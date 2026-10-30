"""Tool framework: typed tools, one registry, failures as data.

* **One source of truth per tool.** A tool's arguments are a Pydantic model.
  The JSON schema the model sees is generated from it, and the same model
  validates what the model sends back, so the two can never drift apart.
* **Failures are results, not crashes.** Bad arguments, "no data yet",
  timeouts and bugs all come back to the model as ``{"error": ...}``, so it
  can correct itself or explain, and one bad call never kills a conversation.
  Internal detail is logged, never put in the result (results go to the model,
  and through it to the user).
* **Bounded.** Each call has a timeout, and results are size-capped, so a
  tool cannot stall the loop or blow up the prompt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from money_agent.core.errors import MoneyAgentError
from money_agent.domain.ports import ToolCallRequest

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 10.0
MAX_RESULT_CHARS = 12_000

ArgsT = TypeVar("ArgsT", bound=BaseModel)


class ToolArgs(BaseModel):
    """Base for tool argument models: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Who a tool call is for.

    Attributes:
        user_id: The user (tools only ever touch this user's data).
    """

    user_id: str


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """The result of one call.

    Attributes:
        call_id: The model's call id.
        name: Tool name.
        result: JSON-serialisable result sent back to the model.
        ok: False when ``result`` is an error.
        latency_ms: Time spent.
    """

    call_id: str
    name: str
    result: dict[str, Any]
    ok: bool
    latency_ms: float = 0.0

    def message(self) -> dict[str, Any]:
        """The OpenAI-format ``tool`` message carrying this result."""
        return {"role": "tool", "tool_call_id": self.call_id, "content": encode(self.result)}


def encode(result: dict[str, Any]) -> str:
    """Serialise a result compactly, capped at ``MAX_RESULT_CHARS``.

    Args:
        result: The result.

    Returns:
        JSON text (an error object if the result was too large).
    """
    text = json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return json.dumps({"error": "result_too_large",
                       "message": "Too much data; narrow the request (dates, category, limit)."})


def error_result(code: str, message: str, **extra: Any) -> dict[str, Any]:
    """A tool error the model can read and act on.

    Args:
        code: Machine-readable code.
        message: What to do about it.
        **extra: More fields.

    Returns:
        ``{"error": code, "message": message, ...}``.
    """
    return {"error": code, "message": message, **extra}


# Schema keys that cost prompt tokens on every model call but don't help the model
# choose or fill a tool. Validation never relies on the sent schema: the Pydantic
# model validates every call on our side, with all its constraints.
_SCHEMA_NOISE = frozenset({"title", "additionalProperties", "maxLength", "minLength"})


def compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """A token-lean JSON schema for the model, from Pydantic's full one.

    Keeps names, types, enums, formats, patterns, numeric ranges and field
    descriptions; inlines ``$ref`` definitions; turns ``anyOf[X, null]`` into
    ``X`` (optional fields are simply not ``required``); drops titles, null
    defaults, docstring descriptions of models and enums, and length limits.

    Args:
        schema: ``model_json_schema()`` output.

    Returns:
        The compact schema (about half the size on this toolkit).
    """
    definitions = schema.get("$defs", {})

    def walk(node: Any, *, field: bool) -> Any:
        if isinstance(node, list):
            return [walk(item, field=field) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = definitions[node["$ref"].rsplit("/", 1)[-1]]
            return walk({k: v for k, v in target.items() if k != "description"}, field=False)
        options = node.get("anyOf")
        if isinstance(options, list):
            concrete = [o for o in options if o != {"type": "null"}]
            if len(concrete) == 1:
                merged = {**concrete[0], **{k: v for k, v in node.items() if k != "anyOf"}}
                return walk(merged, field=field)
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _SCHEMA_NOISE or key == "$defs" or (key == "default" and value is None):
                continue
            if key == "description" and not field:
                continue  # model/enum docstrings repeat the tool description
            if key == "properties":
                out[key] = {name: walk(prop, field=True) for name, prop in value.items()}
            else:
                out[key] = walk(value, field=False)
        return out

    return walk(schema, field=False)  # type: ignore[no-any-return]


class Tool(ABC, Generic[ArgsT]):
    """A function the agent can call."""

    def __init__(self, name: str, description: str, args_model: type[ArgsT], *,
                 timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        """Describe the tool.

        Args:
            name: Function name the model uses (``^[a-z_]{1,64}$``).
            description: When and why to use it (the model reads this).
            args_model: Argument model.
            timeout_s: Time limit per call.

        Raises:
            ValueError: If the name is not a valid function name.
        """
        if not name.replace("_", "").isalpha() or not name.islower() or len(name) > 64:
            raise ValueError(f"invalid tool name {name!r}")
        self.name = name
        self.description = description
        self.args_model = args_model
        self.timeout_s = timeout_s

    def definition(self) -> dict[str, Any]:
        """The OpenAI-format tool definition."""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": compact_schema(self.args_model.model_json_schema())}}

    @abstractmethod
    async def run(self, context: ToolContext, args: ArgsT) -> dict[str, Any]:
        """Do the work.

        Args:
            context: Who the call is for.
            args: Validated arguments.

        Returns:
            A JSON-serialisable result.

        Raises:
            MoneyAgentError: Expected failures (reported to the model by code).
        """

    async def invoke(self, context: ToolContext, call: ToolCallRequest) -> ToolOutcome:
        """Validate, run under the timeout, and turn every failure into a result.

        Args:
            context: Who the call is for.
            call: The model's call.

        Returns:
            The outcome (never raises for tool-level failures).
        """
        started = time.perf_counter()
        result, ok = await self._execute(context, call)
        latency = (time.perf_counter() - started) * 1000
        log = logger.info if ok else logger.warning
        log("tool call", extra={"tool": self.name, "ok": ok, "latency_ms": round(latency, 1),
                                "error_code": None if ok else result.get("error")})
        return ToolOutcome(call.id, self.name, result, ok, latency)

    async def _execute(self, context: ToolContext, call: ToolCallRequest
                       ) -> tuple[dict[str, Any], bool]:
        try:
            raw = json.loads(call.arguments or "{}")
            args = self.args_model.model_validate(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            problems = ([f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
                         for e in exc.errors()] if isinstance(exc, ValidationError)
                        else ["arguments are not valid JSON"])
            return error_result("invalid_arguments", "Fix the arguments and call again.",
                                problems=problems[:5]), False
        try:
            async with asyncio.timeout(self.timeout_s):
                return await self.run(context, args), True
        except TimeoutError:
            return error_result("timeout", "The tool took too long. Try a narrower request."), \
                False
        except MoneyAgentError as exc:
            return error_result(exc.code, exc.public_message), False
        except Exception:  # one broken tool must not end the conversation (logged)
            logger.exception("tool crashed", extra={"tool": self.name})
            return error_result("tool_failed", "This tool failed. Answer without it, and say "
                                               "that this information is unavailable."), False


@dataclass
class ToolRegistry:
    """The tools offered to the model, by name."""

    tools: dict[str, Tool[Any]] = field(default_factory=dict)

    @classmethod
    def of(cls, tools: Iterable[Tool[Any]]) -> ToolRegistry:
        """Build a registry.

        Args:
            tools: Tools (names must be unique).

        Returns:
            The registry.

        Raises:
            ValueError: On a duplicate name.
        """
        registry = cls()
        for tool in tools:
            if tool.name in registry.tools:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            registry.tools[tool.name] = tool
        return registry

    def definitions(self) -> list[dict[str, Any]]:
        """OpenAI-format definitions of every tool."""
        return [tool.definition() for tool in self.tools.values()]

    async def invoke(self, context: ToolContext, call: ToolCallRequest) -> ToolOutcome:
        """Run one call (an unknown tool is reported to the model, not raised).

        Args:
            context: Who the call is for.
            call: The model's call.

        Returns:
            The outcome.
        """
        tool = self.tools.get(call.name)
        if tool is None:
            logger.warning("unknown tool requested", extra={"tool": call.name[:64]})
            return ToolOutcome(call.id, call.name, error_result(
                "unknown_tool", f"There is no tool named {call.name[:64]!r}. "
                                f"Available: {', '.join(sorted(self.tools))}."), ok=False)
        return await tool.invoke(context, call)
