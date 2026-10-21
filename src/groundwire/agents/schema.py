"""A small JSON Schema checker for tool-call arguments.

Tools declare their arguments as JSON Schema. Models sometimes produce
arguments that do not match it (a string where a number is expected, a
missing required field). Catching that in the gateway gives the agent a
clear error instead of a crash, or a silently wrong action, inside the tool.

Scope, stated plainly: this implements the subset that tool definitions use
in practice (``type``, ``properties``, ``required``, ``enum``, ``const``,
``additionalProperties: false``, ``items``, ``minimum``/``maximum``,
``minLength``/``maxLength``, ``minItems``/``maxItems``). Keywords it does not
know (``$ref``, ``oneOf``, ``pattern``, ``format``...) are **ignored, not
failed**: a gateway that rejects valid calls because of an unsupported
keyword would break working agents. For full validation, use a JSON Schema
library in the tool itself.
"""

from __future__ import annotations

from typing import Any

_MAX_ERRORS = 5

_TYPE_CHECKS: dict[str, type | tuple[type, ...]] = {
    "string": str, "boolean": bool, "object": dict, "array": list,
}


def _type_ok(value: Any, expected: str) -> bool:
    """Check a JSON type name (``bool`` is not a number in JSON Schema).

    Args:
        value: Decoded JSON value.
        expected: JSON Schema type name.

    Returns:
        True if the value has that type (unknown type names pass).
    """
    if expected == "null":
        return value is None
    if expected == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and value.is_integer())
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    check = _TYPE_CHECKS.get(expected)
    return True if check is None else isinstance(value, check)


def validate(value: Any, schema: Any, path: str = "") -> list[str]:
    """Validate ``value`` against the supported subset of ``schema``.

    Args:
        value: Decoded JSON value.
        schema: JSON Schema (anything that is not a dict accepts everything).
        path: Location for messages (dotted).

    Returns:
        Human-readable problems (empty when valid), at most a few, and never
        containing argument values (they may hold personal data).
    """
    if not isinstance(schema, dict):
        return []
    where = path or "arguments"
    errors: list[str] = []

    expected = schema.get("type")
    types = expected if isinstance(expected, list) else [expected] if expected else []
    if types and not any(_type_ok(value, str(t)) for t in types):
        return [f"{where} should be {' or '.join(map(str, types))}"]

    if "enum" in schema and isinstance(schema["enum"], list) and value not in schema["enum"]:
        errors.append(f"{where} is not one of the allowed values")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{where} is not the required constant")

    if isinstance(value, int | float) and not isinstance(value, bool):
        if isinstance(schema.get("minimum"), int | float) and value < schema["minimum"]:
            errors.append(f"{where} is below the minimum {schema['minimum']}")
        if isinstance(schema.get("maximum"), int | float) and value > schema["maximum"]:
            errors.append(f"{where} is above the maximum {schema['maximum']}")
    if isinstance(value, str):
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
            errors.append(f"{where} is shorter than {schema['minLength']} characters")
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            errors.append(f"{where} is longer than {schema['maxLength']} characters")
    if isinstance(value, list):
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            errors.append(f"{where} has fewer than {schema['minItems']} items")
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            errors.append(f"{where} has more than {schema['maxItems']} items")
        for index, item in enumerate(value):
            errors.extend(validate(item, schema.get("items"), f"{where}[{index}]"))
    if isinstance(value, dict):
        errors.extend(_validate_object(value, schema, path))
    return errors[:_MAX_ERRORS]


def _validate_object(value: dict[str, Any], schema: dict[str, Any], path: str) -> list[str]:
    """Check ``required``, ``properties`` and ``additionalProperties: false``.

    Args:
        value: Decoded object.
        schema: Object schema.
        path: Location prefix.

    Returns:
        Problems found.
    """
    errors: list[str] = []
    prefix = f"{path}." if path else ""
    raw_properties = schema.get("properties")
    properties: dict[str, Any] = raw_properties if isinstance(raw_properties, dict) else {}
    for name in schema.get("required") or []:
        if isinstance(name, str) and name not in value:
            errors.append(f"missing required argument '{prefix}{name}'")
    for name, item in value.items():
        if name in properties:
            errors.extend(validate(item, properties[name], f"{prefix}{name}"))
        elif schema.get("additionalProperties") is False:
            errors.append(f"unexpected argument '{prefix}{name}'")
    return errors
