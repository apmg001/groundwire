"""HTTP routes.

* ``POST /v1/chat/completions``: OpenAI-compatible, streaming and non-streaming.
* ``GET  /v1/models``: configured route aliases, in OpenAI list format.
* ``GET  /healthz``: liveness (process is up). Never touches providers.
* ``GET  /readyz``: readiness (every route has a provider whose circuit is not
  open). Kubernetes stops sending traffic while this is failing.
* ``GET  /metrics``: Prometheus exposition.

Approval routes (``/v1/approvals``) live in :mod:`groundwire.api.approvals`.

Handlers are thin: parse, call the service, frame the response.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from groundwire import __version__
from groundwire.container import Container
from groundwire.core.context import get_request_id
from groundwire.core.errors import GroundwireError
from groundwire.domain.models import ChatCompletionRequest, ChatCompletionResponse
from groundwire.services.chat import StreamSession

logger = logging.getLogger(__name__)

router = APIRouter()

PROVIDER_HEADER = "X-Groundwire-Provider"


def get_container(request: Request) -> Container:
    """FastAPI dependency returning the app's container.

    Args:
        request: Current request.

    Returns:
        The container built during startup.
    """
    container: Container = request.app.state.container
    return container


# --------------------------------------------------------------------------- #
# Chat completions
# --------------------------------------------------------------------------- #


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    body: ChatCompletionRequest,
    container: Container = Depends(get_container),  # noqa: B008 - FastAPI idiom
) -> Response:
    """Create a chat completion.

    Args:
        body: OpenAI-compatible request body.
        container: Injected container.

    Returns:
        JSON response, or an SSE stream when ``stream`` is true.
    """
    started = time.perf_counter()
    route = body.model or container.config.default_route
    try:
        if body.stream:
            session = await container.chat.stream(body)
        else:
            result: ChatCompletionResponse = await container.chat.complete(body)
    except GroundwireError as exc:
        _record(container, route=route, provider="", status=exc.http_status, started=started,
                stream=body.stream, error_code=exc.code)
        raise

    if body.stream:
        # Recorded when the stream starts (time to first byte): later chunks are the
        # model's pace, and in-band stream errors are counted in the metrics.
        _record(container, route=route, provider=session.provider, status=200,
                started=started, stream=True)
        return StreamingResponse(
            _sse(session),
            media_type="text/event-stream",
            headers={
                PROVIDER_HEADER: session.provider,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
            },
        )

    _record(container, route=str(result.groundwire.get("route", route)),
            provider=str(result.groundwire.get("provider", "")), status=200, started=started,
            metadata=result.groundwire)
    return JSONResponse(
        content=result.model_dump(exclude_none=True),
        headers={PROVIDER_HEADER: str(result.groundwire.get("provider", ""))},
    )


def _record(container: Container, *, route: str, provider: str, status: int, started: float,
            stream: bool = False, error_code: str = "",
            metadata: Mapping[str, Any] | None = None) -> None:
    """Add one request to the dashboard's activity log (no-op when it is off).

    Never raises: a dashboard problem must not turn a served request into an error.

    Args:
        container: App container.
        route: Route alias.
        provider: Provider that answered ("" if none).
        status: HTTP status returned.
        started: ``perf_counter`` at request start.
        stream: Streamed response.
        error_code: Error code for failures.
        metadata: The response's ``groundwire`` metadata.
    """
    if container.activity is None:
        return
    try:
        container.activity.record(request_id=get_request_id(), route=route, provider=provider,
                                  status=status, stream=stream, error_code=error_code,
                                  latency_ms=(time.perf_counter() - started) * 1000,
                                  metadata=metadata)
    except Exception:  # defensive: see docstring
        logger.warning("could not record dashboard activity", exc_info=True)


async def _sse(session: StreamSession) -> AsyncIterator[bytes]:
    """Frame stream payloads as server-sent events.

    Args:
        session: Started stream.

    Yields:
        Encoded ``data:`` lines, ending with ``data: [DONE]``.
    """
    async for payload in session.events:
        yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()
    yield b"data: [DONE]\n\n"


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


@router.get("/v1/models")
async def list_models(container: Container = Depends(get_container)) -> dict[str, Any]:  # noqa: B008
    """List route aliases as OpenAI "models".

    Args:
        container: Injected container.

    Returns:
        OpenAI-style model list.
    """
    routes = [{"id": route, "object": "model", "owned_by": "groundwire", "providers": chain}
              for route, chain in container.router.routes.items()]
    cascades = [{"id": name, "object": "model", "owned_by": "groundwire",
                 "providers": tiers["cheap"] + tiers["strong"], "cascade": tiers}
                for name, tiers in container.router.cascades.items()]
    return {"object": "list", "data": routes + cascades}


# --------------------------------------------------------------------------- #
# Health and metrics
# --------------------------------------------------------------------------- #


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe.

    Returns:
        Static OK payload with the version.
    """
    return {"status": "ok", "version": __version__}


@router.get("/readyz")
async def readyz(container: Container = Depends(get_container)) -> JSONResponse:  # noqa: B008
    """Readiness probe: every route and cascade needs at least one non-open circuit.

    Args:
        container: Injected container.

    Returns:
        200 with per-provider circuit states, or 503 listing unavailable routes.
    """
    states = container.router.breaker_states()
    unavailable = [
        route for route, chain in container.router.routes.items()
        if all(states.get(p) == "open" for p in chain)
    ]
    # A cascade is down only when both tiers are (either tier can serve alone).
    unavailable += [
        name for name, tiers in container.router.cascades.items()
        if all(states.get(p) == "open" for chain in tiers.values() for p in chain)
    ]
    status_code = 503 if unavailable else 200
    return JSONResponse(
        status_code=status_code,
        content={"status": "unavailable" if unavailable else "ready",
                 "circuits": states, "unavailable_routes": unavailable},
    )


@router.get("/metrics")
async def metrics(container: Container = Depends(get_container)) -> Response:  # noqa: B008
    """Prometheus metrics endpoint.

    Args:
        container: Injected container.

    Returns:
        Metrics in text exposition format.
    """
    return Response(content=container.metrics.render(),
                    media_type="text/plain; version=0.0.4; charset=utf-8")
