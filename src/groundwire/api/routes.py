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
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from groundwire import __version__
from groundwire.container import Container
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
    if body.stream:
        session = await container.chat.stream(body)
        return StreamingResponse(
            _sse(session),
            media_type="text/event-stream",
            headers={
                PROVIDER_HEADER: session.provider,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
            },
        )

    result: ChatCompletionResponse = await container.chat.complete(body)
    return JSONResponse(
        content=result.model_dump(exclude_none=True),
        headers={PROVIDER_HEADER: str(result.groundwire.get("provider", ""))},
    )


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
