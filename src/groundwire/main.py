"""Application factory and process entrypoint.

``create_app`` builds a FastAPI app whose lifespan wires the container at
startup and closes provider pools at shutdown. Configuration errors abort
startup with a clear log line instead of surfacing on the first request.

Run locally::

    GROUNDWIRE_CONFIG_PATH=config/groundwire.yaml groundwire
    # or
    uvicorn groundwire.main:app_factory --factory --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from groundwire import __version__
from groundwire.api.approvals import router as approvals_router
from groundwire.api.errors import register_exception_handlers
from groundwire.api.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from groundwire.api.routes import router
from groundwire.container import Container, build_container
from groundwire.core.config import GatewayConfig, Settings, load_gateway_config
from groundwire.core.deployment import check_deployment, effective_workers
from groundwire.core.errors import ConfigurationError
from groundwire.core.logging import configure_logging

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 4 * 1024 * 1024  # 4 MiB: generous for RAG context, bounded for safety

ContainerFactory = Callable[[], Container]


def create_app(
    *,
    settings: Settings | None = None,
    config: GatewayConfig | None = None,
    container_factory: ContainerFactory | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        settings: Process settings; read from the environment if omitted.
        config: Gateway config; loaded from ``settings.config_path`` if omitted.
        container_factory: Builds the container at startup (tests inject fakes).

    Returns:
        The configured app. Components are built when the lifespan starts.
    """
    settings = settings or Settings()

    def default_factory() -> Container:
        gateway_config = config or load_gateway_config(settings.config_path)
        # Runs in every worker: also catches `uvicorn --workers N` / WEB_CONCURRENCY,
        # which bypass the `groundwire` entrypoint's own check.
        check_deployment(gateway_config,
                         workers=effective_workers(settings.workers, os.environ))
        return build_container(gateway_config)

    factory = container_factory or default_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            container = factory()
        except ConfigurationError as exc:
            logger.critical("startup aborted: configuration error", extra={"error": exc.message})
            raise
        app.state.container = container
        logger.info("groundwire started", extra={"version": __version__,
                                                 "routes": list(container.router.routes)})
        try:
            yield
        finally:
            await container.aclose()
            logger.info("groundwire stopped")

    app = FastAPI(title="Groundwire", version=__version__, lifespan=lifespan,
                  docs_url="/docs", redoc_url=None)
    app.include_router(router)
    app.include_router(approvals_router)
    register_exception_handlers(app)
    # Order: outermost first. Request context wraps everything so even 413s
    # get a request ID and an access-log line.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)
    app.add_middleware(RequestContextMiddleware)
    return app


def app_factory() -> FastAPI:
    """Uvicorn ``--factory`` entrypoint: configure logging, then build the app.

    Returns:
        The app.
    """
    settings = Settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    return create_app(settings=settings)


def run() -> None:
    """Console-script entrypoint (``groundwire``).

    Validates the config for the configured worker count *before* spawning
    workers, so a bad deployment fails once, clearly, instead of in N workers.

    Raises:
        SystemExit: With status 2 when the configuration is invalid.
    """
    settings = Settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    try:
        check_deployment(load_gateway_config(settings.config_path),
                         workers=effective_workers(settings.workers, os.environ))
    except ConfigurationError as exc:
        logger.critical("startup aborted: configuration error", extra={"error": exc.message})
        sys.exit(2)
    uvicorn.run(
        "groundwire.main:app_factory",
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_config=None,  # keep our JSON logging
        proxy_headers=True,
        timeout_graceful_shutdown=30,
    )


if __name__ == "__main__":
    run()
