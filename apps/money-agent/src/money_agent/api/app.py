"""Application factory.

``create_app()`` builds a fully wired FastAPI app: settings -> container ->
routes, middleware and the static chat page. Resources open in the lifespan
(before the first request) and close on shutdown. Tests pass their own
container, so no global state is involved.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from money_agent.api import middleware
from money_agent.api.routes import router
from money_agent.container import Container, build_container
from money_agent.core.config import load_settings

STATIC_DIR = resources.files("money_agent.api") / "static"


def create_app(container: Container | None = None) -> FastAPI:
    """Build the app.

    Args:
        container: A pre-built container (tests); built from the environment otherwise.

    Returns:
        The application.

    Raises:
        ConfigurationError: Invalid settings (only when building from the environment).
    """
    wired = container or build_container(load_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await wired.start()
        try:
            yield
        finally:
            await wired.close()

    app = FastAPI(title="Groundwire Money", version="0.1.0", lifespan=lifespan,
                  description="A personal-finance agent whose answers are checked against "
                              "your data before you see them.")
    app.state.container = wired
    middleware.install(app)
    app.include_router(router)
    static = str(STATIC_DIR)
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(f"{static}/index.html")

    return app
