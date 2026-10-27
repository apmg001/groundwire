"""Process entry point: ``python -m money_agent.main`` (or the ``money-agent`` script)."""

from __future__ import annotations

import logging
import sys

import uvicorn

from money_agent.api.app import create_app
from money_agent.container import build_container
from money_agent.core.config import load_settings
from money_agent.core.errors import ConfigurationError
from money_agent.core.logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> int:
    """Configure logging, validate settings, and serve.

    Returns:
        Process exit code (2 on invalid configuration).
    """
    try:
        settings = load_settings()
    except ConfigurationError as exc:
        configure_logging()
        logger.error("cannot start: %s", exc)
        return 2
    configure_logging(settings.log_level, json_output=settings.log_json)
    app = create_app(build_container(settings))
    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None,
                proxy_headers=True, server_header=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
