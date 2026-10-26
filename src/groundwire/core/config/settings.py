"""Process-level settings, read from ``GROUNDWIRE_*`` environment variables.

This is what changes per deployment or container (bind address, log level,
the path to the gateway YAML). Behaviour lives in :class:`GatewayConfig`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from pydantic import (
    Field,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-level settings read from ``GROUNDWIRE_*`` environment variables.

    Attributes:
        config_path: Path to the gateway YAML config.
        host: Interface to bind.
        port: Port to bind.
        log_level: Root log level name.
        log_json: Emit JSON logs (production) instead of human-readable text.
        workers: Uvicorn worker processes.
    """

    model_config = SettingsConfigDict(env_prefix="GROUNDWIRE_", extra="ignore")

    config_path: Path = Path("config/groundwire.yaml")
    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 8000
    log_level: str = "INFO"
    log_json: bool = True
    workers: Annotated[int, Field(ge=1, le=64)] = 1

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in logging.getLevelNamesMapping():
            raise ValueError(f"unknown log level '{value}'")
        return level
