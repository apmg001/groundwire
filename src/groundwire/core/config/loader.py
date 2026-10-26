"""Load and validate the gateway YAML file.

All validation happens at load time, so a bad config never reaches traffic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    ValidationError,
)

from groundwire.core.config.gateway import GatewayConfig
from groundwire.core.errors import ConfigurationError

logger = logging.getLogger(__name__)


def load_gateway_config(path: Path) -> GatewayConfig:
    """Load and validate the gateway YAML configuration.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.

    Raises:
        ConfigurationError: If the file is missing, unreadable, not valid YAML,
            or fails schema/cross-reference validation.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigurationError(f"config file not found: {path}") from exc
    except OSError as exc:
        raise ConfigurationError(f"cannot read config file {path}: {exc}") from exc

    try:
        raw: Any = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(f"config root in {path} must be a mapping")

    return parse_gateway_config(raw, source=str(path))


def parse_gateway_config(raw: dict[str, Any], *, source: str = "<dict>") -> GatewayConfig:
    """Validate an already-parsed config mapping.

    Args:
        raw: Parsed configuration mapping.
        source: Label used in error messages.

    Returns:
        The validated configuration.

    Raises:
        ConfigurationError: If validation fails.
    """
    try:
        config = GatewayConfig.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigurationError(f"invalid gateway config ({source}): {problems}") from exc

    logger.info(
        "gateway config loaded",
        extra={
            "source": source,
            "providers": [p.name for p in config.enabled_providers()],
            "routes": list(config.routes),
        },
    )
    return config
