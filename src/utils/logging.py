"""Centralized logging utilities for the MEWS project."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(
    *,
    level: str = "INFO",
    output_dir: str = "outputs",
    filename: str = "risk_system.log",
    log_format: str = _DEFAULT_FORMAT,
    force: bool = False,
) -> None:
    """Configure root logging handlers with a shared format.

    Parameters
    ----------
    level:
        Logging level name (e.g. ``"INFO"`` or ``"DEBUG"``).
    output_dir:
        Directory where file logs will be written.
    filename:
        Name of the log file within ``output_dir``.
    log_format:
        Format string used for log records.
    force:
        If ``True``, existing handlers are removed prior to configuration.
    """

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    log_directory = Path(output_dir)
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / filename

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[
            logging.FileHandler(str(log_path), mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=force,
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-specific logger configured via :func:`configure_logging`."""

    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
