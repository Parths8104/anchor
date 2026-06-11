"""Structured logging via structlog.

Why structlog: JSON-serializable, key-value structured logs are the
production standard. They work natively with log aggregators (CloudWatch,
Datadog, Loki) without parse-regex gymnastics.
"""

import logging
import sys

import structlog

from anchor.config import get_settings


def configure_logging() -> None:
    """Configure structlog + stdlib logging for the application.

    Idempotent: safe to call multiple times.
    """
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer()
            if sys.stdout.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structured logger, defaulting to the calling module's name."""
    return structlog.get_logger(name)
