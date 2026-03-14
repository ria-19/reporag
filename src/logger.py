"""
Logging configuration.

Single setup_logging() call in lifespan().
All modules call get_logger(__name__) — standard Python pattern.

WHY structured format?
  "2024-01-15 10:23:45 | INFO | indexer | Flushed 32 chunks"
  Grep-friendly. Level + module visible at a glance.
  In production: swap to JSON formatter for log aggregators.
"""

from __future__ import annotations

import logging
import sys

from src.config import settings


def setup_logging() -> None:
    """
    Configure root logger. Call once at startup in lifespan().
    All subsequent get_logger() calls inherit this config.

    WHY root logger not per-module?
    Configuring root propagates to all children automatically.
    Per-module config = repeated boilerplate everywhere.
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if setup_logging() called twice
    if not root.handlers:
        root.addHandler(handler)

    # Quiet noisy third-party loggers
    # WHY? sentence-transformers and lancedb log at DEBUG
    # constantly — drowns out our own logs.
    for noisy in ("sentence_transformers", "lancedb", "kuzu", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Standard usage in every module:
        from src.logger import get_logger
        logger = get_logger(__name__)

    __name__ gives "src.indexer", "src.storage.lance_store" etc.
    Visible in log output — immediately shows which module logged.
    """
    return logging.getLogger(name)