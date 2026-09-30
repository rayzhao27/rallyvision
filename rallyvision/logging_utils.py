"""Console + file logging with a compact, Colab-friendly format.

Handlers are attached to the single package-root logger ``rallyvision``; every
module uses ``get_logger(__name__)`` and propagates up to it. That way a script
can call :func:`setup_logging` once, after imports, and still capture output
from modules that were imported earlier.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

ROOT_LOGGER_NAME = "rallyvision"
_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_FILE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def _root_logger() -> logging.Logger:
    return logging.getLogger(ROOT_LOGGER_NAME)


def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    name: str = ROOT_LOGGER_NAME,
) -> logging.Logger:
    """Attach console (and optionally file) handlers to the package logger.

    Safe to call repeatedly: handlers are never duplicated, and adding a log
    file later does not disturb the existing console handler.
    """
    root = _root_logger()
    root.setLevel(level)
    root.propagate = False

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter(_CONSOLE_FORMAT, "%H:%M:%S"))
        root.addHandler(stream)
    for handler in root.handlers:
        handler.setLevel(level)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        target = str(log_file.resolve())
        existing = {
            getattr(h, "baseFilename", None)
            for h in root.handlers
            if isinstance(h, logging.FileHandler)
        }
        if target not in existing:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
            root.addHandler(file_handler)

    return logging.getLogger(name)


def get_logger(name: str = ROOT_LOGGER_NAME) -> logging.Logger:
    """Return a module logger, configuring the package root on first use."""
    if not _root_logger().handlers:
        setup_logging()
    return logging.getLogger(name)


def set_level(level: int) -> None:
    root = _root_logger()
    root.setLevel(level)
    for handler in root.handlers:
        handler.setLevel(level)


def banner(text: str, char: str = "=", width: int = 72) -> str:
    text = f" {text} "
    pad = max(0, width - len(text))
    left = pad // 2
    return char * left + text + char * (pad - left)


def log_banner(logger: logging.Logger, text: str) -> None:
    logger.info(banner(text))


def log_block(logger: logging.Logger, title: str, body: str) -> None:
    logger.info(banner(title))
    for line in body.splitlines():
        logger.info(line)
