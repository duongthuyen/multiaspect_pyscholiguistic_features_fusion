"""Shared logging configuration for the mental-health-fusion pipeline.

Call setup_logging() once at the start of each entry-point script.  All
other modules just call logging.getLogger(__name__) and inherit whatever
handlers this function installs on the root logger.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(
    log_file: Path | None = None,
    level: int = logging.INFO,
) -> None:
    """Configure the root logger with a console handler and optional file handler.

    Safe to call multiple times — duplicate handlers of the same type and path
    are not added.
    """
    fmt = logging.Formatter(_LOG_FORMAT)
    root = logging.getLogger()
    root.setLevel(level)

    existing_files = {
        getattr(h, "baseFilename", None)
        for h in root.handlers
        if isinstance(h, logging.FileHandler)
    }
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )

    if not has_stream:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)

    if log_file is not None:
        abs_path = str(log_file.resolve())
        if abs_path not in existing_files:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
