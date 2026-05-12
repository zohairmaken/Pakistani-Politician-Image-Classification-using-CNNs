"""
utils.py
========
Shared utility functions used across the pipeline:
- Logging setup
- Directory creation
- Image validation helpers
"""
import io

import logging
import os
import sys
from pathlib import Path

from config import LOG_FORMAT, DATE_FORMAT, LOG_LEVEL, LOGS_DIR


def setup_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """
    Create and return a logger with console + optional file handler.

    Args:
        name:     Logger name (usually __name__).
        log_file: Optional filename inside LOGS_DIR.

    Returns:
        Configured logging.Logger instance.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── console handler (force UTF-8 so box-drawing chars work on Windows) ──
    utf8_stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    ) if hasattr(sys.stdout, "buffer") else sys.stdout
    ch = logging.StreamHandler(utf8_stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # ── optional file handler ──
    if log_file:
        fh = logging.FileHandler(os.path.join(LOGS_DIR, log_file), encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def ensure_dirs(*paths: str) -> None:
    """Create one or more directories (including parents) if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def is_image_file(path: str) -> bool:
    """Return True if the file extension looks like an image."""
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(directory: str) -> int:
    """Recursively count image files inside a directory."""
    return sum(1 for f in Path(directory).rglob("*") if is_image_file(str(f)))


def class_image_counts(root_dir: str) -> dict[str, int]:
    """
    Return a dict {class_name: image_count} for every sub-folder of root_dir.
    """
    counts = {}
    root = Path(root_dir)
    if not root.exists():
        return counts
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            counts[sub.name] = count_images(str(sub))
    return counts
