import io
import logging
import os
import sys
from pathlib import Path

def setup_logger(name: str, config: dict, log_file: str | None = None) -> logging.Logger:
    """
    Create and return a logger with console + optional file handler.
    """
    log_dir = config['paths']['logs_dir']
    log_level = config['logging']['level']
    log_format = config['logging']['format']
    date_format = config['logging']['date_format']
    
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Console handler
    utf8_stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    ) if hasattr(sys.stdout, "buffer") else sys.stdout
    ch = logging.StreamHandler(utf8_stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional file handler
    if log_file:
        fh = logging.FileHandler(os.path.join(log_dir, log_file), encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def ensure_dirs(*paths: str) -> None:
    """Create one or more directories if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def is_image_file(path: str) -> bool:
    """Return True if the file extension looks like an image."""
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def count_images(directory: str) -> int:
    """Recursively count image files inside a directory."""
    return sum(1 for f in Path(directory).rglob("*") if is_image_file(str(f)))

def class_image_counts(root_dir: str) -> dict[str, int]:
    """Return a dict {class_name: image_count} for every sub-folder of root_dir."""
    counts = {}
    root = Path(root_dir)
    if not root.exists():
        return counts
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            counts[sub.name] = count_images(str(sub))
    return counts
