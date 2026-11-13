"""
BreastGuard AI — Shared Utilities & Helper Functions
"""
from __future__ import annotations

import logging
import sys
import time
from functools import wraps
from typing import Callable


def setup_logger(name: str = "breastguard", level: int = logging.INFO) -> logging.Logger:
    """Configure a standardized logger with clean console formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s : %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def time_execution(func: Callable) -> Callable:
    """Decorator to log function execution duration."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        logging.getLogger(func.__module__).info(
            "%s executed in %.2f seconds", func.__name__, duration
        )
        return result
    return wrapper


def format_percentage(val: float, precision: int = 1) -> str:
    """Format decimal probability into percentage string."""
    return f"{val * 100:.{precision}f}%"


def format_ci(mean: float, std: float, precision: int = 4) -> str:
    """Format mean ± standard deviation string."""
    return f"{mean:.{precision}f} ± {std:.{precision}f}"

