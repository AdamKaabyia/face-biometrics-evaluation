#!/usr/bin/env python3
"""
logger.py — Centralized logging configuration for face biometrics evaluation
"""

import logging
import sys
from typing import Optional

def setup_logger(name: Optional[str] = None, level: int = logging.INFO, format_str: Optional[str] = None) -> logging.Logger:
    """Setup and return a configured logger.

    Args:
        name: Logger name
        level: Logging level
        format_str: Custom format string

    Returns:
        Configured logger instance
    """
    if format_str is None:
        format_str = "%(asctime)s [%(levelname)s] %(message)s"

    if name is None:
        name = __name__

    # Create logger
    logger = logging.getLogger(name)

    # Check if logger already has handlers to avoid duplication
    if not logger.handlers:
        logger.setLevel(level)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(format_str, datefmt="%H:%M:%S")
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

    return logger

# Create default logger instance
logger = setup_logger()