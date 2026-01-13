"""Logging utilities for experiments"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with consistent formatting

    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        console: Whether to also log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default settings

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ProgressLogger:
    """Context manager for logging progress with counts"""

    def __init__(self, logger: logging.Logger, task_name: str, total: int):
        self.logger = logger
        self.task_name = task_name
        self.total = total
        self.count = 0

    def __enter__(self):
        self.logger.info(f"Starting: {self.task_name} (total: {self.total})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.info(f"Completed: {self.task_name} ({self.count}/{self.total})")
        else:
            self.logger.error(f"Failed: {self.task_name} at {self.count}/{self.total}")
        return False

    def update(self, increment: int = 1):
        """Update progress count"""
        self.count += increment
        if self.count % max(1, self.total // 10) == 0:  # Log every 10%
            self.logger.info(f"Progress: {self.task_name} - {self.count}/{self.total} ({self.count/self.total:.1%})")
