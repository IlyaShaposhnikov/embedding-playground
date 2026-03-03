"""
Logging configuration module for Embedding Visualizer.

This module provides centralized logging setup for the entire application,
separating user-facing interface messages (on console)
from system/process logs (to file).
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    verbose: bool = False,
    log_file: Path = None,
    console_level: int = logging.WARNING
) -> None:
    """
    Setup application logging with console and file handlers.

    Args:
        verbose: If True, increases console level to INFO, otherwise WARNING.
        Also increases file level to DEBUG.
        log_file: Optional path to write detailed logs to file.
        If None, defaults to logs/embedding_visualizer.log
        console_level: Minimum logging level for console output.
        Default is WARNING to show only important messages to user.
    """
    if log_file is None:
        log_file = Path("logs/embedding_visualizer.log")

    # Determine log levels
    console_log_level = logging.INFO if verbose else console_level
    file_log_level = logging.DEBUG if verbose else logging.INFO

    # Clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set root logger level to lowest level (DEBUG) to allow filtering
    root_logger.setLevel(logging.DEBUG)

    # Create formatter for detailed file logs
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create formatter for concise console logs
    # (for user-facing errors/warnings)
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # Console handler - for user-facing warnings and errors only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler - for detailed system logs
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Suppress overly verbose loggers from external libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    # Suppress gensim's detailed loading logs unless in verbose mode
    if not verbose:
        logging.getLogger('gensim').setLevel(logging.WARNING)
