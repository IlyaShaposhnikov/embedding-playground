import sys
import logging
from pathlib import Path

from src.core.logging_config import setup_logging
from src.core.model_manager import ModelManager
from src.cli import interactive_shell


def main() -> int:
    print("Embedding Visualizer — Initializing...")

    # Warnings/errors go to console, detailed logs to file
    setup_logging(
        verbose=False, log_file=Path("logs/embedding_visualizer.log")
    )
    logger = logging.getLogger(__name__)
    logger.info("-" * 60)
    logger.info("Embedding Visualizer")
    logger.info("Models will be loaded on first use if missing.")

    # Initialize model manager for lazy loading
    model_manager = ModelManager()

    print("Starting Interactive Shell...")
    interactive_shell(model_manager)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("\nInterrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logging.getLogger(__name__).error(
            f"\nUnexpected error: {type(e).__name__}: {e}"
        )
        sys.exit(1)
