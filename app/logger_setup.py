"""
Logging configuration.
"""

import logging


def setup_logging():
    # Set up basic logging with level INFO and a simple formatter
    logging.basicConfig(
        level=logging.INFO,
        # timestamp - logger name - log level - message
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and return a named logger for the application
    logger = logging.getLogger("mlflow_app")
    return logger
