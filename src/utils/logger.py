"""Structured experiment logger utility.

This module provides logging handlers to output training and feature selection
metrics into the logs/ directory.
"""

import logging
import os
import json
from typing import Dict, Any


def setup_logger(name: str, log_dir: str = "logs/", log_level: str = "INFO") -> logging.Logger:
    """Configures and returns a logger instance writing to log_dir.

    Args:
        name: Name of the logger module.
        log_dir: Directory where log files are stored.
        log_level: Severity level for output logging.

    Returns:
        The configured Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicating handlers
    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        formatter = logging.Formatter(
            "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream handler for console stdout
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def log_experiment_results(
    filepath: str,
    parameters: Dict[str, Any],
    metrics: Dict[str, float]
) -> None:
    """Writes experimental metrics and metadata out as a structured JSON file.

    Args:
        filepath: Target JSON file path to write results.
        parameters: Input hyperparameters or trial values.
        metrics: Resulting accuracy, precision, recall, F1, and latency.
    """
    results = {
        "parameters": parameters,
        "metrics": metrics
    }
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)
