"""Structured experiment logger utility.

This module provides logging handlers and the ExperimentLogger class to log
hyperparameters, evaluation metrics, and feature selection masks to the logs/ folder.
"""

import logging
import os
import json
from typing import Dict, Any, List, Optional
import numpy as np


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


class ExperimentLogger:
    """Logs experimental trials, hyperparameter settings, and outputs summary files."""

    def __init__(self, experiment_name: str, log_dir: str = "logs/") -> None:
        """Initializes the experiment logger.

        Args:
            experiment_name: Descriptive name of the current research experiment run.
            log_dir: Folder path where log files will be saved.
        """
        self.experiment_name: str = experiment_name
        self.log_dir: str = log_dir
        self.hyperparams: Dict[str, Any] = {}
        self.metrics_history: List[Dict[str, Any]] = []
        self.feature_mask: Optional[List[int]] = None
        self.feature_names: Optional[List[str]] = None
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger: logging.Logger = setup_logger(self.experiment_name, self.log_dir)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Logs hyperparameters for the current run.

        Args:
            params: Dictionary of configuration hyperparameter values.
        """
        self.hyperparams.update(params)
        self.logger.info(f"Hyperparameters registered: {params}")

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Logs evaluation metrics for a specific epoch or execution step.

        Args:
            metrics: Dictionary of metric results.
            step: Step index (e.g. epoch number or run number).
        """
        entry = {"step": step, "metrics": metrics}
        self.metrics_history.append(entry)
        self.logger.info(f"Step {step} Metrics registered: {metrics}")

    def log_feature_mask(self, mask: np.ndarray, feature_names: List[str]) -> None:
        """Logs the active BWOA feature selection mask.

        Args:
            mask: Binary feature selection mask of shape (n_features,).
            feature_names: List of all feature name strings.
        """
        self.feature_mask = mask.astype(int).tolist()
        self.feature_names = feature_names
        selected = [name for m, name in zip(self.feature_mask, self.feature_names) if m == 1]
        self.logger.info(f"Selected {len(selected)} features out of {len(feature_names)}: {selected}")

    def save_experiment_summary(self, output_path: str) -> None:
        """Saves a detailed trial summary report to JSON and text formats.

        Args:
            output_path: Base filepath (without extension) where outputs will be saved.
        """
        summary = {
            "experiment_name": self.experiment_name,
            "hyperparameters": self.hyperparams,
            "metrics_history": self.metrics_history,
            "feature_mask": self.feature_mask,
            "feature_names": self.feature_names
        }
        
        # Determine json and txt file paths
        json_path = output_path if output_path.endswith(".json") else f"{output_path}.json"
        txt_path = output_path.replace(".json", "") + ".txt" if output_path.endswith(".json") else f"{output_path}.txt"
        
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        # 1. Save JSON
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)
        self.logger.info(f"JSON summary saved to: {json_path}")
            
        # 2. Save Text
        with open(txt_path, "w") as f:
            f.write(f"=== EXPERIMENT SUMMARY: {self.experiment_name} ===\n\n")
            f.write("Hyperparameters:\n")
            for k, v in self.hyperparams.items():
                f.write(f"  {k}: {v}\n")
            f.write("\nMetrics History:\n")
            for entry in self.metrics_history:
                step = entry["step"]
                mets = entry["metrics"]
                f.write(f"  Step {step}:\n")
                for mk, mv in mets.items():
                    f.write(f"    {mk}: {mv}\n")
            if self.feature_mask and self.feature_names:
                selected = [name for m, name in zip(self.feature_mask, self.feature_names) if m == 1]
                f.write("\nFeature Selection:\n")
                f.write(f"  Total Original Features: {len(self.feature_names)}\n")
                f.write(f"  Selected Features Count: {len(selected)}\n")
                f.write(f"  Selected List: {selected}\n")
        self.logger.info(f"Text summary saved to: {txt_path}")
