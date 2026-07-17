"""Unit tests for the performance evaluation metrics module.

This module contains test cases to verify classification score calculators and
average latency estimations.
"""

import unittest
import numpy as np
from src.evaluation.metrics import calculate_metrics


class TestMetrics(unittest.TestCase):
    """Tests the calculate_metrics helper functions and latency estimates."""

    def test_calculate_metrics_exact(self) -> None:
        """Verifies metrics on a simple, known prediction sequence."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        # 4 out of 5 correct => accuracy = 0.8
        # TP = 2, FP = 0, FN = 1 => precision = 1.0, recall = 2/3 = 0.666
        metrics = calculate_metrics(y_true, y_pred, inference_time_ms=10.0)
        
        self.assertAlmostEqual(metrics["accuracy"], 0.8)
        self.assertAlmostEqual(metrics["latency_per_sample_ms"], 2.0)  # 10.0 / 5
        self.assertTrue(metrics["precision"] > 0.0)
        self.assertTrue(metrics["recall"] > 0.0)
        self.assertTrue(metrics["f1"] > 0.0)


if __name__ == "__main__":
    unittest.main()
