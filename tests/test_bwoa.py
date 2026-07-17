"""Unit tests for the Binary Whale Optimization Algorithm (BWOA).

This module contains test cases to verify the optimization execution, V-shaped
transfer function math, and output shapes of BWOA.
"""

import unittest
import numpy as np
from src.optimization.bwoa import BinaryWhaleOptimizer
from src.optimization.fitness import FeatureFitnessEvaluator


class TestBinaryWhaleOptimizer(unittest.TestCase):
    """Tests the BinaryWhaleOptimizer class methods and properties."""

    def setUp(self) -> None:
        """Sets up test data and optimization parameter instances."""
        np.random.seed(42)
        self.n_samples: int = 40
        self.n_features: int = 10
        self.X_train = np.random.rand(self.n_samples, self.n_features)
        self.y_train = np.random.randint(0, 2, size=(self.n_samples,))
        self.X_val = np.random.rand(20, self.n_features)
        self.y_val = np.random.randint(0, 2, size=(20,))
        
        self.evaluator = FeatureFitnessEvaluator(alpha=0.88)
        self.optimizer = BinaryWhaleOptimizer(
            n_agents=5,
            n_features=self.n_features,
            max_iter=3,
            fitness_fn=self.evaluator.calculate_fitness
        )

    def test_transfer_function_bounds(self) -> None:
        """Verifies the V-shaped transfer function maps values to [0, 1]."""
        velocities = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        probabilities = self.optimizer._transfer_function(velocities)
        
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))
        self.assertEqual(probabilities[2], 0.0)  # V-shape at zero is 0.0

    def test_optimization_loop(self) -> None:
        """Verifies that optimize returns a valid mask and fitness history."""
        best_mask, history = self.optimizer.optimize(
            self.X_train, self.y_train, self.X_val, self.y_val
        )
        
        self.assertEqual(best_mask.shape, (self.n_features,))
        self.assertEqual(len(history), 3)
        self.assertTrue(np.all((best_mask == 0) | (best_mask == 1)))


if __name__ == "__main__":
    unittest.main()
