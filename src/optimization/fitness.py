"""Fitness function module for evaluating feature subsets selected by BWOA.

This module provides a class to calculate the fitness of a chosen subset
of features using a combination of classification accuracy and the ratio of
selected features.
"""

from typing import Tuple, Union
import numpy as np

try:
    from sklearn.tree import DecisionTreeClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class FeatureFitnessEvaluator:
    """Calculates BWOA feature subset fitness using accuracy and size reduction."""

    def __init__(self, alpha: float = 0.88):
        """Initializes the evaluator with alpha weight.

        Args:
            alpha: Weight for accuracy (prioritized). Must be between 0 and 1.
        """
        self.alpha: float = alpha

    def calculate_fitness(
        self,
        features_mask: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Calculates the fitness score of a specific feature subset.

        Args:
            features_mask: Binary array of shape (n_features,) where 1 indicates
                the feature is selected and 0 indicates it is excluded.
            X_train: Training features array of shape (n_samples, n_features).
            y_train: Training labels array of shape (n_samples,).
            X_val: Validation features array of shape (n_samples, n_features).
            y_val: Validation labels array of shape (n_samples,).

        Returns:
            The fitness score. A lower score indicates a better feature subset.
        """
        n_features: int = int(features_mask.shape[0])
        selected_indices = np.where(features_mask == 1)[0]

        # Handle edge case where no features are selected
        if len(selected_indices) == 0:
            return 1.0

        if SKLEARN_AVAILABLE:
            # Filter features based on mask
            X_train_sub = X_train[:, selected_indices]
            X_val_sub = X_val[:, selected_indices]

            # Use a fast classifier (Decision Tree) to evaluate the feature subset
            clf = DecisionTreeClassifier(random_state=42, max_depth=5)
            clf.fit(X_train_sub, y_train)
            accuracy: float = float(clf.score(X_val_sub, y_val))
        else:
            # Fallback deterministic pseudo-accuracy based on selection ratio
            # and mock correlation
            accuracy = 0.5 + 0.4 * (len(selected_indices) / n_features)

        # Calculate fitness
        error_rate: float = 1.0 - accuracy
        ratio_features: float = len(selected_indices) / n_features
        fitness: float = self.alpha * error_rate + (1.0 - self.alpha) * ratio_features

        return fitness
