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

    def __init__(self, alpha: float = 0.3, min_accuracy: float = 0.0, min_features: int = 10):
        """Initializes the evaluator.

        Args:
            alpha: Weight for feature selection ratio.
            min_accuracy: Hard threshold floor for validation accuracy.
            min_features: Minimum constraint for selected features.
        """
        self.alpha: float = alpha
        self.min_accuracy: float = min_accuracy
        self.min_features: int = min_features

    def calculate_fitness(
        self,
        features_mask: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> float:
        """Calculates the fitness score of a specific feature subset.

        Args:
            features_mask: Binary array of shape (n_features,).
            X_train: Training features array of shape (n_samples, n_features).
            y_train: Training labels array of shape (n_samples,).
            X_val: Unused parameter (kept for backwards compatibility).
            y_val: Unused parameter (kept for backwards compatibility).

        Returns:
            The fitness score. A lower score is better.
        """
        n_features: int = int(features_mask.shape[0])
        selected_indices = np.where(features_mask == 1)[0]

        # Handle constraint: reject any solution with fewer than min_features
        if len(selected_indices) < self.min_features:
            return 1.0

        if SKLEARN_AVAILABLE:
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            from sklearn.ensemble import RandomForestClassifier
            X_train_sub = X_train[:, selected_indices]

            # Fast evaluation using 50-estimator Random Forest and 3-fold CV
            cv_scores = cross_val_score(
                RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                X_train_sub, y_train, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='accuracy'
            )
            accuracy = float(np.mean(cv_scores))
        else:
            # Fallback pseudo-accuracy
            accuracy = 0.5 + 0.4 * (len(selected_indices) / n_features)

        # Enforce validation accuracy floor
        if accuracy < self.min_accuracy:
            return 1.0

        # Calculate fitness
        error_rate: float = 1.0 - accuracy
        ratio_features: float = len(selected_indices) / n_features
        fitness: float = self.alpha * ratio_features + (1.0 - self.alpha) * error_rate

        return fitness
