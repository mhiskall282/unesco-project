"""Data loader for the BATADAL water distribution system cyberattack dataset.

This module handles loading, parsing, and preprocessing of BATADAL data.
"""

from typing import Tuple
import numpy as np
import pandas as pd


class BatadalLoader:
    """Handles loading and preprocessing of BATADAL dataset."""

    def __init__(self, file_path: str):
        """Initializes the loader with the dataset path.

        Args:
            file_path: Path to the BATADAL CSV dataset.
        """
        self.file_path: str = file_path

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Loads and preprocesses the BATADAL dataset.

        Returns:
            A tuple of (features, labels) as numpy arrays.
        """
        try:
            df = pd.read_csv(self.file_path)
            # Remove timestamp column if present
            if "DATETIME" in df.columns:
                df = df.drop(columns=["DATETIME"])
            
            # Standard BATADAL label column is 'ATT_FLAG'
            label_col = "ATT_FLAG" if "ATT_FLAG" in df.columns else df.columns[-1]
            X = df.drop(columns=[label_col]).values
            y = df[label_col].values
            return X, y
        except FileNotFoundError:
            print("BATADAL dataset file not found. Generating mock data.")
            return self._generate_mock_data()

    def _generate_mock_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates mock BATADAL dataset features and labels for testing."""
        np.random.seed(42)
        n_samples, n_features = 120, 43
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, size=(n_samples,))
        return X, y
