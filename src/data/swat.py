"""Data loader for the Secure Water Treatment (SWaT) industrial control dataset.

This module handles loading, parsing, and preprocessing of the SWaT dataset.
"""

from typing import Tuple
import numpy as np
import pandas as pd


class SWaTLoader:
    """Handles loading and preprocessing of SWaT dataset."""

    def __init__(self, file_path: str):
        """Initializes the loader with the dataset path.

        Args:
            file_path: Path to the SWaT CSV dataset.
        """
        self.file_path: str = file_path

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Loads and preprocesses the SWaT dataset.

        Returns:
            A tuple of (features, labels) as numpy arrays.
        """
        try:
            df = pd.read_csv(self.file_path)
            # Remove timestamp column if present
            if "Timestamp" in df.columns:
                df = df.drop(columns=["Timestamp"])
            
            # Separate features and labels
            # Standard SWaT label column is typically named 'Normal/Attack' or 'Label'
            label_col = "Normal/Attack" if "Normal/Attack" in df.columns else df.columns[-1]
            X = df.drop(columns=[label_col]).values
            y = (df[label_col] != "Normal").astype(int).values
            return X, y
        except FileNotFoundError:
            print("SWaT dataset file not found. Generating mock data.")
            return self._generate_mock_data()

    def _generate_mock_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates mock SWaT dataset features and labels for testing."""
        np.random.seed(42)
        n_samples, n_features = 150, 51
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, size=(n_samples,))
        return X, y
        
