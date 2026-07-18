"""Data loader and preprocessing module for the SWaT dataset.

Note: The Secure Water Treatment (SWaT) dataset requires manual download authorization.
Researchers must request access directly from the iTrust Centre for Research in
Cyber Security, Singapore University of Technology and Design (SUTD).
"""

import os
from typing import Tuple, List, Any, Optional
import numpy as np
import pandas as pd

# Safe import wrapper for scikit-learn classes
try:
    from sklearn.model_selection import train_test_split as sklearn_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

    class StandardScaler:
        """Fallback StandardScaler if scikit-learn is not installed."""
        def __init__(self) -> None:
            self.mean_ = None
            self.scale_ = None
        def fit(self, X: np.ndarray) -> "StandardScaler":
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0) + 1e-8
            return self
        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            self.fit(X)
            return self.transform(X)
        def transform(self, X: np.ndarray) -> np.ndarray:
            return (X - self.mean_) / self.scale_

    def sklearn_split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fallback train_test_split if scikit-learn is not installed."""
        np.random.seed(random_state)
        n_samples = len(X)
        shuffled_indices = np.random.permutation(n_samples)
        split_idx = int(n_samples * (1 - test_size))
        train_idx = shuffled_indices[:split_idx]
        test_idx = shuffled_indices[split_idx:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class SWaTLoader:
    """Handles loading and preprocessing of SWaT industrial water treatment dataset."""

    def __init__(self) -> None:
        """Initializes the loader and creates standard sensor/actuator feature listings."""
        # SWaT dataset contains 51 sensor and actuator values
        self.feature_names: List[str] = [
            "FIT101", "LIT101", "MV101", "P101", "P102", "FIT201", "MCA201",
            "LIT301", "MV301", "MV302", "MV303", "MV304", "P301", "P302",
            "AIT401", "AIT402", "FIT401", "LIT401", "P401", "P402", "P403",
            "P404", "UV401", "AIT501", "AIT502", "AIT503", "AIT504", "FIT501",
            "FIT502", "FIT503", "FIT504", "PIT501", "PIT502", "PIT503", "P501",
            "P502", "P503", "P504", "P505", "P506", "P507", "P508", "P509",
            "P510", "P511", "P512", "P513", "P514", "P515", "P516", "P517"
        ]

    def load(self, path: str) -> pd.DataFrame:
        """Loads raw SWaT CSV dataset from disk.

        If the file does not exist, generates a synthetic mock dataframe.

        Args:
            path: Target location of the SWaT CSV file.

        Returns:
            The loaded DataFrame.
        """
        if not os.path.exists(path):
            print(f"SWaT dataset file not found at {path}. Creating mock dataset.")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._create_dummy_file(path)

        df = pd.read_csv(path)
        return df

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Cleans and extracts features and labels from the DataFrame.

        Removes Timestamp column and maps label columns.

        Args:
            df: Input DataFrame loaded from CSV.

        Returns:
            A tuple of (X, y) as numpy arrays.
        """
        df_clean = df.copy()
        
        # Strip trailing/leading whitespaces from column headers
        df_clean.columns = df_clean.columns.str.strip()
        
        # Remove timestamp
        if "Timestamp" in df_clean.columns:
            df_clean = df_clean.drop(columns=["Timestamp"])
            
        # Standard label col is Normal/Attack
        label_col = "Normal/Attack"
        if label_col not in df_clean.columns:
            # Fall back to the last column
            label_col = df_clean.columns[-1]

        # Extract features and targets
        X_df = df_clean.drop(columns=[label_col])
        
        # Convert targets to binary (0 for Normal, 1 for Attack)
        y = (df_clean[label_col].astype(str).str.strip().str.lower() != "normal").astype(int).values
        
        # Store column names if they match expected feature count
        if len(X_df.columns) == 51:
            self.feature_names = list(X_df.columns)
            
        X = X_df.values
        return X, y

    def get_feature_names(self) -> List[str]:
        """Returns the list of feature names for SWaT.

        Returns:
            A list of 51 feature name strings.
        """
        return self.feature_names

    def train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits the SWaT data into training and test splits.

        Args:
            X: Features matrix.
            y: Binary target label array.
            test_size: Test split ratio size.

        Returns:
            A tuple of (X_train, X_test, y_train, y_test).
        """
        return sklearn_split(X, y, test_size=test_size, random_state=42, stratify=y)

    def normalize(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fits StandardScaler on training data and transforms both splits.

        Args:
            X_train: Training inputs.
            X_test: Test inputs.

        Returns:
            A tuple of normalized (X_train_scaled, X_test_scaled).
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def _create_dummy_file(self, filepath: str) -> None:
        """Generates synthetic SWaT CSV structure for testing."""
        np.random.seed(42)
        n_samples = 100
        data = np.random.rand(n_samples, 51)
        columns = self.feature_names
        
        df = pd.DataFrame(data, columns=columns)
        df["Timestamp"] = pd.date_range("2026-01-01", periods=n_samples, freq="s")
        
        # Add labels
        labels = np.random.choice(["Normal", "Attack"], size=n_samples, p=[0.8, 0.2])
        df["Normal/Attack"] = labels
        
        df.to_csv(filepath, index=False)
