"""Data loader and preprocessing module for the BATADAL dataset.

Note: The BATADAL (Battle of the Attack Detection Algorithms) dataset is
available for download at http://www.batadal.net or through the researchers'
associated public repositories.
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


class BATADALLoader:
    """Handles loading and preprocessing of the BATADAL water network SCADA dataset."""

    def __init__(self) -> None:
        """Initializes the loader and creates standard sensor/actuator feature listings."""
        # BATADAL dataset contains 43 sensor and actuator values
        self.feature_names: List[str] = [
            "L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7", "F_PU1",
            "S_PU1", "F_PU2", "S_PU2", "F_PU3", "S_PU3", "F_PU4", "S_PU4",
            "F_PU5", "S_PU5", "F_PU6", "S_PU6", "F_PU7", "S_PU7", "F_PU8",
            "S_PU8", "F_PU9", "S_PU9", "F_PU10", "S_PU10", "F_PU11", "S_PU11",
            "F_V2", "S_V2", "P_J280", "P_J269", "P_J300", "P_J256", "P_J289",
            "P_J415", "P_J302", "P_J306", "P_J342", "P_J14", "P_J422", "P_J12"
        ]

    def load(self, path: str) -> pd.DataFrame:
        """Loads raw BATADAL CSV dataset from disk.

        If the file does not exist, generates a synthetic mock dataset.

        Args:
            path: Target location of the BATADAL CSV file.

        Returns:
            The loaded DataFrame.
        """
        if not os.path.exists(path):
            print(f"BATADAL dataset file not found at {path}. Creating mock dataset.")
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
        for ts_col in ["DATETIME", "Datetime", "datetime", "Timestamp", "timestamp"]:
            if ts_col in df_clean.columns:
                df_clean = df_clean.drop(columns=[ts_col])
                
        # Standard label col is ATT_FLAG
        label_col = "ATT_FLAG"
        if label_col not in df_clean.columns:
            # Fall back to the last column
            label_col = df_clean.columns[-1]

        # Extract features and targets
        X_df = df_clean.drop(columns=[label_col])
        
        # Convert targets to binary: BATADAL labels are often -999 (no label), 0 (normal), 1 (attack)
        # or -1 (normal), 1 (attack). We map 1 to 1 (Attack) and everything else to 0 (Normal).
        y = (df_clean[label_col].values == 1).astype(int)
        
        # Store column names if they match expected feature count
        if len(X_df.columns) == 43:
            self.feature_names = list(X_df.columns)
            
        X = X_df.values
        return X, y

    def get_feature_names(self) -> List[str]:
        """Returns the list of feature names for BATADAL.

        Returns:
            A list of 43 feature name strings.
        """
        return self.feature_names

    def train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits the BATADAL data into training and test splits.

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
        """Generates synthetic BATADAL CSV structure for testing."""
        np.random.seed(42)
        n_samples = 100
        data = np.random.rand(n_samples, 43)
        columns = self.feature_names
        
        df = pd.DataFrame(data, columns=columns)
        df["DATETIME"] = pd.date_range("2026-01-01", periods=n_samples, freq="h")
        
        # Add labels (-1 or 1, or 0 or 1)
        labels = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
        df["ATT_FLAG"] = labels
        
        df.to_csv(filepath, index=False)


# Backwards compatibility alias
BatadalLoader = BATADALLoader
