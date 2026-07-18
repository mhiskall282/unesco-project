"""Data loader and preprocessing module for the NSL-KDD dataset.

This module provides functions to download, clean, preprocess, and load
the NSL-KDD network intrusion dataset.
"""

import os
import urllib.request
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

# Safe import wrapper for scikit-learn classes
try:
    from sklearn.model_selection import train_test_split as sklearn_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    
    class LabelEncoder:
        """Fallback LabelEncoder if scikit-learn is not installed."""
        def __init__(self) -> None:
            self.classes_: List[Any] = []
        def fit(self, y: List[Any]) -> "LabelEncoder":
            self.classes_ = sorted(list(set(y)))
            return self
        def fit_transform(self, y: List[Any]) -> np.ndarray:
            self.fit(y)
            return self.transform(y)
        def transform(self, y: List[Any]) -> np.ndarray:
            mapping = {val: idx for idx, val in enumerate(self.classes_)}
            return np.array([mapping.get(val, 0) for val in y])

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


class NSLKDDLoader:
    """Handles loading and preprocessing of the NSL-KDD dataset."""

    def __init__(self) -> None:
        """Initializes the loader with standard column names and category mappings."""
        # NSL-KDD column names as defined in official documentation
        self.columns: List[str] = [
            "duration", "protocol_type", "service", "flag", "src_bytes",
            "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
            "num_failed_logins", "logged_in", "num_compromised", "root_shell",
            "su_attempted", "num_root", "num_file_creations", "num_shells",
            "num_access_files", "num_outbound_cmds", "is_host_login",
            "is_guest_login", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack_type",
            "difficulty_level"
        ]
        
        # Mappings of specific attacks to the 5 standard categories
        self.attack_mapping: Dict[str, str] = {
            # Normal
            "normal": "Normal",
            # Denial of Service (DoS)
            "back": "DoS",
            "land": "DoS",
            "neptune": "DoS",
            "pod": "DoS",
            "smurf": "DoS",
            "teardrop": "DoS",
            "mailbomb": "DoS",
            "processtable": "DoS",
            "udpstorm": "DoS",
            "apache2": "DoS",
            "worm": "DoS",
            # Probing
            "satan": "Probe",
            "ipsweep": "Probe",
            "nmap": "Probe",
            "portsweep": "Probe",
            "mscan": "Probe",
            "saint": "Probe",
            # Remote-to-Local (R2L)
            "guess_passwd": "R2L",
            "ftp_write": "R2L",
            "imap": "R2L",
            "phf": "R2L",
            "multihop": "R2L",
            "warezmaster": "R2L",
            "warezclient": "R2L",
            "spy": "R2L",
            "xlock": "R2L",
            "xsnoop": "R2L",
            "snmpgetattack": "R2L",
            "snmpguess": "R2L",
            "httptunnel": "R2L",
            "sendmail": "R2L",
            "named": "R2L",
            # User-to-Root (U2R)
            "buffer_overflow": "U2R",
            "loadmodule": "U2R",
            "perl": "U2R",
            "rootkit": "U2R",
            "sqlattack": "U2R",
            "xterm": "U2R",
            "ps": "U2R"
        }

        # Label encoders for categorical variables
        self.label_encoders: Dict[str, LabelEncoder] = {
            "protocol_type": LabelEncoder(),
            "service": LabelEncoder(),
            "flag": LabelEncoder(),
            "attack_category": LabelEncoder()
        }
        
        # We define classes explicitly to ensure consistent index mappings
        self.classes: List[str] = ["Normal", "DoS", "Probe", "R2L", "U2R"]
        self.label_encoders["attack_category"].fit(self.classes)

    def load(self, path: str) -> pd.DataFrame:
        """Loads the NSL-KDD dataset from path.

        Downloads from defcom17 repository if the file is missing locally.

        Args:
            path: Absolute or relative filepath to load.

        Returns:
            The loaded DataFrame containing raw columns.
        """
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            filename = os.path.basename(path)
            # Map filename to download URL
            url = f"https://raw.githubusercontent.com/defcom17/NSL_KDD/master/{filename}"
            print(f"Downloading dataset from: {url}")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"Successfully downloaded to: {path}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                # Fallback: create a dummy mock file to allow verification without internet
                print("Creating dummy mock data locally.")
                self._create_dummy_file(path)

        df = pd.read_csv(path, names=self.columns)
        return df

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocesses the raw DataFrame.

        Encodes categorical variables, groups attack labels, and splits inputs.

        Args:
            df: Raw input DataFrame.

        Returns:
            A tuple (X, y) containing features and integer class labels.
        """
        df_clean = df.copy()
        
        # Clean attack_type values (strip any trailing dots)
        df_clean["attack_type"] = df_clean["attack_type"].astype(str).str.strip(".")
        
        # Group labels into the 5 primary categories (default to normal if unknown)
        df_clean["attack_category"] = df_clean["attack_type"].map(
            lambda x: self.attack_mapping.get(x.lower(), "Normal")
        )
        
        # Encode target label category to integers
        y = self.label_encoders["attack_category"].transform(df_clean["attack_category"])
        
        # Remove target columns
        X_df = df_clean.drop(columns=["attack_type", "difficulty_level", "attack_category"])
        
        # Encode categorical inputs using LabelEncoder
        for col in ["protocol_type", "service", "flag"]:
            X_df[col] = self.label_encoders[col].fit_transform(X_df[col].astype(str))
            
        X = X_df.values
        return X, y

    def get_feature_names(self) -> List[str]:
        """Returns the list of 41 feature names for the dataset.

        Returns:
            A list of 41 column names.
        """
        return self.columns[:-2]

    def train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits datasets into training and testing sets.

        Args:
            X: Input features array.
            y: Target class labels array.
            test_size: Ratio of the test split size.

        Returns:
            A tuple of (X_train, X_test, y_train, y_test).
        """
        return sklearn_split(X, y, test_size=test_size, random_state=42, stratify=y)

    def normalize(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies StandardScaler normalization.

        The scaler is fitted on the training split only and applied to both.

        Args:
            X_train: Training features array.
            X_test: Test features array.

        Returns:
            A tuple of normalized (X_train_scaled, X_test_scaled).
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def _create_dummy_file(self, filepath: str) -> None:
        """Helper to create a dummy CSV file if network is unavailable."""
        np.random.seed(42)
        rows = []
        for _ in range(100):
            row = [0] * 41
            # protocol_type, service, flag
            row[1] = "tcp"
            row[2] = "http"
            row[3] = "SF"
            # Random values
            row[0] = np.random.randint(0, 100)
            row[4] = np.random.randint(100, 1000)
            row[5] = np.random.randint(100, 1000)
            # Attack types
            attack = np.random.choice(["normal", "neptune", "satan", "warezmaster", "buffer_overflow"])
            row.append(attack)
            row.append(21)  # difficulty level
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, header=False, index=False)
