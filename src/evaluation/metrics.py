"""Evaluation metrics module for intrusion detection models.

This module provides functions to calculate accuracy, precision, recall, F1-Score,
and inference latency of deep learning classifiers.
"""

import time
from typing import Dict, Union
import numpy as np

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    inference_time_ms: float = 0.0,
) -> Dict[str, float]:
    """Computes basic model metrics: accuracy, precision, recall, and F1.

    Args:
        y_true: True labels of shape (n_samples,).
        y_pred: Predicted labels of shape (n_samples,).
        inference_time_ms: Cumulative latency for evaluation (in ms).

    Returns:
        A dictionary with metrics names mapping to float values.
    """
    # Ensure binary or single label format
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    if y_true.ndim > 1 and y_true.shape[-1] > 1:
        y_true = np.argmax(y_true, axis=-1)

    if SKLEARN_AVAILABLE:
        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    else:
        # Custom numpy-only implementation to prevent crash when sklearn is absent
        n_samples = len(y_true)
        if n_samples == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "latency_per_sample_ms": 0.0,
            }
        
        # Binary or Multiclass mapping
        correct = np.sum(y_true == y_pred)
        accuracy = float(correct / n_samples)
        
        # Calculate weighted metrics manually
        classes = np.unique(y_true)
        precision_list = []
        recall_list = []
        f1_list = []
        weights = []
        
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            support = np.sum(y_true == c)
            
            p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            
            precision_list.append(p)
            recall_list.append(r)
            f1_list.append(f)
            weights.append(support)
            
        weights_arr = np.array(weights)
        total_weight = np.sum(weights_arr)
        
        if total_weight > 0:
            precision = float(np.sum(np.array(precision_list) * weights_arr) / total_weight)
            recall = float(np.sum(np.array(recall_list) * weights_arr) / total_weight)
            f1 = float(np.sum(np.array(f1_list) * weights_arr) / total_weight)
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0

    # Mean latency per sample
    n_samples = max(1, len(y_true))
    latency_per_sample = float(inference_time_ms / n_samples)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_per_sample_ms": latency_per_sample,
    }


def evaluate_with_latency(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, float]:
    """Evaluates a model's prediction metrics alongside its average inference latency.

    Args:
        model: Trained model with predict() interface.
        X_val: Validation dataset inputs.
        y_val: Validation labels.

    Returns:
        A dictionary containing classification metrics and average latency.
    """
    start_time = time.perf_counter()
    y_pred_probs = model.predict(X_val)
    end_time = time.perf_counter()

    elapsed_ms = (end_time - start_time) * 1000.0
    
    if y_pred_probs.shape[-1] > 1:
        y_pred = np.argmax(y_pred_probs, axis=-1)
    else:
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    return calculate_metrics(y_val, y_pred, inference_time_ms=elapsed_ms)
