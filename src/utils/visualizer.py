"""Visualization utilities for evaluating model performance.

This module provides the ExperimentVisualizer class and separate helper functions
to plot confusion matrices, ROC curves, feature importances, optimization
convergence, and training histories. All figures are saved to disk.
"""

import os
from typing import List, Optional, Union, Dict, Any
import numpy as np

# Try importing matplotlib safely
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Try importing sklearn metrics for ROC curve calculations
try:
    from sklearn.metrics import roc_curve, auc, confusion_matrix as sklearn_cm
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    save_path: Optional[str] = None
) -> Optional[Any]:
    """Plots and saves/shows a confusion matrix.

    Args:
        cm: Confusion matrix array.
        classes: List of class name strings.
        save_path: Filepath where the plot is saved.

    Returns:
        The matplotlib Figure object or None if matplotlib is not available.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib is not available. Confusion matrix plotting skipped.")
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                f.write("Placeholder confusion matrix plot.")
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells with values
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
            
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    return fig


def plot_convergence_history(
    history: List[float],
    save_path: Optional[str] = None
) -> Optional[Any]:
    """Plots BWOA convergence fitness history over iterations.

    Args:
        history: Fitness score list per iteration.
        save_path: Filepath where the plot is saved.

    Returns:
        The matplotlib Figure object or None if matplotlib is not available.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib is not available. Convergence history plotting skipped.")
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                f.write("Placeholder convergence history plot.")
        return None

    fig = plt.figure(figsize=(8, 5))
    plt.plot(history, marker="o", linestyle="-", color="b")
    plt.title("BWOA Feature Selection Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness Score")
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    return fig


class ExperimentVisualizer:
    """Orchestrates creation and saving of all metric and evaluation plots."""

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_path: str
    ) -> Optional[Any]:
        """Computes and plots a confusion matrix.

        Args:
            y_true: Ground truth target labels.
            y_pred: Predicted class labels.
            class_names: Name labels for target classes.
            save_path: Filepath to save figure.

        Returns:
            The matplotlib Figure object or None if matplotlib is not available.
        """
        # Ensure label indexes are used
        if y_true.ndim > 1 and y_true.shape[-1] > 1:
            y_true = np.argmax(y_true, axis=-1)
        if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
            y_pred = np.argmax(y_pred, axis=-1)

        if SKLEARN_AVAILABLE:
            cm = sklearn_cm(y_true, y_pred)
        else:
            n_classes = len(class_names)
            cm = np.zeros((n_classes, n_classes), dtype=int)
            for t, p in zip(y_true, y_pred):
                if 0 <= t < n_classes and 0 <= p < n_classes:
                    cm[int(t), int(p)] += 1

        return plot_confusion_matrix(cm, class_names, save_path)

    @staticmethod
    def plot_roc_curves(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str],
        save_path: str
    ) -> Optional[Any]:
        """Plots ROC curves for single-class or multi-class predictions.

        Args:
            y_true: Ground truth target labels.
            y_prob: Predicted probability array.
            class_names: Name labels for target classes.
            save_path: Filepath to save figure.

        Returns:
            The matplotlib Figure object or None if matplotlib is not available.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib is not available. ROC curves plotting skipped.")
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w") as f:
                    f.write("Placeholder ROC curves plot.")
            return None

        fig = plt.figure(figsize=(8, 6))
        
        # Check if binary or multi-class
        n_classes = len(class_names)
        
        # Ensure ground truth is integer array
        if y_true.ndim > 1 and y_true.shape[-1] > 1:
            y_true_indices = np.argmax(y_true, axis=-1)
        else:
            y_true_indices = y_true.astype(int)

        if SKLEARN_AVAILABLE:
            if n_classes == 2 or (y_prob.ndim == 1 or y_prob.shape[-1] == 1):
                # Binary classification
                probs = y_prob[:, 1] if (y_prob.ndim > 1 and y_prob.shape[-1] > 1) else y_prob
                fpr, tpr, _ = roc_curve(y_true_indices, probs)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
            else:
                # Multi-class classification
                for i in range(n_classes):
                    # One-vs-rest binary targets
                    y_true_binary = (y_true_indices == i).astype(int)
                    if np.sum(y_true_binary) > 0:
                        fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, lw=2, label=f"Class {class_names[i]} (area = {roc_auc:.2f})")
        else:
            # Fallback mock lines
            plt.plot([0, 1], [0, 1], "k--", label="Default baseline")
            plt.plot([0, 0.1, 1], [0, 0.9, 1], lw=2, label="Mock classifier (area = 0.90)")

        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curves")
        plt.legend(loc="lower right")
        plt.grid(True)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        return fig

    @staticmethod
    def plot_feature_importance(
        feature_mask: np.ndarray,
        feature_names: List[str],
        save_path: str
    ) -> Optional[Any]:
        """Plots a horizontal bar chart showing BWOA selected features.

        Args:
            feature_mask: Binary mask array (1: selected, 0: not selected).
            feature_names: Names of all features.
            save_path: Filepath to save figure.

        Returns:
            The matplotlib Figure object or None if matplotlib is not available.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib is not available. Feature importance plotting skipped.")
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w") as f:
                    f.write("Placeholder feature importance plot.")
            return None

        selected_indices = np.where(feature_mask == 1)[0]
        selected_names = [feature_names[idx] for idx in selected_indices]
        
        fig = plt.figure(figsize=(10, 8))
        
        if not selected_names:
            plt.text(0.5, 0.5, "No Features Selected", ha="center", va="center")
        else:
            y_pos = np.arange(len(selected_names))
            # Just plotting dummy weights (constant 1.0) since mask is binary selection
            plt.barh(y_pos, np.ones(len(selected_names)), align="center", color="teal")
            plt.yticks(y_pos, selected_names)
            plt.xlabel("Selection Status (1: Selected)")
            plt.title("BWOA Selected Feature Subsets")
            plt.grid(axis="x", linestyle="--")
            
        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        return fig

    @staticmethod
    def plot_fitness_history(fitness_history: List[float], save_path: str) -> Optional[Any]:
        """Plots the BWOA best fitness convergence curve.

        Args:
            fitness_history: Record of best fitness per iteration.
            save_path: Filepath to save figure.

        Returns:
            The matplotlib Figure object or None if matplotlib is not available.
        """
        return plot_convergence_history(fitness_history, save_path)

    @staticmethod
    def plot_training_history(history: Union[Dict[str, Any], Any], save_path: str) -> Optional[Any]:
        """Plots learning curves for validation/training accuracy and loss.

        Args:
            history: Dictionary or Keras History object.
            save_path: Filepath to save figure.

        Returns:
            The matplotlib Figure object or None if matplotlib is not available.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib is not available. Training history plotting skipped.")
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w") as f:
                    f.write("Placeholder training history plot.")
            return None

        hist = history.history if hasattr(history, "history") else history
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        if "loss" in hist:
            ax1.plot(hist["loss"], label="Train Loss", color="tomato")
        if "val_loss" in hist:
            ax1.plot(hist["val_loss"], label="Val Loss", color="royalblue")
        ax1.set_title("Model Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.grid(True)

        # Accuracy plot
        acc_key = "accuracy" if "accuracy" in hist else "acc"
        val_acc_key = "val_accuracy" if "val_accuracy" in hist else "val_acc"
        
        if acc_key in hist:
            ax2.plot(hist[acc_key], label="Train Accuracy", color="tomato")
        if val_acc_key in hist:
            ax2.plot(hist[val_acc_key], label="Val Accuracy", color="royalblue")
        ax2.set_title("Model Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="lower right")
        ax2.grid(True)
        
        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        return fig
