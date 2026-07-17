"""Visualization utilities for evaluating model performance.

This module provides functions to plot confusion matrices, ROC curves,
and learning/BWOA optimization paths.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    save_path: Optional[str] = None
) -> None:
    """Plots and saves/shows a confusion matrix.

    Args:
        cm: Confusion matrix array.
        classes: List of class name strings.
        save_path: Filepath where the plot is saved.
    """
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
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_convergence_history(
    history: List[float],
    save_path: Optional[str] = None
) -> None:
    """Plots BWOA convergence fitness history over iterations.

    Args:
        history: Fitness score list per iteration.
        save_path: Filepath where the plot is saved.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history, marker="o", linestyle="-", color="b")
    plt.title("BWOA Feature Selection Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness Score")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
