"""
Visualization utilities for Behavior Cloning (BC) training and evaluation.

This module provides functions for plotting training metrics, predictions, and label distributions
for the passive walker behavior cloning implementation.
"""

from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np


def plot_label_hist(
    labels: np.ndarray,
    bins: int = 40,
    save: Optional[str] = None,
    joint_name: str = "Hip",
    title: str = "FSM Joint-Action Distribution"
) -> None:
    """Plot histogram of recorded FSM joint-action labels.

    Args:
        labels: Array of shape (N, 1) or (N,) containing action values
        bins: Number of histogram bins
        save: Optional path to save the plot
        joint_name: Name of the joint being plotted
        title: Plot title
    """
    vals = labels.flatten()
    plt.figure()
    plt.hist(vals, bins=bins, alpha=0.8)
    plt.title(title)
    plt.xlabel(f"{joint_name} action value")
    plt.ylabel("Count")
    
    if save:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"[utils] saved label hist → {save}")
    else:
        plt.show()


def plot_loss_curve(
    losses: List[float],
    save: Optional[str] = None,
    title: str = "BC Training Loss Curve"
) -> None:
    """Plot behavior cloning training loss over epochs.

    Args:
        losses: List of loss values, one per epoch
        save: Optional path to save the plot
        title: Plot title
    """
    plt.figure()
    plt.plot(np.arange(1, len(losses) + 1), losses, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, linestyle='--', alpha=0.4)
    
    if save:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"[utils] saved loss curve → {save}")
    else:
        plt.show()


def plot_pred_vs_true(
    true: np.ndarray,
    pred: np.ndarray,
    save: Optional[str] = None,
    title: str = "BC Prediction vs. True FSM Action"
) -> None:
    """Plot scatter of BC predictions against ground-truth FSM labels.

    Args:
        true: Array of shape (N, 1) or (N,) containing true action values
        pred: Array of shape (N, 1) or (N,) containing predicted action values
        save: Optional path to save the plot
        title: Plot title
    """
    t = true.flatten()
    p = pred.flatten()
    
    plt.figure()
    plt.scatter(t, p, s=4, alpha=0.6)
    
    # Plot diagonal line for reference
    mn, mx = min(t.min(), p.min()), max(t.max(), p.max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1, color='k')
    
    plt.title(title)
    plt.xlabel("True action")
    plt.ylabel("Predicted action")
    
    if save:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"[utils] saved pred-vs-true → {save}")
    else:
        plt.show()
