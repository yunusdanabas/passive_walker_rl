# passive_walker/bc/plotters.py

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def plot_label_hist(labels: np.ndarray, bins: int = 40, save: str = None, joint_name: str = "Hip", title: str = "FSM Joint-Action Distribution"):
    """
    Histogram of the recorded FSM hip-action labels.
    labels: shape (N, 1) or (N,)
    """
    vals = labels.flatten()
    plt.figure()
    plt.hist(vals, bins=bins, alpha=0.8)
    plt.title(title)
    plt.xlabel(f"{joint_name} action value")
    plt.ylabel("Count")
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"[plotters] saved label hist → {save}")
    else:
        plt.show()


def plot_loss_curve(losses: list[float], save: str = None):
    """
    Plot BC training loss over epochs.
    losses: list of loss values, one per epoch.
    """
    plt.figure()
    plt.plot(np.arange(1, len(losses)+1), losses, marker='o')
    plt.title("BC Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, linestyle='--', alpha=0.4)
    if save:
        # Create directory if it doesn't exist
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"[plotters] saved loss curve → {save}")
    else:
        plt.show()


def plot_pred_vs_true(true: np.ndarray, pred: np.ndarray, save: str = None, title: str = "BC Prediction vs. True FSM Action"):
    """
    Scatter of BC predictions vs. ground-truth FSM labels.
    true, pred: arrays of same shape (N, 1) or (N,)
    """
    t = true.flatten()
    p = pred.flatten()
    plt.figure()
    plt.scatter(t, p, s=4, alpha=0.6)
    mn, mx = min(t.min(), p.min()), max(t.max(), p.max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1, color='k')
    plt.title(title)
    plt.xlabel("True action")
    plt.ylabel("Predicted action")
    if save:
        # Create directory if it doesn't exist
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"[plotters] saved pred-vs-true → {save}")
    else:
        plt.show()
