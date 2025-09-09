"""
Torch MLP scaffold (no training logic here).
"""

try:
    import torch
    import torch.nn as nn
except Exception as e:
    torch = None
    nn = None


class TorchMLP(nn.Module if nn else object):
    """Simple MLP with GELU activations and Tanh output head."""
    
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        if torch is None:
            raise ImportError("PyTorch not available. Install torch or use --backend jax.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim), nn.Tanh()
        )


class TorchMLPLarge(nn.Module if nn else object):
    """Larger MLP with dropout and batch normalization for robustness."""
    
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, dropout: float = 0.1):
        if torch is None:
            raise ImportError("PyTorch not available. Install torch or use --backend jax.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, out_dim), nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
