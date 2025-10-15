"""
PyTorch Neural Network Models for BC

Defines MLP architectures for behavior cloning with different complexity levels.
Supports both simple and large architectures with regularization techniques.
"""

try:
    import torch
    import torch.nn as nn
except Exception as e:
    torch = None
    nn = None


class TorchMLP(nn.Module if nn else object):
    """
    Simple MLP for BC with GELU activations and Tanh output.
    
    Architecture: Input -> Hidden -> Hidden -> Output
    - Uses GELU activation for smooth gradients
    - Tanh output ensures actions stay in [-1, 1] range
    - Lightweight design for fast training
    """
    
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        """
        Initialize simple MLP.
        
        Args:
            in_dim: Input dimension (observation space)
            out_dim: Output dimension (action space)
            hidden: Hidden layer size
        """
        if torch is None:
            raise ImportError("PyTorch not available. Install torch or use --backend jax.")
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), 
            nn.GELU(),
            nn.Linear(hidden, hidden), 
            nn.GELU(),
            nn.Linear(hidden, out_dim), 
            nn.Tanh()
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.net(x)


class TorchMLPLarge(nn.Module if nn else object):
    """
    Large MLP with regularization for robust BC training.
    
    Architecture: Input -> Hidden -> Hidden -> Hidden//2 -> Output
    - Batch normalization for stable training
    - Dropout for regularization
    - Deeper network for complex control policies
    - GELU activations for smooth gradients
    - Tanh output ensures actions stay in [-1, 1] range
    """
    
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, dropout: float = 0.1):
        """
        Initialize large MLP with regularization.
        
        Args:
            in_dim: Input dimension (observation space)
            out_dim: Output dimension (action space)
            hidden: Hidden layer size
            dropout: Dropout probability
        """
        if torch is None:
            raise ImportError("PyTorch not available. Install torch or use --backend jax.")
        super().__init__()
        
        self.net = nn.Sequential(
            # First hidden layer
            nn.Linear(in_dim, hidden), 
            nn.BatchNorm1d(hidden), 
            nn.GELU(), 
            nn.Dropout(dropout),
            
            # Second hidden layer
            nn.Linear(hidden, hidden), 
            nn.BatchNorm1d(hidden), 
            nn.GELU(), 
            nn.Dropout(dropout),
            
            # Third hidden layer (smaller)
            nn.Linear(hidden, hidden//2), 
            nn.BatchNorm1d(hidden//2), 
            nn.GELU(), 
            nn.Dropout(dropout),
            
            # Output layer
            nn.Linear(hidden//2, out_dim), 
            nn.Tanh()
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.net(x)