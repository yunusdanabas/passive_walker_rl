"""
PyTorch Temporal Models for Behavior Cloning

Implements LSTM, GRU, and BiLSTM with proper weight initialization and support
for both sequence training and single-step inference.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    torch = None
    nn = None
    F = None


class TorchLSTM(nn.Module if nn else object):
    """
    LSTM for behavior cloning.
    
    Processes temporal sequences with LSTM layers and outputs actions in [-1, 1].
    Supports variable-length sequences and single-step inference.
    """
    
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int = 128, 
                 num_layers: int = 1, dropout: float = 0.1, bidirectional: bool = False):
        """
        Initialize LSTM model.
        
        Args:
            in_dim: Input dimension (observation space)
            out_dim: Output dimension (action space)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        if torch is None:
            raise ImportError("PyTorch not available. Install torch or use --backend jax.")
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layer
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.output_layer = nn.Linear(lstm_output_size, out_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
                - For sequences: (batch, seq_len, in_dim)
                - For single step: (batch, in_dim)
            hidden: Optional hidden state tuple (h_0, c_0)
            
        Returns:
            Output tensor with actions in [-1, 1] range
            If sequence input: (batch, seq_len, out_dim)
            If single step: (batch, out_dim)
        """
        # Handle single-step inference
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            single_step = True
        else:
            single_step = False
        
        # LSTM forward pass
        lstm_out, new_hidden = self.lstm(x, hidden)
        
        # Apply output layer
        output = self.output_layer(lstm_out)
        output = torch.tanh(output)
        
        # Remove sequence dimension for single-step inference
        if single_step:
            output = output.squeeze(1)
        
        return output, new_hidden
    
    def get_initial_hidden(self, batch_size, device=None):
        """Get initial hidden state."""
        if device is None:
            device = next(self.parameters()).device
        
        num_directions = 2 if self.bidirectional else 1
        h_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)


class TorchGRU(nn.Module if nn else object):
    """
    GRU for BC with configurable architecture and temporal modeling.
    
    Architecture: Input -> GRU -> Hidden -> Output
    - Supports variable sequence lengths with proper padding/masking
    - Tanh output ensures actions stay in [-1, 1] range
    - Configurable hidden size, layers, dropout
    - Both sequence and single-step inference modes
    """
    
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int = 128, 
                 num_layers: int = 1, dropout: float = 0.1, bidirectional: bool = False):
        """
        Initialize GRU model.
        
        Args:
            in_dim: Input dimension (observation space)
            out_dim: Output dimension (action space)
            hidden_size: GRU hidden dimension
            num_layers: Number of GRU layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
        """
        if torch is None:
            raise ImportError("PyTorch not available. Install torch or use --backend jax.")
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layer
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        self.output_layer = nn.Linear(gru_output_size, out_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
                - For sequences: (batch, seq_len, in_dim)
                - For single step: (batch, in_dim)
            hidden: Optional hidden state tensor
            
        Returns:
            Output tensor with actions in [-1, 1] range
            If sequence input: (batch, seq_len, out_dim)
            If single step: (batch, out_dim)
        """
        # Handle single-step inference
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            single_step = True
        else:
            single_step = False
        
        # GRU forward pass
        gru_out, new_hidden = self.gru(x, hidden)
        
        # Apply output layer
        output = self.output_layer(gru_out)
        output = torch.tanh(output)
        
        # Remove sequence dimension for single-step inference
        if single_step:
            output = output.squeeze(1)
        
        return output, new_hidden
    
    def get_initial_hidden(self, batch_size, device=None):
        """Get initial hidden state."""
        if device is None:
            device = next(self.parameters()).device
        
        num_directions = 2 if self.bidirectional else 1
        h_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=device)
        return h_0


class TorchBiLSTM(nn.Module if nn else object):
    """
    Bidirectional LSTM for BC with enhanced temporal modeling.
    
    Architecture: Input -> BiLSTM -> Hidden -> Output
    - Uses bidirectional processing for better temporal understanding
    - Particularly useful for offline training with full episode sequences
    - Supports variable sequence lengths with proper padding/masking
    """
    
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int = 128, 
                 num_layers: int = 1, dropout: float = 0.1):
        """
        Initialize bidirectional LSTM model.
        
        Args:
            in_dim: Input dimension (observation space)
            out_dim: Output dimension (action space)
            hidden_size: LSTM hidden dimension (per direction)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        if torch is None:
            raise ImportError("PyTorch not available. Install torch or use --backend jax.")
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Output layer (2 * hidden_size because bidirectional)
        self.output_layer = nn.Linear(2 * hidden_size, out_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch, seq_len, in_dim)
            hidden: Optional hidden state tuple (h_0, c_0)
            
        Returns:
            Output tensor with actions in [-1, 1] range (batch, seq_len, out_dim)
        """
        # Bidirectional LSTM forward pass
        lstm_out, new_hidden = self.lstm(x, hidden)
        
        # Apply output layer
        output = self.output_layer(lstm_out)
        output = torch.tanh(output)
        
        return output, new_hidden
    
    def get_initial_hidden(self, batch_size, device=None):
        """Get initial hidden state."""
        if device is None:
            device = next(self.parameters()).device
        
        # Bidirectional: 2 directions
        h_0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)


def create_temporal_model(model_type: str, in_dim: int, out_dim: int, **kwargs):
    """
    Factory function to create temporal models.
    
    Args:
        model_type: Type of model ("lstm", "gru", "bilstm")
        in_dim: Input dimension
        out_dim: Output dimension
        **kwargs: Additional model parameters
        
    Returns:
        Instantiated model
    """
    if model_type.lower() == "lstm":
        return TorchLSTM(in_dim, out_dim, **kwargs)
    elif model_type.lower() == "gru":
        return TorchGRU(in_dim, out_dim, **kwargs)
    elif model_type.lower() == "bilstm":
        return TorchBiLSTM(in_dim, out_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'lstm', 'gru', or 'bilstm'")


def compute_temporal_loss(predictions, targets, mask=None, loss_type="l1"):
    """
    Compute loss for temporal models with optional masking.
    
    Args:
        predictions: Model predictions (batch, seq_len, out_dim)
        targets: Ground truth targets (batch, seq_len, out_dim)
        mask: Optional mask for valid timesteps (batch, seq_len)
        loss_type: Type of loss ("l1", "mse", "smooth_l1")
        
    Returns:
        Loss value
    """
    if loss_type == "l1":
        loss = F.l1_loss(predictions, targets, reduction='none')
    elif loss_type == "mse":
        loss = F.mse_loss(predictions, targets, reduction='none')
    elif loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(predictions, targets, reduction='none')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Apply mask if provided
    if mask is not None:
        loss = loss * mask.unsqueeze(-1)
        # Average over valid timesteps
        loss = loss.sum() / mask.sum()
    else:
        loss = loss.mean()
    
    return loss


def create_padding_mask(sequence_lengths, max_length):
    """
    Create padding mask for variable-length sequences.
    
    Args:
        sequence_lengths: Length of each sequence in batch (batch,)
        max_length: Maximum sequence length
        
    Returns:
        Mask tensor (batch, max_length) where True indicates valid timesteps
    """
    batch_size = len(sequence_lengths)
    mask = torch.arange(max_length, device=sequence_lengths.device).expand(batch_size, max_length) < sequence_lengths.unsqueeze(1)
    return mask


