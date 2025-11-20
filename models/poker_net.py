"""Base neural network architecture for poker."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for ResNet-style architecture."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual
        out += residual
        out = F.relu(out)
        return out


class PokerNet(nn.Module):
    """Base poker neural network with ResNet-style architecture."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.LayerNorm(hidden_dim)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output heads will be added by subclasses
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through base network."""
        # Handle both batched and unbatched inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        out = self.input_proj(x)
        out = self.input_bn(out)
        
        for res_block in self.res_blocks:
            out = res_block(out)
        
        return out

