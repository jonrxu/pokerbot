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
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        residual = x
        # Handle batch size 1 for BatchNorm
        if x.size(0) == 1:
            self.bn1.eval()
            self.bn2.eval()
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        if x.size(0) == 1:
            self.bn1.train()
            self.bn2.train()
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
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
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
        # BatchNorm doesn't work with batch size 1, so use eval mode or skip
        if x.size(0) == 1:
            self.input_bn.eval()
        out = self.input_bn(out)
        if x.size(0) == 1:
            self.input_bn.train()
        
        for res_block in self.res_blocks:
            if x.size(0) == 1:
                res_block.bn1.eval()
                res_block.bn2.eval()
            out = res_block(out)
            if x.size(0) == 1:
                res_block.bn1.train()
                res_block.bn2.train()
        
        return out

