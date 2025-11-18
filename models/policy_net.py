"""Policy network for Deep CFR.
Approximates the average strategy over all iterations.
"""

import torch
import torch.nn as nn
from .poker_net import PokerNet


class PolicyNet(PokerNet):
    """Neural network for predicting action probabilities (Average Strategy)."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 6,
                 max_actions: int = 20):
        super().__init__(input_dim, hidden_dim, num_layers)
        self.max_actions = max_actions
        
        # Policy head: outputs action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, max_actions)
        )
        
        self._init_head_weights()
    
    def _init_head_weights(self):
        """Initialize output head weights."""
        for m in self.policy_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass returning policy logits.
        
        Args:
            x: Input state encoding
            
        Returns:
            logits: Tensor of shape (batch_size, max_actions)
        """
        features = super().forward(x)
        logits = self.policy_head(features)
        return logits
    
    def get_policy(self, x, legal_actions_mask=None):
        """Get softmax policy for legal actions."""
        features = super().forward(x)
        logits = self.policy_head(features)
        
        if legal_actions_mask is not None:
            # Mask illegal actions with large negative value
            logits = logits.masked_fill(~legal_actions_mask, float('-inf'))
        
        return torch.softmax(logits, dim=-1)

