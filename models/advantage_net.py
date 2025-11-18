"""Advantage network for Deep CFR.
In the Deep CFR paper, this is referred to as the 'Value Network' V(I), 
but it outputs a vector of advantages/regrets for each action, not a single scalar.
We call it AdvantageNet to avoid confusion with scalar value functions.
"""

import torch
import torch.nn as nn
from .poker_net import PokerNet


class AdvantageNet(PokerNet):
    """Neural network for predicting advantage (regret) vectors."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 6,
                 max_actions: int = 20):
        super().__init__(input_dim, hidden_dim, num_layers)
        self.max_actions = max_actions
        
        # Advantage head: outputs vector of advantages (one for each action)
        # Corresponds to predicted cumulative regrets R(I, a)
        self.advantage_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, max_actions)
        )
        
        self._init_head_weights()
    
    def _init_head_weights(self):
        """Initialize output head weights."""
        for m in self.advantage_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass returning advantage vector.
        
        Args:
            x: Input state encoding
            
        Returns:
            advantages: Tensor of shape (batch_size, max_actions)
        """
        features = super().forward(x)
        advantages = self.advantage_head(features)
        return advantages
    
    def get_advantages(self, x, legal_actions_mask=None):
        """Get advantages for legal actions."""
        features = super().forward(x)
        advantages = self.advantage_head(features)
        
        if legal_actions_mask is not None:
            # Mask illegal actions
            # We use a float mask (1.0 for legal, 0.0 for illegal)
            advantages = advantages * legal_actions_mask.float()
        
        return advantages

