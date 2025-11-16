"""Value and policy network for Deep CFR."""

import torch
import torch.nn as nn
from .poker_net import PokerNet


class ValuePolicyNet(PokerNet):
    """Neural network with separate value and policy heads."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 6,
                 max_actions: int = 20):
        super().__init__(input_dim, hidden_dim, num_layers)
        self.max_actions = max_actions
        
        # Value head: outputs expected utility
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Policy head: outputs action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, max_actions)
        )
        
        self._init_head_weights()
    
    def _init_head_weights(self):
        """Initialize output head weights."""
        for m in [self.value_head, self.policy_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        """Forward pass returning value and policy."""
        features = super().forward(x)
        
        value = self.value_head(features)
        policy_logits = self.policy_head(features)
        
        return value, policy_logits
    
    def get_value(self, x):
        """Get value estimate only."""
        features = super().forward(x)
        return self.value_head(features)
    
    def get_policy(self, x, legal_actions_mask=None):
        """Get policy probabilities for legal actions."""
        features = super().forward(x)
        policy_logits = self.policy_head(features)
        
        if legal_actions_mask is not None:
            # Mask illegal actions
            policy_logits = policy_logits.masked_fill(~legal_actions_mask, float('-inf'))
        
        return torch.softmax(policy_logits, dim=-1)
    
    def get_action_probs(self, x, legal_actions_mask=None):
        """Get action probabilities compatible with regret matching."""
        return self.get_policy(x, legal_actions_mask)

