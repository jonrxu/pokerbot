"""Deep CFR algorithm implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from poker_game.game import PokerGame, GameState, Action
from poker_game.information_set import InformationSet, get_information_set
from models.advantage_net import AdvantageNet
from models.policy_net import PolicyNet
from poker_game.state_encoder import StateEncoder


class DeepCFR:
    """Deep Counterfactual Regret Minimization algorithm.
    
    Implements Canonical Deep CFR with 2 networks:
    1. AdvantageNet: Estimates cumulative regrets (advantages) R(I, a).
    2. PolicyNet: Approximates average strategy (final policy) sigma_avg(I, a).
    """
    
    def __init__(self, 
                 advantage_net: AdvantageNet,
                 policy_net: PolicyNet,
                 state_encoder: StateEncoder,
                 game: PokerGame,
                 learning_rate: float = 1e-4,
                 device: str = 'cpu'):
        self.advantage_net = advantage_net
        self.policy_net = policy_net
        self.state_encoder = state_encoder
        self.game = game
        self.device = device
        
        # Move networks to device
        self.advantage_net.to(device)
        self.policy_net.to(device)
        
        # Optimizers
        self.advantage_optimizer = optim.Adam(self.advantage_net.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Exploration parameter (Linear CFR exploration floor)
        self.exploration = 0.1
        
        # Helper for scaling values to neural net range (approx -1 to 1)
        # We use starting stack (20,000 chips) as the scale, so CFVs are O(1).
        self.CFV_SCALE = 20000.0

    def regret_matching(self, regrets: Dict[int, float]) -> Dict[int, float]:
        """Convert regrets to action probabilities using regret matching."""
        positive_regrets = {a: max(0, r) for a, r in regrets.items()}
        total = sum(positive_regrets.values())
        
        if total > 0:
            return {a: r / total for a, r in positive_regrets.items()}
        else:
            # Uniform distribution if no positive regrets
            num_actions = len(regrets)
            return {a: 1.0 / num_actions for a in regrets.keys()}

    def get_strategy_from_advantage_net(self, state: GameState, legal_actions: List[Tuple[Action, int]]) -> Dict[int, float]:
        """Get current exploration strategy using Advantage Network."""
        if not legal_actions:
            return {}

        player = state.current_player
        
        # Encode state
        state_encoding = self.state_encoder.encode(state, player)
        state_tensor = torch.tensor(state_encoding, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get predicted advantages (regrets)
        with torch.no_grad():
            advantages_tensor = self.advantage_net.get_advantages(state_tensor)
            advantages_np = advantages_tensor.cpu().numpy()[0]  # Shape: (max_actions,)
            
        # Map back to legal actions
        # legal_actions is a list of (Action, amount), indices 0..len-1
        valid_advantages = {}
        for idx in range(len(legal_actions)):
            if idx < len(advantages_np):
                valid_advantages[idx] = float(advantages_np[idx])
            else:
                valid_advantages[idx] = 0.0
                
        # Apply Regret Matching on predicted advantages
        strategy = self.regret_matching(valid_advantages)
        
        # Add exploration floor (epsilon-greedy-ish mix)
        # This ensures we don't get stuck in local optima early on
        num_legal = len(legal_actions)
        mixed_strategy = {}
        for idx in range(num_legal):
            mixed_strategy[idx] = (
                (1 - self.exploration) * strategy.get(idx, 0.0) +
                (self.exploration * (1.0 / num_legal))
            )
            
        # Normalize
        total_prob = sum(mixed_strategy.values())
        if total_prob > 1e-9:
            return {k: v / total_prob for k, v in mixed_strategy.items()}
        return {k: 1.0/num_legal for k in mixed_strategy}

    def traverse_external_sampling(self, 
                                 state: GameState, 
                                 player: int,
                                 buffers: Dict[str, List]) -> float:
        """
        Traverse the game tree using External Sampling MCCFR.
        
        Args:
            state: Current game state
            player: The trainee (traversal player)
            buffers: Dictionary to append training data to
            
        Returns:
            Expected value of this node (counterfactual value)
        """
        # Terminal Node
        if state.is_terminal:
            payoffs = self.game.get_payoff(state)
            return payoffs[player]

        legal_actions = self.game.get_legal_actions(state)
        if not legal_actions:
            return 0.0
            
        current_player = state.current_player
        
        # --- Case 1: Opponent or Chance Node ---
        if current_player != player:
            # External Sampling: Sample ONE action
            
            # If it's the opponent's turn, they play according to their Average Strategy (PolicyNet)
            # If chance, handled by game logic (implied sampling in future steps if distinct)
            # Here opponent model is our PolicyNet.
            
            state_encoding = self.state_encoder.encode(state, current_player)
            state_tensor = torch.tensor(state_encoding, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # PolicyNet outputs probabilities
                probs = self.policy_net.get_policy(state_tensor).cpu().numpy()[0]
            
            # Filter to legal actions and renormalize
            legal_probs = []
            for idx in range(len(legal_actions)):
                if idx < len(probs):
                    legal_probs.append(probs[idx])
                else:
                    legal_probs.append(0.0)
            
            total = sum(legal_probs)
            if total > 1e-6:
                # Renormalize
                legal_probs = np.array(legal_probs, dtype=np.float64)
                legal_probs /= legal_probs.sum()
            else:
                # Fallback to uniform
                legal_probs = np.ones(len(legal_actions), dtype=np.float64) / len(legal_actions)
                
            # Sample action
            action_idx = np.random.choice(len(legal_actions), p=legal_probs)
            action, amount = legal_actions[action_idx]
            
            next_state = self.game.apply_action(state, action, amount)
            
            # Recursive call
            return self.traverse_external_sampling(next_state, player, buffers)

        # --- Case 2: Traversal Player Node (Our Turn) ---
        else:
            # We traverse ALL actions to compute counterfactual regrets
            
            # 1. Get current strategy from Advantage Net (Exploration Strategy)
            # This is used to weight the node value, but NOT to choose paths (we take all)
            strategy = self.get_strategy_from_advantage_net(state, legal_actions)
            
            # 2. Calculate value for each action by recursion
            action_values = np.zeros(len(legal_actions))
            
            for idx, (action, amount) in enumerate(legal_actions):
                next_state = self.game.apply_action(state, action, amount)
                
                # Recursive call: get value of taking this action
                action_values[idx] = self.traverse_external_sampling(next_state, player, buffers)
            
            # 3. Calculate node value (weighted average under current strategy)
            node_value = sum(strategy[idx] * action_values[idx] for idx in range(len(legal_actions)))
            
            # 4. Calculate Counterfactual Regrets / Advantages
            # Advantage(a) = Value(a) - Value(Node)
            advantages = action_values - node_value
            
            # 5. Store Training Data
            state_encoding = self.state_encoder.encode(state, player)
            
            # Store for Advantage Net: (State, Advantage Vector)
            # We assume training happens on SCALED values to keep gradients stable.
            # Advantages can be large (chips), so scale them.
            scaled_advantages = advantages / self.CFV_SCALE
            
            full_advantage_vector = np.zeros(self.advantage_net.max_actions)
            for idx, adv in enumerate(scaled_advantages):
                if idx < self.advantage_net.max_actions:
                    full_advantage_vector[idx] = adv
            
            buffers['advantage'].append((state_encoding, full_advantage_vector))
            
            # Store for Policy Net: (State, Strategy Probabilities)
            # This trains the policy net to mimic the average strategy.
            # In standard Deep CFR, we weight this by iteration t (linear weighting). 
            # We'll handle weighting in the loss or sampling later.
            full_strategy_vector = np.zeros(self.policy_net.max_actions)
            for idx, prob in strategy.items():
                if idx < self.policy_net.max_actions:
                    full_strategy_vector[idx] = prob
                    
            buffers['policy'].append((state_encoding, full_strategy_vector))
            
            return node_value
