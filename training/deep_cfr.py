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
                 advantage_learning_rate: float = None,
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
        advantage_lr = advantage_learning_rate if advantage_learning_rate is not None else learning_rate
        self.advantage_optimizer = optim.Adam(self.advantage_net.parameters(), lr=advantage_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Exploration parameter (Linear CFR exploration floor)
        self.exploration = 0.1
        
    def set_exploration(self, exploration: float):
        """Set exploration rate."""
        self.exploration = exploration
        
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

    def get_strategy_from_advantage_net(self, state: GameState, legal_actions: List[Tuple[Action, int]], exploration_rate: float = None) -> Dict[int, float]:
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
                # Inverse Symlog Scaling: y = sign(x) * (exp(|x|) - 1)
                # Clamp to prevent overflow (exp(710) overflows float64)
                val = float(advantages_np[idx])
                clamped_abs = np.clip(np.abs(val), 0, 100)  # e^100 ≈ 2e43, way bigger than any poker stack
                real_val = np.sign(val) * (np.exp(clamped_abs) - 1)
                valid_advantages[idx] = real_val
            else:
                valid_advantages[idx] = 0.0
                
        # Apply Regret Matching on predicted advantages
        strategy = self.regret_matching(valid_advantages)
        
        # Add exploration floor (epsilon-greedy-ish mix)
        # This ensures we don't get stuck in local optima early on
        epsilon = exploration_rate if exploration_rate is not None else self.exploration
        num_legal = len(legal_actions)
        mixed_strategy = {}
        for idx in range(num_legal):
            mixed_strategy[idx] = (
                (1 - epsilon) * strategy.get(idx, 0.0) +
                (epsilon * (1.0 / num_legal))
            )
            
        # Normalize
        total_prob = sum(mixed_strategy.values())
        if total_prob > 1e-9:
            return {k: v / total_prob for k, v in mixed_strategy.items()}
        return {k: 1.0/num_legal for k in mixed_strategy}

    def traverse_outcome_sampling(self, 
                                state: GameState, 
                                player: int,
                                buffers: Dict[str, List],
                                iteration_weight: float = 1.0,
                                depth: int = 0,
                                max_depth: int = 40,
                                sample_reach: float = 1.0,
                                exploration_rate: float = None) -> float:
        """
        Traverse the game tree using Outcome Sampling MCCFR.
        
        This samples ONE path through the tree, avoiding exponential explosion.
        It uses importance sampling to estimate regrets.
        
        Args:
            state: Current game state
            player: The trainee (traversal player)
            buffers: Dictionary to append training data to
            iteration_weight: Weight for policy samples (Linear CFR)
            depth: Current recursion depth
            max_depth: Maximum recursion depth
            sample_reach: Probability of reaching this history (pi^{-i})
            
        Returns:
            Utility of the sampled path
        """
        # Safety check: prevent infinite recursion
        if depth >= max_depth:
            payoffs = self.game.get_payoff(state) if state.is_terminal else [0.0, 0.0]
            return payoffs[player] if isinstance(payoffs, (list, tuple)) else 0.0
        
        # Terminal Node
        if state.is_terminal:
            payoffs = self.game.get_payoff(state)
            return payoffs[player]

        legal_actions = self.game.get_legal_actions(state)
        if not legal_actions:
            return 0.0
            
        current_player = state.current_player
        
        # Safety check
        if current_player is None:
            return 0.0
        
        # --- Case 1: Opponent or Chance Node ---
        if current_player != player:
            # Sample ONE action according to opponent's strategy (PolicyNet)
            
            state_encoding = self.state_encoder.encode(state, current_player)
            state_tensor = torch.tensor(state_encoding, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                probs = self.policy_net.get_policy_probs(state_tensor).cpu().numpy()[0]
            
            # Filter to legal actions and renormalize
            legal_probs = []
            for idx in range(len(legal_actions)):
                if idx < len(probs):
                    legal_probs.append(probs[idx])
                else:
                    legal_probs.append(0.0)
            
            total = sum(legal_probs)
            if total > 1e-6:
                legal_probs = np.array(legal_probs, dtype=np.float64)
                legal_probs /= legal_probs.sum()
            else:
                legal_probs = np.ones(len(legal_actions), dtype=np.float64) / len(legal_actions)
            
            # Sample action
            action_idx = np.random.choice(len(legal_actions), p=legal_probs)
            action, amount = legal_actions[action_idx]
            
            next_state = self.game.apply_action(state, action, amount)
            
            # Recursive call
            return self.traverse_outcome_sampling(next_state, player, buffers, iteration_weight, depth + 1, max_depth, sample_reach, exploration_rate)

        # --- Case 2: Traversal Player Node (Our Turn) ---
        else:
            # Outcome Sampling: Sample ONE action, but compute regrets using Importance Sampling
            
            # 1. Get current strategy from Advantage Net
            strategy = self.get_strategy_from_advantage_net(state, legal_actions, exploration_rate)
            
            # Convert strategy dict to list of probs aligned with legal_actions
            strategy_probs = np.zeros(len(legal_actions))
            for idx in range(len(legal_actions)):
                strategy_probs[idx] = strategy.get(idx, 0.0)
            
            # Ensure sum is 1.0
            if strategy_probs.sum() > 0:
                strategy_probs /= strategy_probs.sum()
            else:
                strategy_probs = np.ones(len(legal_actions)) / len(legal_actions)
                
            # 2. Sample ONE action
            action_idx = np.random.choice(len(legal_actions), p=strategy_probs)
            action, amount = legal_actions[action_idx]
            chosen_prob = strategy_probs[action_idx]
            
            # 3. Recurse
            next_state = self.game.apply_action(state, action, amount)
            utility = self.traverse_outcome_sampling(next_state, player, buffers, iteration_weight, depth + 1, max_depth, sample_reach, exploration_rate)
            
            # 4. Compute Regrets (Outcome Sampling Formula)
            # r(I, a) = (u / prob(a)) - u   if a was sampled
            # r(I, a) = -u                  if a was NOT sampled
            # Note: This is the "sampled counterfactual value" estimate.
            # CFV(a) = u / prob(a) if sampled, else 0
            # CFV(I) = u
            # Regret = CFV(a) - CFV(I)
            
            # To reduce variance, we can use the "baseline" form, but standard OS is:
            # sampled_cfv = utility / chosen_prob
            # regret[a] = sampled_cfv - utility  (if a == chosen)
            # regret[a] = 0 - utility            (if a != chosen)
            
            # Wait, the formula is:
            # r(I, a) = w * (u - v(I)) ? No.
            # Standard OS:
            # \tilde{v}(I, a) = u / \sigma(a) if a sampled, else 0.
            # \tilde{v}(I) = u
            # \tilde{r}(I, a) = \tilde{v}(I, a) - \tilde{v}(I)
            
            # However, we must be careful with small probabilities causing huge variance.
            # We add an epsilon to exploration (handled in get_strategy) so chosen_prob > epsilon.
            
            advantages = np.zeros(len(legal_actions))
            
            # Weighted utility for the sampled action
            # We clip the probability to avoid division by zero or massive explosions
            safe_prob = max(chosen_prob, 0.01) 
            weighted_utility = utility / safe_prob
            
            for idx in range(len(legal_actions)):
                if idx == action_idx:
                    advantages[idx] = weighted_utility - utility
                else:
                    advantages[idx] = 0.0 - utility
            
            # 5. Store Training Data
            state_encoding = self.state_encoder.encode(state, player)
            
            full_advantage_vector = np.zeros(self.advantage_net.max_actions)
            for idx, adv in enumerate(advantages):
                if idx < self.advantage_net.max_actions:
                    scaled_adv = np.sign(adv) * np.log1p(np.abs(adv))
                    full_advantage_vector[idx] = scaled_adv
            
            buffers['advantage'].append((state_encoding, full_advantage_vector))
            
            # Store Policy Data
            full_strategy_vector = np.zeros(self.policy_net.max_actions)
            for idx, prob in strategy.items():
                if idx < self.policy_net.max_actions:
                    full_strategy_vector[idx] = prob
                    
            buffers['policy'].append((state_encoding, full_strategy_vector, float(iteration_weight)))
            
            return utility

    def traverse_external_sampling(self, 
                                 state: GameState, 
                                 player: int,
                                 buffers: Dict[str, List],
                                 iteration_weight: float = 1.0,
                                 depth: int = 0,
                                 max_depth: int = 40) -> float:
        """
        Traverse the game tree using External Sampling MCCFR.
        """
        # Safety check: prevent infinite recursion
        if depth >= max_depth:
            payoffs = self.game.get_payoff(state) if state.is_terminal else [0.0, 0.0]
            return payoffs[player] if isinstance(payoffs, (list, tuple)) else 0.0
        
        # Terminal Node
        if state.is_terminal:
            payoffs = self.game.get_payoff(state)
            return payoffs[player]

        legal_actions = self.game.get_legal_actions(state)
        if not legal_actions:
            return 0.0
            
        current_player = state.current_player
        
        if current_player is None:
            if state.is_terminal:
                payoffs = self.game.get_payoff(state)
                return payoffs[player]
            else:
                return 0.0
        
        # --- Case 1: Opponent or Chance Node ---
        if current_player != player:
            state_encoding = self.state_encoder.encode(state, current_player)
            state_tensor = torch.tensor(state_encoding, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                probs = self.policy_net.get_policy_probs(state_tensor).cpu().numpy()[0]
            
            legal_probs = []
            for idx in range(len(legal_actions)):
                if idx < len(probs):
                    legal_probs.append(probs[idx])
                else:
                    legal_probs.append(0.0)
            
            total = sum(legal_probs)
            if total > 1e-6:
                legal_probs = np.array(legal_probs, dtype=np.float64)
                legal_probs /= legal_probs.sum()
            else:
                legal_probs = np.ones(len(legal_actions), dtype=np.float64) / len(legal_actions)
            
            action_idx = np.random.choice(len(legal_actions), p=legal_probs)
            action, amount = legal_actions[action_idx]
            
            next_state = self.game.apply_action(state, action, amount)
            
            return self.traverse_external_sampling(next_state, player, buffers, iteration_weight, depth + 1, max_depth)
        
        # --- Case 2: Traversal Player Node (Our Turn) ---
        else:
            strategy = self.get_strategy_from_advantage_net(state, legal_actions)
            
            action_values = np.zeros(len(legal_actions))
            
            for idx, (action, amount) in enumerate(legal_actions):
                next_state = self.game.apply_action(state, action, amount)
                
                if next_state.current_player == current_player and not next_state.is_terminal:
                    if next_state.is_terminal:
                        payoffs = self.game.get_payoff(next_state)
                        action_values[idx] = payoffs[player]
                    else:
                        action_values[idx] = 0.0
                else:
                    action_values[idx] = self.traverse_external_sampling(
                        next_state, player, buffers, iteration_weight, depth + 1, max_depth
                    )
            
            node_value = sum(strategy[idx] * action_values[idx] for idx in range(len(legal_actions)))
            
            advantages = action_values - node_value
            
            state_encoding = self.state_encoder.encode(state, player)
            
            full_advantage_vector = np.zeros(self.advantage_net.max_actions)
            for idx, adv in enumerate(advantages):
                if idx < self.advantage_net.max_actions:
                    scaled_adv = np.sign(adv) * np.log1p(np.abs(adv))
                    full_advantage_vector[idx] = scaled_adv
            
            buffers['advantage'].append((state_encoding, full_advantage_vector))
            
            full_strategy_vector = np.zeros(self.policy_net.max_actions)
            for idx, prob in strategy.items():
                if idx < self.policy_net.max_actions:
                    full_strategy_vector[idx] = prob
                    
            buffers['policy'].append((state_encoding, full_strategy_vector, float(iteration_weight)))
            
            return node_value
