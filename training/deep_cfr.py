"""Deep CFR algorithm implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from poker_game.game import PokerGame, GameState, Action
from poker_game.information_set import InformationSet, get_information_set
from models.value_policy_net import ValuePolicyNet
from poker_game.state_encoder import StateEncoder


class DeepCFR:
    """Deep Counterfactual Regret Minimization algorithm."""
    
    def __init__(self, 
                 value_net: ValuePolicyNet,
                 policy_net: ValuePolicyNet,
                 state_encoder: StateEncoder,
                 game: PokerGame,
                 learning_rate: float = 1e-4,
                 device: str = 'cpu'):
        self.value_net = value_net
        self.policy_net = policy_net
        self.state_encoder = state_encoder
        self.game = game
        self.device = device
        
        # Move networks to device
        self.value_net.to(device)
        self.policy_net.to(device)
        
        # Optimizers
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Regret and strategy accumulators
        self.regret_memory: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.strategy_memory: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.counterfactual_values: Dict[str, float] = defaultdict(float)
        
        # Training buffers
        self.value_buffer: List[Tuple[np.ndarray, float]] = []
        self.policy_buffer: List[Tuple[np.ndarray, np.ndarray]] = []
        
        # Exploration parameter
        self.exploration = 0.6
    
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
    
    def get_strategy(self, info_set: InformationSet, legal_actions: List[Tuple[Action, int]]) -> Dict[int, float]:
        """Get current strategy for an information set."""
        key = info_set.key
        
        # Get regrets for this information set
        regrets = self.regret_memory[key]
        
        # If no regrets yet, return uniform
        if not regrets or len(regrets) != len(legal_actions):
            return {i: 1.0 / len(legal_actions) for i in range(len(legal_actions))}
        
        # Regret matching
        return self.regret_matching(regrets)
    
    def sample_action(self, strategy: Dict[int, float]) -> int:
        """Sample an action from strategy distribution."""
        actions = list(strategy.keys())
        probs = [strategy[a] for a in actions]
        return np.random.choice(actions, p=probs)
    
    def traverse(self, state: GameState, player: int, reach_prob: float, 
                 sample_strategy: bool = True) -> Tuple[float, List]:
        """Traverse game tree and compute counterfactual values."""
        if state.is_terminal:
            payoffs = self.game.get_payoff(state)
            return payoffs[player], []
        
        info_set = get_information_set(state, player)
        legal_actions = self.game.get_legal_actions(state)
        
        if len(legal_actions) == 0:
            return 0.0, []
        
        # Get strategy
        strategy = self.get_strategy(info_set, legal_actions)
        
        # Add exploration
        if sample_strategy:
            exploration_strategy = {}
            for i in range(len(legal_actions)):
                exploration_strategy[i] = (
                    self.exploration * (1.0 / len(legal_actions)) +
                    (1 - self.exploration) * strategy.get(i, 0.0)
                )
            strategy = exploration_strategy
            # Renormalize
            total = sum(strategy.values())
            strategy = {k: v / total for k, v in strategy.items()}
        
        # Compute counterfactual values
        cf_values = {}
        node_value = 0.0
        trajectory_data = []
        
        for action_idx, (action, amount) in enumerate(legal_actions):
            action_prob = strategy.get(action_idx, 0.0)
            if action_prob == 0:
                continue
            
            next_state = self.game.apply_action(state, action, amount)
            
            if next_state.current_player == player:
                # Same player acts again
                cf_value, sub_trajectory = self.traverse(
                    next_state, player, reach_prob * action_prob, sample_strategy
                )
            else:
                # Opponent acts
                opp_reach = reach_prob * action_prob
                cf_value, sub_trajectory = self.traverse(
                    next_state, player, opp_reach, sample_strategy
                )
            
            cf_values[action_idx] = cf_value
            node_value += action_prob * cf_value
            
            # Store trajectory data
            trajectory_data.append({
                'info_set': info_set,
                'action': action_idx,
                'action_prob': action_prob,
                'cf_value': cf_value,
                'reach_prob': reach_prob
            })
            trajectory_data.extend(sub_trajectory)
        
        # Update regrets
        key = info_set.key
        for action_idx, cf_value in cf_values.items():
            regret = cf_value - node_value
            self.regret_memory[key][action_idx] += regret * reach_prob
        
        # Update counterfactual value
        self.counterfactual_values[key] = node_value
        
        return node_value, trajectory_data
    
    def update_networks(self, batch_size: int = 32):
        """Update value and policy networks from buffers."""
        if len(self.value_buffer) == 0 and len(self.policy_buffer) == 0:
            return
        
        # Update value network
        if len(self.value_buffer) >= batch_size:
            indices = np.random.choice(len(self.value_buffer), batch_size, replace=False)
            batch_states = [self.value_buffer[i][0] for i in indices]
            batch_values = [self.value_buffer[i][1] for i in indices]
            
            states_tensor = torch.tensor(np.stack(batch_states), dtype=torch.float32).to(self.device)
            values_tensor = torch.tensor(batch_values, dtype=torch.float32).to(self.device).unsqueeze(1)
            
            self.value_optimizer.zero_grad()
            predicted_values, _ = self.value_net(states_tensor)
            value_loss = nn.MSELoss()(predicted_values, values_tensor)
            value_loss.backward()
            self.value_optimizer.step()
        
        # Update policy network
        if len(self.policy_buffer) >= batch_size:
            indices = np.random.choice(len(self.policy_buffer), batch_size, replace=False)
            batch_states = [self.policy_buffer[i][0] for i in indices]
            batch_probs = [self.policy_buffer[i][1] for i in indices]
            
            states_tensor = torch.tensor(np.stack(batch_states), dtype=torch.float32).to(self.device)
            probs_tensor = torch.tensor(np.stack(batch_probs), dtype=torch.float32).to(self.device)
            
            self.policy_optimizer.zero_grad()
            _, policy_logits = self.policy_net(states_tensor)
            policy_loss = nn.CrossEntropyLoss()(policy_logits, probs_tensor.argmax(dim=1))
            # Also use KL divergence for better policy matching
            policy_probs = torch.softmax(policy_logits, dim=1)
            kl_loss = nn.KLDivLoss(reduction='batchmean')(
                torch.log(policy_probs + 1e-8), probs_tensor
            )
            total_loss = policy_loss + kl_loss
            total_loss.backward()
            self.policy_optimizer.step()
    
    def compute_average_strategy(self, info_set: InformationSet, 
                                legal_actions: List[Tuple[Action, int]]) -> Dict[int, float]:
        """Compute average strategy from accumulated regrets."""
        key = info_set.key
        
        # Get accumulated strategy
        strategy_sum = self.strategy_memory.get(key, {})
        total = sum(strategy_sum.values())
        
        if total > 0:
            return {a: s / total for a, s in strategy_sum.items()}
        else:
            # Uniform if no strategy accumulated
            return {i: 1.0 / len(legal_actions) for i in range(len(legal_actions))}
    
    def get_average_policy(self, state: GameState, player: int) -> Dict[int, float]:
        """Get average policy for current state."""
        info_set = get_information_set(state, player)
        legal_actions = self.game.get_legal_actions(state)
        return self.compute_average_strategy(info_set, legal_actions)

