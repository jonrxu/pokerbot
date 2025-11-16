"""Evaluator for head-to-head agent matches."""

import random
import numpy as np
import torch
from typing import List, Dict, Optional
from poker_game.game import PokerGame, GameState, Action
from poker_game.information_set import InformationSet, get_information_set
from models.value_policy_net import ValuePolicyNet
from poker_game.state_encoder import StateEncoder
from evaluation.metrics import Metrics


class Evaluator:
    """Evaluates agents in head-to-head matches."""
    
    def __init__(self, game: PokerGame, state_encoder: StateEncoder):
        self.game = game
        self.state_encoder = state_encoder
        self.metrics = Metrics()
    
    def evaluate_agents(self,
                        agent1_net: ValuePolicyNet,
                        agent2_net: ValuePolicyNet,
                        num_games: int = 100,
                        device: str = 'cpu') -> Dict:
        """Evaluate two agents head-to-head."""
        agent1_wins = 0
        agent2_wins = 0
        agent1_payoff = 0.0
        agent2_payoff = 0.0
        
        agent1_net.to(device)
        agent2_net.to(device)
        agent1_net.eval()
        agent2_net.eval()
        
        for game_num in range(num_games):
            state = self.game.reset()
            
            # Randomly assign agents to positions
            if random.random() < 0.5:
                agents = [agent1_net, agent2_net]
                agent_ids = ['agent1', 'agent2']
            else:
                agents = [agent2_net, agent1_net]
                agent_ids = ['agent2', 'agent1']
            
            # Play game
            while not state.is_terminal:
                player = state.current_player
                agent_net = agents[player]
                
                # Get legal actions
                legal_actions = self.game.get_legal_actions(state)
                if len(legal_actions) == 0:
                    break
                
                # Encode state
                state_encoding = self.state_encoder.encode(state, player)
                state_tensor = torch.tensor(state_encoding, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Get action probabilities
                with torch.no_grad():
                    _, policy_logits = agent_net(state_tensor)
                    action_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                
                # Sample action (or take best action)
                action_idx = self._sample_action(action_probs, legal_actions)
                action, amount = legal_actions[action_idx]
                
                # Record action
                self.metrics.record_action(agent_ids[player], action.name)
                
                # Apply action
                state = self.game.apply_action(state, action, amount)
            
            # Get payoffs
            payoffs = self.game.get_payoff(state)
            
            # Record results
            agent1_payoff += payoffs[0] if agent_ids[0] == 'agent1' else payoffs[1]
            agent2_payoff += payoffs[1] if agent_ids[1] == 'agent2' else payoffs[0]
            
            if payoffs[0] > payoffs[1]:
                if agent_ids[0] == 'agent1':
                    agent1_wins += 1
                else:
                    agent2_wins += 1
            elif payoffs[1] > payoffs[0]:
                if agent_ids[1] == 'agent1':
                    agent1_wins += 1
                else:
                    agent2_wins += 1
        
        # Record metrics
        self.metrics.record_game_result('agent1', agent1_payoff)
        self.metrics.record_game_result('agent2', agent2_payoff)
        
        return {
            'agent1_wins': agent1_wins,
            'agent2_wins': agent2_wins,
            'agent1_payoff': agent1_payoff / num_games,
            'agent2_payoff': agent2_payoff / num_games,
            'agent1_win_rate': agent1_wins / num_games,
            'agent2_win_rate': agent2_wins / num_games,
            'metrics': self.metrics.get_summary()
        }
    
    def _sample_action(self, action_probs: np.ndarray, legal_actions: List) -> int:
        """Sample an action from probabilities, masking illegal actions."""
        # Create mask for legal actions
        mask = np.zeros(len(action_probs))
        for i in range(len(legal_actions)):
            if i < len(mask):
                mask[i] = 1.0
        
        # Mask and renormalize
        masked_probs = action_probs * mask
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            # Uniform if all masked
            masked_probs = mask / mask.sum()
        
        # Sample
        return np.random.choice(len(masked_probs), p=masked_probs)
    
    def evaluate_agent_pool(self,
                            agent_pool: List[ValuePolicyNet],
                            num_games_per_matchup: int = 50) -> Dict:
        """Evaluate a pool of agents in round-robin tournament."""
        results = {}
        
        for i, agent1 in enumerate(agent_pool):
            for j, agent2 in enumerate(agent_pool):
                if i >= j:
                    continue
                
                matchup_result = self.evaluate_agents(
                    agent1, agent2, num_games=num_games_per_matchup
                )
                results[f"agent_{i}_vs_agent_{j}"] = matchup_result
        
        return results

