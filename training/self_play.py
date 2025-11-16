"""Self-play trajectory generation."""

import random
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

from poker_game.game import PokerGame, GameState, Action
from poker_game.information_set import InformationSet, get_information_set
from training.deep_cfr import DeepCFR


class SelfPlayGenerator:
    """Generates self-play trajectories for training."""
    
    def __init__(self, game: PokerGame, deep_cfr: DeepCFR, num_trajectories: int = 1000):
        self.game = game
        self.deep_cfr = deep_cfr
        self.num_trajectories = num_trajectories
    
    def generate_trajectories(self) -> List[Dict]:
        """Generate trajectories via self-play."""
        trajectories = []
        
        for _ in range(self.num_trajectories):
            state = self.game.reset()
            trajectory = self._play_hand(state)
            trajectories.append(trajectory)
        
        return trajectories
    
    def _play_hand(self, initial_state: GameState) -> Dict:
        """Play a single hand and collect trajectory data."""
        trajectory = {
            'states': [],
            'actions': [],
            'info_sets': [],
            'payoffs': None,
            'player': None
        }
        
        state = initial_state
        current_player = state.current_player
        
        while not state.is_terminal:
            player = state.current_player
            info_set = get_information_set(state, player)
            legal_actions = self.game.get_legal_actions(state)
            
            if len(legal_actions) == 0:
                break
            
            # Get strategy from Deep CFR
            strategy = self.deep_cfr.get_strategy(info_set, legal_actions)
            
            # Sample action
            action_idx = self.deep_cfr.sample_action(strategy)
            action, amount = legal_actions[action_idx]
            
            # Store trajectory data
            trajectory['states'].append(state)
            trajectory['actions'].append((action_idx, action, amount))
            trajectory['info_sets'].append(info_set)
            
            # Apply action
            state = self.game.apply_action(state, action, amount)
        
        # Get final payoffs
        payoffs = self.game.get_payoff(state)
        trajectory['payoffs'] = payoffs
        trajectory['player'] = current_player
        
        return trajectory
    
    def generate_trajectories_parallel(self, num_workers: int = 4) -> List[Dict]:
        """Generate trajectories in parallel (for distributed training)."""
        # This will be used by Modal workers
        trajectories_per_worker = self.num_trajectories // num_workers
        all_trajectories = []
        
        for _ in range(num_workers):
            worker_trajectories = []
            for _ in range(trajectories_per_worker):
                state = self.game.reset()
                trajectory = self._play_hand(state)
                worker_trajectories.append(trajectory)
            all_trajectories.extend(worker_trajectories)
        
        return all_trajectories

