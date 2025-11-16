"""Metrics for evaluating poker bot performance."""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class Metrics:
    """Metrics calculator for poker bot evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.win_count = defaultdict(int)
        self.total_payoff = defaultdict(float)
        self.games_played = 0
        self.action_frequencies = defaultdict(lambda: defaultdict(int))
        self.exploitability_scores = []
    
    def record_game_result(self, agent_id: str, payoff: float):
        """Record result of a single game."""
        self.win_count[agent_id] += 1 if payoff > 0 else 0
        self.total_payoff[agent_id] += payoff
        self.games_played += 1
    
    def record_action(self, agent_id: str, action: str):
        """Record an action taken by an agent."""
        self.action_frequencies[agent_id][action] += 1
    
    def get_win_rate(self, agent_id: str) -> float:
        """Get win rate for an agent."""
        if self.games_played == 0:
            return 0.0
        return self.win_count[agent_id] / self.games_played
    
    def get_average_payoff(self, agent_id: str) -> float:
        """Get average payoff for an agent."""
        if self.games_played == 0:
            return 0.0
        return self.total_payoff[agent_id] / self.games_played
    
    def get_action_distribution(self, agent_id: str) -> Dict[str, float]:
        """Get action distribution for an agent."""
        total = sum(self.action_frequencies[agent_id].values())
        if total == 0:
            return {}
        return {action: count / total for action, count in self.action_frequencies[agent_id].items()}
    
    def compute_exploitability(self, agent_strategy: Dict, opponent_strategy: Dict) -> float:
        """Compute exploitability (deviation from Nash equilibrium).
        
        Simplified exploitability metric - in practice, this would require
        solving for best response against the agent's strategy.
        """
        # Placeholder - would need full game tree traversal
        # For now, return a simple metric based on strategy consistency
        exploitability = 0.0
        
        # Check for obvious exploits (e.g., always folding, always calling)
        for info_set, strategy in agent_strategy.items():
            if len(strategy) > 0:
                max_prob = max(strategy.values())
                min_prob = min(strategy.values())
                # High variance in strategy suggests exploitability
                exploitability += (max_prob - min_prob) ** 2
        
        return exploitability / max(len(agent_strategy), 1)
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics."""
        summary = {
            'games_played': self.games_played,
            'agents': {}
        }
        
        for agent_id in set(list(self.win_count.keys()) + list(self.total_payoff.keys())):
            summary['agents'][agent_id] = {
                'win_rate': self.get_win_rate(agent_id),
                'average_payoff': self.get_average_payoff(agent_id),
                'action_distribution': self.get_action_distribution(agent_id)
            }
        
        return summary

