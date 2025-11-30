"""GTO opponent agent that uses the Deep CFR model."""

import torch
import os
from typing import Tuple, List
from poker_game.game import PokerGame, GameState, Action
from poker_game.state_encoder import StateEncoder
from poker_game.information_set import get_information_set
from models.value_policy_net import ValuePolicyNet
from training.deep_cfr import DeepCFR


class GTOAgent:
    """GTO opponent using a trained Deep CFR checkpoint.

    This agent loads a Deep CFR checkpoint and uses the average strategy
    (from strategy_memory) to make near-optimal GTO decisions.
    """

    def __init__(
        self,
        game: PokerGame,
        checkpoint_path: str = None,
        device: str = 'cpu',
        exploration: float = 0.0,  # No exploration for pure GTO play
    ):
        """Initialize GTO agent.

        Args:
            game: PokerGame instance
            checkpoint_path: Path to Deep CFR checkpoint
            device: 'cpu' or 'cuda'
            exploration: Exploration parameter (0.0 for pure GTO)
        """
        self.game = game
        self.device = device
        self.exploration = exploration

        # Initialize encoder and networks
        self.state_encoder = StateEncoder()
        input_dim = self.state_encoder.feature_dim

        value_net = ValuePolicyNet(input_dim=input_dim).to(device)
        policy_net = ValuePolicyNet(input_dim=input_dim).to(device)

        # Initialize Deep CFR
        self.deep_cfr = DeepCFR(
            value_net=value_net,
            policy_net=policy_net,
            state_encoder=self.state_encoder,
            game=game,
            device=device
        )

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        else:
            import logging
            logging.warning(
                "No checkpoint provided or checkpoint not found. "
                "GTOAgent will use untrained networks (random play)."
            )

    def _load_checkpoint(self, checkpoint_path: str):
        """Load Deep CFR checkpoint."""
        from collections import defaultdict
        import logging

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # Load networks
            if 'value_net_state' in checkpoint:
                self.deep_cfr.value_net.load_state_dict(checkpoint['value_net_state'])
            if 'policy_net_state' in checkpoint:
                self.deep_cfr.policy_net.load_state_dict(checkpoint['policy_net_state'])

            # Load strategy memory (this is what we use for GTO play)
            if 'strategy_memory' in checkpoint:
                if isinstance(checkpoint['strategy_memory'], dict):
                    self.deep_cfr.strategy_memory = defaultdict(
                        lambda: defaultdict(float),
                        {
                            k: defaultdict(float, v) if isinstance(v, dict) else v
                            for k, v in checkpoint['strategy_memory'].items()
                        }
                    )
                else:
                    self.deep_cfr.strategy_memory = checkpoint['strategy_memory']

            # Load regret memory (used as fallback)
            if 'regret_memory' in checkpoint:
                if isinstance(checkpoint['regret_memory'], dict):
                    self.deep_cfr.regret_memory = defaultdict(
                        lambda: defaultdict(float),
                        {
                            k: defaultdict(float, v) if isinstance(v, dict) else v
                            for k, v in checkpoint['regret_memory'].items()
                        }
                    )
                else:
                    self.deep_cfr.regret_memory = checkpoint['regret_memory']

            logging.info(f"Successfully loaded GTO checkpoint from {checkpoint_path}")

        except Exception as e:
            logging.error(f"Failed to load GTO checkpoint: {e}")
            raise

    def get_action(
        self,
        state: GameState,
        legal_actions: List[Tuple[Action, int]]
    ) -> Tuple[Action, int]:
        """Get GTO action for the current state.

        Uses the average strategy from strategy_memory, which converges to
        Nash equilibrium in CFR.
        """
        if not legal_actions:
            return (Action.FOLD, 0)

        player = state.current_player

        # Get information set
        info_set = get_information_set(state, player)

        # Get strategy from Deep CFR
        # First try average strategy (strategy_memory), then current strategy (regret-based)
        strategy = self.deep_cfr.get_average_strategy(info_set, legal_actions)

        if strategy is None or len(strategy) == 0:
            # Fallback to current strategy if no average strategy available
            strategy = self.deep_cfr.get_strategy(info_set, legal_actions)

        if strategy is None or len(strategy) == 0:
            # Ultimate fallback: uniform random
            import random
            return random.choice(legal_actions)

        # Sample action according to strategy
        import numpy as np
        actions = list(strategy.keys())
        probs = np.array([strategy[a] for a in actions])

        # Normalize probabilities
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(probs)) / len(probs)

        # Sample action
        chosen_action = np.random.choice(len(actions), p=probs)
        return actions[chosen_action]
