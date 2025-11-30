"""Legacy state encoder for backwards compatibility with old checkpoints.

This encoder matches the 178-feature encoding used in checkpoint_iter_19.pt
and earlier checkpoints.
"""

import numpy as np
import torch
from typing import List, Tuple
from .game import GameState, Action


class LegacyStateEncoder:
    """Legacy StateEncoder (178 features) for loading old checkpoints.

    This matches the encoding from checkpoint_iter_19.pt which expected
    178 input features instead of the current 167.

    The difference appears to be in the betting history encoding or
    additional features that were later removed.
    """

    RANKS = 13
    SUITS = 4
    MAX_HISTORY = 25  # Old version used 25 instead of 20

    def __init__(self):
        # This encoder produces 178 features to match old checkpoints
        # Calculation:
        # - Hole cards: 2 * (13 + 4) = 34
        # - Community cards: 5 * (13 + 4) = 85
        # - Betting history: 25 * 2 = 50
        # - Pot/stack ratios: 4
        # - Position: 1
        # - Street: 1
        # - Current bets: 2
        # - Extra feature: 1 (for backwards compatibility)
        # Total: 34 + 85 + 50 + 4 + 1 + 1 + 2 + 1 = 178
        self.feature_dim = 178

    def _calculate_feature_dim(self) -> int:
        """Calculate total feature dimension."""
        return 178

    def encode(self, state: GameState, player: int) -> np.ndarray:
        """Encode game state for a specific player (legacy 178-feature format)."""
        features = []

        # Encode hole cards (one-hot for rank and suit)
        hole_cards = state.hole_cards[player]
        for rank, suit in hole_cards:
            rank_onehot = np.zeros(self.RANKS)
            rank_onehot[rank] = 1.0
            suit_onehot = np.zeros(self.SUITS)
            suit_onehot[suit] = 1.0
            features.extend(rank_onehot)
            features.extend(suit_onehot)

        # Encode community cards
        community_onehot = np.zeros(5 * (self.RANKS + self.SUITS))
        for i, (rank, suit) in enumerate(state.community_cards):
            if i < 5:
                offset = i * (self.RANKS + self.SUITS)
                rank_onehot = np.zeros(self.RANKS)
                rank_onehot[rank] = 1.0
                suit_onehot = np.zeros(self.SUITS)
                suit_onehot[suit] = 1.0
                community_onehot[offset:offset+self.RANKS] = rank_onehot
                community_onehot[offset+self.RANKS:offset+self.RANKS+self.SUITS] = suit_onehot
        features.extend(community_onehot)

        # Encode betting history (MAX_HISTORY = 25 for old checkpoints)
        history_features = np.zeros(self.MAX_HISTORY * 2)
        for i, (p, action, amount) in enumerate(state.betting_history[:self.MAX_HISTORY]):
            # Action encoding
            action_map = {
                Action.FOLD: 0.0,
                Action.CHECK: 1.0,
                Action.CALL: 2.0,
                Action.BET: 3.0,
                Action.RAISE: 4.0
            }
            action_val = action_map.get(action, 0.0)

            # Amount (normalized)
            amount_norm = amount / state.big_blind if state.big_blind > 0 else 0.0

            history_features[i * 2] = action_val
            history_features[i * 2 + 1] = amount_norm
        features.extend(history_features)

        # Pot and stack ratios
        total_chips = sum(state.stacks) + state.pot
        if total_chips > 0:
            pot_ratio = state.pot / total_chips
            stack_ratio_self = state.stacks[player] / total_chips
            stack_ratio_opp = state.stacks[1 - player] / total_chips
            to_call = (state.current_bets[1 - player] - state.current_bets[player]) / total_chips
        else:
            pot_ratio = 0.0
            stack_ratio_self = 0.0
            stack_ratio_opp = 0.0
            to_call = 0.0
        features.extend([pot_ratio, stack_ratio_self, stack_ratio_opp, to_call])

        # Position
        position = 1.0 if state.button == player else 0.0
        features.append(position)

        # Street
        street_norm = state.street / 3.0
        features.append(street_norm)

        # Current bets
        bet_self = state.current_bets[player] / state.big_blind if state.big_blind > 0 else 0.0
        bet_opp = state.current_bets[1 - player] / state.big_blind if state.big_blind > 0 else 0.0
        features.extend([bet_self, bet_opp])

        # Extra feature for backwards compatibility (always 0)
        # This brings total to 178 to match old checkpoints
        features.append(0.0)

        # Verify we have exactly 178 features
        assert len(features) == 178, f"Expected 178 features, got {len(features)}"

        return np.array(features, dtype=np.float32)

    def encode_batch(self, states: List[Tuple[GameState, int]]) -> torch.Tensor:
        """Encode a batch of states."""
        encoded = [self.encode(state, player) for state, player in states]
        return torch.tensor(np.stack(encoded), dtype=torch.float32)
