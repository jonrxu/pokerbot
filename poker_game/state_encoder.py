"""State encoding for neural network input."""

import numpy as np
import torch
from typing import List, Tuple
from .game import GameState, Action

# Try to import treys for hand evaluation
try:
    from treys import Card, Evaluator
    HAS_TREYS = True
except ImportError:
    HAS_TREYS = False


class StateEncoder:
    """Encodes poker game state into neural network input tensor."""
    
    RANKS = 13
    SUITS = 4
    MAX_HISTORY = 20  # Maximum betting history length
    
    def __init__(self):
        self.feature_dim = self._calculate_feature_dim()
        if HAS_TREYS:
            self.evaluator = Evaluator()
    
    def _calculate_feature_dim(self) -> int:
        """Calculate total feature dimension."""
        # Hole cards: 2 * (13 + 4) = 34
        # Community cards: 5 * (13 + 4) = 85
        # Betting history: MAX_HISTORY * (action_type + amount) = 20 * 2 = 40
        # Pot/stack ratios: 4
        # Position: 1
        # Street: 1
        # Current bets: 2
        # Hand Strength features:
        # - Strength (normalized rank): 1
        # - Hand Class (one-hot): 10
        return 34 + 85 + (self.MAX_HISTORY * 2) + 4 + 1 + 1 + 2 + 1 + 10
    
    def _get_treys_card(self, rank: int, suit: int):
        """Convert internal card representation to treys Card."""
        if not HAS_TREYS:
            return 0
        # rank 0..12 -> '2'..'A'
        # suit 0..3 -> 's','h','d','c'
        r_char = '23456789TJQKA'[rank]
        s_char = 'shdc'[suit]
        return Card.new(f"{r_char}{s_char}")

    def encode(self, state: GameState, player: int) -> np.ndarray:
        """Encode game state for a specific player."""
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
        
        # Encode betting history
        history_features = np.zeros(self.MAX_HISTORY * 2)
        for i, (p, action, amount) in enumerate(state.betting_history[:self.MAX_HISTORY]):
            # Action encoding: FOLD=0, CHECK=1, CALL=2, BET=3, RAISE=4
            action_map = {
                Action.FOLD: 0.0,
                Action.CHECK: 1.0,
                Action.CALL: 2.0,
                Action.BET: 3.0,
                Action.RAISE: 4.0
            }
            action_val = action_map.get(action, 0.0)
            # Normalize amount by big blind
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
        
        # Position (1 if button, 0 if big blind)
        position = 1.0 if state.button == player else 0.0
        features.append(position)
        
        # Street (normalized 0-1)
        street_norm = state.street / 3.0
        features.append(street_norm)
        
        # Current bets (normalized)
        bet_self = state.current_bets[player] / state.big_blind if state.big_blind > 0 else 0.0
        bet_opp = state.current_bets[1 - player] / state.big_blind if state.big_blind > 0 else 0.0
        features.extend([bet_self, bet_opp])

        # --- Explicit Hand Features (The "Sight" Fix) ---
        hand_strength = 0.0
        hand_class_onehot = np.zeros(10)
        
        if HAS_TREYS:
            try:
                hole = [self._get_treys_card(r, s) for r, s in hole_cards]
                board = [self._get_treys_card(r, s) for r, s in state.community_cards]
                
                if len(board) >= 3:
                    # Post-flop: Use full evaluation
                    rank = self.evaluator.evaluate(board, hole)
                    # rank is 1 (best) to 7462 (worst)
                    # Normalize to 0.0 - 1.0 (1.0 is best)
                    hand_strength = 1.0 - (rank / 7462.0)
                    
                    rank_class = self.evaluator.get_rank_class(rank)
                    # treys returns 1 (Royal Flush) to 9 (High Card)
                    # We map 1->9 (index 9), 9->1 (index 1), or just use index = 10 - rank_class
                    # Let's just use 0-based index: index = rank_class - 1
                    if 1 <= rank_class <= 10:
                         hand_class_onehot[rank_class - 1] = 1.0
                         
                else:
                    # Pre-flop: Heuristics
                    r1, s1 = hole_cards[0]
                    r2, s2 = hole_cards[1]
                    
                    # Pair
                    if r1 == r2:
                        hand_strength = 0.5 + (r1 / 26.0) # 0.5 to 1.0 approx
                        hand_class_onehot[8] = 1.0 # Treat as "Pair" class (index 8 for Pair in standard list usually)
                    else:
                        high_card = max(r1, r2)
                        hand_strength = high_card / 26.0 # 0.0 to 0.5 approx
                        hand_class_onehot[9] = 1.0 # High Card
                        
                        # Suited bonus
                        if s1 == s2:
                            hand_strength += 0.05
            except Exception:
                # Fallback if evaluation fails
                pass
        
        features.append(hand_strength)
        features.extend(hand_class_onehot)
        
        return np.array(features, dtype=np.float32)
    
    def encode_batch(self, states: List[Tuple[GameState, int]]) -> torch.Tensor:
        """Encode a batch of states."""
        encoded = [self.encode(state, player) for state, player in states]
        return torch.tensor(np.stack(encoded), dtype=torch.float32)
