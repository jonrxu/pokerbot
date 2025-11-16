"""Information set representation for CFR."""

from typing import List, Tuple, Optional
from .game import GameState, Action


class InformationSet:
    """Represents an information set (all states indistinguishable to a player)."""
    
    def __init__(self, player: int, state: GameState):
        self.player = player
        self.hole_cards = tuple(sorted(state.hole_cards[player]))
        self.community_cards = tuple(sorted(state.community_cards))
        self.street = state.street
        self.betting_history = tuple(state.betting_history)
        self.pot = state.pot
        self.stacks = tuple(state.stacks)
        self.current_bets = tuple(state.current_bets)
        self.button = state.button
        
        # Create hashable key
        self.key = self._create_key()
    
    def _create_key(self) -> str:
        """Create a unique key for this information set."""
        # Key components: player, hole cards, community cards, street, betting history
        parts = [
            f"p{self.player}",
            f"h{self.hole_cards}",
            f"c{self.community_cards}",
            f"s{self.street}",
            f"bh{self.betting_history}",
            f"pot{self.pot}",
            f"stacks{self.stacks}",
            f"bets{self.current_bets}",
            f"btn{self.button}"
        ]
        return "|".join(str(p) for p in parts)
    
    def __hash__(self):
        return hash(self.key)
    
    def __eq__(self, other):
        if not isinstance(other, InformationSet):
            return False
        return self.key == other.key
    
    def __repr__(self):
        return f"InfoSet(player={self.player}, street={self.street}, key={self.key[:50]}...)"


def get_information_set(state: GameState, player: int) -> InformationSet:
    """Extract information set for a player from game state."""
    return InformationSet(player, state)

