"""Poker game logic for heads-up Texas Hold'em."""

import random
from enum import Enum
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


class Action(Enum):
    """Poker actions."""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


@dataclass
class GameState:
    """Represents the current state of a poker game."""
    # Cards
    hole_cards: List[Tuple[int, int]] = field(default_factory=list)  # (rank, suit) for each player
    community_cards: List[Tuple[int, int]] = field(default_factory=list)
    deck: List[Tuple[int, int]] = field(default_factory=list)
    
    # Stacks and pot
    stacks: List[int] = field(default_factory=lambda: [20000, 20000])  # Starting stacks
    pot: int = 0
    current_bets: List[int] = field(default_factory=lambda: [0, 0])
    
    # Betting
    current_player: int = 0  # 0 or 1
    button: int = 0  # Button position
    street: int = 0  # 0=preflop, 1=flop, 2=turn, 3=river
    betting_history: List[Tuple[int, Action, int]] = field(default_factory=list)  # (player, action, amount)
    last_raise_amount: int = 0
    min_raise: int = 0
    max_raise: int = 0
    
    # Game state
    is_terminal: bool = False
    winner: Optional[int] = None
    showdown: bool = False
    
    # Game config
    small_blind: int = 50
    big_blind: int = 100
    is_limit: bool = False  # True for limit, False for no-limit


class PokerGame:
    """Heads-up Texas Hold'em poker game."""
    
    RANKS = 13  # 2-A
    SUITS = 4
    
    def __init__(self, small_blind: int = 50, big_blind: int = 100, is_limit: bool = False, 
                 starting_stack: int = 20000):
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.is_limit = is_limit
        self.starting_stack = starting_stack
        self.reset()
    
    def reset(self) -> GameState:
        """Reset game to initial state."""
        # Create deck
        deck = [(rank, suit) for rank in range(self.RANKS) for suit in range(self.SUITS)]
        random.shuffle(deck)
        
        # Deal hole cards
        hole_cards = [[deck.pop(), deck.pop()] for _ in range(2)]
        
        # Post blinds
        stacks = [self.starting_stack, self.starting_stack]
        pot = self.small_blind + self.big_blind
        current_bets = [self.big_blind, self.small_blind]
        stacks[0] -= self.big_blind
        stacks[1] -= self.small_blind
        
        # Determine button (alternates)
        button = random.randint(0, 1)
        current_player = 1 - button  # Big blind acts first preflop
        
        self.state = GameState(
            hole_cards=hole_cards,
            community_cards=[],
            deck=deck,
            stacks=stacks,
            pot=pot,
            current_bets=current_bets,
            current_player=current_player,
            button=button,
            street=0,
            betting_history=[],
            last_raise_amount=self.big_blind - self.small_blind,
            min_raise=self.big_blind,
            max_raise=stacks[current_player] if not self.is_limit else self.big_blind * 2,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            is_limit=self.is_limit
        )
        
        return self.state
    
    def get_legal_actions(self, state: GameState) -> List[Tuple[Action, int]]:
        """Get legal actions for current player."""
        # Terminal states have no legal actions
        if state.is_terminal or state.current_player is None:
            return []

        actions = []
        player = state.current_player
        to_call = state.current_bets[1 - player] - state.current_bets[player]

        # Can always fold
        actions.append((Action.FOLD, 0))

        # Can check/call
        if to_call == 0:
            actions.append((Action.CHECK, 0))
        else:
            if state.stacks[player] >= to_call:
                actions.append((Action.CALL, to_call))

        # Can bet/raise
        if to_call == 0:
            # Betting with a simplified, abstracted set of sizes:
            # small bet (min_bet), pot bet, and all-in (subject to constraints).
            min_bet = state.big_blind if state.street == 0 else state.big_blind
            max_bet = min(state.stacks[player], state.max_raise)
            if state.is_limit:
                max_bet = min(max_bet, state.big_blind * (2 if state.street < 2 else 4))

            if min_bet <= max_bet and state.stacks[player] >= min_bet:
                bet_candidates = set()

                # Small bet
                bet_candidates.add(min_bet)

                # Pot bet (at least min_bet)
                pot_size = max(min_bet, state.pot)
                
                # Add fractional pot bets (0.5x, 0.75x)
                # Ensure they are at least min_bet
                half_pot = max(min_bet, int(state.pot * 0.5))
                three_quarter_pot = max(min_bet, int(state.pot * 0.75))
                
                bet_candidates.add(half_pot)
                bet_candidates.add(three_quarter_pot)
                bet_candidates.add(pot_size)

                # All-in
                bet_candidates.add(state.stacks[player])

                for bet_size in sorted(bet_candidates):
                    if min_bet <= bet_size <= max_bet and state.stacks[player] >= bet_size:
                        actions.append((Action.BET, bet_size))
        else:
            # Raising with a simplified set of sizes:
            # minimum raise and all-in (subject to constraints).
            # Here, 'min_total' is the minimum total amount this player must have in the pot.
            min_total = state.current_bets[1 - player] + state.min_raise
            max_additional = min(state.stacks[player], state.max_raise)
            min_additional = min_total - state.current_bets[player]

            if min_additional <= max_additional and state.stacks[player] >= min_additional:
                raise_totals = set()

                # Minimum raise total
                raise_totals.add(min_total)

                # All-in total: current contributed chips + remaining stack
                all_in_total = state.current_bets[player] + state.stacks[player]
                raise_totals.add(all_in_total)

                for total in sorted(raise_totals):
                    additional = total - state.current_bets[player]
                    if additional >= min_additional and additional <= max_additional and state.stacks[player] >= additional:
                        actions.append((Action.RAISE, total))

        return actions
    
    def apply_action(self, state: GameState, action: Action, amount: int = 0) -> GameState:
        """Apply an action to the game state."""
        # Create new state with shallow copy of lists to prevent state corruption
        # We use list[:] for fast slicing instead of copy.deepcopy() which is slow
        new_state = GameState(**state.__dict__)
        
        # Copy mutable lists that are modified in-place or shared
        new_state.deck = state.deck[:] 
        new_state.community_cards = state.community_cards[:]
        # Note: hole_cards is a list of lists/tuples. The outer list needs copying.
        # The inner items are tuples/lists. If we modify inner items, we need deep copy.
        # But we usually just read hole cards or replace the whole entry.
        # Let's be safe for hole_cards since it's small (2 items).
        new_state.hole_cards = [h[:] if isinstance(h, list) else h for h in state.hole_cards]
        
        new_state.betting_history = state.betting_history[:]
        new_state.stacks = state.stacks[:]
        new_state.current_bets = state.current_bets[:]
        
        player = new_state.current_player
        
        if action == Action.FOLD:
            new_state.is_terminal = True
            new_state.winner = 1 - player
            return new_state
        
        # Update bets and stacks
        if action == Action.CHECK:
            pass  # No change
        elif action == Action.CALL:
            to_call = new_state.current_bets[1 - player] - new_state.current_bets[player]
            new_state.current_bets[player] += to_call
            new_state.stacks[player] -= to_call
            new_state.pot += to_call
        elif action == Action.BET:
            new_state.current_bets[player] += amount
            new_state.stacks[player] -= amount
            new_state.pot += amount
            new_state.last_raise_amount = amount
            new_state.min_raise = amount
        elif action == Action.RAISE:
            total_to_put = amount
            already_in = new_state.current_bets[player]
            additional = total_to_put - already_in
            new_state.current_bets[player] = total_to_put
            new_state.stacks[player] -= additional
            new_state.pot += additional
            new_state.last_raise_amount = total_to_put - new_state.current_bets[1 - player]
            new_state.min_raise = new_state.last_raise_amount
        
        # Record action
        new_state.betting_history.append((player, action, amount))
        
        # Check if betting round is complete
        if len(new_state.betting_history) >= 2:
            last_two = new_state.betting_history[-2:]
            # If both players have acted and bets are equal
            if (last_two[0][0] != last_two[1][0] and 
                new_state.current_bets[0] == new_state.current_bets[1]):
                # Move to next street or showdown
                if new_state.street < 3:
                    new_state.street += 1
                    new_state.current_player = 1 - new_state.button  # Button acts first post-flop
                    new_state.current_bets = [0, 0]
                    new_state.last_raise_amount = 0
                    new_state.min_raise = new_state.big_blind
                    new_state.max_raise = min(new_state.stacks[0], new_state.stacks[1])
                    if new_state.is_limit:
                        new_state.max_raise = new_state.big_blind * (2 if new_state.street < 2 else 4)
                    
                    # Deal community cards
                    if new_state.street == 1:  # Flop
                        new_state.community_cards = [new_state.deck.pop() for _ in range(3)]
                    elif new_state.street == 2:  # Turn
                        new_state.community_cards.append(new_state.deck.pop())
                    elif new_state.street == 3:  # River
                        new_state.community_cards.append(new_state.deck.pop())
                else:
                    # Showdown
                    new_state.is_terminal = True
                    new_state.showdown = True
                    new_state.current_player = None
            else:
                # Switch players
                new_state.current_player = 1 - new_state.current_player
        else:
            # Switch players
            new_state.current_player = 1 - new_state.current_player
        
        return new_state
    
    def evaluate_hand(self, hole_cards: List[Tuple[int, int]], 
                     community_cards: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
        """Evaluate hand strength. Returns (hand_rank, tiebreaker)."""
        all_cards = hole_cards + community_cards
        if len(all_cards) < 5:
            # Not enough cards, return high card
            ranks = sorted([c[0] for c in all_cards], reverse=True)
            return (0, ranks)
        
        # Generate all 5-card combinations
        from itertools import combinations
        best_rank = -1
        best_tiebreaker = []
        
        for combo in combinations(all_cards, 5):
            ranks = sorted([c[0] for c in combo], reverse=True)
            suits = [c[1] for c in combo]
            
            # Count ranks
            rank_counts = {}
            for r in ranks:
                rank_counts[r] = rank_counts.get(r, 0) + 1
            
            counts = sorted(rank_counts.values(), reverse=True)
            is_flush = len(set(suits)) == 1
            is_straight = False
            
            # Check straight
            unique_ranks = sorted(set(ranks))
            if len(unique_ranks) >= 5:
                for i in range(len(unique_ranks) - 4):
                    if unique_ranks[i+4] - unique_ranks[i] == 4:
                        is_straight = True
                        break
                # A-2-3-4-5 straight
                if unique_ranks == [0, 1, 2, 3, 12]:
                    is_straight = True
            
            # Determine hand rank
            if is_straight and is_flush:
                if ranks[0] == 12 and ranks[1] == 11:  # A-K-Q-J-10
                    hand_rank = 9  # Royal flush
                else:
                    hand_rank = 8  # Straight flush
                tiebreaker = sorted(ranks, reverse=True)
            elif counts == [4, 1]:
                hand_rank = 7  # Four of a kind
                four_kind = [r for r, c in rank_counts.items() if c == 4][0]
                kicker = [r for r, c in rank_counts.items() if c == 1][0]
                tiebreaker = [four_kind, kicker]
            elif counts == [3, 2]:
                hand_rank = 6  # Full house
                three_kind = [r for r, c in rank_counts.items() if c == 3][0]
                pair = [r for r, c in rank_counts.items() if c == 2][0]
                tiebreaker = [three_kind, pair]
            elif is_flush:
                hand_rank = 5  # Flush
                tiebreaker = sorted(ranks, reverse=True)
            elif is_straight:
                hand_rank = 4  # Straight
                tiebreaker = sorted(ranks, reverse=True)
            elif counts == [3, 1, 1]:
                hand_rank = 3  # Three of a kind
                three_kind = [r for r, c in rank_counts.items() if c == 3][0]
                kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
                tiebreaker = [three_kind] + kickers
            elif counts == [2, 2, 1]:
                hand_rank = 2  # Two pair
                pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
                kicker = [r for r, c in rank_counts.items() if c == 1][0]
                tiebreaker = pairs + [kicker]
            elif counts == [2, 1, 1, 1]:
                hand_rank = 1  # Pair
                pair = [r for r, c in rank_counts.items() if c == 2][0]
                kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
                tiebreaker = [pair] + kickers
            else:
                hand_rank = 0  # High card
                tiebreaker = sorted(ranks, reverse=True)
            
            if hand_rank > best_rank or (hand_rank == best_rank and tiebreaker > best_tiebreaker):
                best_rank = hand_rank
                best_tiebreaker = tiebreaker
        
        return (best_rank, best_tiebreaker)
    
    def get_payoff(self, state: GameState) -> List[float]:
        """Get final payoffs for each player (net profit)."""
        if not state.is_terminal:
            return [0.0, 0.0]
        
        starting_stack = self.starting_stack
        
        # Calculate payoffs accounting for pot distribution
        # When someone folds, the pot goes to the winner
        # When there's a showdown, the pot is split based on hand strength
        payoffs = [0.0, 0.0]
        
        if state.winner is not None:
            # Someone folded - winner gets the pot
            payoffs[state.winner] = state.pot
        
        elif state.showdown:
            # Showdown - evaluate hands
            hand0 = self.evaluate_hand(state.hole_cards[0], state.community_cards)
            hand1 = self.evaluate_hand(state.hole_cards[1], state.community_cards)
            
            if hand0 > hand1:
                payoffs[0] = state.pot
            elif hand1 > hand0:
                payoffs[1] = state.pot
            else:
                # Split pot
                payoffs[0] = state.pot / 2
                payoffs[1] = state.pot / 2
        
        # Convert to net profit (accounting for chips invested)
        # Stacks already reflect chips invested, so we need to add the pot winnings
        # and subtract starting stack
        net_payoffs = [
            state.stacks[0] + payoffs[0] - starting_stack,
            state.stacks[1] + payoffs[1] - starting_stack
        ]
        
        return net_payoffs

