"""Text-based interface for the simplified heads-up NLHE environment.

This module turns `PokerGame` states into compact text prompts for an LLM
and maps between legal actions and simple action tokens like `A0`, `A1`, etc.

The intended usage is:
    - Call `build_action_options(state)` to get a list of `ActionOption`s.
    - Call `format_state_prompt(state, action_options, current_player)` to
      obtain the prompt string to feed to the LLM.
    - Parse the LLM's reply (e.g. "A1") with `parse_action_token(...)` to
      recover the chosen `(Action, amount)` for `PokerGame.apply_action`.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from poker_game.game import PokerGame, GameState, Action


@dataclass
class ActionOption:
    """A single legal action exposed to the LLM.

    Attributes:
        token: Short identifier like "A0", "A1", ...
        action: The underlying `Action` enum.
        amount: The associated amount (for BET/RAISE/CALL), or 0 for FOLD/CHECK.
        description: Human-readable description for the prompt.
    """

    token: str
    action: Action
    amount: int
    description: str


def _street_name(street: int) -> str:
    """Return human-readable street name."""
    if street == 0:
        return "Preflop"
    if street == 1:
        return "Flop"
    if street == 2:
        return "Turn"
    if street == 3:
        return "River"
    return f"Street-{street}"


def _card_to_str(card: Tuple[int, int]) -> str:
    """Convert (rank, suit) to a compact string like 'Ah' or 'Td'."""
    rank, suit = card
    ranks = "23456789TJQKA"
    suits = "cdhs"  # clubs, diamonds, hearts, spades
    return f"{ranks[rank]}{suits[suit]}"


def _cards_to_str(cards: List[Tuple[int, int]]) -> str:
    return " ".join(_card_to_str(c) for c in cards) if cards else "(none)"


def build_action_options(game: PokerGame, state: GameState) -> List[ActionOption]:
    """Build `ActionOption`s for all legal actions in the given state.

    The tokens are deterministic per call: A0, A1, ..., in the same order
    as returned by `game.get_legal_actions(state)`.
    """
    legal_actions = game.get_legal_actions(state)
    options: List[ActionOption] = []

    for idx, (action, amount) in enumerate(legal_actions):
        token = f"A{idx}"

        if action == Action.FOLD:
            desc = "Fold and give up the pot."
        elif action == Action.CHECK:
            desc = "Check (no chips added)."
        elif action == Action.CALL:
            desc = f"Call {amount} to match the opponent's bet."
        elif action == Action.BET:
            desc = f"Bet {amount} into the pot."
        elif action == Action.RAISE:
            desc = f"Raise total to {amount}."
        else:
            desc = f"{action.value} {amount}"

        options.append(ActionOption(token=token, action=action, amount=amount, description=desc))

    return options


def format_state_prompt(
    game: PokerGame,
    state: GameState,
    action_options: List[ActionOption],
    current_player: Optional[int] = None,
) -> str:
    """Format the current game state as a prompt for the LLM.

    The prompt includes:
        - Blinds and starting stacks
        - Current stacks and pot
        - Position info (button, who acts)
        - Board + hole cards for the acting player
        - Simple betting history
        - List of legal action tokens and their descriptions
        - A strict instruction to output exactly one action token.
    """
    if current_player is None:
        current_player = state.current_player

    street = _street_name(state.street)
    board_str = _cards_to_str(state.community_cards)
    hero_hole = _cards_to_str(state.hole_cards[current_player])

    stacks_str = f"P0={state.stacks[0]}, P1={state.stacks[1]}"
    bets_str = f"P0={state.current_bets[0]}, P1={state.current_bets[1]}"

    # Simple betting history as lines like "P0 BET 300"
    history_lines: List[str] = []
    for p, act, amt in state.betting_history[-6:]:  # last few actions for brevity
        if act in (Action.FOLD, Action.CHECK):
            history_lines.append(f"P{p} {act.name}")
        else:
            history_lines.append(f"P{p} {act.name} {amt}")
    history_str = "\n".join(history_lines) if history_lines else "(no previous actions this hand)"

    # Legal action section
    actions_section_lines = []
    token_list = []
    for opt in action_options:
        actions_section_lines.append(f"- {opt.token}: {opt.description}")
        token_list.append(opt.token)

    actions_block = "\n".join(actions_section_lines) if actions_section_lines else "(no legal actions)"
    token_choices = ", ".join(token_list) if token_list else ""

    instruction = (
        "Respond with exactly ONE action token from the list above, "
        "and nothing else. For example: A0"
    )

    prompt = (
        "You are playing heads-up no-limit Texas Hold'em.\n"
        f"Blinds: small blind {state.small_blind}, big blind {state.big_blind}.\n"
        f"Street: {street}\n"
        f"Button: Player {state.button}\n"
        f"Acting player: Player {current_player}\n"
        f"Stacks: {stacks_str}\n"
        f"Current bets: {bets_str}\n"
        f"Pot size: {state.pot}\n"
        f"Board cards: {board_str}\n"
        f"Your hole cards: {hero_hole}\n"
        "Recent betting history (most recent last):\n"
        f"{history_str}\n\n"
        "Legal actions:\n"
        f"{actions_block}\n\n"
        f"Valid tokens: {token_choices}\n"
        f"{instruction}\n"
    )

    return prompt


def parse_action_token(token: str, action_options: List[ActionOption]) -> Optional[Tuple[Action, int]]:
    """Given a token string from the LLM, return the corresponding action.

    Returns:
        (Action, amount) if the token is valid, otherwise None.
    """
    token = token.strip()
    for opt in action_options:
        if token == opt.token:
            return opt.action, opt.amount
    return None


