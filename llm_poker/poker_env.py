"""Text-based poker environment suitable for LLM + RL.

This module wraps `PokerGame` and the text interface into a simple
reset/step environment that can be plugged into an external RL loop
such as Tinker's RL API.

Design:
    - Observations are *strings* (prompts) describing the current state
      for the acting player, plus the list of legal action tokens.
    - Actions are *strings* like "A0", "A1", which are mapped back to
      `(Action, amount)` via `ActionOption`s.
    - Reward is given *only at terminal* by default, as normalized chip
      profit for the training player, plus optional bonuses/penalties
      for formatting/validity.

The environment supports different opponent types:
    - 'self': same policy controls both players (self-play; the RL loop
      will simply act at every decision point).
    - 'random': opponent chooses a random legal action.
    - 'always_call': opponent calls/checks when possible, otherwise folds.
    - 'baseline': simple TAG baseline agent.

The RL framework is expected to:
    - Call `reset()` to get the initial prompt.
    - At each step, feed the prompt to the model, get an action token,
      and call `step(action_token)`.
    - Stop when `done` is True and use `reward` for learning.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import random

from poker_game.game import PokerGame, GameState, Action
from llm_poker.text_interface import (
    ActionOption,
    build_action_options,
    format_state_prompt,
    parse_action_token,
)


@dataclass
class StepResult:
    """Result of a single environment step."""

    prompt: str
    reward: float
    done: bool
    info: dict


class RandomAgent:
    """Random baseline opponent."""

    def get_action(self, state: GameState, legal_actions: List[Tuple[Action, int]]) -> Tuple[Action, int]:
        if not legal_actions:
            return (Action.FOLD, 0)
        return random.choice(legal_actions)


class AlwaysCallAgent:
    """Always-check/call baseline opponent."""

    def get_action(self, state: GameState, legal_actions: List[Tuple[Action, int]]) -> Tuple[Action, int]:
        if not legal_actions:
            return (Action.FOLD, 0)

        to_call = state.current_bets[1 - state.current_player] - state.current_bets[state.current_player]
        if to_call == 0:
            for act, amt in legal_actions:
                if act == Action.CHECK:
                    return act, amt
        else:
            for act, amt in legal_actions:
                if act == Action.CALL:
                    return act, amt

        # Fallback: fold
        return (Action.FOLD, 0)


class BaselineAgent:
    """Simple TAG-style baseline opponent (similar to bootstrap BaselineAgent)."""

    def __init__(self, game: PokerGame):
        self.game = game

    def get_action(self, state: GameState, legal_actions: List[Tuple[Action, int]]) -> Tuple[Action, int]:
        if not legal_actions:
            return (Action.FOLD, 0)

        player = state.current_player
        hole_cards = state.hole_cards[player]

        hand_strength = self._evaluate_hand_strength(hole_cards, state.community_cards)
        to_call = state.current_bets[1 - player] - state.current_bets[player]

        if hand_strength > 0.7:
            # Strong hand - bet/raise
            if to_call == 0:
                bet_actions = [a for a in legal_actions if a[0] in (Action.BET, Action.RAISE)]
                if bet_actions:
                    return random.choice(bet_actions)
            else:
                raise_actions = [a for a in legal_actions if a[0] == Action.RAISE]
                if raise_actions:
                    return random.choice(raise_actions)
                call_actions = [a for a in legal_actions if a[0] == Action.CALL]
                if call_actions:
                    return call_actions[0]
        elif hand_strength > 0.4:
            # Medium hand - call/check
            if to_call == 0:
                check_actions = [a for a in legal_actions if a[0] == Action.CHECK]
                if check_actions:
                    return check_actions[0]
            else:
                call_actions = [a for a in legal_actions if a[0] == Action.CALL]
                if call_actions and to_call < state.stacks[player] * 0.1:
                    return call_actions[0]
                return (Action.FOLD, 0)
        else:
            # Weak hand - fold or check
            if to_call == 0:
                check_actions = [a for a in legal_actions if a[0] == Action.CHECK]
                if check_actions:
                    return check_actions[0]
            return (Action.FOLD, 0)

        # Fallback: first legal action
        return legal_actions[0]

    def _evaluate_hand_strength(self, hole_cards, community_cards) -> float:
        """Very rough hand strength heuristic in [0, 1]."""
        ranks = [card[0] for card in hole_cards]
        # Pairs are relatively strong
        if ranks[0] == ranks[1]:
            return 0.6 + min(ranks[0], 12) / 12.0 * 0.3
        max_rank = max(ranks)
        if max_rank >= 10:
            return 0.4 + (max_rank - 10) / 2.0 * 0.2
        # Suited
        if hole_cards[0][1] == hole_cards[1][1]:
            return 0.3
        return 0.2


class PokerTextEnv:
    """Text-based heads-up NLHE environment for LLM RL.

    This environment is agnostic to any specific RL library; it just
    exposes a simple:
        - `reset() -> prompt`
        - `step(action_token: str) -> StepResult`
    API, which can be wrapped by Tinker's RL loop.
    """

    def __init__(
        self,
        small_blind: int = 50,
        big_blind: int = 100,
        starting_stack: int = 20000,
        opponent_type: str = "self",  # 'self', 'random', 'always_call', 'baseline'
        formatting_bonus: float = 0.05,
        illegal_penalty: float = -0.5,
        payoff_scale: float = 20000.0,
    ):
        self.game = PokerGame(
            small_blind=small_blind,
            big_blind=big_blind,
            is_limit=False,
            starting_stack=starting_stack,
        )
        self.opponent_type = opponent_type
        self.formatting_bonus = formatting_bonus
        self.illegal_penalty = illegal_penalty
        self.payoff_scale = payoff_scale

        # Internal state
        self.state: Optional[GameState] = None
        self.last_action_options: List[ActionOption] = []

        # Opponent agent (used when opponent_type != 'self')
        if opponent_type == "random":
            self._opponent = RandomAgent()
        elif opponent_type == "always_call":
            self._opponent = AlwaysCallAgent()
        elif opponent_type == "baseline":
            self._opponent = BaselineAgent(self.game)
        else:
            self._opponent = None  # self-play

    @property
    def current_player(self) -> int:
        """Return the index of the player whose turn it is."""
        if self.state is None:
            return 0
        return self.state.current_player if self.state.current_player is not None else 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self) -> str:
        """Start a new hand and return the initial prompt for the acting player."""
        self.state = self.game.reset()
        self.last_action_options = build_action_options(self.game, self.state)
        prompt = format_state_prompt(self.game, self.state, self.last_action_options)
        return prompt

    def step(self, action_token: str) -> StepResult:
        """Apply the agent's action token and advance the game.

        For simplicity:
            - Reward is 0 for all non-terminal steps.
            - At terminal, reward is normalized chips won by Player 0.
            - Invalid tokens incur `illegal_penalty` and immediately end the hand.
        """
        if self.state is None:
            raise RuntimeError("Environment must be reset() before step().")

        # Interpret the token
        parsed = parse_action_token(action_token, self.last_action_options)
        if parsed is None:
            # Illegal / badly formatted action
            # Terminate the episode with a penalty.
            info = {"error": "invalid_action_token", "token": action_token}
            prompt = ""  # no next observation
            return StepResult(prompt=prompt, reward=self.illegal_penalty, done=True, info=info)

        action, amount = parsed
        state = self.state

        # Apply the agent's action
        state = self.game.apply_action(state, action, amount)

        # If the hand ended, compute reward
        if state.is_terminal:
            payoffs = self.game.get_payoff(state)
            # Reward logic:
            # If self-play, we return P0's reward by default in 'reward' field,
            # but 'payoffs' info contains both. The RL loop must handle attribution.
            raw_reward = payoffs[0]
            reward = max(-self.payoff_scale, min(self.payoff_scale, raw_reward)) / self.payoff_scale
            self.state = state
            return StepResult(prompt="", reward=reward, done=True, info={"payoffs": payoffs})

        # Otherwise, we may need to let the opponent act (if not self-play)
        if self.opponent_type != "self":
            # Opponent acts until it is our turn again or the hand ends.
            while (not state.is_terminal) and state.current_player == 1:
                legal_actions = self.game.get_legal_actions(state)
                if not legal_actions:
                    break
                opp_action, opp_amount = self._opponent.get_action(state, legal_actions)
                state = self.game.apply_action(state, opp_action, opp_amount)

            if state.is_terminal:
                payoffs = self.game.get_payoff(state)
                raw_reward = payoffs[0]
                reward = max(-self.payoff_scale, min(self.payoff_scale, raw_reward)) / self.payoff_scale
                self.state = state
                return StepResult(prompt="", reward=reward, done=True, info={"payoffs": payoffs})

        # Now it's our turn again (or P1's turn in self-play); build the next prompt.
        self.state = state
        self.last_action_options = build_action_options(self.game, state)
        prompt = format_state_prompt(self.game, state, self.last_action_options)

        # Optional small formatting bonus for a valid action
        reward = self.formatting_bonus
        return StepResult(prompt=prompt, reward=reward, done=False, info={})
