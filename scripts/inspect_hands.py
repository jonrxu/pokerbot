#!/usr/bin/env python3
"""Inspect detailed hand rollouts for a given checkpoint vs a baseline.

This runs a small number of games in Modal and prints human-readable
hand histories (hole cards, board, actions, and payoffs) so we can
visually sanity-check the agent's decisions.
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import modal  # type: ignore
from modal_deploy.config import checkpoint_volume, image


inspect_app = modal.App("poker-bot-hand-inspect")


@inspect_app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    cpu=2,
    memory=4096,
    timeout=600,
)
def inspect_hands_modal(
    iteration: int,
    baseline_type: str = "always_call",  # 'random', 'always_call', 'baseline', or iteration number
    num_hands: int = 5,
):
    """Run a few hands and print detailed histories."""
    import random
    import torch
    import numpy as np

    from poker_game.game import PokerGame, GameState, Action
    from poker_game.state_encoder import StateEncoder
    from models.value_policy_net import ValuePolicyNet

    # Simple baseline agents (copied from evaluate_benchmarks logic)
    class RandomAgent:
        def get_action(self, state: GameState, legal_actions):
            if len(legal_actions) == 0:
                return (Action.FOLD, 0)
            return random.choice(legal_actions)

    class AlwaysCallAgent:
        def get_action(self, state: GameState, legal_actions):
            to_call = state.current_bets[1 - state.current_player] - state.current_bets[state.current_player]
            if to_call == 0:
                for action, amount in legal_actions:
                    if action == Action.CHECK:
                        return (action, amount)
            else:
                for action, amount in legal_actions:
                    if action == Action.CALL:
                        return (action, amount)
            return (Action.FOLD, 0)

    class BaselineAgent:
        """Simple baseline agent for warm-start training (TAG-style)."""

        def __init__(self, game: PokerGame):
            self.game = game

        def get_action(self, state: GameState, legal_actions):
            if len(legal_actions) == 0:
                return (Action.FOLD, 0)

            player = state.current_player
            hole_cards = state.hole_cards[player]

            hand_strength = self._evaluate_hand_strength(hole_cards, state.community_cards)
            to_call = state.current_bets[1 - player] - state.current_bets[player]

            if hand_strength > 0.7:
                # Strong hand - bet/raise
                if to_call == 0:
                    bet_actions = [a for a in legal_actions if a[0] in [Action.BET, Action.RAISE]]
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
                    else:
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

        def _evaluate_hand_strength(self, hole_cards, community_cards):
            ranks = [card[0] for card in hole_cards]
            if ranks[0] == ranks[1]:
                return 0.6 + min(ranks[0], 12) / 12.0 * 0.3
            max_rank = max(ranks)
            if max_rank >= 10:
                return 0.4 + (max_rank - 10) / 2.0 * 0.2
            if hole_cards[0][1] == hole_cards[1][1]:
                return 0.3
            return 0.2

    def card_to_str(card):
        """Convert (rank, suit) into something like 'Ah' or 'Td'."""
        rank, suit = card
        RANKS = "23456789TJQKA"
        SUITS = "cdhs"  # clubs, diamonds, hearts, spades
        return f"{RANKS[rank]}{SUITS[suit]}"

    def cards_to_str(cards):
        return " ".join(card_to_str(c) for c in cards)

    def load_network(checkpoint_path: str, encoder: StateEncoder) -> ValuePolicyNet:
        if not torch.cuda.is_available():
            device = "cpu"
        else:
            device = "cuda"

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        input_dim = encoder.feature_dim
        net = ValuePolicyNet(input_dim=input_dim)
        if "policy_net_state" not in checkpoint:
            raise ValueError(f"Checkpoint missing policy_net_state: {checkpoint_path}")
        net.load_state_dict(checkpoint["policy_net_state"])
        net.to(device)
        net.eval()
        return net

    print(f"=== Inspecting hands for iteration {iteration} vs {baseline_type} ===")
    print(f"Number of hands: {num_hands}")
    print()

    game = PokerGame(small_blind=50, big_blind=100, is_limit=False)
    state_encoder = StateEncoder()

    # Load current bot
    current_checkpoint = f"/checkpoints/checkpoint_iter_{iteration}.pt"
    current_net = load_network(current_checkpoint, state_encoder)

    # Setup baseline opponent
    use_network_baseline = False
    baseline_agent = None
    baseline_net = None

    if baseline_type == "random":
        baseline_agent = RandomAgent()
    elif baseline_type == "always_call":
        baseline_agent = AlwaysCallAgent()
    elif baseline_type == "baseline":
        baseline_agent = BaselineAgent(game)
    else:
        # Treat as iteration number for a network baseline
        baseline_checkpoint = f"/checkpoints/checkpoint_iter_{baseline_type}.pt"
        baseline_net = load_network(baseline_checkpoint, state_encoder)
        use_network_baseline = True

    # Device for nets
    device = next(current_net.parameters()).device

    for hand_idx in range(num_hands):
        state = game.reset()
        # Randomly assign whether current net is player 0 or 1
        current_is_player0 = random.random() < 0.5

        print(f"--- Hand {hand_idx + 1} ---")
        print(f"Button: Player {state.button}")
        print(f"Current agent is player {'0' if current_is_player0 else '1'}")
        print(f"Hole cards:")
        print(f"  P0: {cards_to_str(state.hole_cards[0])}")
        print(f"  P1: {cards_to_str(state.hole_cards[1])}")
        print(f"Stacks: P0={state.stacks[0]}, P1={state.stacks[1]}")
        print(f"Pot: {state.pot}")

        last_street = state.street

        while not state.is_terminal:
            player = state.current_player
            legal_actions = game.get_legal_actions(state)
            if not legal_actions:
                break

            # Detect street change to show board
            if state.street != last_street:
                street_name = ["Preflop", "Flop", "Turn", "River"][state.street]
                print(f"  == {street_name} ==  Board: {cards_to_str(state.community_cards)}")
                last_street = state.street

            is_current = (player == 0 and current_is_player0) or (player == 1 and not current_is_player0)

            if is_current:
                # Current bot's move
                enc = state_encoder.encode(state, player)
                state_tensor = (
                    torch.tensor(enc, dtype=torch.float32, device=device)
                    .unsqueeze(0)
                )
                with torch.no_grad():
                    _, policy_logits = current_net(state_tensor)
                    action_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

                num_legal = len(legal_actions)
                legal_probs = action_probs[:num_legal]
                if legal_probs.sum() > 0:
                    legal_probs = legal_probs / legal_probs.sum()
                else:
                    legal_probs = np.ones(num_legal) / num_legal
                action_idx = np.random.choice(num_legal, p=legal_probs)
                action, amount = legal_actions[action_idx]
                actor = "CURRENT"
            else:
                # Baseline move
                if use_network_baseline:
                    enc = state_encoder.encode(state, player)
                    state_tensor = (
                        torch.tensor(enc, dtype=torch.float32, device=device)
                        .unsqueeze(0)
                    )
                    with torch.no_grad():
                        _, policy_logits = baseline_net(state_tensor)
                        action_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

                    num_legal = len(legal_actions)
                    legal_probs = action_probs[:num_legal]
                    if legal_probs.sum() > 0:
                        legal_probs = legal_probs / legal_probs.sum()
                    else:
                        legal_probs = np.ones(num_legal) / num_legal
                    action_idx = np.random.choice(num_legal, p=legal_probs)
                    action, amount = legal_actions[action_idx]
                else:
                    action, amount = baseline_agent.get_action(state, legal_actions)
                actor = "BASELINE"

            print(
                f"  Player {player} ({actor}) -> {action.name}"
                + (f" {amount}" if amount else "")
                + f" | Pot={state.pot}, Bets={state.current_bets}, Stacks={state.stacks}"
            )

            state = game.apply_action(state, action, amount)

        # Final board and payoffs
        print(f"Final board: {cards_to_str(state.community_cards)}")
        payoffs = game.get_payoff(state)
        print(f"Payoffs: P0={payoffs[0]}, P1={payoffs[1]}")
        if current_is_player0:
            current_payoff = payoffs[0]
        else:
            current_payoff = payoffs[1]
        print(f"Current agent payoff this hand: {current_payoff}")
        print()

    return {"status": "ok", "iteration": iteration, "baseline_type": baseline_type}


def main():
    parser = argparse.ArgumentParser(description="Inspect detailed hand rollouts.")
    parser.add_argument("--iteration", type=int, default=40, help="Checkpoint iteration to load as current agent.")
    parser.add_argument(
        "--baseline",
        type=str,
        default="always_call",
        choices=["random", "always_call", "baseline"],
        help="Baseline type: random, always_call, or baseline (TAG).",
    )
    parser.add_argument("--num-hands", type=int, default=5, help="Number of hands to print.")

    args = parser.parse_args()

    print("=" * 80)
    print("HAND INSPECTION")
    print("=" * 80)
    print(f"Iteration: {args.iteration}")
    print(f"Baseline: {args.baseline}")
    print(f"Num hands: {args.num_hands}")
    print()

    with inspect_app.run():
        inspect_hands_modal.remote(
            iteration=args.iteration,
            baseline_type=args.baseline,
            num_hands=args.num_hands,
        )


if __name__ == "__main__":
    main()


