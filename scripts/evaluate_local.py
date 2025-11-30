#!/usr/bin/env python3
"""Evaluate a local checkpoint against baseline opponents."""

import sys
import os
import torch
import numpy as np
import random
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from poker_game.game import PokerGame, GameState, Action
from poker_game.state_encoder import StateEncoder
from poker_game.information_set import get_information_set
from models.value_policy_net import ValuePolicyNet
from training.deep_cfr import DeepCFR


class RandomAgent:
    """Random baseline agent."""

    def get_action(self, state: GameState, legal_actions):
        if len(legal_actions) == 0:
            return (Action.FOLD, 0)
        return random.choice(legal_actions)


class AlwaysCallAgent:
    """Always calls/checks."""

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
    """Simple tight-aggressive baseline."""

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
            if to_call == 0:
                check_actions = [a for a in legal_actions if a[0] == Action.CHECK]
                if check_actions:
                    return check_actions[0]
            return (Action.FOLD, 0)

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


def load_agent(checkpoint_path: str, game: PokerGame, encoder: StateEncoder):
    """Load an agent from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    input_dim = encoder.feature_dim
    value_net = ValuePolicyNet(input_dim=input_dim)
    policy_net = ValuePolicyNet(input_dim=input_dim)

    if 'value_net_state' in checkpoint:
        value_net.load_state_dict(checkpoint['value_net_state'])
    if 'policy_net_state' in checkpoint:
        policy_net.load_state_dict(checkpoint['policy_net_state'])

    agent = DeepCFR(
        value_net=value_net,
        policy_net=policy_net,
        state_encoder=encoder,
        game=game,
        device='cpu'
    )

    if 'strategy_memory' in checkpoint:
        agent.strategy_memory = checkpoint['strategy_memory']

    print(f"  ✓ Loaded successfully")
    return agent


def evaluate_matchup(agent, opponent, game, num_games=1000, opponent_name="Opponent"):
    """Play games between agent and opponent."""
    print(f"\nEvaluating vs {opponent_name} ({num_games} games)...")

    agent_wins = 0
    agent_payoff = 0.0

    for _ in range(num_games):
        state = game.reset()

        # Randomly assign positions
        agent_is_p0 = random.random() < 0.5

        while not state.is_terminal:
            player = state.current_player
            legal_actions = game.get_legal_actions(state)

            if len(legal_actions) == 0:
                break

            # Get action
            if (player == 0 and agent_is_p0) or (player == 1 and not agent_is_p0):
                # Agent's turn
                info_set = get_information_set(state, player)
                strategy = agent.get_average_strategy(info_set, legal_actions)

                if strategy:
                    actions = list(strategy.keys())
                    probs = np.array([strategy[a] for a in actions])
                    if probs.sum() > 0:
                        probs = probs / probs.sum()
                    else:
                        probs = np.ones(len(probs)) / len(probs)
                    chosen_idx = np.random.choice(len(actions), p=probs)
                    action, amount = actions[chosen_idx]
                else:
                    action, amount = random.choice(legal_actions)
            else:
                # Opponent's turn
                action, amount = opponent.get_action(state, legal_actions)

            state = game.apply_action(state, action, amount)

        # Get payoffs
        payoffs = game.get_payoff(state)

        if agent_is_p0:
            agent_payoff += payoffs[0]
            if payoffs[0] > payoffs[1]:
                agent_wins += 1
        else:
            agent_payoff += payoffs[1]
            if payoffs[1] > payoffs[0]:
                agent_wins += 1

    win_rate = agent_wins / num_games
    avg_payoff = agent_payoff / num_games

    return {
        'opponent': opponent_name,
        'win_rate': win_rate,
        'avg_payoff': avg_payoff,
        'num_games': num_games
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate local checkpoint against baselines')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--num-games', type=int, default=1000,
                       help='Number of games per matchup (default: 1000)')

    args = parser.parse_args()

    print("=" * 80)
    print("BASELINE EVALUATION (LOCAL)")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Games per matchup: {args.num_games}")
    print()

    # Initialize game
    game = PokerGame(small_blind=50, big_blind=100, is_limit=False)
    encoder = StateEncoder()

    # Load agent
    try:
        agent = load_agent(args.checkpoint, game, encoder)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Define baselines
    baselines = [
        (RandomAgent(), "Random Agent"),
        (AlwaysCallAgent(), "Always Call Agent"),
        (BaselineAgent(game), "Baseline TAG Agent"),
    ]

    results = []

    # Evaluate against each baseline
    for opponent, name in baselines:
        result = evaluate_matchup(agent, opponent, game, args.num_games, name)
        results.append(result)

        print(f"  Win Rate: {result['win_rate']:.2%}")
        print(f"  Avg Payoff: {result['avg_payoff']:+.2f} chips/game")

        if result['win_rate'] > 0.70:
            print(f"  ✓✓ Excellent! Strongly dominating (>70% win rate)")
        elif result['win_rate'] > 0.60:
            print(f"  ✓ Very good! Significantly better (>60% win rate)")
        elif result['win_rate'] > 0.55:
            print(f"  ✓ Good! Moderately better (>55% win rate)")
        elif result['win_rate'] > 0.50:
            print(f"  ⚠ Slightly better (>50% win rate)")
        else:
            print(f"  ✗ Needs improvement (<50% win rate)")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Opponent':<30} {'Win Rate':<15} {'Avg Payoff':<15}")
    print("-" * 80)

    for result in results:
        print(f"{result['opponent']:<30} {result['win_rate']:<15.2%} {result['avg_payoff']:<+15.2f}")

    # Overall assessment
    print()
    print("OVERALL ASSESSMENT:")
    print("-" * 80)
    avg_win_rate = np.mean([r['win_rate'] for r in results])

    if avg_win_rate > 0.70:
        print("🔥 EXCELLENT! Model is very strong against weak opponents")
    elif avg_win_rate > 0.60:
        print("✓ GOOD! Model has learned solid poker fundamentals")
    elif avg_win_rate > 0.50:
        print("⚠ OKAY: Model is learning but needs more training")
    else:
        print("✗ NEEDS WORK: Model struggling against weak opponents")

    print(f"Average win rate across all baselines: {avg_win_rate:.2%}")


if __name__ == "__main__":
    main()
