#!/usr/bin/env python3
"""Analyze exploitative strategy to understand what it learned.

This script compares the exploitative model's strategy to GTO baseline
to identify specific exploitative patterns.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from poker_game.game import PokerGame, GameState, Action
from poker_game.state_encoder import StateEncoder
from poker_game.legacy_state_encoder import LegacyStateEncoder
from poker_game.information_set import get_information_set
from models.value_policy_net import ValuePolicyNet
from training.deep_cfr import DeepCFR


def load_agent(checkpoint_path, encoder, game, device='cpu'):
    """Load an agent from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    value_net = ValuePolicyNet(input_dim=encoder.feature_dim).to(device)
    policy_net = ValuePolicyNet(input_dim=encoder.feature_dim).to(device)

    if 'value_net_state' in checkpoint:
        value_net.load_state_dict(checkpoint['value_net_state'], strict=False)
    if 'policy_net_state' in checkpoint:
        policy_net.load_state_dict(checkpoint['policy_net_state'], strict=False)

    agent = DeepCFR(
        value_net=value_net,
        policy_net=policy_net,
        state_encoder=encoder,
        game=game,
        device=device
    )

    if 'strategy_memory' in checkpoint:
        agent.strategy_memory = checkpoint.get('strategy_memory', {})

    return agent


def analyze_situation(game, exploitative, gto, num_samples=1000):
    """Analyze strategies in different poker situations."""

    situations = {
        'preflop_strong': [],
        'preflop_weak': [],
        'postflop_hit': [],
        'postflop_miss': [],
        'facing_bet': [],
        'bluff_spot': [],
    }

    strategy_diffs = defaultdict(list)

    for _ in range(num_samples):
        state = game.reset()

        # Categorize situation
        hole_cards = state.hole_cards[0]
        ranks = sorted([card[0] for card in hole_cards], reverse=True)

        # Check if strong hand
        is_pair = ranks[0] == ranks[1]
        is_high = ranks[0] >= 10  # J or better

        legal_actions = game.get_legal_actions(state)
        if not legal_actions:
            continue

        info_set = get_information_set(state, 0)

        # Get strategies
        exploit_strat = exploitative.get_average_strategy(info_set, legal_actions)
        gto_strat = gto.get_average_strategy(info_set, legal_actions)

        if not exploit_strat or not gto_strat:
            continue

        # Calculate differences
        for action in legal_actions:
            exploit_prob = exploit_strat.get(action, 0)
            gto_prob = gto_strat.get(action, 0)
            diff = exploit_prob - gto_prob

            situation_key = None

            # Categorize
            if state.street == 0:  # Preflop
                if is_pair or is_high:
                    situation_key = 'preflop_strong'
                else:
                    situation_key = 'preflop_weak'
            else:  # Postflop
                # Simple heuristic: did we improve?
                if is_pair or is_high:
                    situation_key = 'postflop_hit'
                else:
                    situation_key = 'postflop_miss'

            # Check if facing bet
            to_call = state.current_bets[1] - state.current_bets[0]
            if to_call > 0:
                situation_key = 'facing_bet'

            # Check if bluff spot (weak hand, aggressive action)
            if not (is_pair or is_high) and action[0] in [Action.BET, Action.RAISE]:
                situation_key = 'bluff_spot'

            if situation_key:
                strategy_diffs[situation_key].append({
                    'action': action,
                    'exploit_prob': exploit_prob,
                    'gto_prob': gto_prob,
                    'diff': diff
                })

    return strategy_diffs


def print_analysis(strategy_diffs):
    """Print strategy analysis."""

    print("=" * 80)
    print("EXPLOITATIVE STRATEGY ANALYSIS")
    print("=" * 80)
    print()

    for situation, diffs in strategy_diffs.items():
        if not diffs:
            continue

        print(f"\n{situation.upper().replace('_', ' ')}")
        print("-" * 80)

        # Group by action type
        action_groups = defaultdict(list)
        for d in diffs:
            action_type = d['action'][0].name
            action_groups[action_type].append(d['diff'])

        for action_type, diff_list in action_groups.items():
            avg_diff = np.mean(diff_list)

            if abs(avg_diff) > 0.05:  # Significant difference
                direction = "MORE" if avg_diff > 0 else "LESS"
                print(f"  {action_type}: {direction} often ({avg_diff:+.2%} vs GTO)")

                if abs(avg_diff) > 0.15:
                    print(f"    🔥 MAJOR EXPLOIT!")

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print("Key Exploits:")
    print("  • BLUFF_SPOT differences: Adjusting bluffing frequency")
    print("  • FACING_BET differences: Exploiting opponent's bet sizing")
    print("  • PREFLOP differences: Adjusting hand selection vs GTO")
    print()
    print("What to look for:")
    print("  ✓ MORE aggressive vs tight GTO = Exploiting passivity")
    print("  ✓ LESS aggressive vs GTO = Avoiding GTO's traps")
    print("  ✓ Different bluffing frequency = Exploiting fold/call tendencies")


def main():
    parser = argparse.ArgumentParser(description='Analyze exploitative strategy')
    parser.add_argument('exploitative', help='Path to exploitative checkpoint')
    parser.add_argument('gto', help='Path to GTO checkpoint')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of situations to sample')

    args = parser.parse_args()

    print("Loading models...")
    game = PokerGame(small_blind=50, big_blind=100, is_limit=False)

    # Load exploitative (current encoder)
    exploit_encoder = StateEncoder()
    exploitative = load_agent(args.exploitative, exploit_encoder, game)
    print(f"  ✓ Loaded exploitative model")

    # Load GTO (legacy encoder)
    gto_encoder = LegacyStateEncoder()
    gto = load_agent(args.gto, gto_encoder, game)
    print(f"  ✓ Loaded GTO model")

    print(f"\nAnalyzing {args.samples} situations...")
    strategy_diffs = analyze_situation(game, exploitative, gto, args.samples)

    print_analysis(strategy_diffs)


if __name__ == '__main__':
    main()
