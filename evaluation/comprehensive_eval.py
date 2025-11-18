#!/usr/bin/env python3
"""Comprehensive evaluation of iteration 50 against iter 195 and baselines."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modal_deploy.config import checkpoint_volume, image
import modal

# Create a single evaluation app
eval_app = modal.App("poker-bot-comprehensive-eval")


@eval_app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    cpu=4,
    memory=8192,
    timeout=3600,
)
def check_checkpoint_exists(checkpoint_name: str) -> bool:
    """Check if a checkpoint exists on Modal."""
    return os.path.exists(f"/checkpoints/{checkpoint_name}")


@eval_app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    cpu=4,
    memory=8192,
    timeout=3600,
)
def evaluate_against_baseline_modal(
    current_iteration: int,
    baseline_type: str,  # 'random', 'always_call', 'baseline', or iteration number
    num_games: int = 2000
):
    """Evaluate current bot against a baseline opponent."""
    import os
    import torch
    import numpy as np
    import random
    from poker_game.game import PokerGame, GameState, Action
    from poker_game.state_encoder import StateEncoder
    from models.value_policy_net import ValuePolicyNet
    
    # Define baseline agents inline
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
                    else:
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
    
    print(f"Evaluating iteration {current_iteration} vs {baseline_type}")
    print(f"Games: {num_games}")
    
    # Initialize game
    game = PokerGame(small_blind=50, big_blind=100, is_limit=False)
    state_encoder = StateEncoder()
    
    def load_network(checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        input_dim = state_encoder.feature_dim
        policy_net = ValuePolicyNet(input_dim=input_dim)
        if 'policy_net_state' in checkpoint:
            policy_net.load_state_dict(checkpoint['policy_net_state'])
        else:
            raise ValueError(f"Checkpoint missing policy_net_state: {checkpoint_path}")
        return policy_net
    
    # Load current bot
    current_checkpoint = f"/checkpoints/checkpoint_iter_{current_iteration}.pt"
    current_net = load_network(current_checkpoint)
    current_net.eval()
    
    # Setup baseline opponent
    baseline_net = None
    if baseline_type == 'random':
        baseline_agent = RandomAgent()
        use_network = False
    elif baseline_type == 'always_call':
        baseline_agent = AlwaysCallAgent()
        use_network = False
    elif baseline_type == 'baseline':
        baseline_agent = BaselineAgent(game)
        use_network = False
    else:
        baseline_checkpoint = f"/checkpoints/checkpoint_iter_{baseline_type}.pt"
        baseline_net = load_network(baseline_checkpoint)
        baseline_net.eval()
        use_network = True
    
    # Run evaluation
    current_wins = 0
    current_payoff = 0.0
    baseline_payoff = 0.0
    
    for game_num in range(num_games):
        state = game.reset()
        
        if random.random() < 0.5:
            current_is_player0 = True
        else:
            current_is_player0 = False
        
        while not state.is_terminal:
            player = state.current_player
            legal_actions = game.get_legal_actions(state)
            
            if len(legal_actions) == 0:
                break
            
            if (player == 0 and current_is_player0) or (player == 1 and not current_is_player0):
                state_encoding = state_encoder.encode(state, player)
                state_tensor = torch.tensor(state_encoding, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    _, policy_logits = current_net(state_tensor)
                    action_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                
                num_legal = len(legal_actions)
                if num_legal == 0:
                    break
                
                legal_probs = action_probs[:num_legal]
                if legal_probs.sum() > 0:
                    legal_probs = legal_probs / legal_probs.sum()
                else:
                    legal_probs = np.ones(num_legal) / num_legal
                
                action_idx = np.random.choice(num_legal, p=legal_probs)
                action, amount = legal_actions[action_idx]
            else:
                if use_network:
                    state_encoding = state_encoder.encode(state, player)
                    state_tensor = torch.tensor(state_encoding, dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        _, policy_logits = baseline_net(state_tensor)
                        action_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                    
                    num_legal = len(legal_actions)
                    if num_legal == 0:
                        break
                    
                    legal_probs = action_probs[:num_legal]
                    if legal_probs.sum() > 0:
                        legal_probs = legal_probs / legal_probs.sum()
                    else:
                        legal_probs = np.ones(num_legal) / num_legal
                    
                    action_idx = np.random.choice(num_legal, p=legal_probs)
                    action, amount = legal_actions[action_idx]
                else:
                    action, amount = baseline_agent.get_action(state, legal_actions)
            
            state = game.apply_action(state, action, amount)
        
        payoffs = game.get_payoff(state)
        
        if current_is_player0:
            current_payoff += payoffs[0]
            baseline_payoff += payoffs[1]
            if payoffs[0] > payoffs[1]:
                current_wins += 1
        else:
            current_payoff += payoffs[1]
            baseline_payoff += payoffs[0]
            if payoffs[1] > payoffs[0]:
                current_wins += 1
    
    win_rate = current_wins / num_games
    avg_payoff = current_payoff / num_games
    
    return {
        'current_iteration': current_iteration,
        'baseline_type': baseline_type,
        'num_games': num_games,
        'win_rate': win_rate,
        'avg_payoff': avg_payoff,
        'total_payoff': current_payoff
    }


@eval_app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    cpu=4,
    memory=8192,
    timeout=3600,
)
def evaluate_checkpoints_modal(
    iteration1: int,
    iteration2: int,
    num_games: int = 1000
):
    """Evaluate two checkpoints against each other."""
    import os
    import torch
    import numpy as np
    from poker_game.game import PokerGame
    from poker_game.state_encoder import StateEncoder
    from models.value_policy_net import ValuePolicyNet
    from evaluation.evaluator import Evaluator
    
    print(f"Evaluating iteration {iteration1} vs iteration {iteration2}")
    print(f"Games: {num_games}")
    
    game = PokerGame(small_blind=50, big_blind=100, is_limit=False)
    state_encoder = StateEncoder()
    evaluator = Evaluator(game, state_encoder)
    
    def load_network(checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        input_dim = state_encoder.feature_dim
        policy_net = ValuePolicyNet(input_dim=input_dim)
        if 'policy_net_state' in checkpoint:
            policy_net.load_state_dict(checkpoint['policy_net_state'])
        else:
            raise ValueError(f"Checkpoint missing policy_net_state: {checkpoint_path}")
        return policy_net
    
    checkpoint1_path = f"/checkpoints/checkpoint_iter_{iteration1}.pt"
    checkpoint2_path = f"/checkpoints/checkpoint_iter_{iteration2}.pt"
    
    print(f"Loading checkpoint 1: {checkpoint1_path}")
    network1 = load_network(checkpoint1_path)
    
    print(f"Loading checkpoint 2: {checkpoint2_path}")
    network2 = load_network(checkpoint2_path)
    
    print("Running evaluation...")
    result = evaluator.evaluate_agents(
        network1, network2, num_games=num_games, device='cpu'
    )
    
    return {
        'iteration1': iteration1,
        'iteration2': iteration2,
        'num_games': num_games,
        'iteration1_win_rate': result['agent1_win_rate'],
        'iteration2_win_rate': result['agent2_win_rate'],
        'iteration1_avg_payoff': result['agent1_payoff'],
        'iteration2_avg_payoff': result['agent2_payoff'],
        'full_result': result
    }


def main():
    """Run comprehensive evaluation."""
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive evaluation')
    parser.add_argument('--current', type=int, default=50,
                       help='Current iteration (default: 50)')
    parser.add_argument('--old', type=int, default=195,
                       help='Old iteration to compare (default: 195)')
    parser.add_argument('--num-games', type=int, default=2000,
                       help='Number of games per matchup (default: 2000)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPREHENSIVE EVALUATION")
    print("="*80)
    print(f"Current iteration: {args.current}")
    print(f"Old iteration: {args.old}")
    print(f"Games per matchup: {args.num_games}")
    print()
    
    # Check if checkpoints exist
    print("Checking checkpoint availability...")
    with eval_app.run():
        current_exists = check_checkpoint_exists.remote(f"checkpoint_iter_{args.current}.pt")
        old_exists = check_checkpoint_exists.remote(f"checkpoint_iter_{args.old}.pt")
    
    if not current_exists:
        print(f"✗ ERROR: checkpoint_iter_{args.current}.pt not found on Modal!")
        return
    if not old_exists:
        print(f"⚠ WARNING: checkpoint_iter_{args.old}.pt not found on Modal!")
        print("  Will skip comparison with old checkpoint.")
        compare_old = False
    else:
        compare_old = True
        print(f"✓ Found checkpoint_iter_{args.current}.pt")
        print(f"✓ Found checkpoint_iter_{args.old}.pt")
    
    print()
    
    results = {}
    
    # PART 1: Evaluate against baselines
    print("="*80)
    print("PART 1: EVALUATION AGAINST BASELINES")
    print("="*80)
    print()
    
    baselines = [
        ('random', 'Random Agent'),
        ('always_call', 'Always Call Agent'),
        ('baseline', 'Baseline Agent (TAG)'),
    ]
    
    with eval_app.run():
        for baseline_key, baseline_name in baselines:
            print(f"Evaluating Iter {args.current} vs {baseline_name}...")
            print("-" * 80)
            
            try:
                result = evaluate_against_baseline_modal.remote(
                    args.current, baseline_key, args.num_games
                )
                
                win_rate = result['win_rate']
                avg_payoff = result['avg_payoff']
                
                results[f"iter_{args.current}_vs_{baseline_key}"] = result
                
                print(f"  Win Rate: {win_rate:.2%}")
                print(f"  Avg Payoff: {avg_payoff:.2f} chips/game")
                
                # Assessment
                if win_rate > 0.70:
                    print(f"  ✓✓ Excellent! Strongly dominating (>70% win rate)")
                elif win_rate > 0.60:
                    print(f"  ✓ Very good! Significantly better (>60% win rate)")
                elif win_rate > 0.55:
                    print(f"  ✓ Good! Moderately better (>55% win rate)")
                elif win_rate > 0.50:
                    print(f"  ⚠ Slightly better (>50% win rate)")
                elif win_rate < 0.45:
                    print(f"  ✗ WORSE! Current bot is losing (<45% win rate)")
                else:
                    print(f"  ≈ Roughly even (45-55% win rate)")
                
                print()
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                print()
    
    # PART 2: Evaluate against old checkpoint
    if compare_old:
        print("="*80)
        print("PART 2: EVALUATION AGAINST OLD CHECKPOINT")
        print("="*80)
        print()
        
        print(f"Evaluating Iter {args.current} vs Iter {args.old}...")
        print("-" * 80)
        
        try:
            with eval_app.run():
                result = evaluate_checkpoints_modal.remote(
                    args.current, args.old, args.num_games
                )
            
            win_rate = result['iteration1_win_rate']
            avg_payoff = result['iteration1_avg_payoff']
            
            results[f"iter_{args.current}_vs_{args.old}"] = result
            
            print(f"  Win Rate (Iter {args.current}): {win_rate:.2%}")
            print(f"  Avg Payoff (Iter {args.current}): {avg_payoff:.2f} chips/game")
            print(f"  Win Rate (Iter {args.old}): {result['iteration2_win_rate']:.2%}")
            print(f"  Avg Payoff (Iter {args.old}): {result['iteration2_avg_payoff']:.2f} chips/game")
            
            # Assessment
            if win_rate > 0.55:
                print(f"  ✓ Current bot is significantly better (>55% win rate)")
            elif win_rate > 0.50:
                print(f"  ⚠ Current bot is slightly better (>50% win rate)")
            elif win_rate < 0.45:
                print(f"  ✗ Current bot is WORSE (<45% win rate) - concerning!")
            else:
                print(f"  ≈ Results are roughly even (45-55% win rate)")
            
            print()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # SUMMARY
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Matchup':<40} {'Win Rate':<15} {'Avg Payoff':<15}")
    print("-" * 80)
    
    for matchup, result in results.items():
        if 'win_rate' in result:
            win_rate = result['win_rate']
            payoff = result['avg_payoff']
        else:
            win_rate = result['iteration1_win_rate']
            payoff = result['iteration1_avg_payoff']
        
        print(f"{matchup:<40} {win_rate:<15.2%} {payoff:<15.2f}")
    
    print()
    print("INTERPRETATION:")
    print("-" * 80)
    print("Against weak opponents (random, always_call, baseline):")
    print("  - Should see 60-70%+ win rates if bot is strong")
    print("  - Lower rates suggest bot needs more training or has issues")
    print()
    if compare_old:
        print("Against old checkpoint:")
        print("  - Should see 55-65%+ win rates showing improvement")
        print("  - Near 50% suggests limited improvement")
        print("  - Below 50% suggests regression - investigate!")
        print()
    print("If win rates are low across all baselines:")
    print("  - Bot may not be converging properly")
    print("  - Consider checking training process and hyperparameters")
    
    return results


if __name__ == "__main__":
    main()
