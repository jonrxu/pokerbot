#!/usr/bin/env python3
"""Evaluate current bot against baseline opponents and early iterations."""

import sys
import os
import torch
import numpy as np
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modal_deploy.config import checkpoint_volume, image
import modal
from poker_game.game import PokerGame, GameState, Action
from poker_game.state_encoder import StateEncoder
from models.value_policy_net import ValuePolicyNet


# Create Modal app for evaluation
eval_app = modal.App("poker-bot-benchmark-eval")


class RandomAgent:
    """Random baseline agent - plays completely randomly."""
    
    def get_action(self, state: GameState, legal_actions):
        """Return random legal action."""
        if len(legal_actions) == 0:
            return (Action.FOLD, 0)
        return random.choice(legal_actions)


class AlwaysCallAgent:
    """Always calls/checks - very weak baseline."""
    
    def get_action(self, state: GameState, legal_actions):
        """Always call or check."""
        to_call = state.current_bets[1 - state.current_player] - state.current_bets[state.current_player]
        
        if to_call == 0:
            # Check if possible
            for action, amount in legal_actions:
                if action == Action.CHECK:
                    return (action, amount)
        else:
            # Call if possible
            for action, amount in legal_actions:
                if action == Action.CALL:
                    return (action, amount)
        
        # Fallback: fold
        return (Action.FOLD, 0)


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
    import random
    from poker_game.game import PokerGame, GameState, Action
    from poker_game.state_encoder import StateEncoder
    from models.value_policy_net import ValuePolicyNet
    
    # Define BaselineAgent inline since bootstrap module may not be available
    class BaselineAgent:
        """Simple baseline agent for warm-start training."""
        
        def __init__(self, game: PokerGame):
            self.game = game
        
        def get_action(self, state: GameState, legal_actions):
            """Get action using simple heuristic strategy."""
            if len(legal_actions) == 0:
                return (Action.FOLD, 0)
            
            player = state.current_player
            hole_cards = state.hole_cards[player]
            
            # Simple tight-aggressive strategy
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
            """Evaluate hand strength (0-1)."""
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
        """Load network from checkpoint."""
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
        # It's an iteration number
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
        
        # Randomly assign positions
        if random.random() < 0.5:
            current_is_player0 = True
        else:
            current_is_player0 = False
        
        while not state.is_terminal:
            player = state.current_player
            legal_actions = game.get_legal_actions(state)
            
            if len(legal_actions) == 0:
                break
            
            # Get action
            if (player == 0 and current_is_player0) or (player == 1 and not current_is_player0):
                # Current bot's turn
                state_encoding = state_encoder.encode(state, player)
                state_tensor = torch.tensor(state_encoding, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    _, policy_logits = current_net(state_tensor)
                    action_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                
                # Sample action - map network output to legal actions
                # The network outputs probabilities for action indices 0..max_actions-1
                # Action index i corresponds to the i-th legal action
                # Mask out actions beyond the number of legal actions
                num_legal = len(legal_actions)
                if num_legal == 0:
                    break
                
                # Only use probabilities for the first num_legal actions
                legal_probs = action_probs[:num_legal]
                if legal_probs.sum() > 0:
                    legal_probs = legal_probs / legal_probs.sum()
                else:
                    # Uniform if all probabilities are zero
                    legal_probs = np.ones(num_legal) / num_legal
                
                action_idx = np.random.choice(num_legal, p=legal_probs)
                action, amount = legal_actions[action_idx]
            else:
                # Baseline opponent's turn
                if use_network:
                    state_encoding = state_encoder.encode(state, player)
                    state_tensor = torch.tensor(state_encoding, dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        _, policy_logits = baseline_net(state_tensor)
                        action_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                    
                    # Same logic as above
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
        
        # Get payoffs
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


def main():
    """Run benchmark evaluations."""
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate against benchmarks')
    parser.add_argument('--current', type=int, default=195,
                       help='Current iteration (default: 195)')
    parser.add_argument('--num-games', type=int, default=2000,
                       help='Number of games per matchup (default: 2000)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("BENCHMARK EVALUATION")
    print("="*80)
    print(f"Current iteration: {args.current}")
    print(f"Games per matchup: {args.num_games}")
    print()
    
    # Evaluate against different baselines
    baselines = [
        ('random', 'Random Agent'),
        ('always_call', 'Always Call Agent'),
        ('baseline', 'Baseline Agent (TAG)'),
        ('1', 'Iteration 1 (very early)'),
        ('2', 'Iteration 2'),
        ('5', 'Iteration 5'),
        ('10', 'Iteration 10'),
    ]
    
    results = {}
    
    with eval_app.run():
        for baseline_key, baseline_name in baselines:
            print(f"Evaluating vs {baseline_name}...")
            print("-" * 80)
            
            try:
                result = evaluate_against_baseline_modal.remote(
                    args.current, baseline_key, args.num_games
                )
                
                win_rate = result['win_rate']
                avg_payoff = result['avg_payoff']
                
                results[baseline_name] = result
                
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
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Opponent':<30} {'Win Rate':<15} {'Avg Payoff':<15}")
    print("-" * 80)
    
    for name, result in results.items():
        win_rate = result['win_rate']
        payoff = result['avg_payoff']
        print(f"{name:<30} {win_rate:<15.2%} {payoff:<15.2f}")
    
    print()
    print("INTERPRETATION:")
    print("-" * 80)
    print("Against weak opponents (random, always_call, baseline):")
    print("  - Should see 60-70%+ win rates if bot is strong")
    print("  - Lower rates suggest bot needs more training")
    print()
    print("Against early iterations:")
    print("  - Should see 55-65%+ win rates showing improvement")
    print("  - Near 50% suggests limited improvement")
    print()
    print("If win rates are low across all baselines:")
    print("  - Bot may not be converging properly")
    print("  - Consider checking training process")
    
    return results


if __name__ == "__main__":
    main()

