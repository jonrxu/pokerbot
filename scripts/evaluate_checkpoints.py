#!/usr/bin/env python3
"""Evaluate current bot against previous checkpoints to measure improvement."""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modal_deploy.config import checkpoint_volume, image
import modal
from poker_game.game import PokerGame
from poker_game.state_encoder import StateEncoder
from models.value_policy_net import ValuePolicyNet
from evaluation.evaluator import Evaluator


# Create Modal app for evaluation
eval_app = modal.App("poker-bot-evaluation")

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
    from poker_game.game import PokerGame
    from poker_game.state_encoder import StateEncoder
    from models.value_policy_net import ValuePolicyNet
    from evaluation.evaluator import Evaluator
    
    print(f"Evaluating iteration {iteration1} vs iteration {iteration2}")
    print(f"Games: {num_games}")
    
    # Initialize game and encoder
    game = PokerGame(small_blind=50, big_blind=100, is_limit=False)
    state_encoder = StateEncoder()
    evaluator = Evaluator(game, state_encoder)
    
    def load_network(checkpoint_path):
        """Load network from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        input_dim = state_encoder.feature_dim
        
        # Create policy network (we use policy for evaluation)
        policy_net = ValuePolicyNet(input_dim=input_dim)
        
        if 'policy_net_state' in checkpoint:
            policy_net.load_state_dict(checkpoint['policy_net_state'])
        else:
            raise ValueError(f"Checkpoint missing policy_net_state: {checkpoint_path}")
        
        return policy_net
    
    # Load networks
    checkpoint1_path = f"/checkpoints/checkpoint_iter_{iteration1}.pt"
    checkpoint2_path = f"/checkpoints/checkpoint_iter_{iteration2}.pt"
    
    print(f"Loading checkpoint 1: {checkpoint1_path}")
    network1 = load_network(checkpoint1_path)
    
    print(f"Loading checkpoint 2: {checkpoint2_path}")
    network2 = load_network(checkpoint2_path)
    
    # Run evaluation
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
    """Run evaluations."""
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate checkpoints')
    parser.add_argument('--current', type=int, default=195,
                       help='Current iteration (default: 195)')
    parser.add_argument('--compare', type=int, nargs='+', default=[10, 100],
                       help='Iterations to compare against (default: 10, 100)')
    parser.add_argument('--num-games', type=int, default=1000,
                       help='Number of games per matchup (default: 1000)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CHECKPOINT EVALUATION")
    print("="*80)
    print(f"Current iteration: {args.current}")
    print(f"Comparing against: {args.compare}")
    print(f"Games per matchup: {args.num_games}")
    print()
    
    results = {}
    
    with eval_app.run():
        for old_iter in args.compare:
            print(f"Evaluating: Iteration {args.current} vs Iteration {old_iter}")
            print("-" * 80)
            
            try:
                result = evaluate_checkpoints_modal.remote(
                    args.current, old_iter, args.num_games
                )
                
                win_rate = result['iteration1_win_rate']
                avg_payoff = result['iteration1_avg_payoff']
                
                results[f"{args.current}_vs_{old_iter}"] = result
                
                print(f"  Win Rate: {win_rate:.2%}")
                print(f"  Avg Payoff: {avg_payoff:.2f} chips/game")
                
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
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    for matchup, result in results.items():
        win_rate = result['iteration1_win_rate']
        payoff = result['iteration1_avg_payoff']
        print(f"{matchup}:")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Avg Payoff: {payoff:.2f} chips/game")
        print()
    
    return results


if __name__ == "__main__":
    main()
