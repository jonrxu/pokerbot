"""Main training script for Deep CFR poker bot."""

import argparse
import torch
import os
from pathlib import Path

from poker_game.game import PokerGame
from poker_game.state_encoder import StateEncoder
from models.value_policy_net import ValuePolicyNet
from training.trainer import Trainer
from checkpoints.checkpoint_manager import CheckpointManager
from evaluation.evaluator import Evaluator
from evaluation.metrics import Metrics


def create_checkpoint_callback(checkpoint_manager: CheckpointManager):
    """Create checkpoint callback function."""
    def callback(trainer: Trainer, iteration: int):
        training_state = trainer.get_training_state()
        checkpoint_manager.save_checkpoint(
            iteration=iteration,
            value_net_state=training_state['value_net_state'],
            policy_net_state=training_state['policy_net_state'],
            value_optimizer_state=training_state['value_optimizer_state'],
            policy_optimizer_state=training_state['policy_optimizer_state'],
            regret_memory=training_state['regret_memory'],
            strategy_memory=training_state['strategy_memory'],
            counterfactual_values=training_state['counterfactual_values'],
            metrics={'iteration': iteration}
        )
        print(f"Checkpoint saved at iteration {iteration}")
    return callback


def main():
    parser = argparse.ArgumentParser(description='Train Deep CFR poker bot')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--trajectories', type=int, default=1000, help='Trajectories per iteration')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=int, default=None, help='Resume from iteration')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--update-freq', type=int, default=10, help='Network update frequency')
    parser.add_argument('--checkpoint-freq', type=int, default=100, help='Checkpoint frequency')
    parser.add_argument('--evaluate-freq', type=int, default=200, help='Evaluation frequency')
    parser.add_argument('--eval-games', type=int, default=100, help='Number of games for evaluation')
    
    args = parser.parse_args()
    
    # Initialize components
    print("Initializing components...")
    game = PokerGame(small_blind=50, big_blind=100, is_limit=False)
    state_encoder = StateEncoder()
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)
    
    # Create or load networks
    input_dim = state_encoder.feature_dim
    value_net = ValuePolicyNet(input_dim=input_dim)
    policy_net = ValuePolicyNet(input_dim=input_dim)
    
    # Load checkpoint if resuming
    start_iteration = 0
    if args.resume is not None:
        checkpoint = checkpoint_manager.load_checkpoint(args.resume)
        if checkpoint:
            value_net.load_state_dict(checkpoint['value_net_state'])
            policy_net.load_state_dict(checkpoint['policy_net_state'])
            start_iteration = checkpoint['iteration'] + 1
            print(f"Resuming from iteration {start_iteration}")
    elif checkpoint_manager.get_latest_iteration() >= 0:
        # Try to load latest checkpoint
        checkpoint = checkpoint_manager.load_latest_checkpoint()
        if checkpoint:
            value_net.load_state_dict(checkpoint['value_net_state'])
            policy_net.load_state_dict(checkpoint['policy_net_state'])
            start_iteration = checkpoint['iteration'] + 1
            print(f"Resuming from latest checkpoint at iteration {start_iteration}")
    
    # Move to device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    value_net.to(device)
    policy_net.to(device)
    
    # Create trainer
    trainer = Trainer(
        game=game,
        state_encoder=state_encoder,
        value_net=value_net,
        policy_net=policy_net,
        device=device,
        learning_rate=args.learning_rate,
        trajectories_per_iteration=args.trajectories,
        network_update_frequency=args.update_freq
    )
    
    # Load training state if resuming
    if start_iteration > 0 and checkpoint:
        trainer.load_training_state({
            'iteration': checkpoint['iteration'],
            'value_net_state': checkpoint['value_net_state'],
            'policy_net_state': checkpoint['policy_net_state'],
            'value_optimizer_state': checkpoint.get('value_optimizer_state', {}),
            'policy_optimizer_state': checkpoint.get('policy_optimizer_state', {}),
            'regret_memory': checkpoint.get('regret_memory', {}),
            'strategy_memory': checkpoint.get('strategy_memory', {}),
            'counterfactual_values': checkpoint.get('counterfactual_values', {})
        })
    
    # Create checkpoint callback
    checkpoint_callback = create_checkpoint_callback(checkpoint_manager)
    
    # Evaluation setup
    evaluator = Evaluator(game, state_encoder)
    baseline_net = ValuePolicyNet(input_dim=input_dim).to(device)
    
    # Training loop
    print(f"\nStarting training: {args.iterations} iterations")
    print(f"Device: {device}, Trajectories per iteration: {args.trajectories}")
    
    for iteration in range(start_iteration, args.iterations):
        # Train iteration
        stats = trainer.train_iteration()
        
        # Checkpoint
        if (iteration + 1) % args.checkpoint_freq == 0:
            checkpoint_callback(trainer, iteration + 1)
        
        # Evaluate
        if (iteration + 1) % args.evaluate_freq == 0:
            print(f"\nEvaluating at iteration {iteration + 1}...")
            eval_result = evaluator.evaluate_agents(
                trainer.deep_cfr.value_net,
                baseline_net,
                num_games=args.eval_games,
                device=device
            )
            print(f"Evaluation results: Agent win rate = {eval_result['agent1_win_rate']:.2%}, "
                  f"Average payoff = {eval_result['agent1_payoff']:.2f}")
        
        # Print progress
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{args.iterations}: {stats}")
    
    # Final checkpoint
    checkpoint_callback(trainer, args.iterations)
    print("\nTraining complete!")


if __name__ == '__main__':
    main()

