#!/usr/bin/env python3
"""Local diagnostic script to understand gameplay and advantage computation."""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from poker_game.game import PokerGame, Action, GameState
from poker_game.state_encoder import StateEncoder
from models.advantage_net import AdvantageNet
from models.policy_net import PolicyNet
from training.deep_cfr import DeepCFR

def play_game_with_analysis(iteration=15, num_games=5):
    """Play games and show detailed analysis."""
    print(f"="*80)
    print(f"PLAYING GAMES WITH ITERATION {iteration}")
    print(f"="*80)
    
    # Load checkpoint from Modal (or local if available)
    checkpoint_path = f"checkpoints/checkpoint_iter_{iteration}.pt"
    
    # For now, we'll create a script that can run on Modal or locally
    # If running locally, we need to download checkpoint first
    print(f"\nNote: This script expects checkpoint at: {checkpoint_path}")
    print("If running locally, download checkpoint from Modal first.")
    print()
    
    # Initialize
    game = PokerGame()
    encoder = StateEncoder()
    input_dim = encoder.feature_dim
    
    adv_net = AdvantageNet(input_dim)
    pol_net = PolicyNet(input_dim)
    
    # Try to load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        adv_net.load_state_dict(checkpoint['advantage_net_state'])
        pol_net.load_state_dict(checkpoint['policy_net_state'])
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"✗ Checkpoint not found. Using random initialization.")
        print("  This will show what a random network does.")
    
    adv_net.eval()
    pol_net.eval()
    
    deep_cfr = DeepCFR(adv_net, pol_net, encoder, game)
    
    # Play games
    for game_num in range(num_games):
        print(f"\n{'='*80}")
        print(f"GAME {game_num + 1}")
        print(f"{'='*80}")
        
        state = game.reset()
        print(f"Initial State:")
        print(f"  Player 0 Hole Cards: {state.hole_cards[0]}")
        print(f"  Player 1 Hole Cards: {state.hole_cards[1]}")
        print(f"  Pot: {state.pot}, Stacks: {state.stacks}")
        print()
        
        move_num = 0
        while not state.is_terminal and move_num < 50:  # Safety limit
            player = state.current_player
            legal_actions = game.get_legal_actions(state)
            
            if not legal_actions:
                break
            
            print(f"Move {move_num + 1}: Player {player}'s turn")
            print(f"  Street: {['Preflop', 'Flop', 'Turn', 'River'][state.street]}")
            if state.community_cards:
                print(f"  Board: {state.community_cards}")
            print(f"  Pot: {state.pot}, Current Bets: {state.current_bets}")
            print(f"  Stacks: {state.stacks}")
            print(f"  Legal Actions: {legal_actions}")
            
            # Get network predictions
            encoding = encoder.encode(state, player)
            tensor = torch.tensor(encoding, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                # Advantage Net
                advantages_raw = adv_net(tensor).numpy()[0]
                # Inverse Symlog
                advantages_real = []
                for adv in advantages_raw:
                    clamped = np.clip(np.abs(adv), 0, 100)
                    real = np.sign(adv) * (np.exp(clamped) - 1)
                    advantages_real.append(real)
                
                # Policy Net
                logits = pol_net(tensor)
                probs = torch.softmax(logits, dim=1).numpy()[0]
            
            print(f"\n  Network Predictions:")
            action_names = {0: "FOLD", 1: "CHECK", 2: "CALL", 3: "BET", 4: "RAISE"}
            
            print(f"    Advantages (Real Chips):")
            for idx, (action, amount) in enumerate(legal_actions):
                action_name = action_names.get(action.value, "UNKNOWN")
                adv_real = advantages_real[idx] if idx < len(advantages_real) else 0.0
                prob = probs[idx] if idx < len(probs) else 0.0
                print(f"      {action_name:6} (amount={amount:4d}): Advantage={adv_real:8.2f}, Prob={prob:.4f}")
            
            # Choose action (greedy for analysis, or sample)
            action_probs_legal = [probs[i] if i < len(probs) else 0.0 for i in range(len(legal_actions))]
            if sum(action_probs_legal) > 0:
                action_probs_legal = np.array(action_probs_legal)
                action_probs_legal = action_probs_legal / action_probs_legal.sum()
                action_idx = np.random.choice(len(legal_actions), p=action_probs_legal)
            else:
                action_idx = 0
            
            action, amount = legal_actions[action_idx]
            print(f"\n  → Chosen Action: {action_names.get(action.value, 'UNKNOWN')} (amount={amount})")
            
            state = game.apply_action(state, action, amount)
            move_num += 1
        
        # Final payoffs
        payoffs = game.get_payoff(state)
        print(f"\n  Final Payoffs: Player 0: {payoffs[0]}, Player 1: {payoffs[1]}")
        print(f"  Winner: Player {0 if payoffs[0] > payoffs[1] else 1}")

def analyze_advantage_loss_computation(iteration=15):
    """Analyze how advantage loss is computed."""
    print(f"\n{'='*80}")
    print(f"ANALYZING ADVANTAGE LOSS COMPUTATION")
    print(f"{'='*80}")
    
    # Simulate a simple scenario
    game = PokerGame()
    encoder = StateEncoder()
    input_dim = encoder.feature_dim
    
    adv_net = AdvantageNet(input_dim)
    pol_net = PolicyNet(input_dim)
    
    checkpoint_path = f"checkpoints/checkpoint_iter_{iteration}.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        adv_net.load_state_dict(checkpoint['advantage_net_state'])
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"✗ Checkpoint not found. Using random initialization.")
    
    adv_net.eval()
    
    # Create a simple state
    state = game.reset()
    player = state.current_player
    legal_actions = game.get_legal_actions(state)
    
    print(f"\nExample State:")
    print(f"  Player: {player}")
    print(f"  Legal Actions: {legal_actions}")
    
    # Encode state
    encoding = encoder.encode(state, player)
    tensor = torch.tensor(encoding, dtype=torch.float32).unsqueeze(0)
    
    # Get network prediction
    with torch.no_grad():
        advantages_raw = adv_net(tensor).numpy()[0]
    
    print(f"\nNetwork Output (Raw/Symlog Space):")
    for idx, (action, amount) in enumerate(legal_actions):
        action_name = {0: "FOLD", 1: "CHECK", 2: "CALL", 3: "BET", 4: "RAISE"}.get(action.value, "UNKNOWN")
        raw_val = advantages_raw[idx] if idx < len(advantages_raw) else 0.0
        print(f"  {action_name}: {raw_val:.4f}")
    
    # Simulate what the target would be during training
    print(f"\nSimulated Training Target (Symlog Space):")
    print("  During traversal, we compute:")
    print("    action_value = traverse(next_state)")
    print("    node_value = sum(strategy * action_values)")
    print("    advantage = action_value - node_value")
    print("    target = symlog(advantage) = sign(adv) * log(1 + |adv|)")
    
    # Example: If we had advantages of [100, -50, 200] chips
    example_advantages_chips = [100, -50, 200]
    print(f"\n  Example Advantages (Chips): {example_advantages_chips}")
    
    symlog_targets = []
    for adv in example_advantages_chips:
        symlog_val = np.sign(adv) * np.log1p(np.abs(adv))
        symlog_targets.append(symlog_val)
        print(f"    Advantage {adv:6.0f} chips -> Symlog Target: {symlog_val:.4f}")
    
    print(f"\n  Network predicts: {[f'{x:.4f}' for x in advantages_raw[:len(example_advantages_chips)]]}")
    print(f"  Targets would be: {[f'{x:.4f}' for x in symlog_targets]}")
    
    # Compute MSE loss
    if len(advantages_raw) >= len(symlog_targets):
        mse = np.mean((advantages_raw[:len(symlog_targets)] - symlog_targets) ** 2)
        print(f"\n  MSE Loss: {mse:.4f}")
        print(f"  RMSE: {np.sqrt(mse):.4f}")
        print(f"  Interpretation: Network is off by ~{np.sqrt(mse):.4f} in Symlog space")
        
        # Convert back to chips
        print(f"\n  What does this mean in chips?")
        for i, (pred_raw, target_symlog) in enumerate(zip(advantages_raw[:len(symlog_targets)], symlog_targets)):
            # Inverse transform
            pred_real = np.sign(pred_raw) * (np.exp(np.clip(np.abs(pred_raw), 0, 100)) - 1)
            target_real = example_advantages_chips[i]
            error_chips = abs(pred_real - target_real)
            print(f"    Action {i}: Predicted {pred_real:8.2f} chips, Target {target_real:8.2f} chips, Error: {error_chips:8.2f} chips")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=15, help='Checkpoint iteration')
    parser.add_argument('--num-games', type=int, default=3, help='Number of games to play')
    parser.add_argument('--analyze-loss', action='store_true', help='Analyze loss computation')
    args = parser.parse_args()
    
    if args.analyze_loss:
        analyze_advantage_loss_computation(args.iteration)
    else:
        play_game_with_analysis(args.iteration, args.num_games)
        analyze_advantage_loss_computation(args.iteration)

