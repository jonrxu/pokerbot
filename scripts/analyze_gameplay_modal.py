#!/usr/bin/env python3
"""Modal diagnostic script to understand gameplay and advantage computation."""

import modal
from modal_deploy.config import checkpoint_volume, image
import torch
import numpy as np
from poker_game.game import PokerGame, Action
from poker_game.state_encoder import StateEncoder
from models.advantage_net import AdvantageNet
from models.policy_net import PolicyNet
from training.deep_cfr import DeepCFR

app = modal.App("gameplay-analysis")

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume}
)
def analyze_gameplay(iteration=15, num_games=3):
    """Play games and show detailed analysis."""
    print(f"="*80)
    print(f"PLAYING GAMES WITH ITERATION {iteration}")
    print(f"="*80)
    
    checkpoint_path = f"/checkpoints/checkpoint_iter_{iteration}.pt"
    
    # Initialize
    game = PokerGame()
    encoder = StateEncoder()
    input_dim = encoder.feature_dim
    
    adv_net = AdvantageNet(input_dim)
    pol_net = PolicyNet(input_dim)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        adv_net.load_state_dict(checkpoint['advantage_net_state'])
        pol_net.load_state_dict(checkpoint['policy_net_state'])
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return
    
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
        while not state.is_terminal and move_num < 50:
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
                adv_raw = advantages_raw[idx] if idx < len(advantages_raw) else 0.0
                prob = probs[idx] if idx < len(probs) else 0.0
                print(f"      {action_name:6} (amount={amount:4d}): Raw={adv_raw:7.3f}, Real={adv_real:8.2f} chips, Prob={prob:.4f}")
            
            # Choose action
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

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume}
)
def analyze_loss_computation(iteration=15):
    """Analyze how advantage loss is computed."""
    print(f"\n{'='*80}")
    print(f"ANALYZING ADVANTAGE LOSS COMPUTATION (Iteration {iteration})")
    print(f"{'='*80}")
    
    game = PokerGame()
    encoder = StateEncoder()
    input_dim = encoder.feature_dim
    
    adv_net = AdvantageNet(input_dim)
    
    checkpoint_path = f"/checkpoints/checkpoint_iter_{iteration}.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        adv_net.load_state_dict(checkpoint['advantage_net_state'])
        print(f"✓ Loaded checkpoint")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return
    
    adv_net.eval()
    
    # Sample a few states
    print(f"\nSampling states from actual gameplay...")
    
    states_analyzed = []
    for _ in range(5):
        state = game.reset()
        player = state.current_player
        legal_actions = game.get_legal_actions(state)
        
        if not legal_actions:
            continue
        
        encoding = encoder.encode(state, player)
        tensor = torch.tensor(encoding, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            advantages_raw = adv_net(tensor).numpy()[0]
        
        # Simulate what targets would be
        # In real training, targets come from traversal
        # For analysis, let's see what the network predicts
        
        states_analyzed.append({
            'state': state,
            'legal_actions': legal_actions,
            'advantages_raw': advantages_raw[:len(legal_actions)]
        })
    
    print(f"\nExample Network Outputs (Raw/Symlog Space):")
    action_names = {0: "FOLD", 1: "CHECK", 2: "CALL", 3: "BET", 4: "RAISE"}
    
    for i, sample in enumerate(states_analyzed[:3]):
        print(f"\n  State {i+1}:")
        print(f"    Street: {['Preflop', 'Flop', 'Turn', 'River'][sample['state'].street]}")
        print(f"    Legal Actions: {sample['legal_actions']}")
        for idx, (action, amount) in enumerate(sample['legal_actions']):
            action_name = action_names.get(action.value, "UNKNOWN")
            raw_val = sample['advantages_raw'][idx]
            # Inverse transform
            real_val = np.sign(raw_val) * (np.exp(np.clip(np.abs(raw_val), 0, 100)) - 1)
            print(f"      {action_name:6}: Raw={raw_val:7.3f}, Real={real_val:8.2f} chips")
    
    print(f"\n{'='*80}")
    print("LOSS COMPUTATION EXPLANATION")
    print(f"{'='*80}")
    print("""
During training:
1. Traverse game tree (traverse_external_sampling)
2. For each state, compute:
   - action_values[i] = traverse(next_state after action i)
   - node_value = sum(strategy[i] * action_values[i])
   - advantages[i] = action_values[i] - node_value  (in CHIPS)
3. Transform to Symlog:
   - target[i] = sign(advantages[i]) * log(1 + |advantages[i]|)
4. Network predicts: pred[i] (already in Symlog space)
5. Loss = MSE(pred, target) = mean((pred[i] - target[i])^2)

Why loss decreases but performance is bad?
- Loss measures how well network predicts Symlog(advantages)
- But if advantages themselves are wrong (bad strategy), loss can still decrease
- Example: If true advantage is +100 chips, but we compute it as +10 chips,
  the network learns to predict symlog(10) = 2.4, loss is low, but strategy is wrong!
    """)
    
    # Show example
    print("\nExample:")
    true_advantage = 100  # chips
    wrong_advantage = 10  # chips (computed incorrectly)
    
    true_target = np.sign(true_advantage) * np.log1p(np.abs(true_advantage))
    wrong_target = np.sign(wrong_advantage) * np.log1p(np.abs(wrong_advantage))
    
    print(f"  True advantage: {true_advantage} chips -> Target: {true_target:.4f}")
    print(f"  Wrong advantage: {wrong_advantage} chips -> Target: {wrong_target:.4f}")
    print(f"  If network predicts {wrong_target:.4f}, loss is low, but strategy is wrong!")

@app.local_entrypoint()
def main(iteration: int = 15, num_games: int = 3, analyze_loss: bool = False):
    if analyze_loss:
        analyze_loss_computation.remote(iteration)
    else:
        analyze_gameplay.remote(iteration, num_games)
        analyze_loss_computation.remote(iteration)

