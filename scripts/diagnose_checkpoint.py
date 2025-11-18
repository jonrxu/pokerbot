#!/usr/bin/env python3
"""Comprehensive diagnostic script to inspect checkpoint weights, values, and logic."""

import modal
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modal_deploy.config import image, checkpoint_volume

app = modal.App("checkpoint-diagnostic")

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=600,
    memory=8192
)
def run_diagnostic(checkpoint_filename: str = None):
    """Run a suite of rigorous checks on a checkpoint."""
    import os
    import torch
    import numpy as np
    from poker_game.game import PokerGame, GameState, Action
    from poker_game.state_encoder import StateEncoder
    from models.value_policy_net import ValuePolicyNet
    
    results = []
    def log(msg):
        print(msg)
        results.append(msg)
    
    # 1. Load Checkpoint
    if checkpoint_filename:
        latest = checkpoint_filename
    else:
        checkpoints = sorted([f for f in os.listdir("/checkpoints") if f.startswith("checkpoint_iter_")])
        if not checkpoints:
            return "No checkpoints found."
        latest = checkpoints[-1]
    
    log("="*80)
    log(f"ANALYZING CHECKPOINT: {latest}")
    log("="*80)
    
    try:
        checkpoint_path = f"/checkpoints/{latest}"
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        return f"Failed to load checkpoint: {e}"
    
    # 2. Initialize Game & Networks
    game = PokerGame()
    encoder = StateEncoder()
    input_dim = encoder.feature_dim
    
    value_net = ValuePolicyNet(input_dim)
    policy_net = ValuePolicyNet(input_dim)
    
    try:
        value_net.load_state_dict(checkpoint['value_net_state'])
        policy_net.load_state_dict(checkpoint['policy_net_state'])
    except RuntimeError as e:
        log(f"⚠ Error loading state dict (likely architecture mismatch): {e}")
        # Continue with uninitialized nets for other checks if possible, or return
        return "\n".join(results)

    value_net.eval()
    policy_net.eval()
    
    # [A] NETWORK WEIGHT ANALYSIS
    log("\n[1] NETWORK WEIGHT ANALYSIS")
    for name, net in [("Value", value_net), ("Policy", policy_net)]:
        total_params = 0
        dead_params = 0
        max_weight = 0.0
        for param in net.parameters():
            data = param.data
            total_params += data.numel()
            dead_params += torch.sum(data == 0).item()
            max_weight = max(max_weight, torch.max(torch.abs(data)).item())
            
        log(f"  {name} Net: Max Weight={max_weight:.4f}, Dead Params={dead_params}/{total_params} ({dead_params/total_params*100:.1f}%)")
        if max_weight > 100:
            log(f"  ⚠ WARNING: {name} net has very large weights! Potential explosion.")
    
    # [B] VALUE NETWORK CONSISTENCY CHECK
    log("\n[2] VALUE NETWORK CONSISTENCY")
    # Case A: Pocket Aces (Pre-flop)
    state_aces = game.reset()
    state_aces.hole_cards[0] = [(12, 0), (12, 1)] # AA
    state_aces.current_player = 0
    
    # Case B: 7-2 Offsuit (Pre-flop)
    state_trash = game.reset()
    state_trash.hole_cards[0] = [(5, 0), (0, 1)] # 7-2
    state_trash.current_player = 0
    
    enc_aces = torch.tensor(encoder.encode(state_aces, 0), dtype=torch.float32).unsqueeze(0)
    enc_trash = torch.tensor(encoder.encode(state_trash, 0), dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        val_aces, _ = value_net(enc_aces)
        val_trash, _ = value_net(enc_trash)
        
    log(f"  Value(AA): {val_aces.item():.4f}")
    log(f"  Value(72o): {val_trash.item():.4f}")
    
    if val_aces.item() > val_trash.item():
        log("  ✓ PASS: AA valued higher than 72o")
    else:
        log("  ✗ FAIL: Value network thinks 72o >= AA (or roughly equal)")

    # [C] POLICY NETWORK ENTROPY CHECK
    log("\n[3] POLICY NETWORK ENTROPY")
    state_neutral = game.reset()
    enc_neutral = torch.tensor(encoder.encode(state_neutral, 0), dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        _, logits = policy_net(enc_neutral)
        probs = torch.softmax(logits, dim=1)[0].numpy()
        
    log(f"  Probabilities (Pre-flop neutral): {probs}")
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    log(f"  Entropy: {entropy:.4f}")
    
    if entropy < 0.1:
        log("  ⚠ WARNING: Low entropy. Policy might have collapsed to a single action.")
    elif entropy > 1.5: 
        log("  ✓ PASS: High entropy (exploring/mixed strategy).")
    else:
        log("  ✓ PASS: Moderate entropy.")

    # [D] INTERNAL MEMORY INSPECTION
    log("\n[4] INTERNAL MEMORY STATISTICS")
    
    # Regret Memory
    reg_mem = checkpoint.get('regret_memory', {})
    log(f"  Regret Memory Items: {len(reg_mem)}")
    if reg_mem:
        all_regrets = []
        for v in reg_mem.values():
            if isinstance(v, dict): all_regrets.extend(v.values())
            else: all_regrets.append(v)
        if all_regrets:
            r_arr = np.array(all_regrets)
            log(f"    Regret Stats: Mean={r_arr.mean():.4f}, Std={r_arr.std():.4f}, Max={r_arr.max():.4f}")

    # Strategy Memory
    strat_mem = checkpoint.get('strategy_memory', {})
    log(f"  Strategy Memory Items: {len(strat_mem)}")
    
    # CF Values
    val_mem = checkpoint.get('counterfactual_values', {})
    log(f"  CF Values Stored: {len(val_mem)}")
    if val_mem:
        vals = list(val_mem.values())
        v_arr = np.array([v for v in vals if abs(v) > 1e-6])
        if len(v_arr) > 0:
            log(f"    CF Value Stats: Mean={v_arr.mean():.4f}, Std={v_arr.std():.4f}, Range=[{v_arr.min():.4f}, {v_arr.max():.4f}]")
        else:
            log("    ⚠ ALL CF VALUES ARE ZERO!")

    return "\n".join(results)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose checkpoint')
    parser.add_argument('--file', type=str, default=None, help='Specific checkpoint file (e.g. checkpoint_iter_50.pt)')
    args = parser.parse_args()
    
    print("Starting diagnostic...", file=sys.stderr)
    with app.run():
        print(run_diagnostic.remote(args.file))

