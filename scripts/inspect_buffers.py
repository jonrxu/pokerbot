#!/usr/bin/env python3
"""Inspect training data buffers to diagnose loss trends."""

import modal
import torch
import numpy as np
from collections import defaultdict
import sys

# Use shared configuration
from modal_deploy.config import image, checkpoint_volume

app = modal.App("buffer-inspector")

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=600
)
def inspect_buffers():
    import os
    import pickle
    
    results = []
    def log(msg):
        print(msg)
        results.append(msg)
    
    log("\n[1] INSPECTING REPLAY BUFFERS")
    
    # Load replay buffers if available
    value_buffer = []
    policy_buffer = []
    
    try:
        with open("/checkpoints/value_replay.pkl", "rb") as f:
            try:
                value_buffer = pickle.load(f)
            except EOFError:
                log("⚠ Value buffer file is empty or corrupted (EOFError).")
    except FileNotFoundError:
        log("Value buffer not found.")
        
    try:
        with open("/checkpoints/policy_replay.pkl", "rb") as f:
            try:
                policy_buffer = pickle.load(f)
            except EOFError:
                log("⚠ Policy buffer file is empty or corrupted (EOFError).")
    except FileNotFoundError:
        log("Policy buffer not found.")
            
    log(f"Value Buffer Size: {len(value_buffer)}")
    log(f"Policy Buffer Size: {len(policy_buffer)}")
    
    if not value_buffer and not policy_buffer:
        return "\n".join(results)
    
    try:
        if value_buffer:
            # --- Analyze Value Targets ---
            log("\n[2] VALUE TARGET ANALYSIS")
            # Buffer is a deque, convert to list for slicing
            value_list = list(value_buffer)
            
            # Buffer format: (state_encoding, target_value)
            targets = [v[1] for v in value_list[-5000:]] # Look at last 5000 samples
            targets = np.array(targets)
            
            log(f"Sample Targets (last 10): {targets[-10:]}")
            log(f"Statistics:")
            log(f"  Mean: {np.mean(targets):.4f}")
            log(f"  Std Dev: {np.std(targets):.4f}")
            log(f"  Min: {np.min(targets):.4f}")
            log(f"  Max: {np.max(targets):.4f}")
            log(f"  % Zero/Near-Zero (<0.01): {np.mean(np.abs(targets) < 0.01) * 100:.1f}%")
            
            # Check if targets are distinct enough
            unique_vals = len(np.unique(targets))
            log(f"  Unique Values: {unique_vals}")
            
            if np.std(targets) < 0.05:
                log("⚠ WARNING: Value targets are extremely clustered! Network is learning to predict a constant.")
            else:
                log("✓ PASS: Value targets show variance.")

        if policy_buffer:
            # --- Analyze Policy Targets ---
            log("\n[3] POLICY TARGET ANALYSIS")
            policy_list = list(policy_buffer)
            # Buffer format: (state_encoding, action_probs)
            # action_probs is a numpy array or list
            policy_targets = [p[1] for p in policy_list[-1000:]] # Last 1000 samples
            
            # Check for oscillation/uniformity in targets
            uniform_targets = 0
            deterministic_targets = 0 # One action has prob > 0.9
            
            for prob_dist in policy_targets:
                probs = np.array(prob_dist)
                if np.max(probs) > 0.9:
                    deterministic_targets += 1
                
                # Uniformity check
                if len(probs) > 0:
                    expected = 1.0 / len(probs)
                    if np.allclose(probs, expected, atol=0.05):
                        uniform_targets += 1
            
            log(f"Analyzed {len(policy_targets)} policy targets:")
            log(f"  Uniform Targets: {uniform_targets} ({uniform_targets/len(policy_targets)*100:.1f}%)")
            log(f"  Deterministic Targets (>0.9): {deterministic_targets} ({deterministic_targets/len(policy_targets)*100:.1f}%)")
            log(f"  Mixed Targets: {len(policy_targets) - uniform_targets - deterministic_targets}")
            
            if uniform_targets > len(policy_targets) * 0.5:
                 log("⚠ WARNING: Majority of policy targets are uniform. Average strategy might be failing to accumulate correctly.")

    except Exception as e:
        log(f"Error inspecting buffers: {e}")
        import traceback
        log(traceback.format_exc())
    
    return "\n".join(results)

if __name__ == "__main__":
    with app.run():
        print(inspect_buffers.remote())
