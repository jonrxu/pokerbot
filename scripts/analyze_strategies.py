#!/usr/bin/env python3
"""Detailed strategy analysis to verify bug fix."""

import modal

app = modal.App("strategy-analysis")
checkpoint_volume = modal.Volume.from_name("poker-bot-checkpoints", create_if_missing=False)

@app.function(
    image=modal.Image.debian_slim(python_version="3.10").pip_install("torch", "numpy"),
    volumes={"/checkpoints": checkpoint_volume},
    timeout=300
)
def analyze_strategies():
    """Analyze strategies to check if bug is fixed."""
    import torch
    import os
    import numpy as np
    import sys
    
    output_lines = []
    
    def log(msg):
        print(msg)
        output_lines.append(msg)
        sys.stdout.flush()
    
    # Find latest checkpoint
    checkpoints = sorted([f for f in os.listdir("/checkpoints") if f.startswith("checkpoint_iter_")])
    if not checkpoints:
        log("No checkpoints found!")
        return "\n".join(output_lines)
        
    latest = checkpoints[-1]
    log("="*80)
    log(f"ANALYZING CHECKPOINT: {latest}")
    log("="*80)
    
    try:
        checkpoint = torch.load(f"/checkpoints/{latest}", map_location='cpu', weights_only=False)
    except Exception as e:
        log(f"Failed to load checkpoint: {e}")
        return "\n".join(output_lines)
    
    # Analyze Strategy Memory
    strat_mem = checkpoint.get('strategy_memory', {})
    log(f"\nTotal info sets with strategies: {len(strat_mem)}")
    
    if not strat_mem:
        log("No strategies found!")
        return "\n".join(output_lines)
    
    # Analyze each strategy
    uniform_count = 0
    non_uniform_count = 0
    highly_skewed_count = 0  # One action > 70%
    sample_non_uniform = []
    sample_uniform = []
    
    for key, strategy in list(strat_mem.items())[:1000]:  # Sample first 1000
        if not isinstance(strategy, dict):
            continue
            
        values = list(strategy.values())
        if not values:
            continue
            
        total = sum(values)
        if total == 0:
            continue
            
        # Normalize
        probs = [v / total for v in values]
        num_actions = len(probs)
        
        # Check if uniform (within 5% tolerance)
        expected_prob = 1.0 / num_actions
        is_uniform = all(abs(p - expected_prob) < 0.05 for p in probs)
        
        # Check if highly skewed
        max_prob = max(probs)
        is_skewed = max_prob > 0.70
        
        if is_uniform:
            uniform_count += 1
            if len(sample_uniform) < 3:
                sample_uniform.append((key[:50], probs))
        else:
            non_uniform_count += 1
            if len(sample_non_uniform) < 5:
                sample_non_uniform.append((key[:50], probs))
            if is_skewed:
                highly_skewed_count += 1
    
    log("\n" + "="*80)
    log("STRATEGY ANALYSIS RESULTS")
    log("="*80)
    log(f"\nUniform strategies (random play): {uniform_count}")
    log(f"Non-uniform strategies (learning!): {non_uniform_count}")
    log(f"Highly skewed strategies (>70% on one action): {highly_skewed_count}")
    
    if uniform_count + non_uniform_count > 0:
        pct_uniform = (uniform_count / (uniform_count + non_uniform_count)) * 100
        pct_non_uniform = (non_uniform_count / (uniform_count + non_uniform_count)) * 100
        log(f"\nPercentage uniform: {pct_uniform:.1f}%")
        log(f"Percentage non-uniform: {pct_non_uniform:.1f}%")
    
    log("\n" + "-"*80)
    log("Sample UNIFORM Strategies (BAD - indicates bug):")
    log("-"*80)
    for key, probs in sample_uniform[:3]:
        log(f"  {key}...")
        for i, p in enumerate(probs):
            log(f"    Action {i}: {p*100:.1f}%")
    
    log("\n" + "-"*80)
    log("Sample NON-UNIFORM Strategies (GOOD - indicates learning!):")
    log("-"*80)
    for key, probs in sample_non_uniform[:5]:
        log(f"  {key}...")
        sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        for i, p in sorted_probs:
            log(f"    Action {i}: {p*100:.1f}%")
    
    # Check regrets
    reg_mem = checkpoint.get('regret_memory', {})
    log("\n" + "="*80)
    log("REGRET ANALYSIS")
    log("="*80)
    log(f"Total info sets with regrets: {len(reg_mem)}")
    
    if reg_mem:
        all_regrets = []
        for regrets_dict in list(reg_mem.values())[:1000]:
            if isinstance(regrets_dict, dict):
                all_regrets.extend(regrets_dict.values())
        
        if all_regrets:
            r_arr = np.array(all_regrets)
            log(f"\nRegret Statistics (sample of {len(all_regrets)}):")
            log(f"  Mean: {r_arr.mean():.4f}")
            log(f"  Std: {r_arr.std():.4f}")
            log(f"  Min: {r_arr.min():.4f}")
            log(f"  Max: {r_arr.max():.4f}")
            log(f"  Non-zero regrets: {np.sum(np.abs(r_arr) > 0.01)}/{len(all_regrets)}")
    
    log("\n" + "="*80)
    log("VERDICT")
    log("="*80)
    if non_uniform_count > uniform_count:
        log("✓ BUG APPEARS FIXED! Non-uniform strategies dominate.")
        log("  The bot is learning actual strategies, not just random play.")
    elif non_uniform_count > 0:
        log("⚠ PARTIALLY FIXED: Some non-uniform strategies found.")
        log("  This is early in training - more iterations needed.")
    else:
        log("✗ BUG STILL PRESENT: All strategies are uniform.")
        log("  The bot is still playing randomly.")
    
    return "\n".join(output_lines)

if __name__ == "__main__":
    import sys
    print("Starting strategy analysis...", file=sys.stderr)
    with app.run():
        result = analyze_strategies.remote()
        print("\n" + "="*80)
        print("STRATEGY ANALYSIS RESULTS:")
        print("="*80)
        print(result)
        print("="*80)


