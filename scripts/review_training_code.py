#!/usr/bin/env python3
"""Review training code for correctness and best practices."""

import sys
import os
from pathlib import Path
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def review_training_code():
    """Review training code for correctness."""
    
    print("="*80)
    print("TRAINING CODE REVIEW")
    print("="*80)
    print()
    
    # Read training file
    train_file = project_root / "modal_deploy" / "train.py"
    with open(train_file, 'r') as f:
        train_code = f.read()
    
    # Read config file
    config_file = project_root / "modal_deploy" / "config.py"
    with open(config_file, 'r') as f:
        config_code = f.read()
    
    issues = []
    recommendations = []
    
    # 1. Check trajectory generation
    print("1. TRAJECTORY GENERATION")
    print("-" * 80)
    
    # Check default trajectories_per_iteration
    default_traj_match = re.search(r'trajectories_per_iteration:\s*int\s*=\s*(\d+)', train_code)
    if default_traj_match:
        default_traj = int(default_traj_match.group(1))
        print(f"   Default trajectories per iteration: {default_traj}")
        if default_traj < 2000:
            issues.append(f"Default trajectories ({default_traj}) is less than recommended 2000")
        elif default_traj > 10000:
            recommendations.append(f"Consider reducing trajectories ({default_traj}) if training is slow")
    else:
        issues.append("Could not find default trajectories_per_iteration")
    
    # Check trajectory validation
    if 'validate_trajectory' in train_code:
        print("   ✓ Trajectory validation function exists")
    else:
        issues.append("Missing trajectory validation function")
    
    # Check error handling for invalid trajectories
    if 'invalid_count' in train_code and 'valid_trajectories' in train_code:
        print("   ✓ Invalid trajectory handling exists")
    else:
        issues.append("Missing invalid trajectory handling")
    
    print()
    
    # 2. Check replay buffer
    print("2. REPLAY BUFFER")
    print("-" * 80)
    
    # Check ReplayBuffer class
    if 'class ReplayBuffer' in train_code:
        print("   ✓ ReplayBuffer class exists")
        
        # Check maxlen
        maxlen_match = re.search(r'maxlen\s*=\s*(\d+)', train_code)
        if maxlen_match:
            maxlen = int(maxlen_match.group(1))
            print(f"   Replay buffer maxlen: {maxlen}")
            if maxlen < 100000:
                issues.append(f"Replay buffer maxlen ({maxlen}) may be too small")
            elif maxlen > 500000:
                recommendations.append(f"Replay buffer maxlen ({maxlen}) is large, ensure memory is sufficient")
        else:
            issues.append("Could not find replay buffer maxlen")
        
        # Check save/load
        if 'def save' in train_code and 'def load' in train_code:
            print("   ✓ Replay buffer save/load methods exist")
        else:
            issues.append("Missing replay buffer save/load methods")
    else:
        issues.append("Missing ReplayBuffer class")
    
    # Check replay buffer usage
    if 'value_replay' in train_code and 'policy_replay' in train_code:
        print("   ✓ Replay buffers are used for value and policy")
    else:
        issues.append("Replay buffers not used in training")
    
    # Check replay buffer loading
    if 'value_replay.load' in train_code and 'policy_replay.load' in train_code:
        print("   ✓ Replay buffers are loaded from checkpoints")
    else:
        issues.append("Replay buffers not loaded from checkpoints")
    
    print()
    
    # 3. Check learning rates
    print("3. LEARNING RATES")
    print("-" * 80)
    
    # Check learning rate values
    lr_matches = re.findall(r'learning_rate\s*=\s*([\d.e-]+)', train_code)
    value_lr_matches = re.findall(r'value_learning_rate\s*=\s*([\d.e-]+)', train_code)
    
    if lr_matches:
        lr = float(lr_matches[0])
        print(f"   Policy learning rate: {lr}")
        if lr > 1e-3:
            issues.append(f"Policy learning rate ({lr}) may be too high")
        elif lr < 1e-6:
            issues.append(f"Policy learning rate ({lr}) may be too low")
        else:
            print("   ✓ Policy learning rate is reasonable")
    
    if value_lr_matches:
        value_lr = float(value_lr_matches[0])
        print(f"   Value learning rate: {value_lr}")
        if value_lr > 1e-3:
            issues.append(f"Value learning rate ({value_lr}) may be too high")
        elif value_lr < 1e-6:
            issues.append(f"Value learning rate ({value_lr}) may be too low")
        else:
            print("   ✓ Value learning rate is reasonable")
    
    print()
    
    # 4. Check batch size
    print("4. BATCH SIZE")
    print("-" * 80)
    
    # Check default batch size
    batch_size_match = re.search(r'batch_size:\s*int\s*=\s*(\d+)', train_code)
    if batch_size_match:
        batch_size = int(batch_size_match.group(1))
        print(f"   Default batch size: {batch_size}")
        if batch_size < 16:
            issues.append(f"Batch size ({batch_size}) may be too small")
        elif batch_size > 128:
            recommendations.append(f"Batch size ({batch_size}) is large, ensure GPU memory is sufficient")
        else:
            print("   ✓ Batch size is reasonable")
    else:
        issues.append("Could not find default batch size")
    
    # Check config batch size
    config_batch_match = re.search(r'"batch_size":\s*(\d+)', config_code)
    if config_batch_match:
        config_batch = int(config_batch_match.group(1))
        print(f"   Config batch size: {config_batch}")
    
    print()
    
    # 5. Check value scaling
    print("5. VALUE SCALING")
    print("-" * 80)
    
    if 'CFV_SCALE' in train_code:
        scale_match = re.search(r'CFV_SCALE\s*=\s*([\d.]+)', train_code)
        if scale_match:
            scale = float(scale_match.group(1))
            print(f"   CFV_SCALE: {scale}")
            if scale < 1000:
                issues.append(f"CFV_SCALE ({scale}) may be too small")
            elif scale > 100000:
                issues.append(f"CFV_SCALE ({scale}) may be too large")
            else:
                print("   ✓ CFV_SCALE is reasonable")
        else:
            issues.append("CFV_SCALE defined but value not found")
    else:
        issues.append("Missing CFV_SCALE - value targets may be unstable")
    
    # Check if scaling is applied
    if 'CFV_SCALE' in train_code and ('terminal_cf_value' in train_code or 'batch_values_clipped' in train_code):
        print("   ✓ Value scaling appears to be applied")
    else:
        issues.append("Value scaling may not be applied correctly")
    
    print()
    
    # 6. Check number of updates
    print("6. TRAINING UPDATES")
    print("-" * 80)
    
    # Check num_updates
    num_updates_match = re.search(r'num_updates\s*=\s*(\d+)', train_code)
    if num_updates_match:
        num_updates = int(num_updates_match.group(1))
        print(f"   Number of updates per iteration: {num_updates}")
        if num_updates < 100:
            issues.append(f"Number of updates ({num_updates}) may be too low")
        elif num_updates > 5000:
            recommendations.append(f"Number of updates ({num_updates}) is high, training may be slow")
        else:
            print("   ✓ Number of updates is reasonable")
    else:
        issues.append("Could not find num_updates")
    
    print()
    
    # 7. Check gradient clipping
    print("7. GRADIENT CLIPPING")
    print("-" * 80)
    
    if 'grad_norm' in train_code or 'clip_grad_norm' in train_code or 'torch.nn.utils.clip_grad_norm' in train_code:
        print("   ✓ Gradient clipping appears to be implemented")
    else:
        issues.append("Missing gradient clipping - may have exploding gradients")
    
    print()
    
    # 8. Check mixed precision training
    print("8. MIXED PRECISION TRAINING")
    print("-" * 80)
    
    if 'autocast' in train_code or 'GradScaler' in train_code or 'use_amp' in train_code:
        print("   ✓ Mixed precision training (AMP) appears to be implemented")
    else:
        recommendations.append("Consider adding mixed precision training for faster GPU training")
    
    print()
    
    # 9. Check checkpoint saving
    print("9. CHECKPOINT SAVING")
    print("-" * 80)
    
    if 'checkpoint_path' in train_code and 'torch.save' in train_code:
        print("   ✓ Checkpoint saving appears to be implemented")
        
        # Check if regret_memory and strategy_memory are saved
        if 'regret_memory' in train_code and 'strategy_memory' in train_code:
            print("   ✓ Regret and strategy memory appear to be saved")
        else:
            issues.append("Regret/strategy memory may not be saved in checkpoints")
    else:
        issues.append("Checkpoint saving may not be implemented")
    
    print()
    
    # 10. Check CFR math
    print("10. CFR MATH VERIFICATION")
    print("-" * 80)
    
    # Check if regret matching is used
    if 'regret_matching' in train_code or 'regret_memory' in train_code:
        print("   ✓ Regret matching appears to be implemented")
    else:
        issues.append("Regret matching may not be implemented")
    
    # Check if average strategy is computed
    if 'compute_average_strategy' in train_code or 'strategy_memory' in train_code:
        print("   ✓ Average strategy computation appears to be implemented")
    else:
        issues.append("Average strategy computation may not be implemented")
    
    print()
    
    # SUMMARY
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    if issues:
        print(f"\n⚠ ISSUES FOUND ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("\n✓ No critical issues found!")
    
    if recommendations:
        print(f"\n💡 RECOMMENDATIONS ({len(recommendations)}):")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print("\n✓ No recommendations")
    
    print()
    print("="*80)
    print("TRAINING HYPERPARAMETERS SUMMARY")
    print("="*80)
    
    # Extract key hyperparameters
    if default_traj_match:
        print(f"Trajectories per iteration: {default_traj_match.group(1)}")
    if batch_size_match:
        print(f"Batch size: {batch_size_match.group(1)}")
    if lr_matches:
        print(f"Policy learning rate: {lr_matches[0]}")
    if value_lr_matches:
        print(f"Value learning rate: {value_lr_matches[0]}")
    if num_updates_match:
        print(f"Updates per iteration: {num_updates_match.group(1)}")
    if maxlen_match:
        print(f"Replay buffer maxlen: {maxlen_match.group(1)}")
    if scale_match:
        print(f"CFV_SCALE: {scale_match.group(1)}")
    
    print()
    
    return len(issues) == 0


if __name__ == "__main__":
    review_training_code()

