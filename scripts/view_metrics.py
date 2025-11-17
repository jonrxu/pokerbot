#!/usr/bin/env python3
"""View training metrics from local download."""

import json
import sys
from pathlib import Path

def load_metrics(metrics_dir: str = "./local_metrics"):
    """Load and display metrics."""
    metrics_file = Path(metrics_dir) / "training_metrics.jsonl"
    summary_file = Path(metrics_dir) / "summary.json"
    
    if not summary_file.exists():
        print("No metrics found. Run: ./scripts/check_status.sh")
        return
    
    # Load summary
    with open(summary_file) as f:
        summary = json.load(f)
    
    print("="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Latest Iteration: {summary.get('latest_iteration', 'N/A')}")
    print(f"Total Iterations: {summary.get('total_iterations', 0)}")
    print(f"Last Updated: {summary.get('last_updated', 'N/A')}")
    print()
    
    if 'latest_metrics' in summary:
        lm = summary['latest_metrics']
        print("Latest Metrics:")
        print(f"  Value Loss: {lm.get('value_loss', 0):.6f}")
        print(f"  Policy Loss: {lm.get('policy_loss', 0):.6f}")
        print(f"  Trajectories: {lm.get('trajectories_generated', 0)}")
        print()
    
    print("Best Metrics:")
    print(f"  Best Value Loss: {summary.get('best_value_loss', float('inf')):.6f}")
    print(f"  Best Policy Loss: {summary.get('best_policy_loss', float('inf')):.6f}")
    print()
    
    # Load recent metrics
    if metrics_file.exists():
        print("Recent Iterations (last 10):")
        print("-"*60)
        with open(metrics_file) as f:
            lines = f.readlines()
            for line in lines[-10:]:
                m = json.loads(line)
                print(f"Iter {m['iteration']:4d} | "
                      f"Value: {m.get('value_loss', 0):8.4f} | "
                      f"Policy: {m.get('policy_loss', 0):8.4f} | "
                      f"Traj: {m.get('trajectories_generated', 0):5d}")

if __name__ == "__main__":
    metrics_dir = sys.argv[1] if len(sys.argv) > 1 else "./local_metrics"
    load_metrics(metrics_dir)

