import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def plot_metrics(jsonl_file):
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    
    # Filter to current run (last 50 iterations)
    # Assuming data is appended, we take the last entries corresponding to the current run
    # The current run started from iteration 0 (bootstrap).
    # We look for a sequence of 1, 2, 3... in 'iteration'
    
    current_run = []
    last_iter = -1
    for entry in reversed(data):
        it = entry.get('iteration', 0)
        if it == 1 and last_iter != -1:
             # Found start of run
             current_run.append(entry)
             break
        current_run.append(entry)
        last_iter = it
        
    current_run.reverse()
    
    # If we couldn't find a clear start, just take all data or last 50
    if not current_run:
        current_run = data[-50:]
        
    # Extract metrics
    iterations = []
    policy_losses = []
    advantage_losses = []
    
    for entry in current_run:
        iterations.append(entry.get('iteration', 0))
        policy_losses.append(entry.get('policy_loss', 0))
        advantage_losses.append(entry.get('advantage_loss', 0))
        
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Policy Loss
    ax1.plot(iterations, policy_losses, 'b-o', label='Policy Loss')
    ax1.set_title('Policy Loss (Phase 3)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Advantage Loss
    ax2.plot(iterations, advantage_losses, 'g-o', label='Advantage Loss')
    ax2.set_title('Advantage Loss (Phase 3)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_yscale('log')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('metrics/training_metrics_phase3.png')
    print("Graph saved to metrics/training_metrics_phase3.png")

if __name__ == "__main__":
    plot_metrics('metrics/training_metrics_phase3.jsonl')

