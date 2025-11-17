#!/usr/bin/env python3
"""Plot training metrics from Modal volume."""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from modal_deploy.check_results import app, check_training_results


def plot_metrics(output_file='training_metrics.png', show_plot=False):
    """Fetch metrics from Modal and create visualization."""
    
    with app.run():
        result = check_training_results.remote()
        
        if not result.get('all_entries'):
            print('No metrics found. Is training running?')
            return
        
        metrics = result['all_entries']
        
        # Filter to full-scale iterations (10k trajectories)
        full_scale = [m for m in metrics if m.get('trajectories_generated', 0) >= 10000]
        
        if len(full_scale) == 0:
            print('No full-scale iterations found yet')
            return
        
        iterations = [m.get('iteration', 0) for m in full_scale]
        value_losses = [m.get('value_loss', 0) for m in full_scale]
        policy_losses = [m.get('policy_loss', 0) for m in full_scale]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Training Progress: Deep CFR Poker Bot', fontsize=16, fontweight='bold')
        
        # Plot Policy Loss (more important metric)
        ax1.plot(iterations, policy_losses, 'b-o', linewidth=2.5, markersize=5, 
                label='Policy Loss', alpha=0.8)
        ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Policy Loss', fontsize=12, color='b', fontweight='bold')
        ax1.set_title('Policy Loss Over Time (Strategy Learning)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Add trend line for policy loss
        if len(iterations) > 1:
            z = np.polyfit(iterations, policy_losses, 1)
            p = np.poly1d(z)
            ax1.plot(iterations, p(iterations), 'r--', alpha=0.6, linewidth=2, 
                    label=f'Trend Line (slope: {z[0]:.6f})')
            ax1.legend(loc='upper right', fontsize=10)
        
        # Add improvement annotation
        if len(policy_losses) > 0:
            improvement = ((policy_losses[0] - policy_losses[-1]) / policy_losses[0] * 100) if policy_losses[0] > 0 else 0
            current = policy_losses[-1]
            best = min(policy_losses)
            
            info_text = f'Current: {current:.4f}\nBest: {best:.4f}\nImprovement: {improvement:.1f}%'
            ax1.text(0.02, 0.98, info_text, 
                    transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Plot Value Loss
        ax2.plot(iterations, value_losses, 'g-o', linewidth=2.5, markersize=5, 
                label='Value Loss', alpha=0.8)
        ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Value Loss', fontsize=12, color='g', fontweight='bold')
        ax2.set_title('Value Loss Over Time (Prediction Accuracy)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Use log scale for value loss if range is large
        if max(value_losses) / min([v for v in value_losses if v > 0]) > 100:
            ax2.set_yscale('log')
            ax2.set_ylabel('Value Loss (log scale)', fontsize=12, color='g', fontweight='bold')
        
        # Add value loss info
        val_min = min(value_losses)
        val_max = max(value_losses)
        val_current = value_losses[-1]
        val_info = f'Current: {val_current:.2f}\nRange: {val_min:.2f} - {val_max:.2f}\n(Volatility is normal)'
        ax2.text(0.02, 0.98, val_info, 
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Add overall stats at bottom
        stats_text = f'Total Iterations: {len(full_scale)} | Iterations {iterations[0]}-{iterations[-1]}'
        fig.text(0.5, 0.01, stats_text, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05)
        
        # Save figure
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f'✓ Graph saved to: {output_file}')
        print(f'  Policy Loss: {policy_losses[0]:.4f} → {policy_losses[-1]:.4f} ({improvement:.1f}% improvement)')
        print(f'  Value Loss: {value_losses[0]:.2f} → {value_losses[-1]:.2f}')
        print(f'  Total iterations plotted: {len(full_scale)}')
        
        if show_plot:
            plt.show()
        else:
            plt.close()


if __name__ == '__main__':
    output_file = sys.argv[1] if len(sys.argv) > 1 else 'training_metrics.png'
    show = '--show' in sys.argv
    plot_metrics(output_file, show_plot=show)

