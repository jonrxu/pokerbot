#!/usr/bin/env python3
"""Plot training metrics from Modal volume."""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from modal_deploy.check_results import app, check_training_results


def plot_metrics(output_file=None, show_plot=False):
    """Fetch metrics from Modal and create visualization."""
    
    # Default to metrics/ folder with timestamp
    if output_file is None:
        metrics_dir = Path('metrics')
        metrics_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = metrics_dir / f'training_metrics_{timestamp}.png'
    else:
        output_file = Path(output_file)
        # If no directory specified, save to metrics/ folder
        if output_file.parent == Path('.'):
            metrics_dir = Path('metrics')
            metrics_dir.mkdir(exist_ok=True)
            output_file = metrics_dir / output_file.name
    
    with app.run():
        result = check_training_results.remote()
        
        if not result.get('all_entries'):
            print('No metrics found. Is training running?')
            return
        
        metrics = result['all_entries']
        
        # Filter to current training run (2000 trajectories per iteration)
        # Accept any iterations with trajectories >= 1000 (to include our 2000 traj/iter runs)
        current_run = [m for m in metrics if m.get('trajectories_generated', 0) >= 1000]
        
        if len(current_run) == 0:
            print('No training iterations found yet')
            return
        
        # Use only the latest entry per iteration to avoid duplicates
        latest_per_iter = {}
        for m in current_run:
            it = m.get('iteration', 0)
            if it not in latest_per_iter or m.get('timestamp', '') > latest_per_iter[it].get('timestamp', ''):
                latest_per_iter[it] = m
        
        # Extract metrics from unique iterations
        sorted_iters = sorted(latest_per_iter.keys())
        
        # Filter to only show latest run (iterations 1-50, since metrics are 1-indexed)
        # Metrics are stored with iteration + 1, so iteration 0 in code = iteration 1 in metrics
        # Find the start of the latest run by looking for iteration 1
        if 1 in sorted_iters:
            # Latest run starts at iteration 1, show all iterations up to 50
            latest_run_iters = [it for it in sorted_iters if 1 <= it <= 50]
        else:
            # If no iteration 1, take the latest 50 iterations
            latest_run_iters = sorted_iters[-50:] if len(sorted_iters) >= 50 else sorted_iters
        
        # Filter out iterations with invalid metrics (0 or missing values)
        valid_iters = []
        valid_advantage_losses = []
        valid_policy_losses = []
        
        for it in latest_run_iters:
            advantage_loss = latest_per_iter[it].get('advantage_loss', latest_per_iter[it].get('value_loss', 0))
            policy_loss = latest_per_iter[it].get('policy_loss', 0)
            # Only include iterations with valid (non-zero) metrics
            if advantage_loss > 0 and policy_loss > 0:
                valid_iters.append(it)
                valid_advantage_losses.append(advantage_loss)
                valid_policy_losses.append(policy_loss)
        
        if len(valid_iters) == 0:
            print('No valid metrics found for latest run')
            return
        
        iterations = valid_iters
        advantage_losses = valid_advantage_losses
        policy_losses = valid_policy_losses
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Training Progress: Deep CFR Poker Bot (Latest Run)', fontsize=16, fontweight='bold')
        
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
        
        # Plot Advantage Loss
        ax2.plot(iterations, advantage_losses, 'g-o', linewidth=2.5, markersize=5, 
                label='Advantage Loss', alpha=0.8)
        ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Advantage Loss', fontsize=12, color='g', fontweight='bold')
        ax2.set_title('Advantage Loss Over Time (Regret Prediction)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Use log scale for advantage loss if range is large
        if max(advantage_losses) / min([v for v in advantage_losses if v > 0]) > 100:
            ax2.set_yscale('log')
            ax2.set_ylabel('Advantage Loss (log scale)', fontsize=12, color='g', fontweight='bold')
        
        # Add advantage loss info
        val_min = min(advantage_losses)
        val_max = max(advantage_losses)
        val_current = advantage_losses[-1]
        val_info = f'Current: {val_current:.2f}\nRange: {val_min:.2f} - {val_max:.2f}\n(Volatility is normal)'
        ax2.text(0.02, 0.98, val_info, 
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Add overall stats at bottom
        stats_text = f'Unique Iterations: {len(iterations)} | Range: {iterations[0]}-{iterations[-1]}'
        fig.text(0.5, 0.01, stats_text, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05)
        
        # Check for evaluation metrics
        eval_random_wr = []
        eval_call_wr = []
        eval_iterations = []
        
        for it in latest_run_iters:
            m = latest_per_iter[it]
            if 'eval_random_win_rate' in m:
                eval_iterations.append(it)
                eval_random_wr.append(m.get('eval_random_win_rate', 0))
                eval_call_wr.append(m.get('eval_always_call_win_rate', 0))
        
        # Add evaluation subplot if available
        if len(eval_iterations) > 0:
            ax3 = fig.add_subplot(3, 1, 3)
            ax3.plot(eval_iterations, [w * 100 for w in eval_random_wr], 'r-o', 
                    linewidth=2.5, markersize=5, label='vs Random', alpha=0.8)
            ax3.plot(eval_iterations, [w * 100 for w in eval_call_wr], 'm-s', 
                    linewidth=2.5, markersize=5, label='vs Always Call', alpha=0.8)
            ax3.axhline(y=50, color='k', linestyle='--', alpha=0.3, label='Break-even')
            ax3.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
            ax3.set_title('Evaluation: Win Rate vs Baselines', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, linestyle='--')
            ax3.legend(loc='best', fontsize=10)
            ax3.set_ylim([0, 100])
            
            if len(eval_random_wr) > 0:
                eval_info = f'Latest vs Random: {eval_random_wr[-1]*100:.1f}%\nLatest vs Always Call: {eval_call_wr[-1]*100:.1f}%'
                ax3.text(0.02, 0.98, eval_info, 
                        transform=ax3.transAxes, fontsize=10,
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
            
            fig.suptitle('Training Progress: Deep CFR Poker Bot (Replay Buffer + Feature Encoder)', 
                        fontsize=16, fontweight='bold')
            plt.subplots_adjust(hspace=0.4, bottom=0.05)
        else:
            fig.suptitle('Training Progress: Deep CFR Poker Bot (Replay Buffer + Feature Encoder)', 
                        fontsize=16, fontweight='bold')
            plt.subplots_adjust(bottom=0.05)
        
        # Save figure
        output_file_str = str(output_file)
        plt.savefig(output_file_str, dpi=150, bbox_inches='tight')
        print(f'✓ Graph saved to: {output_file_str}')
        print(f'  Policy Loss: {policy_losses[0]:.4f} → {policy_losses[-1]:.4f} ({improvement:.1f}% change)')
        print(f'  Advantage Loss: {advantage_losses[0]:.4f} → {advantage_losses[-1]:.4f}')
        print(f'  Unique iterations plotted: {len(iterations)}')
        
        if len(eval_iterations) > 0:
            print(f'\n  Evaluation Results:')
            print(f'    vs Random: {eval_random_wr[-1]*100:.1f}% win rate (iter {eval_iterations[-1]})')
            print(f'    vs Always Call: {eval_call_wr[-1]*100:.1f}% win rate (iter {eval_iterations[-1]})')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return output_file_str


if __name__ == '__main__':
    output_file = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else None
    show = '--show' in sys.argv
    result_file = plot_metrics(output_file, show_plot=show)
    if result_file:
        print(f"\n✓ Metrics graph saved to: {result_file}")

