#!/usr/bin/env python3
"""Plot comprehensive training curves including losses, payoff, and win rate."""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import subprocess


def plot_training_curves(
    metrics_file: str = None,
    output_file: str = 'training_curves.png',
    show_plot: bool = False
):
    """Plot training curves with losses, payoff, and win rate.

    Args:
        metrics_file: Path to metrics JSON (if None, downloads from Modal)
        output_file: Output image path
        show_plot: Whether to display plot
    """
    # Load metrics
    if metrics_file is None:
        print("Downloading metrics from Modal...")
        result = subprocess.run(
            ['modal', 'volume', 'ls', 'poker-exploitative-cfr-checkpoints'],
            capture_output=True,
            text=True
        )

        # Find metrics files
        all_files = result.stdout.split('\n')
        metrics_files = [
            line.split()[0] for line in all_files
            if 'metrics_' in line and '.json' in line
        ]

        if not metrics_files:
            print("No metrics found! Run training first.")
            return

        # Prefer final metrics
        final_metrics = [f for f in metrics_files if 'metrics_final.json' in f]
        if final_metrics:
            latest = final_metrics[0]
            print(f"Found final metrics: {latest}")
        else:
            iter_metrics = [f for f in metrics_files if 'metrics_iter_' in f]
            latest = sorted(iter_metrics)[-1]
            print(f"Found latest iteration metrics: {latest}")

        # Download
        subprocess.run([
            'modal', 'volume', 'get',
            '--force',
            'poker-exploitative-cfr-checkpoints',
            latest,
            './temp_metrics.json',
        ])
        metrics_file = './temp_metrics.json'

    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    if not metrics:
        print("No metrics found!")
        return

    # Extract data
    iterations = [m['iteration'] for m in metrics]
    avg_payoffs = [m['avg_payoff'] for m in metrics]
    win_rates = [m['win_rate'] * 100 for m in metrics]
    value_losses = [m.get('value_loss', 0) for m in metrics]
    policy_losses = [m.get('policy_loss', 0) for m in metrics]

    # Create figure with 4 subplots (2x2)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Exploitative Training: Full Analysis', fontsize=16, fontweight='bold')

    # 1. Average Payoff
    ax1.plot(iterations, avg_payoffs, 'b-o', linewidth=2, markersize=4, alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break Even')
    ax1.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Avg Payoff (chips/hand)', fontsize=11, fontweight='bold')
    ax1.set_title('Average Payoff vs GTO', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend()

    # Add rolling average
    if len(avg_payoffs) >= 10:
        window = min(10, len(avg_payoffs) // 5)
        rolling_avg = np.convolve(avg_payoffs, np.ones(window)/window, mode='valid')
        ax1.plot(iterations[window-1:], rolling_avg, 'g-', linewidth=2.5,
                label=f'{window}-iteration moving avg', alpha=0.8)
        ax1.legend()

    # 2. Win Rate
    ax2.plot(iterations, win_rates, 'g-o', linewidth=2, markersize=4, alpha=0.7)
    ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% (Break Even)')
    ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Win Rate vs GTO', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()

    # Add rolling average
    if len(win_rates) >= 10:
        window = min(10, len(win_rates) // 5)
        rolling_avg = np.convolve(win_rates, np.ones(window)/window, mode='valid')
        ax2.plot(iterations[window-1:], rolling_avg, 'orange', linewidth=2.5,
                label=f'{window}-iteration moving avg', alpha=0.8)
        ax2.legend()

    # 3. Value Loss
    if any(v > 0 for v in value_losses):
        ax3.plot(iterations, value_losses, 'purple', linewidth=2, markersize=3,
                marker='o', alpha=0.7)
        ax3.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Value Loss (MSE)', fontsize=11, fontweight='bold')
        ax3.set_title('Value Network Loss', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_yscale('log')  # Log scale for loss

        # Add trend line
        if len(iterations) > 1:
            z = np.polyfit(iterations, np.log(value_losses), 1)
            trend = np.exp(z[1] + z[0] * np.array(iterations))
            ax3.plot(iterations, trend, 'r--', linewidth=2, alpha=0.6,
                    label='Exponential trend')
            ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No value loss data', ha='center', va='center',
                transform=ax3.transAxes, fontsize=14)
        ax3.set_xlabel('Iteration', fontsize=11)
        ax3.set_ylabel('Value Loss', fontsize=11)
        ax3.set_title('Value Network Loss', fontsize=12, fontweight='bold')

    # 4. Policy Loss
    if any(p > 0 for p in policy_losses):
        ax4.plot(iterations, policy_losses, 'orange', linewidth=2, markersize=3,
                marker='o', alpha=0.7)
        ax4.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Policy Loss (KL Div)', fontsize=11, fontweight='bold')
        ax4.set_title('Policy Network Loss', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_yscale('log')  # Log scale for loss

        # Add trend line
        if len(iterations) > 1:
            z = np.polyfit(iterations, np.log(policy_losses), 1)
            trend = np.exp(z[1] + z[0] * np.array(iterations))
            ax4.plot(iterations, trend, 'r--', linewidth=2, alpha=0.6,
                    label='Exponential trend')
            ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No policy loss data', ha='center', va='center',
                transform=ax4.transAxes, fontsize=14)
        ax4.set_xlabel('Iteration', fontsize=11)
        ax4.set_ylabel('Policy Loss', fontsize=11)
        ax4.set_title('Policy Network Loss', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'✓ Graph saved to: {output_file}')

    # Calculate statistics
    overall_avg_payoff = np.mean(avg_payoffs)
    overall_win_rate = np.mean(win_rates)
    recent_n = min(10, len(avg_payoffs))
    recent_avg_payoff = np.mean(avg_payoffs[-recent_n:])
    recent_win_rate = np.mean(win_rates[-recent_n:])

    # Print summary
    print()
    print("=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total Iterations: {len(iterations)}")
    print(f"Overall Avg Payoff: {overall_avg_payoff:+.2f} chips/hand")
    print(f"Overall Win Rate: {overall_win_rate:.2f}%")
    print(f"Recent Avg Payoff (last {recent_n}): {recent_avg_payoff:+.2f} chips/hand")
    print(f"Recent Win Rate (last {recent_n}): {recent_win_rate:.2f}%")

    if value_losses and any(v > 0 for v in value_losses):
        print(f"Final Value Loss: {value_losses[-1]:.6f}")
    if policy_losses and any(p > 0 for p in policy_losses):
        print(f"Final Policy Loss: {policy_losses[-1]:.6f}")

    print()
    print("Interpretation:")
    if len(iterations) < 50:
        print("  ⚠️ Small sample size - results may be noisy")

    if overall_avg_payoff > 100:
        print("  🔥 Excellent! Strong exploitation of GTO")
    elif overall_avg_payoff > 50:
        print("  ✓ Good! Solid exploitative edge")
    elif overall_avg_payoff > 0:
        print("  ⚠ Marginal: Slightly positive, may need more training")
    else:
        print("  ✗ Negative: Check training setup")

    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot training curves')
    parser.add_argument('--metrics', type=str, default=None,
                       help='Path to metrics JSON')
    parser.add_argument('--output', type=str, default='training_curves.png',
                       help='Output image path')
    parser.add_argument('--show', action='store_true',
                       help='Display plot')

    args = parser.parse_args()
    plot_training_curves(args.metrics, args.output, args.show)


if __name__ == '__main__':
    main()
