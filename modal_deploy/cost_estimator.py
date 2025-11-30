#!/usr/bin/env python3
"""Modal cost estimator for exploitative CFR training.

This script helps estimate costs before running expensive training jobs.
"""

import argparse


class ModalCostEstimator:
    """Estimate Modal costs for poker bot training."""

    # Modal pricing (as of 2024)
    # See: https://modal.com/pricing
    A10G_COST_PER_HOUR = 1.10  # $/hour for A10G GPU
    T4_COST_PER_HOUR = 0.60    # $/hour for T4 GPU (cheaper alternative)
    CPU_COST_PER_HOUR = 0.0003  # $/hour per vCPU
    MEMORY_COST_PER_GB_HOUR = 0.0004  # $/GB/hour

    def __init__(self, gpu_type='T4'):
        """Initialize cost estimator.

        Args:
            gpu_type: 'A10G' or 'T4'
        """
        self.gpu_type = 'T4'

        if gpu_type == 'A10G':
            self.gpu_cost = self.A10G_COST_PER_HOUR
        elif gpu_type == 'T4':
            self.gpu_cost = self.T4_COST_PER_HOUR
        else:
            raise ValueError(f"Unknown GPU type: {gpu_type}")

    def estimate_training_time(
        self,
        num_iterations: int,
        trajectories_per_iteration: int,
    ) -> float:
        """Estimate training time in hours.

        Based on empirical measurements:
        - A10G: ~30 seconds per iteration with 1000 trajectories
        - T4: ~60 seconds per iteration with 1000 trajectories

        Args:
            num_iterations: Number of training iterations
            trajectories_per_iteration: Trajectories per iteration

        Returns:
            Estimated hours
        """
        # Base time per iteration (seconds)
        if self.gpu_type == 'A10G':
            base_seconds = 30
        else:  # T4
            base_seconds = 60

        # Scale by trajectory count
        scaling_factor = trajectories_per_iteration / 1000.0
        seconds_per_iteration = base_seconds * scaling_factor

        # Total time
        total_seconds = num_iterations * seconds_per_iteration
        total_hours = total_seconds / 3600.0

        return total_hours

    def estimate_cost(
        self,
        num_iterations: int,
        trajectories_per_iteration: int,
        cpu_cores: int = 4,
        memory_gb: int = 16,
    ) -> dict:
        """Estimate total cost for training run.

        Args:
            num_iterations: Number of training iterations
            trajectories_per_iteration: Trajectories per iteration
            cpu_cores: Number of CPU cores
            memory_gb: Memory in GB

        Returns:
            Dictionary with cost breakdown
        """
        hours = self.estimate_training_time(num_iterations, trajectories_per_iteration)

        # Calculate component costs
        gpu_cost = self.gpu_cost * hours
        cpu_cost = self.CPU_COST_PER_HOUR * cpu_cores * hours
        memory_cost = self.MEMORY_COST_PER_GB_HOUR * memory_gb * hours

        total_cost = gpu_cost + cpu_cost + memory_cost

        return {
            'hours': hours,
            'gpu_cost': gpu_cost,
            'cpu_cost': cpu_cost,
            'memory_cost': memory_cost,
            'total_cost': total_cost,
            'gpu_type': self.gpu_type,
        }

    def print_estimate(self, estimate: dict):
        """Print cost estimate in readable format."""
        print("=" * 80)
        print("MODAL COST ESTIMATE")
        print("=" * 80)
        print(f"GPU Type: {estimate['gpu_type']}")
        print(f"Estimated Runtime: {estimate['hours']:.2f} hours")
        print()
        print("Cost Breakdown:")
        print(f"  GPU:    ${estimate['gpu_cost']:.2f}")
        print(f"  CPU:    ${estimate['cpu_cost']:.4f}")
        print(f"  Memory: ${estimate['memory_cost']:.4f}")
        print("-" * 80)
        print(f"  TOTAL:  ${estimate['total_cost']:.2f}")
        print("=" * 80)


def main():
    """CLI for cost estimation."""
    parser = argparse.ArgumentParser(
        description='Estimate Modal costs for exploitative CFR training'
    )
    parser.add_argument(
        '--iterations', type=int, default=1000,
        help='Number of training iterations (default: 1000)'
    )
    parser.add_argument(
        '--trajectories', type=int, default=1000,
        help='Trajectories per iteration (default: 1000)'
    )
    parser.add_argument(
        '--gpu', type=str, default='A10G', choices=['A10G', 'T4'],
        help='GPU type (default: A10G)'
    )
    parser.add_argument(
        '--budget', type=float, default=None,
        help='Budget limit in dollars (optional)'
    )

    args = parser.parse_args()

    # Create estimator
    estimator = ModalCostEstimator(gpu_type=args.gpu)

    # Estimate cost
    estimate = estimator.estimate_cost(
        num_iterations=args.iterations,
        trajectories_per_iteration=args.trajectories,
    )

    # Print estimate
    estimator.print_estimate(estimate)

    # Check budget
    if args.budget is not None:
        print()
        if estimate['total_cost'] <= args.budget:
            print(f"✓ Within budget! (${estimate['total_cost']:.2f} <= ${args.budget:.2f})")
        else:
            print(f"✗ OVER BUDGET! (${estimate['total_cost']:.2f} > ${args.budget:.2f})")
            overage = estimate['total_cost'] - args.budget
            print(f"  Overage: ${overage:.2f}")

            # Suggest alternatives
            print()
            print("Suggestions to reduce cost:")

            # Option 1: Reduce iterations
            reduced_iters = int(args.iterations * args.budget / estimate['total_cost'])
            print(f"  1. Reduce iterations to {reduced_iters}")

            # Option 2: Use T4 GPU
            if args.gpu == 'A10G':
                t4_estimator = ModalCostEstimator(gpu_type='T4')
                t4_estimate = t4_estimator.estimate_cost(
                    args.iterations, args.trajectories
                )
                print(f"  2. Use T4 GPU instead (${t4_estimate['total_cost']:.2f})")

            # Option 3: Reduce trajectories
            reduced_trajs = int(args.trajectories * args.budget / estimate['total_cost'])
            print(f"  3. Reduce trajectories to {reduced_trajs} per iteration")

    # Provide quick test recommendation
    print()
    print("=" * 80)
    print("RECOMMENDATION: Start with a Quick Test")
    print("=" * 80)
    print()
    print("Before running the full training, do a quick test:")

    # Quick test estimate
    quick_estimate = estimator.estimate_cost(
        num_iterations=10,
        trajectories_per_iteration=100,
    )

    print(f"  Iterations: 10")
    print(f"  Trajectories: 100")
    print(f"  Cost: ${quick_estimate['total_cost']:.2f}")
    print(f"  Time: {quick_estimate['hours'] * 60:.1f} minutes")
    print()
    print("This will:")
    print("  ✓ Verify your setup works")
    print("  ✓ Validate checkpoint loading")
    print("  ✓ Test the training loop")
    print("  ✓ Cost almost nothing")
    print()
    print("Command:")
    print("  modal run modal_deploy/train_exploitative_cfr.py::main \\")
    print("    --gto-checkpoint checkpoint_iter_19.pt \\")
    print("    --num-iterations 10 \\")
    print("    --trajectories-per-iteration 100")


if __name__ == '__main__':
    main()
