"""Test training in Modal with extensive logging and validation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modal_deploy.train import app, training_worker

def main():
    """Run a small test training job."""
    print("="*60)
    print("MODAL TRAINING TEST")
    print("="*60)
    print("\nThis will run a small test training job (3 iterations, 100 trajectories)")
    print("to validate that everything is working correctly.\n")
    
    # Run a very small test
    with app.run():
        result = training_worker.spawn(
            num_iterations=3,
            trajectories_per_iteration=100,  # Small for testing
            num_workers=2,  # Just 2 workers for testing
            start_iteration=0,
            batch_size=32
        )
        
        print("\nWaiting for training to complete...")
        print("Check Modal dashboard for logs: https://modal.com/apps")
        print("\nYou can also check logs with:")
        print("  modal app logs poker-bot-training")
        
        # Wait for result (with timeout)
        try:
            final_result = result.get(timeout=1800)  # 30 minute timeout
            print("\n" + "="*60)
            print("TRAINING TEST COMPLETE!")
            print("="*60)
            print(f"Final result: {final_result}")
        except Exception as e:
            print(f"\nError: {e}")
            print("Check Modal logs for details")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

