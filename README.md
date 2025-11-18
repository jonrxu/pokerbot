# Deep CFR Poker Bot

A heads-up poker bot trained using Deep Counterfactual Regret Minimization (Deep CFR) with neural networks, trained via distributed self-play on Modal.

## Features

- **Deep CFR Algorithm**: State-of-the-art Nash equilibrium training
- **Neural Networks**: ResNet-style architecture (512 hidden units, 6 residual layers)
- **Distributed Training**: Parallel trajectory generation + GPU training on Modal
- **Robustness**: Comprehensive error handling, retry logic, and validation
- **Checkpointing**: Automatic saves with resume capability
- **Observability**: Detailed metrics logging and monitoring
- **Self-Play**: Learns from millions of games without human data

## Setup

1. **Create conda environment:**
```bash
conda create -n poker_bot python=3.10
conda activate poker_bot
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up Modal:**
```bash
modal token set
# Create volume (if not exists)
modal volume create poker-bot-checkpoints
```

4. **Verify setup:**
```bash
# Test with small job
./scripts/start_training.sh 2 10 1 "" 8
```

## Quick Start: Phase 1 Training

**Start asynchronous training job** (Nash equilibrium - runs independently):

```bash
# Main training job (recommended)
./scripts/start_training.sh 1000 10000 4 "" 32

# Or with custom parameters
./scripts/start_training.sh [iterations] [trajectories] [workers] [resume_from] [batch_size]
```

**Monitor progress:**
```bash
# Check status and download latest metrics
./scripts/check_status.sh

# View metrics locally
python scripts/view_metrics.py

# Stop training if needed
./scripts/stop_training.sh
```

**Training Details:**
- **Iterations:** 1,000
- **Trajectories per iteration:** 10,000
- **Total games:** 10,000,000 hands
- **Workers:** 4 parallel CPU workers
- **GPU:** T4 GPU for network training
- **Expected duration:** 3-7 days (~80-160 hours)
- **Expected cost:** $8-16 (within $30 budget)

## LLM Training with Tinker

To train the LLM-based agent using Reinforcement Learning on the Tinker platform:

1.  **Setup Tinker**: Ensure you have the `poker_bot_tinker` environment active and your `TINKER_API_KEY` set.
2.  **Self-Play Training**: Run the self-play training loop where the model plays against itself.
    ```bash
    python scripts/train_selfplay.py \
      --model-name Qwen/Qwen3-4B-Instruct-2507 \
      --num-batches 10 \
      --episodes-per-batch 16 \
      --log-path /tmp/tinker-examples/rl_poker_selfplay
    ```
3.  **Evaluation**: Evaluate the trained model against baselines or checkpoints.
    ```bash
    python scripts/tinker_vs_checkpoint.py \
      --checkpoint-iter 195 \
      --rl-log-path /tmp/tinker-examples/rl_poker_selfplay
    ```

## Robustness Features

The training system includes comprehensive error handling:

- ✅ **Checkpoint loading**: Automatic fallback to new networks if corrupted
- ✅ **Trajectory validation**: Skips invalid trajectories gracefully
- ✅ **Worker retry logic**: Retries failed workers up to 2 times
- ✅ **Volume commit retries**: 3 retry attempts for checkpoint persistence
- ✅ **Atomic checkpoint writes**: Prevents corruption during saves
- ✅ **Gradient clipping**: Prevents exploding gradients (max_norm=1.0)
- ✅ **NaN/Inf detection**: Skips invalid loss values automatically
- ✅ **Metrics validation**: Filters invalid metrics before logging
- ✅ **GPU fallback**: Warns when falling back to CPU

## Project Structure

- `poker_game/`: Game logic and state management
- `models/`: Neural network architectures (ResNet-style)
- `training/`: Deep CFR algorithm and training loops
- `modal_deploy/`: Modal deployment functions and configuration
- `checkpoints/`: Checkpoint management utilities
- `evaluation/`: Evaluation and exploitability metrics
- `scripts/`: Helper scripts for training management
  - `start_training.sh`: Start async training job
  - `check_status.sh`: Monitor training progress
  - `view_metrics.py`: View training metrics
  - `stop_training.sh`: Stop running training job
- `bootstrap.py`: Bootstrap utilities for warm-start training

## Expected Bot Quality

After Phase 1 training (1,000 iterations):

- **vs. Random opponents:** 70-80%+ win rate
- **vs. Weak players:** 60-70%+ win rate  
- **vs. Strong players:** 50-55% (near Nash, hard to exploit)
- **Exploitability:** < 0.1 big blinds per hand
- **Capabilities:** Pre-flop selection, post-flop betting, value betting, bluffing, pot odds, position play

