#!/bin/bash
# Start asynchronous training job on Modal

set -e

# Default values
ITERATIONS=${1:-1000}
TRAJECTORIES=${2:-10000}
WORKERS=${3:-16}  # Increased to 16 for faster parallel trajectory generation
RESUME_FROM=${4:-}
BATCH_SIZE=${5:-128}  # Increased to 128 for better GPU utilization with A10G

echo "=========================================="
echo "Poker Bot Training - Phase 1: Nash Equilibrium"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Iterations: $ITERATIONS"
echo "  Trajectories per iteration: $TRAJECTORIES"
echo "  Workers: $WORKERS"
echo "  Batch size: $BATCH_SIZE"
if [ -n "$RESUME_FROM" ]; then
    echo "  Resuming from iteration: $RESUME_FROM"
fi
echo ""

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate poker_bot

# Start async training job
echo "Submitting job to Modal..."
if [ -n "$RESUME_FROM" ]; then
    modal run modal_train.py::app.main \
        --num-iterations $ITERATIONS \
        --trajectories-per-iteration $TRAJECTORIES \
        --num-workers $WORKERS \
        --batch-size $BATCH_SIZE \
        --resume-from $RESUME_FROM \
        --deploy
else
    modal run modal_train.py::app.main \
        --num-iterations $ITERATIONS \
        --trajectories-per-iteration $TRAJECTORIES \
        --num-workers $WORKERS \
        --batch-size $BATCH_SIZE \
        --deploy
fi

echo ""
echo "✓ Job submitted successfully!"
echo ""
echo "To monitor progress:"
echo "  1. Check Modal dashboard: https://modal.com/apps"
echo "  2. View logs: modal app logs poker-bot-training"
echo "  3. Download metrics: modal volume download poker-bot-checkpoints /checkpoints/metrics ./local_metrics"
echo ""

