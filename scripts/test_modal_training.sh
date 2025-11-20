#!/bin/bash
# Small test training job in Modal with extensive logging

set -e

echo "============================================================"
echo "MODAL TRAINING TEST"
echo "============================================================"
echo ""
echo "This will run a small test (3 iterations, 100 trajectories)"
echo "to validate training works correctly in Modal."
echo ""
echo "Check logs with: modal app logs poker-bot-training"
echo ""

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate poker_bot

# Run small test job
modal run modal_deploy/train.py::app.main --deploy \
    --num-iterations 3 \
    --trajectories-per-iteration 100 \
    --num-workers 2 \
    --batch-size 32

echo ""
echo "Test job submitted! Check Modal dashboard for progress:"
echo "https://modal.com/apps"
echo ""
echo "View logs: modal app logs poker-bot-training"

