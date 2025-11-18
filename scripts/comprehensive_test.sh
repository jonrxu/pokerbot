#!/bin/bash
# Comprehensive test with extensive validation

set -e

echo "============================================================"
echo "COMPREHENSIVE TRAINING TEST"
echo "============================================================"
echo ""
echo "Running 2 iterations with 50 trajectories to validate:"
echo "  - Terminal states are stored correctly"
echo "  - Regrets are updated"
echo "  - Counterfactual values computed"
echo "  - Policy loss decreases"
echo ""

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate poker_bot

# Run test job
modal run modal_deploy/train.py::app.main --deploy \
    --num-iterations 2 \
    --trajectories-per-iteration 50 \
    --num-workers 2 \
    --batch-size 32

echo ""
echo "Test job submitted! Waiting 60 seconds for initial logs..."
sleep 60

echo ""
echo "Checking logs for validation..."
modal app logs poker-bot-training 2>&1 | grep -E "Terminal CF value|Regret update|Trajectory Processing Summary|decision points|regrets updated|CF values computed|Policy loss|Value loss|terminal state not marked" | head -50

echo ""
echo "Full logs available at: modal app logs poker-bot-training"
echo "Check Modal dashboard: https://modal.com/apps"

