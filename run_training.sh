#!/bin/bash
# Training script for Deep CFR Poker Bot

echo "Deep CFR Poker Bot Training"
echo "=========================="

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    echo "Error: Modal CLI not found. Install with: pip install modal"
    exit 1
fi

# Set Modal token if not set
if [ -z "$MODAL_TOKEN_ID" ]; then
    echo "Warning: MODAL_TOKEN_ID not set. Run: modal token set"
fi

# Parse arguments
MODE=${1:-local}  # local or modal
ITERATIONS=${2:-1000}
TRAJECTORIES=${3:-1000}

if [ "$MODE" == "modal" ]; then
    echo "Starting distributed training on Modal..."
    echo "Iterations: $ITERATIONS"
    echo "Trajectories per iteration: $TRAJECTORIES"
    
    modal run modal_train.py::main \
        --num-iterations $ITERATIONS \
        --trajectories-per-iteration $TRAJECTORIES \
        --num-workers 4
else
    echo "Starting local training..."
    echo "Iterations: $ITERATIONS"
    echo "Trajectories per iteration: $TRAJECTORIES"
    
    python train.py \
        --iterations $ITERATIONS \
        --trajectories $TRAJECTORIES \
        --device cpu
fi

