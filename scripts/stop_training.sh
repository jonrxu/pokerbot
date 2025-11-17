#!/bin/bash
# Stop running training job

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate poker_bot

echo "Stopping training job..."
modal app stop poker-bot-training 2>/dev/null && \
    echo "✓ Training job stopped" || \
    echo "No active training job found"

