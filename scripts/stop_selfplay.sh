#!/bin/bash
# Stops the self-play training process

echo "Stopping self-play training..."
pkill -f "scripts/train_selfplay.py"
echo "Done."

