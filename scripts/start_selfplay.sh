#!/bin/bash
# Starts self-play training in the background

# Use the provided API key
export TINKER_API_KEY="tml-LXqec8g7qYUUKPJs5WwiLPQtH6WcthESiz7YYuiEG47XcuQT5ewV0kyGAdkFSftMFAAAA"
LOG_FILE="selfplay_training.log"
PYTHON_EXEC="/Users/jonathanxu/anaconda3/envs/poker_bot_tinker/bin/python"

echo "Starting self-play training in background..."
# Use -u to unbuffer output so logs appear immediately
nohup "$PYTHON_EXEC" -u scripts/train_selfplay.py \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --num-batches 50 \
  --episodes-per-batch 32 \
  --log-path /tmp/tinker-examples/rl_poker_selfplay_full \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training started with PID $PID"
echo "Logs are being written to $LOG_FILE"
echo "Run './scripts/monitor_selfplay.sh' to watch progress."
