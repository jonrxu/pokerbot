#!/bin/bash
# Check training status and download metrics

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate poker_bot

echo "Checking training status..."
echo ""

# Check Modal app status
echo "=== Modal App Status ==="
modal app list 2>/dev/null | grep poker-bot-training || echo "No active app found"

echo ""
echo "=== Recent Logs (last 30 lines) ==="
modal app logs poker-bot-training --tail 30 2>/dev/null || echo "No logs available yet"

echo ""
echo "=== Downloading Metrics ==="
mkdir -p ./local_metrics
if modal volume download poker-bot-checkpoints /checkpoints/metrics ./local_metrics 2>/dev/null; then
    echo "✓ Metrics downloaded to ./local_metrics"
    
    if [ -f "./local_metrics/summary.json" ]; then
        echo ""
        echo "=== Training Summary ==="
        python scripts/view_metrics.py ./local_metrics
    else
        echo "No summary file found yet (training may not have started)"
    fi
else
    echo "No metrics available yet (training may not have started)"
fi

echo ""
echo "=== Downloading Logs ==="
mkdir -p ./local_logs
modal volume download poker-bot-checkpoints /checkpoints/training.log ./local_logs/training.log 2>/dev/null && \
    echo "✓ Logs downloaded to ./local_logs/training.log" || \
    echo "No log file available yet"

