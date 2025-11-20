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
echo "=== Training Metrics ==="
echo "Run 'python scripts/plot_metrics.py' to generate metrics graph"
echo "Or check Modal dashboard: https://modal.com/apps"

