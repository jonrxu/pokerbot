#!/bin/bash
# Check Modal credits and provide cost-aware recommendations

echo "========================================================================"
echo "MODAL CREDITS & COST CHECK"
echo "========================================================================"
echo ""

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal is not installed!"
    echo ""
    echo "Install it with:"
    echo "  pip install modal"
    echo "  modal token new"
    exit 1
fi

# Check if authenticated
if ! modal profile current &> /dev/null; then
    echo "❌ Not logged in to Modal!"
    echo ""
    echo "Login with:"
    echo "  modal token new"
    exit 1
fi

echo "✓ Modal is installed and authenticated"
echo ""

# Show current profile
echo "Current profile:"
modal profile current
echo ""

# Check volumes
echo "========================================================================"
echo "CHECKING VOLUMES"
echo "========================================================================"
echo ""

echo "Your GTO checkpoint volume:"
modal volume ls poker-bot-checkpoints 2>&1 | grep -E "checkpoint_iter|Volume not found" || echo "Volume exists but is empty"
echo ""

echo "Your exploitative checkpoint volume:"
modal volume ls poker-exploitative-cfr-checkpoints 2>&1 | grep -E "exploitative_cfr|metrics_iter|Volume not found|Volume exists" || echo "Volume exists but is empty"
echo ""

# Estimate costs
echo "========================================================================"
echo "COST ESTIMATES"
echo "========================================================================"
echo ""

python3 modal_deploy/cost_estimator.py --iterations 10 --trajectories 100 2>&1 | grep -A 20 "MODAL COST ESTIMATE"
echo ""
echo "For FULL training:"
python3 modal_deploy/cost_estimator.py --iterations 1000 --trajectories 1000 2>&1 | grep "TOTAL:" | head -1

echo ""
echo "========================================================================"
echo "RECOMMENDATIONS"
echo "========================================================================"
echo ""
echo "Before running expensive training:"
echo "  1. ✓ Check your Modal credits at: https://modal.com/settings/billing"
echo "  2. ✓ Always run a quick test first (\$0.09)"
echo "  3. ✓ Monitor costs via Modal dashboard"
echo "  4. ✓ Set up billing alerts if available"
echo ""
echo "Quick test command (recommended):"
echo "  modal run modal_deploy/train_exploitative_cfr.py::main \\"
echo "    --gto-checkpoint checkpoint_iter_19.pt \\"
echo "    --num-iterations 10 \\"
echo "    --trajectories-per-iteration 100"
echo ""
