# Modal Deployment Guide: Exploitative CFR Training

Complete guide to training your exploitative poker bot on Modal with cost controls and benchmarking.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup Modal](#setup-modal)
3. [Cost Estimation](#cost-estimation)
4. [Upload Checkpoint](#upload-checkpoint)
5. [Run Training](#run-training)
6. [Monitor Progress](#monitor-progress)
7. [Download Results](#download-results)
8. [Evaluate Model](#evaluate-model)
9. [Cost Controls](#cost-controls)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### What You Need

- ✅ Trained GTO checkpoint (e.g., `checkpoint_iter_19.pt`)
- ✅ Modal account with credits
- ✅ Modal CLI installed

### Check Your Checkpoint

```bash
# Verify checkpoint exists
ls checkpoints/checkpoint_iter_19.pt

# Check checkpoint contents
python3 -c "
import torch
checkpoint = torch.load('checkpoints/checkpoint_iter_19.pt', weights_only=False)
print('Keys:', list(checkpoint.keys()))
print('Has value_net_state:', 'value_net_state' in checkpoint)
print('Has policy_net_state:', 'policy_net_state' in checkpoint)
"
```

Expected output:
```
Keys: ['iteration', 'value_net_state', 'policy_net_state', 'value_optimizer_state', 'policy_optimizer_state', 'regret_memory', 'strategy_memory', 'counterfactual_values', 'metrics']
Has value_net_state: True
Has policy_net_state: True
```

---

## Setup Modal

### Install Modal

```bash
# Install Modal
pip install modal

# Login to Modal
modal token new
```

This will open your browser to authenticate. Follow the prompts.

### Verify Installation

```bash
# Check Modal is working
modal profile current

# Check Modal volumes
modal volume ls
```

---

## Cost Estimation

**IMPORTANT:** Always estimate costs before running expensive jobs!

### Estimate Your Training Run

```bash
# Estimate full training (1000 iterations)
python3 modal_deploy/cost_estimator.py \
  --iterations 1000 \
  --trajectories 1000 \
  --gpu A10G

# Example output:
# ================================================================================
# MODAL COST ESTIMATE
# ================================================================================
# GPU Type: A10G
# Estimated Runtime: 8.33 hours
#
# Cost Breakdown:
#   GPU:    $9.17
#   CPU:    $0.0001
#   Memory: $0.0005
# --------------------------------------------------------------------------------
#   TOTAL:  $9.17
# ================================================================================
```

### Set a Budget

```bash
# Check if within budget
python3 modal_deploy/cost_estimator.py \
  --iterations 1000 \
  --trajectories 1000 \
  --budget 10.00

# If over budget, script will suggest alternatives
```

### Cost Breakdown by Configuration

| Configuration | Iterations | Trajectories | GPU | Time | Cost |
|--------------|-----------|--------------|-----|------|------|
| **Quick Test** | 10 | 100 | A10G | ~5 min | **$0.09** |
| **Small** | 100 | 500 | A10G | ~25 min | **$0.46** |
| **Medium** | 500 | 1000 | A10G | ~4.2 hrs | **$4.58** |
| **Full** | 1000 | 1000 | A10G | ~8.3 hrs | **$9.17** |
| **Budget (T4)** | 1000 | 1000 | T4 | ~16.7 hrs | **$10.00** |

---

## Upload Checkpoint

### Upload GTO Checkpoint to Modal

```bash
# Upload your GTO checkpoint
modal volume put poker-bot-checkpoints \
  checkpoints/checkpoint_iter_19.pt \
  checkpoint_iter_19.pt

# Verify upload
modal volume ls poker-bot-checkpoints | grep checkpoint_iter_19
```

Expected output:
```
checkpoint_iter_19.pt  (uploaded successfully)
```

---

## Run Training

### Step 1: Quick Test (Recommended)

**Always start with a quick test to verify everything works!**

```bash
# 10 iterations, 100 trajectories (~5 minutes, ~$0.09)
modal run modal_deploy/train_exploitative_cfr.py::main \
  --gto-checkpoint checkpoint_iter_19.pt \
  --num-iterations 10 \
  --trajectories-per-iteration 100
```

**What to look for:**
- ✅ Checkpoint loads successfully
- ✅ Training starts without errors
- ✅ Metrics are logged each iteration
- ✅ Checkpoints are saved

If the quick test succeeds, proceed to full training!

### Step 2: Full Training

```bash
# Full training (1000 iterations, 1000 trajectories)
modal run modal_deploy/train_exploitative_cfr.py::main \
  --gto-checkpoint checkpoint_iter_19.pt \
  --num-iterations 1000 \
  --trajectories-per-iteration 1000
```

### Alternative Configurations

```bash
# Medium training (500 iterations, faster)
modal run modal_deploy/train_exploitative_cfr.py::main \
  --gto-checkpoint checkpoint_iter_19.pt \
  --num-iterations 500 \
  --trajectories-per-iteration 1000

# Budget training (use T4 GPU instead of A10G)
# Note: Edit train_exploitative_cfr.py line 56 to use gpu="T4"
modal run modal_deploy/train_exploitative_cfr.py::main \
  --gto-checkpoint checkpoint_iter_19.pt \
  --num-iterations 1000 \
  --trajectories-per-iteration 1000
```

---

## Monitor Progress

### Real-time Logs

Modal will show real-time logs in your terminal:

```
================================================================================
Iteration 1/1000
================================================================================
Avg Payoff: +45.23 chips
Win Rate: 52.34%
Trajectories: 1000
Value Buffer: 8234
Policy Buffer: 8234
```

**What to look for:**

- **Avg Payoff**: Should be positive and increasing (means we're beating the GTO player)
  - Starting: ~0-50 chips
  - Good progress: 100-200 chips
  - Excellent: 200+ chips

- **Win Rate**: Should be > 50% and increasing
  - Starting: ~50-52%
  - Good: 55-60%
  - Excellent: 60%+

### Check Progress Online

You can also view logs on the Modal dashboard:

```bash
# Get your Modal app URL
modal app list
```

Visit: https://modal.com/apps

---

## Download Results

### Download Final Model

```bash
# Download final trained model
modal volume get poker-exploitative-cfr-checkpoints \
  exploitative_cfr_final.pt \
  ./exploitative_vs_iter19.pt
```

### Download Intermediate Checkpoints

```bash
# List all checkpoints
modal volume ls poker-exploitative-cfr-checkpoints

# Download specific iteration
modal volume get poker-exploitative-cfr-checkpoints \
  exploitative_cfr_iter_0500.pt \
  ./exploitative_iter_500.pt
```

### Download Metrics

```bash
# Download metrics for plotting
modal volume get poker-exploitative-cfr-checkpoints \
  metrics_iter_1000.json \
  ./metrics.json
```

---

## Evaluate Model

### Load and Use Your Trained Model

```python
import torch
from poker_game.game import PokerGame
from poker_game.state_encoder import StateEncoder
from poker_game.information_set import get_information_set
from models.value_policy_net import ValuePolicyNet
from training.deep_cfr import DeepCFR

# Load checkpoint
checkpoint = torch.load('exploitative_vs_iter19.pt')

# Initialize game and encoder
game = PokerGame(small_blind=50, big_blind=100, is_limit=False)
encoder = StateEncoder()

# Create networks
input_dim = encoder.feature_dim
value_net = ValuePolicyNet(input_dim=input_dim)
policy_net = ValuePolicyNet(input_dim=input_dim)

# Load trained weights
value_net.load_state_dict(checkpoint['value_net_state'])
policy_net.load_state_dict(checkpoint['policy_net_state'])

# Create exploitative agent
exploitative_agent = DeepCFR(
    value_net=value_net,
    policy_net=policy_net,
    state_encoder=encoder,
    game=game,
)

# Load strategy memory
exploitative_agent.strategy_memory = checkpoint['strategy_memory']

# Use it to play!
state = game.reset()
legal_actions = game.get_legal_actions(state)
info_set = get_information_set(state, 0)
strategy = exploitative_agent.get_average_strategy(info_set, legal_actions)

print(f"Exploitative strategy: {strategy}")
```

### Benchmark Against Baselines

```bash
# Evaluate against various opponents
python3 scripts/evaluate_benchmarks.py \
  --current 1000 \
  --num-games 2000
```

This will test your model against:
- Random agent
- Always-call agent
- Baseline TAG agent
- Early iteration models (iter 1, 2, 5, 10)

**Expected results:**
- vs Random: 70-80% win rate
- vs Always-call: 75-85% win rate
- vs Baseline: 60-70% win rate
- vs Early iterations: 55-65% win rate

---

## Cost Controls

### Set Budget Limits

Modal doesn't have built-in budget limits, so **monitor your spending**:

1. **Check your balance before training**
   ```bash
   # Visit Modal dashboard
   open https://modal.com/settings/billing
   ```

2. **Set up alerts**
   - Go to Modal dashboard > Settings > Billing
   - Set up email alerts for spending thresholds

3. **Use quick tests first**
   - Always run small test runs before committing to full training
   - Verify costs match estimates

### Kill a Running Job

If a job is running too long or costing too much:

```bash
# List running apps
modal app list

# Stop a specific app
modal app stop poker-exploitative-cfr-training
```

### Monitor Costs

```bash
# Check Modal usage
modal stats
```

---

## Troubleshooting

### Issue: "Checkpoint not found"

```bash
# Check checkpoint exists in volume
modal volume ls poker-bot-checkpoints | grep checkpoint_iter_19

# If missing, re-upload
modal volume put poker-bot-checkpoints \
  checkpoints/checkpoint_iter_19.pt \
  checkpoint_iter_19.pt
```

### Issue: "Failed to load checkpoint"

The checkpoint might be corrupted. Verify locally:

```python
import torch

# Try loading locally
checkpoint = torch.load('checkpoints/checkpoint_iter_19.pt', weights_only=False)

# Check contents
print("Keys:", checkpoint.keys())

# Should have:
# - value_net_state
# - policy_net_state
# - strategy_memory
# - regret_memory
```

### Issue: "Out of memory"

Reduce batch size or trajectories:

```bash
# Edit train_exploitative_cfr.py line 64
# Change: batch_size: int = 64
# To:     batch_size: int = 32

# Or reduce trajectories
modal run modal_deploy/train_exploitative_cfr.py::main \
  --gto-checkpoint checkpoint_iter_19.pt \
  --num-iterations 1000 \
  --trajectories-per-iteration 500  # Reduced from 1000
```

### Issue: "Modal timeout"

For very long training runs, increase timeout:

```python
# Edit train_exploitative_cfr.py line 57
# Change: timeout=36000,  # 10 hours
# To:     timeout=72000,  # 20 hours
```

### Issue: "Training not improving"

Check your metrics:

1. **Avg Payoff staying near 0?**
   - Model might not be learning to exploit
   - Try increasing learning rate
   - Verify GTO checkpoint is loaded correctly

2. **Win Rate < 50%?**
   - Model is losing to GTO (this is bad!)
   - Check training loop for bugs
   - Verify reward calculation

3. **Losses not decreasing?**
   - Learning rate might be too high or too low
   - Try adjusting in train_exploitative_cfr.py (lines 65-66)

---

## Quick Reference

### Command Cheat Sheet

```bash
# 1. Estimate cost
python3 modal_deploy/cost_estimator.py --iterations 1000 --trajectories 1000

# 2. Upload checkpoint
modal volume put poker-bot-checkpoints checkpoints/checkpoint_iter_19.pt checkpoint_iter_19.pt

# 3. Quick test
modal run modal_deploy/train_exploitative_cfr.py::main --gto-checkpoint checkpoint_iter_19.pt --num-iterations 10 --trajectories-per-iteration 100

# 4. Full training
modal run modal_deploy/train_exploitative_cfr.py::main --gto-checkpoint checkpoint_iter_19.pt --num-iterations 1000 --trajectories-per-iteration 1000

# 5. Download result
modal volume get poker-exploitative-cfr-checkpoints exploitative_cfr_final.pt ./exploitative_vs_iter19.pt

# 6. Evaluate
python3 scripts/evaluate_benchmarks.py --current 1000 --num-games 2000
```

---

## Next Steps

After training completes:

1. **Download your model**
2. **Benchmark against baselines** - verify it beats weak opponents
3. **Test against your GTO model** - should have 55-60%+ win rate
4. **Analyze the strategy** - see what exploits it learned
5. **Use in your poker bot** - integrate into your main application

---

## Support

- **Modal Docs**: https://modal.com/docs
- **Modal Discord**: https://discord.gg/modal
- **Pricing**: https://modal.com/pricing

Good luck with your training! 🚀
