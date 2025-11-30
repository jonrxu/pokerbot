# Quick Start: Run Exploitative Training on Modal

**Get your exploitative poker bot training on Modal in 5 minutes!**

---

## ⚡ TL;DR - Run This Now

```bash
# 1. Install Modal
pip install modal
modal token new

# 2. Estimate cost first!
python3 modal_deploy/cost_estimator.py --iterations 1000 --trajectories 1000

# 3. Upload your GTO checkpoint
modal volume put poker-bot-checkpoints \
  checkpoints/checkpoint_iter_19.pt \
  checkpoint_iter_19.pt

# 4. Quick test (5 min, $0.09)
modal run modal_deploy/train_exploitative_cfr.py::main \
  --gto-checkpoint checkpoint_iter_19.pt \
  --num-iterations 10 \
  --trajectories-per-iteration 100

# 5. If test works, run full training (8 hrs, ~$9)
modal run modal_deploy/train_exploitative_cfr.py::main \
  --gto-checkpoint checkpoint_iter_19.pt \
  --num-iterations 1000 \
  --trajectories-per-iteration 1000

# 6. Download result
modal volume get poker-exploitative-cfr-checkpoints \
  exploitative_cfr_final.pt \
  ./exploitative_vs_iter19.pt

# 7. Plot results
python3 scripts/plot_exploitative_metrics.py

# 8. Benchmark against baselines
python3 scripts/evaluate_benchmarks.py --current 1000 --num-games 2000
```

---

## 📊 What You Get

After training completes, you'll have:

1. **Trained exploitative model** that beats your GTO player
2. **Metrics and checkpoints** saved every 100 iterations
3. **Performance graphs** showing payoff and win rate over time
4. **Benchmark results** vs random, baseline, and early iteration opponents

---

## 💰 Cost Breakdown

| Configuration | Time | Cost | Use Case |
|--------------|------|------|----------|
| **Quick Test** (10 iter, 100 traj) | 5 min | $0.09 | Verify setup works |
| **Small** (100 iter, 500 traj) | 25 min | $0.46 | Quick experiment |
| **Medium** (500 iter, 1000 traj) | 4.2 hrs | $4.58 | Decent model |
| **Full** (1000 iter, 1000 traj) | 8.3 hrs | $9.17 | Best results |

**Budget Tip:** Start with Quick Test, then Medium, then Full only if needed.

---

## 📈 What to Expect

### During Training

You'll see logs like this every iteration:

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

**Good signs:**
- ✅ Avg Payoff > 0 and increasing
- ✅ Win Rate > 50% and increasing
- ✅ No errors or crashes

**Bad signs:**
- ❌ Avg Payoff negative or decreasing
- ❌ Win Rate < 50%
- ❌ Errors loading checkpoint

### After Training

Expected performance:

- **vs GTO Opponent**: 55-60% win rate, +100-200 chips/hand
- **vs Random**: 70-80% win rate
- **vs Always-Call**: 75-85% win rate
- **vs Baseline**: 60-70% win rate

---

## 🔧 Troubleshooting

### "Checkpoint not found"

```bash
# Verify it's uploaded
modal volume ls poker-bot-checkpoints | grep checkpoint_iter_19
```

If not there, re-upload:
```bash
modal volume put poker-bot-checkpoints \
  checkpoints/checkpoint_iter_19.pt \
  checkpoint_iter_19.pt
```

### "Out of credits"

Check your Modal balance:
```bash
open https://modal.com/settings/billing
```

Add credits or use a smaller configuration.

### "Training not improving"

1. Check logs - is avg payoff increasing?
2. Verify GTO checkpoint loaded correctly
3. Try running longer (more iterations)
4. Adjust learning rate in `train_exploitative_cfr.py`

---

## 📚 Full Documentation

For detailed information, see:
- **[MODAL_DEPLOYMENT_GUIDE.md](MODAL_DEPLOYMENT_GUIDE.md)** - Complete deployment guide
- **[modal_deploy/cost_estimator.py](modal_deploy/cost_estimator.py)** - Cost estimation tool
- **[scripts/plot_exploitative_metrics.py](scripts/plot_exploitative_metrics.py)** - Plotting script
- **[scripts/evaluate_benchmarks.py](scripts/evaluate_benchmarks.py)** - Benchmark evaluation

---

## 🚀 Ready to Go!

Your checkpoint (`checkpoint_iter_19.pt`) will be the frozen GTO opponent that the exploitative agent learns to beat.

**Next steps:**
1. Run the quick test to verify everything works
2. If successful, launch full training
3. Monitor progress via logs or Modal dashboard
4. Download and evaluate your trained model
5. Use it to crush your opponents! 🎯

Good luck! 🍀
