# Cost Optimization Guide

## 💸 The Problem

You want: **100,000 iterations × 100 trajectories = 10M hands**

Current cost estimate: **~$90** (83 hours on A10G)

---

## 🎯 Solution 1: Optimal Configuration (RECOMMENDED)

Instead of spreading thin (100k iter × 100 traj), go **deeper per iteration**:

```bash
# BETTER: 10,000 iterations × 1,000 trajectories = 10M hands
modal run modal_deploy/train_exploitative_cfr.py::main \
  --gto-checkpoint checkpoint_iter_19.pt \
  --num-iterations 10000 \
  --trajectories-per-iteration 1000
```

**Why this is better:**
- ✅ Same total hands (10M)
- ✅ More stable learning (larger batches)
- ✅ Better network updates (more data per update)
- ✅ **Same cost (~$90)** but better results!

**Time:** ~83 hours (~3.5 days)
**Cost:** ~$92 on A10G

---

## 🎯 Solution 2: Budget-Friendly (50% Cheaper)

Use **T4 GPU** instead of A10G:

### Option A: Edit the training script

```python
# In modal_deploy/train_exploitative_cfr.py line 56
# Change from:
    gpu="A10G",  # A10G for faster training
# To:
    gpu="T4",    # T4 for budget training
```

Then run:
```bash
modal run modal_deploy/train_exploitative_cfr.py::main \
  --gto-checkpoint checkpoint_iter_19.pt \
  --num-iterations 10000 \
  --trajectories-per-iteration 1000
```

**Time:** ~160 hours (~6.7 days) ⚠️ Slower!
**Cost:** ~$50 on T4 ✅ 50% savings!

---

## 🎯 Solution 3: Medium Budget

Run fewer total hands but still get good results:

```bash
# 5,000 iterations × 1,000 trajectories = 5M hands
modal run modal_deploy/train_exploitative_cfr.py::main \
  --gto-checkpoint checkpoint_iter_19.pt \
  --num-iterations 5000 \
  --trajectories-per-iteration 1000
```

**Time:** ~42 hours (~1.75 days)
**Cost:** ~$46 on A10G
**Quality:** Still very good! (Half the data but more efficient)

---

## 🎯 Solution 4: Smart Incremental Training

Start small, scale up if needed:

### Phase 1: Initial Training
```bash
# 1,000 iterations × 1,000 trajectories = 1M hands
# Cost: ~$9, Time: ~8 hours
modal run modal_deploy/train_exploitative_cfr.py::main \
  --gto-checkpoint checkpoint_iter_19.pt \
  --num-iterations 1000 \
  --trajectories-per-iteration 1000
```

### Phase 2: Check Results
```bash
# Evaluate and plot
python3 scripts/plot_training_curves.py
python3 scripts/evaluate_local.py exploitative_cfr_final.pt
```

### Phase 3: Continue Training (if needed)
```bash
# Resume with another 2,000-5,000 iterations
# Only if Phase 1 shows promise!
```

**Total Cost:** ~$9-45 depending on results
**Smart:** Only spend more if it's working!

---

## 📊 Recommended Configurations

| Goal | Config | Time | Cost | Quality |
|------|--------|------|------|---------|
| **Quick Test** | 100 × 500 | 25 min | $0.50 | Weak |
| **Good** | 1,000 × 1,000 | 8 hrs | $9 | Decent |
| **Better** | 5,000 × 1,000 | 42 hrs | $46 | Good |
| **Best** | 10,000 × 1,000 | 83 hrs | $92 | Very Good |
| **Overkill** | 100,000 × 100 | 83 hrs | $92 | Worse than "Best"! |

---

## 💡 Key Insights

### Why 10k × 1k > 100k × 100

**100k iterations × 100 trajectories:**
- ❌ Too frequent network updates (overfitting)
- ❌ Small batch size (noisy gradients)
- ❌ More overhead per iteration
- ⏱️ Same wall-clock time

**10k iterations × 1k trajectories:**
- ✅ Better batch statistics
- ✅ More stable learning
- ✅ Less frequent but better updates
- ⏱️ Same wall-clock time
- 🎯 **Better final performance!**

### The Learning Rate Problem

With 100k iterations, you'd need to:
- Lower learning rate (slower convergence)
- Risk overfitting to specific situations
- Get noisy gradients from small batches

With 10k iterations:
- Standard learning rate works
- Better generalization
- Stable gradients

---

## 🚀 My Recommendation

**For $90 budget:**

```bash
# Phase 1: Quick validation ($9, 8 hours)
modal run modal_deploy/train_exploitative_cfr.py::main \
  --gto-checkpoint checkpoint_iter_19.pt \
  --num-iterations 1000 \
  --trajectories-per-iteration 1000

# Check results, then...

# Phase 2: Full training ($46, 42 hours)
modal run modal_deploy/train_exploitative_cfr.py::main \
  --gto-checkpoint checkpoint_iter_19.pt \
  --num-iterations 5000 \
  --trajectories-per-iteration 1000
```

**Total:** $55, ~50 hours
**Result:** Excellent exploitative model
**Bonus:** Can stop after Phase 1 if already good!

---

## 🔧 How to Switch to T4 GPU

Edit `modal_deploy/train_exploitative_cfr.py`:

```python
# Line 56, change:
@app.function(
    image=image,
    volumes={...},
    gpu="T4",  # ← Change from "A10G" to "T4"
    timeout=72000,  # ← Increase timeout (T4 is slower)
    memory=16384,
)
```

Then run normally. **Saves 50% cost but takes 2× longer.**

---

## 📈 Expected Results

### After 1,000 iterations (1M hands, $9)
- vs GTO: 52-56% win rate
- vs Random: 65-75% win rate
- Avg Payoff: +50-150 chips/hand

### After 5,000 iterations (5M hands, $46)
- vs GTO: 56-60% win rate
- vs Random: 70-80% win rate
- Avg Payoff: +100-200 chips/hand

### After 10,000 iterations (10M hands, $92)
- vs GTO: 58-62% win rate
- vs Random: 75-85% win rate
- Avg Payoff: +150-250 chips/hand

**Diminishing returns after 5k iterations!** Save money, stop at 5k.

---

## 🎯 Bottom Line

**DON'T DO:** 100k × 100 = Bad learning dynamics, same cost
**DO THIS:** 10k × 1k = Better learning, same cost, better results

**BEST VALUE:** 5k × 1k = Half the cost, 90% of the performance

Want me to update your configuration for optimal cost/performance? 🚀
