# Evaluation Guide: Loss Curves & Baseline Comparisons

## 📉 1. Loss Curves (NEW!)

I just added loss tracking to your training! Now you'll see:
- **Value Loss** - How well the network predicts payoffs
- **Policy Loss** - How well the network learns strategies

### Plot Everything (Recommended)

```bash
# Download metrics and plot all curves
python3 scripts/plot_training_curves.py
```

**Output:** `training_curves.png` with 4 subplots:
1. Average Payoff (with moving average)
2. Win Rate (with moving average)
3. Value Loss (log scale, with trend)
4. Policy Loss (log scale, with trend)

### What Good Losses Look Like

**Value Loss (MSE):**
- Start: 10,000 - 100,000
- Early (100 iter): 1,000 - 10,000
- Mid (500 iter): 100 - 1,000
- Late (1000 iter): 10 - 100
- ✅ Should **decrease** steadily

**Policy Loss (KL Divergence):**
- Start: 1.0 - 5.0
- Early (100 iter): 0.5 - 2.0
- Mid (500 iter): 0.1 - 0.5
- Late (1000 iter): 0.01 - 0.1
- ✅ Should **decrease** steadily

**Red Flags:**
- ❌ Losses increasing over time
- ❌ Losses stuck at same value (not learning)
- ❌ Losses spiking wildly (instability)

---

## 🎯 2. Baseline Comparisons

You have an excellent benchmarking script already! It tests your model against:
- **Random Agent** - Plays completely randomly
- **Always-Call Agent** - Calls/checks every time
- **Baseline TAG Agent** - Tight-aggressive heuristic
- **Early Iterations** - Your model from iterations 1, 2, 5, 10

### Run Baseline Evaluation

```bash
# After training completes, download your model
modal volume get poker-exploitative-cfr-checkpoints \
  exploitative_cfr_final.pt \
  ./exploitative_cfr_final.pt

# Then evaluate against baselines (2000 games each)
python3 scripts/evaluate_benchmarks.py \
  --current 1000 \
  --num-games 2000
```

**Note:** Replace `1000` with your final iteration number.

### Expected Results

After 1000 iterations of good training:

| Opponent | Expected Win Rate | What It Means |
|----------|------------------|---------------|
| **Random** | 70-80% | Should dominate random play |
| **Always-Call** | 75-85% | Should punish passive play |
| **Baseline TAG** | 60-70% | Should beat simple strategy |
| **Iteration 1** | 60-70% | Should improve significantly |
| **Iteration 10** | 55-65% | Should show clear improvement |

**Interpretation:**
- ✅ **Dominating weak opponents** (70%+) = Model learned poker fundamentals
- ✅ **Beating early iterations** (55%+) = Model improved over training
- ✅ **Consistent edge** across all = Robust strategy
- ❌ **<60% vs weak opponents** = Model needs more training
- ❌ **<50% vs early iterations** = Model got worse (bug!)

---

## 📊 3. Complete Evaluation Workflow

### Step 1: Train
```bash
modal run modal_deploy/train_exploitative_cfr.py::main \
  --gto-checkpoint checkpoint_iter_19.pt \
  --num-iterations 1000 \
  --trajectories-per-iteration 1000
```

### Step 2: Plot Training Curves
```bash
# Download and plot all metrics
python3 scripts/plot_training_curves.py
```

Check:
- ✅ Losses decreasing
- ✅ Payoff increasing
- ✅ Win rate > 50%

### Step 3: Download Model
```bash
modal volume get poker-exploitative-cfr-checkpoints \
  exploitative_cfr_final.pt \
  ./exploitative_cfr_final.pt
```

### Step 4: Benchmark Against Baselines
```bash
python3 scripts/evaluate_benchmarks.py \
  --current 1000 \
  --num-games 2000
```

Expected output:
```
================================================================================
BENCHMARK EVALUATION
================================================================================
Current iteration: 1000
Games per matchup: 2000

Evaluating vs Random Agent...
--------------------------------------------------------------------------------
  Win Rate: 75.50%
  Avg Payoff: 1250.25 chips/game
  ✓✓ Excellent! Strongly dominating (>70% win rate)

Evaluating vs Always Call Agent...
--------------------------------------------------------------------------------
  Win Rate: 79.25%
  Avg Payoff: 1450.75 chips/game
  ✓✓ Excellent! Strongly dominating (>70% win rate)

... (etc)
```

### Step 5: Compare to GTO (Optional)
```bash
# Play 10,000 hands against the GTO checkpoint
# (This will show if you're actually exploiting GTO or just lucky)
```

---

## 🔍 4. Interpreting Combined Results

### Scenario A: Successful Training ✅

**Training Curves:**
- Losses: Steadily decreasing
- Payoff: +150 chips/hand, upward trend
- Win Rate: 56%, upward trend

**Baselines:**
- vs Random: 75% win rate
- vs Always-Call: 80% win rate
- vs Baseline: 65% win rate
- vs Early iterations: 60% win rate

**Conclusion:** 🎉 Model learned strong exploitative strategy!

---

### Scenario B: Overfit to GTO ⚠️

**Training Curves:**
- Losses: Decreasing (good)
- Payoff: +200 chips/hand vs GTO
- Win Rate: 58% vs GTO

**Baselines:**
- vs Random: 52% win rate ❌
- vs Always-Call: 55% win rate ❌
- vs Baseline: 48% win rate ❌

**Conclusion:** Model overfit to the specific GTO opponent. Not generalizing to other strategies. Need more diverse training opponents.

---

### Scenario C: Needs More Training ⚠️

**Training Curves:**
- Losses: Still high (not converged)
- Payoff: +30 chips/hand, noisy
- Win Rate: 51%, noisy

**Baselines:**
- vs Random: 60% win rate (okay)
- vs Always-Call: 65% win rate (okay)
- vs Baseline: 52% win rate (marginal)

**Conclusion:** Training working but needs more iterations. Run 500-1000 more.

---

### Scenario D: Something Wrong ❌

**Training Curves:**
- Losses: Increasing or flat
- Payoff: Negative or zero
- Win Rate: <50%

**Baselines:**
- vs Random: <55% ❌
- vs Always-Call: <60% ❌

**Conclusion:** Bug in training! Check:
- GTO checkpoint loaded correctly?
- Networks updating?
- Learning rates reasonable?

---

## 📈 5. Quick Reference Commands

```bash
# 1. Plot all training curves (NEW!)
python3 scripts/plot_training_curves.py

# 2. Plot just payoff/winrate (original)
python3 scripts/plot_exploitative_metrics.py

# 3. Benchmark against baselines
python3 scripts/evaluate_benchmarks.py --current 1000 --num-games 2000

# 4. Download model from Modal
modal volume get poker-exploitative-cfr-checkpoints \
  exploitative_cfr_final.pt ./exploitative_cfr_final.pt

# 5. Check what's in Modal volume
modal volume ls poker-exploitative-cfr-checkpoints
```

---

## 💡 Pro Tips

1. **Loss curves are your canary** - If losses aren't decreasing, payoff won't improve
2. **Baseline evaluation is crucial** - Good vs GTO doesn't mean good at poker!
3. **Compare multiple checkpoints** - Test iter 100, 500, 1000 to see improvement
4. **High variance is normal** - That's why we average over many games
5. **Trust the trends, not single numbers** - Look at moving averages

---

## 🚀 Next Run Will Include Loss Tracking!

Your next training run will automatically track losses and you can plot them with:
```bash
python3 scripts/plot_training_curves.py
```

Ready to run full training with loss tracking? 🎯
