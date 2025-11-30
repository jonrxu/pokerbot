# Modal-Only Exploitative LLM Training (No Tinker!)

**Train an exploitative poker LLM entirely on Modal - no external APIs, no Tinker costs!**

---

## ✨ What's Different?

| Feature | Tinker Version | Modal-Only Version |
|---------|----------------|-------------------|
| **LLM Hosting** | Tinker API | HuggingFace on Modal GPU |
| **Training** | Tinker LoRA | PyTorch Policy Gradient |
| **Cost** | Modal + Tinker API | Modal only |
| **External APIs** | Required | None |
| **Setup** | API key needed | Just Modal |

**Result: Simpler, cheaper, fully self-contained!**

---

## 🚀 Quick Start (3 Steps)

### Step 1: Setup Modal

```bash
# Install Modal
pip install modal

# Authenticate
modal setup
```

### Step 2: Upload GTO Checkpoint

```bash
# Upload your trained GTO checkpoint to Modal
modal volume put poker-bot-checkpoints \
  checkpoints/checkpoint_1000.pt \
  checkpoint_1000.pt
```

### Step 3: Launch Training

```bash
# Run training (entirely on Modal - no external APIs!)
modal run modal_deploy/train_exploitative_modal_only.py::main \
  --gto-checkpoint checkpoint_1000.pt \
  --num-epochs 20 \
  --episodes-per-epoch 32
```

**That's it!** Training runs on Modal GPU. No API keys, no external services.

---

## 📊 What Happens During Training

```
1. Modal provisions T4 GPU instance
2. Downloads Qwen2.5-3B-Instruct from HuggingFace
3. Loads your GTO checkpoint as opponent
4. For each epoch:
   - Collects 32 episodes (LLM vs GTO)
   - Calculates policy gradients
   - Updates LLM weights
   - Saves checkpoint every 5 epochs
5. Saves final trained model
6. Shuts down automatically
```

**Training time:** ~2-4 hours for 20 epochs (32 episodes each)

---

## 💰 Cost Estimate

**Modal GPU pricing (T4):**
- ~$0.60/hour
- 20 epochs × 32 episodes ≈ 3-4 hours
- **Total: ~$2-3**

**No Tinker API costs!** 🎉

Compare to Tinker version: ~$20-30 total (Modal + Tinker API)

---

## 📥 Download Trained Model

After training completes:

```bash
# List checkpoints
modal volume ls poker-exploitative-modal-checkpoints

# Download final model
modal volume get poker-exploitative-modal-checkpoints \
  exploitative_model_final \
  ./my_exploitative_llm/

# Or download a specific epoch
modal volume get poker-exploitative-modal-checkpoints \
  exploitative_model_epoch_020 \
  ./my_model_epoch_20/
```

---

## 🎛️ Configuration Options

### Basic Usage

```bash
modal run modal_deploy/train_exploitative_modal_only.py::main \
  --gto-checkpoint checkpoint_1000.pt \
  --num-epochs 20 \
  --episodes-per-epoch 32
```

### All Options

```bash
modal run modal_deploy/train_exploitative_modal_only.py::main \
  --model-name "Qwen/Qwen2.5-3B-Instruct" \
  --gto-checkpoint checkpoint_1000.pt \
  --num-epochs 50 \
  --episodes-per-epoch 64 \
  --learning-rate 1e-5
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-name` | `Qwen/Qwen2.5-3B-Instruct` | HuggingFace model to use |
| `--gto-checkpoint` | Required | Your GTO checkpoint name |
| `--num-epochs` | 20 | Number of training epochs |
| `--episodes-per-epoch` | 32 | Episodes per epoch |
| `--learning-rate` | 1e-5 | Learning rate |

---

## 🎯 Expected Performance

| Epochs | Avg Return | Win Rate | Status |
|--------|------------|----------|--------|
| 0 | ~0.0 | ~50% | Baseline (random) |
| 5 | +0.05 to +0.10 | 51-53% | Learning basics |
| 10 | +0.10 to +0.20 | 53-56% | Starting to exploit |
| 20 | +0.20 to +0.40 | 56-60% | Good exploitation |
| 50+ | +0.40 to +0.60 | 60-65% | Strong exploitation |

---

## 🔧 Architecture Details

### Components

1. **PokerLLMAgent** (`llm_poker/llm_agent.py`)
   - Loads HuggingFace causal LM (Qwen 2.5)
   - Generates actions from text prompts
   - Runs on Modal GPU

2. **PolicyGradientTrainer** (`llm_poker/policy_gradient.py`)
   - REINFORCE algorithm
   - Computes discounted returns
   - Updates policy to favor high-reward actions

3. **ExploitativePokerEnv** (`llm_poker/exploitative_env.py`)
   - Same reward system as Tinker version
   - Bluff, aggression, fold equity bonuses
   - GTO opponent from Deep CFR

### Training Algorithm (REINFORCE)

```
For each epoch:
  1. Collect episodes with current policy:
     - LLM generates actions from prompts
     - Track (prompt, action, reward) tuples

  2. Compute returns for each step:
     - G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...

  3. Update policy (gradient ascent):
     - ∇_θ J ≈ Σ G_t * ∇_θ log π_θ(a_t|s_t)
     - Increase probability of high-return actions

  4. Save checkpoint and repeat
```

---

## 🎮 Using Your Trained Model

### Load and Play

```python
from llm_poker.llm_agent import PokerLLMAgent

# Load trained model
agent = PokerLLMAgent.load("./my_exploitative_llm/")

# Play a hand
prompt = """You are playing heads-up no-limit Texas Hold'em.
Blinds: small blind 50, big blind 100.
Street: Flop
Board cards: Kh 7d 2c
Your hole cards: As Ah
Pot size: 350

Legal actions:
- A0: Check (no chips added).
- A1: Bet 175 into the pot.
- A2: Bet 350 into the pot.

Respond with exactly ONE action token from the list above."""

action = agent.get_action_token(prompt)
print(f"LLM chose: {action}")
```

---

## 🆘 Troubleshooting

### Issue: "GTO checkpoint not found"

```bash
# Check what's in your volume
modal volume ls poker-bot-checkpoints

# Upload checkpoint
modal volume put poker-bot-checkpoints \
  checkpoints/YOUR_CHECKPOINT.pt \
  YOUR_CHECKPOINT.pt
```

### Issue: "Out of memory"

**Solution:** Use smaller batch size:
```bash
modal run ... --episodes-per-epoch 16
```

Or use smaller model:
```bash
modal run ... --model-name "Qwen/Qwen2.5-1.5B-Instruct"
```

### Issue: Training not improving

**Try:**
1. **More episodes:** `--episodes-per-epoch 64`
2. **Higher learning rate:** `--learning-rate 5e-5`
3. **More epochs:** `--num-epochs 50`

### Issue: Win rate < 50%

**Possible causes:**
1. GTO checkpoint too strong - need more training
2. Learning rate too high - try `--learning-rate 5e-6`
3. Model too small - try larger Qwen model

---

## 📈 Monitoring Training

Training outputs real-time metrics:

```
================================================================================
EPOCH 10/20
================================================================================
Collecting 32 episodes...
100%|████████████████████████████████████████| 32/32 [02:15<00:00,  4.24s/it]

Episode Stats:
  Avg Return: +0.1845
  Win Rate: 56.25%
  Wins: 18, Losses: 14

Training on collected episodes...

Training Metrics:
  Loss: 0.0234
  Num Steps: 128

Saving checkpoint to /checkpoints/exploitative_epoch_010.pt
  ⭐ New best avg return: 0.1845
```

**Key metrics to watch:**
- **Avg Return** - Should increase over time (target: >0.2)
- **Win Rate** - Should be >55% (exploiting GTO)
- **Loss** - Should decrease initially, then stabilize

---

## 🔬 Advanced: Customizing Rewards

Edit `modal_deploy/train_exploitative_modal_only.py` to change reward bonuses:

```python
# Ultra-aggressive bluffer
train_exploitative_modal.remote(
    bluff_bonus=1.0,           # Big bluff rewards
    aggression_bonus=0.3,       # Lots of aggression
    fold_equity_bonus=0.5,      # Love making them fold
    exploitative_sizing_bonus=0.4,  # Crazy bet sizes
)
```

```python
# Subtle exploiter
train_exploitative_modal.remote(
    bluff_bonus=0.2,           # Occasional bluffs
    aggression_bonus=0.05,      # Selective aggression
    fold_equity_bonus=0.2,      # Some fold equity
    exploitative_sizing_bonus=0.8,  # Focus on sizing tells
)
```

---

## 🎯 Next Steps

1. ✅ **Train initial model** (20 epochs)
2. ✅ **Evaluate performance** (check win rate)
3. ✅ **Iterate on rewards** if needed
4. ✅ **Train longer** (50+ epochs) for production
5. ✅ **Download and deploy** your exploitative bot!

---

## 💡 Tips for Best Results

1. **Start small** - Test with 5 epochs first
2. **Monitor metrics** - Watch for improving win rate
3. **Adjust learning rate** - If loss explodes, reduce LR
4. **Train longer** - 50+ epochs for best performance
5. **Save money** - Use smaller model (1.5B) for testing

---

## 📊 Comparison: Tinker vs Modal-Only

| Aspect | Tinker Version | Modal-Only |
|--------|---------------|------------|
| **Setup complexity** | Medium (API key) | Easy (just Modal) |
| **Cost (20 epochs)** | ~$20-30 | ~$2-3 |
| **Training speed** | Fast (optimized) | Medium |
| **Customization** | Limited | Full control |
| **Dependencies** | Tinker API | Just transformers |
| **Reliability** | Depends on API | Fully self-contained |

**Verdict:** Modal-only is simpler, cheaper, and gives you full control!

---

## 🎉 Summary

**Before (Tinker):**
```bash
export TINKER_API_KEY="..."
python scripts/train_exploitative_llm.py --gto-checkpoint ...
# Cost: $20-30
```

**After (Modal-only):**
```bash
modal run modal_deploy/train_exploitative_modal_only.py::main \
  --gto-checkpoint checkpoint_1000.pt
# Cost: $2-3, no API needed!
```

**Simpler. Cheaper. Self-contained. 🚀**

---

Questions? Check the troubleshooting section or review the code in:
- `llm_poker/llm_agent.py` - LLM wrapper
- `llm_poker/policy_gradient.py` - Training algorithm
- `modal_deploy/train_exploitative_modal_only.py` - Modal deployment
