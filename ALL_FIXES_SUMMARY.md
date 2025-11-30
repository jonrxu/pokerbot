# Complete Fixes Summary

## All Errors Fixed

### 1. ✅ Architecture Mismatch (178 vs 167 features)
**Error:**
```
size mismatch for input_proj.weight: copying a param with shape torch.Size([512, 178])
from checkpoint, the shape in current model is torch.Size([512, 167])
```

**Fix:**
- Created `LegacyStateEncoder` with 178 features
- GTO opponent uses legacy encoder (178)
- Exploitative agent uses current encoder (167)

---

### 2. ✅ Missing `get_average_strategy` Method
**Error:**
```
AttributeError: 'DeepCFR' object has no attribute 'get_average_strategy'
```

**Fix:**
- Added `get_average_strategy()` to DeepCFR
- Added `get_action_strategy()` to DeepCFR
- Both return action-based dictionaries

---

### 3. ✅ Value Network Returns Tuple
**Error:**
```
AttributeError: 'tuple' object has no attribute 'item'
```

**Root Cause:**
- `value_net(state_tensor)` returns `(value, policy_logits)` not just value
- Code was calling `.item()` on a tuple

**Fix:**
```python
# Before:
predicted_value = self.exploitative_agent.value_net(state_tensor).item()

# After:
predicted_value, _ = self.exploitative_agent.value_net(state_tensor)
predicted_value = predicted_value.item()
```

---

### 4. ✅ Regret Memory Key Mismatch
**Error:**
- Code was using action tuples as keys instead of integer indices

**Fix:**
```python
# Before:
for action in legal_actions:
    self.exploitative_agent.regret_memory[info_set][action] += regret

# After:
for action_idx in range(len(legal_actions)):
    self.exploitative_agent.regret_memory[info_set.key][action_idx] += regret
```

---

### 5. ✅ Missing Training Methods
**Error:**
```
AttributeError: 'DeepCFR' object has no attribute 'train_value_network'
```

**Fix:**
- Added `train_value_network(value_buffer, batch_size)` to DeepCFR
- Added `train_policy_network(policy_buffer, batch_size)` to DeepCFR

---

## Files Modified

### New Files
1. **poker_game/legacy_state_encoder.py**
   - 178-feature encoder for backwards compatibility

### Modified Files

1. **training/deep_cfr.py**
   - Added `get_average_strategy()` - returns Dict[Tuple[Action, int], float]
   - Added `get_action_strategy()` - returns Dict[Tuple[Action, int], float]
   - Added `train_value_network()` - trains value network from buffer
   - Added `train_policy_network()` - trains policy network from buffer

2. **training/exploitative_trainer.py**
   - Uses `LegacyStateEncoder` for GTO opponent (178 features)
   - Uses `StateEncoder` for exploitative agent (167 features)
   - Fixed value network call to unpack tuple
   - Fixed regret memory updates to use integer indices
   - Uses `get_action_strategy()` for exploitative agent
   - Uses `get_average_strategy()` for GTO opponent

3. **modal_deploy/train_exploitative_cfr.py**
   - Saves metrics at each checkpoint

---

## Testing

Run the quick test:

```bash
modal run modal_deploy/train_exploitative_cfr.py::main \
  --gto-checkpoint checkpoint_iter_19.pt \
  --num-iterations 10 \
  --trajectories-per-iteration 100
```

**Expected output:**
```
Using LegacyStateEncoder for GTO opponent (178 features)
Loading GTO checkpoint from: /gto_checkpoints/checkpoint_iter_19.pt
  ⚠ Policy network architecture mismatch, loading with strict=False
    Missing keys: ['value_head.0.weight', ...]
✓ Successfully loaded GTO checkpoint
Initialized exploitative trainer vs GTO checkpoint
  Exploitative agent: 167 features
  GTO opponent: 178 features
  Device: cuda
================================================================================
Iteration 1/10
================================================================================
Avg Payoff: +XX.XX chips    ← Should be positive!
Win Rate: XX.XX%             ← Should be > 50%
Trajectories: 100
Value Buffer: XXXX
Policy Buffer: XXXX
================================================================================
Iteration 2/10
...
```

---

## What Success Looks Like

✅ **No errors during initialization**
✅ **GTO checkpoint loads (even with warnings)**
✅ **Training iterations complete**
✅ **Avg Payoff > 0** (beating GTO)
✅ **Win Rate > 50%** (winning more than losing)
✅ **Checkpoints save successfully**

---

## If You Still See Errors

1. **Missing keys warning** - This is OK! GTO opponent is frozen, so missing layers don't matter
2. **CUDA out of memory** - Reduce batch_size or trajectories_per_iteration
3. **Other errors** - Share the full error message

---

## Ready to Train!

All fixes are applied. The code should now run successfully! 🚀

Try it now:
```bash
modal run modal_deploy/train_exploitative_cfr.py::main \
  --gto-checkpoint checkpoint_iter_19.pt \
  --num-iterations 10 \
  --trajectories-per-iteration 100
```
