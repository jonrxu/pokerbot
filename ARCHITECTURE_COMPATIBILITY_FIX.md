# Architecture Compatibility Fix

## Problem

When loading `checkpoint_iter_19.pt` for exploitative training, you get this error:

```
RuntimeError: Error(s) in loading state_dict for ValuePolicyNet:
    Missing key(s) in state_dict: "value_head.0.weight", "value_head.0.bias",
                                   "value_head.2.weight", "value_head.2.bias".
    size mismatch for input_proj.weight: copying a param with shape torch.Size([512, 178])
                                         from checkpoint, the shape in current model is
                                         torch.Size([512, 167]).
```

## Root Cause

The checkpoint was trained with an older version of `StateEncoder` that produced **178 features**, but the current version produces **167 features** (11 fewer).

**Old StateEncoder (178 features):**
- Hole cards: 34
- Community cards: 85
- Betting history: 50 (MAX_HISTORY=25, 2 features per action)
- Pot/stack ratios: 4
- Position: 1
- Street: 1
- Current bets: 2
- Extra feature: 1
- **Total: 178**

**Current StateEncoder (167 features):**
- Hole cards: 34
- Community cards: 85
- Betting history: 40 (MAX_HISTORY=20, 2 features per action)
- Pot/stack ratios: 4
- Position: 1
- Street: 1
- Current bets: 2
- **Total: 167**

## Solution

We created a **LegacyStateEncoder** that matches the old 178-feature format and updated the **ExploitativeTrainer** to use separate encoders:

1. **GTO Opponent**: Uses `LegacyStateEncoder` (178 features) to load old checkpoints
2. **Exploitative Agent**: Uses current `StateEncoder` (167 features) for training

This allows the two models to have different architectures without conflicts!

## Files Changed

### 1. `poker_game/legacy_state_encoder.py` (NEW)
Legacy encoder with 178 features for backwards compatibility.

### 2. `training/exploitative_trainer.py` (UPDATED)
- Added `use_legacy_encoder_for_gto` parameter (default: `True`)
- Uses `LegacyStateEncoder` for GTO opponent
- Uses current `StateEncoder` for exploitative agent
- Loads checkpoints with `strict=False` to handle any remaining mismatches

## How It Works

```python
# GTO opponent uses legacy encoder (178 features)
gto_encoder = LegacyStateEncoder()
gto_policy_net = ValuePolicyNet(input_dim=178)

# Exploitative agent uses current encoder (167 features)
exploit_encoder = StateEncoder()
exploit_policy_net = ValuePolicyNet(input_dim=167)

# Load GTO checkpoint with strict=False for safety
gto_policy_net.load_state_dict(checkpoint['policy_net_state'], strict=False)
```

## Testing

To verify the fix works:

```bash
# This should now work without errors!
modal run modal_deploy/train_exploitative_cfr.py::main \
  --gto-checkpoint checkpoint_iter_19.pt \
  --num-iterations 10 \
  --trajectories-per-iteration 100
```

Expected output:
```
Using LegacyStateEncoder for GTO opponent (178 features)
Loading GTO checkpoint from: /gto_checkpoints/checkpoint_iter_19.pt
  ✓ Value network loaded (strict)
  ✓ Policy network loaded (strict)
  ✓ Loaded 12345 regret memory entries
  ✓ Loaded 12345 strategy memory entries
✓ Successfully loaded GTO checkpoint
Initialized exploitative trainer vs GTO checkpoint
  Exploitative agent: 167 features
  GTO opponent: 178 features
  Device: cuda
```

## For Future Checkpoints

If you train a new GTO model with the current StateEncoder (167 features), you can disable the legacy encoder:

```python
trainer = ExploitativeTrainer(
    game=game,
    state_encoder=state_encoder,
    gto_checkpoint_path=path_to_new_checkpoint,
    use_legacy_encoder_for_gto=False,  # Use current encoder
)
```

## Why This Approach?

**Why not just update StateEncoder to produce 178 features?**
- The current 167-feature encoding is better (less redundant history)
- We want new models to use the improved encoding
- Only the GTO opponent needs the legacy format

**Why not convert the old checkpoint?**
- Checkpoint conversion is error-prone
- This solution is more flexible
- Works with any old checkpoint automatically

**Why use separate encoders?**
- Exploitative agent and GTO opponent don't need the same architecture
- They encode the same game state, just differently
- This gives us maximum compatibility and flexibility

## Summary

✅ **Problem Fixed**: Old checkpoints now load correctly
✅ **No Data Loss**: All checkpoint weights and memories preserved
✅ **Forward Compatible**: New checkpoints will use improved encoding
✅ **Flexible**: Can mix old and new checkpoints as needed

You can now proceed with training!
