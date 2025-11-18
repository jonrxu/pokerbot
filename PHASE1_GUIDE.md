# Phase 1: Nash Equilibrium Training Guide

## Overview

This guide covers starting Phase 1 training - building a strong Nash equilibrium agent using Deep CFR.

## Quick Start

### 1. Start Training Job

```bash
# Basic usage (default: 1000 iterations, 10000 trajectories, 4 workers, batch size 32)
./scripts/start_training.sh

# Custom configuration
./scripts/start_training.sh 1000 10000 4 "" 32
# Arguments: [iterations] [trajectories] [workers] [resume_from] [batch_size]

# Resume from checkpoint
./scripts/start_training.sh 1000 10000 4 500 32
```

**The job runs asynchronously** - you can close the terminal after submission!

### 2. Monitor Progress

```bash
# Check status and download metrics
./scripts/check_status.sh

# View metrics summary
python scripts/view_metrics.py

# View logs directly
modal app logs poker-bot-training --tail 50
```

### 3. Stop Training (if needed)

```bash
./scripts/stop_training.sh
```

## Metrics Tracked

All metrics are logged to `/checkpoints/metrics/` in the Modal volume:

- **training_metrics.jsonl**: Line-by-line metrics for each iteration
  - Iteration number
  - Value loss
  - Policy loss
  - Trajectories generated
  - Checkpoint paths
  - Timestamps

- **summary.json**: Current summary statistics
  - Latest iteration
  - Best losses
  - Total iterations completed

- **training.log**: Detailed training logs
  - Worker progress
  - Training updates
  - Errors and warnings

## Observability

### Real-time Monitoring

1. **Modal Dashboard**: https://modal.com/apps
   - See active functions
   - Monitor resource usage
   - View function logs

2. **Downloaded Metrics**: `./local_metrics/`
   - Run `./scripts/check_status.sh` to download
   - View with `python scripts/view_metrics.py`

3. **Training Logs**: `./local_logs/training.log`
   - Detailed iteration-by-iteration progress
   - Worker completion status
   - Training statistics

### Key Metrics to Watch

- **Value Loss**: Should decrease over time (converging to true values)
- **Policy Loss**: Should stabilize (policy converging)
- **Trajectories Generated**: Should match expected count
- **Checkpoint Frequency**: Every 100 iterations

## Expected Training Timeline

For 1000 iterations with 10,000 trajectories/iteration:

- **Per iteration**: ~5-10 minutes
  - Trajectory generation: ~3-5 min (4 workers)
  - Network training: ~2-5 min (GPU)
  
- **Total time**: ~80-160 hours
- **Cost estimate**: ~$8-16 (depending on GPU usage)

## Troubleshooting

### Job Not Starting
- Check Modal authentication: `modal token get`
- Verify volume exists: `modal volume list`
- Check logs: `modal app logs poker-bot-training`

### Metrics Not Updating
- Metrics are saved every iteration
- Run `./scripts/check_status.sh` to download latest
- Check if training is actually running in Modal dashboard

### Out of Memory
- Reduce batch size: `--batch-size 16`
- Reduce trajectories per iteration
- Reduce number of workers

### Training Too Slow
- Increase number of workers (up to available quota)
- Use more powerful GPU (A10G instead of T4)
- Reduce trajectories per iteration for faster iterations

## Next Steps After Phase 1

Once Phase 1 completes:

1. Evaluate agent performance
2. Check exploitability metrics
3. Compare against baseline
4. Proceed to Phase 2: Multi-agent evolution

See `TRAINING_PLAN.md` for details.

