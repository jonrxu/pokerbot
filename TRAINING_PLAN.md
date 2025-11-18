# Poker Bot Training Plan

## Current Status

✅ **Completed:**
- Deep CFR implementation with neural networks
- Self-play trajectory generation
- Distributed training infrastructure on Modal
- Checkpointing and resume capability
- Basic evaluation framework

## Phase 1: Bootstrap Training (Nash Equilibrium)

**Goal:** Train a strong Nash-equilibrium agent using Deep CFR

### Steps:
1. **Initial Training Run**
   - Start with 1000 iterations
   - 10,000 trajectories per iteration
   - 4 parallel workers
   - Monitor convergence (exploitability should decrease over time)

2. **Hyperparameter Tuning**
   - Learning rate: Try 1e-4, 5e-5, 1e-5
   - Network architecture: Adjust hidden dimensions (256, 512, 1024)
   - Batch size: 32, 64, 128
   - Network update frequency: Every 10, 20, 50 iterations

3. **Convergence Criteria**
   - Exploitability < threshold (e.g., < 0.1 big blinds per hand)
   - Win rate stabilizes against baseline
   - Strategy consistency across multiple runs

**Command:**
```bash
conda activate poker_bot
modal run modal_train.py::app.main \
  --num-iterations 1000 \
  --trajectories-per-iteration 10000 \
  --num-workers 4
```

## Phase 2: Multi-Agent Evolution (Exploitative Strategies)

**Goal:** Train multiple agents with different strategies and evolve exploitative play

### Strategy:
1. **Create Agent Pool**
   - Train 3-5 agents with different:
     - Initialization seeds
     - Exploration rates
     - Network architectures
     - Training schedules

2. **Round-Robin Tournaments**
   - Agents play head-to-head matches
   - Track win rates and exploitability
   - Identify strongest strategies

3. **Evolutionary Selection**
   - Keep top-performing agents
   - Mutate/retrain weaker agents
   - Introduce new agents periodically

4. **Exploitative Adaptation**
   - Agents learn opponent-specific strategies
   - Adapt to exploit weaknesses
   - Develop meta-strategies

### Implementation:
- Extend `bootstrap.py` to create agent pool
- Add tournament evaluation in `evaluation/evaluator.py`
- Create evolutionary selection mechanism
- Implement opponent modeling

## Phase 3: Advanced Features

### 3.1 Exploitability Computation
- Implement proper exploitability calculation
- Use best-response computation
- Track exploitability over time

### 3.2 Action Masking
- Ensure illegal actions are properly masked
- Improve policy network training
- Better action probability distributions

### 3.3 Memory Optimization
- Compress regret memories for large games
- Use approximation techniques if needed
- Optimize checkpoint sizes

### 3.4 Evaluation Metrics
- Head-to-head win rates
- Exploitability scores
- Strategy visualization
- Action frequency analysis

## Training Schedule

### Week 1-2: Bootstrap Phase
- Run initial training (1000 iterations)
- Tune hyperparameters
- Achieve basic Nash equilibrium

### Week 3-4: Multi-Agent Phase
- Create agent pool
- Run tournaments
- Identify best strategies

### Week 5+: Exploitative Phase
- Implement opponent modeling
- Develop exploitative strategies
- Continuous improvement

## Cost Management

With $30 compute budget:
- **Modal Pricing:** ~$0.10/hour for T4 GPU, ~$0.05/hour for CPU
- **Estimated:** ~300 GPU hours or ~600 CPU hours
- **Strategy:** Use spot instances, efficient batching, checkpoint frequently

## Monitoring

Track:
- Training loss (value and policy)
- Exploitability over iterations
- Win rates in head-to-head matches
- Checkpoint sizes and save frequency
- Modal costs per iteration

## Next Steps

1. Run initial bootstrap training
2. Evaluate agent performance
3. Implement multi-agent pool
4. Begin evolutionary training
5. Develop exploitative strategies

