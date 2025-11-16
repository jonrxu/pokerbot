# Deep CFR Poker Bot

A heads-up poker bot trained using Deep Counterfactual Regret Minimization (Deep CFR) with neural networks, trained via distributed self-play on Modal.

## Features

- Deep CFR algorithm for Nash equilibrium play
- Multi-agent self-play training
- Distributed training on Modal
- Checkpointing and resume capability
- Evaluation and exploitability metrics

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Modal:
```bash
modal token set
```

3. Run training:
```bash
python -m modal.train
```

## Project Structure

- `poker_game/`: Game logic and state management
- `models/`: Neural network architectures
- `training/`: Deep CFR and training loops
- `modal/`: Modal deployment functions
- `checkpoints/`: Checkpoint management
- `evaluation/`: Evaluation and metrics

