"""Entry point for Modal training.

Usage:
    modal run modal_train.py::app.main --num-iterations 1000 --trajectories-per-iteration 10000 --num-workers 4
"""

# Import modules so Modal uploads them
import poker_game
import models
import training
import evaluation
import checkpoints

# Import the Modal app
from modal_deploy.train import app

if __name__ == "__main__":
    pass

