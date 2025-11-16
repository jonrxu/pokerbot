"""Entry point for Modal training."""

import modal

# Import the Modal app
from modal.train import app

if __name__ == "__main__":
    # Run training via Modal
    with app.run():
        # This will be called via: modal run modal_train.py
        pass

