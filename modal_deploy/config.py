"""Modal configuration for distributed training."""

import modal
import os

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Modal image with dependencies and local code
# Use add_local_dir to include code directories in the image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "treys>=0.1.8",
    )
    .add_local_dir(
        os.path.join(project_root, "poker_game"),
        "/root/poker_game"
    )
    .add_local_dir(
        os.path.join(project_root, "models"),
        "/root/models"
    )
    .add_local_dir(
        os.path.join(project_root, "training"),
        "/root/training"
    )
    .add_local_dir(
        os.path.join(project_root, "evaluation"),
        "/root/evaluation"
    )
    .add_local_dir(
        os.path.join(project_root, "checkpoints"),
        "/root/checkpoints"
    )
    .add_local_dir(
        os.path.join(project_root, "modal_deploy"),
        "/root/modal_deploy"
    )
)

# Volume for checkpoints and data
checkpoint_volume = modal.Volume.from_name("poker-bot-checkpoints", create_if_missing=True)

# GPU configuration
# Options: "T4" (cheaper, slower), "A10G" (2x faster), "A100" (4x faster, most expensive)
GPU_CONFIG = {
    "gpu": "A10G",  # Upgraded to A10G for 2x faster training
    "memory": 16384,  # 16GB RAM
}

CPU_CONFIG = {
    "cpu": 4,  # 4 CPU cores for trajectory generation
    "memory": 8192,  # 8GB RAM
}

# Training configuration
TRAINING_CONFIG = {
    "trajectories_per_worker": 1000,
    "batch_size": 64,  # Increased from 32 for faster GPU training
    "network_update_frequency": 10,
    "checkpoint_frequency": 100,
}

