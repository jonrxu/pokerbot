"""Modal configuration for distributed training."""

import modal

# Modal image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
    )
)

# Volume for checkpoints and data
checkpoint_volume = modal.Volume.from_name("poker-bot-checkpoints", create_if_missing=True)

# GPU configuration
GPU_CONFIG = {
    "gpu": "T4",  # Use T4 GPU for training
    "memory": 16384,  # 16GB RAM
}

CPU_CONFIG = {
    "cpu": 4,  # 4 CPU cores for trajectory generation
    "memory": 8192,  # 8GB RAM
}

# Training configuration
TRAINING_CONFIG = {
    "trajectories_per_worker": 1000,
    "batch_size": 32,
    "network_update_frequency": 10,
    "checkpoint_frequency": 100,
}

