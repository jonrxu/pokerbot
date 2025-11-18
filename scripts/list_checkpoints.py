#!/usr/bin/env python3
"""List all checkpoints on Modal."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modal_deploy.config import checkpoint_volume, image
import modal

app = modal.App("list-checkpoints")

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=60,
)
def list_checkpoints():
    import os
    import glob
    
    checkpoint_dir = "/checkpoints"
    checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint_iter_*.pt")
    checkpoints.sort(key=lambda x: int(x.split("_iter_")[1].split(".pt")[0]) if x.split("_iter_")[1].split(".pt")[0].isdigit() else 0)
    
    print("Available checkpoints on Modal:")
    print("-" * 80)
    for cp in checkpoints:
        size_mb = os.path.getsize(cp) / (1024*1024)
        iter_num = cp.split("_iter_")[1].split(".pt")[0]
        print(f"  checkpoint_iter_{iter_num}.pt ({size_mb:.1f} MB)")
    
    return [cp.split("_iter_")[1].split(".pt")[0] for cp in checkpoints]

if __name__ == "__main__":
    with app.run():
        list_checkpoints.remote()

