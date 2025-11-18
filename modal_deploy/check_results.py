"""Check results from async training job."""

import modal
import json
from modal_deploy.config import checkpoint_volume

app = modal.App("poker-bot-training")

@app.function(
    image=modal.Image.debian_slim(python_version="3.10"),
    volumes={"/checkpoints": checkpoint_volume},
    timeout=30,
)
def check_training_results():
    """Check if training results exist."""
    import os
    
    results = {
        "metrics_exists": os.path.exists("/checkpoints/metrics/training_metrics.jsonl"),
        "summary_exists": os.path.exists("/checkpoints/metrics/summary.json"),
        "log_exists": os.path.exists("/checkpoints/training.log"),
        "checkpoints": []
    }
    
    if os.path.exists("/checkpoints"):
        checkpoints = [f for f in os.listdir("/checkpoints") if f.startswith("checkpoint_")]
        results["checkpoints"] = sorted(checkpoints)
    
    if results["summary_exists"]:
        with open("/checkpoints/metrics/summary.json") as f:
            results["summary"] = json.load(f)
    
    if results["metrics_exists"]:
        with open("/checkpoints/metrics/training_metrics.jsonl") as f:
            lines = f.readlines()
            results["num_metrics_entries"] = len(lines)
            if lines:
                results["last_entry"] = json.loads(lines[-1])
                results["all_entries"] = [json.loads(line) for line in lines]
    
    # Read last 30 lines of log if it exists
    if results["log_exists"]:
        with open("/checkpoints/training.log") as f:
            all_lines = f.readlines()
            results["log_tail"] = "".join(all_lines[-30:])
            results["total_log_lines"] = len(all_lines)
    
    return results

if __name__ == "__main__":
    with app.run():
        result = check_training_results.remote()
        print(json.dumps(result, indent=2))

