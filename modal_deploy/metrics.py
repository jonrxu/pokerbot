"""Metrics tracking and logging for training."""

import json
import os
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict


class MetricsLogger:
    """Logs training metrics to files for observability."""
    
    def __init__(self, log_dir: str = "/checkpoints/metrics"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_file = os.path.join(log_dir, "training_metrics.jsonl")
        self.summary_file = os.path.join(log_dir, "summary.json")
        self.metrics_history: List[Dict] = []
    
    def log_iteration(self, iteration: int, metrics: Dict[str, Any]):
        """Log metrics for a single iteration."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            **metrics
        }
        
        # Append to JSONL file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Keep in memory
        self.metrics_history.append(log_entry)
        
        # Update summary
        self._update_summary()
    
    def _update_summary(self):
        """Update summary statistics."""
        if not self.metrics_history:
            return
        
        latest = self.metrics_history[-1]
        summary = {
            "latest_iteration": latest["iteration"],
            "total_iterations": len(self.metrics_history),
            "latest_metrics": latest,
            "best_value_loss": min(m.get("value_loss", float("inf")) for m in self.metrics_history if "value_loss" in m),
            "best_policy_loss": min(m.get("policy_loss", float("inf")) for m in self.metrics_history if "policy_loss" in m),
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)
    
    def get_summary(self) -> Dict:
        """Get current summary."""
        if os.path.exists(self.summary_file):
            with open(self.summary_file, "r") as f:
                return json.load(f)
        return {}
    
    def get_latest_metrics(self, n: int = 10) -> List[Dict]:
        """Get latest N metrics."""
        return self.metrics_history[-n:] if self.metrics_history else []

