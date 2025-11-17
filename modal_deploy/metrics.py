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
        
        # Append to JSONL file with error handling
        try:
            # Simple append - JSONL files are append-only so corruption risk is minimal
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            # If write fails, just keep in memory - don't fail training
            pass
        
        # Keep in memory
        self.metrics_history.append(log_entry)
        
        # Update summary with error handling
        try:
            self._update_summary()
        except Exception as e:
            # Don't fail if summary update fails
            pass
    
    def _update_summary(self):
        """Update summary statistics."""
        if not self.metrics_history:
            return
        
        latest = self.metrics_history[-1]
        
        # Calculate best losses, handling NaN/Inf
        value_losses = [m.get("value_loss") for m in self.metrics_history if "value_loss" in m and isinstance(m.get("value_loss"), (int, float))]
        policy_losses = [m.get("policy_loss") for m in self.metrics_history if "policy_loss" in m and isinstance(m.get("policy_loss"), (int, float))]
        
        # Filter out NaN/Inf
        value_losses = [v for v in value_losses if v == v and v != float('inf') and v != float('-inf')]
        policy_losses = [v for v in policy_losses if v == v and v != float('inf') and v != float('-inf')]
        
        summary = {
            "latest_iteration": latest["iteration"],
            "total_iterations": len(self.metrics_history),
            "latest_metrics": latest,
            "best_value_loss": min(value_losses) if value_losses else float("inf"),
            "best_policy_loss": min(policy_losses) if policy_losses else float("inf"),
            "last_updated": datetime.now().isoformat()
        }
        
        # Write to temp file first, then rename for atomicity
        temp_file = self.summary_file + ".tmp"
        try:
            with open(temp_file, "w") as f:
                json.dump(summary, f, indent=2)
            os.rename(temp_file, self.summary_file)
        except Exception as e:
            # Fallback: direct write
            try:
                with open(self.summary_file, "w") as f:
                    json.dump(summary, f, indent=2)
            except:
                pass
    
    def get_summary(self) -> Dict:
        """Get current summary."""
        if os.path.exists(self.summary_file):
            with open(self.summary_file, "r") as f:
                return json.load(f)
        return {}
    
    def get_latest_metrics(self, n: int = 10) -> List[Dict]:
        """Get latest N metrics."""
        return self.metrics_history[-n:] if self.metrics_history else []

