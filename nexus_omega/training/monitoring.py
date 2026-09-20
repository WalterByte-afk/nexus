"""
Monitoring utilities for tracking training progress.

Handles metrics collection, visualization, and logging.
Keeps things simple and practical.
"""

import torch
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json
from pathlib import Path


class MetricsTracker:
    """
    Track and aggregate metrics during training.

    Collects metrics, computes running averages, and stores history.
    """

    def __init__(self):
        self.metrics = defaultdict(list)
        self.current_epoch_metrics = defaultdict(list)
        self.epoch = 0

    def update(self, metrics: Dict[str, float]):
        """Add new metric values."""
        for key, value in metrics.items():
            self.current_epoch_metrics[key].append(value)

    def batch_update(self, name: str, value: float):
        """Update a single metric for current batch."""
        self.current_epoch_metrics[name].append(value)

    def epoch_end(self) -> Dict[str, float]:
        """
        Compute epoch averages and store in history.

        Returns averaged metrics for the epoch.
        """
        epoch_metrics = {}

        for key, values in self.current_epoch_metrics.items():
            if values:
                avg = sum(values) / len(values)
                epoch_metrics[key] = avg
                self.metrics[key].append(avg)

        # Clear current epoch
        self.current_epoch_metrics = defaultdict(list)
        self.epoch += 1

        return epoch_metrics

    def get_history(self, metric_name: str) -> List[float]:
        """Get history of a specific metric."""
        return self.metrics.get(metric_name, [])

    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value of a metric."""
        history = self.get_history(metric_name)
        return history[-1] if history else None

    def get_best(self, metric_name: str, mode: str = 'min') -> Optional[float]:
        """Get best value of a metric."""
        history = self.get_history(metric_name)
        if not history:
            return None

        if mode == 'min':
            return min(history)
        else:
            return max(history)


class TrainingMonitor:
    """
    Comprehensive training monitor.

    Tracks metrics, estimates time remaining, and provides summary stats.
    """

    def __init__(self, log_dir: Optional[str] = None):
        self.tracker = MetricsTracker()
        self.log_dir = Path(log_dir) if log_dir else None
        self.start_time = None
        self.epoch_start_time = None
        self.batch_times = []

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def start_training(self):
        """Call at the start of training."""
        self.start_time = time.time()
        print("Training started...")

    def start_epoch(self, epoch: int, total_epochs: int):
        """Call at the start of each epoch."""
        self.epoch_start_time = time.time()
        self.tracker.epoch = epoch
        print(f"\nEpoch {epoch + 1}/{total_epochs}")

    def log_batch(self, batch_idx: int, total_batches: int, loss: float, lr: float = None):
        """Log batch metrics."""
        batch_start = time.time()

        # Store loss
        self.tracker.batch_update('loss', loss)
        if lr is not None:
            self.tracker.batch_update('learning_rate', lr)

        # Progress display
        progress = (batch_idx + 1) / total_batches
        bar_len = 30
        filled = int(bar_len * progress)
        bar = '=' * filled + '-' * (bar_len - filled)

        avg_loss = sum(self.tracker.current_epoch_metrics['loss']) / len(self.tracker.current_epoch_metrics['loss'])

        print(f"\r  [{bar}] {batch_idx + 1}/{total_batches} - loss: {avg_loss:.6f}", end='')

    def end_epoch(self) -> Dict[str, float]:
        """Call at the end of each epoch."""
        elapsed = time.time() - self.epoch_start_time

        # Get averaged metrics
        metrics = self.tracker.epoch_end()
        metrics['epoch_time'] = elapsed

        # Print summary
        print(f"\n  Epoch completed in {elapsed:.2f}s")
        for key, value in metrics.items():
            if key != 'epoch_time':
                print(f"  {key}: {value:.6f}")

        # Save to log if enabled
        if self.log_dir:
            self._save_epoch_log(metrics)

        return metrics

    def end_training(self):
        """Call at the end of training."""
        total_time = time.time() - self.start_time
        print(f"\nTraining completed in {total_time:.2f}s ({total_time/60:.2f}m)")

        # Print final summary
        print("\nFinal metrics:")
        for key, values in self.tracker.metrics.items():
            if values:
                print(f"  {key}: {values[-1]:.6f} (best: {min(values) if 'loss' in key else max(values):.6f})")

    def _save_epoch_log(self, metrics: Dict[str, float]):
        """Save epoch metrics to log file."""
        log_file = self.log_dir / 'metrics.jsonl'
        entry = {
            'timestamp': time.time(),
            **metrics,
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')


class GradientMonitor:
    """
    Monitor gradient statistics during training.

    Useful for debugging training issues.
    """

    def __init__(self):
        self.grad_stats = defaultdict(list)

    def track_gradients(self, model: torch.nn.Module):
        """Track gradient statistics for a model."""
        total_norm = 0.0
        param_count = 0
        max_grad = 0.0
        min_grad = float('inf')

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1

                param_max = param.grad.data.max().item()
                param_min = param.grad.data.min().item()

                max_grad = max(max_grad, param_max)
                min_grad = min(min_grad, param_min)

        total_norm = total_norm ** 0.5

        stats = {
            'grad_norm': total_norm,
            'max_grad': max_grad,
            'min_grad': min_grad,
            'param_count': param_count,
        }

        for key, value in stats.items():
            self.grad_stats[key].append(value)

        return stats

    def get_stats(self) -> Dict[str, List[float]]:
        """Get all tracked gradient statistics."""
        return dict(self.grad_stats)


class MemoryMonitor:
    """
    Monitor GPU memory usage during training.

    Helps identify memory bottlenecks.
    """

    def __init__(self):
        self.memory_stats = defaultdict(list)

    def track_memory(self):
        """Track current memory usage."""
        stats = {}

        if torch.cuda.is_available():
            stats['allocated_gb'] = torch.cuda.memory_allocated() / 1024 ** 3
            stats['cached_gb'] = torch.cuda.memory_reserved() / 1024 ** 3
            stats['max_allocated_gb'] = torch.cuda.max_memory_allocated() / 1024 ** 3

            for key, value in stats.items():
                self.memory_stats[key].append(value)

        return stats

    def reset_peak_stats(self):
        """Reset peak memory stats."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def get_summary(self) -> Dict[str, float]:
        """Get memory summary."""
        if not self.memory_stats['allocated_gb']:
            return {}

        return {
            'peak_allocated_gb': max(self.memory_stats['allocated_gb']),
            'avg_allocated_gb': sum(self.memory_stats['allocated_gb']) / len(self.memory_stats['allocated_gb']),
            'peak_cached_gb': max(self.memory_stats['cached_gb']) if self.memory_stats['cached_gb'] else 0,
        }


if __name__ == "__main__":
    print("Testing monitoring utilities...")

    # Test metrics tracker
    tracker = MetricsTracker()

    # Simulate training
    for epoch in range(3):
        for batch in range(5):
            loss = 1.0 / (epoch + 1) + 0.1 * (batch / 5)
            tracker.batch_update('loss', loss)

        metrics = tracker.epoch_end()
        print(f"Epoch {epoch}: {metrics}")

    print(f"\nLoss history: {tracker.get_history('loss')}")
    print(f"Best loss: {tracker.get_best('loss', mode='min')}")

    # Test training monitor
    monitor = TrainingMonitor(log_dir='test_logs')
    monitor.start_training()

    for epoch in range(2):
        monitor.start_epoch(epoch, 2)
        for batch in range(10):
            loss = 0.5 / (epoch + 1)
            monitor.log_batch(batch, 10, loss)
        monitor.end_epoch()

    monitor.end_training()

    print("\nAll monitoring tests passed!")
