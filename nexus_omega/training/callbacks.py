"""
Training callbacks for monitoring and controlling the training process.

Handles things like early stopping, checkpointing, and metrics logging.
Nothing fancy, just practical utilities.
"""

import torch
import time
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import json


class Callback:
    """Base callback class."""

    def on_train_begin(self, trainer):
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, trainer):
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, batch_idx: int, trainer):
        """Called at the start of each batch."""
        pass

    def on_batch_end(self, batch_idx: int, loss: float, trainer):
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """
    Stop training when monitored metric stops improving.

    Monitors a metric (like validation loss) and stops if it doesn't
    improve for 'patience' epochs.
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        if self.monitor not in metrics:
            return

        current = metrics[self.monitor]

        if self.mode == 'min':
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best {self.monitor}: {self.best_value:.6f}")


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.

    Can save best model based on a metric, or save every N epochs.
    """

    def __init__(
        self,
        filepath: str,
        monitor: Optional[str] = None,
        mode: str = 'min',
        save_best_only: bool = True,
        save_freq: int = 1,
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq

        self.best_value = float('inf') if mode == 'min' else float('-inf')

        # Create directory if needed
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        # Check if we should save
        should_save = False

        if self.save_best_only and self.monitor:
            if self.monitor not in metrics:
                return

            current = metrics[self.monitor]

            if self.mode == 'min':
                improved = current < self.best_value
            else:
                improved = current > self.best_value

            if improved:
                self.best_value = current
                should_save = True
        else:
            # Save every N epochs
            if (epoch + 1) % self.save_freq == 0:
                should_save = True

        if should_save:
            self._save_checkpoint(epoch, metrics, trainer)

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], trainer):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'metrics': metrics,
        }

        filepath = str(self.filepath).replace('{epoch}', str(epoch))
        torch.save(checkpoint, filepath)
        print(f"\nCheckpoint saved: {filepath}")


class MetricsLogger(Callback):
    """
    Log metrics to a file during training.

    Saves metrics as JSONL for easy analysis later.
    """

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Clear old log
        if self.log_file.exists():
            self.log_file.unlink()

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        log_entry = {
            'epoch': epoch,
            'timestamp': time.time(),
            **metrics,
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


class LearningRateScheduler(Callback):
    """
    Adjust learning rate during training.

    Supports several common schedules.
    """

    def __init__(
        self,
        schedule: str = 'constant',
        initial_lr: float = 1e-4,
        warmup_epochs: int = 0,
        decay_rate: float = 0.1,
        decay_epochs: Optional[List[int]] = None,
    ):
        self.schedule = schedule
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.decay_epochs = decay_epochs or []

    def on_epoch_begin(self, epoch: int, trainer):
        lr = self._compute_lr(epoch)

        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = lr

    def _compute_lr(self, epoch: int) -> float:
        """Compute learning rate for current epoch."""
        # Warmup phase
        if epoch < self.warmup_epochs:
            return self.initial_lr * (epoch + 1) / self.warmup_epochs

        # Main schedule
        if self.schedule == 'constant':
            return self.initial_lr

        elif self.schedule == 'step':
            # Decay at specific epochs
            multiplier = 1.0
            for decay_epoch in self.decay_epochs:
                if epoch >= decay_epoch:
                    multiplier *= self.decay_rate
            return self.initial_lr * multiplier

        elif self.schedule == 'exponential':
            # Exponential decay
            return self.initial_lr * (self.decay_rate ** epoch)

        elif self.schedule == 'cosine':
            # Cosine annealing
            import math
            return self.initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / 100))

        return self.initial_lr


class GradientClipper(Callback):
    """
    Clip gradients to prevent exploding gradients.

    Simple but effective for training stability.
    """

    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm

    def on_batch_end(self, batch_idx: int, loss: float, trainer):
        torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(),
            self.max_norm,
        )


class ProgressBar(Callback):
    """
    Simple progress display during training.

    Shows current epoch, batch, loss, etc.
    """

    def __init__(self, update_freq: int = 10):
        self.update_freq = update_freq
        self.epoch_start_time = None
        self.batch_losses = []

    def on_epoch_begin(self, epoch: int, trainer):
        self.epoch_start_time = time.time()
        self.batch_losses = []
        print(f"\nEpoch {epoch + 1}/{trainer.num_epochs}")

    def on_batch_end(self, batch_idx: int, loss: float, trainer):
        self.batch_losses.append(loss)

        if (batch_idx + 1) % self.update_freq == 0:
            avg_loss = sum(self.batch_losses) / len(self.batch_losses)
            print(f"  Batch {batch_idx + 1}: loss={avg_loss:.6f}", end='\r')

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        elapsed = time.time() - self.epoch_start_time
        avg_loss = sum(self.batch_losses) / len(self.batch_losses)

        print(f"\n  Train loss: {avg_loss:.6f}")

        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        print(f"  Time: {elapsed:.2f}s")


class CallbackList:
    """
    Manages multiple callbacks.

    Just a container that calls all callbacks in order.
    """

    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks

    def on_train_begin(self, trainer):
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer):
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_begin(self, epoch: int, trainer):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, trainer)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, metrics, trainer)

    def on_batch_begin(self, batch_idx: int, trainer):
        for cb in self.callbacks:
            cb.on_batch_begin(batch_idx, trainer)

    def on_batch_end(self, batch_idx: int, loss: float, trainer):
        for cb in self.callbacks:
            cb.on_batch_end(batch_idx, loss, trainer)

    def should_stop(self) -> bool:
        """Check if any callback wants to stop training."""
        return any(
            getattr(cb, 'should_stop', False)
            for cb in self.callbacks
        )


if __name__ == "__main__":
    print("Testing callbacks...")

    # Test early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=3)

    # Simulate training
    class MockTrainer:
        pass

    trainer = MockTrainer()

    # Loss improves then plateaus
    losses = [1.0, 0.8, 0.6, 0.5, 0.5, 0.5, 0.5]

    for epoch, loss in enumerate(losses):
        metrics = {'val_loss': loss}
        early_stop.on_epoch_end(epoch, metrics, trainer)

        if early_stop.should_stop:
            print(f"Stopped at epoch {epoch}")
            break

    print("\nCallbacks test passed!")
