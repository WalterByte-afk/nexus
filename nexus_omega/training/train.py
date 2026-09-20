"""
Training utilities for NEXUS-Ω.

Includes training loop, learning rate schedulers, and continual learning support.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from nexus_omega.config.settings import TrainingConfig, SystemConfig
from nexus_omega.core.omega_system import NEXUSOmega
from nexus_omega.utils.efficiency_metrics import EfficiencyTracker


class NEXUSTrainer:
    """
    Trainer for NEXUS-Ω with support for:
    - Pre-training
    - Lifelong/continual learning
    - EWC loss for anti-forgetting
    - Efficiency tracking
    """

    def __init__(
        self,
        model: NEXUSOmega,
        config: Optional[TrainingConfig] = None,
        sys_config: Optional[SystemConfig] = None,
        device: str = "auto",
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self.sys_config = sys_config or SystemConfig()
        self.device = self._setup_device(device)

        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.pretrain_steps,
            eta_min=self.config.learning_rate * 0.1,
        )

        # Efficiency tracker
        self.tracker = EfficiencyTracker(self.device)

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Metrics history
        self.history: Dict[str, List[float]] = {}

    def _setup_device(self, device: str) -> str:
        """Setup computing device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result

    def pretrain(
        self,
        dataloader: DataLoader,
        num_steps: Optional[int] = None,
        log_interval: int = 100,
    ):
        """
        Pre-train the model on a dataset.

        Args:
            dataloader: Training data loader
            num_steps: Number of steps (None = one epoch)
            log_interval: How often to log
        """
        self.model.train()
        self.tracker.start()

        num_steps = num_steps or len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_steps:
                break

            batch = self.to_device(batch)

            # Forward pass
            self.optimizer.zero_grad()

            output = self.model(
                batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                enable_online_learning=False,
            )

            # Compute loss (cross-entropy)
            loss = self._compute_loss(output, batch)

            # Add EWC loss if available
            if self.global_step > 1000:
                ewc_loss = self.model.compute_ewc_loss()
                loss = loss + 0.01 * ewc_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )

            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            if batch_idx % log_interval == 0:
                metrics = self._log_batch(batch_idx, loss, output)

            self.global_step += 1

        self.epoch += 1
        metrics = self.tracker.stop(num_tokens=num_steps * batch.get("input_ids").shape[0])
        return metrics

    def continual_train(
        self,
        input_ids: torch.Tensor,
        num_steps: int = 100,
    ):
        """
        Train in continual learning mode (online learning during inference).

        Args:
            input_ids: Input tensor [batch, seq_len]
            num_steps: Number of online steps
        """
        self.model.train()
        self.tracker.start()

        for step in range(num_steps):
            # Forward with online learning enabled
            output = self.model(
                input_ids,
                enable_online_learning=True,
            )

            # Compute loss
            loss = self._compute_loss(output, {"input_ids": input_ids})

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Update
            self.optimizer.step()

            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")

        metrics = self.tracker.stop(num_tokens=num_steps * input_ids.shape[0])
        return metrics

    def _compute_loss(
        self,
        output: "NEXUSOutput",
        batch: Dict,
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        logits = output.logits
        targets = batch["input_ids"]

        # Flatten for loss computation
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        loss = nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=-100,
        )

        # Add auxiliary losses
        if output.aux_losses is not None:
            loss = loss + output.aux_losses

        return loss

    def _log_batch(
        self,
        batch_idx: int,
        loss: torch.Tensor,
        output: "NEXUSOutput",
    ) -> Dict:
        """Log batch metrics."""
        metrics = {
            "loss": loss.item(),
            "tokens_processed": self.global_step,
            "epoch": self.epoch,
        }

        if output.metrics:
            metrics.update(output.metrics)

        # Print
        if batch_idx % 100 == 0:
            print(f"Step {batch_idx}: Loss = {loss.item():.4f}")

        # Store history
        for k, v in metrics.items():
            if isinstance(v, (float, int)):
                if k not in self.history:
                    self.history[k] = []
                self.history[k].append(v)

        return metrics

    def consolidate(self):
        """Run consolidation cycle."""
        report = self.model.consolidate()
        print(f"Consolidation: {report.num_consolidated} layers transferred")
        return report


def create_dataloader(
    texts: List[str],
    tokenizer: object,
    batch_size: int = 32,
    max_length: int = 128,
) -> DataLoader:
    """
    Create a dataloader from texts.

    Args:
        texts: List of text strings
        tokenizer: Tokenizer with encode method
        batch_size: Batch size
        max_length: Maximum sequence length

    Returns:
        PyTorch DataLoader
    """
    from torch.utils.data import Dataset, DataLoader

    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            encoded = self.tokenizer.encode(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            return {
                "input_ids": torch.tensor(encoded),
                "attention_mask": torch.ones(len(encoded)),
            }

    dataset = TextDataset(texts, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    print("Testing training utilities...")

    # Create a simple model
    from nexus_omega.core.omega_system import create_nexus_omega

    model = create_nexus_omega("small", vocab_size=1000)
    trainer = NEXUSTrainer(model)

    # Create dummy data
    dummy_texts = ["Hello world"] * 100
    # (Would use real tokenizer here)

    print(f"Model parameters: {model.count_parameters()}")

    print("✓ Training utilities tests passed!")
