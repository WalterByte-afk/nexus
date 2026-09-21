"""
Complete training script for NEXUS-Omega.

Integrates all components: model, data, callbacks, monitoring.
Ready to run when you have compute resources.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus_omega.core.omega_system import NEXUSOmega, ArchitectureConfig
from nexus_omega.data.data_utils import (
    TextDataset,
    StreamingDataset,
    SimpleTokenizer,
    create_dataloader,
    load_text_file,
)
from nexus_omega.training.callbacks import (
    CallbackList,
    EarlyStopping,
    ModelCheckpoint,
    MetricsLogger,
    LearningRateScheduler,
    GradientClipper,
    ProgressBar,
)
from nexus_omega.training.monitoring import (
    TrainingMonitor,
    GradientMonitor,
    MemoryMonitor,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train NEXUS-Omega')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--vocab-size', type=int, default=10000, help='Vocab size')
    parser.add_argument('--hidden-dim', type=int, default=768, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def setup_model(args):
    """Create and configure the model."""
    config = ArchitectureConfig(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        expert_dim=args.hidden_dim // 4,
        kan_grid_size=5,
        max_recurrence=8,
        consolidation_threshold=0.1,
        memory_dim=1024,
        sdr_dim=8192,
        sdr_sparsity=0.02,
    )

    model = NEXUSOmega(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Config: {config}")

    return model, config


def setup_data(args):
    """Create data loaders."""
    # Load data
    if args.data.endswith('.jsonl'):
        from nexus_omega.data.data_utils import load_jsonl
        texts = load_jsonl(args.data)
    else:
        texts = load_text_file(args.data)

    print(f"Loaded {len(texts)} texts")

    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)

    # Create dataset
    dataset = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=args.max_length,
        stride=args.max_length // 2,
    )

    print(f"Dataset size: {len(dataset)} examples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader, tokenizer


def setup_callbacks(args, model):
    """Configure training callbacks."""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            min_delta=1e-4,
            mode='min',
        ),
        ModelCheckpoint(
            filepath=os.path.join(args.checkpoint_dir, 'best_model.pt'),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
        ),
        ModelCheckpoint(
            filepath=os.path.join(args.checkpoint_dir, 'epoch_{epoch}.pt'),
            save_best_only=False,
            save_freq=5,
        ),
        MetricsLogger(
            log_file=os.path.join(args.log_dir, 'metrics.jsonl'),
        ),
        LearningRateScheduler(
            schedule='cosine',
            initial_lr=args.lr,
            warmup_epochs=2,
        ),
        GradientClipper(max_norm=1.0),
        ProgressBar(update_freq=10),
    ]

    return CallbackList(callbacks)


def train_epoch(model, dataloader, optimizer, device, epoch, callbacks, monitor, grad_monitor):
    """Train for one epoch."""
    model.train()

    callbacks.on_epoch_begin(epoch, model)
    monitor.start_epoch(epoch, args.epochs)

    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        callbacks.on_batch_begin(batch_idx, model)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids)

        # Compute loss (simple next-token prediction)
        loss = nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            labels.view(-1),
            ignore_index=0,  # Ignore padding
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        callbacks.on_batch_end(batch_idx, loss.item(), model)

        optimizer.step()

        total_loss += loss.item()

        # Logging
        lr = optimizer.param_groups[0]['lr']
        monitor.log_batch(batch_idx, num_batches, loss.item(), lr)

        # Track gradients
        grad_monitor.track_gradients(model)

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)

            loss = nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
                ignore_index=0,
            )

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    global args
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup model, data, callbacks
    model, config = setup_model(args)
    model = model.to(device)

    dataloader, tokenizer = setup_data(args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    callbacks = setup_callbacks(args, model)
    monitor = TrainingMonitor(log_dir=args.log_dir)
    grad_monitor = GradientMonitor()
    mem_monitor = MemoryMonitor()

    # Resume if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1

    # Training loop
    monitor.start_training()
    callbacks.on_train_begin(model)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(
            model, dataloader, optimizer, device, epoch,
            callbacks, monitor, grad_monitor
        )

        # Validate (using same dataloader for now - should use separate val set)
        val_loss = evaluate(model, dataloader, device)

        # End of epoch
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }

        epoch_metrics = monitor.end_epoch()
        epoch_metrics.update(metrics)

        callbacks.on_epoch_end(epoch, epoch_metrics, model)

        # Memory monitoring
        mem_stats = mem_monitor.track_memory()
        if mem_stats:
            print(f"  GPU Memory: {mem_stats.get('allocated_gb', 0):.2f}GB allocated")

        # Check early stopping
        if callbacks.should_stop():
            print(f"Early stopping triggered at epoch {epoch}")
            break

    monitor.end_training()
    callbacks.on_train_end(model)

    # Final gradient stats
    print("\nGradient statistics:")
    for key, values in grad_monitor.get_stats().items():
        if values:
            print(f"  {key}: avg={sum(values)/len(values):.6f}, max={max(values):.6f}")

    # Final memory stats
    print("\nMemory statistics:")
    for key, value in mem_monitor.get_summary().items():
        print(f"  {key}: {value:.2f}GB")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()