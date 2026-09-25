"""
Lifelong Learning Training Script for NEXUS-Omega.

Tests the brain-inspired continual learning capabilities:
1. Online learning during inference (Hebbian updates)
2. Dual-plasticity weight separation (fast + slow memory)
3. Synaptic consolidation during "sleep mode"
4. Catastrophic forgetting resistance
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nexus_omega.core.omega_system import NEXUSOmega, create_nexus_omega
from nexus_omega.config.settings import ArchitectureConfig
from nexus_omega.data.continual_datasets import ContinualLearningBenchmark
from nexus_omega.memory.consolidation import SynapticConsolidation, SleepMode
import numpy as np
from tqdm import tqdm


class LifelongLearner:
    """
    Implements the full lifelong learning loop.

    Training phases:
    1. Task learning with online updates
    2. Consolidation during sleep mode
    3. Evaluation on all tasks
    """

    def __init__(
        self,
        model: NEXUSOmega,
        device: str = "cuda",
        plasticity_lr: float = 0.001,
        consolidation_freq: int = 500,
    ):
        self.model = model.to(device)
        self.device = device
        self.plasticity_lr = plasticity_lr

        # Standard optimizer for base weights
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.0001,
            weight_decay=0.01
        )

        # Consolidation system
        self.consolidation = SynapticConsolidation(
            model,
            consolidation_threshold=0.5,
            consolidation_frequency=consolidation_freq
        )

        self.sleep_mode = SleepMode(
            self.consolidation,
            sleep_threshold=0.6,
            min_steps_between_sleep=100
        )

        # Metrics
        self.step = 0
        self.task_history = []

    def train_task(
        self,
        task_dataloader,
        task_id: int,
        num_epochs: int = 3,
        enable_online_learning: bool = True,
    ):
        """
        Train on a single task with online learning enabled.

        Args:
            task_dataloader: DataLoader for this task
            task_id: Task identifier
            num_epochs: Training epochs
            enable_online_learning: Enable Hebbian updates during forward pass
        """
        self.model.train()

        print(f"\n=== Training Task {task_id} ===")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(task_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass with online learning
                self.optimizer.zero_grad()

                output = self.model(
                    batch_x,
                    enable_online_learning=enable_online_learning
                )

                # Extract logits
                if hasattr(output, 'logits'):
                    logits = output.logits
                else:
                    logits = output

                # For sequential tasks: average over sequence dimension
                if len(logits.shape) == 3:
                    logits = logits.mean(dim=1)

                # Binary classification loss
                loss = nn.functional.cross_entropy(logits, batch_y)

                # Add EWC loss if we have Fisher information
                if len(self.consolidation.fisher_dict) > 0:
                    ewc_loss = self.consolidation.ewc_loss()
                    loss = loss + 0.01 * ewc_loss

                # Backward and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                self.step += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                # Check if we should consolidate
                if self.consolidation.should_consolidate():
                    print("\n[CONSOLIDATION] Running sleep mode...")
                    report = self.sleep_mode.enter_sleep_mode()
                    print(f"  Consolidated {report.num_consolidated} modules")
                    print(f"  Protected {report.protected_weights} important weights")

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        # Compute Fisher Information after task completion
        print("\n[FISHER] Computing importance scores...")
        self.consolidation.compute_fisher_information(task_dataloader, num_samples=100, device=self.device)

        # Run final consolidation for this task
        print("[SLEEP] Final consolidation for this task...")
        report = self.sleep_mode.enter_sleep_mode()
        print(f"  Transferred knowledge: {report.transferred_knowledge:.4f}")

        self.task_history.append(task_id)

    def sequential_training(
        self,
        benchmark: ContinualLearningBenchmark,
        num_epochs_per_task: int = 3,
    ):
        """
        Train on tasks sequentially and measure catastrophic forgetting.

        This is the key test of lifelong learning.
        """
        print("\n" + "="*60)
        print("LIFELONG LEARNING EXPERIMENT")
        print("="*60)

        results = {
            "task_accuracies": {},
            "forgetting_over_time": [],
        }

        for task_id in range(benchmark.num_tasks):
            # Train on this task
            self.train_task(
                benchmark.train_loaders[task_id],
                task_id,
                num_epochs=num_epochs_per_task,
                enable_online_learning=True
            )

            # Evaluate on ALL tasks (including previous ones)
            print(f"\n[EVAL] Testing on all {task_id+1} tasks seen so far...")
            metrics = benchmark.evaluate_all_tasks(self.model, self.device)

            print(f"\nResults after Task {task_id}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            results["task_accuracies"][f"after_task_{task_id}"] = metrics
            results["forgetting_over_time"].append(metrics["catastrophic_forgetting"])

        # Final summary
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)

        final_metrics = benchmark.evaluate_all_tasks(self.model, self.device)

        print(f"\nFinal Average Accuracy: {final_metrics['avg_accuracy']:.4f}")
        print(f"Catastrophic Forgetting: {final_metrics['catastrophic_forgetting']:.4f}")

        print("\nForgetting over time:")
        for i, forgetting in enumerate(results["forgetting_over_time"]):
            print(f"  After task {i}: {forgetting:.4f}")

        # Check success criteria
        print("\n" + "="*60)
        if final_metrics['catastrophic_forgetting'] < 0.05:
            print("SUCCESS: Catastrophic forgetting < 5% :D")
        elif final_metrics['catastrophic_forgetting'] < 0.15:
            print("GOOD: Catastrophic forgetting < 15%")
        else:
            print(f"NEEDS WORK: Forgetting = {final_metrics['catastrophic_forgetting']:.2%}")

        if final_metrics['avg_accuracy'] > 0.7:
            print("SUCCESS: Average accuracy > 70%")

        print("="*60)

        return results


def main():
    print("NEXUS-Omega Lifelong Learning Training")
    print("="*60)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create small model for testing
    print("\nCreating NEXUS-Omega model...")
    config = ArchitectureConfig(
        hidden_dim=256,
        num_layers=4,
        num_experts=64,
        max_depth_inference=3,
        depth_policy="fixed",  # Use fixed depth to avoid numerical issues
    )

    model = NEXUSOmega(
        vocab_size=1000,
        arch_config=config,
    )
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count['total']:,}")

    # Create continual learning benchmark
    print("\nSetting up continual learning benchmark...")
    benchmark = ContinualLearningBenchmark(
        dataset_type="text",
        num_tasks=3,  # Start with 3 tasks
        batch_size=16,
    )

    # Create lifelong learner
    learner = LifelongLearner(
        model,
        device=device,
        plasticity_lr=0.001,
        consolidation_freq=100,
    )

    # Run sequential training
    results = learner.sequential_training(
        benchmark,
        num_epochs_per_task=2,  # Quick test
    )

    print("\nTraining complete!")

    # Save model
    save_path = "checkpoints/nexus_omega_lifelong.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'results': results,
    }, save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
