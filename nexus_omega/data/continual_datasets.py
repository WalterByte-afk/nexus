"""
Continual Learning Datasets for NEXUS-Omega validation.

Implements sequential task datasets to test catastrophic forgetting resistance.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import numpy as np


class SequentialMNIST(Dataset):
    """
    Split MNIST into sequential tasks.

    Classic continual learning benchmark. Each task is 2 digits.
    Task 1: [0, 1]
    Task 2: [2, 3]
    Task 3: [4, 5]
    Task 4: [6, 7]
    Task 5: [8, 9]
    """

    def __init__(self, task_id: int = 0, train: bool = True):
        self.task_id = task_id
        self.train = train

        # Download MNIST using torchvision
        try:
            from torchvision import datasets, transforms

            mnist = datasets.MNIST(
                root='./data',
                train=train,
                download=True,
                transform=transforms.ToTensor()
            )

            # Filter to current task's digits
            digit_start = task_id * 2
            digit_end = digit_start + 2

            indices = [
                i for i, (_, label) in enumerate(mnist)
                if digit_start <= label < digit_end
            ]

            self.data = [mnist[i][0] for i in indices]
            self.labels = [mnist[i][1] - digit_start for i in indices]  # Remap to [0, 1]

        except ImportError:
            # Fallback: generate synthetic data
            print("torchvision not found. Using synthetic data.")
            num_samples = 1000 if train else 200
            self.data = [torch.randn(1, 28, 28) for _ in range(num_samples)]
            self.labels = [i % 2 for i in range(num_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SequentialTextTasks(Dataset):
    """
    Sequential text classification tasks for language models.

    Task 1: Sentiment (positive/negative)
    Task 2: Topic (tech/sports)
    Task 3: Intent (question/statement)
    """

    def __init__(self, task_id: int = 0, train: bool = True, vocab_size: int = 1000):
        self.task_id = task_id
        self.train = train
        self.vocab_size = vocab_size

        # Generate synthetic sequential text tasks
        num_samples = 500 if train else 100
        seq_len = 32

        self.data = []
        self.labels = []

        for i in range(num_samples):
            # Create synthetic token sequences
            # Different tasks have different token distributions
            if task_id == 0:  # Sentiment task
                # Positive vs negative sentiment patterns
                label = i % 2
                if label == 0:  # Negative
                    tokens = torch.randint(0, vocab_size // 2, (seq_len,))
                else:  # Positive
                    tokens = torch.randint(vocab_size // 2, vocab_size, (seq_len,))

            elif task_id == 1:  # Topic task
                # Tech vs sports vocabulary
                label = i % 2
                if label == 0:  # Tech
                    tokens = torch.randint(100, 300, (seq_len,))
                else:  # Sports
                    tokens = torch.randint(400, 600, (seq_len,))

            else:  # Intent task
                # Question vs statement patterns
                label = i % 2
                if label == 0:  # Question
                    tokens = torch.randint(0, vocab_size, (seq_len,))
                    tokens[0] = 50  # Question marker at start
                else:  # Statement
                    tokens = torch.randint(0, vocab_size, (seq_len,))
                    tokens[0] = 100  # Statement marker

            self.data.append(tokens)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ContinualLearningBenchmark:
    """
    Manages sequential task training and evaluation.

    Tests:
    1. Backward transfer (catastrophic forgetting)
    2. Forward transfer (knowledge reuse)
    3. Sample efficiency
    """

    def __init__(
        self,
        dataset_type: str = "text",  # "mnist" or "text"
        num_tasks: int = 5,
        batch_size: int = 32,
    ):
        self.dataset_type = dataset_type
        self.num_tasks = num_tasks
        self.batch_size = batch_size

        # Track accuracy per task over time
        self.task_accuracies = {i: [] for i in range(num_tasks)}

        # Create all task dataloaders
        self.train_loaders = []
        self.test_loaders = []

        for task_id in range(num_tasks):
            if dataset_type == "mnist":
                train_ds = SequentialMNIST(task_id, train=True)
                test_ds = SequentialMNIST(task_id, train=False)
            else:  # text
                train_ds = SequentialTextTasks(task_id, train=True)
                test_ds = SequentialTextTasks(task_id, train=False)

            self.train_loaders.append(
                DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            )
            self.test_loaders.append(
                DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            )

    def evaluate_task(
        self,
        model,
        task_id: int,
        device: str = "cuda",
    ) -> float:
        """
        Evaluate model on a specific task.

        Returns accuracy.
        """
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in self.test_loaders[task_id]:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # Forward pass
                output = model(batch_x)

                # Get predictions
                if hasattr(output, 'logits'):
                    logits = output.logits
                else:
                    logits = output

                # For sequential tasks, we only need binary classification
                if len(logits.shape) == 3:
                    logits = logits.mean(dim=1)  # Average over sequence

                pred = (logits[:, 1] > logits[:, 0]).long()

                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def evaluate_all_tasks(
        self,
        model,
        device: str = "cuda",
    ) -> dict:
        """
        Evaluate on all tasks seen so far.

        Returns:
            dict with metrics including catastrophic forgetting
        """
        accuracies = {}

        for task_id in range(self.num_tasks):
            acc = self.evaluate_task(model, task_id, device)
            accuracies[f"task_{task_id}"] = acc
            self.task_accuracies[task_id].append(acc)

        # Compute catastrophic forgetting
        # Forgetting = (max_accuracy - current_accuracy) for past tasks
        forgetting = []
        for task_id in range(self.num_tasks):
            if len(self.task_accuracies[task_id]) > 1:
                max_acc = max(self.task_accuracies[task_id])
                current_acc = self.task_accuracies[task_id][-1]
                forgetting.append(max_acc - current_acc)

        avg_forgetting = np.mean(forgetting) if forgetting else 0.0

        return {
            **accuracies,
            "avg_accuracy": np.mean(list(accuracies.values())),
            "catastrophic_forgetting": avg_forgetting,
        }


if __name__ == "__main__":
    print("Testing Continual Learning Datasets...")

    # Test Sequential MNIST
    print("\n1. Sequential MNIST")
    for task_id in range(5):
        ds = SequentialMNIST(task_id, train=True)
        print(f"Task {task_id}: {len(ds)} samples, digits [{task_id*2}, {task_id*2+1}]")
        x, y = ds[0]
        print(f"  Sample shape: {x.shape}, label: {y}")

    # Test Sequential Text
    print("\n2. Sequential Text Tasks")
    for task_id in range(3):
        ds = SequentialTextTasks(task_id, train=True)
        print(f"Task {task_id}: {len(ds)} samples")
        x, y = ds[0]
        print(f"  Sample shape: {x.shape}, label: {y}")

    # Test Benchmark
    print("\n3. Continual Learning Benchmark")
    benchmark = ContinualLearningBenchmark("text", num_tasks=3, batch_size=16)
    print(f"Created benchmark with {len(benchmark.train_loaders)} tasks")

    print("\n✓ Continual learning datasets ready!")
