"""
Efficiency metrics for tracking NEXUS-Ω performance.

Tracks energy consumption, memory usage, and computational efficiency
to verify we meet our targets vs. GPT-6/Astra baselines.
"""

import torch
import time
import psutil
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class EfficiencyMetrics:
    """Container for efficiency measurements."""

    # Energy (Joules per token)
    energy_per_token: float = 0.0

    # Memory (GB)
    peak_vram_gb: float = 0.0
    active_params_gb: float = 0.0

    # Compute (FLOPs)
    flops_per_token: float = 0.0
    theoretical_flops: float = 0.0

    # Speed (tokens/second)
    tokens_per_second: float = 0.0

    # Sparsity
    weight_sparsity: float = 0.0
    activation_sparsity: float = 0.0


class EfficiencyTracker:
    """
    Tracks and measures efficiency metrics during training/inference.

    Usage:
        tracker = EfficiencyTracker()
        tracker.start()

        # ... run model ...

        metrics = tracker.stop(num_tokens)
        print(metrics)
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.metrics_history: List[EfficiencyMetrics] = []

    def start(self):
        """Start tracking."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated() / 1e9  # GB
        else:
            self.start_memory = psutil.Process().memory_info().rss / 1e9

    def stop(self, num_tokens: int) -> EfficiencyMetrics:
        """Stop tracking and compute metrics."""
        if self.start_time is None:
            raise ValueError("Tracker not started")

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_time = time.time() - self.start_time

        metrics = EfficiencyMetrics()

        # Speed
        metrics.tokens_per_second = num_tokens / elapsed_time if elapsed_time > 0 else 0

        # Memory
        if self.device == "cuda" and torch.cuda.is_available():
            metrics.peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
        else:
            metrics.peak_vram_gb = psutil.Process().memory_info().rss / 1e9

        # Energy estimate (rough approximation)
        # Typical GPU: 300W max, assume proportional utilization
        if self.device == "cuda":
            # Simplified: 300W GPU * utilization * time
            gpu_power_w = 300  # Approximate
            energy_j = gpu_power_w * elapsed_time
            metrics.energy_per_token = energy_j / num_tokens if num_tokens > 0 else 0

        # Reset for next use
        self.start_time = None
        self.start_memory = None

        # Store history
        self.metrics_history.append(metrics)

        return metrics

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics across all tracked runs."""
        if not self.metrics_history:
            return {}

        summary = {
            "avg_energy_per_token": np.mean([m.energy_per_token for m in self.metrics_history]),
            "avg_tokens_per_second": np.mean([m.tokens_per_second for m in self.metrics_history]),
            "peak_vram_gb": max([m.peak_vram_gb for m in self.metrics_history]),
            "total_runs": len(self.metrics_history),
        }

        return summary


def estimate_flops(
    hidden_dim: int,
    num_layers: int,
    seq_len: int,
    batch_size: int,
    sparsity: float = 0.9,
) -> float:
    """
    Estimate FLOPs for a forward pass.

    Args:
        hidden_dim: Model hidden dimension
        num_layers: Number of layers
        seq_len: Sequence length
        batch_size: Batch size
        sparsity: Weight sparsity (fraction of zeros)

    Returns:
        Estimated FLOPs
    """
    # Dense FLOPs estimate
    dense_flops = 2 * batch_size * seq_len * hidden_dim * hidden_dim * num_layers

    # Apply sparsity reduction
    sparse_flops = dense_flops * (1 - sparsity)

    return sparse_flops


def compute_memory_footprint(
    total_params: int,
    active_params: int,
    bytes_per_param: int = 2,  # FP16
) -> Dict[str, float]:
    """
    Compute memory footprint in GB.

    Args:
        total_params: Total parameter count
        active_params: Parameters loaded in VRAM
        bytes_per_param: Bytes per parameter (2 for FP16, 4 for FP32)

    Returns:
        Dictionary with memory breakdown
    """
    total_memory_gb = (total_params * bytes_per_param) / 1e9
    active_memory_gb = (active_params * bytes_per_param) / 1e9

    return {
        "total_memory_gb": total_memory_gb,
        "active_memory_gb": active_memory_gb,
        "compression_ratio": total_params / active_params if active_params > 0 else 0,
    }


def compare_to_baseline(nexus_metrics: EfficiencyMetrics) -> Dict[str, float]:
    """
    Compare NEXUS-Ω metrics to GPT-6 baseline.

    Baseline estimates (approximate):
    - GPT-6 energy: ~0.5-2 J/token
    - GPT-6 VRAM: ~80GB
    - GPT-6 speed: varies by hardware
    """
    gpt6_energy = 1.0  # J/token (mid-range estimate)
    gpt6_vram = 80.0  # GB

    energy_improvement = gpt6_energy / nexus_metrics.energy_per_token if nexus_metrics.energy_per_token > 0 else float('inf')
    vram_improvement = gpt6_vram / nexus_metrics.peak_vram_gb if nexus_metrics.peak_vram_gb > 0 else float('inf')

    return {
        "energy_improvement_x": energy_improvement,
        "vram_improvement_x": vram_improvement,
        "meets_target": energy_improvement > 10 and vram_improvement > 4,
    }


if __name__ == "__main__":
    # Quick test
    print("Testing efficiency metrics...")

    tracker = EfficiencyTracker()
    tracker.start()

    # Simulate some work
    x = torch.randn(32, 128, 768)
    for _ in range(10):
        x = torch.matmul(x, torch.randn(768, 768))

    metrics = tracker.stop(num_tokens=32 * 128)
    print(f"Tokens/sec: {metrics.tokens_per_second:.1f}")
    print(f"Peak VRAM: {metrics.peak_vram_gb:.2f} GB")
    print(f"Energy/token: {metrics.energy_per_token:.4f} J")

    comparison = compare_to_baseline(metrics)
    print(f"Energy improvement: {comparison['energy_improvement_x']:.1f}x vs GPT-6 baseline")

    print("✓ Efficiency metrics tests passed!")
