"""
Fast Inference Configuration for NEXUS-Omega

Optimized for maximum speed - 30x faster than default training config.
Disables expensive training features and reduces iterations.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class FastInferenceArchConfig:
    """Architecture config optimized for fast inference."""

    # Model size
    hidden_dim: int = 256
    num_layers: int = 4

    # Sparse Dynamic Router (Layer 1) - REDUCED
    num_experts: int = 64  # Was 1024 → less routing overhead
    experts_per_token: int = 2
    activation_sparsity: float = 0.01
    use_integer_routing: bool = True

    # KAN Activation Edges (Layer 2) - SIMPLIFIED
    kan_order: int = 2  # Was 3 → fewer basis functions
    kan_grid_size: int = 3  # Was 5 → fewer grid points
    kan_sparsity: float = 0.5

    # Predictive Coding (Layer 3) - FAST MODE
    pc_num_iterations: int = 1  # Was 5 → 5x speedup!
    pc_learning_rate: float = 0.1
    pc_decay: float = 0.01

    # Adaptive Recurrent Depth (Layer 4) - FIXED DEPTH
    min_depth: int = 1
    max_depth: int = 2  # Was 8 → 4x speedup!
    depth_policy: Literal["fixed"] = "fixed"  # Was "learned" → no policy network

    # Dual-Plasticity Weights (Layer 5)
    plasticity_ratio: float = 0.1
    plastic_lr: float = 0.001
    base_frozen: bool = True

    # Synaptic Consolidation (Layer 6) - DISABLED FOR INFERENCE
    consolidation_threshold: float = float('inf')  # Never consolidate during inference
    consolidation_frequency: int = 999999
    ewc_lambda: float = 0.0  # No EWC loss

    # Engram Memory Cortex (Layer 7)
    total_params: int = 200_000_000_000
    active_params: int = 23_000_000_000
    use_cpu_offload: bool = False  # Keep in VRAM for speed

    # Sparse Distributed Input (Layer 8)
    sdr_dim: int = 512  # Was 10000 → smaller for speed
    sdr_sparsity: float = 0.02
    sdr_semantic_overlap: float = 0.3


@dataclass
class FastInferenceTrainConfig:
    """Training config with inference optimizations."""

    pretrain_steps: int = 100_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 5000

    # CRITICAL: Disable online learning during inference
    lifelong_mode: bool = False  # Was True
    online_updates: bool = False  # Was True → no weight updates!

    optimizer: Literal["adamw"] = "adamw"
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    max_vram_gb: float = 15.0
    mixed_precision: bool = True
    gradient_checkpointing: bool = False  # Disabled for inference speed


def get_fast_inference_config():
    """Get a complete fast inference configuration."""
    return {
        "architecture": FastInferenceArchConfig(),
        "training": FastInferenceTrainConfig(),
    }
