"""
NEXUS-Ω Architecture Settings

This file contains all hyperparameters and configuration for the NEXUS-Ω system.
Designed for Google Colab free tier initially, scalable to more powerful hardware.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ArchitectureConfig:
    """Core architecture hyperparameters."""

    # Model size
    hidden_dim: int = 768  # Base hidden dimension
    num_layers: int = 12   # Number of NEXUS-Ω blocks

    # Sparse Dynamic Router (Layer 1)
    num_experts: int = 1024  # Micro-experts for routing
    experts_per_token: int = 2  # Top-k experts to activate
    activation_sparsity: float = 0.01  # 1% activation rate
    use_integer_routing: bool = True  # Integer-only routing (no float softmax)

    # KAN Activation Edges (Layer 2)
    kan_order: int = 3  # B-spline order
    kan_grid_size: int = 5  # Number of grid points
    kan_sparsity: float = 0.5  # Sparsity in KAN connections

    # Predictive Coding (Layer 3)
    pc_num_iterations: int = 5  # Predictive coding inference steps
    pc_learning_rate: float = 0.1  # Local Hebbian learning rate
    pc_decay: float = 0.01  # Weight decay for stability

    # Adaptive Recurrent Depth (Layer 4)
    min_depth: int = 1  # Minimum recurrent loops
    max_depth: int = 8  # Maximum recurrent loops
    depth_policy: Literal["learned", "fixed", "adaptive"] = "learned"

    # Dual-Plasticity Weights (Layer 5)
    plasticity_ratio: float = 0.1  # Ratio of plastic to base weights
    plastic_lr: float = 0.001  # Learning rate for plastic weights
    base_frozen: bool = True  # Keep base weights frozen

    # Synaptic Consolidation (Layer 6)
    consolidation_threshold: float = 0.8  # Fisher information threshold
    consolidation_frequency: int = 1000  # Steps between consolidation
    ewc_lambda: float = 100.0  # Elastic weight consolidation strength

    # Engram Memory Cortex (Layer 7)
    total_params: int = 200_000_000_000  # 200B total parameters
    active_params: int = 23_000_000_000  # 23B active in VRAM
    use_cpu_offload: bool = True  # Offload cold weights to CPU/RAM

    # Sparse Distributed Input (Layer 8)
    sdr_dim: int = 10000  # SDR dimensionality
    sdr_sparsity: float = 0.02  # 2% active bits
    sdr_semantic_overlap: float = 0.3  # Overlap for similar inputs


@dataclass
class TrainingConfig:
    """Training and optimization settings."""

    # Pre-training
    pretrain_steps: int = 100_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 5000

    # Lifelong learning
    lifelong_mode: bool = True  # Enable continuous learning
    online_updates: bool = True  # Update during inference

    # Optimization
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Hardware constraints
    max_vram_gb: float = 15.0  # Google Colab free tier limit
    mixed_precision: bool = True  # FP16 mixed precision
    gradient_checkpointing: bool = True  # Save memory


@dataclass
class SystemConfig:
    """System-level configuration."""

    # Device
    device: Literal["cuda", "cpu", "auto"] = "auto"
    seed: int = 42

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000

    # Metrics tracking
    track_energy: bool = True  # Track energy consumption
    track_forgetting: bool = True  # Track catastrophic forgetting
    track_memory: bool = True  # Track memory usage


# Default configuration
DEFAULT_CONFIG = {
    "architecture": ArchitectureConfig(),
    "training": TrainingConfig(),
    "system": SystemConfig(),
}


def get_config(config_name: str = "default"):
    """Get a configuration by name."""
    if config_name == "default":
        return DEFAULT_CONFIG
    elif config_name == "small":
        # Smaller model for testing
        cfg = DEFAULT_CONFIG.copy()
        cfg["architecture"].hidden_dim = 384
        cfg["architecture"].num_layers = 6
        cfg["architecture"].num_experts = 256
        return cfg
    elif config_name == "large":
        # Larger model for better performance
        cfg = DEFAULT_CONFIG.copy()
        cfg["architecture"].hidden_dim = 1536
        cfg["architecture"].num_layers = 24
        cfg["architecture"].num_experts = 2048
        return cfg
    else:
        raise ValueError(f"Unknown config: {config_name}")
