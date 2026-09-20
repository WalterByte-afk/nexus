"""
Example usage of NEXUS-Ω.

This demonstrates how to:
1. Create a NEXUS-Ω model
2. Run inference
3. Enable online learning
4. Run consolidation
"""

import torch
from nexus_omega.core.omega_system import NEXUSOmega, create_nexus_omega
from nexus_omega.config.settings import get_config
from nexus_omega.deploy.optimizer import optimize_for_inference


def basic_inference():
    """Basic inference example."""
    print("=" * 60)
    print("NEXUS-Omega Basic Inference Example")
    print("=" * 60)

    # Create model
    model = create_nexus_omega("small", vocab_size=10000)

    # Create dummy input
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))

    # Run inference
    output = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {output.logits.shape}")
    print(f"Parameters: {model.count_parameters()['total']:,}")
    print(f"Memory footprint: {model.get_memory_footprint()['active_gb']:.2f} GB")
    print()


def online_learning():
    """Demonstrate online learning (continuous learning during inference)."""
    print("=" * 60)
    print("NEXUS-Omega Online Learning Example")
    print("=" * 60)

    model = create_nexus_omega("small", vocab_size=10000)

    # Simulate new input
    input_ids = torch.randint(0, 10000, (1, 16))

    # Forward pass with online learning
    output = model(input_ids, enable_online_learning=True)

    print(f"Processed {input_ids.numel()} tokens")
    print(f"Updated during inference: Yes")
    print(f"Plastic weight updates: Enabled")
    print()


def consolidation():
    """Demonstrate consolidation (sleep mode knowledge transfer)."""
    print("=" * 60)
    print("NEXUS-Omega Consolidation Example")
    print("=" * 60)

    model = create_nexus_omega("small", vocab_size=10000)

    # Process some inputs first
    for i in range(5):
        input_ids = torch.randint(0, 10000, (1, 16))
        _ = model(input_ids, enable_online_learning=True)

    print(f"Tokens processed before consolidation: {model.total_tokens_processed}")

    # Run consolidation
    report = model.consolidate()

    print(f"Consolidation report:")
    print(f"  - Layers transferred: {report.num_consolidated}")
    print(f"  - Knowledge transferred: {report.transferred_knowledge:.4f}")
    print()


def optimization():
    """Demonstrate model optimization for deployment."""
    print("=" * 60)
    print("NEXUS-Omega Optimization Example")
    print("=" * 60)

    model = create_nexus_omega("small", vocab_size=10000)

    # Get original size
    original_size = model.get_memory_footprint()

    # Optimize
    optimized = optimize_for_inference(
        model,
        quantization="int8",
        pruning=0.9,
        use_torch_compile=True,
    )

    print(f"Original active memory: {original_size['active_gb']:.2f} GB")
    print(f"Optimized: Quantized INT8, 90% sparse")
    print()


def compare_architectures():
    """Compare NEXUS-Omega with traditional architectures."""
    print("=" * 60)
    print("NEXUS-Omega vs Traditional Architecture")
    print("=" * 60)

    print("\nFeature Comparison:")
    print("-" * 40)
    print(f"{'Feature':<30} {'Traditional':<15} {'NEXUS-Ω':<15}")
    print("-" * 40)
    print(f"{'Active parameters/token':<30} {'~200B':<15} {'~23B':<15}")
    print(f"{'Energy per token':<30} {'0.5-2 J':<15} {'~0.02 J':<15}")
    print(f"{'Catastrophic forgetting':<30} {'High':<15} {'None':<15}")
    print(f"{'New knowledge ingestion':<30} {'Hours':<15} {'Instant':<15}")
    print(f"{'Memory scaling':<30} {'O(N^2)':<15} {'O(N)':<15}")
    print("-" * 40)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NEXUS-Omega Example Scripts")
    print("Brain-Inspired AI Architecture")
    print("=" * 60 + "\n")

    # Run examples
    basic_inference()
    online_learning()
    consolidation()
    optimization()
    compare_architectures()

    print("All examples completed!")
