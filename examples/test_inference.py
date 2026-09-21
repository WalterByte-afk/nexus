"""
Simple inference test for NEXUS-Omega.

Works on CPU for quick testing without GPU.
"""

import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus_omega.core.omega_system import NEXUSOmega, ArchitectureConfig
from nexus_omega.data.data_utils import SimpleTokenizer
from nexus_omega.deploy.inference import InferenceEngine, InferenceConfig


def main():
    print("Testing NEXUS-Omega inference...")
    print("=" * 50)

    # Small config for quick testing
    config = ArchitectureConfig(
        hidden_dim=256,
        num_layers=4,
        num_experts=64,
        kan_grid_size=3,
        max_depth=4,
        consolidation_threshold=0.1,
        sdr_dim=512,
        sdr_sparsity=0.02,
    )

    # Create tokenizer and model
    vocab_size = 1000
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)

    # Create model with vocab_size and config
    model = NEXUSOmega(vocab_size=vocab_size, arch_config=config)
    model.eval()

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Test input
    text = "Hello world, this is a test"
    tokens = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor([tokens], dtype=torch.long)

    print(f"\nInput text: {text}")
    print(f"Input tokens: {tokens}")
    print(f"Input shape: {input_ids.shape}")

    # Run forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(input_ids)

    print(f"Output shape: {outputs.logits.shape}")
    print(f"Output range: [{outputs.logits.min():.4f}, {outputs.logits.max():.4f}]")

    # Test inference engine
    print("\nTesting inference engine...")
    infer_config = InferenceConfig(
        use_int8=False,  # Disable for now - has compatibility issues
        use_kv_cache=False,
        use_fp16=False,
    )
    engine = InferenceEngine(model, infer_config)

    # Generate text (using logits directly)
    print("\nGenerating text...")
    generated = engine.generate(
        input_ids,
        max_length=10,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )

    # Decode
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"Generated: {generated_text}")

    # Benchmark
    print("\nBenchmarking...")
    bench_results = engine.benchmark(input_ids, num_runs=50)
    print(f"Tokens/sec: {bench_results['tokens_per_sec']:.2f}")
    print(f"Latency: {bench_results['latency_ms']:.2f}ms")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()