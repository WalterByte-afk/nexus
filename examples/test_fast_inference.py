"""
ULTRA-FAST Inference Test for NEXUS-Omega

Uses optimized config: 1 PC iteration, fixed depth=2, no online learning.
Expected: 30x faster than default config.
"""

import os
import sys
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus_omega.core.omega_system import NEXUSOmega
from nexus_omega.config.fast_inference_config import get_fast_inference_config
from nexus_omega.deploy.inference import InferenceEngine, InferenceConfig


def main():
    print("=" * 60)
    print("NEXUS-Omega ULTRA-FAST Inference Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Get optimized config
    config = get_fast_inference_config()
    arch_config = config["architecture"]
    train_config = config["training"]

    print(f"\nConfig optimizations:")
    print(f"  - PC iterations: {arch_config.pc_num_iterations} (was 5)")
    print(f"  - Max depth: {arch_config.max_depth} (was 8)")
    print(f"  - Depth policy: {arch_config.depth_policy} (was learned)")
    print(f"  - Num experts: {arch_config.num_experts} (was 1024)")
    print(f"  - Online learning: {train_config.online_updates} (was True)")

    # Create model
    print(f"\nCreating model...")
    vocab_size = 1000
    model = NEXUSOmega(
        vocab_size=vocab_size,
        arch_config=arch_config,
        train_config=train_config
    )
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Create input
    input_ids = torch.randint(0, vocab_size, (1, 10), device=device)

    # Warmup
    print(f"\nWarming up GPU...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids, enable_online_learning=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark forward pass
    print(f"\nBenchmarking forward pass (50 iterations)...")
    start = time.time()

    with torch.no_grad():
        for _ in range(50):
            _ = model(input_ids, enable_online_learning=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start
    tokens_per_sec = (50 * 10) / elapsed

    print(f"✓ 50 passes in {elapsed:.2f}s")
    print(f"✓ {elapsed/50*1000:.1f}ms per pass")
    print(f"✓ {tokens_per_sec:.1f} tokens/sec")

    # Test generation
    print(f"\nTesting generation (3 tokens)...")
    infer_config = InferenceConfig(
        use_int8=False,
        use_kv_cache=False,
        use_fp16=False,
        use_torch_compile=False,  # Test without compile first
    )
    engine = InferenceEngine(model, infer_config)

    start = time.time()
    generated = engine.generate(
        input_ids,
        max_length=3,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )
    elapsed = time.time() - start

    print(f"✓ Generated 3 tokens in {elapsed:.2f}s ({elapsed/3:.2f}s per token)")

    if elapsed / 3 < 1.0:
        print(f"\n🎉 SUCCESS! Under 1 second per token!")
    else:
        print(f"\n⚠️  Still slow - needs more optimization")

    print(f"\n{'='*60}")
    print("Test complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
