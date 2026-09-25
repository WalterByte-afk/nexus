"""
Comprehensive benchmark script for NEXUS-Omega optimizations.

Tests different optimization strategies and measures speedup.
"""

import os
import sys
import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import warnings

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus_omega.core.omega_system import NEXUSOmega, ArchitectureConfig
from nexus_omega.deploy.inference import InferenceEngine, InferenceConfig
from nexus_omega.deploy.optimizer import (
    apply_torch_compile,
    optimize_for_inference,
    profile_model,
    CUDAGraphCapture,
)


def create_test_model() -> Tuple[NEXUSOmega, torch.Tensor]:
    """Create a small test model and sample input."""
    config = ArchitectureConfig(
        hidden_dim=256,
        num_layers=4,
        num_experts=64,
        kan_grid_size=3,
        max_depth=4,
    )

    model = NEXUSOmega(vocab_size=1000, arch_config=config)
    input_ids = torch.randint(0, 1000, (1, 10), dtype=torch.long)

    return model, input_ids


def benchmark_configuration(
    name: str,
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup: int = 10,
    runs: int = 50,
) -> Dict[str, float]:
    """Benchmark a specific configuration."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {name}")
    print(f"{'=' * 60}")

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    start = time.time()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_ids)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start

    num_tokens = input_ids.numel()
    tokens_processed = num_tokens * runs

    results = {
        "total_time_s": elapsed,
        "avg_latency_ms": (elapsed / runs) * 1000,
        "tokens_per_second": tokens_processed / elapsed,
        "seconds_per_token": elapsed / tokens_processed,
    }

    print(f"  Total time: {results['total_time_s']:.2f}s")
    print(f"  Avg latency: {results['avg_latency_ms']:.2f}ms")
    print(f"  Tokens/sec: {results['tokens_per_second']:.2f}")
    print(f"  Sec/token: {results['seconds_per_token']:.4f}s")

    return results


def main():
    print("=" * 80)
    print("NEXUS-Omega Optimization Benchmark Suite")
    print("=" * 80)

    # System info
    print("\nSystem Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"  Python version: {sys.version.split()[0]}")
    print(f"  torch.compile available: {hasattr(torch, 'compile')}")

    # Check Python version
    if sys.version_info >= (3, 12):
        print("\n[WARNING] Python 3.12+ detected!")
        print("   torch.compile requires Python <=3.11 for full optimization.")
        print("   Please downgrade for 5-15x speedup!\n")

    # Create model
    print("\nCreating test model...")
    model, input_ids = create_test_model()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_ids = input_ids.to(device)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")

    # Store results
    all_results = {}

    # 1. Baseline (no optimization)
    print("\n" + "=" * 80)
    print("PHASE 1: Baseline Performance")
    print("=" * 80)

    model_baseline = model
    results_baseline = benchmark_configuration(
        "Baseline (no optimization)",
        model_baseline,
        input_ids,
        warmup=5,
        runs=20,
    )
    all_results["baseline"] = results_baseline
    baseline_speed = results_baseline["seconds_per_token"]

    # 2. FP16 Mixed Precision
    if torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("PHASE 2: FP16 Mixed Precision")
        print("=" * 80)

        model_fp16 = create_test_model()[0].to(device).half()
        results_fp16 = benchmark_configuration(
            "FP16 Mixed Precision",
            model_fp16,
            input_ids,
            warmup=5,
            runs=20,
        )
        all_results["fp16"] = results_fp16
        speedup_fp16 = baseline_speed / results_fp16["seconds_per_token"]
        print(f"  → Speedup: {speedup_fp16:.2f}x over baseline")

    # 3. torch.compile (if available)
    if hasattr(torch, 'compile') and sys.version_info < (3, 12):
        print("\n" + "=" * 80)
        print("PHASE 3: torch.compile Optimization")
        print("=" * 80)

        for mode in ["default", "reduce-overhead", "max-autotune"]:
            print(f"\n--- torch.compile mode: {mode} ---")

            model_compiled = create_test_model()[0].to(device)
            model_compiled.eval()

            try:
                print(f"Compiling model (this may take 30-120s)...")
                compile_start = time.time()
                model_compiled = torch.compile(model_compiled, mode=mode, dynamic=True)

                # First inference triggers compilation
                with torch.no_grad():
                    _ = model_compiled(input_ids)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                compile_time = time.time() - compile_start
                print(f"Compilation time: {compile_time:.1f}s")

                results_compiled = benchmark_configuration(
                    f"torch.compile (mode={mode})",
                    model_compiled,
                    input_ids,
                    warmup=3,
                    runs=20,
                )
                all_results[f"compile_{mode}"] = results_compiled
                speedup_compiled = baseline_speed / results_compiled["seconds_per_token"]
                print(f"  → Speedup: {speedup_compiled:.2f}x over baseline")

            except Exception as e:
                print(f"  ✗ Failed: {e}")

    elif hasattr(torch, 'compile'):
        print("\n⚠️  Skipping torch.compile (Python 3.12+ not supported)")
    else:
        print("\n⚠️  Skipping torch.compile (PyTorch < 2.0)")

    # 4. CUDA Graphs
    if torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("PHASE 4: CUDA Graphs")
        print("=" * 80)

        try:
            model_cuda_graph = create_test_model()[0].to(device).eval()

            print("Capturing CUDA graph...")
            capture = CUDAGraphCapture(model_cuda_graph, input_ids.shape)

            # Benchmark using CUDA graph
            graph_results = []

            # Warmup
            for _ in range(5):
                _ = capture.run(input_ids)

            torch.cuda.synchronize()

            # Benchmark
            start = time.time()
            for _ in range(50):
                _ = capture.run(input_ids)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            results_cuda_graph = {
                "total_time_s": elapsed,
                "avg_latency_ms": (elapsed / 50) * 1000,
                "tokens_per_second": (input_ids.numel() * 50) / elapsed,
                "seconds_per_token": elapsed / (input_ids.numel() * 50),
            }

            all_results["cuda_graphs"] = results_cuda_graph
            speedup_graph = baseline_speed / results_cuda_graph["seconds_per_token"]

            print(f"  Total time: {results_cuda_graph['total_time_s']:.2f}s")
            print(f"  Avg latency: {results_cuda_graph['avg_latency_ms']:.2f}ms")
            print(f"  Tokens/sec: {results_cuda_graph['tokens_per_second']:.2f}")
            print(f"  Sec/token: {results_cuda_graph['seconds_per_token']:.4f}s")
            print(f"  → Speedup: {speedup_graph:.2f}x over baseline")

        except Exception as e:
            print(f"  ✗ CUDA graph capture failed: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Optimization Results")
    print("=" * 80)

    print(f"\n{'Configuration':<40} {'Sec/Token':<15} {'Speedup':<10}")
    print("-" * 70)

    for name, results in all_results.items():
        speedup = baseline_speed / results["seconds_per_token"]
        print(f"{name:<40} {results['seconds_per_token']:.6f}s      {speedup:.2f}x")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if sys.version_info >= (3, 12):
        print("\n[CRITICAL] Downgrade to Python 3.11 to enable torch.compile")
        print("   Expected speedup: 5-15x")

    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        if capability[0] < 7:
            print("\n[WARNING] GPU Compute Capability < 7.0 detected")
            print("   Consider upgrading GPU for better performance")
            print("   (Tensor Cores available on Compute 7.0+)")

    print("\nNext steps:")
    print("  1. Apply vectorized operations to remove Python loops")
    print("  2. Enable torch.compile with reduce-overhead mode")
    print("  3. Use CUDA graphs for static inference paths")
    print("  4. Profile with torch.profiler to find remaining bottlenecks")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
