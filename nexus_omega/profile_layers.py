"""
Comprehensive Layer Profiling for NEXUS-Omega
Uses torch.cuda.Event() for accurate GPU timing measurements.
"""

import os
import sys
import torch
import torch.nn as nn
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nexus_omega.config.settings import ArchitectureConfig
from nexus_omega.core.omega_system import NEXUSOmega, NEXUSOmegaBlock
from nexus_omega.routing.sparse_router import SparseDynamicRouter
from nexus_omega.activations.kan_edges import KANLayer
from nexus_omega.learning.predictive_coding import PredictiveCodingModule
from nexus_omega.recurrence.adaptive_depth import AdaptiveRecurrentDepth
from nexus_omega.memory.dual_plasticity import DualPlasticityWeights
from nexus_omega.encoding.sdr_encoding import SparseDistributedEncoder

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration for profiling
PROFILE_CONFIG = ArchitectureConfig(
    hidden_dim=512,
    num_layers=4,
    num_experts=1024,
    experts_per_token=2,
    activation_sparsity=0.01,
    kan_grid_size=5,
    kan_order=3,
    pc_num_iterations=5,
    pc_learning_rate=0.1,
    pc_decay=0.01,
    min_depth=1,
    max_depth=8,
    depth_policy="learned",
    plasticity_ratio=0.1,
    plastic_lr=0.01,
    base_frozen=True,
    consolidation_threshold=0.1,
    ewc_lambda=1000.0,
    consolidation_frequency=100,
    sdr_dim=1024,
    sdr_sparsity=0.02,
)

# Batch sizes to test
BATCH_SIZES = [1, 2, 4, 8]
SEQ_LENS = [32, 64, 128, 256]

# Number of warmup runs and measurement runs
WARMUP_RUNS = 10
MEASURE_RUNS = 50


@dataclass
class LayerTiming:
    """Timing results for a single layer."""
    layer_name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    num_runs: int
    batch_size: int
    seq_len: int


def time_with_cuda_events(module: nn.Module, inputs: tuple, num_runs: int = MEASURE_RUNS, warmup: int = WARMUP_RUNS) -> List[float]:
    """Time a module using CUDA events for accurate GPU timing."""
    if not torch.cuda.is_available():
        # Fallback to CPU timing
        times = []
        module.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(warmup):
                _ = module(*inputs)

            # Measure
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = module(*inputs)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
        return times

    # CUDA timing with events
    module.eval()
    torch.cuda.synchronize()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(*inputs)

    torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = module(*inputs)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            times.append(elapsed_ms)

    return times


def profile_layer(layer: nn.Module, layer_name: str, inputs: tuple, batch_size: int, seq_len: int) -> LayerTiming:
    """Profile a single layer and return timing statistics."""
    print(f"  Profiling {layer_name} (batch={batch_size}, seq={seq_len})...")
    times = time_with_cuda_events(layer, inputs)

    return LayerTiming(
        layer_name=layer_name,
        mean_ms=sum(times) / len(times),
        std_ms=torch.tensor(times).std().item(),
        min_ms=min(times),
        max_ms=max(times),
        num_runs=len(times),
        batch_size=batch_size,
        seq_len=seq_len,
    )


def profile_sparse_router(batch_size: int, seq_len: int) -> LayerTiming:
    """Profile the Sparse Dynamic Router."""
    layer = SparseDynamicRouter(
        hidden_dim=PROFILE_CONFIG.hidden_dim,
        num_experts=PROFILE_CONFIG.num_experts,
        experts_per_token=PROFILE_CONFIG.experts_per_token,
        sparsity=PROFILE_CONFIG.activation_sparsity,
    ).to(device)

    x = torch.randn(batch_size, seq_len, PROFILE_CONFIG.hidden_dim, device=device)

    return profile_layer(layer, "SparseDynamicRouter", (x,), batch_size, seq_len)


def profile_kan_layer(batch_size: int, seq_len: int) -> LayerTiming:
    """Profile the KAN Layer."""
    layer = KANLayer(
        in_features=PROFILE_CONFIG.hidden_dim,
        out_features=PROFILE_CONFIG.hidden_dim,
        grid_size=PROFILE_CONFIG.kan_grid_size,
        spline_order=PROFILE_CONFIG.kan_order,
    ).to(device)

    x = torch.randn(batch_size, seq_len, PROFILE_CONFIG.hidden_dim, device=device)

    return profile_layer(layer, "KANLayer", (x,), batch_size, seq_len)


def profile_predictive_coding(batch_size: int, seq_len: int) -> LayerTiming:
    """Profile the Predictive Coding Module."""
    layer = PredictiveCodingModule(
        hidden_dim=PROFILE_CONFIG.hidden_dim,
        num_iterations=PROFILE_CONFIG.pc_num_iterations,
        learning_rate=PROFILE_CONFIG.pc_learning_rate,
        decay=PROFILE_CONFIG.pc_decay,
    ).to(device)

    x = torch.randn(batch_size, seq_len, PROFILE_CONFIG.hidden_dim, device=device)

    return profile_layer(layer, "PredictiveCodingModule", (x,), batch_size, seq_len)


def profile_adaptive_depth(batch_size: int, seq_len: int) -> LayerTiming:
    """Profile the Adaptive Recurrent Depth."""
    layer = AdaptiveRecurrentDepth(
        hidden_dim=PROFILE_CONFIG.hidden_dim,
        min_depth=PROFILE_CONFIG.min_depth,
        max_depth=PROFILE_CONFIG.max_depth,
        policy=PROFILE_CONFIG.depth_policy,
    ).to(device)

    x = torch.randn(batch_size, seq_len, PROFILE_CONFIG.hidden_dim, device=device)

    return profile_layer(layer, "AdaptiveRecurrentDepth", (x,), batch_size, seq_len)


def profile_dual_plasticity(batch_size: int, seq_len: int) -> LayerTiming:
    """Profile the Dual-Plasticity Weights."""
    layer = DualPlasticityWeights(
        in_features=PROFILE_CONFIG.hidden_dim,
        out_features=PROFILE_CONFIG.hidden_dim,
        plasticity_ratio=PROFILE_CONFIG.plasticity_ratio,
        plastic_lr=PROFILE_CONFIG.plastic_lr,
        base_frozen=PROFILE_CONFIG.base_frozen,
    ).to(device)

    x = torch.randn(batch_size, seq_len, PROFILE_CONFIG.hidden_dim, device=device)

    return profile_layer(layer, "DualPlasticityWeights", (x,), batch_size, seq_len)


def profile_sdr_encoder(batch_size: int, seq_len: int) -> LayerTiming:
    """Profile the SDR Encoder."""
    layer = SparseDistributedEncoder(
        input_dim=PROFILE_CONFIG.hidden_dim,
        sdr_dim=PROFILE_CONFIG.sdr_dim,
        sparsity=PROFILE_CONFIG.sdr_sparsity,
    ).to(device)

    x = torch.randn(batch_size, seq_len, PROFILE_CONFIG.hidden_dim, device=device)

    return profile_layer(layer, "SparseDistributedEncoder", (x,), batch_size, seq_len)


def profile_full_block(batch_size: int, seq_len: int) -> LayerTiming:
    """Profile a full NEXUS-Omega block (all layers combined)."""
    block = NEXUSOmegaBlock(
        hidden_dim=PROFILE_CONFIG.hidden_dim,
        config=PROFILE_CONFIG,
        block_id=0,
    ).to(device)

    x = torch.randn(batch_size, seq_len, PROFILE_CONFIG.hidden_dim, device=device)

    return profile_layer(block, "NEXUSOmegaBlock", (x,), batch_size, seq_len)


def profile_full_model(batch_size: int, seq_len: int) -> LayerTiming:
    """Profile the full NEXUS-Omega model."""
    model = NEXUSOmega(
        vocab_size=50000,
        arch_config=PROFILE_CONFIG,
    ).to(device)

    input_ids = torch.randint(0, 50000, (batch_size, seq_len), device=device)

    return profile_layer(model, "NEXUSOmega", (input_ids,), batch_size, seq_len)


def run_comprehensive_profile():
    """Run comprehensive profiling across all layers and configurations."""
    results = {}

    print("=" * 70)
    print("NEXUS-Omega Layer Profiling")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Config: hidden_dim={PROFILE_CONFIG.hidden_dim}, experts={PROFILE_CONFIG.num_experts}")
    print(f"Profile iterations: {MEASURE_RUNS} runs after {WARMUP_RUNS} warmup")
    print()

    # Test configurations
    test_configs = [
        (1, 32),
        (1, 64),
        (1, 128),
        (2, 32),
        (2, 64),
        (4, 32),
    ]

    for batch_size, seq_len in test_configs:
        config_key = f"batch{batch_size}_seq{seq_len}"
        results[config_key] = {}

        print(f"\n{'='*70}")
        print(f"Configuration: batch_size={batch_size}, seq_len={seq_len}")
        print(f"{'='*70}")

        # Profile individual layers
        layers_to_profile = [
            ("SparseDynamicRouter", profile_sparse_router),
            ("KANLayer", profile_kan_layer),
            ("PredictiveCodingModule", profile_predictive_coding),
            ("AdaptiveRecurrentDepth", profile_adaptive_depth),
            ("DualPlasticityWeights", profile_dual_plasticity),
            ("SparseDistributedEncoder", profile_sdr_encoder),
        ]

        layer_results = {}
        for layer_name, profile_fn in layers_to_profile:
            try:
                timing = profile_fn(batch_size, seq_len)
                layer_results[layer_name] = timing
                print(f"  {layer_name}: {timing.mean_ms:.2f}ms ± {timing.std_ms:.2f}ms (min={timing.min_ms:.2f}, max={timing.max_ms:.2f})")
            except Exception as e:
                print(f"  {layer_name}: ERROR - {e}")
                layer_results[layer_name] = None

        # Profile full block
        try:
            block_timing = profile_full_block(batch_size, seq_len)
            layer_results["NEXUSOmegaBlock"] = block_timing
            print(f"  NEXUSOmegaBlock: {block_timing.mean_ms:.2f}ms ± {block_timing.std_ms:.2f}ms")
        except Exception as e:
            print(f"  NEXUSOmegaBlock: ERROR - {e}")
            layer_results["NEXUSOmegaBlock"] = None

        # Profile full model
        try:
            model_timing = profile_full_model(batch_size, seq_len)
            layer_results["NEXUSOmega"] = model_timing
            print(f"  NEXUSOmega (full): {model_timing.mean_ms:.2f}ms ± {model_timing.std_ms:.2f}ms")
        except Exception as e:
            print(f"  NEXUSOmega (full): ERROR - {e}")
            layer_results["NEXUSOmega"] = None

        results[config_key] = layer_results

    return results


def analyze_results(results: Dict) -> Dict:
    """Analyze profiling results to identify bottlenecks."""
    analysis = {
        "bottleneck_layers": [],
        "layer_breakdown": {},
        "scaling_analysis": {},
        "recommendations": [],
    }

    # Find the most time-consuming layer per configuration
    for config_key, layer_results in results.items():
        config_analysis = {}
        max_time = 0
        bottleneck = None

        for layer_name, timing in layer_results.items():
            if timing is None:
                continue
            config_analysis[layer_name] = {
                "mean_ms": timing.mean_ms,
                "std_ms": timing.std_ms,
                "min_ms": timing.min_ms,
                "max_ms": timing.max_ms,
            }
            if timing.mean_ms > max_time:
                max_time = timing.mean_ms
                bottleneck = layer_name

        analysis["layer_breakdown"][config_key] = config_analysis
        if bottleneck:
            analysis["bottleneck_layers"].append({
                "config": config_key,
                "bottleneck": bottleneck,
                "time_ms": max_time,
            })

    # Identify consistent bottlenecks across configs
    bottleneck_counts = {}
    for b in analysis["bottleneck_layers"]:
        bottleneck_counts[b["bottleneck"]] = bottleneck_counts.get(b["bottleneck"], 0) + 1

    analysis["consistent_bottlenecks"] = sorted(bottleneck_counts.items(), key=lambda x: x[1], reverse=True)

    return analysis


def print_detailed_report(results: Dict, analysis: Dict):
    """Print a detailed profiling report."""
    print("\n" + "=" * 70)
    print("DETAILED PROFILING REPORT")
    print("=" * 70)

    # Layer timing breakdown per config
    for config_key, layer_results in results.items():
        print(f"\n--- {config_key} ---")
        for layer_name, timing in layer_results.items():
            if timing:
                print(f"  {layer_name:35s}: {timing.mean_ms:8.2f}ms ± {timing.std_ms:.2f}ms  "
                      f"(min={timing.min_ms:.2f}, max={timing.max_ms:.2f})")

    # Bottleneck analysis
    print("\n" + "-" * 70)
    print("BOTTLENECK ANALYSIS")
    print("-" * 70)

    for b in analysis["bottleneck_layers"]:
        print(f"  {b['config']}: {b['bottleneck']} = {b['time_ms']:.2f}ms")

    print("\nConsistent bottlenecks across configurations:")
    for layer, count in analysis["consistent_bottlenecks"]:
        print(f"  {layer}: {count} times")

    # Calculate speedup potential
    print("\n" + "-" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("-" * 70)

    # Analyze specific layers for root causes
    print("\n1. PREDICTIVE CODING MODULE:")
    print("   - 5 iterations of predictor + error_corrector forward passes")
    print("   - Hebbian update: matmul(pre.T @ post) on [batch*seq, hidden] every iteration")
    print("   - O(num_iterations * batch * seq * hidden^2) complexity")
    print("   - Suggestion: Reduce iterations, vectorize Hebbian update, or use low-rank approx")

    print("\n2. ADAPTIVE RECURRENT DEPTH:")
    print("   - Nested loops: for batch -> for seq -> for depth (1-8)")
    print("   - .item() call on line 97 causes GPU-CPU synchronization!")
    print("   - GRUCell called sequentially per token, not batched")
    print("   - Suggestion: Vectorize across batch+seq, remove .item(), use batched RNN")

    print("\n3. SPARSE DYNAMIC ROUTER:")
    print("   - Sequential Python loops over tokens (line 90) and experts (line 97)")
    print("   - Expert weights indexed individually: self.expert_down[expert_id]")
    print("   - No vectorization - each token processed separately")
    print("   - Suggestion: Batch expert computation, use scatter/gather, or Triton kernel")

    print("\n" + "-" * 70)
    print("PRIORITY RANKING (by estimated speedup impact)")
    print("-" * 70)
    print("1. SPARSE ROUTER: Vectorize token loop -> 10-50x speedup")
    print("2. ADAPTIVE DEPTH: Remove .item() + batch recurrent -> 5-20x speedup")
    print("3. PREDICTIVE CODING: Vectorize Hebbian update -> 2-5x speedup")
    print("4. FULL MODEL: torch.compile + fused kernels -> 2-3x speedup")


def save_results(results: Dict, analysis: Dict, output_path: str = "profiling_results.json"):
    """Save profiling results to JSON."""
    # Convert LayerTiming objects to dicts
    serializable_results = {}
    for config_key, layer_results in results.items():
        serializable_results[config_key] = {}
        for layer_name, timing in layer_results.items():
            if timing:
                serializable_results[config_key][layer_name] = {
                    "layer_name": timing.layer_name,
                    "mean_ms": timing.mean_ms,
                    "std_ms": timing.std_ms,
                    "min_ms": timing.min_ms,
                    "max_ms": timing.max_ms,
                    "num_runs": timing.num_runs,
                    "batch_size": timing.batch_size,
                    "seq_len": timing.seq_len,
                }

    with open(output_path, 'w') as f:
        json.dump({
            "results": serializable_results,
            "analysis": analysis,
            "config": {
                "hidden_dim": PROFILE_CONFIG.hidden_dim,
                "num_experts": PROFILE_CONFIG.num_experts,
                "experts_per_token": PROFILE_CONFIG.experts_per_token,
                "pc_num_iterations": PROFILE_CONFIG.pc_num_iterations,
                "max_depth": PROFILE_CONFIG.max_depth,
                "device": str(device),
            }
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    # Run comprehensive profiling
    results = run_comprehensive_profile()

    # Analyze
    analysis = analyze_results(results)

    # Print report
    print_detailed_report(results, analysis)

    # Save results
    save_results(results, analysis)

    print("\n✓ Profiling complete!")