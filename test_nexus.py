"""
Test script for NEXUS-Omega architecture.

Runs basic tests to verify all components work correctly.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    from nexus_omega.config.settings import ArchitectureConfig
    from nexus_omega.utils.sparse_utils import top_k_percentage
    from nexus_omega.routing.sparse_router import SparseDynamicRouter
    from nexus_omega.activations.kan_edges import KANLayer
    from nexus_omega.learning.predictive_coding import PredictiveCodingModule
    from nexus_omega.recurrence.adaptive_depth import AdaptiveRecurrentDepth
    from nexus_omega.memory.dual_plasticity import DualPlasticityWeights
    from nexus_omega.memory.consolidation import SynapticConsolidation
    from nexus_omega.memory.engram_cortex import EngramMemoryCortex
    from nexus_omega.encoding.sdr_encoding import SparseDistributedEncoder
    from nexus_omega.core.omega_system import NEXUSOmega, create_nexus_omega

    print("[PASS] All imports successful")


def test_individual_layers():
    """Test each layer individually."""
    print("\nTesting individual layers...")

    hidden_dim = 512
    batch, seq_len = 2, 10
    x = torch.randn(batch, seq_len, hidden_dim)

    # Layer 1: Sparse Router
    from nexus_omega.routing.sparse_router import SparseDynamicRouter
    router = SparseDynamicRouter(hidden_dim, num_experts=64, experts_per_token=2)
    out = router(x)
    assert out.output.shape == x.shape, "Sparse router shape mismatch"
    print("  [PASS] Layer 1: Sparse Dynamic Router")

    # Layer 2: KAN
    from nexus_omega.activations.kan_edges import KANLayer
    kan = KANLayer(hidden_dim, hidden_dim, grid_size=5)
    out = kan(x)
    assert out.output.shape == x.shape, "KAN shape mismatch"
    print("  [PASS] Layer 2: KAN Activation Edges")

    # Layer 3: Predictive Coding
    from nexus_omega.learning.predictive_coding import PredictiveCodingModule
    pc = PredictiveCodingModule(hidden_dim, num_iterations=3)
    out = pc(x)
    assert out.output.shape == x.shape, "Predictive coding shape mismatch"
    print("  [PASS] Layer 3: Predictive Coding")

    # Layer 4: Adaptive Recurrence
    from nexus_omega.recurrence.adaptive_depth import AdaptiveRecurrentDepth
    ard = AdaptiveRecurrentDepth(hidden_dim, min_depth=1, max_depth=4, policy="fixed")
    out = ard(x)
    assert out.output.shape == x.shape, "Adaptive recurrence shape mismatch"
    print("  [PASS] Layer 4: Adaptive Recurrent Depth")

    # Layer 5: Dual-Plasticity
    from nexus_omega.memory.dual_plasticity import DualPlasticityWeights
    dp = DualPlasticityWeights(hidden_dim, hidden_dim, plasticity_ratio=0.1)
    out = dp(x)
    assert out.output.shape == x.shape, "Dual-plasticity shape mismatch"
    print("  [PASS] Layer 5: Dual-Plasticity Weights")

    # Layer 7: Engram Memory
    from nexus_omega.memory.engram_cortex import EngramMemoryCortex
    mem = EngramMemoryCortex(total_params=1_000_000, active_params=100_000, hidden_dim=hidden_dim)
    out = mem(x)
    assert out.shape == x.shape, "Engram memory shape mismatch"
    print("  [PASS] Layer 7: Engram Memory Cortex")

    # Layer 8: SDR Encoding
    from nexus_omega.encoding.sdr_encoding import SparseDistributedEncoder
    sdr = SparseDistributedEncoder(hidden_dim, sdr_dim=5000, sparsity=0.02)
    out = sdr(x)
    assert out.shape[:-1] == x.shape[:-1], "SDR encoding shape mismatch"
    print("  [PASS] Layer 8: Sparse Distributed Input")

    print("[PASS] All layers working correctly")


def test_full_model():
    """Test the full NEXUS-Omega model."""
    print("\nTesting full NEXUS-Omega model...")

    from nexus_omega.core.omega_system import create_nexus_omega

    # Create small model
    model = create_nexus_omega("small", vocab_size=1000)

    # Test inference
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    output = model(input_ids)

    assert output.logits.shape == (batch_size, seq_len, 1000), "Output shape mismatch"
    assert output.hidden_states.shape == (batch_size, seq_len, model.arch_config.hidden_dim), "Hidden states shape mismatch"

    print(f"  [PASS] Forward pass successful")
    print(f"  Parameters: {model.count_parameters()['total']:,}")
    print(f"  Memory: {model.get_memory_footprint()['active_gb']:.2f} GB")

    # Test online learning
    output = model(input_ids, enable_online_learning=True)
    print(f"  [PASS] Online learning enabled")

    # Test consolidation
    report = model.consolidate()
    print(f"  [PASS] Consolidation successful")

    print("[PASS] Full model working correctly")


def test_utilities():
    """Test utility functions."""
    print("\nTesting utilities...")

    from nexus_omega.utils.sparse_utils import top_k_percentage, compute_sparsity
    from nexus_omega.utils.efficiency_metrics import EfficiencyTracker

    # Test sparse utils
    x = torch.randn(100, 512)
    sparse_x = top_k_percentage(x, k=0.05)
    sparsity = compute_sparsity(sparse_x)
    assert sparsity > 0.9, "Sparsity too low"
    print("  [PASS] Sparse utilities")

    # Test efficiency tracker
    tracker = EfficiencyTracker()
    tracker.start()
    _ = torch.randn(100, 512) @ torch.randn(512, 512)
    metrics = tracker.stop(num_tokens=100)
    assert metrics.tokens_per_second > 0, "Tokens per second should be positive"
    print("  [PASS] Efficiency metrics")

    print("[PASS] Utilities working correctly")


def test_optimization():
    """Test optimization utilities."""
    print("\nTesting optimization...")

    from nexus_omega.deploy.optimizer import quantize_model, prune_model

    # Create simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
    )

    # Test quantization
    quantized = quantize_model(model, "int8")
    print("  [PASS] Quantization")

    # Test pruning
    pruned = prune_model(model, sparsity=0.9)
    print("  [PASS] Pruning")

    print("[PASS] Optimization working correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("NEXUS-Omega Test Suite")
    print("=" * 60)

    try:
        test_imports()
        test_individual_layers()
        test_full_model()
        test_utilities()
        test_optimization()

        print("\n" + "=" * 60)
        print("[PASS] ALL TESTS PASSED!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
