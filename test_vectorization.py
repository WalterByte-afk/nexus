"""
Quick test to verify vectorized implementations work correctly.
"""
import torch
import sys
import os
sys.path.insert(0, 'C:/Users/abdullah/Desktop/ai')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from nexus_omega.routing.sparse_router import SparseDynamicRouter
from nexus_omega.recurrence.adaptive_depth import AdaptiveRecurrentDepth
from nexus_omega.learning.predictive_coding import PredictiveCodingModule

print("=" * 60)
print("Testing Vectorized Implementations")
print("=" * 60)

# Test 1: Sparse Router (vectorized expert computation)
print("\n1. Testing SparseDynamicRouter (vectorized)...")
try:
    router = SparseDynamicRouter(hidden_dim=512, num_experts=64, experts_per_token=2)
    x = torch.randn(2, 10, 512)
    output = router(x)
    assert output.output.shape == x.shape, f"Shape mismatch: {output.output.shape} != {x.shape}"
    print(f"   [OK] Input: {x.shape} -> Output: {output.output.shape}")
    print(f"   [OK] Routing entropy: {output.metrics['routing_entropy']:.4f}")
    print("   SUCCESS: Vectorized expert routing works!")
except Exception as e:
    print(f"   [FAIL] FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Adaptive Depth (vectorized with torch.where)
print("\n2. Testing AdaptiveRecurrentDepth (vectorized)...")
try:
    ard = AdaptiveRecurrentDepth(hidden_dim=512, min_depth=1, max_depth=8, policy="learned")
    x = torch.randn(2, 10, 512)
    output = ard(x)
    assert output.output.shape == x.shape, f"Shape mismatch: {output.output.shape} != {x.shape}"
    print(f"   [OK] Input: {x.shape} -> Output: {output.output.shape}")
    print(f"   [OK] Avg depth: {output.metrics['avg_depth']:.2f}")
    print(f"   [OK] Depth range: [{output.metrics['min_depth_used']}, {output.metrics['max_depth_used']}]")
    print("   SUCCESS: Vectorized adaptive depth works!")
except Exception as e:
    print(f"   [FAIL] FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Predictive Coding (reduced iterations for inference)
print("\n3. Testing PredictiveCodingModule (inference optimization)...")
try:
    pc = PredictiveCodingModule(hidden_dim=512, num_iterations=5, inference_iterations=1)
    x = torch.randn(2, 10, 512)

    # Test training mode (should use 5 iterations)
    pc.train()
    output_train = pc(x)
    assert output_train.metrics['iterations'] == 5, f"Training iterations should be 5, got {output_train.metrics['iterations']}"
    print(f"   [OK] Training mode: {output_train.metrics['iterations']} iterations")

    # Test inference mode (should use 1 iteration)
    pc.eval()
    output_eval = pc(x)
    assert output_eval.metrics['iterations'] == 1, f"Inference iterations should be 1, got {output_eval.metrics['iterations']}"
    print(f"   [OK] Inference mode: {output_eval.metrics['iterations']} iteration (5x speedup!)")
    print(f"   [OK] Prediction error: {output_eval.metrics['prediction_error']:.6f}")
    print("   SUCCESS: Reduced iterations for inference works!")
except Exception as e:
    print(f"   [FAIL] FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All vectorization tests completed!")
print("=" * 60)
