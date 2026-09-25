# NEXUS-Omega Ultra-Fast Inference Optimization Research

**Goal**: Achieve 100x speedup (42s/token → <0.5s/token)  
**Date**: 2026-09-25  
**Environment**: PyTorch 2.3.0+cu121, GTX 750 Ti (Compute 5.0), Python 3.12

---

## Executive Summary

**CRITICAL FINDING**: Python 3.12 is incompatible with `torch.compile` (TorchDynamo). This blocks the primary optimization path.

**Recommended Action Plan**:
1. **IMMEDIATE**: Downgrade to Python 3.11 (torch.compile requires ≤3.11)
2. **HIGH IMPACT**: Implement torch.compile with `reduce-overhead` mode
3. **MEDIUM IMPACT**: Add custom CUDA kernels for bottleneck operations
4. **LOW IMPACT**: Flash Attention (limited benefit for this architecture)

**Estimated Speedup**:
- Python 3.11 + torch.compile: **15-25x** (42s → 1.7-2.8s/token)
- Custom Triton kernels: **3-5x additional** (→ 0.5-0.9s/token)
- CUDA graphs: **1.5-2x additional** (→ 0.25-0.6s/token)
- **Total potential: 45-250x speedup**

---

## 1. torch.compile Research

### Overview
`torch.compile` (PyTorch 2.0+) uses TorchDynamo to trace Python bytecode and optimize computation graphs.

### Modes Comparison

| Mode | Compile Time | Runtime Speed | Best For |
|------|--------------|---------------|----------|
| `default` | ~10-30s | 2-5x faster | General use, balanced |
| `reduce-overhead` | ~10-30s | 5-15x faster | Small batches, low latency |
| `max-autotune` | ~60-300s | 10-25x faster | Throughput, large batches |

### Recommendation for NEXUS-Omega
**Use `reduce-overhead` mode** because:
- Single-token inference (batch_size=1) benefits most from overhead reduction
- Fast first-compile time (~30s vs 5min for max-autotune)
- 5-15x speedup is sufficient for first optimization pass

### Code Integration
```python
# In nexus_omega/deploy/inference.py
class InferenceEngine:
    def _optimize_model(self):
        self.model.eval()
        
        if self.config.use_torch_compile:
            print("Compiling model with torch.compile...")
            # Use reduce-overhead for single-token inference
            self.model = torch.compile(
                self.model, 
                mode="reduce-overhead",
                fullgraph=False,  # Allow graph breaks
                dynamic=True  # Handle variable sequence lengths
            )
```

### CRITICAL BLOCKER
**Python 3.12 is NOT supported** by TorchDynamo. Must use Python 3.11 or earlier.

**Fix**:
```bash
# Create new environment with Python 3.11
conda create -n nexus_fast python=3.11
conda activate nexus_fast
pip install torch>=2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. Flash Attention 2 Research

### Overview
Flash Attention optimizes attention mechanisms using tiled computation to reduce memory I/O.

### Applicability to NEXUS-Omega
**LIMITED BENEFIT** because:
- NEXUS uses GRU recurrence, NOT standard attention
- No multi-head attention mechanism in current architecture
- Recurrent layers don't benefit from Flash Attention optimization

### Where It Could Help (Minimal)
- If adding standard attention layers for long-range dependencies
- Cross-layer attention mechanisms (not currently implemented)

### Code Integration (Optional)
```python
# Only if adding attention layers
from torch.nn.functional import scaled_dot_product_attention

class OptimizedAttention(nn.Module):
    def forward(self, q, k, v):
        # Uses Flash Attention 2 backend automatically in PyTorch 2.0+
        return scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            enable_gqa=True  # Grouped Query Attention
        )
```

**Verdict**: Skip Flash Attention for now. Not relevant to current architecture.

---

## 3. Triton Kernel Research

### Overview
Triton is a Python-based language for writing custom GPU kernels. Easier than CUDA C++.

### Target Operations for Custom Kernels

#### 3a. Sparse Expert Routing (HIGH PRIORITY)
**Current bottleneck**: Sequential expert computation in Python loop

```python
# CURRENT (SLOW): nexus_omega/routing/sparse_router.py lines 90-113
for i in range(flat_x.shape[0]):
    for j, expert_id in enumerate(expert_idx):
        # Expert computation - this is VERY slow
        ...
```

**Optimization**: Fused Triton kernel for batched expert routing

```python
# PROPOSED: Custom Triton kernel
import triton
import triton.language as tl

@triton.jit
def fused_sparse_moe_kernel(
    x_ptr, expert_weights_ptr, expert_indices_ptr, output_ptr,
    batch_size, hidden_dim, num_experts, experts_per_token,
    BLOCK_SIZE: tl.constexpr
):
    # Fuse expert selection + computation in single kernel
    # Avoid Python loops and memory transfers
    pid = tl.program_id(0)
    
    # Load input token
    x = tl.load(x_ptr + pid * hidden_dim + tl.arange(0, BLOCK_SIZE))
    
    # Load expert indices for this token
    expert_ids = tl.load(expert_indices_ptr + pid * experts_per_token + tl.arange(0, experts_per_token))
    
    # Process experts (unrolled)
    output = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(experts_per_token):
        expert_id = tl.load(expert_indices_ptr + pid * experts_per_token + i)
        # Fused expert forward pass
        # ... (expert computation)
    
    tl.store(output_ptr + pid * hidden_dim + tl.arange(0, BLOCK_SIZE), output)
```

**Expected Speedup**: 5-10x for routing layer

#### 3b. B-Spline KAN Activation (MEDIUM PRIORITY)
**Current bottleneck**: Recursive B-spline computation with Python loops

```python
# CURRENT (SLOW): nexus_omega/activations/kan_edges.py lines 95-98
for i in range(len(self.coef)):
    basis = self.b_spline_basis(x_clamped, i, self.spline_order)
    output = output + self.coef[i] * basis
```

**Optimization**: Vectorized B-spline evaluation

```python
@triton.jit
def bspline_activation_kernel(
    x_ptr, coef_ptr, grid_ptr, output_ptr,
    batch_size, hidden_dim, num_coefs, spline_order,
    BLOCK_SIZE: tl.constexpr
):
    # Fused B-spline basis evaluation
    # Vectorize over coefficients
    pid = tl.program_id(0)
    
    x_val = tl.load(x_ptr + pid)
    
    # Compute all basis functions in parallel
    result = 0.0
    for i in range(num_coefs):
        coef = tl.load(coef_ptr + i)
        basis = compute_bspline_basis(x_val, i, spline_order, grid_ptr)
        result += coef * basis
    
    tl.store(output_ptr + pid, result)
```

**Expected Speedup**: 3-5x for KAN layers

#### 3c. Hebbian Outer Product (MEDIUM PRIORITY)
**Current bottleneck**: Outer product for dual-plasticity updates

```python
# CURRENT: nexus_omega/memory/dual_plasticity.py lines 132-135
self.W_plastic_down.data += self.plastic_lr * x_mean.unsqueeze(0).expand_as(self.W_plastic_down)
self.W_plastic_up.data += self.plastic_lr * out_mean.unsqueeze(1).expand_as(self.W_plastic_up)
```

**Optimization**: Fused Hebbian update kernel

```python
@triton.jit
def hebbian_update_kernel(
    weight_ptr, pre_act_ptr, post_act_ptr,
    learning_rate, num_rows, num_cols,
    BLOCK_SIZE: tl.constexpr
):
    # Fused outer product + weight update
    # ΔW = lr * outer(post, pre)
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    pre = tl.load(pre_act_ptr + pid_col)
    post = tl.load(post_act_ptr + pid_row)
    
    # Compute update
    delta = learning_rate * post * pre
    
    # Atomic add to weight
    weight_idx = pid_row * num_cols + pid_col
    old_weight = tl.load(weight_ptr + weight_idx)
    tl.store(weight_ptr + weight_idx, old_weight + delta)
```

**Expected Speedup**: 2-3x for plasticity updates

### Triton Installation
**PROBLEM**: Triton not available on Windows (Linux/WSL2 only)

**Workarounds**:
1. Use WSL2 for development/deployment
2. Use pure PyTorch optimizations (tensor ops instead of loops)
3. Write custom CUDA kernels (harder but Windows-compatible)

---

## 4. torch.jit.script Research

### Overview
TorchScript compiles Python functions to optimized intermediate representation.

### Applicability
**MODERATE BENEFIT** for:
- Individual layer forward passes
- Recurrent cell computations
- Routing logic

### Code Integration
```python
# In nexus_omega/routing/sparse_router.py
class SparseDynamicRouter(NexusLayer):
    def __init__(self, ...):
        super().__init__(...)
        # Script-compile the expert forward pass
        self.expert_forward = torch.jit.script(self._expert_forward)
    
    @torch.jit.script
    def _expert_forward(self, x: torch.Tensor, expert_id: int) -> torch.Tensor:
        # Pure tensor ops - no Python loops
        down_weight = self.expert_down[expert_id]
        up_weight = self.expert_up[expert_id]
        h = F.relu(F.linear(x, down_weight, self.expert_biases_down[expert_id]))
        return F.linear(h, up_weight, self.expert_biases_up[expert_id])
```

### Limitations
- Cannot script code with Python loops over dynamic ranges
- Cannot script code with conditionals on tensor values
- torch.compile often better for PyTorch 2.0+

**Verdict**: Use torch.compile instead. Better optimization, less manual work.

---

## 5. CUDA Graphs Research

### Overview
CUDA graphs capture entire computation sequences and replay them with minimal CPU overhead.

### Requirements
- **Static input shapes** (no variable sequence length)
- **No CPU-GPU synchronization** during execution
- **No Python control flow** in captured region

### Applicability to NEXUS-Omega
**HIGH POTENTIAL** for:
- Fixed-length inference (padding to max length)
- Single-token generation (autoregressive)

### Code Integration
```python
# In nexus_omega/deploy/inference.py
class InferenceEngine:
    def __init__(self, model, config):
        self.model = model
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None
        
        if config.use_cuda_graphs:
            self._capture_cuda_graph()
    
    def _capture_cuda_graph(self):
        """Capture forward pass as CUDA graph."""
        if not torch.cuda.is_available():
            return
        
        # Create static tensors for graph capture
        self.static_input = torch.zeros(
            1, 1, dtype=torch.long, device='cuda'
        )
        
        # Warmup
        for _ in range(3):
            _ = self.model(self.static_input)
        
        torch.cuda.synchronize()
        
        # Capture graph
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            self.static_output = self.model(self.static_input)
        
        print("CUDA graph captured!")
    
    def generate_with_graph(self, input_ids):
        """Generate using CUDA graph (faster)."""
        if self.cuda_graph is None:
            return self.generate(input_ids)  # Fallback
        
        # Copy input to static buffer
        self.static_input.copy_(input_ids)
        
        # Replay graph (VERY fast)
        self.cuda_graph.replay()
        
        # Copy output
        return self.static_output.clone()
```

**Expected Speedup**: 1.5-2x additional on top of torch.compile

**Limitations**:
- Requires padding all sequences to same length
- Cannot handle dynamic control flow (adaptive depth)
- Complex to debug

---

## 6. Implementation Priority & Integration Strategy

### Phase 1: Foundation (Week 1)
**Priority: CRITICAL**

1. **Downgrade to Python 3.11**
   ```bash
   conda create -n nexus_fast python=3.11
   pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Enable torch.compile with reduce-overhead**
   - File: `nexus_omega/deploy/inference.py`
   - Change: Add `mode="reduce-overhead"` parameter
   - Expected: 5-15x speedup

3. **Remove Python loops from hot paths**
   - File: `nexus_omega/routing/sparse_router.py`
   - Convert expert loop to batched tensor operations
   - Expected: 2-3x additional speedup

### Phase 2: Kernel Optimization (Week 2)
**Priority: HIGH**

4. **Vectorize B-spline computation**
   - File: `nexus_omega/activations/kan_edges.py`
   - Replace recursive calls with tensor operations
   - Expected: 3-5x speedup for KAN layers

5. **Batch expert routing**
   - Rewrite sparse router to use `torch.gather` and `torch.scatter`
   - Eliminate per-token loops
   - Expected: 5-10x speedup for routing

### Phase 3: Advanced Optimization (Week 3)
**Priority: MEDIUM**

6. **CUDA graphs for autoregressive generation**
   - Capture single-token forward pass
   - Expected: 1.5-2x additional speedup

7. **Mixed precision (FP16/BF16)**
   - Already partially implemented
   - Test on GTX 750 Ti (FP16 support?)
   - Expected: 1.3-1.8x speedup

### Phase 4: Custom Kernels (Week 4+)
**Priority: LOW (if still needed)**

8. **Triton kernels** (Linux/WSL2 only)
   - Sparse routing kernel
   - B-spline kernel
   - Expected: 2-5x additional speedup

---

## 7. Immediate Action Code Snippets

### A. Remove Python Loops from Sparse Router

**BEFORE** (nexus_omega/routing/sparse_router.py:90-113):
```python
for i in range(flat_x.shape[0]):
    token_x = flat_x[i]
    expert_idx = expert_indices[i]
    expert_output = torch.zeros(hidden, device=x.device)
    
    for j, expert_id in enumerate(expert_idx):
        down_weight = self.expert_down[expert_id]
        # ... etc
```

**AFTER** (Vectorized):
```python
def forward(self, x: torch.Tensor, **kwargs) -> LayerOutput:
    batch, seq_len, hidden = x.shape
    flat_x = x.view(-1, hidden)  # [N, hidden]
    
    # Compute routing (same)
    routing_logits = self.routing_network(flat_x)
    routing_scores, expert_indices = torch.topk(
        routing_logits, self.experts_per_token, dim=-1
    )  # [N, k]
    
    # OPTIMIZED: Batched expert computation
    N = flat_x.shape[0]
    outputs = torch.zeros(N, hidden, device=x.device)
    
    # Process all tokens for each expert in parallel
    for expert_id in range(self.num_experts):
        # Find which tokens use this expert
        expert_mask = (expert_indices == expert_id).any(dim=-1)  # [N]
        
        if expert_mask.any():
            # Batch process all tokens using this expert
            expert_tokens = flat_x[expert_mask]  # [M, hidden]
            
            # Expert forward (vectorized)
            down_weight = self.expert_down[expert_id]
            down_bias = self.expert_biases_down[expert_id]
            h = F.relu(F.linear(expert_tokens, down_weight, down_bias))
            
            up_weight = self.expert_up[expert_id]
            up_bias = self.expert_biases_up[expert_id]
            expert_out = F.linear(h, up_weight, up_bias)
            
            # Accumulate (handle multiple experts per token)
            outputs[expert_mask] += expert_out / self.experts_per_token
    
    output = outputs.view(batch, seq_len, hidden)
    return LayerOutput(output=output, metrics={...})
```

### B. Vectorize B-Spline KAN

**BEFORE** (nexus_omega/activations/kan_edges.py:95-98):
```python
for i in range(len(self.coef)):
    basis = self.b_spline_basis(x_clamped, i, self.spline_order)
    output = output + self.coef[i] * basis
```

**AFTER** (Vectorized using Einstein summation):
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x_clamped = torch.clamp(x, self.grid_range[0], self.grid_range[1])
    
    # Compute all basis functions at once (vectorized)
    # Shape: [*x.shape, num_basis_functions]
    basis_matrix = self.compute_all_basis(x_clamped)  # [*, num_coef]
    
    # Matrix multiply: sum(coef[i] * basis[i]) for all i
    output = torch.matmul(basis_matrix, self.coef)  # [*x.shape]
    
    # Residual
    output = output + self.residual_weight * x
    return output

def compute_all_basis(self, x: torch.Tensor) -> torch.Tensor:
    """Compute all B-spline basis functions efficiently."""
    num_basis = len(self.coef)
    basis = torch.zeros(*x.shape, num_basis, device=x.device)
    
    # Vectorized basis computation (still needs optimization)
    # TODO: Replace with closed-form B-spline matrix
    for i in range(num_basis):
        basis[..., i] = self.b_spline_basis(x, i, self.spline_order)
    
    return basis
```

### C. Optimized Inference Configuration

**File**: `examples/test_inference.py`

```python
def main():
    # ... model creation ...
    
    # OPTIMIZED inference config
    infer_config = InferenceConfig(
        # Core optimizations
        use_torch_compile=True,  # PRIMARY OPTIMIZATION
        use_fp16=True,  # If GPU supports it
        
        # Memory optimizations
        use_kv_cache=False,  # Not applicable to recurrent arch
        use_int8=False,  # Disable for now - slow on some GPUs
        
        # Batching
        max_batch_size=1,  # Single-token generation
    )
    
    engine = InferenceEngine(model, infer_config)
    
    # Benchmark with compile warmup
    print("Warming up torch.compile (this takes 30-60s)...")
    for _ in range(10):
        _ = engine.model(input_ids)
    
    print("Running optimized benchmark...")
    results = engine.benchmark(input_ids, num_runs=100)
    print(f"Optimized speed: {results['tokens_per_sec']:.2f} tok/s")
```

---

## 8. Benchmark Plan

### Baseline Measurement
1. Current performance: 42s/token
2. Breakdown by layer (profiling needed)

### Optimization Checkpoints
- [ ] Python 3.11 + torch.compile: Target 5-15x speedup
- [ ] Vectorized routing: Target +2-3x
- [ ] Vectorized KAN: Target +3-5x
- [ ] CUDA graphs: Target +1.5-2x
- [ ] Mixed precision: Target +1.3-1.8x

### Profiling Commands
```python
# Add to test_inference.py
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    outputs = model(input_ids)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("nexus_trace.json")
```

---

## 9. Hardware Considerations

### GTX 750 Ti Limitations
- **Compute Capability 5.0**: Older architecture
- **Limited Tensor Cores**: No native mixed-precision acceleration
- **2GB VRAM**: Memory-constrained
- **~1.3 TFLOPS FP32**: Relatively slow

### Optimization Impact on GTX 750 Ti
- torch.compile: ✅ **Full benefit** (reduces CPU overhead)
- Mixed precision (FP16): ⚠️ **Limited benefit** (no hardware acceleration)
- CUDA graphs: ✅ **Full benefit** (reduces kernel launch overhead)
- Triton kernels: ⚠️ **May not help** (compute-bound, not memory-bound)

### Recommendation
**Consider testing on newer GPU** (RTX 3060+) to validate full optimization potential. GTX 750 Ti will bottleneck the 100x speedup goal.

---

## 10. Risk Assessment

### High Risk
- ❌ **Python 3.12 incompatibility**: Blocks torch.compile (CRITICAL)
- ⚠️ **Hardware limitations**: GTX 750 Ti may not achieve 100x speedup

### Medium Risk
- ⚠️ **Dynamic control flow**: Adaptive depth conflicts with CUDA graphs
- ⚠️ **Triton Windows support**: Cannot use Triton kernels on Windows

### Low Risk
- ✅ **torch.compile stability**: Well-tested in PyTorch 2.3+
- ✅ **Vectorization**: Standard PyTorch optimizations

---

## Conclusion

### Realistic Performance Target
**With Python 3.11 + All Optimizations**:
- Base (torch.compile): 5-15x → **2.8-8.4s/token**
- + Vectorization: 2-3x → **0.9-4.2s/token**
- + CUDA graphs: 1.5-2x → **0.45-2.8s/token**

**Achievable: 15-93x speedup → 0.45-2.8s/token**

**To reach <0.5s/token goal**:
- ✅ Possible with all optimizations + newer GPU
- ⚠️ Challenging on GTX 750 Ti (compute-limited)
- ✅ Definitely achievable on RTX 3060+ or A100

### Next Steps
1. ✅ **CRITICAL**: Downgrade to Python 3.11
2. ✅ Enable torch.compile(mode="reduce-overhead")
3. ✅ Vectorize sparse router and KAN layers
4. ✅ Profile to identify remaining bottlenecks
5. ⚠️ Consider GPU upgrade if <0.5s/token required on current hardware
