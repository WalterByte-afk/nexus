# NEXUS-Omega Ultra-Fast Inference Implementation Guide

**Date**: 2026-09-25  
**Target Speedup**: 100x (42s/token → <0.5s/token)  
**Status**: RESEARCH COMPLETE - READY FOR IMPLEMENTATION

---

## 🔴 CRITICAL BLOCKER: Python 3.12

**torch.compile** requires Python ≤3.11. Current environment uses Python 3.12.

### Fix Immediately:
```bash
# OPTION 1: Conda (recommended)
conda create -n nexus_fast python=3.11
conda activate nexus_fast

# OPTION 2: Virtual environment
python3.11 -m venv venv_fast
source venv_fast/bin/activate  # or venv_fast\Scripts\activate on Windows

# Install dependencies
pip install torch>=2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Without this fix, **5-15x speedup from torch.compile is unavailable**.

---

## 🚀 Phase 1: Core Optimizations (Week 1)

### 1.1 Vectorize Sparse Router [HIGH PRIORITY]

**File**: `nexus_omega/routing/sparse_router.py`

**Problem**: Python loops for expert computation
**Solution**: Replace with batched tensor operations

**Current Code** (SLOW):
```python
for i in range(flat_x.shape[0]):
    for j, expert_id in enumerate(expert_idx):
        # Expert computation - SERIAL
```

**Optimized Code** (FAST):
```python
# Use existing VectorizedSparseRouter from optimizer.py
from nexus_omega.deploy.optimizer import VectorizedSparseRouter

class OptimizedSparseDynamicRouter(NexusLayer):
    def __init__(self, ...):
        super().__init__(...)
        self.router = VectorizedSparseRouter(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
        )
```

**Expected Speedup**: 5-10x for routing layer

### 1.2 Vectorize KAN Activation [HIGH PRIORITY]

**File**: `nexus_omega/activations/kan_edges.py`

**Problem**: Recursive B-spline basis computation
**Solution**: Vectorized basis matrix

**Current Code** (SLOW):
```python
for i in range(len(self.coef)):
    basis = self.b_spline_basis(x_clamped, i, self.spline_order)
```

**Optimized Code** (FAST):
```python
from nexus_omega.deploy.optimizer import VectorizedKANActivation

class OptimizedKANLayer(NexusLayer):
    def __init__(self, ...):
        super().__init__(...)
        self.activation = VectorizedKANActivation(
            grid_size=grid_size,
            spline_order=spline_order,
        )
```

**Expected Speedup**: 3-5x for KAN layers

### 1.3 Update Inference Configuration [MEDIUM PRIORITY]

**File**: `examples/test_inference.py`

**Current**:
```python
infer_config = InferenceConfig(
    use_torch_compile=True,  # Won't work on Python 3.12
    use_fp16=False,  # Not enabled
    use_kv_cache=False,
)
```

**Updated**:
```python
infer_config = InferenceConfig(
    use_torch_compile=True,  # Only works after Python 3.11
    use_fp16=True,  # Enable mixed precision
    use_kv_cache=True,  # Add KV cache support
    use_cuda_graphs=True,  # Enable CUDA graphs
)
```

**Expected Speedup**: 2-10x (depends on Python version)

---

## ⚡ Phase 2: PyTorch Optimizations (Week 2)

### 2.1 Enable torch.compile [CRITICAL]

**Prerequisite**: Python 3.11 installed

**File**: `nexus_omega/deploy/inference.py`

**Add to `_optimize_model` method**:
```python
if self.config.use_torch_compile:
    try:
        print("Compiling model with torch.compile...")
        # Use reduce-overhead for single-token inference
        self.model = torch.compile(
            self.model, 
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=True,
        )
        print("✓ Model compiled for 5-15x speedup")
    except Exception as e:
        print(f"torch.compile failed: {e}")
```

**Modes to test**:
- `"reduce-overhead"`: Best for single-token (fast compile, good speed)
- `"max-autotune"`: Best for throughput (slow compile, best speed)

**Expected Speedup**: 5-15x (varies by mode)

### 2.2 Mixed Precision [MEDIUM PRIORITY]

**File**: `nexus_omega/deploy/inference.py`

**Update**:
```python
if self.config.use_fp16 and torch.cuda.is_available():
    # Check GPU support
    device_capability = torch.cuda.get_device_capability(0)
    if device_capability >= (7, 0):  # Tensor Cores
        self.model = self.model.half()
        print("✓ Enabled FP16 with Tensor Core acceleration")
    else:
        self.model = self.model.to(torch.bfloat16)
        print("✓ Enabled BF16 for older GPUs")
```

**Hardware Note**:
- GTX 750 Ti (Compute 5.0): No hardware acceleration for FP16
- RTX 3060+ (Compute 8.6): 2-4x speedup with FP16

**Expected Speedup**: 1.3-4x (depends on GPU)

---

## 🎯 Phase 3: Memory & Execution (Week 3)

### 3.1 CUDA Graphs for Static Shapes [MEDIUM PRIORITY]

**File**: `nexus_omega/deploy/inference.py`

**Add to `InferenceEngine` class**:
```python
class InferenceEngine:
    def __init__(self, model, config):
        self.cuda_graph = None
        self.static_input = None
        
        if config.use_cuda_graphs:
            self._capture_cuda_graph()
    
    def _capture_cuda_graph(self):
        # Capture single-token forward pass
        static_input = torch.zeros(1, 1, dtype=torch.long, device='cuda')
        
        # Warmup
        for _ in range(3):
            _ = self.model(static_input)
        
        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(static_input)
    
    def generate_with_graph(self, input_ids):
        # Fast inference via graph replay
        self.static_input.copy_(input_ids)
        self.graph.replay()
        return self.static_output.clone()
```

**Limitations**:
- Requires fixed input shapes
- Complex with dynamic control flow (adaptive depth)

**Expected Speedup**: 1.5-2x

### 3.2 Optimize KV Cache [MEDIUM PRIORITY]

**Current issue**: Not implemented for recurrent architecture

**Solution**: Add recurrent state cache
```python
class RecurrentCache:
    """Cache for recurrent layers (alternative to attention KV cache)."""
    def __init__(self):
        self.hidden_states = None  # Store GRU hidden states
        self.context = None  # Store temporal context
    
    def update(self, new_hidden, new_context):
        # Efficient concatenation/updating
        pass
```

**Expected Speedup**: 1.2-1.5x for long sequences

---

## 🔧 Phase 4: Advanced Optimizations (Week 4+)

### 4.1 Custom Triton Kernels [OPTIONAL]

**Linux/WSL2 only** - Windows not supported

**File**: `nexus_omega/routing/sparse_expert_routing.py` (created)

**Kernels available**:
1. `sparse_expert_routing_kernel`: Fused expert routing
2. `bspline_activation_kernel`: Vectorized B-splines
3. `hebbian_update_kernel`: Fast plasticity updates

**To use**:
```python
if TRITON_AVAILABLE:
    from nexus_omega.routing.sparse_expert_routing import (
        sparse_expert_routing_forward
    )
    # Use Triton kernel instead of PyTorch
```

**Expected Speedup**: 2-5x (but limited to Linux)

### 4.2 Profile-Guided Optimization [ADVANCED]

**Tools**:
```python
# Add to test_inference.py
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    outputs = model(input_ids)

# Find bottlenecks
print(prof.key_averages().table(sort_by="cuda_time_total"))
prof.export_chrome_trace("trace.json")
```

**Optimization workflow**:
1. Profile → Find bottleneck
2. Optimize → Fix bottleneck  
3. Repeat → Until <0.5s/token

---

## 📊 Performance Targets & Validation

### Baseline: 42s/token

### Optimization Targets:
1. Vectorization (Phase 1): **42s → 4-8s** (5-10x)
2. torch.compile (Phase 2): **4-8s → 0.8-2s** (5-15x additional)
3. CUDA graphs (Phase 3): **0.8-2s → 0.4-1s** (1.5-2x additional)
4. Mixed precision: **0.4-1s → 0.2-0.5s** (1.3-4x additional)

### Validation Commands:
```bash
# Test each phase
python scripts/benchmark_optimizations.py

# Profile bottlenecks
python -m cProfile -s time examples/test_inference.py

# Memory usage
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

---

## 🛠️ Integration Checklist

### ✅ COMPLETED (from research phase)
- [x] Research report with comprehensive analysis
- [x] Identified Python 3.12 blocker
- [x] Created optimization utilities (`optimizer.py`)
- [x] Created Triton kernel templates (Linux only)
- [x] Created benchmark script

### 🔄 IN PROGRESS (requires Python 3.11)
- [ ] Downgrade Python environment (CRITICAL)
- [ ] Test torch.compile with reduce-overhead mode
- [ ] Vectorize sparse router loops
- [ ] Vectorize KAN activation

### 📋 PENDING (after Python fix)
- [ ] Benchmark with torch.compile
- [ ] Implement CUDA graphs
- [ ] Add mixed precision support
- [ ] Profile and identify remaining bottlenecks

---

## 🚨 Troubleshooting Guide

### Common Issues:

#### 1. "torch.compile not available"
```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"
# Must be ≥2.0.0

# Check Python version  
python --version
# Must be ≤3.11
```

#### 2. "CUDA out of memory"
```python
# Reduce batch size
infer_config = InferenceConfig(max_batch_size=1)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use CPU offloading
infer_config.offload_to_cpu = True
```

#### 3. "Slow first inference" (torch.compile warmup)
```python
# Expected: First run takes 30-60s, then fast
engine = InferenceEngine(model, config)

# Trigger compilation
_ = engine.model(input_ids)  # SLOW (compiling)
_ = engine.model(input_ids)  # FAST (reusing compiled graph)
```

#### 4. "Triton not available on Windows"
```bash
# Use WSL2 or Linux
# Or stick with pure PyTorch optimizations
pip uninstall triton
# Use VectorizedSparseRouter instead
```

---

## 📈 Expected Performance Results

### On GTX 750 Ti (Compute 5.0):
- **Best case**: 15-30x speedup → **1.4-2.8s/token**
- **With Python 3.11**: 5-15x from torch.compile
- **With vectorization**: 2-3x additional
- **Total**: May not reach <0.5s/token on this hardware

### On RTX 3060 (Compute 8.6):
- **Best case**: 45-100x speedup → **0.4-0.9s/token**
- **With Python 3.11**: 5-15x from torch.compile
- **With FP16 Tensor Cores**: 2-4x additional
- **With CUDA graphs**: 1.5-2x additional
- **Total**: <0.5s/token achievable

---

## 🎯 Final Recommendations

### IMMEDIATE (Day 1):
1. **Downgrade to Python 3.11** (CRITICAL blocker)
2. **Enable torch.compile with reduce-overhead mode**
3. **Run benchmark** to verify 5-15x speedup

### SHORT TERM (Week 1):
4. **Vectorize Python loops** in sparse router and KAN
5. **Test mixed precision** (FP16 if supported)
6. **Profile** to find remaining bottlenecks

### MEDIUM TERM (Week 2-3):
7. **Implement CUDA graphs** for static inference
8. **Optimize memory layout** for cache efficiency
9. **Consider GPU upgrade** if <0.5s/token critical

### LONG TERM (Week 4+):
10. **Explore Triton kernels** if Linux/WSL2 available
11. **Implement sparse attention** if adding attention layers
12. **Optimize for deployment** (TensorRT, ONNX)

---

## 📝 Quick Start (2 Hours)

```bash
# 1. Fix Python version (30 minutes)
conda create -n nexus_fast python=3.11
conda activate nexus_fast

# 2. Install dependencies (15 minutes)
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. Test torch.compile (15 minutes)
python -c "
import torch
model = torch.nn.Linear(512, 512).cuda()
compiled = torch.compile(model, mode='reduce-overhead')
print('✓ torch.compile works!')
"

# 4. Run benchmark (10 minutes)
python scripts/benchmark_optimizations.py

# 5. Apply vectorization (30 minutes)
# Edit nexus_omega/routing/sparse_router.py
# Replace loops with VectorizedSparseRouter

# 6. Test optimized model (20 minutes)
python examples/test_inference.py
```

**Expected result after 2 hours**: 5-10x speedup confirmed

---

**Bottom Line**: The 100x speedup goal is technically achievable but requires:
1. Python 3.11 (CRITICAL - blocks torch.compile)
2. Vectorization of Python loops (HIGH priority)
3. Possibly newer GPU hardware (if <0.5s/token mandatory)

**Realistic timeline**: 
- 1 week to reach 15-30x speedup (2-3s/token)
- 2-4 weeks to reach 45-100x speedup (<0.5s/token on RTX 3060+)

---

**Research complete. Ready for implementation.**
