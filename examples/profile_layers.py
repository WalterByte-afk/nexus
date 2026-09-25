"""Profile each layer to find the bottleneck"""
import torch
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus_omega.core.omega_system import NEXUSOmega
from nexus_omega.config.fast_inference_config import FastInferenceArchConfig

config = FastInferenceArchConfig(
    hidden_dim=128,
    num_layers=2,
    num_experts=8,
)

device = torch.device("cuda")
model = NEXUSOmega(vocab_size=100, arch_config=config).to(device).eval()
input_ids = torch.randint(0, 100, (1, 5), device=device)

# Warmup
with torch.no_grad():
    _ = model(input_ids, enable_online_learning=False)

print("Profiling each layer...")
print("="*60)

# Profile each block
for block_idx, block in enumerate(model.blocks):
    x = model.token_embedding(input_ids) + model.position_embedding(torch.arange(5, device=device).unsqueeze(0))

    # Time sparse router
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = block.sparse_router(x)
    torch.cuda.synchronize()
    router_time = (time.time() - start) / 10

    # Time KAN
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = block.kan_activation(x)
    torch.cuda.synchronize()
    kan_time = (time.time() - start) / 10

    # Time predictive coding
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = block.predictive_coding(x, enable_online_learning=False)
    torch.cuda.synchronize()
    pc_time = (time.time() - start) / 10

    # Time adaptive recurrence
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = block.adaptive_recurrence(x)
    torch.cuda.synchronize()
    recur_time = (time.time() - start) / 10

    print(f"Block {block_idx}:")
    print(f"  Sparse Router:     {router_time*1000:6.1f}ms")
    print(f"  KAN Activation:    {kan_time*1000:6.1f}ms")
    print(f"  Predictive Coding: {pc_time*1000:6.1f}ms")
    print(f"  Adaptive Recur:    {recur_time*1000:6.1f}ms")
    print(f"  TOTAL:             {(router_time+kan_time+pc_time+recur_time)*1000:6.1f}ms")
    print()
