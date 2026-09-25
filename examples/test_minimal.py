"""Ultra-minimal diagnostic test"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Step 1: Importing NEXUSOmega...")
from nexus_omega.core.omega_system import NEXUSOmega
from nexus_omega.config.fast_inference_config import FastInferenceArchConfig

print("Step 2: Creating tiny config...")
config = FastInferenceArchConfig(
    hidden_dim=128,  # TINY
    num_layers=2,    # Just 2 layers
    num_experts=8,   # Just 8 experts
)

print("Step 3: Creating model...")
model = NEXUSOmega(vocab_size=100, arch_config=config)

print("Step 4: Moving to CUDA...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print("Step 5: Creating input...")
input_ids = torch.randint(0, 100, (1, 5), device=device)

print("Step 6: Forward pass...")
with torch.no_grad():
    output = model(input_ids, enable_online_learning=False)

print(f"SUCCESS! Output shape: {output.logits.shape}")
print(f"Model works! Now testing speed...")

import time
start = time.time()
with torch.no_grad():
    for i in range(10):
        _ = model(input_ids, enable_online_learning=False)
        if i == 0:
            print(f"  First pass done in {time.time()-start:.2f}s")
torch.cuda.synchronize() if torch.cuda.is_available() else None
elapsed = time.time() - start

print(f"\n10 passes: {elapsed:.2f}s")
print(f"Per pass: {elapsed/10*1000:.0f}ms")
print(f"Tokens/sec: {50/elapsed:.1f}")
