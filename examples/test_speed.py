"""
Simple Speed Test for NEXUS-Omega with Optimized Config

Quick benchmark to see tokens/second after all layer optimizations.
"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus_omega.core.omega_system import NEXUSOmega
from nexus_omega.config.fast_inference_config import get_fast_inference_config
from nexus_omega.deploy.inference import InferenceEngine, InferenceConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

config = get_fast_inference_config()
arch_config = config["architecture"]
train_config = config["training"]

model = NEXUSOmega(vocab_size=1000, arch_config=arch_config)
model = model.to(device)
model.eval()

input_ids = torch.randint(0, 1000, (1, 10), device=device)

# Warmup
with torch.no_grad():
    for _ in range(3):
        _ = model(input_ids, enable_online_learning=False)

torch.cuda.synchronize() if torch.cuda.is_available() else None

# Benchmark
start = time.time()
with torch.no_grad():
    for _ in range(50):
        _ = model(input_ids, enable_online_learning=False)
torch.cuda.synchronize() if torch.cuda.is_available() else None

elapsed = time.time() - start
tokens_per_sec = (50 * 10) / elapsed

print(f"\n{'='*50}")
print(f"50 forward passes: {elapsed:.2f}s")
print(f"Per pass: {elapsed/50*1000:.1f}ms")
print(f"Tokens/sec: {tokens_per_sec:.1f}")
print(f"{'='*50}")