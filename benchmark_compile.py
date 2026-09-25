import torch
import time
import os
from nexus_omega.core.omega_system import create_nexus_omega

# Set environment variable to allow running on CPU if GPU is not available
if not torch.cuda.is_available():
    print("Warning: CUDA not available. Running on CPU (compilation will be slower/less effective).")
    device = torch.device('cpu')
else:
    device = torch.device('cuda')

# Create model
model = create_nexus_omega("small", vocab_size=10000).to(device)
model.eval()

# Dummy input
input_ids = torch.randint(0, 10000, (1, 32)).to(device)

def run_benchmark(model, num_runs=100):
    # Warmup
    for _ in range(10):
        _ = model(input_ids)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_runs):
        _ = model(input_ids)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start_time
    # tokens per run = numel of input
    return (num_runs * input_ids.numel()) / elapsed

# 1. Uncompiled
print("Measuring uncompiled...")
uncompiled_tps = run_benchmark(model)
print(f"Uncompiled: {uncompiled_tps:.2f} tokens/sec")

# 2. Compiled
print("\nMeasuring compiled (mode='max-autotune')...")
try:
    start_compile = time.time()
    # Using dynamic=False for max-autotune often helps if shapes are fixed
    compiled_model = torch.compile(model, mode="max-autotune")
    # Need to run once to trigger compilation
    _ = compiled_model(input_ids)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    compile_time = time.time() - start_compile

    compiled_tps = run_benchmark(compiled_model)
    print(f"Compile time: {compile_time:.2f} seconds")
    print(f"Compiled: {compiled_tps:.2f} tokens/sec")
    print(f"Speedup: {compiled_tps/uncompiled_tps:.2f}x")
except Exception as e:
    print(f"Compilation failed: {e}")

    # Try reduce-overhead if max-autotune fails
    print("\nAttempting compile with mode='reduce-overhead'...")
    try:
        start_compile = time.time()
        compiled_model = torch.compile(model, mode="reduce-overhead")
        _ = compiled_model(input_ids)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        compile_time = time.time() - start_compile

        compiled_tps = run_benchmark(compiled_model)
        print(f"Compile time: {compile_time:.2f} seconds")
        print(f"Compiled: {compiled_tps:.2f} tokens/sec")
        print(f"Speedup: {compiled_tps/uncompiled_tps:.2f}x")
    except Exception as e:
        print(f"Compilation failed: {e}")
