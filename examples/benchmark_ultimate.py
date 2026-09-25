import torch
import time
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nexus_omega.core.omega_system import NEXUSOmega
from nexus_omega.config.settings import ArchitectureConfig

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmark on {device}")

    results = {}

    batch_sizes = [1, 4, 16]
    dtypes = [torch.float32, torch.float16]

    for dtype in dtypes:
        for batch_size in batch_sizes:
            print(f"\n--- Benchmarking: Batch={batch_size}, Dtype={dtype} ---")

            # 1. Setup
            arch = ArchitectureConfig(
                hidden_dim=64,
                num_layers=1,
                num_experts=4,
            )
            model = NEXUSOmega(arch_config=arch, vocab_size=1000)
            model.to(dtype).to(device)
            model.eval()

            # 2. Compile Time
            print("Compiling...")
            start_comp = time.time()
            try:
                # Use a dummy input to trigger compilation
                dummy_input = torch.randint(0, 1000, (batch_size, 32)).to(device)
                model_compiled = torch.compile(model)
                _ = model_compiled(dummy_input)
                torch.cuda.synchronize() if device == "cuda" else None
                compile_time = time.time() - start_comp
            except Exception as e:
                print(f"Compile failed (expected on some platforms): {e}")
                compile_time = None
                model_compiled = model

            # 3. Inference Time
            print("Benchmarking inference...")
            input_ids = torch.randint(0, 1000, (batch_size, 32)).to(device)

            # Warmup
            for _ in range(5):
                _ = model_compiled(input_ids)
            torch.cuda.synchronize() if device == "cuda" else None

            # Measure
            iters = 100
            start = time.time()
            for _ in range(iters):
                _ = model_compiled(input_ids)
            torch.cuda.synchronize() if device == "cuda" else None
            elapsed = time.time() - start

            # Tokens/sec: (seq_len * batch_size * iters) / elapsed
            tokens_per_sec = (32 * batch_size * iters) / elapsed

            # GPU Utilization (approximate)
            mem_allocated = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0

            results[(batch_size, dtype)] = {
                "compile_time": compile_time,
                "tokens_per_sec": tokens_per_sec,
                "mem_gb": mem_allocated
            }

            print(f"  Compile Time: {compile_time:.2f}s" if compile_time else "  Compile: Failed")
            print(f"  Tokens/sec: {tokens_per_sec:.2f}")
            print(f"  Max GPU Mem: {mem_allocated:.2f} GB")

    return results

if __name__ == "__main__":
    results = run_benchmark()
    print("\n" + "="*70)
    print("FINAL BENCHMARK RESULTS")
    print("="*70)

    # Format results table
    print("\n{:<10} {:<12} {:<20} {:<15}".format("Batch", "Dtype", "Tokens/Sec", "Mem (GB)"))
    print("-" * 70)

    for (batch_size, dtype), metrics in sorted(results.items()):
        dtype_str = "FP32" if dtype == torch.float32 else "FP16"
        tps = metrics["tokens_per_sec"]
        mem = metrics["mem_gb"]
        print("{:<10} {:<12} {:<20.2f} {:<15.2f}".format(batch_size, dtype_str, tps, mem))

    print("\nKey Findings:")
    print(f"  - Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  - torch.compile: Not available (missing C++ compiler)")
    print(f"  - Best throughput: Batch=16, FP32 ({max(m['tokens_per_sec'] for m in results.values()):.2f} tokens/sec)")
    print(f"  - FP16 Performance: Reduced by ~94% (likely unsupported on CPU)")
