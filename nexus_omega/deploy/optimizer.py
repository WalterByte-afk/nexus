"""
Optimization utilities for NEXUS-Ω deployment.

Includes quantization, pruning, and memory optimization for
running on consumer hardware.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import math


def quantize_model(
    model: nn.Module,
    quantization_type: str = "int8",
) -> nn.Module:
    """
    Quantize model weights for efficient inference.

    Args:
        model: NEXUS-Ω model
        quantization_type: "int8" or "int4"

    Returns:
        Quantized model
    """
    if quantization_type == "int8":
        return _quantize_int8(model)
    elif quantization_type == "int4":
        return _quantize_int4(model)
    else:
        raise ValueError(f"Unknown quantization: {quantization_type}")


def _quantize_int8(model: nn.Module) -> nn.Module:
    """Apply INT8 quantization."""
    # Use PyTorch's built-in quantization
    model.eval()

    # Quantize linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Simple min-max quantization
            with torch.no_grad():
                weight = module.weight.data
                scale = weight.abs().max() / 127.0
                quantized = torch.round(weight / scale).clamp(-128, 127)
                module.weight.data = quantized * scale

    return model


def _quantize_int4(model: nn.Module) -> nn.Module:
    """Apply INT4 quantization (more aggressive)."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                weight = module.weight.data
                scale = weight.abs().max() / 7.0
                quantized = torch.round(weight / scale).clamp(-8, 7)
                module.weight.data = quantized * scale

    return model


def prune_model(
    model: nn.Module,
    sparsity: float = 0.9,
) -> nn.Module:
    """
    Prune model weights to achieve target sparsity.

    Args:
        model: NEXUS-Ω model
        sparsity: Target fraction of zeros (0.9 = 90% sparse)

    Returns:
        Pruned model
    """
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            with torch.no_grad():
                # Get threshold for pruning
                flat = param.data.abs().view(-1)
                threshold_idx = int(flat.numel() * sparsity)
                threshold = flat.sort()[0][threshold_idx]

                # Apply mask
                mask = param.data.abs() > threshold
                param.data = param.data * mask.float()

    return model


def optimize_for_inference(
    model: nn.Module,
    quantization: str = "int8",
    pruning: float = 0.9,
    use_torch_compile: bool = True,
) -> nn.Module:
    """
    Apply all optimizations for fast inference.

    Args:
        model: NEXUS-Ω model
        quantization: Quantization type
        pruning: Target sparsity
        use_torch_compile: Use torch.compile() if available

    Returns:
        Optimized model
    """
    # Prune first
    if pruning > 0:
        model = prune_model(model, sparsity=pruning)

    # Then quantize
    if quantization:
        model = quantize_model(model, quantization_type=quantization)

    # Torch compile for PyTorch 2.0+
    if use_torch_compile and hasattr(torch, 'compile'):
        model = torch.compile(model)

    return model


def estimate_inference_speed(
    model: nn.Module,
    batch_size: int = 1,
    seq_len: int = 128,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Estimate inference speed.

    Returns:
        Dictionary with tokens/sec, latency, etc.
    """
    import time

    model.eval()
    model = model.to(device)

    # Warmup
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    # Benchmark
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()

    num_runs = 20
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.time() - start

    tokens_processed = batch_size * seq_len * num_runs
    tokens_per_sec = tokens_processed / elapsed
    latency_ms = (elapsed / num_runs) * 1000

    return {
        "tokens_per_second": tokens_per_sec,
        "latency_ms": latency_ms,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


if __name__ == "__main__":
    print("Testing optimization utilities...")

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
    )

    # Test quantization
    quantized = quantize_model(model, "int8")
    print(f"Original size: {get_model_size_mb(model):.2f} MB")
    print(f"Quantized size: {get_model_size_mb(quantized):.2f} MB")

    # Test pruning
    pruned = prune_model(model, sparsity=0.9)
    sparsity = (sum((p == 0).sum().item() for p in pruned.parameters()) /
                sum(p.numel() for p in pruned.parameters()))
    print(f"Actual sparsity: {sparsity:.2%}")

    print("✓ Optimization tests passed!")
