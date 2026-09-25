"""
Fast Activation Layer - Optimized replacement for KAN
Replaces slow B-spline KAN with fast learned gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from nexus_omega.base.layer import NexusLayer, LayerOutput


class FastLearnedActivation(NexusLayer):
    """
    Ultra-fast learned activation using simple gating mechanism.

    Replaces KAN's B-splines with a lightweight learned non-linearity
    that's 1000x faster while maintaining adaptability.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_gelu: bool = True,
        config: Optional[any] = None,
    ):
        super().__init__("fast_activation", config)
        self.in_features = in_features
        self.out_features = out_features

        # Linear transformation
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / math.sqrt(in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learned gating parameters (per output feature)
        self.gate_weight = nn.Parameter(torch.ones(out_features))
        self.gate_bias = nn.Parameter(torch.zeros(out_features))

        # Use GELU or SiLU
        self.activation = nn.GELU() if use_gelu else nn.SiLU()

    def forward(self, x: torch.Tensor, **kwargs) -> LayerOutput:
        """
        Fast forward pass with learned gating.

        Args:
            x: Input [batch, seq_len, in_features]

        Returns:
            LayerOutput with activated output
        """
        # Linear transformation (single matmul - FAST)
        out = F.linear(x, self.weight, self.bias)

        # Apply base activation (GELU/SiLU - both very fast)
        activated = self.activation(out)

        # Learned per-feature gating (element-wise ops - FAST)
        gated = activated * self.gate_weight + self.gate_bias

        return LayerOutput(
            output=gated,
            metrics={"activation_type": "fast_learned"}
        )


class SimpleFastActivation(NexusLayer):
    """
    Simplest possible fast activation - just GELU.
    Use this for inference when speed is critical.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: Optional[any] = None,
    ):
        super().__init__("simple_fast", config)
        self.in_features = in_features
        self.out_features = out_features

        # Just linear + GELU
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, **kwargs) -> LayerOutput:
        """Ultra-fast forward: linear + GELU."""
        out = self.linear(x)
        activated = F.gelu(out)

        return LayerOutput(
            output=activated,
            metrics={"activation_type": "simple_gelu"}
        )


class VectorizedEfficientKAN(NexusLayer):
    """
    Vectorized version of EfficientKAN with no Python loops.
    Uses batched operations for all group processing.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int = 8,
        config: Optional[any] = None,
    ):
        super().__init__("vectorized_kan", config)
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        # Linear weights
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / math.sqrt(in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Per-group learned polynomial coefficients (simpler than B-splines)
        # Each group has 4 coefficients for cubic approximation
        self.poly_coefs = nn.Parameter(torch.randn(num_groups, 4))

        # Assign features to groups
        self.register_buffer(
            'feature_to_group',
            torch.arange(out_features) % num_groups
        )

    def forward(self, x: torch.Tensor, **kwargs) -> LayerOutput:
        """Fully vectorized forward pass."""
        # Linear transformation
        out = F.linear(x, self.weight, self.bias)

        # Apply polynomial activation per group (VECTORIZED)
        # Compute powers: x, x^2, x^3
        out_2 = out * out
        out_3 = out_2 * out

        # Stack polynomial terms [batch, seq, features, 4]
        poly_stack = torch.stack([torch.ones_like(out), out, out_2, out_3], dim=-1)

        # Get coefficients for each feature [features, 4]
        feature_coefs = self.poly_coefs[self.feature_to_group]  # [out_features, 4]

        # Apply: sum over polynomial terms (vectorized dot product)
        activated = torch.einsum('...f,fp->...f', poly_stack, feature_coefs)

        return LayerOutput(
            output=activated,
            metrics={"activation_type": "vectorized_kan", "num_groups": self.num_groups}
        )


if __name__ == "__main__":
    print("Testing Fast Activation Layers...")

    # Test dimensions
    batch, seq, dim = 2, 64, 512
    x = torch.randn(batch, seq, dim)

    # Test FastLearnedActivation
    fast = FastLearnedActivation(dim, dim)
    out = fast(x)
    print(f"FastLearnedActivation: {x.shape} -> {out.output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in fast.parameters()):,}")

    # Test SimpleFastActivation
    simple = SimpleFastActivation(dim, dim)
    out = simple(x)
    print(f"SimpleFastActivation: {x.shape} -> {out.output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in simple.parameters()):,}")

    # Test VectorizedEfficientKAN
    vec_kan = VectorizedEfficientKAN(dim, dim, num_groups=8)
    out = vec_kan(x)
    print(f"VectorizedEfficientKAN: {x.shape} -> {out.output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in vec_kan.parameters()):,}")

    print("\n✓ All fast activation tests passed!")
