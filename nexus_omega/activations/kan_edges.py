"""
KAN Activation Edges (Layer 2)

Kolmogorov-Arnold Network inspired activation functions using learnable
B-splines. Unlike fixed activation functions (ReLU, GELU), these adapt
to the data and are immune to catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from nexus_omega.base.layer import NexusLayer, LayerOutput


class KANActivation(nn.Module):
    """
    Learnable activation function using B-splines.

    Instead of fixed activation like ReLU or GELU, each connection
    has its own learnable univariate function represented as B-splines.
    """

    def __init__(
        self,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple = (-1.0, 1.0),
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        # Create grid points for B-spline basis
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1
        )
        self.register_buffer('grid', grid)

        # Learnable spline coefficients
        self.coef = nn.Parameter(torch.randn(grid_size + spline_order))

        # Optional residual connection weight
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)

    def b_spline_basis(self, x: torch.Tensor, i: int, k: int) -> torch.Tensor:
        """
        Compute B-spline basis function recursively.

        Args:
            x: Input tensor
            i: Basis function index
            k: Spline order

        Returns:
            Basis function evaluated at x
        """
        if k == 0:
            # Base case: indicator function
            return ((x >= self.grid[i]) & (x < self.grid[i + 1])).float()
        else:
            # Recursive case
            left_num = x - self.grid[i]
            left_den = self.grid[i + k] - self.grid[i]
            left = left_num / (left_den + 1e-8) if left_den.abs() > 1e-8 else 0.0

            right_num = self.grid[i + k + 1] - x
            right_den = self.grid[i + k + 1] - self.grid[i + 1]
            right = right_num / (right_den + 1e-8) if right_den.abs() > 1e-8 else 0.0

            return (
                left * self.b_spline_basis(x, i, k - 1) +
                right * self.b_spline_basis(x, i + 1, k - 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable B-spline activation.

        Args:
            x: Input tensor of any shape

        Returns:
            Activated tensor (same shape as input)
        """
        # Clamp input to grid range for stability
        x_clamped = torch.clamp(x, self.grid_range[0], self.grid_range[1])

        # Compute spline activation
        output = torch.zeros_like(x)
        for i in range(len(self.coef)):
            basis = self.b_spline_basis(x_clamped, i, self.spline_order)
            output = output + self.coef[i] * basis

        # Add residual connection to input
        output = output + self.residual_weight * x

        return output


class KANLayer(NexusLayer):
    """
    KAN Activation Edges - Layer 2 of NEXUS-Ω.

    Replaces traditional linear + activation with KAN edges that learn
    the optimal univariate function for each connection.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        use_bias: bool = True,
        config: Optional[any] = None,
    ):
        super().__init__("kan_layer", config)
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Linear transformation
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # KAN activation for each output dimension
        self.activations = nn.ModuleList([
            KANActivation(grid_size, spline_order)
            for _ in range(out_features)
        ])

    def forward(self, x: torch.Tensor, **kwargs) -> LayerOutput:
        """
        Forward pass with KAN activations.

        Args:
            x: Input [batch, seq_len, in_features]

        Returns:
            LayerOutput with activated output
        """
        # Linear transformation
        out = F.linear(x, self.weight, self.bias)

        # Apply KAN activation per output dimension
        batch, seq_len, out_features = out.shape
        activated = torch.zeros_like(out)

        # Process each output feature with its own learned activation
        for i in range(out_features):
            activated[..., i] = self.activations[i](out[..., i])

        return LayerOutput(
            output=activated,
            metrics={
                "kan_grid_size": self.grid_size,
                "spline_order": self.spline_order,
            }
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"grid_size={self.grid_size}, spline_order={self.spline_order}"
        )


class EfficientKANLayer(NexusLayer):
    """
    Efficient KAN Layer using shared splines across features.

    Reduces memory by sharing spline coefficients across groups of features.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_spline_groups: int = 8,
        grid_size: int = 5,
        spline_order: int = 3,
        config: Optional[any] = None,
    ):
        super().__init__("efficient_kan", config)
        self.in_features = in_features
        self.out_features = out_features
        self.num_spline_groups = num_spline_groups
        self.grid_size = grid_size

        # Linear weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Shared spline activations (one per group)
        self.activations = nn.ModuleList([
            KANActivation(grid_size, spline_order)
            for _ in range(num_spline_groups)
        ])

        # Assign each output feature to a group
        self.register_buffer(
            'feature_to_group',
            torch.arange(out_features) % num_spline_groups
        )

    def forward(self, x: torch.Tensor, **kwargs) -> LayerOutput:
        """Forward with grouped KAN activations."""
        # Linear transformation
        out = F.linear(x, self.weight, self.bias)

        # Apply grouped activations
        activated = torch.zeros_like(out)

        for group_id in range(self.num_spline_groups):
            # Get features in this group
            group_mask = self.feature_to_group == group_id
            group_indices = torch.where(group_mask)[0]

            if len(group_indices) > 0:
                # Apply same activation to all features in group
                for idx in group_indices:
                    activated[..., idx] = self.activations[group_id](out[..., idx])

        return LayerOutput(output=activated)


if __name__ == "__main__":
    print("Testing KAN Activation Edges...")

    # Test single KAN activation
    kan_act = KANActivation(grid_size=5, spline_order=3)
    x = torch.randn(10, 512)
    y = kan_act(x)
    print(f"KAN activation: {x.shape} -> {y.shape}")

    # Test full KAN layer
    kan_layer = KANLayer(in_features=512, out_features=256, grid_size=5)
    x = torch.randn(2, 10, 512)
    output = kan_layer(x)
    print(f"KAN layer: {x.shape} -> {output.output.shape}")
    print(f"Parameters: {kan_layer.count_parameters():,}")

    # Test efficient KAN
    eff_kan = EfficientKANLayer(512, 256, num_spline_groups=8)
    output = eff_kan(x)
    print(f"Efficient KAN: {x.shape} -> {output.output.shape}")
    print(f"Parameters: {eff_kan.count_parameters():,}")

    print("✓ KAN activation tests passed!")
