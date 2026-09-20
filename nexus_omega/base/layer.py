"""
Base Layer class for all NEXUS-Ω layers.

All 8 invented layers inherit from this base class, providing
a consistent interface for the OmegaSystem composition.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class LayerOutput:
    """Container for layer outputs with metadata."""

    output: torch.Tensor
    aux_loss: Optional[torch.Tensor] = None  # Auxiliary losses (e.g., load balancing)
    metrics: Optional[Dict[str, float]] = None  # Layer-specific metrics


class NexusLayer(nn.Module, ABC):
    """
    Abstract base class for all NEXUS-Ω layers.

    Provides:
    - Consistent forward interface
    - Parameter counting
    - Efficiency tracking hooks
    - Consolidation interface
    """

    def __init__(self, name: str, config: Optional[Any] = None):
        super().__init__()
        self.name = name
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> LayerOutput:
        """
        Forward pass for this layer.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            LayerOutput with output tensor and optional metadata
        """
        pass

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters in this layer."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_sparsity(self) -> float:
        """Compute average weight sparsity in this layer."""
        total_params = 0
        zero_params = 0

        for param in self.parameters():
            total_params += param.numel()
            zero_params += (param.abs() < 1e-8).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0

    def consolidate(self, fisher_info: Optional[Dict[str, torch.Tensor]] = None):
        """
        Run consolidation step (for dual-plasticity layers).

        Override this in layers that support consolidation.
        """
        pass

    def get_fisher_information(self) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information for important weights.

        Used for Elastic Weight Consolidation (EWC).
        Override this for layers that support EWC.
        """
        return {}

    def reset_plastic_weights(self):
        """Reset plastic weights (for dual-plasticity layers)."""
        pass

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return f"name={self.name}, params={self.count_parameters():,}"


class CompositeLayer(NexusLayer):
    """
    Layer that combines multiple sub-layers.

    Used for building complex layer stacks.
    """

    def __init__(self, name: str, layers: list[NexusLayer], config: Optional[Any] = None):
        super().__init__(name, config)
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, **kwargs) -> LayerOutput:
        """Sequential forward through all sub-layers."""
        total_aux_loss = 0.0
        all_metrics = {}

        for layer in self.layers:
            output = layer(x, **kwargs)
            x = output.output

            if output.aux_loss is not None:
                total_aux_loss = total_aux_loss + output.aux_loss

            if output.metrics:
                all_metrics.update(output.metrics)

        return LayerOutput(
            output=x,
            aux_loss=total_aux_loss if total_aux_loss != 0 else None,
            metrics=all_metrics if all_metrics else None
        )


if __name__ == "__main__":
    # Test the base class with a simple implementation
    class TestLayer(NexusLayer):
        def __init__(self, hidden_dim):
            super().__init__("test_layer")
            self.linear = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x: torch.Tensor, **kwargs) -> LayerOutput:
            return LayerOutput(output=self.linear(x))

    layer = TestLayer(512)
    x = torch.randn(2, 10, 512)
    output = layer(x)

    print(f"Layer: {layer}")
    print(f"Parameters: {layer.count_parameters():,}")
    print(f"Output shape: {output.output.shape}")
    print("✓ Base layer tests passed!")
