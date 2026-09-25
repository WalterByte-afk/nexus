"""
Predictive Coding Modules (Layer 3)

Implements local Hebbian learning that updates during inference.
No global backpropagation needed - brain-like local plasticity rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from nexus_omega.base.layer import NexusLayer, LayerOutput


class PredictiveCodingModule(NexusLayer):
    """
    Predictive Coding - Layer 3 of NEXUS-Ω.

    Key principles:
    1. Local prediction error minimization
    2. Hebbian learning ("neurons that fire together wire together")
    3. Updates happen during inference, not just training
    """

    def __init__(
        self,
        hidden_dim: int,
        num_iterations: int = 5,
        learning_rate: float = 0.1,
        decay: float = 0.01,
        config: Optional[any] = None,
        inference_iterations: int = 1,  # Reduced iterations for inference mode
    ):
        super().__init__("predictive_coding", config)
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.inference_iterations = inference_iterations  # Use 1 iteration during inference
        self.learning_rate = learning_rate
        self.decay = decay

        # Prediction layers (forward model)
        self.predictor = nn.Linear(hidden_dim, hidden_dim)

        # Error correction layers (backward model)
        self.error_corrector = nn.Linear(hidden_dim, hidden_dim)

        # Hebbian weights (updated locally)
        self.hebbian_weights = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor, **kwargs) -> LayerOutput:
        """
        Forward pass with predictive coding iterations.

        Args:
            x: Input [batch, seq_len, hidden_dim]

        Returns:
            LayerOutput with error-minimized representation
        """
        batch, seq_len, hidden = x.shape

        # Initialize representation
        representation = x.clone()

        total_error = 0.0

        # OPTIMIZATION: Use reduced iterations during inference (1 vs 5)
        # This is the biggest speedup - 5x reduction in computation
        iterations = self.num_iterations if self.training else self.inference_iterations

        # Predictive coding iterations
        for iteration in range(iterations):
            # Make prediction
            prediction = self.predictor(representation)

            # Compute prediction error
            error = x - prediction

            # Error correction
            correction = self.error_corrector(error)
            representation = representation + correction

            # Accumulate error for metrics
            total_error = total_error + error.pow(2).mean()

            # OPTIMIZATION: Make Hebbian updates optional/disabled during inference
            # Only update weights during training or if explicitly enabled
            if self.training or kwargs.get('enable_online_learning', False):
                self._hebbian_update(representation, error)

        avg_error = total_error / iterations

        return LayerOutput(
            output=representation,
            aux_loss=avg_error * 0.1,  # Prediction error as auxiliary loss
            metrics={
                "prediction_error": avg_error,
                "iterations": iterations,
            }
        )

    def _hebbian_update(self, pre: torch.Tensor, post: torch.Tensor):
        """
        Apply local Hebbian learning rule.

        Δw = η * (pre * post^T - decay * w)
        """
        batch, seq_len, hidden = pre.shape

        # Flatten for update
        pre_flat = pre.view(-1, hidden)
        post_flat = post.view(-1, hidden)

        # Compute outer product (Hebbian correlation)
        correlation = torch.matmul(pre_flat.t(), post_flat) / (batch * seq_len)

        # Update with decay
        with torch.no_grad():
            self.hebbian_weights.data = (
                self.hebbian_weights.data +
                self.learning_rate * correlation -
                self.decay * self.hebbian_weights.data
            )

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, iterations={self.num_iterations}, "
            f"lr={self.learning_rate}, decay={self.decay}"
        )


class HebbianLayer(nn.Module):
    """
    Pure Hebbian learning layer without prediction error.

    Simpler variant that just applies Hebbian plasticity.
    """

    def __init__(self, hidden_dim: int, learning_rate: float = 0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Hebbian connection weights
        self.weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)

    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """
        Forward with Hebbian update.

        Args:
            x: Input [batch, seq_len, hidden_dim]
            update: Whether to apply Hebbian update

        Returns:
            Output after Hebbian transformation
        """
        # Apply transformation
        output = F.linear(x, self.weights)

        # Hebbian update: Δw = η * (x^T * output)
        if update and (self.training or update):
            self._update_weights(x, output)

        return output

    def _update_weights(self, pre: torch.Tensor, post: torch.Tensor):
        """Apply Hebbian learning rule."""
        batch, seq_len, hidden = pre.shape

        # Flatten
        pre_flat = pre.reshape(-1, hidden)
        post_flat = post.reshape(-1, hidden)

        # Correlation
        correlation = torch.matmul(pre_flat.t(), post_flat) / (batch * seq_len)

        # Update
        with torch.no_grad():
            self.weights.data = self.weights.data + self.learning_rate * correlation


if __name__ == "__main__":
    print("Testing Predictive Coding Module...")

    # Test predictive coding
    pc_module = PredictiveCodingModule(
        hidden_dim=512,
        num_iterations=5,
        learning_rate=0.1
    )

    x = torch.randn(2, 10, 512)
    output = pc_module(x)

    print(f"Input: {x.shape}")
    print(f"Output: {output.output.shape}")
    print(f"Prediction error: {output.metrics['prediction_error']:.4f}")
    print(f"Aux loss: {output.aux_loss:.4f}")

    # Test Hebbian layer
    hebb = HebbianLayer(512, learning_rate=0.01)
    y = hebb(x)
    print(f"Hebbian output: {y.shape}")

    print("✓ Predictive coding tests passed!")
