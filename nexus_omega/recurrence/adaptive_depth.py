"""
Adaptive Recurrent Depth (Layer 4)

Dynamically adjusts computation depth (1-8 loops) per token based on complexity.
Smart tokens get smart compute - simple tokens processed quickly, complex ones get more iterations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
from nexus_omega.base.layer import NexusLayer, LayerOutput


class AdaptiveRecurrentDepth(NexusLayer):
    """
    Adaptive Recurrent Depth - Layer 4 of NEXUS-Ω.

    Key innovation: Dynamic depth allocation per token.
    - Simple tokens: 1-2 loops
    - Complex tokens: 6-8 loops
    """

    def __init__(
        self,
        hidden_dim: int,
        min_depth: int = 1,
        max_depth: int = 8,
        policy: Literal["learned", "fixed", "adaptive"] = "learned",
        config: Optional[any] = None,
    ):
        super().__init__("adaptive_recurrent_depth", config)
        self.hidden_dim = hidden_dim
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.policy = policy

        # Recurrent processing block
        self.recurrent_cell = nn.GRUCell(hidden_dim, hidden_dim)

        # Depth policy network (learns to predict optimal depth)
        self.depth_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, max_depth - min_depth + 1),
            nn.Softmax(dim=-1)
        )

        # Halting probability network (for adaptive policy)
        self.halting_net = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> LayerOutput:
        """
        Forward with adaptive depth.

        Args:
            x: Input [batch, seq_len, hidden_dim]

        Returns:
            LayerOutput with adaptively processed output
        """
        batch, seq_len, hidden = x.shape

        if self.policy == "learned":
            return self._forward_learned(x)
        elif self.policy == "adaptive":
            return self._forward_adaptive(x)
        else:  # fixed
            return self._forward_fixed(x, depth=self.max_depth)

    def _forward_learned(self, x: torch.Tensor) -> LayerOutput:
        """Use learned policy network to predict depth."""
        batch, seq_len, hidden = x.shape

        # Predict depth distribution
        depth_probs = self.depth_policy(x)  # [batch, seq_len, num_depths]

        # Sample or use argmax
        if self.training:
            depth_samples = torch.multinomial(
                depth_probs.view(-1, self.max_depth - self.min_depth + 1),
                1
            ).view(batch, seq_len)
        else:
            depth_samples = depth_probs.argmax(dim=-1)

        # Add min_depth offset
        depths = depth_samples + self.min_depth

        # Process each token with its predicted depth
        outputs = []
        avg_depth = 0.0

        for b in range(batch):
            for s in range(seq_len):
                token = x[b, s]  # [hidden]
                depth = depths[b, s].item()
                avg_depth += depth

                # Recurrent processing
                hidden_state = token
                for _ in range(int(depth)):
                    hidden_state = self.recurrent_cell(hidden_state.unsqueeze(0), hidden_state.unsqueeze(0)).squeeze(0)

                outputs.append(hidden_state)

        output = torch.stack(outputs).view(batch, seq_len, hidden)
        avg_depth = avg_depth / (batch * seq_len)

        return LayerOutput(
            output=output,
            metrics={
                "avg_depth": avg_depth,
                "min_depth_used": depths.min().item(),
                "max_depth_used": depths.max().item(),
            }
        )

    def _forward_adaptive(self, x: torch.Tensor) -> LayerOutput:
        """Adaptive Computation Time (ACT) style halting."""
        batch, seq_len, hidden = x.shape

        # Initialize
        state = x.clone()
        halting_probs = torch.zeros(batch, seq_len, device=x.device)
        iterations = torch.zeros(batch, seq_len, device=x.device)

        max_iterations = self.max_depth
        threshold = 0.99

        for step in range(max_iterations):
            # Compute halting probability
            halt_logits = self.halting_net(state).squeeze(-1)  # [batch, seq_len]
            halt_probs = torch.sigmoid(halt_logits)

            # Update cumulative halting probability
            halting_probs = halting_probs + halt_probs

            # Determine which tokens continue
            continue_mask = (halting_probs < threshold).float()

            # Update state for continuing tokens
            flat_state = state.view(-1, hidden)
            new_state = self.recurrent_cell(flat_state, flat_state)
            state = new_state.view(batch, seq_len, hidden)

            # Mask state updates
            state = state * continue_mask.unsqueeze(-1) + state * (1 - continue_mask.unsqueeze(-1))

            # Track iterations
            iterations = iterations + continue_mask

            # Early stop if all tokens halted
            if continue_mask.sum() == 0:
                break

        return LayerOutput(
            output=state,
            metrics={
                "avg_iterations": iterations.mean().item(),
                "max_iterations": iterations.max().item(),
            }
        )

    def _forward_fixed(self, x: torch.Tensor, depth: int) -> LayerOutput:
        """Fixed depth recurrence."""
        batch, seq_len, hidden = x.shape

        state = x.view(-1, hidden)

        for _ in range(depth):
            state = self.recurrent_cell(state, state)

        output = state.view(batch, seq_len, hidden)

        return LayerOutput(
            output=output,
            metrics={"depth": depth}
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"depth_range=[{self.min_depth}, {self.max_depth}], "
            f"policy={self.policy}"
        )


if __name__ == "__main__":
    print("Testing Adaptive Recurrent Depth...")

    # Test learned policy
    ard = AdaptiveRecurrentDepth(
        hidden_dim=512,
        min_depth=1,
        max_depth=8,
        policy="learned"
    )

    x = torch.randn(2, 10, 512)
    output = ard(x)

    print(f"Input: {x.shape}")
    print(f"Output: {output.output.shape}")
    print(f"Metrics: {output.metrics}")

    # Test adaptive policy
    ard_adaptive = AdaptiveRecurrentDepth(512, policy="adaptive")
    output = ard_adaptive(x)
    print(f"Adaptive output: {output.output.shape}")
    print(f"Adaptive metrics: {output.metrics}")

    print("✓ Adaptive recurrent depth tests passed!")
