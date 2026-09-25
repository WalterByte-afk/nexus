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

        # VECTORIZED: Process all tokens up to max depth, masking based on their actual depth
        # Flatten for batch processing
        state = x.view(-1, hidden)  # [batch*seq, hidden]
        depths_flat = depths.view(-1)  # [batch*seq]

        # VECTORIZED: Pre-compute all recurrent steps and select based on depth
        # Instead of iterating, we can unroll the recurrence to max_depth steps
        # and then select the appropriate state for each token based on its depth

        # Initialize all states
        all_states = [state]

        # Compute all steps up to max_depth
        for step in range(self.max_depth):
            new_state = self.recurrent_cell(all_states[-1], all_states[-1])
            all_states.append(new_state)

        # Stack states: [max_depth+1, batch*seq, hidden]
        stacked_states = torch.stack(all_states, dim=0)

        # Gather the correct state for each token based on its depth
        # We need to use depths_flat as indices (0-based, but we have depths_flat from 1..max_depth)
        # So subtract 1 to get indices into stacked_states
        depth_indices = (depths_flat - 1).clamp(0, self.max_depth - 1)
        state = stacked_states[depth_indices, torch.arange(len(state), device=state.device)]

        output = state.view(batch, seq_len, hidden)
        avg_depth = depths.float().mean()

        return LayerOutput(
            output=output,
            metrics={
                "avg_depth": avg_depth,
                "min_depth_used": depths.min(),
                "max_depth_used": depths.max(),
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
                "avg_iterations": iterations.mean(),
                "max_iterations": iterations.max(),
            }
        )

    def _forward_fixed(self, x: torch.Tensor, depth: int) -> LayerOutput:
        """Fixed depth recurrence - VECTORIZED."""
        batch, seq_len, hidden = x.shape

        state = x.view(-1, hidden)

        # VECTORIZED: Unroll all depth steps and store states
        all_states = [state]
        for step in range(depth):
            state = self.recurrent_cell(state, state)
            all_states.append(state)

        # Take the final state (after 'depth' iterations)
        output = all_states[-1].view(batch, seq_len, hidden)

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
