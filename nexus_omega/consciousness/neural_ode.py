"""
Continuous-Time Neural ODEs - Moving beyond discrete token processing.

Instead of h_{t+1} = f(h_t, x_t), we use:
dh(t)/dt = f(h(t), x(t), t, theta)

This allows the network to operate in continuous time, handling irregular
temporal dependencies and enabling true dynamic adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple, Optional


class NeuralODEFunc(nn.Module):
    """
    The function f(h, t) that defines the dynamics of the hidden state.

    This is what gets integrated over time using RK4 or other ODE solvers.
    """

    def __init__(self, hidden_dim: int = 768, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # The dynamics network
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),  # Bounded dynamics for stability
            ])

        self.dynamics = nn.Sequential(*layers)

        # Time embedding - the network knows "when" it is
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute dh/dt at time t.

        Args:
            t: Current time [1] (scalar broadcast to batch)
            h: Hidden state [batch, seq, hidden]

        Returns:
            dh/dt with same shape as h
        """
        # Time conditioning
        t_embed = self.time_embed(t.view(1, 1, 1).expand_as(h[..., :1]))

        # Dynamics
        h_with_time = h + t_embed
        dh_dt = self.dynamics(h_with_time)

        return dh_dt


class RK4Integrator:
    """
    4th-order Runge-Kutta integration for solving ODEs.

    Given dh/dt = f(h, t), compute h(t + dt) from h(t).

    The RK4 method:
    k1 = f(t, h)
    k2 = f(t + dt/2, h + dt*k1/2)
    k3 = f(t + dt/2, h + dt*k2/2)
    k4 = f(t + dt, h + dt*k3)
    h_next = h + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    """

    @staticmethod
    def step(
        func: Callable,
        t: float,
        h: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        Single RK4 integration step.

        Args:
            func: The ODE function f(t, h)
            t: Current time
            h: Current state
            dt: Time step size

        Returns:
            h at time t + dt
        """
        k1 = func(torch.tensor([t], device=h.device, dtype=h.dtype), h)
        k2 = func(torch.tensor([t + dt/2], device=h.device, dtype=h.dtype), h + dt * k1 / 2)
        k3 = func(torch.tensor([t + dt/2], device=h.device, dtype=h.dtype), h + dt * k2 / 2)
        k4 = func(torch.tensor([t + dt], device=h.device, dtype=h.dtype), h + dt * k3)

        h_next = h + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        return h_next


class ContinuousTimeLayer(nn.Module):
    """
    A neural network layer that operates in continuous time.

    Instead of discrete transformations, this integrates an ODE from t=0 to t=1
    to compute the output.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        integration_time: float = 1.0,
        num_steps: int = 10,
        method: str = "rk4",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.integration_time = integration_time
        self.num_steps = num_steps
        self.method = method

        # The ODE function
        self.ode_func = NeuralODEFunc(hidden_dim)

        # Integrator
        if method == "rk4":
            self.integrator = RK4Integrator()
        else:
            raise ValueError(f"Unknown integration method: {method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Integrate from t=0 to t=integration_time.

        Args:
            x: Initial state [batch, seq, hidden]

        Returns:
            Final state after integration
        """
        h = x
        dt = self.integration_time / self.num_steps
        t = 0.0

        for _ in range(self.num_steps):
            h = self.integrator.step(self.ode_func, t, h, dt)
            t += dt

        return h


class AdaptiveTimeProcessor(nn.Module):
    """
    Processes sequences with adaptive time steps - learns how fast to "think".

    Some inputs need slow, careful processing (integration_time = 2.0).
    Others can be processed quickly (integration_time = 0.5).

    The network learns this adaptively.
    """

    def __init__(self, hidden_dim: int = 768, num_steps: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps

        # Time predictor - learns how long to think
        self.time_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Between 0 and 1
        )

        # Scale to reasonable range [0.1, 2.0]
        self.time_scale = 1.9
        self.time_offset = 0.1

        # The ODE processor
        self.ode_func = NeuralODEFunc(hidden_dim)
        self.integrator = RK4Integrator()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process with adaptive integration time.

        Args:
            x: Input [batch, seq, hidden]

        Returns:
            output: Processed state
            integration_times: How long each sample was processed
        """
        # Predict integration time for each sample
        time_raw = self.time_predictor(x.mean(dim=1))  # [batch, 1]
        integration_time = time_raw * self.time_scale + self.time_offset

        # Process each sample with its predicted time
        batch_size = x.size(0)
        outputs = []

        for i in range(batch_size):
            h = x[i:i+1]
            t_total = integration_time[i].item()
            dt = t_total / self.num_steps
            t = 0.0

            for _ in range(self.num_steps):
                h = self.integrator.step(self.ode_func, t, h, dt)
                t += dt

            outputs.append(h)

        output = torch.cat(outputs, dim=0)

        return output, integration_time


class TemporalSelfAttention(nn.Module):
    """
    Self-attention that operates in continuous time.

    Instead of attending to discrete positions, attends to continuous
    time intervals with learned temporal kernels.
    """

    def __init__(self, hidden_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # QKV projections
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)

        # Temporal kernel - learned basis functions for time
        self.temporal_kernel = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, num_heads),
            nn.Softmax(dim=-1),
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Temporal self-attention.

        Args:
            x: Input [batch, seq, hidden]
            timestamps: Time of each token [batch, seq] (optional)

        Returns:
            Attended output
        """
        batch_size, seq_len, _ = x.shape

        # If no timestamps provided, use uniform spacing
        if timestamps is None:
            timestamps = torch.linspace(0, 1, seq_len, device=x.device)
            timestamps = timestamps.unsqueeze(0).expand(batch_size, -1)

        # Compute QKV
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each is [batch, seq, heads, head_dim]

        # Compute attention scores
        scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / (self.head_dim ** 0.5)

        # Temporal modulation
        # For each query-key pair, compute temporal distance and modulate
        time_diff = (timestamps.unsqueeze(2) - timestamps.unsqueeze(1)).abs()  # [batch, seq, seq]
        temporal_weight = self.temporal_kernel(time_diff.unsqueeze(-1))  # [batch, seq, seq, heads]
        temporal_weight = temporal_weight.permute(0, 3, 1, 2)  # [batch, heads, seq, seq]

        # Modulate attention with temporal kernel
        scores = scores * temporal_weight

        # Attention
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bhqk,bkhd->bqhd', attn, v)

        # Reshape and project
        out = out.reshape(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)

        return out


if __name__ == "__main__":
    print("Testing Continuous-Time Neural Layers...")

    # Test 1: Basic Neural ODE
    print("\n1. Neural ODE Layer")
    ode_layer = ContinuousTimeLayer(hidden_dim=256, num_steps=10)
    x = torch.randn(2, 10, 256)
    out = ode_layer(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")

    # Test 2: Adaptive Time Processing
    print("\n2. Adaptive Time Processor")
    adaptive = AdaptiveTimeProcessor(hidden_dim=256, num_steps=10)
    out, times = adaptive(x)
    print(f"Output: {out.shape}")
    print(f"Integration times: {times.squeeze().tolist()}")

    # Test 3: Temporal Self-Attention
    print("\n3. Temporal Self-Attention")
    temp_attn = TemporalSelfAttention(hidden_dim=256, num_heads=8)
    timestamps = torch.linspace(0, 1, 10).unsqueeze(0).expand(2, -1)
    out = temp_attn(x, timestamps)
    print(f"Output: {out.shape}")

    print("\nContinuous-time processing ready - no more discrete tokens!")
