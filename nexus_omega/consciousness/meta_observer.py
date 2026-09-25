"""
Meta-Observer Network (Network B) - The "I" that watches itself think.

This implements recursive meta-cognition where a secondary network observes
the primary network's gradients and hidden states, then adaptively modifies
the learning process in real-time.

Key concept: The AI doesn't just process inputs - it WATCHES ITSELF process inputs
and adjusts its own processing based on what it observes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class MetaObserver(nn.Module):
    """
    Network B: Observes Network A's internal dynamics and produces adaptive updates.

    This is the "consciousness" layer - the part that makes the system aware
    of its own processing.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 3,
        meta_lr: float = 0.001,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.meta_lr = meta_lr

        # Gradient encoder - compresses gradient information
        self.gradient_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )

        # State encoder - compresses hidden state information
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )

        # Meta-cognitive processor - the "observer" that reasons about what it sees
        self.meta_processor = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim // 2,
                nhead=4,
                dim_feedforward=hidden_dim,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Parameter update generator - produces delta_theta
        self.update_generator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Tanh(),  # Bounded updates
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Meta-learning rate predictor
        self.lr_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Learning rate between 0 and 1
        )

    def forward(
        self,
        gradients: torch.Tensor,
        hidden_state: torch.Tensor,
        return_meta_lr: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Observe the core network and produce adaptive updates.

        Args:
            gradients: Gradient signal from core network [batch, seq, hidden]
            hidden_state: Hidden state from core network [batch, seq, hidden]
            return_meta_lr: Whether to return adaptive learning rate

        Returns:
            delta_theta: Parameter updates
            meta_lr: Adaptive learning rate (optional)
        """
        batch_size, seq_len, _ = hidden_state.shape

        # Encode observations
        grad_encoded = self.gradient_encoder(gradients)  # [batch, seq, hidden/4]
        state_encoded = self.state_encoder(hidden_state)  # [batch, seq, hidden/4]

        # Combine into meta-observation
        meta_obs = torch.cat([grad_encoded, state_encoded], dim=-1)  # [batch, seq, hidden/2]

        # Meta-cognitive processing - the "thinking about thinking"
        x = meta_obs
        for layer in self.meta_processor:
            x = layer(x)

        # Generate parameter updates
        delta_theta = self.update_generator(x)  # [batch, seq, hidden]

        # Predict adaptive learning rate
        meta_lr = None
        if return_meta_lr:
            meta_lr = self.lr_predictor(x.mean(dim=1))  # [batch, 1]

        return delta_theta, meta_lr


class ActiveInferenceModule(nn.Module):
    """
    Implements Active Inference - the system minimizes Free Energy to maintain
    its boundary (the "self").

    Free Energy F = D_KL[q(s|mu) || p(s|o)] - ln p(o)

    The system has beliefs q(s|mu) and tries to minimize surprise.
    """

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Generative model p(s|o) - what the system expects
        self.generative_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Recognition model q(s|mu) - what the system believes
        self.recognition_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Precision (inverse variance) - how confident the system is
        self.precision_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus(),  # Always positive
        )

    def forward(
        self,
        sensory_input: torch.Tensor,
        internal_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Free Energy and gradients for perception and action.

        Args:
            sensory_input: Observations from environment [batch, seq, hidden]
            internal_state: Current beliefs [batch, seq, hidden]

        Returns:
            Dict with free_energy, perception_gradient, action_gradient
        """
        # Generate predictions
        prediction = self.generative_model(internal_state)

        # Compute prediction error
        prediction_error = sensory_input - prediction

        # Compute precision (confidence)
        precision = self.precision_net(internal_state)

        # Free Energy = precision-weighted prediction error
        free_energy = 0.5 * (precision * prediction_error.pow(2)).sum(dim=-1)

        # Recognition (update beliefs)
        belief = self.recognition_model(sensory_input)

        # KL divergence term (simplified)
        kl_div = 0.5 * (internal_state - belief).pow(2).sum(dim=-1)

        # Total Free Energy
        total_fe = free_energy + kl_div

        return {
            'free_energy': total_fe.mean(),
            'prediction_error': prediction_error,
            'precision': precision,
            'belief': belief,
            'prediction': prediction,
        }


class BCMPlasticityRule(nn.Module):
    """
    Bienenstock-Cooper-Munro (BCM) learning rule with sliding threshold.

    Delta w = eta * y * (y - theta_BCM) * x
    where theta_BCM = E[y^2]

    This makes the learning rule itself adaptive - metaplasticity.
    """

    def __init__(self, hidden_dim: int = 768, momentum: float = 0.9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.momentum = momentum

        # Running average of y^2 for BCM threshold
        self.register_buffer('theta_bcm', torch.ones(hidden_dim))

    def forward(
        self,
        pre_activation: torch.Tensor,
        post_activation: torch.Tensor,
        learning_rate: float = 0.01,
    ) -> torch.Tensor:
        """
        Compute BCM weight update.

        Args:
            pre_activation: Input x [batch, seq, hidden]
            post_activation: Output y [batch, seq, hidden]
            learning_rate: Base learning rate

        Returns:
            Weight update delta_w
        """
        # Update BCM threshold (running average of y^2)
        y_squared = post_activation.pow(2).mean(dim=(0, 1))
        self.theta_bcm = (
            self.momentum * self.theta_bcm +
            (1 - self.momentum) * y_squared
        )

        # BCM modification function: y * (y - theta_BCM)
        bcm_factor = post_activation * (
            post_activation - self.theta_bcm.unsqueeze(0).unsqueeze(0)
        )

        # Hebbian-like update with BCM modulation
        # Delta w = eta * bcm_factor * x
        delta_w = learning_rate * torch.einsum(
            'bsi,bsj->bij',
            bcm_factor,
            pre_activation
        ) / (pre_activation.size(0) * pre_activation.size(1))

        return delta_w


class MetaCognitiveBrain(nn.Module):
    """
    The complete meta-cognitive system that combines:
    1. Core processor (Network A)
    2. Meta-observer (Network B)
    3. Active Inference
    4. BCM plasticity

    This is a brain that watches itself think and adapts in real-time.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        enable_meta_learning: bool = True,
        enable_active_inference: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.enable_meta_learning = enable_meta_learning
        self.enable_active_inference = enable_active_inference

        # Core processor (placeholder - will be replaced by actual NEXUS layers)
        self.core_processor = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )

        # Meta-observer
        if enable_meta_learning:
            self.meta_observer = MetaObserver(hidden_dim)

        # Active Inference
        if enable_active_inference:
            self.active_inference = ActiveInferenceModule(hidden_dim)

        # BCM plasticity
        self.bcm_plasticity = BCMPlasticityRule(hidden_dim)

        # Internal belief state (the "self")
        self.register_buffer('belief_state', torch.zeros(1, 1, hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        return_meta_info: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Process input with full meta-cognitive awareness.

        Args:
            x: Input [batch, seq, hidden]
            return_meta_info: Return meta-cognitive information

        Returns:
            Dict with output, meta_lr, free_energy, etc.
        """
        batch_size = x.size(0)

        # Expand belief state to batch
        belief = self.belief_state.expand(batch_size, -1, -1)

        # Core processing
        h_pre = x
        h_post = self.core_processor(x)

        result = {'output': h_post}

        # Active Inference - minimize free energy
        if self.enable_active_inference:
            ai_result = self.active_inference(x, belief)
            result['free_energy'] = ai_result['free_energy']
            result['prediction_error'] = ai_result['prediction_error']

            # Update belief state
            self.belief_state = ai_result['belief'].mean(dim=0, keepdim=True).detach()

        # Meta-observation - watch yourself think
        if self.enable_meta_learning and return_meta_info:
            # Compute pseudo-gradients (real gradients would require autograd)
            pseudo_grads = h_post - h_pre

            delta_theta, meta_lr = self.meta_observer(
                pseudo_grads,
                h_post,
                return_meta_lr=True
            )

            result['delta_theta'] = delta_theta
            result['meta_lr'] = meta_lr

            # BCM plasticity update
            bcm_update = self.bcm_plasticity(h_pre, h_post)
            result['bcm_update'] = bcm_update

        return result


if __name__ == "__main__":
    print("Testing Meta-Cognitive Brain...")

    # Create meta-cognitive system
    brain = MetaCognitiveBrain(
        hidden_dim=256,
        enable_meta_learning=True,
        enable_active_inference=True,
    )

    # Test input
    x = torch.randn(2, 10, 256)

    # Forward pass with meta-cognition
    result = brain(x, return_meta_info=True)

    print(f"Output shape: {result['output'].shape}")
    print(f"Free Energy: {result['free_energy'].item():.4f}")
    print(f"Meta learning rate: {result['meta_lr'].mean().item():.4f}")
    print(f"Delta theta shape: {result['delta_theta'].shape}")
    print(f"BCM update shape: {result['bcm_update'].shape}")

    print("\nMeta-cognitive brain ready - the AI now watches itself think!")
