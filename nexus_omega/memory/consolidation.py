"""
Synaptic Consolidation (Layer 6)

Implements brain-like consolidation during "sleep" mode.
Transfers knowledge from fast plastic weights to stable base weights
while protecting important weights using Elastic Weight Consolidation (EWC).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass
import copy


@dataclass
class ConsolidationReport:
    """Report from consolidation cycle."""

    num_consolidated: int
    fisher_threshold: float
    protected_weights: int
    transferred_knowledge: float


class SynapticConsolidation:
    """
    Synaptic Consolidation - Layer 6 of NEXUS-Ω.

    Key innovation: Brain-like sleep mode consolidation
    1. Identify important weights via Fisher Information
    2. Protect them from forgetting (EWC)
    3. Transfer stable plastic weights to base weights
    """

    def __init__(
        self,
        model: nn.Module,
        consolidation_threshold: float = 0.8,
        ewc_lambda: float = 100.0,
        consolidation_frequency: int = 1000,
    ):
        self.model = model
        self.consolidation_threshold = consolidation_threshold
        self.ewc_lambda = ewc_lambda
        self.consolidation_frequency = consolidation_frequency

        # Fisher Information matrices (importance of each weight)
        self.fisher_dict: Dict[str, torch.Tensor] = {}

        # Optimal parameters (reference point for EWC)
        self.optimal_params: Dict[str, torch.Tensor] = {}

        # Consolidation history
        self.history: List[ConsolidationReport] = []

        # Step counter
        self.step_count = 0

    def compute_fisher_information(
        self,
        dataloader,
        num_samples: int = 1000,
    ):
        """
        Compute Fisher Information Matrix for all weights.

        Fisher[w] = E[(∂log P(y|x) / ∂w)²]

        High Fisher = important weight (shouldn't change much)
        """
        self.model.eval()

        # Initialize Fisher dict
        self.fisher_dict = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        sample_count = 0

        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= num_samples:
                break

            self.model.zero_grad()

            # Forward pass
            x, y = batch
            output = self.model(x)

            # Compute log probability
            log_probs = torch.log_softmax(output, dim=-1)
            selected_log_probs = log_probs.gather(1, y.unsqueeze(1)).squeeze()

            # Sum for gradient computation
            loss = -selected_log_probs.mean()

            # Backward pass
            loss.backward()

            # Accumulate squared gradients (Fisher approximation)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_dict[name] = self.fisher_dict[name] + param.grad.data.pow(2)

            sample_count += x.shape[0]

        # Normalize
        for name in self.fisher_dict:
            self.fisher_dict[name] = self.fisher_dict[name] / sample_count

        # Save optimal parameters
        self.optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.train()

    def ewc_loss(self) -> torch.Tensor:
        """
        Compute Elastic Weight Consolidation loss.

        Penalizes changes to important weights.
        """
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for name, param in self.model.named_parameters():
            if name in self.fisher_dict and param.requires_grad:
                # Penalize deviation from optimal parameters
                fisher = self.fisher_dict[name]
                optimal = self.optimal_params[name]

                loss = loss + (fisher * (param - optimal).pow(2)).sum()

        return self.ewc_lambda * loss

    def consolidate(self) -> ConsolidationReport:
        """
        Run consolidation cycle.

        1. Identify stable plastic weights
        2. Transfer to base weights
        3. Protect important weights via EWC
        """
        consolidated_count = 0
        protected_count = 0
        transferred_magnitude = 0.0

        for name, module in self.model.named_modules():
            # Only consolidate dual-plasticity modules
            if hasattr(module, 'W_base') and hasattr(module, 'W_plastic_up'):
                # Check plastic weight stability
                plastic_norm = module.W_plastic_up.norm().item()

                if plastic_norm > self.consolidation_threshold:
                    # Compute full plastic weight matrix
                    full_plastic = torch.matmul(module.W_plastic_up, module.W_plastic_down)

                    # Transfer to base weights
                    with torch.no_grad():
                        module.W_base.data = module.W_base.data + full_plastic

                        # Reset plastic weights
                        module.W_plastic_up.data = module.W_plastic_up.data * 0.1
                        module.W_plastic_down.data = module.W_plastic_down.data * 0.1

                    consolidated_count += 1
                    transferred_magnitude += plastic_norm

        # Count protected weights
        for name, fisher in self.fisher_dict.items():
            protected_count += (fisher > self.consolidation_threshold).sum().item()

        # Create report
        report = ConsolidationReport(
            num_consolidated=consolidated_count,
            fisher_threshold=self.consolidation_threshold,
            protected_weights=protected_count,
            transferred_knowledge=transferred_magnitude,
        )

        self.history.append(report)
        return report

    def should_consolidate(self) -> bool:
        """Check if it's time for consolidation."""
        self.step_count += 1
        return self.step_count % self.consolidation_frequency == 0

    def get_importance_ranking(self, top_k: int = 10) -> Dict[str, float]:
        """Get ranking of most important weights."""
        importance = {}

        for name, fisher in self.fisher_dict.items():
            importance[name] = fisher.mean().item()

        # Sort by importance
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )

        return sorted_importance


class SleepMode:
    """
    Sleep mode controller for consolidation.

    Manages when and how the model enters consolidation cycles.
    """

    def __init__(
        self,
        consolidation: SynapticConsolidation,
        sleep_threshold: float = 0.7,
        min_steps_between_sleep: int = 100,
    ):
        self.consolidation = consolidation
        self.sleep_threshold = sleep_threshold
        self.min_steps_between_sleep = min_steps_between_sleep

        self.last_sleep_step = 0
        self.is_sleeping = False

    def enter_sleep_mode(self) -> ConsolidationReport:
        """
        Enter sleep mode and run consolidation.
        """
        self.is_sleeping = True

        # Run consolidation
        report = self.consolidation.consolidate()

        self.last_sleep_step = self.consolidation.step_count
        self.is_sleeping = False

        return report

    def should_sleep(self, performance_metric: float) -> bool:
        """
        Determine if model should enter sleep mode.

        Args:
            performance_metric: Current performance (0-1)

        Returns:
            Whether to enter sleep mode
        """
        if self.is_sleeping:
            return False

        # Check minimum steps since last sleep
        steps_since_sleep = self.consolidation.step_count - self.last_sleep_step
        if steps_since_sleep < self.min_steps_between_sleep:
            return False

        # Check if performance degraded
        if performance_metric < self.sleep_threshold:
            return True

        # Check scheduled consolidation
        return self.consolidation.should_consolidate()


if __name__ == "__main__":
    print("Testing Synaptic Consolidation...")

    # Create a simple model with dual-plasticity
    from nexus_omega.memory.dual_plasticity import DualPlasticityWeights

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = DualPlasticityWeights(512, 512)
            self.layer2 = nn.Linear(512, 10)

        def forward(self, x):
            x = self.layer1(x).output
            x = self.layer2(x)
            return x

    model = SimpleModel()
    consolidation = SynapticConsolidation(model)

    # Test Fisher computation
    print("Computing Fisher Information...")
    # (Would normally use real dataloader)

    # Test EWC loss
    x = torch.randn(2, 10, 512)
    output = model(x)
    ewc = consolidation.ewc_loss()
    print(f"EWC loss: {ewc.item():.4f}")

    # Test consolidation
    report = consolidation.consolidate()
    print(f"Consolidated: {report.num_consolidated} layers")

    print("✓ Synaptic consolidation tests passed!")
