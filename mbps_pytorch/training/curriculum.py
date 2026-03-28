"""Training Curriculum for MBPS.

Controls loss weight scheduling across the 4 training phases:
    Phase A (1-20): Semantic only
    Phase B (21-40): Add instance with gradient projection
    Phase C (41-60): Full model with bridge + consistency + PQ
    Phase D: Self-training refinement
"""

from __future__ import annotations

from typing import Dict, NamedTuple


class PhaseConfig(NamedTuple):
    """Configuration for a single training phase."""

    alpha: float  # semantic weight
    beta: float   # instance weight
    gamma: float  # bridge weight
    delta: float  # consistency weight
    epsilon: float  # PQ proxy weight
    use_gradient_projection: bool
    use_bridge: bool
    use_consistency: bool
    use_pq_loss: bool


# Default phase configs from SKILL.md
PHASE_CONFIGS = {
    "A": PhaseConfig(1.0, 0.0, 0.0, 0.0, 0.0, False, False, False, False),
    "B": PhaseConfig(1.0, 1.0, 0.0, 0.0, 0.0, True, False, False, False),
    "C": PhaseConfig(0.8, 1.0, 0.1, 0.3, 0.2, False, True, True, True),
    "D": PhaseConfig(0.6, 1.0, 0.1, 0.4, 0.3, False, True, True, True),
}


class TrainingCurriculum:
    """Manages loss weight scheduling and phase transitions.

    Args:
        phase_a_end: Last epoch of Phase A.
        phase_b_end: Last epoch of Phase B.
        total_epochs: Total training epochs.
    """

    def __init__(
        self,
        phase_a_end: int = 20,
        phase_b_end: int = 40,
        total_epochs: int = 60,
    ):
        self.phase_a_end = phase_a_end
        self.phase_b_end = phase_b_end
        self.total_epochs = total_epochs

    def get_phase(self, epoch: int) -> str:
        """Get current training phase.

        Args:
            epoch: Current epoch (1-indexed).

        Returns:
            Phase name ('A', 'B', or 'C').
        """
        if epoch <= self.phase_a_end:
            return "A"
        elif epoch <= self.phase_b_end:
            return "B"
        else:
            return "C"

    def get_config(self, epoch: int) -> PhaseConfig:
        """Get phase configuration for current epoch.

        Handles smooth transitions:
        - Phase B: beta ramps from 0->1 linearly
        - Phase transitions: smooth weight interpolation

        Args:
            epoch: Current epoch.

        Returns:
            PhaseConfig with appropriate weights.
        """
        phase = self.get_phase(epoch)
        config = PHASE_CONFIGS[phase]

        if phase == "B":
            # Linear ramp-up of instance weight
            progress = (epoch - self.phase_a_end) / (
                self.phase_b_end - self.phase_a_end
            )
            beta = progress * config.beta
            return config._replace(beta=beta)

        return config

    def get_loss_weights(self, epoch: int) -> Dict[str, float]:
        """Get loss weights as a dictionary.

        Args:
            epoch: Current epoch.

        Returns:
            Dict of loss weight names -> values.
        """
        config = self.get_config(epoch)
        return {
            "alpha": config.alpha,
            "beta": config.beta,
            "gamma": config.gamma,
            "delta": config.delta,
            "epsilon": config.epsilon,
        }

    def should_use_bridge(self, epoch: int) -> bool:
        """Check if bridge should be active."""
        return self.get_config(epoch).use_bridge

    def should_use_gradient_projection(self, epoch: int) -> bool:
        """Check if gradient projection should be used."""
        return self.get_config(epoch).use_gradient_projection
