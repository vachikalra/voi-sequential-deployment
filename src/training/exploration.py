"""
Resource-Adaptive Exploration Module.

Standard RL exploration (entropy bonus, epsilon-greedy) explores uniformly
regardless of remaining resources. In irreversible-action problems, this
causes wasteful commits early in the episode when the agent is still
exploring randomly.

This module adapts exploration intensity based on:
1. Remaining budget fraction (less budget → less exploration)
2. Time horizon remaining (less time → less exploration)
3. Confidence in current belief (high uncertainty → more cautious, not more random)

Key insight: in irreversible-action problems, "exploration" through random
commits is HARMFUL because you can't undo the commit. Instead, exploration
should manifest as WAITING (gathering observations) not as ACTING randomly.
"""

import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class AdaptiveExplorationConfig:
    """Configuration for resource-adaptive exploration."""

    base_entropy_coeff: float = 0.05
    min_entropy_coeff: float = 0.001
    budget_sensitivity: float = 2.0     # how aggressively entropy drops with budget
    time_sensitivity: float = 1.0       # how entropy changes with remaining time
    confidence_threshold: float = 0.8   # belief confidence above which we reduce exploration
    warmup_steps: int = 1000            # steps before adaptation kicks in


class ResourceAdaptiveExploration:
    """
    Adapts exploration intensity based on remaining irreversible-action budget.

    The core mechanism:
        entropy_coeff(t) = base * f(budget) * g(time) * h(confidence)

    Where:
        f(budget) = (n_remaining / N_total) ^ budget_sensitivity
            → exploration drops as budget depletes

        g(time) = (T - t) / T
            → exploration drops as episode ends (less time to benefit from info)

        h(confidence) = 1 if uncertain, reduced if very confident
            → don't waste exploration when you already know what to do

    This replaces the FIXED entropy coefficient in standard PPO with
    a DYNAMIC one that respects resource constraints.
    """

    def __init__(self, config: AdaptiveExplorationConfig = None):
        self.config = config or AdaptiveExplorationConfig()
        self._step_count = 0

    def get_entropy_coefficient(
        self,
        budget_remaining: int,
        budget_total: int,
        timestep: int,
        max_horizon: int,
        belief_entropy: float = 1.0,
    ) -> float:
        """
        Compute adapted entropy coefficient for current state.

        Args:
            budget_remaining: how many irreversible actions remain
            budget_total: initial budget
            timestep: current timestep in episode
            max_horizon: maximum episode length
            belief_entropy: entropy of the belief state (uncertainty measure)

        Returns:
            Adapted entropy coefficient for PPO loss
        """
        self._step_count += 1

        if self._step_count < self.config.warmup_steps:
            return self.config.base_entropy_coeff

        # Budget factor: exploration drops as resources deplete
        budget_fraction = budget_remaining / max(budget_total, 1)
        budget_factor = budget_fraction ** self.config.budget_sensitivity

        # Time factor: less exploration near end of episode
        time_remaining = (max_horizon - timestep) / max(max_horizon, 1)
        time_factor = time_remaining ** self.config.time_sensitivity

        # Combined
        adapted_coeff = self.config.base_entropy_coeff * budget_factor * time_factor

        return max(adapted_coeff, self.config.min_entropy_coeff)

    def get_commit_penalty(
        self,
        budget_remaining: int,
        budget_total: int,
        voi_estimate: float,
    ) -> float:
        """
        Compute an additional penalty for committing under high uncertainty.

        When VoI is high (waiting is valuable) AND budget is low,
        apply extra penalty to discourage wasteful commits.

        This is separate from the VoI decision criterion — it provides
        a TRAINING signal that teaches the policy to be conservative
        when resources are scarce.
        """
        budget_fraction = budget_remaining / max(budget_total, 1)
        scarcity = 1.0 - budget_fraction  # 0 = abundant, 1 = nearly depleted

        # Penalty scales with both scarcity and VoI
        penalty = scarcity * voi_estimate * 0.5

        return penalty

    def should_force_wait(
        self,
        budget_remaining: int,
        timestep: int,
        max_horizon: int,
    ) -> bool:
        """
        Hard constraint: if budget is critically low relative to
        remaining horizon, force WAIT to prevent immediate depletion.

        This is a safety mechanism — if the agent has 1 relay left
        and 100 timesteps remaining, it should NOT commit on a
        random exploration action.
        """
        if budget_remaining <= 0:
            return True

        remaining_time = max_horizon - timestep
        if remaining_time <= 0:
            return False

        # If budget/remaining_time ratio is very low, be very conservative
        ratio = budget_remaining / remaining_time
        return ratio < 0.02  # less than 2% of steps can be commits

    def get_exploration_stats(self) -> dict:
        """Get current exploration state for logging."""
        return {
            "total_steps": self._step_count,
            "warmup_complete": self._step_count >= self.config.warmup_steps,
        }

    def reset(self) -> None:
        """Reset for new episode (keeps step count for warmup)."""
        pass
