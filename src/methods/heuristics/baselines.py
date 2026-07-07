"""
Heuristic baselines for relay deployment.

These represent the "current practice" — simple rule-based strategies
that the learned VoI agent must outperform.
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseHeuristic(ABC):
    """Base class for heuristic deployment strategies."""

    @abstractmethod
    def decide(self, observation: dict) -> int:
        """
        Decide whether to deploy (1) or wait (0).

        Args:
            observation: environment observation dict

        Returns:
            action: 0 (WAIT) or 1 (COMMIT)
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for new episode."""
        ...


class FixedIntervalHeuristic(BaseHeuristic):
    """
    Deploy a relay every K timesteps regardless of conditions.

    This is the simplest possible strategy and represents a lower bound
    on deployment intelligence. Equivalent to spacing relays evenly
    without considering environmental conditions.
    """

    def __init__(self, interval: int = 15):
        self.interval = interval
        self._step_count = 0

    def decide(self, observation: dict) -> int:
        self._step_count += 1
        if self._step_count >= self.interval:
            self._step_count = 0
            return 1
        return 0

    def reset(self) -> None:
        self._step_count = 0


class SignalThresholdHeuristic(BaseHeuristic):
    """
    Deploy when signal strength drops below a threshold.

    This represents the reactive approach used in most real-world
    relay systems (including the physical prototype that inspired
    this research). The relay is deployed when SINR approaches
    the minimum acceptable quality.
    """

    def __init__(self, sinr_threshold_db: float = 15.0):
        self.threshold = sinr_threshold_db

    def decide(self, observation: dict) -> int:
        min_sinr = observation.get("min_sinr", np.array([50.0]))
        if isinstance(min_sinr, np.ndarray):
            min_sinr = min_sinr[0]

        if min_sinr <= self.threshold:
            return 1
        return 0

    def reset(self) -> None:
        pass


class GreedyLookaheadHeuristic(BaseHeuristic):
    """
    Deploy at the position that maximizes local signal improvement.

    Slightly smarter than threshold: estimates whether deploying NOW
    or in the next few steps gives better signal quality. Uses a
    simple predictive model based on signal decay rate.

    This represents a "smart engineer" baseline — better than reactive
    but without learning or deep planning.
    """

    def __init__(
        self,
        sinr_threshold_db: float = 18.0,
        lookahead_steps: int = 5,
        decay_rate_estimate: float = 2.0,  # estimated dB loss per step
    ):
        self.threshold = sinr_threshold_db
        self.lookahead = lookahead_steps
        self.decay_rate = decay_rate_estimate
        self._prev_sinr = None
        self._estimated_decay = decay_rate_estimate

    def decide(self, observation: dict) -> int:
        min_sinr = observation.get("min_sinr", np.array([50.0]))
        if isinstance(min_sinr, np.ndarray):
            min_sinr = min_sinr[0]

        budget_frac = observation.get("budget_remaining", np.array([1.0]))
        if isinstance(budget_frac, np.ndarray):
            budget_frac = budget_frac[0]

        # Update decay estimate
        if self._prev_sinr is not None:
            observed_decay = self._prev_sinr - min_sinr
            self._estimated_decay = 0.9 * self._estimated_decay + 0.1 * max(observed_decay, 0)
        self._prev_sinr = min_sinr

        # Predict signal in lookahead steps
        predicted_future_sinr = min_sinr - self._estimated_decay * self.lookahead

        # Deploy if:
        # 1. Signal is already below threshold, OR
        # 2. Signal will drop below threshold within lookahead AND budget allows waiting
        if min_sinr <= self.threshold:
            return 1
        elif predicted_future_sinr <= self.threshold and budget_frac > 0.3:
            # Deploy now (before it gets worse) if we have budget headroom
            return 1

        return 0

    def reset(self) -> None:
        self._prev_sinr = None
        self._estimated_decay = 2.0
