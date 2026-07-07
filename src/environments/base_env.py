"""
Abstract base environment for resource-constrained POMDPs with irreversible actions.

This defines the shared interface that all domain-specific environments must implement.
The key constraint: actions of type COMMIT are permanent and decrement a finite budget.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class EnvironmentConfig:
    """Configuration for irreversible-action POMDP environments."""

    max_horizon: int = 200
    initial_budget: int = 10
    observation_noise_std: float = 0.1
    seed: Optional[int] = None


class IrreversibleActionPOMDP(gym.Env, ABC):
    """
    Base class for POMDPs with irreversible actions and finite budgets.

    The agent has two actions at each timestep:
        0: WAIT  — do nothing, observe, advance time
        1: COMMIT — execute irreversible action, consume one budget unit

    Subclasses implement domain-specific:
        - State transitions
        - Observation generation
        - Reward computation
        - Termination conditions
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    WAIT = 0
    COMMIT = 1

    def __init__(self, config: EnvironmentConfig, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(2)
        self.observation_space = self._build_observation_space()

        self._budget_remaining = config.initial_budget
        self._timestep = 0
        self._total_reward = 0.0
        self._committed_actions = []
        self._rng = np.random.default_rng(config.seed)

    @abstractmethod
    def _build_observation_space(self) -> spaces.Space:
        """Define the observation space for this domain."""
        ...

    @abstractmethod
    def _get_true_state(self) -> dict[str, Any]:
        """Return the full (hidden) environment state."""
        ...

    @abstractmethod
    def _generate_observation(self) -> np.ndarray:
        """Generate a partial, potentially noisy observation of the true state."""
        ...

    @abstractmethod
    def _compute_reward(self, action: int) -> float:
        """Compute immediate reward for the given action."""
        ...

    @abstractmethod
    def _apply_commit(self) -> None:
        """Apply the irreversible action to the environment state."""
        ...

    @abstractmethod
    def _advance_time(self) -> None:
        """Advance the environment forward one timestep (called for both WAIT and COMMIT)."""
        ...

    @abstractmethod
    def _check_terminal(self) -> bool:
        """Check if the episode has reached a terminal state (beyond budget/horizon)."""
        ...

    @abstractmethod
    def _compute_terminal_reward(self) -> float:
        """Compute any terminal bonus/penalty at end of episode."""
        ...

    @abstractmethod
    def _reset_domain(self, seed: Optional[int] = None) -> None:
        """Reset domain-specific state for a new episode."""
        ...

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._budget_remaining = self.config.initial_budget
        self._timestep = 0
        self._total_reward = 0.0
        self._committed_actions = []

        self._reset_domain(seed)

        obs = self._generate_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one environment step.

        Returns: (observation, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        if action == self.COMMIT:
            if self._budget_remaining <= 0:
                reward = -1.0
                terminated = True
                obs = self._generate_observation()
                return obs, reward, terminated, False, self._get_info()

            self._apply_commit()
            self._budget_remaining -= 1
            self._committed_actions.append(self._timestep)

        reward = self._compute_reward(action)
        self._advance_time()
        self._timestep += 1

        terminated = self._check_terminal()
        truncated = self._timestep >= self.config.max_horizon

        if terminated or truncated:
            reward += self._compute_terminal_reward()

        self._total_reward += reward
        obs = self._generate_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_info(self) -> dict[str, Any]:
        """Return info dict with metadata about current state."""
        return {
            "timestep": self._timestep,
            "budget_remaining": self._budget_remaining,
            "budget_used": self.config.initial_budget - self._budget_remaining,
            "total_reward": self._total_reward,
            "committed_at": list(self._committed_actions),
        }

    @property
    def budget_remaining(self) -> int:
        return self._budget_remaining

    @property
    def budget_fraction(self) -> float:
        return self._budget_remaining / self.config.initial_budget

    @property
    def time_fraction(self) -> float:
        return self._timestep / self.config.max_horizon
