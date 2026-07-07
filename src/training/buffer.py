"""
Rollout buffer for PPO training.

Stores transitions collected during environment interaction and computes
Generalized Advantage Estimation (GAE) for policy gradient updates.
"""

import numpy as np
import torch
from typing import Optional


class RolloutBuffer:
    """
    Stores rollout data and computes advantages using GAE.

    For recurrent policies, stores sequences per episode rather than
    individual transitions.
    """

    def __init__(
        self,
        buffer_size: int = 2048,
        obs_dim: int = 24,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.voi_estimates = np.zeros(buffer_size, dtype=np.float32)

        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        voi: float = 0.0,
    ) -> None:
        """Add a single transition to the buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)
        self.voi_estimates[self.ptr] = voi
        self.ptr += 1

        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_advantages(self, last_value: float = 0.0) -> None:
        """
        Compute GAE advantages and discounted returns.

        GAE(γ,λ): A_t = Σ_{l=0}^{T-t} (γλ)^l * δ_{t+l}
        where δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        """
        last_gae = 0.0
        n = self.ptr if not self.full else self.buffer_size

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = 0.0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            # TD error
            delta = (
                self.rewards[t]
                + self.gamma * next_value * (1 - self.dones[t])
                - self.values[t]
            )

            # GAE
            last_gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * last_gae
            self.advantages[t] = last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get(self) -> dict:
        """Get all stored data as a dictionary of numpy arrays."""
        n = self.ptr if not self.full else self.buffer_size
        return {
            "observations": self.observations[:n],
            "actions": self.actions[:n],
            "rewards": self.rewards[:n],
            "values": self.values[:n],
            "log_probs": self.log_probs[:n],
            "dones": self.dones[:n],
            "advantages": self.advantages[:n],
            "returns": self.returns[:n],
            "voi_estimates": self.voi_estimates[:n],
        }

    def reset(self) -> None:
        """Reset buffer for next rollout collection."""
        self.ptr = 0
        self.full = False
