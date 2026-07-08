"""
Value-of-Information Estimator.

Learns to predict how much decision quality will improve from one additional
observation. Trained via hindsight: after each episode, we can compute the
REALIZED information value at each timestep and use it as a regression target.

The VoI at time t is defined as:
    VoI(t) = max_a Q(o_{1:t+1}, a) - max_a Q(o_{1:t}, a)

This is the improvement in optimal Q-value after receiving observation o_{t+1}.
We train a network to PREDICT this quantity from o_{1:t} alone (before seeing o_{t+1}).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class VoIEstimatorConfig:
    """Configuration for VoI estimator training."""
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs_per_update: int = 5
    target_update_rate: float = 0.005  # Polyak averaging for target network
    history_buffer_size: int = 10000
    min_samples_to_train: int = 500


class VoIEstimator:
    """
    Trains the VoI head using hindsight-computed targets.

    After each episode rollout:
    1. For each timestep t, we have the belief state b_t
    2. We compute Q(b_t, COMMIT) and Q(b_t, WAIT) using the value network
    3. At t+1, we compute Q(b_{t+1}, COMMIT) and Q(b_{t+1}, WAIT)
    4. The realized VoI is: max_a Q(b_{t+1}, a) - max_a Q(b_t, a)
    5. If VoI > 0: waiting was valuable (information improved decisions)
       If VoI ≈ 0: waiting gave no new useful info
    6. We train the VoI head to predict this quantity from b_t

    This is a form of HINDSIGHT EXPERIENCE where we label past states
    with information that was only available later.
    """

    def __init__(
        self,
        voi_network: nn.Module,
        value_network: nn.Module,
        config: VoIEstimatorConfig = None,
    ):
        self.voi_net = voi_network
        self.value_net = value_network
        self.config = config or VoIEstimatorConfig()

        self.optimizer = torch.optim.Adam(
            self.voi_net.parameters(), lr=self.config.learning_rate
        )

        # History buffer for (belief_state, realized_voi) pairs
        self._belief_buffer: List[torch.Tensor] = []
        self._voi_target_buffer: List[float] = []

    def compute_hindsight_voi(
        self,
        belief_sequence: torch.Tensor,
        reward_sequence: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        """
        Compute realized VoI for each timestep in a completed episode.

        Args:
            belief_sequence: [T, belief_dim] — belief states over episode
            reward_sequence: [T] — rewards received
            gamma: discount factor

        Returns:
            voi_targets: [T-1] — realized VoI for each non-terminal step
        """
        with torch.no_grad():
            # Compute values at each timestep
            values = self.value_net(belief_sequence).squeeze(-1)  # [T]

            # VoI at time t = max(V(t+1) - V(t), 0)
            # (information can only help, never hurt, in expectation)
            value_improvements = values[1:] - values[:-1]  # [T-1]
            voi_targets = torch.clamp(value_improvements, min=0.0)

        return voi_targets

    def store_episode(
        self,
        belief_sequence: torch.Tensor,
        reward_sequence: torch.Tensor,
        gamma: float = 0.99,
    ) -> None:
        """Store computed VoI targets from a completed episode."""
        voi_targets = self.compute_hindsight_voi(belief_sequence, reward_sequence, gamma)

        # Store belief states (excluding last, which has no VoI target)
        for t in range(len(voi_targets)):
            self._belief_buffer.append(belief_sequence[t].detach())
            self._voi_target_buffer.append(voi_targets[t].item())

        # Trim buffer if too large
        if len(self._belief_buffer) > self.config.history_buffer_size:
            self._belief_buffer = self._belief_buffer[-self.config.history_buffer_size:]
            self._voi_target_buffer = self._voi_target_buffer[-self.config.history_buffer_size:]

    def update(self) -> Optional[float]:
        """
        Train VoI head on stored (belief, realized_voi) pairs.

        Returns:
            Mean loss over training batch, or None if insufficient data.
        """
        if len(self._belief_buffer) < self.config.min_samples_to_train:
            return None

        beliefs = torch.stack(self._belief_buffer)
        targets = torch.tensor(self._voi_target_buffer, dtype=torch.float32)

        total_loss = 0.0
        n_updates = 0

        for _ in range(self.config.n_epochs_per_update):
            # Random mini-batch
            indices = torch.randperm(len(beliefs))[:self.config.batch_size]
            batch_beliefs = beliefs[indices]
            batch_targets = targets[indices]

            # Forward pass
            predicted_voi = self.voi_net(batch_beliefs).squeeze(-1)

            # Huber loss (robust to outliers in VoI targets)
            loss = nn.functional.huber_loss(predicted_voi, batch_targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.voi_net.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()
            n_updates += 1

        return total_loss / max(n_updates, 1)

    def get_voi_stats(self) -> dict[str, float]:
        """Get statistics about stored VoI targets (for logging)."""
        if not self._voi_target_buffer:
            return {"mean_voi": 0.0, "std_voi": 0.0, "max_voi": 0.0}

        arr = np.array(self._voi_target_buffer)
        return {
            "mean_voi": float(np.mean(arr)),
            "std_voi": float(np.std(arr)),
            "max_voi": float(np.max(arr)),
            "buffer_size": len(arr),
        }
