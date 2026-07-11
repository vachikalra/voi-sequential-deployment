"""
VoI-Guided PPO Training Loop.

Implements Proximal Policy Optimization augmented with Value-of-Information
estimation. The training procedure alternates between:
1. Collecting rollouts using the VoI-modified policy
2. Computing advantages (GAE)
3. Updating policy/value networks (PPO clipping)
4. Updating VoI estimator from hindsight targets
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ...networks.graph_encoder import RelayGraphEncoder
from ...networks.belief_module import BeliefModule
from ...networks.heads import VoIAugmentedActorCritic
from .voi_estimator import VoIEstimator, VoIEstimatorConfig


@dataclass
class VoIPPOConfig:
    """Hyperparameters for VoI-guided PPO training."""

    # PPO core
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5
    max_grad_norm: float = 0.5

    # VoI specific
    voi_weight: float = 1.0
    voi_loss_coefficient: float = 0.25
    voi_update_frequency: int = 5  # update VoI head every N policy updates

    # Training
    n_steps_per_rollout: int = 2048
    n_epochs_per_update: int = 10
    batch_size: int = 64
    total_timesteps: int = 1_000_000

    # Architecture
    observation_dim: int = 24
    graph_embedding_dim: int = 64
    belief_hidden_dim: int = 128
    head_hidden_dim: int = 64

    # Curriculum
    use_curriculum: bool = True
    curriculum_stages: int = 4

    # Logging
    log_interval: int = 10
    save_interval: int = 100
    eval_episodes: int = 20


class VoIPPOAgent:
    """
    Value-of-Information guided PPO agent.

    Architecture:
        Observation → [Feature Encoder] → [Graph Encoder (GAT)] →
        [Belief Module (GRU)] → [Policy + Value + VoI Heads]

    The VoI head estimates how much waiting would improve decision quality.
    This estimate modifies the policy's action selection, creating a
    conservative bias that prevents premature commitment of finite resources.
    """

    def __init__(self, config: VoIPPOConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)

        # Build networks
        self.observation_encoder = nn.Sequential(
            nn.Linear(config.observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.belief_hidden_dim),
            nn.ReLU(),
        ).to(self.device)

        self.belief_module = BeliefModule(
            input_dim=config.belief_hidden_dim,
            hidden_dim=config.belief_hidden_dim,
        ).to(self.device)

        self.actor_critic = VoIAugmentedActorCritic(
            belief_dim=config.belief_hidden_dim,
            hidden_dim=config.head_hidden_dim,
            voi_weight=config.voi_weight,
        ).to(self.device)

        # Optimizers
        all_params = (
            list(self.observation_encoder.parameters())
            + list(self.belief_module.parameters())
            + list(self.actor_critic.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=config.learning_rate)

        # VoI estimator (uses hindsight training)
        self.voi_estimator = VoIEstimator(
            voi_network=self.actor_critic.voi_head,
            value_network=self.actor_critic.value_head,
            config=VoIEstimatorConfig(),
        )

        # Training state
        self._hidden_state: Optional[torch.Tensor] = None
        self._episode_beliefs: list[torch.Tensor] = []
        self._episode_rewards: list[float] = []
        self._update_count = 0

    def reset(self) -> None:
        """Reset hidden state for new episode."""
        self._hidden_state = None
        self._episode_beliefs = []
        self._episode_rewards = []

    def get_action(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> tuple[int, dict]:
        """
        Select action given observation.

        Returns:
            action: 0 (WAIT) or 1 (COMMIT)
            info: dict with log_prob, value, voi estimates
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

            # Encode observation
            obs_embedding = self.observation_encoder(obs_tensor)

            # Update belief state
            belief, self._hidden_state = self.belief_module(
                obs_embedding, self._hidden_state
            )

            # Store belief for VoI training
            self._episode_beliefs.append(belief.squeeze(0))

            # Get action with VoI guidance
            action, log_prob, value = self.actor_critic.get_action(
                belief, deterministic=deterministic
            )

            # Get VoI estimate for logging
            voi = self.actor_critic.voi_head(belief)

        return action.item(), {
            "log_prob": log_prob.item(),
            "value": value.item(),
            "voi": voi.item(),
        }

    def store_reward(self, reward: float) -> None:
        """Store reward for VoI hindsight computation."""
        self._episode_rewards.append(reward)

    def end_episode(self) -> None:
        """Process completed episode for VoI training."""
        if len(self._episode_beliefs) > 1:
            belief_seq = torch.stack(self._episode_beliefs)
            reward_seq = torch.tensor(self._episode_rewards, dtype=torch.float32)
            self.voi_estimator.store_episode(belief_seq, reward_seq, self.config.gamma)

    def update(self, rollout_buffer: dict) -> dict[str, float]:
        """
        PPO update step.

        Args:
            rollout_buffer: dict with keys:
                observations, actions, rewards, dones,
                log_probs, values, advantages, returns

        Returns:
            Dictionary of training metrics
        """
        observations = torch.FloatTensor(rollout_buffer["observations"]).to(self.device)
        actions = torch.LongTensor(rollout_buffer["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_buffer["log_probs"]).to(self.device)
        advantages = torch.FloatTensor(rollout_buffer["advantages"]).to(self.device)
        returns = torch.FloatTensor(rollout_buffer["returns"]).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_voi_loss = 0.0

        for _ in range(self.config.n_epochs_per_update):
            # Random mini-batches
            indices = torch.randperm(len(observations))
            for start in range(0, len(observations), self.config.batch_size):
                batch_idx = indices[start:start + self.config.batch_size]

                batch_obs = observations[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Forward pass (simplified: no recurrence in update for efficiency)
                obs_emb = self.observation_encoder(batch_obs)
                # Use obs_emb directly as belief proxy during batch update
                belief = obs_emb  # TODO: implement proper recurrent update

                log_probs, values, entropy, voi = (
                    self.actor_critic.evaluate_actions(belief, batch_actions)
                )

                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = (
                    policy_loss
                    + self.config.value_loss_coefficient * value_loss
                    + self.config.entropy_coefficient * entropy_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.observation_encoder.parameters())
                    + list(self.belief_module.parameters())
                    + list(self.actor_critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        # Update VoI estimator periodically
        self._update_count += 1
        voi_loss = None
        if self._update_count % self.config.voi_update_frequency == 0:
            voi_loss = self.voi_estimator.update()

        n_batches = max(1, self.config.n_epochs_per_update * (len(observations) // self.config.batch_size))
        return {
            "policy_loss": total_policy_loss / n_batches,
            "value_loss": total_value_loss / n_batches,
            "entropy": total_entropy / n_batches,
            "voi_loss": voi_loss if voi_loss is not None else 0.0,
            "voi_stats": self.voi_estimator.get_voi_stats(),
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "observation_encoder": self.observation_encoder.state_dict(),
            "belief_module": self.belief_module.state_dict(),
            "actor_critic": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "update_count": self._update_count,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.observation_encoder.load_state_dict(checkpoint["observation_encoder"])
        self.belief_module.load_state_dict(checkpoint["belief_module"])
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._update_count = checkpoint["update_count"]
