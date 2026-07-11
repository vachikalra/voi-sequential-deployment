"""
Model-Based RL Baseline: World Model + Planning.

Implements a lightweight world model that learns environment dynamics,
then uses tree search (Monte Carlo Tree Search variant) to plan deployment decisions.

This provides a comparison point that represents the "planning" approach to
resource-constrained decision-making, contrasting with the VoI-augmented
model-free approach.

Architecture:
    - Transition model: predicts next state given (state, action)
    - Reward model: predicts immediate reward
    - Planning: MCTS-style rollout using the learned model
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class WorldModelConfig:
    """Configuration for the world model baseline."""
    obs_dim: int = 26
    hidden_dim: int = 128
    latent_dim: int = 64
    ensemble_size: int = 3
    planning_horizon: int = 5
    n_simulations: int = 50
    learning_rate: float = 1e-3
    replay_buffer_size: int = 50000
    batch_size: int = 128
    model_train_freq: int = 250
    min_buffer_size: int = 1000


class TransitionModel(nn.Module):
    """Predicts next observation given current observation and action."""

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.obs_dim = obs_dim

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action.unsqueeze(-1).float()], dim=-1)
        delta = self.net(x)
        return obs + delta  # predict residual


class RewardModel(nn.Module):
    """Predicts reward given observation and action."""

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action.unsqueeze(-1).float()], dim=-1)
        return self.net(x).squeeze(-1)


class DoneModel(nn.Module):
    """Predicts episode termination."""

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class EnsembleWorldModel(nn.Module):
    """Ensemble of world models for uncertainty estimation."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.transition_models = nn.ModuleList([
            TransitionModel(config.obs_dim, config.hidden_dim)
            for _ in range(config.ensemble_size)
        ])
        self.reward_models = nn.ModuleList([
            RewardModel(config.obs_dim, config.hidden_dim)
            for _ in range(config.ensemble_size)
        ])
        self.done_model = DoneModel(config.obs_dim, config.hidden_dim)

    def predict(
        self, obs: torch.Tensor, action: torch.Tensor, model_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next state, reward, and done probability."""
        if model_idx is not None:
            next_obs = self.transition_models[model_idx](obs, action)
            reward = self.reward_models[model_idx](obs, action)
        else:
            # Average across ensemble
            next_obs = torch.stack([m(obs, action) for m in self.transition_models]).mean(0)
            reward = torch.stack([m(obs, action) for m in self.reward_models]).mean(0)

        done_prob = self.done_model(next_obs)
        return next_obs, reward, done_prob

    def get_uncertainty(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute prediction disagreement across ensemble (epistemic uncertainty)."""
        predictions = torch.stack([m(obs, action) for m in self.transition_models])
        return predictions.std(dim=0).mean(dim=-1)


class ReplayBuffer:
    """Simple replay buffer for model training."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.idx = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.dones[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[indices]),
            torch.LongTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_obs[indices]),
            torch.FloatTensor(self.dones[indices]),
        )


class ModelBasedAgent:
    """
    Model-based RL agent using learned world model + tree search planning.
    
    Decision loop:
    1. Maintain replay buffer of real experiences
    2. Periodically train world model on buffer
    3. At each decision point, simulate future trajectories using model
    4. Choose action that maximizes expected cumulative reward over planning horizon
    """

    def __init__(self, config: WorldModelConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.world_model = EnsembleWorldModel(config).to(device)
        self.buffer = ReplayBuffer(config.replay_buffer_size, config.obs_dim)
        self.optimizer = optim.Adam(self.world_model.parameters(), lr=config.learning_rate)
        self.total_steps = 0
        self.model_trained = False

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, dict]:
        """Select action using model-based planning."""
        if self.buffer.size < self.config.min_buffer_size or not self.model_trained:
            # Random exploration until model has enough data
            action = np.random.randint(2)
            return action, {"value": 0.0, "log_prob": np.log(0.5), "voi": 0.0}

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Plan using MCTS-style rollouts
        action_values = self._plan(obs_tensor)

        if deterministic:
            action = int(action_values.argmax())
        else:
            # Softmax sampling
            probs = F.softmax(action_values / 0.5, dim=0)
            action = int(torch.multinomial(probs, 1).item())

        value = float(action_values.max())
        log_prob = float(torch.log(F.softmax(action_values, dim=0)[action]))

        return action, {"value": value, "log_prob": log_prob, "voi": 0.0}

    def _plan(self, obs: torch.Tensor) -> torch.Tensor:
        """Simulate trajectories for each action and return expected values."""
        action_values = torch.zeros(2)

        with torch.no_grad():
            for action in range(2):
                total_reward = 0.0
                for sim in range(self.config.n_simulations):
                    reward = self._simulate_trajectory(obs, action)
                    total_reward += reward
                action_values[action] = total_reward / self.config.n_simulations

        return action_values

    def _simulate_trajectory(self, start_obs: torch.Tensor, first_action: int) -> float:
        """Simulate a single trajectory using the world model."""
        obs = start_obs.clone()
        action_tensor = torch.LongTensor([first_action]).to(self.device)
        cumulative_reward = 0.0
        discount = 0.99

        for step in range(self.config.planning_horizon):
            if step == 0:
                action = action_tensor
            else:
                # Use ensemble uncertainty to guide exploration in simulation
                uncertainty_0 = self.world_model.get_uncertainty(
                    obs, torch.LongTensor([0]).to(self.device)
                )
                uncertainty_1 = self.world_model.get_uncertainty(
                    obs, torch.LongTensor([1]).to(self.device)
                )
                # Choose action with higher uncertainty (optimistic exploration)
                action = torch.LongTensor([1 if uncertainty_1 > uncertainty_0 else 0]).to(self.device)

            # Random model from ensemble for trajectory diversity
            model_idx = np.random.randint(self.config.ensemble_size)
            next_obs, reward, done_prob = self.world_model.predict(obs, action, model_idx)

            cumulative_reward += (discount ** step) * reward.item()
            obs = next_obs

            if done_prob.item() > 0.5:
                break

        return cumulative_reward

    def store_transition(self, obs, action, reward, next_obs, done):
        """Store a transition in the replay buffer."""
        self.buffer.add(obs, action, reward, next_obs, done)
        self.total_steps += 1

        if (self.total_steps % self.config.model_train_freq == 0
                and self.buffer.size >= self.config.min_buffer_size):
            self._train_model()

    def _train_model(self, n_updates: int = 20):
        """Train the world model on replay buffer data."""
        self.world_model.train()

        for _ in range(n_updates):
            obs, actions, rewards, next_obs, dones = self.buffer.sample(self.config.batch_size)
            obs = obs.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_obs = next_obs.to(self.device)
            dones = dones.to(self.device)

            total_loss = 0.0

            for i in range(self.config.ensemble_size):
                # Transition loss
                pred_next = self.transition_models[i](obs, actions)
                trans_loss = F.mse_loss(pred_next, next_obs)

                # Reward loss
                pred_reward = self.reward_models[i](obs, actions)
                reward_loss = F.mse_loss(pred_reward, rewards)

                total_loss += trans_loss + reward_loss

            # Done loss
            pred_done = self.world_model.done_model(next_obs)
            done_loss = F.binary_cross_entropy(pred_done, dones)
            total_loss += done_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
            self.optimizer.step()

        self.model_trained = True
        self.world_model.eval()

    @property
    def transition_models(self):
        return self.world_model.transition_models

    @property
    def reward_models(self):
        return self.world_model.reward_models

    def reset(self):
        """Reset episode-level state (nothing needed for model-based)."""
        pass

    def save(self, path: str):
        torch.save({
            "world_model": self.world_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]
        self.model_trained = True
