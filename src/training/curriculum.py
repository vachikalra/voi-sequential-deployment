"""
Curriculum Learning Scheduler.

Controls the progression of environment difficulty during training.
Starts with simple environments (short tunnels, generous budgets)
and gradually increases complexity as the agent improves.

The curriculum is critical for irreversible-action problems because:
- Random exploration in hard environments wastes all resources immediately
- Reward signal is too sparse in complex mines for the agent to learn
- Gradual difficulty increase provides a learning gradient at every stage
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..environments.mine_topology import MineTopologyConfig


@dataclass
class CurriculumStage:
    """Definition of a single curriculum difficulty stage."""

    name: str
    mine_depth: float
    branch_probability: float
    initial_budget: int
    max_horizon: int
    observation_noise_std: float
    promotion_threshold: float  # mean reward needed to advance


DEFAULT_CURRICULUM = [
    CurriculumStage(
        name="Stage 1: Linear Tunnels",
        mine_depth=80.0,
        branch_probability=0.0,
        initial_budget=10,
        max_horizon=60,
        observation_noise_std=0.05,
        promotion_threshold=0.7,
    ),
    CurriculumStage(
        name="Stage 2: Simple Branching",
        mine_depth=150.0,
        branch_probability=0.08,
        initial_budget=8,
        max_horizon=100,
        observation_noise_std=0.08,
        promotion_threshold=0.6,
    ),
    CurriculumStage(
        name="Stage 3: Complex Topology",
        mine_depth=250.0,
        branch_probability=0.15,
        initial_budget=7,
        max_horizon=150,
        observation_noise_std=0.1,
        promotion_threshold=0.5,
    ),
    CurriculumStage(
        name="Stage 4: Adversarial Conditions",
        mine_depth=400.0,
        branch_probability=0.25,
        initial_budget=6,
        max_horizon=200,
        observation_noise_std=0.15,
        promotion_threshold=0.0,  # final stage, no promotion
    ),
]


class CurriculumScheduler:
    """
    Manages progression through curriculum stages based on agent performance.

    Tracks rolling mean reward and promotes the agent to harder environments
    once performance stabilizes at or above the promotion threshold.
    """

    def __init__(
        self,
        stages: Optional[list[CurriculumStage]] = None,
        window_size: int = 100,
        min_episodes_per_stage: int = 500,
    ):
        self.stages = stages or DEFAULT_CURRICULUM
        self.window_size = window_size
        self.min_episodes_per_stage = min_episodes_per_stage

        self._current_stage_idx = 0
        self._reward_history: list[float] = []
        self._episodes_in_stage = 0
        self._stage_history: list[tuple[int, int]] = []  # (stage_idx, episode_count)

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self._current_stage_idx]

    @property
    def current_stage_idx(self) -> int:
        return self._current_stage_idx

    @property
    def is_final_stage(self) -> bool:
        return self._current_stage_idx >= len(self.stages) - 1

    def get_mine_config(self) -> MineTopologyConfig:
        """Get mine topology config for current curriculum stage."""
        stage = self.current_stage
        return MineTopologyConfig(
            total_depth=stage.mine_depth,
            branch_probability=stage.branch_probability,
        )

    def report_episode_reward(self, reward: float) -> bool:
        """
        Report completed episode reward. Returns True if stage was promoted.

        Args:
            reward: normalized episode reward (0-1 range ideally)

        Returns:
            promoted: True if agent advanced to next stage
        """
        self._reward_history.append(reward)
        self._episodes_in_stage += 1

        if self.is_final_stage:
            return False

        if self._episodes_in_stage < self.min_episodes_per_stage:
            return False

        # Check promotion criterion
        recent_rewards = self._reward_history[-self.window_size:]
        mean_reward = np.mean(recent_rewards)

        if mean_reward >= self.current_stage.promotion_threshold:
            self._promote()
            return True

        return False

    def _promote(self) -> None:
        """Advance to next curriculum stage."""
        self._stage_history.append(
            (self._current_stage_idx, self._episodes_in_stage)
        )
        self._current_stage_idx = min(
            self._current_stage_idx + 1, len(self.stages) - 1
        )
        self._episodes_in_stage = 0
        self._reward_history = []

    def get_stats(self) -> dict:
        """Get curriculum progress statistics."""
        return {
            "current_stage": self._current_stage_idx,
            "stage_name": self.current_stage.name,
            "episodes_in_stage": self._episodes_in_stage,
            "mean_reward_window": (
                np.mean(self._reward_history[-self.window_size:])
                if self._reward_history else 0.0
            ),
            "promotion_threshold": self.current_stage.promotion_threshold,
            "total_stages": len(self.stages),
            "is_final": self.is_final_stage,
        }
