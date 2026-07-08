"""
Communication Relay Deployment Environment.

Primary evaluation domain: an agent must deploy a finite number of relay nodes
in an underground mine tunnel to maintain end-to-end communication between
a moving rescue team and the surface base station.

The mine topology is revealed progressively as the team advances (fog of war).
Relay placement is irreversible — once deployed, a relay cannot be moved.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..base_env import EnvironmentConfig, IrreversibleActionPOMDP
from ..mine_topology import MineTopologyConfig, MineTopologyGenerator, TunnelSegment
from ..signal_propagation import PropagationConfig, SignalPropagationModel


@dataclass
class RelayDeploymentConfig(EnvironmentConfig):
    """Configuration specific to relay deployment domain."""

    mine_config: MineTopologyConfig = None
    propagation_config: PropagationConfig = None
    team_speed: float = 1.5              # meters per timestep
    visibility_radius: int = 2           # segments ahead the team can see
    uptime_reward_weight: float = 1.0
    deploy_cost_weight: float = 0.1
    quality_penalty_weight: float = 0.5
    terminal_bonus_weight: float = 5.0

    def __post_init__(self):
        if self.mine_config is None:
            self.mine_config = MineTopologyConfig()
        if self.propagation_config is None:
            self.propagation_config = PropagationConfig()


class RelayDeploymentEnv(IrreversibleActionPOMDP):
    """
    Gymnasium environment for sequential relay deployment in underground mines.

    Observation space:
        - signal_readings: SINR at each deployed relay (padded, normalized)
        - team_position: normalized depth fraction
        - team_local_features: rock type, tunnel width, moisture at current position
        - budget_remaining: normalized relay count
        - topology_summary: encoding of discovered topology structure
        - min_sinr: current weakest link in the relay chain
        - time_since_last_deploy: timesteps since last relay was placed

    Action space:
        0: WAIT — continue advancing without deploying
        1: COMMIT — deploy relay at current team position

    Reward:
        + uptime bonus when end-to-end link is maintained
        - cost for each relay deployed
        - penalty proportional to SINR degradation below threshold
        + terminal bonus for maintaining connectivity at episode end
    """

    def __init__(self, config: Optional[RelayDeploymentConfig] = None, **kwargs):
        if config is None:
            config = RelayDeploymentConfig()
        self._domain_config = config
        self._topology_gen = MineTopologyGenerator(config.mine_config)
        self._propagation = SignalPropagationModel(config.propagation_config)

        # Domain state (initialized in _reset_domain)
        self._mine_graph = None
        self._segments = []
        self._path = []  # ordered list of segment indices the team traverses
        self._team_segment_idx = 0
        self._relay_segment_indices = []
        self._discovered_segments = set()
        self._uptime_history = []
        self._sinr_history = []

        super().__init__(config, **kwargs)

    def _build_observation_space(self) -> spaces.Space:
        """Define observation space."""
        max_relays = self._domain_config.initial_budget
        return spaces.Dict({
            "signal_readings": spaces.Box(
                low=-50.0, high=50.0, shape=(max_relays,), dtype=np.float32
            ),
            "team_position": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "local_features": spaces.Box(
                low=0.0, high=1.0, shape=(4,), dtype=np.float32
            ),
            "budget_remaining": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "min_sinr": spaces.Box(
                low=-50.0, high=50.0, shape=(1,), dtype=np.float32
            ),
            "time_since_deploy": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "topology_encoding": spaces.Box(
                low=0.0, high=1.0, shape=(8,), dtype=np.float32
            ),
        })

    def _reset_domain(self, seed: Optional[int] = None) -> None:
        """Generate a new mine and reset team position."""
        self._mine_graph, self._segments = self._topology_gen.generate()

        # Find the longest non-dead-end path from entrance (node 0)
        self._path = self._find_exploration_path()
        self._team_segment_idx = 0
        self._relay_segment_indices = []
        self._discovered_segments = {0}
        self._uptime_history = []
        self._sinr_history = []

    def _find_exploration_path(self) -> List[int]:
        """Find a reasonable exploration path through the mine."""
        if not self._segments:
            return [0]

        # BFS from entrance, following the deepest branch
        graph = self._mine_graph
        visited = {0}
        path = []
        current = 0

        while True:
            neighbors = [
                (n, graph.edges[current, n].get("length", 0))
                for n in graph.successors(current)
                if n not in visited
            ]
            if not neighbors:
                break

            # Follow the longest unvisited branch
            next_node = max(neighbors, key=lambda x: x[1])[0]
            visited.add(next_node)

            # Find the segment index for this edge
            for i, seg in enumerate(self._segments):
                if seg.start_node == current and seg.end_node == next_node:
                    path.append(i)
                    break

            current = next_node

        return path if path else [0]

    def _get_true_state(self) -> dict[str, Any]:
        """Return full environment state (for oracle/debugging)."""
        return {
            "mine_graph": self._mine_graph,
            "all_segments": self._segments,
            "team_segment_idx": self._team_segment_idx,
            "relay_positions": list(self._relay_segment_indices),
            "full_path": self._path,
        }

    def _generate_observation(self) -> dict[str, np.ndarray]:
        """Generate partial observation of current state."""
        max_relays = self._domain_config.initial_budget

        # Signal readings at deployed relays (noisy)
        signal_readings = np.full(max_relays, -50.0, dtype=np.float32)
        if self._relay_segment_indices:
            path_segments = [self._segments[i] for i in self._path[:self._team_segment_idx + 1]]
            if path_segments:
                sinr_values = self._propagation.compute_sinr(
                    path_segments, self._relay_segment_indices, include_fading=True
                )
                for i, sinr in enumerate(sinr_values[:max_relays]):
                    signal_readings[i] = self._propagation.get_noisy_measurement(sinr)

        # Team position (normalized)
        team_pos = np.array(
            [self._team_segment_idx / max(len(self._path), 1)], dtype=np.float32
        )

        # Local geological features at current position
        if self._team_segment_idx < len(self._path):
            current_seg = self._segments[self._path[self._team_segment_idx]]
            local_features = np.array([
                current_seg.rock_type.absorption_coefficient / 2.0,  # normalized
                current_seg.width / 5.0,
                current_seg.moisture_level,
                current_seg.bend_angle / 180.0,
            ], dtype=np.float32)
        else:
            local_features = np.zeros(4, dtype=np.float32)

        # Budget
        budget = np.array([self.budget_fraction], dtype=np.float32)

        # Minimum SINR in chain
        min_sinr = self._get_min_sinr()
        min_sinr_obs = np.array([min_sinr], dtype=np.float32)

        # Time since last deployment
        if self._committed_actions:
            time_since = (self._timestep - self._committed_actions[-1]) / self.config.max_horizon
        else:
            time_since = self._timestep / self.config.max_horizon
        time_since_deploy = np.array([min(time_since, 1.0)], dtype=np.float32)

        # Topology encoding (summary statistics of discovered structure)
        topology_enc = self._encode_discovered_topology()

        return {
            "signal_readings": signal_readings,
            "team_position": team_pos,
            "local_features": local_features,
            "budget_remaining": budget,
            "min_sinr": min_sinr_obs,
            "time_since_deploy": time_since_deploy,
            "topology_encoding": topology_enc,
        }

    def _compute_reward(self, action: int) -> float:
        """Compute step reward."""
        cfg = self._domain_config
        reward = 0.0

        # Uptime reward: +1 if connected
        connected = self._is_chain_connected()
        self._uptime_history.append(connected)
        if connected:
            reward += cfg.uptime_reward_weight

        # Deploy cost
        if action == self.COMMIT:
            reward -= cfg.deploy_cost_weight

        # Quality penalty: penalize low SINR
        min_sinr = self._get_min_sinr()
        self._sinr_history.append(min_sinr)
        if min_sinr < self._propagation.config.sinr_threshold_db:
            deficit = self._propagation.config.sinr_threshold_db - min_sinr
            reward -= cfg.quality_penalty_weight * (deficit / 20.0)

        return reward

    def _compute_terminal_reward(self) -> float:
        """Bonus for maintaining connectivity at episode end."""
        if not self._uptime_history:
            return 0.0
        uptime_ratio = sum(self._uptime_history) / len(self._uptime_history)
        efficiency = uptime_ratio / max(
            1, self._domain_config.initial_budget - self._budget_remaining
        )
        return self._domain_config.terminal_bonus_weight * uptime_ratio

    def _apply_commit(self) -> None:
        """Deploy a relay at the current team position."""
        self._relay_segment_indices.append(self._team_segment_idx)

    def _advance_time(self) -> None:
        """Move team forward along the path."""
        if self._team_segment_idx < len(self._path) - 1:
            self._team_segment_idx += 1
            # Reveal nearby segments (fog of war)
            for i in range(
                self._team_segment_idx,
                min(
                    self._team_segment_idx + self._domain_config.visibility_radius,
                    len(self._path),
                )
            ):
                self._discovered_segments.add(i)

    def _check_terminal(self) -> bool:
        """Episode ends when team reaches end of path or budget exhausted."""
        return self._team_segment_idx >= len(self._path) - 1

    def _get_min_sinr(self) -> float:
        """Get the minimum SINR across all hops in relay chain."""
        if not self._relay_segment_indices and self._team_segment_idx == 0:
            return self._propagation.config.tx_power_dbm - self._propagation.config.noise_floor_dbm

        path_segments = [
            self._segments[self._path[i]]
            for i in range(min(self._team_segment_idx + 1, len(self._path)))
            if i < len(self._path)
        ]
        if not path_segments:
            return 0.0

        sinr_values = self._propagation.compute_sinr(
            path_segments, self._relay_segment_indices, include_fading=False
        )
        return float(np.min(sinr_values)) if len(sinr_values) > 0 else 0.0

    def _is_chain_connected(self) -> bool:
        """Check if end-to-end communication is maintained."""
        min_sinr = self._get_min_sinr()
        return min_sinr >= self._propagation.config.sinr_threshold_db

    def _encode_discovered_topology(self) -> np.ndarray:
        """Encode discovered topology as fixed-size feature vector."""
        n_discovered = len(self._discovered_segments)
        total_segments = len(self._path)

        # Count branches in discovered region
        branch_count = 0
        if self._mine_graph is not None:
            for seg_idx in self._discovered_segments:
                if seg_idx < len(self._path):
                    seg = self._segments[self._path[seg_idx]]
                    if self._mine_graph.out_degree(seg.end_node) > 1:
                        branch_count += 1

        encoding = np.array([
            n_discovered / max(total_segments, 1),            # discovery fraction
            branch_count / max(n_discovered, 1),              # branch density
            len(self._relay_segment_indices) / max(n_discovered, 1),  # relay density
            self._team_segment_idx / max(total_segments, 1),  # progress
            self.budget_fraction,                              # budget remaining
            len(self._relay_segment_indices) / self._domain_config.initial_budget,  # budget used
            1.0 if self._is_chain_connected() else 0.0,       # currently connected
            self._get_min_sinr() / 50.0,                      # normalized min SINR
        ], dtype=np.float32)

        return np.clip(encoding, 0.0, 1.0)

    def get_metrics(self) -> dict[str, float]:
        """Get performance metrics for the episode."""
        uptime = (
            sum(self._uptime_history) / len(self._uptime_history)
            if self._uptime_history else 0.0
        )
        relays_used = self._domain_config.initial_budget - self._budget_remaining
        efficiency = uptime / max(relays_used, 1)
        avg_sinr = np.mean(self._sinr_history) if self._sinr_history else 0.0

        return {
            "communication_uptime": uptime,
            "relays_deployed": relays_used,
            "relay_efficiency": efficiency,
            "average_sinr_db": avg_sinr,
            "min_sinr_db": min(self._sinr_history) if self._sinr_history else 0.0,
            "path_length_segments": len(self._path),
            "final_connected": self._is_chain_connected(),
        }
