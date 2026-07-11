"""
Procedural mine topology generator.

Generates realistic underground mine tunnel networks as directed graphs
with physical properties (tunnel width, rock type, bend angles) that
affect RF signal propagation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


class RockType(Enum):
    """Geological classification affecting RF absorption."""
    SANDSTONE = "sandstone"      # Low absorption (~0.5 dB/m)
    LIMESTONE = "limestone"      # Medium absorption (~1.0 dB/m)
    GRANITE = "granite"          # High absorption (~2.0 dB/m)
    SHALE = "shale"              # Variable (~1.5 dB/m)
    COAL = "coal"                # Low absorption (~0.3 dB/m)

    @property
    def absorption_coefficient(self) -> float:
        """RF absorption in dB per meter."""
        coefficients = {
            "sandstone": 0.5,
            "limestone": 1.0,
            "granite": 2.0,
            "shale": 1.5,
            "coal": 0.3,
        }
        return coefficients[self.value]


@dataclass
class TunnelSegment:
    """A single segment of mine tunnel between two junction points."""
    start_node: int
    end_node: int
    length: float                # meters
    width: float                 # meters (affects waveguide propagation)
    rock_type: RockType
    bend_angle: float            # degrees (cumulative bends in segment)
    moisture_level: float        # 0-1, affects signal attenuation
    is_collapsed: bool = False   # impassable


@dataclass
class MineTopologyConfig:
    """Configuration for procedural mine generation."""
    total_depth: float = 500.0          # max depth in meters
    segment_length_mean: float = 15.0   # mean segment length (shorter segments = longer path)
    segment_length_std: float = 5.0     # std of segment length
    branch_probability: float = 0.15    # P(fork) at each junction
    dead_end_ratio: float = 0.3         # fraction of branches that dead-end
    loop_probability: float = 0.05      # P(tunnel loops back to existing node)
    width_range: tuple = (1.8, 5.0)     # tunnel width range (meters)
    bend_frequency: float = 0.05        # expected bends per meter
    bend_angle_range: tuple = (15, 90)  # bend angle range (degrees)
    collapse_probability: float = 0.02  # P(segment is collapsed/impassable)
    geology_correlation: float = 0.7    # spatial correlation of rock type
    seed: Optional[int] = None


class MineTopologyGenerator:
    """
    Procedural generator for realistic underground mine topologies.

    Generates a directed graph where:
    - Nodes represent junction points (forks, dead ends, entry)
    - Edges represent tunnel segments with physical properties
    - The entry node (node 0) is always the mine entrance / Fresh Air Base
    """

    def __init__(self, config: MineTopologyConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._rock_types = list(RockType)

    def generate(self) -> tuple[nx.DiGraph, list[TunnelSegment]]:
        """
        Generate a random mine topology.

        Returns:
            graph: NetworkX DiGraph with node/edge attributes
            segments: List of TunnelSegment objects with physical properties
        """
        graph = nx.DiGraph()
        segments = []
        node_positions = {}

        # Node 0 is the mine entrance
        graph.add_node(0, depth=0.0, position=(0.0, 0.0), is_entrance=True)
        node_positions[0] = (0.0, 0.0)

        frontier = [(0, 0.0, 0.0)]  # (node_id, current_depth, angle)
        next_node_id = 1
        current_rock = self._rng.choice(self._rock_types)
        max_nodes = 100  # cap complexity for generation speed

        while frontier and next_node_id < max_nodes:
            parent_id, depth, heading = frontier.pop(0)

            if depth >= self.config.total_depth:
                continue

            # Generate segment length
            seg_length = max(
                5.0,
                self._rng.normal(
                    self.config.segment_length_mean,
                    self.config.segment_length_std
                )
            )
            seg_length = min(seg_length, self.config.total_depth - depth)

            # Determine branching
            n_branches = 1
            if self._rng.random() < self.config.branch_probability:
                n_branches = 2
                if self._rng.random() < 0.2:
                    n_branches = 3

            for branch_idx in range(n_branches):
                # Compute new heading
                if branch_idx == 0:
                    new_heading = heading + self._rng.uniform(-20, 20)
                else:
                    new_heading = heading + self._rng.uniform(30, 90) * (
                        1 if branch_idx % 2 == 0 else -1
                    )

                # Create new node
                new_depth = depth + seg_length * abs(np.cos(np.radians(new_heading - heading)))
                new_depth = max(new_depth, depth + 1.0)  # always advance depth
                dx = seg_length * np.sin(np.radians(new_heading))
                dy = seg_length * np.cos(np.radians(new_heading))
                parent_pos = node_positions[parent_id]
                new_pos = (parent_pos[0] + dx, parent_pos[1] + dy)

                # Rock type with spatial correlation
                if self._rng.random() > self.config.geology_correlation:
                    current_rock = self._rng.choice(self._rock_types)

                # Segment properties
                tunnel_width = self._rng.uniform(*self.config.width_range)
                n_bends = self._rng.poisson(self.config.bend_frequency * seg_length)
                total_bend_angle = sum(
                    self._rng.uniform(*self.config.bend_angle_range)
                    for _ in range(n_bends)
                )
                moisture = self._rng.beta(2, 5)  # skewed toward dry
                collapsed = self._rng.random() < self.config.collapse_probability

                # Check for dead end
                is_dead_end = (
                    branch_idx > 0
                    and self._rng.random() < self.config.dead_end_ratio
                )

                # Add to graph
                node_id = next_node_id
                next_node_id += 1

                graph.add_node(
                    node_id,
                    depth=new_depth,
                    position=new_pos,
                    is_entrance=False,
                    is_dead_end=is_dead_end,
                    rock_type=current_rock.value,
                )
                node_positions[node_id] = new_pos

                segment = TunnelSegment(
                    start_node=parent_id,
                    end_node=node_id,
                    length=seg_length,
                    width=tunnel_width,
                    rock_type=current_rock,
                    bend_angle=total_bend_angle,
                    moisture_level=moisture,
                    is_collapsed=collapsed,
                )
                segments.append(segment)

                graph.add_edge(
                    parent_id,
                    node_id,
                    length=seg_length,
                    width=tunnel_width,
                    rock_type=current_rock.value,
                    bend_angle=total_bend_angle,
                    moisture=moisture,
                    collapsed=collapsed,
                    weight=seg_length,
                )

                # Add to frontier unless dead end or collapsed
                if not is_dead_end and not collapsed:
                    frontier.append((node_id, new_depth, new_heading))

                # Check for loop connections
                if (
                    self._rng.random() < self.config.loop_probability
                    and len(graph.nodes) > 5
                ):
                    candidates = [
                        n for n in graph.nodes
                        if n != node_id
                        and n != parent_id
                        and not graph.has_edge(node_id, n)
                    ]
                    if candidates:
                        loop_target = self._rng.choice(candidates)
                        target_pos = node_positions[loop_target]
                        loop_length = np.sqrt(
                            (new_pos[0] - target_pos[0]) ** 2
                            + (new_pos[1] - target_pos[1]) ** 2
                        )
                        if loop_length < self.config.segment_length_mean * 3:
                            graph.add_edge(
                                node_id,
                                loop_target,
                                length=loop_length,
                                width=tunnel_width,
                                rock_type=current_rock.value,
                                bend_angle=45.0,
                                moisture=moisture,
                                collapsed=False,
                                weight=loop_length,
                            )

        return graph, segments

    def get_complexity_metrics(self, graph: nx.DiGraph) -> dict:
        """Compute topology complexity metrics for a generated mine."""
        undirected = graph.to_undirected()
        branch_nodes = [n for n in graph.nodes if graph.out_degree(n) > 1]
        dead_ends = [n for n in graph.nodes if graph.out_degree(n) == 0]

        return {
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "n_branch_points": len(branch_nodes),
            "n_dead_ends": len(dead_ends),
            "max_depth": max(
                (graph.nodes[n].get("depth", 0) for n in graph.nodes), default=0
            ),
            "avg_segment_length": np.mean(
                [d["length"] for _, _, d in graph.edges(data=True)]
            ) if graph.number_of_edges() > 0 else 0,
            "connectivity": nx.is_weakly_connected(graph),
            "longest_path": nx.dag_longest_path_length(graph)
            if nx.is_directed_acyclic_graph(graph) else -1,
        }
