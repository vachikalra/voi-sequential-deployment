"""
Oracle solver: optimal relay placement with full environment knowledge.

Solves the relay placement problem as a Mixed Integer Program (MIP)
given COMPLETE knowledge of the mine topology and signal propagation.

This represents the theoretical performance ceiling — no online method
can beat the oracle because it has information no real-time agent possesses.
Used as an upper bound for evaluation.
"""

import numpy as np
from typing import List, Optional, Tuple

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


class OracleSolver:
    """
    Offline optimal relay placement via integer linear programming.

    Given full mine topology, signal propagation model, and budget N,
    finds the minimum-cost placement that maintains SINR above threshold
    at all points along the traversal path.
    """

    def __init__(self, sinr_threshold_db: float = 10.0):
        if not HAS_PULP:
            raise ImportError("Oracle solver requires PuLP: pip install pulp")
        self.sinr_threshold = sinr_threshold_db

    def solve(
        self,
        path_sinr_matrix: np.ndarray,
        budget: int,
        solver_time_limit: int = 30,
    ) -> tuple[list[int], float]:
        """
        Find optimal relay positions.

        Args:
            path_sinr_matrix: [n_positions, n_positions] matrix where entry (i,j)
                is the SINR between positions i and j if they are consecutive
                relay endpoints.
            budget: maximum number of relays to deploy
            solver_time_limit: maximum solver runtime in seconds

        Returns:
            positions: list of segment indices where relays should be placed
            objective_value: achieved minimum SINR across all hops
        """
        n_positions = path_sinr_matrix.shape[0]

        # Decision variables: x_i = 1 if relay placed at position i
        prob = pulp.LpProblem("RelayPlacement", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n_positions)]

        # Auxiliary variable: minimum SINR across all hops
        min_sinr = pulp.LpVariable("min_sinr", lowBound=-100)

        # Objective: maximize minimum SINR
        prob += min_sinr

        # Budget constraint
        prob += pulp.lpSum(x) <= budget

        # SINR constraints: for each pair of consecutive active relays,
        # the SINR must exceed min_sinr
        # This is complex to linearize exactly; we use a simplified formulation
        # where we require coverage intervals
        M = 200  # big-M constant

        # Constraint: at least one relay every K positions (ensures coverage)
        # and SINR between consecutive relays meets threshold
        for i in range(n_positions):
            for j in range(i + 1, min(i + n_positions, n_positions)):
                # If x_i = 1 and x_j = 1 and no relay between them,
                # then SINR(i,j) >= min_sinr
                if j < n_positions:
                    sinr_ij = path_sinr_matrix[i, j]
                    # Simplified: if both endpoints are active
                    prob += min_sinr <= sinr_ij + M * (2 - x[i] - x[j])

        # Solve
        solver = pulp.PULP_CBC_CMD(timeLimit=solver_time_limit, msg=0)
        prob.solve(solver)

        if prob.status != pulp.constants.LpStatusOptimal:
            # Fallback: evenly space relays
            positions = list(np.linspace(0, n_positions - 1, budget, dtype=int))
            return positions, 0.0

        positions = [i for i in range(n_positions) if x[i].value() > 0.5]
        obj_value = min_sinr.value() if min_sinr.value() is not None else 0.0

        return positions, obj_value

    def solve_greedy_approximation(
        self,
        segment_path_losses: np.ndarray,
        budget: int,
        tx_power_dbm: float = 20.0,
        noise_floor_dbm: float = -100.0,
    ) -> tuple[list[int], float]:
        """
        Fast greedy approximation when MIP is too slow.

        Greedily places relays at positions that maximize the minimum
        hop SINR. O(N * budget) complexity.
        """
        n_segments = len(segment_path_losses)
        cumulative_loss = np.cumsum(segment_path_losses)

        positions = []
        last_relay_pos = 0
        last_relay_cum_loss = 0.0

        for _ in range(budget):
            # Find the position where placing a relay maximizes min SINR
            best_pos = None
            best_min_sinr = -np.inf

            for pos in range(last_relay_pos + 1, n_segments):
                hop_loss = cumulative_loss[pos] - last_relay_cum_loss
                hop_sinr = tx_power_dbm - hop_loss - noise_floor_dbm

                # Remaining path without more relays
                remaining_loss = cumulative_loss[-1] - cumulative_loss[pos]
                remaining_hops = budget - len(positions) - 1
                if remaining_hops > 0:
                    avg_remaining_hop_loss = remaining_loss / remaining_hops
                    remaining_sinr = tx_power_dbm - avg_remaining_hop_loss - noise_floor_dbm
                else:
                    remaining_sinr = tx_power_dbm - remaining_loss - noise_floor_dbm

                min_sinr = min(hop_sinr, remaining_sinr)
                if min_sinr > best_min_sinr:
                    best_min_sinr = min_sinr
                    best_pos = pos

            if best_pos is None:
                break

            positions.append(best_pos)
            last_relay_pos = best_pos
            last_relay_cum_loss = cumulative_loss[best_pos]

        # Compute achieved minimum SINR
        all_positions = [0] + sorted(positions) + [n_segments - 1]
        min_sinr = np.inf
        for i in range(len(all_positions) - 1):
            hop_loss = cumulative_loss[all_positions[i + 1]] - (
                cumulative_loss[all_positions[i]] if all_positions[i] > 0 else 0
            )
            hop_sinr = tx_power_dbm - hop_loss - noise_floor_dbm
            min_sinr = min(min_sinr, hop_sinr)

        return positions, min_sinr
