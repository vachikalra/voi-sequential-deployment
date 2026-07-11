"""
RF signal propagation model for underground tunnel environments.

Implements a physically-motivated model based on:
- Log-distance path loss with underground-specific exponent
- Bend/corner attenuation
- Material-specific absorption
- Waveguide effects in narrow tunnels
- Stochastic fading (Rayleigh + log-normal shadowing)

References:
- Sun & Akyildiz, "Channel modeling and analysis for wireless networks
  in underground mines", IEEE Trans. Commun., 2010
- Forooshani et al., "A survey of wireless communications and propagation
  modeling in underground mines", IEEE Commun. Surveys, 2013
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .mine_topology import RockType, TunnelSegment


@dataclass
class PropagationConfig:
    """Configuration for the RF propagation model."""

    frequency_mhz: float = 900.0         # Operating frequency
    tx_power_dbm: float = 33.0           # Transmit power (~2W)
    noise_floor_dbm: float = -100.0      # Receiver noise floor
    sinr_threshold_db: float = 5.0       # Minimum acceptable SINR
    path_loss_exponent: float = 2.2      # Moderate underground PLE
    reference_distance_m: float = 1.0    # Reference distance for path loss model
    reference_loss_db: float = 20.0      # Path loss at reference distance
    bend_loss_db_per_90deg: float = 3.0  # Signal loss per 90-degree bend
    waveguide_cutoff_width_m: float = 1.0  # Below this, waveguide attenuation dominates
    shadowing_std_db: float = 2.0        # Log-normal shadowing standard deviation
    measurement_noise_std_db: float = 2.0  # Observation noise on SINR measurements
    relay_gain_db: float = 20.0          # Signal boost provided by a relay node


class SignalPropagationModel:
    """
    Computes received signal strength and SINR between nodes in a mine tunnel.

    The model captures:
    1. Distance-dependent path loss (log-distance model)
    2. Material absorption (rock-type dependent)
    3. Bend/corner losses (geometry-dependent)
    4. Waveguide cutoff effects (width-dependent)
    5. Stochastic fading (for realistic variation)
    """

    def __init__(self, config: PropagationConfig, seed: Optional[int] = None):
        self.config = config
        self._rng = np.random.default_rng(seed)

    def compute_path_loss(self, segment: TunnelSegment) -> float:
        """
        Compute total path loss over a tunnel segment in dB.

        PL_total = PL_distance + PL_bends + PL_absorption + PL_waveguide
        """
        pl_distance = self._distance_path_loss(segment.length)
        pl_bends = self._bend_attenuation(segment.bend_angle)
        pl_absorption = self._material_absorption(
            segment.length, segment.rock_type, segment.moisture_level
        )
        pl_waveguide = self._waveguide_attenuation(segment.width, segment.length)

        return pl_distance + pl_bends + pl_absorption + pl_waveguide

    def compute_sinr(
        self,
        segments: list[TunnelSegment],
        relay_positions: list[int],
        include_fading: bool = True,
    ) -> np.ndarray:
        """
        Compute SINR at each link in a relay chain.

        Args:
            segments: ordered list of tunnel segments from source to destination
            relay_positions: indices of segments where relays are placed
            include_fading: whether to add stochastic fading

        Returns:
            Array of SINR values (dB) at each relay hop
        """
        if not relay_positions:
            total_pl = sum(self.compute_path_loss(s) for s in segments)
            sinr = self.config.tx_power_dbm - total_pl - self.config.noise_floor_dbm
            if include_fading:
                sinr += self._rng.normal(0, self.config.shadowing_std_db)
            return np.array([sinr])

        # Compute SINR at each hop between consecutive relays
        positions = [0] + sorted(relay_positions) + [len(segments)]
        sinr_per_hop = []

        for i in range(len(positions) - 1):
            hop_segments = segments[positions[i]:positions[i + 1]]
            if not hop_segments:
                sinr_per_hop.append(float("inf"))
                continue

            hop_pl = sum(self.compute_path_loss(s) for s in hop_segments)
            effective_tx = self.config.tx_power_dbm
            if i > 0:
                effective_tx += self.config.relay_gain_db

            sinr = effective_tx - hop_pl - self.config.noise_floor_dbm

            if include_fading:
                sinr += self._rng.normal(0, self.config.shadowing_std_db)

            sinr_per_hop.append(sinr)

        return np.array(sinr_per_hop)

    def compute_link_quality(self, sinr_db: float) -> float:
        """
        Map SINR to a [0, 1] link quality metric.
        Uses a sigmoid function centered at the SINR threshold.
        """
        x = (sinr_db - self.config.sinr_threshold_db) / 5.0
        return 1.0 / (1.0 + np.exp(-x))

    def is_connected(self, sinr_array: np.ndarray) -> bool:
        """Check if all hops in a relay chain meet minimum SINR."""
        return bool(np.all(sinr_array >= self.config.sinr_threshold_db))

    def get_noisy_measurement(self, true_sinr: float) -> float:
        """Simulate a noisy SINR measurement (what the agent observes)."""
        noise = self._rng.normal(0, self.config.measurement_noise_std_db)
        return true_sinr + noise

    def _distance_path_loss(self, distance_m: float) -> float:
        """Log-distance path loss model: PL(d) = PL_0 + 10*n*log10(d/d_0)"""
        if distance_m <= self.config.reference_distance_m:
            return self.config.reference_loss_db
        return (
            self.config.reference_loss_db
            + 10 * self.config.path_loss_exponent
            * np.log10(distance_m / self.config.reference_distance_m)
        )

    def _bend_attenuation(self, total_bend_degrees: float) -> float:
        """Attenuation due to tunnel bends. Proportional to total bend angle."""
        return self.config.bend_loss_db_per_90deg * (total_bend_degrees / 90.0)

    def _material_absorption(
        self, length_m: float, rock_type: RockType, moisture: float
    ) -> float:
        """
        Frequency-dependent absorption by tunnel wall material.
        Moisture increases absorption significantly.
        """
        base_absorption = rock_type.absorption_coefficient * length_m * 0.3
        moisture_factor = 1.0 + 0.5 * moisture
        return base_absorption * moisture_factor

    def _waveguide_attenuation(self, width_m: float, length_m: float) -> float:
        """
        Below cutoff width, tunnel acts as waveguide with exponential attenuation.
        Above cutoff, this contribution is negligible.
        """
        if width_m >= self.config.waveguide_cutoff_width_m:
            return 0.0
        ratio = self.config.waveguide_cutoff_width_m / width_m
        return 5.0 * (ratio - 1.0) * (length_m / 10.0)

    def estimate_max_range(self, rock_type: RockType, width: float = 3.0) -> float:
        """Estimate maximum communication range for given conditions (no relays)."""
        available_budget = (
            self.config.tx_power_dbm
            - self.config.noise_floor_dbm
            - self.config.sinr_threshold_db
            - self.config.reference_loss_db
        )
        # Solve: available = 10*n*log10(d) + absorption*d
        # Approximate with iterative method
        for d in range(10, 500, 5):
            pl = self._distance_path_loss(d)
            absorption = rock_type.absorption_coefficient * d
            if pl + absorption > available_budget:
                return float(d - 5)
        return 500.0
