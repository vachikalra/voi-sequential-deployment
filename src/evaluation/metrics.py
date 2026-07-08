"""
Evaluation metrics for relay deployment experiments.

Provides standardized computation of all dependent variables used
in the research, with proper statistical treatment.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class EpisodeMetrics:
    """Metrics from a single evaluation episode."""

    communication_uptime: float        # fraction of timesteps with connectivity [0, 1]
    relays_deployed: int               # total relays used
    relay_efficiency: float            # uptime / relays_deployed
    mean_sinr_db: float                # average SINR across episode
    min_sinr_db: float                 # worst-case SINR
    path_coverage_fraction: float      # fraction of path with acceptable signal
    first_dropout_timestep: Optional[int]  # when connectivity first dropped (None = never)
    total_reward: float                # cumulative reward
    episode_length: int                # total timesteps


@dataclass
class ExperimentResults:
    """Aggregated results across multiple evaluation episodes with statistics."""

    method_name: str
    n_episodes: int

    # Primary metrics (mean ± CI)
    uptime_mean: float
    uptime_ci_lower: float
    uptime_ci_upper: float

    efficiency_mean: float
    efficiency_ci_lower: float
    efficiency_ci_upper: float

    relays_used_mean: float
    relays_used_std: float

    reward_mean: float
    reward_std: float

    # Secondary metrics
    mean_sinr_mean: float
    min_sinr_mean: float
    first_dropout_mean: Optional[float]


def compute_episode_metrics(
    uptime_history: List[bool],
    sinr_history: List[float],
    relays_deployed: int,
    total_reward: float,
    sinr_threshold: float = 10.0,
) -> EpisodeMetrics:
    """Compute all metrics for a single completed episode."""

    uptime = sum(uptime_history) / max(len(uptime_history), 1)
    efficiency = uptime / max(relays_deployed, 1)
    mean_sinr = np.mean(sinr_history) if sinr_history else 0.0
    min_sinr = min(sinr_history) if sinr_history else 0.0
    coverage = sum(s >= sinr_threshold for s in sinr_history) / max(len(sinr_history), 1)

    first_dropout = None
    for t, connected in enumerate(uptime_history):
        if not connected:
            first_dropout = t
            break

    return EpisodeMetrics(
        communication_uptime=uptime,
        relays_deployed=relays_deployed,
        relay_efficiency=efficiency,
        mean_sinr_db=mean_sinr,
        min_sinr_db=min_sinr,
        path_coverage_fraction=coverage,
        first_dropout_timestep=first_dropout,
        total_reward=total_reward,
        episode_length=len(uptime_history),
    )


def aggregate_results(
    method_name: str,
    episodes: List[EpisodeMetrics],
    confidence_level: float = 0.95,
) -> ExperimentResults:
    """
    Aggregate episode metrics with confidence intervals.

    Uses bootstrapped confidence intervals for robust estimation.
    """
    from scipy import stats

    n = len(episodes)
    if n == 0:
        raise ValueError("No episodes to aggregate")

    uptimes = [e.communication_uptime for e in episodes]
    efficiencies = [e.relay_efficiency for e in episodes]
    relays = [e.relays_deployed for e in episodes]
    rewards = [e.total_reward for e in episodes]
    sinrs = [e.mean_sinr_db for e in episodes]
    min_sinrs = [e.min_sinr_db for e in episodes]
    dropouts = [e.first_dropout_timestep for e in episodes if e.first_dropout_timestep is not None]

    # Confidence intervals using t-distribution
    def ci(data):
        mean = np.mean(data)
        if len(data) < 2:
            return mean, mean, mean
        se = stats.sem(data)
        t_crit = stats.t.ppf((1 + confidence_level) / 2, df=len(data) - 1)
        return mean, mean - t_crit * se, mean + t_crit * se

    up_mean, up_lo, up_hi = ci(uptimes)
    eff_mean, eff_lo, eff_hi = ci(efficiencies)

    return ExperimentResults(
        method_name=method_name,
        n_episodes=n,
        uptime_mean=up_mean,
        uptime_ci_lower=up_lo,
        uptime_ci_upper=up_hi,
        efficiency_mean=eff_mean,
        efficiency_ci_lower=eff_lo,
        efficiency_ci_upper=eff_hi,
        relays_used_mean=np.mean(relays),
        relays_used_std=np.std(relays),
        reward_mean=np.mean(rewards),
        reward_std=np.std(rewards),
        mean_sinr_mean=np.mean(sinrs),
        min_sinr_mean=np.mean(min_sinrs),
        first_dropout_mean=np.mean(dropouts) if dropouts else None,
    )


def statistical_significance_test(
    results_a: List[EpisodeMetrics],
    results_b: List[EpisodeMetrics],
    metric: str = "communication_uptime",
    alpha: float = 0.05,
) -> dict:
    """
    Test whether method A significantly outperforms method B.

    Uses Welch's t-test (does not assume equal variances) and
    reports effect size (Cohen's d).
    """
    from scipy import stats

    values_a = [getattr(e, metric) for e in results_a]
    values_b = [getattr(e, metric) for e in results_b]

    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (np.std(values_a) ** 2 + np.std(values_b) ** 2) / 2
    )
    cohens_d = (np.mean(values_a) - np.mean(values_b)) / max(pooled_std, 1e-8)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "cohens_d": cohens_d,
        "effect_size": (
            "large" if abs(cohens_d) > 0.8
            else "medium" if abs(cohens_d) > 0.5
            else "small" if abs(cohens_d) > 0.2
            else "negligible"
        ),
        "mean_a": np.mean(values_a),
        "mean_b": np.mean(values_b),
        "a_better": np.mean(values_a) > np.mean(values_b),
    }
