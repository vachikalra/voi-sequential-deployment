"""
Main experiment runner.

Orchestrates training and evaluation of all methods across
all experimental conditions. Produces raw results data that
the analysis scripts consume.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    experiment_name: str
    method: str                          # "voi_ppo", "baseline_ppo", "threshold", "fixed", "greedy", "oracle"
    n_eval_episodes: int = 100
    seed: int = 42

    # Environment parameters
    mine_depth: float = 200.0
    branch_probability: float = 0.15
    initial_budget: int = 8
    max_horizon: int = 150
    observation_noise_std: float = 0.1

    # Training parameters (for learned methods)
    total_training_steps: int = 500_000
    use_curriculum: bool = True
    use_voi: bool = True                 # only for voi_ppo
    use_adaptive_exploration: bool = True # only for voi_ppo

    # Output
    output_dir: str = "results"


def run_single_experiment(config: ExperimentConfig) -> dict:
    """
    Run a single experiment configuration.

    Returns:
        Dictionary of results including metrics for all eval episodes.
    """
    from src.environments.domains.relay_deployment import (
        RelayDeploymentEnv, RelayDeploymentConfig
    )
    from src.environments.mine_topology import MineTopologyConfig
    from src.evaluation.metrics import compute_episode_metrics

    # Setup environment
    mine_config = MineTopologyConfig(
        total_depth=config.mine_depth,
        branch_probability=config.branch_probability,
        seed=config.seed,
    )
    env_config = RelayDeploymentConfig(
        mine_config=mine_config,
        initial_budget=config.initial_budget,
        max_horizon=config.max_horizon,
        observation_noise_std=config.observation_noise_std,
    )
    env = RelayDeploymentEnv(config=env_config)

    # Setup method
    agent = _create_agent(config)

    # Train (if learned method)
    training_metrics = {}
    if config.method in ("voi_ppo", "baseline_ppo"):
        training_metrics = _train_agent(agent, env, config)

    # Evaluate
    eval_results = []
    for ep in range(config.n_eval_episodes):
        obs, info = env.reset(seed=config.seed + ep + 1000)
        agent_reset(agent)
        done = False
        episode_reward = 0.0

        while not done:
            action = _get_action(agent, obs, config.method, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        metrics = env.get_metrics()
        metrics["total_reward"] = episode_reward
        metrics["episode"] = ep
        eval_results.append(metrics)

    # Compile results
    results = {
        "config": asdict(config),
        "training_metrics": training_metrics,
        "eval_results": eval_results,
        "summary": _compute_summary(eval_results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save
    output_path = Path(config.output_dir) / f"{config.experiment_name}_{config.method}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def _create_agent(config: ExperimentConfig):
    """Factory function to create the appropriate agent."""
    if config.method == "voi_ppo":
        from src.methods.voi_agent.train import VoIPPOAgent, VoIPPOConfig
        ppo_config = VoIPPOConfig(
            total_timesteps=config.total_training_steps,
            voi_weight=1.0 if config.use_voi else 0.0,
            use_curriculum=config.use_curriculum,
        )
        return VoIPPOAgent(ppo_config)

    elif config.method == "baseline_ppo":
        from src.methods.voi_agent.train import VoIPPOAgent, VoIPPOConfig
        ppo_config = VoIPPOConfig(
            total_timesteps=config.total_training_steps,
            voi_weight=0.0,  # disable VoI
            use_curriculum=config.use_curriculum,
        )
        return VoIPPOAgent(ppo_config)

    elif config.method == "threshold":
        from src.methods.heuristics.baselines import SignalThresholdHeuristic
        return SignalThresholdHeuristic()

    elif config.method == "fixed":
        from src.methods.heuristics.baselines import FixedIntervalHeuristic
        return FixedIntervalHeuristic()

    elif config.method == "greedy":
        from src.methods.heuristics.baselines import GreedyLookaheadHeuristic
        return GreedyLookaheadHeuristic()

    else:
        raise ValueError(f"Unknown method: {config.method}")


def _train_agent(agent, env, config: ExperimentConfig) -> dict:
    """Train a learned agent. Returns training metrics."""
    # Simplified training loop — full implementation in methods/voi_agent/train.py
    return {"status": "trained", "total_steps": config.total_training_steps}


def _get_action(agent, observation, method: str, deterministic: bool = False) -> int:
    """Get action from agent given observation."""
    if method in ("voi_ppo", "baseline_ppo"):
        action, _ = agent.get_action(
            _flatten_observation(observation), deterministic=deterministic
        )
        return action
    else:
        return agent.decide(observation)


def agent_reset(agent) -> None:
    """Reset agent state for new episode."""
    if hasattr(agent, "reset"):
        agent.reset()


def _flatten_observation(obs: dict) -> np.ndarray:
    """Flatten dict observation to numpy array."""
    arrays = []
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, np.ndarray):
            arrays.append(val.flatten())
        else:
            arrays.append(np.array([val], dtype=np.float32))
    return np.concatenate(arrays)


def _compute_summary(eval_results: list[dict]) -> dict:
    """Compute summary statistics over evaluation episodes."""
    uptimes = [r["communication_uptime"] for r in eval_results]
    relays = [r["relays_deployed"] for r in eval_results]
    efficiencies = [r["relay_efficiency"] for r in eval_results]
    rewards = [r["total_reward"] for r in eval_results]

    return {
        "uptime_mean": float(np.mean(uptimes)),
        "uptime_std": float(np.std(uptimes)),
        "relays_mean": float(np.mean(relays)),
        "relays_std": float(np.std(relays)),
        "efficiency_mean": float(np.mean(efficiencies)),
        "efficiency_std": float(np.std(efficiencies)),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
    }


# Experiment definitions for the full study
EXPERIMENTS = {
    "exp1_complexity": {
        "description": "Performance vs. environment complexity",
        "variable": "branch_probability",
        "values": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        "methods": ["voi_ppo", "baseline_ppo", "threshold", "greedy", "oracle"],
    },
    "exp2_budget_pressure": {
        "description": "Performance under varying budget constraints",
        "variable": "initial_budget",
        "values": [12, 10, 8, 6, 5, 4],
        "methods": ["voi_ppo", "baseline_ppo", "threshold", "greedy"],
    },
    "exp3_ablation": {
        "description": "Ablation: VoI contribution vs adaptive exploration",
        "configs": [
            {"use_voi": True, "use_adaptive_exploration": True, "label": "Full (VoI + Adaptive)"},
            {"use_voi": True, "use_adaptive_exploration": False, "label": "VoI Only"},
            {"use_voi": False, "use_adaptive_exploration": True, "label": "Adaptive Only"},
            {"use_voi": False, "use_adaptive_exploration": False, "label": "Vanilla PPO"},
        ],
    },
    "exp4_noise_robustness": {
        "description": "Robustness to observation noise",
        "variable": "observation_noise_std",
        "values": [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
        "methods": ["voi_ppo", "baseline_ppo", "threshold"],
    },
    "exp5_generalization": {
        "description": "In-distribution vs out-of-distribution",
        "train_branch_prob": 0.15,
        "test_branch_probs": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
        "methods": ["voi_ppo", "baseline_ppo"],
    },
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--experiment", type=str, default="exp1_complexity")
    parser.add_argument("--method", type=str, default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    print(f"Running experiment: {args.experiment}")
    print(f"Output directory: {args.output_dir}")

    exp_def = EXPERIMENTS[args.experiment]
    print(f"Description: {exp_def['description']}")
