"""
Run comprehensive experiments comparing VoI-PPO against baselines.

Generates results for:
1. Main comparison (all methods, default difficulty)
2. Budget pressure sweep (varying budget/path ratio)
3. Ablation (VoI only, adaptive exploration only, full)
4. Noise robustness (varying observation noise)
"""

import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environments.domains.relay_deployment import RelayDeploymentEnv, RelayDeploymentConfig
from src.environments.mine_topology import MineTopologyConfig
from src.methods.voi_agent.train import VoIPPOAgent, VoIPPOConfig
from src.methods.heuristics.baselines import (
    FixedIntervalHeuristic,
    SignalThresholdHeuristic,
    GreedyLookaheadHeuristic,
)
from train_agent import flatten_observation


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate_agent(agent, env, method_name: str, n_episodes: int = 100, seed: int = 0):
    """Run evaluation episodes and return metrics."""
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        if hasattr(agent, "reset"):
            agent.reset()
        done = False
        ep_reward = 0.0
        deploy_times = []
        step = 0

        while not done:
            if method_name in ("voi_ppo", "baseline_ppo"):
                flat_obs = flatten_observation(obs)
                action, action_info = agent.get_action(flat_obs, deterministic=True)
            else:
                action = agent.decide(obs)

            if action == 1:
                deploy_times.append(step)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            step += 1

        metrics = env.get_metrics()
        metrics["total_reward"] = ep_reward
        metrics["deploy_times"] = deploy_times
        metrics["episode_length"] = step
        results.append(metrics)

    return results


def make_env(budget=5, depth=500.0, branch_prob=0.15, noise=0.1, seed=None):
    """Create evaluation environment."""
    mine_config = MineTopologyConfig(
        total_depth=depth,
        branch_probability=branch_prob,
        seed=seed,
    )
    env_config = RelayDeploymentConfig(
        mine_config=mine_config,
        initial_budget=budget,
        max_horizon=50,
        observation_noise_std=noise,
    )
    return RelayDeploymentEnv(config=env_config)


def load_trained_agent(method: str, model_path: str, obs_dim: int):
    """Load a trained agent."""
    use_voi = method == "voi_ppo"
    config = VoIPPOConfig(
        observation_dim=obs_dim,
        voi_weight=1.0 if use_voi else 0.0,
    )
    agent = VoIPPOAgent(config)
    try:
        agent.load(model_path)
        print(f"  Loaded: {model_path}")
    except FileNotFoundError:
        print(f"  WARNING: {model_path} not found, using untrained agent")
    return agent


def experiment_1_main_comparison():
    """Compare all methods on default environment."""
    print("\n" + "=" * 60)
    print("Experiment 1: Main Method Comparison")
    print("=" * 60)

    env = make_env(seed=1000)
    test_obs, _ = env.reset(seed=1000)
    obs_dim = len(flatten_observation(test_obs))

    results = {}

    # Learned methods
    voi_path = Path("checkpoints/voi_ppo/best_model.pt")
    bl_path = Path("checkpoints/baseline_ppo/best_model.pt")

    if voi_path.exists():
        agent = load_trained_agent("voi_ppo", str(voi_path), obs_dim)
        results["voi_ppo"] = evaluate_agent(agent, env, "voi_ppo", n_episodes=200)
    else:
        print("  Skipping VoI-PPO (no trained model)")

    if bl_path.exists():
        agent = load_trained_agent("baseline_ppo", str(bl_path), obs_dim)
        results["baseline_ppo"] = evaluate_agent(agent, env, "baseline_ppo", n_episodes=200)
    else:
        print("  Skipping Baseline PPO (no trained model)")

    # Heuristic baselines
    for name, agent_cls in [
        ("fixed_interval", FixedIntervalHeuristic),
        ("signal_threshold", SignalThresholdHeuristic),
        ("greedy_lookahead", GreedyLookaheadHeuristic),
    ]:
        agent = agent_cls()
        results[name] = evaluate_agent(agent, env, name, n_episodes=200)

    # Print summary
    print("\n  Method           | Uptime  | Reward  | Efficiency")
    print("  " + "-" * 55)
    for name, res in results.items():
        uptimes = [r["communication_uptime"] for r in res]
        rewards = [r["total_reward"] for r in res]
        effs = [r["relay_efficiency"] for r in res]
        print(f"  {name:<18} | {np.mean(uptimes):>5.1%} | {np.mean(rewards):>+7.1f} | {np.mean(effs):>.3f}")

    save_results("experiment_1_comparison.json", results)
    return results


def experiment_2_budget_pressure():
    """Sweep budget levels to test resource pressure resilience."""
    print("\n" + "=" * 60)
    print("Experiment 2: Budget Pressure Sweep")
    print("=" * 60)

    budgets = [3, 4, 5, 6, 7, 8]
    results = {b: {} for b in budgets}

    for budget in budgets:
        print(f"\n  Budget = {budget}:")
        env = make_env(budget=budget, seed=2000)
        test_obs, _ = env.reset(seed=2000)
        obs_dim = len(flatten_observation(test_obs))

        # Heuristics (always available)
        for name, agent_cls in [
            ("fixed_interval", FixedIntervalHeuristic),
            ("signal_threshold", SignalThresholdHeuristic),
        ]:
            agent = agent_cls()
            res = evaluate_agent(agent, env, name, n_episodes=100)
            results[budget][name] = {
                "uptime": float(np.mean([r["communication_uptime"] for r in res])),
                "reward": float(np.mean([r["total_reward"] for r in res])),
            }

        # Trained agents (if available)
        voi_path = Path("checkpoints/voi_ppo/best_model.pt")
        if voi_path.exists():
            agent = load_trained_agent("voi_ppo", str(voi_path), obs_dim)
            res = evaluate_agent(agent, env, "voi_ppo", n_episodes=100)
            results[budget]["voi_ppo"] = {
                "uptime": float(np.mean([r["communication_uptime"] for r in res])),
                "reward": float(np.mean([r["total_reward"] for r in res])),
            }

        bl_path = Path("checkpoints/baseline_ppo/best_model.pt")
        if bl_path.exists():
            agent = load_trained_agent("baseline_ppo", str(bl_path), obs_dim)
            res = evaluate_agent(agent, env, "baseline_ppo", n_episodes=100)
            results[budget]["baseline_ppo"] = {
                "uptime": float(np.mean([r["communication_uptime"] for r in res])),
                "reward": float(np.mean([r["total_reward"] for r in res])),
            }

        for method, data in results[budget].items():
            print(f"    {method:<18}: uptime={data['uptime']:.1%}, reward={data['reward']:+.1f}")

    save_results("experiment_2_budget_pressure.json", results)
    return results


def experiment_3_noise_robustness():
    """Test performance under varying observation noise."""
    print("\n" + "=" * 60)
    print("Experiment 3: Noise Robustness")
    print("=" * 60)

    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    results = {str(n): {} for n in noise_levels}

    for noise in noise_levels:
        print(f"\n  Noise = {noise:.2f}:")
        env = make_env(noise=noise, seed=3000)
        test_obs, _ = env.reset(seed=3000)
        obs_dim = len(flatten_observation(test_obs))

        for name, agent_cls in [
            ("fixed_interval", FixedIntervalHeuristic),
            ("signal_threshold", SignalThresholdHeuristic),
        ]:
            agent = agent_cls()
            res = evaluate_agent(agent, env, name, n_episodes=100)
            results[str(noise)][name] = {
                "uptime": float(np.mean([r["communication_uptime"] for r in res])),
            }

        voi_path = Path("checkpoints/voi_ppo/best_model.pt")
        if voi_path.exists():
            agent = load_trained_agent("voi_ppo", str(voi_path), obs_dim)
            res = evaluate_agent(agent, env, "voi_ppo", n_episodes=100)
            results[str(noise)]["voi_ppo"] = {
                "uptime": float(np.mean([r["communication_uptime"] for r in res])),
            }

    save_results("experiment_3_noise.json", results)
    return results


def save_results(filename: str, data):
    """Save experiment results to JSON."""
    path = RESULTS_DIR / filename
    serializable = {}
    for key, val in data.items():
        if isinstance(val, list):
            serializable[str(key)] = [
                {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                 for k, v in item.items() if k != "deploy_times"}
                for item in val
            ]
        elif isinstance(val, dict):
            serializable[str(key)] = {
                k2: (v2 if not isinstance(v2, dict) else v2)
                for k2, v2 in val.items()
            }
        else:
            serializable[str(key)] = val

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    start = time.time()

    experiment_1_main_comparison()
    experiment_2_budget_pressure()
    experiment_3_noise_robustness()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"All experiments complete! Total time: {elapsed:.1f}s")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'=' * 60}")
