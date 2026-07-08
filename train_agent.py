"""
Complete end-to-end training script.

Run with:
    python train_agent.py --method voi_ppo --steps 500000
    python train_agent.py --method baseline_ppo --steps 500000
"""

import argparse
import time
import json
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch

from src.environments.domains.relay_deployment import (
    RelayDeploymentEnv,
    RelayDeploymentConfig,
)
from src.environments.mine_topology import MineTopologyConfig
from src.methods.voi_agent.train import VoIPPOAgent, VoIPPOConfig
from src.training.buffer import RolloutBuffer
from src.training.curriculum import CurriculumScheduler
from src.training.exploration import ResourceAdaptiveExploration


def flatten_observation(obs: dict) -> np.ndarray:
    """Convert dict observation to flat numpy array."""
    arrays = []
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, np.ndarray):
            arrays.append(np.nan_to_num(val.flatten().astype(np.float32), nan=0.0))
        elif isinstance(val, (int, float)):
            v = 0.0 if np.isnan(val) else float(val)
            arrays.append(np.array([v], dtype=np.float32))
    return np.concatenate(arrays)


def make_env(curriculum: CurriculumScheduler, seed: int = None) -> RelayDeploymentEnv:
    """Create environment from current curriculum stage."""
    stage = curriculum.current_stage
    mine_config = MineTopologyConfig(
        total_depth=stage.mine_depth,
        branch_probability=stage.branch_probability,
        seed=seed,
    )
    env_config = RelayDeploymentConfig(
        mine_config=mine_config,
        initial_budget=stage.initial_budget,
        max_horizon=stage.max_horizon,
        observation_noise_std=stage.observation_noise_std,
    )
    return RelayDeploymentEnv(config=env_config)


def train(
    method: str = "voi_ppo",
    total_steps: int = 500_000,
    rollout_steps: int = 2048,
    seed: int = 42,
    save_dir: str = "checkpoints",
    log_interval: int = 10,
    eval_interval: int = 50,
    use_curriculum: bool = True,
    device: str = "cpu",
):
    """
    Main training loop.

    Alternates between:
    1. Collecting rollouts (agent interacts with environment)
    2. Computing advantages (GAE)
    3. Updating policy (PPO + VoI)
    """
    print(f"{'='*60}")
    print(f"Training: {method}")
    print(f"Total steps: {total_steps:,}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Determine observation dimension by running one step
    curriculum = CurriculumScheduler()
    test_env = make_env(curriculum, seed=seed)
    test_obs, _ = test_env.reset(seed=seed)
    obs_dim = len(flatten_observation(test_obs))
    print(f"Observation dimension: {obs_dim}")

    # Create agent
    use_voi = method == "voi_ppo"
    config = VoIPPOConfig(
        observation_dim=obs_dim,
        total_timesteps=total_steps,
        voi_weight=1.0 if use_voi else 0.0,
        use_curriculum=use_curriculum,
        n_steps_per_rollout=rollout_steps,
    )
    agent = VoIPPOAgent(config, device=device)

    # Adaptive exploration (only for VoI method)
    exploration = ResourceAdaptiveExploration() if use_voi else None

    # Buffer
    buffer = RolloutBuffer(buffer_size=rollout_steps, obs_dim=obs_dim)

    # Tracking
    total_timesteps = 0
    n_updates = 0
    n_episodes = 0
    episode_rewards = []
    episode_uptimes = []
    best_mean_reward = -np.inf
    training_log = []

    save_path = Path(save_dir) / method
    save_path.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = make_env(curriculum, seed=seed)
    obs, info = env.reset(seed=seed)
    flat_obs = flatten_observation(obs)
    agent.reset()

    episode_reward = 0.0
    episode_steps = 0

    start_time = time.time()

    while total_timesteps < total_steps:
        # === COLLECT ROLLOUTS ===
        buffer.reset()

        for step in range(rollout_steps):
            # Get action from agent
            action, action_info = agent.get_action(flat_obs, deterministic=False)

            # Apply resource-adaptive exploration override
            if exploration and use_voi:
                if exploration.should_force_wait(
                    env.budget_remaining,
                    env._timestep,
                    env.config.max_horizon,
                ):
                    action = 0  # force WAIT

            # Step environment
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            next_flat_obs = flatten_observation(next_obs)

            # Store in buffer
            buffer.add(
                obs=flat_obs,
                action=action,
                reward=reward,
                value=action_info["value"],
                log_prob=action_info["log_prob"],
                done=done,
                voi=action_info.get("voi", 0.0),
            )

            # Store reward for VoI training
            agent.store_reward(reward)

            flat_obs = next_flat_obs
            episode_reward += reward
            episode_steps += 1
            total_timesteps += 1

            if done:
                # Episode finished
                n_episodes += 1
                episode_rewards.append(episode_reward)

                # Get metrics
                metrics = env.get_metrics()
                episode_uptimes.append(metrics["communication_uptime"])

                # Report to curriculum
                normalized_reward = metrics["communication_uptime"]
                promoted = curriculum.report_episode_reward(normalized_reward)
                if promoted:
                    print(f"\n🎓 Promoted to: {curriculum.current_stage.name}\n")

                # End episode for VoI training
                agent.end_episode()

                # Reset
                env = make_env(curriculum, seed=seed + n_episodes)
                obs, info = env.reset(seed=seed + n_episodes)
                flat_obs = flatten_observation(obs)
                agent.reset()
                episode_reward = 0.0
                episode_steps = 0

        # === COMPUTE ADVANTAGES ===
        with torch.no_grad():
            _, last_info = agent.get_action(flat_obs, deterministic=True)
            last_value = last_info["value"]
        buffer.compute_advantages(last_value)

        # === UPDATE POLICY ===
        rollout_data = buffer.get()
        update_metrics = agent.update(rollout_data)
        n_updates += 1

        # === LOGGING ===
        if n_updates % log_interval == 0:
            elapsed = time.time() - start_time
            fps = total_timesteps / max(elapsed, 1)

            recent_rewards = episode_rewards[-100:] if episode_rewards else [0]
            recent_uptimes = episode_uptimes[-100:] if episode_uptimes else [0]
            mean_reward = np.mean(recent_rewards)
            mean_uptime = np.mean(recent_uptimes)

            log_entry = {
                "timesteps": total_timesteps,
                "updates": n_updates,
                "episodes": n_episodes,
                "mean_reward_100": float(mean_reward),
                "mean_uptime_100": float(mean_uptime),
                "policy_loss": update_metrics.get("policy_loss", 0),
                "value_loss": update_metrics.get("value_loss", 0),
                "entropy": update_metrics.get("entropy", 0),
                "voi_loss": update_metrics.get("voi_loss", 0),
                "curriculum_stage": curriculum.current_stage_idx,
                "fps": fps,
                "elapsed_s": elapsed,
            }
            training_log.append(log_entry)

            print(
                f"[{total_timesteps:>8,}/{total_steps:,}] "
                f"Ep: {n_episodes:>5} | "
                f"Reward: {mean_reward:>7.3f} | "
                f"Uptime: {mean_uptime:>5.1%} | "
                f"Stage: {curriculum.current_stage_idx} | "
                f"FPS: {fps:.0f}"
            )

            # Save best
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                agent.save(str(save_path / "best_model.pt"))

        # === PERIODIC SAVE ===
        if n_updates % eval_interval == 0:
            agent.save(str(save_path / f"checkpoint_{total_timesteps}.pt"))

    # === TRAINING COMPLETE ===
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Total time: {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    print(f"Total episodes: {n_episodes}")
    print(f"Final mean reward (100): {np.mean(episode_rewards[-100:]):.3f}")
    print(f"Final mean uptime (100): {np.mean(episode_uptimes[-100:]):.1%}")
    print(f"Best mean reward: {best_mean_reward:.3f}")
    print(f"{'='*60}\n")

    # Save final model and training log
    agent.save(str(save_path / "final_model.pt"))
    with open(save_path / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    return training_log


def evaluate(
    method: str = "voi_ppo",
    model_path: str = None,
    n_episodes: int = 100,
    seed: int = 1000,
    complexity: float = 0.15,
    budget: int = 8,
):
    """
    Evaluate a trained agent.

    Returns metrics across n_episodes.
    """
    print(f"\nEvaluating: {method} ({n_episodes} episodes)")

    # Create environment
    mine_config = MineTopologyConfig(
        total_depth=250.0,
        branch_probability=complexity,
        seed=seed,
    )
    env_config = RelayDeploymentConfig(
        mine_config=mine_config,
        initial_budget=budget,
        max_horizon=150,
    )
    env = RelayDeploymentEnv(config=env_config)

    # Create/load agent
    test_obs, _ = env.reset(seed=seed)
    obs_dim = len(flatten_observation(test_obs))

    if method in ("voi_ppo", "baseline_ppo"):
        use_voi = method == "voi_ppo"
        config = VoIPPOConfig(
            observation_dim=obs_dim,
            voi_weight=1.0 if use_voi else 0.0,
        )
        agent = VoIPPOAgent(config)
        if model_path:
            agent.load(model_path)
    elif method == "threshold":
        from src.methods.heuristics.baselines import SignalThresholdHeuristic
        agent = SignalThresholdHeuristic()
    elif method == "fixed":
        from src.methods.heuristics.baselines import FixedIntervalHeuristic
        agent = FixedIntervalHeuristic()
    elif method == "greedy":
        from src.methods.heuristics.baselines import GreedyLookaheadHeuristic
        agent = GreedyLookaheadHeuristic()

    # Run evaluation
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        if hasattr(agent, "reset"):
            agent.reset()
        done = False
        ep_reward = 0.0

        while not done:
            if method in ("voi_ppo", "baseline_ppo"):
                flat_obs = flatten_observation(obs)
                action, _ = agent.get_action(flat_obs, deterministic=True)
            else:
                action = agent.decide(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        metrics = env.get_metrics()
        metrics["total_reward"] = ep_reward
        results.append(metrics)

    # Summary
    uptimes = [r["communication_uptime"] for r in results]
    relays = [r["relays_deployed"] for r in results]
    efficiencies = [r["relay_efficiency"] for r in results]

    print(f"  Uptime:     {np.mean(uptimes):.1%} ± {np.std(uptimes):.1%}")
    print(f"  Relays:     {np.mean(relays):.1f} ± {np.std(relays):.1f}")
    print(f"  Efficiency: {np.mean(efficiencies):.3f} ± {np.std(efficiencies):.3f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate VoI agent")
    parser.add_argument("--method", type=str, default="voi_ppo",
                        choices=["voi_ppo", "baseline_ppo", "threshold", "fixed", "greedy"])
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    if args.eval_only:
        evaluate(
            method=args.method,
            model_path=args.model_path,
            seed=args.seed,
        )
    else:
        train(
            method=args.method,
            total_steps=args.steps,
            seed=args.seed,
            device=args.device,
            save_dir=args.save_dir,
        )
