# Value-of-Information Guided Decision Making in Resource-Constrained POMDPs with Irreversible Actions

## Research Overview

**Research Question**: Can explicitly estimating the value of future information improve commitment timing in sequential decision problems with finite irreversible actions?

**Core Insight**: Standard reinforcement learning treats all actions as equal-cost decisions. But in problems where actions are *irreversible* and resources are *finite*, the act of **waiting** has implicit value — it reveals information that could improve future decisions. This project proposes a method that explicitly estimates this Value of Information (VoI) and incorporates it into the action-selection process.

**Contribution**: A VoI-augmented PPO algorithm that learns *when* to commit irreversible actions by estimating how much decision quality would improve from additional observations. We demonstrate this outperforms standard model-free RL, model-based RL, and hand-coded heuristics across multiple domains with irreversible sequential deployment.

---

## Problem Class: Irreversible-Action POMDPs

We study a specific class of POMDPs where:
1. **Actions are permanent** — once committed, cannot be undone
2. **Resources are finite** — the agent has N total irreversible actions for the entire episode
3. **Observations are partial** — the agent cannot see the full environment state
4. **Information arrives sequentially** — waiting reveals new observations

Formally, the POMDP tuple (S, A, O, T, Ω, R, γ) with additional constraints:
- A = {WAIT, COMMIT}
- COMMIT decrements a finite budget: n_{t+1} = n_t - 1 if a_t = COMMIT
- Episode terminates if n_t = 0 (budget exhausted) or t = T (horizon reached)

### The VoI Decision Criterion

At each timestep, the agent decides:

```
a_t = COMMIT  if  Q(o_{1:t}, COMMIT) > Q(o_{1:t}, WAIT) + VoI(o_{1:t})
a_t = WAIT    otherwise
```

Where VoI is the expected improvement in decision quality from one additional observation:

```
VoI(o_{1:t}) = E_{o_{t+1}} [max_a Q(o_{1:t+1}, a) | WAIT] - max_a Q(o_{1:t}, a)
```

This creates a *conservative bias* — the agent only commits when the immediate value of acting exceeds both the Q-value of waiting AND the expected information gain from waiting.

---

## Methods

### 1. VoI-Guided PPO (Proposed Method)

Our primary contribution. A modified PPO agent with an auxiliary VoI estimation head that modulates action selection. The VoI estimator is trained to predict how much the value function improves after one additional observation step.

### 2. Standard PPO Baseline

Vanilla PPO with entropy bonus for exploration. No explicit reasoning about irreversibility or information value. Represents state-of-the-art model-free RL applied naively to the problem.

### 3. Model-Based RL Baseline

Learns a world model (transition dynamics + observation model), then uses Monte Carlo Tree Search to plan deployment decisions. Represents the "learn a model, then plan" paradigm.

### 4. Heuristic Baselines

- **Fixed-interval**: Commit every K timesteps regardless of observations
- **Threshold**: Commit when a monitored signal crosses a threshold
- **Oracle (upper bound)**: Has full environment knowledge, solves optimal placement offline via integer programming

---

## Evaluation Domains

The method is tested across multiple domains to demonstrate generality:

1. **Communication Relay Deployment** — Deploy relay nodes in an underground tunnel network to maintain signal connectivity under partial observability
2. **Sensor Placement** — Place sensors in an unknown building to maximize coverage with limited sensors
3. **Sequential Resource Allocation** — Allocate finite resources across sequentially-revealed opportunities (secretary problem variant)

---

## Repository Structure

```
├── src/
│   ├── environments/          # POMDP environments (Gymnasium-compatible)
│   │   ├── base_env.py        # Abstract base class for irreversible-action POMDPs
│   │   └── domains/           # Specific instantiations
│   ├── methods/
│   │   ├── voi_agent/         # Proposed method: VoI-guided PPO
│   │   ├── baseline_ppo/      # Standard PPO comparison
│   │   ├── model_based/       # World model + planning comparison
│   │   └── heuristics/        # Rule-based baselines + oracle
│   ├── networks/              # Neural network architectures
│   ├── training/              # Training infrastructure
│   └── evaluation/            # Metrics, statistics, visualization
├── experiments/               # Reproducible experiment scripts
├── demo/                      # Interactive demonstration
├── paper/                     # LaTeX paper source
├── configs/                   # Hyperparameter configurations
└── tests/                     # Unit tests
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Train VoI agent on relay deployment domain
python -m src.methods.voi_agent.train --config configs/training_config.yaml

# Run full experiment suite
python experiments/run_all.py

# Launch interactive demo
python demo/app.py
```

---

## Key Results (Expected)

| Method | Comm. Uptime | Resources Used | Efficiency (Uptime/Resource) |
|--------|-------------|----------------|------------------------------|
| Fixed-interval | ~70% | N (all) | 0.70 |
| Threshold | ~82% | ~0.8N | 0.85 |
| Standard PPO | ~88% | ~0.7N | 0.92 |
| Model-Based RL | ~85% | ~0.6N | 0.94 |
| **VoI-Guided PPO** | **~93%** | **~0.6N** | **0.98** |
| Oracle (ceiling) | ~98% | ~0.5N | 1.00 |

---

## Citation

```bibtex
@inproceedings{kalra2025voi,
  title={Value-of-Information Guided Decision Making in Resource-Constrained POMDPs with Irreversible Actions},
  author={Kalra, Vachi},
  year={2025},
  note={Undergraduate Research Symposium}
}
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric
- Gymnasium
- NetworkX
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- Weights & Biases (optional, for experiment tracking)
