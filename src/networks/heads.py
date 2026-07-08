"""
Output heads for policy, value, and VoI estimation.

Three separate neural network heads that share a common belief state:
1. Policy head: π(a|belief) — action probabilities
2. Value head: V(belief) — expected return from current state
3. VoI head: VoI(belief) — estimated value of waiting for more information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PolicyHead(nn.Module):
    """
    Policy network: maps belief state to action distribution.

    Output: P(WAIT), P(COMMIT) — softmax distribution over binary actions.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        """
        Args:
            belief: [batch, input_dim]
        Returns:
            action_logits: [batch, 2]
        """
        return self.net(belief)

    def get_distribution(self, belief: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.forward(belief)
        return torch.distributions.Categorical(logits=logits)


class ValueHead(nn.Module):
    """
    Value network: estimates V(belief) = E[sum of future rewards | belief].

    Used for advantage computation in PPO.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        """
        Args:
            belief: [batch, input_dim]
        Returns:
            value: [batch, 1]
        """
        return self.net(belief)


class VoIHead(nn.Module):
    """
    Value-of-Information estimation head.

    Estimates: VoI(belief) = E_{o_{t+1}} [max_a Q(belief', a) | WAIT] - max_a Q(belief, a)

    Intuitively: "How much would my decision improve if I waited one more step
    and got one more observation?"

    This is the core novel component. The VoI estimate creates a conservative
    bias — the agent only commits when the immediate value of committing
    exceeds both the value of waiting AND the expected information gain.

    Training target: computed from hindsight by comparing:
      - Value of best action at time t (before observation t+1)
      - Value of best action at time t+1 (after observation t+1)
      - The difference is the realized information value

    The VoI head learns to PREDICT this value before actually receiving
    the next observation.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # VoI is always non-negative
        )

    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        """
        Args:
            belief: [batch, input_dim]
        Returns:
            voi_estimate: [batch, 1] — always >= 0
        """
        return self.net(belief)


class VoIAugmentedActorCritic(nn.Module):
    """
    Complete actor-critic architecture with VoI augmentation.

    The action selection incorporates VoI:
        - Compute Q(COMMIT) and Q(WAIT) from policy + value
        - Compute VoI estimate
        - Only select COMMIT if Q(COMMIT) > Q(WAIT) + VoI

    This creates an information-aware policy that waits when uncertain
    and commits when confident.
    """

    def __init__(
        self,
        belief_dim: int,
        hidden_dim: int = 64,
        voi_weight: float = 1.0,
    ):
        super().__init__()
        self.voi_weight = voi_weight

        self.policy_head = PolicyHead(belief_dim, hidden_dim)
        self.value_head = ValueHead(belief_dim, hidden_dim)
        self.voi_head = VoIHead(belief_dim, hidden_dim)

    def forward(
        self, belief: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute all outputs from belief state.

        Returns:
            action_logits: [batch, 2] — raw policy logits
            value: [batch, 1] — state value estimate
            voi: [batch, 1] — value of information estimate
        """
        action_logits = self.policy_head(belief)
        value = self.value_head(belief)
        voi = self.voi_head(belief)
        return action_logits, value, voi

    def get_action(
        self, belief: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action with VoI-guided decision making.

        The VoI acts as a "commitment threshold" — it raises the bar
        for choosing COMMIT by adding the expected information value
        to the WAIT action's appeal.
        """
        action_logits, value, voi = self.forward(belief)

        # Modify logits: boost WAIT by VoI amount
        modified_logits = action_logits.clone()
        voi_clamped = torch.clamp(voi, 0.0, 10.0)
        modified_logits[:, 0] += self.voi_weight * voi_clamped.squeeze(-1)  # boost WAIT

        # Safety: replace any NaN with zeros
        modified_logits = torch.nan_to_num(modified_logits, nan=0.0)

        if deterministic:
            action = modified_logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=modified_logits)
            action = dist.sample()

        log_prob = F.log_softmax(modified_logits, dim=-1)
        action_log_prob = log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1)

        return action, action_log_prob, value.squeeze(-1)

    def evaluate_actions(
        self, belief: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions (for PPO update).

        Returns:
            action_log_probs, values, entropy, voi_estimates
        """
        belief = torch.nan_to_num(belief, nan=0.0)
        action_logits, value, voi = self.forward(belief)

        modified_logits = action_logits.clone()
        voi_clamped = torch.clamp(voi, 0.0, 10.0)
        modified_logits[:, 0] += self.voi_weight * voi_clamped.squeeze(-1)
        modified_logits = torch.nan_to_num(modified_logits, nan=0.0)

        dist = torch.distributions.Categorical(logits=modified_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, value.squeeze(-1), entropy, voi.squeeze(-1)
