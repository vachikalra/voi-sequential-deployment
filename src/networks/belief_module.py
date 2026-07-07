"""
Belief module: GRU-based recurrent network for maintaining belief state
under partial observability.

In a POMDP, the agent needs to maintain a summary of its observation history
(belief state) to make optimal decisions. The GRU learns to compress the
sequence of observations into a fixed-size hidden state that captures
relevant information about unseen parts of the environment.
"""

import torch
import torch.nn as nn


class BeliefModule(nn.Module):
    """
    Recurrent belief state module.

    Takes current observation embedding and previous hidden state,
    outputs updated belief state. This captures:
    - Temporal patterns in signal degradation
    - Memory of topology structure seen so far
    - Implicit prediction of unseen environment regions
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        observation_embedding: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update belief state with new observation.

        Args:
            observation_embedding: [batch, seq_len, input_dim] or [batch, input_dim]
            hidden: [n_layers, batch, hidden_dim] or None

        Returns:
            belief_state: [batch, hidden_dim] (latest belief)
            hidden: [n_layers, batch, hidden_dim] (for next step)
        """
        if observation_embedding.dim() == 2:
            observation_embedding = observation_embedding.unsqueeze(1)

        if hidden is None:
            hidden = self.init_hidden(observation_embedding.shape[0],
                                     observation_embedding.device)

        output, hidden_new = self.gru(observation_embedding, hidden)

        # Take the last timestep output as belief state
        belief_state = output[:, -1, :]
        belief_state = self.layer_norm(belief_state)

        return belief_state, hidden_new

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state to zeros."""
        return torch.zeros(
            self.n_layers, batch_size, self.hidden_dim, device=device
        )


class AttentiveBeliefModule(nn.Module):
    """
    Enhanced belief module with self-attention over observation history.

    Instead of relying solely on GRU compression, this module maintains
    a buffer of past observation embeddings and uses attention to selectively
    recall relevant past information when making decisions.

    Useful when early observations are critical for late decisions
    (e.g., remembering a branch point encountered 50 steps ago).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_attention_heads: int = 4,
        memory_length: int = 50,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_length = memory_length

        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, batch_first=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_attention_heads,
            batch_first=True,
        )

        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self._memory_buffer: torch.Tensor | None = None

    def forward(
        self,
        observation_embedding: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            observation_embedding: [batch, input_dim]
            hidden: [1, batch, hidden_dim]

        Returns:
            belief_state: [batch, hidden_dim]
            hidden: [1, batch, hidden_dim]
        """
        batch_size = observation_embedding.shape[0]

        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_dim,
                                device=observation_embedding.device)
            self._memory_buffer = torch.zeros(
                batch_size, 0, self.hidden_dim,
                device=observation_embedding.device,
            )

        # GRU update
        obs_seq = observation_embedding.unsqueeze(1)
        gru_out, hidden_new = self.gru(obs_seq, hidden)
        gru_belief = gru_out.squeeze(1)

        # Append to memory buffer
        self._memory_buffer = torch.cat(
            [self._memory_buffer, gru_belief.unsqueeze(1)], dim=1
        )
        if self._memory_buffer.shape[1] > self.memory_length:
            self._memory_buffer = self._memory_buffer[:, -self.memory_length:, :]

        # Self-attention over memory
        if self._memory_buffer.shape[1] > 1:
            query = gru_belief.unsqueeze(1)  # [B, 1, H]
            attn_out, _ = self.attention(query, self._memory_buffer, self._memory_buffer)
            attn_belief = attn_out.squeeze(1)
        else:
            attn_belief = gru_belief

        # Combine GRU and attention-based beliefs
        combined = self.combine(torch.cat([gru_belief, attn_belief], dim=-1))

        return combined, hidden_new

    def reset_memory(self) -> None:
        """Clear memory buffer for new episode."""
        self._memory_buffer = None
