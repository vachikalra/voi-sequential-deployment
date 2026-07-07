"""
Graph Attention Network (GAT) encoder for relay topology.

Encodes the discovered relay graph into a fixed-size latent representation.
Nodes represent deployed relays + team position. Edges represent tunnel
segments between them with signal quality features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Single GAT attention head."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [batch, n_nodes, in_features]
            adj: Adjacency matrix [batch, n_nodes, n_nodes]

        Returns:
            Updated node features [batch, n_nodes, out_features]
        """
        h = self.W(x)  # [batch, n_nodes, out_features]
        batch_size, n_nodes, out_dim = h.shape

        # Compute attention coefficients
        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1)  # [B, N, N, F]
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)  # [B, N, N, F]
        e = self.leaky_relu(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))

        # Mask non-adjacent nodes
        mask = (adj == 0)
        e = e.masked_fill(mask, float("-inf"))

        # Softmax attention
        alpha = F.softmax(e, dim=-1)
        alpha = alpha.masked_fill(mask, 0.0)
        alpha = self.dropout(alpha)

        # Weighted aggregation
        out = torch.bmm(alpha, h)
        return out


class MultiHeadGAT(nn.Module):
    """Multi-head Graph Attention Network."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads

        # First layer: multi-head attention
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(in_features, hidden_features, dropout)
            for _ in range(n_heads)
        ])

        # Intermediate layers
        self.intermediate_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.intermediate_layers.append(nn.ModuleList([
                GraphAttentionLayer(hidden_features * n_heads, hidden_features, dropout)
                for _ in range(n_heads)
            ]))

        # Output projection
        self.output_proj = nn.Linear(hidden_features * n_heads, out_features)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [batch, n_nodes, in_features]
            adj: Adjacency matrix [batch, n_nodes, n_nodes]

        Returns:
            Graph embedding [batch, out_features]
        """
        # Multi-head attention (first layer)
        head_outputs = [head(x, adj) for head in self.attention_heads]
        h = torch.cat(head_outputs, dim=-1)  # [B, N, hidden*heads]
        h = F.elu(h)

        # Intermediate layers
        for layer_heads in self.intermediate_layers:
            head_outputs = [head(h, adj) for head in layer_heads]
            h = torch.cat(head_outputs, dim=-1)
            h = F.elu(h)

        # Global graph pooling (mean + max)
        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1).values
        h_global = h_mean + h_max

        # Project to output dimension
        out = self.output_proj(h_global)
        out = self.layer_norm(out)
        return out


class RelayGraphEncoder(nn.Module):
    """
    Encodes the relay deployment graph into a latent vector.

    Each node in the graph represents either:
    - A deployed relay (features: SINR, position, time deployed)
    - The team's current position (features: local geology, signal quality)
    - The base station (features: fixed)

    Edges represent tunnel segments between consecutive relays.
    """

    def __init__(
        self,
        node_feature_dim: int = 6,
        edge_feature_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gat = MultiHeadGAT(
            in_features=hidden_dim,
            hidden_features=hidden_dim // n_heads,
            out_features=output_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )

    def forward(
        self, node_features: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_features: [batch, max_nodes, node_feature_dim]
            adj: [batch, max_nodes, max_nodes]

        Returns:
            Graph embedding: [batch, output_dim]
        """
        h = self.node_encoder(node_features)
        return self.gat(h, adj)
