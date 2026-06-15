from __future__ import annotations

import torch
from torch import nn


class EEGAttentionProbe(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        num_channels: int,
        electrode_positions: torch.Tensor,
        num_classes: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        channel_id_dim: int,
        position_dim: int,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.register_buffer("electrode_positions", electrode_positions.float())
        self.channel_ids = nn.Embedding(num_channels, channel_id_dim)
        self.position_encoder = nn.Sequential(
            nn.Linear(3, position_dim),
            nn.GELU(),
            nn.Linear(position_dim, position_dim),
        )
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim + channel_id_dim + position_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, channel_embeddings: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = channel_embeddings.shape
        if channels != self.num_channels:
            channel_embeddings = self._align_channels(channel_embeddings, channels)
            channels = self.num_channels

        ids = torch.arange(channels, device=channel_embeddings.device)
        channel_features = self.channel_ids(ids).unsqueeze(0).expand(batch, -1, -1)
        position_features = self.position_encoder(self.electrode_positions[:channels])
        position_features = position_features.unsqueeze(0).expand(batch, -1, -1)
        tokens = torch.cat([channel_embeddings, channel_features, position_features], dim=-1)
        tokens = self.input_projection(tokens)

        query = self.query.expand(batch, -1, -1)
        pooled, _ = self.attention(query=query, key=tokens, value=tokens, need_weights=False)
        pooled = self.norm(pooled.squeeze(1))
        return self.classifier(pooled)

    def _align_channels(self, channel_embeddings: torch.Tensor, channels: int) -> torch.Tensor:
        if channels > self.num_channels:
            return channel_embeddings[:, : self.num_channels]
        repeat_count = self.num_channels - channels
        padding = channel_embeddings[:, -1:, :].expand(-1, repeat_count, -1)
        return torch.cat([channel_embeddings, padding], dim=1)


class ConcatenatedLinearProbe(nn.Module):
    def __init__(self, *, embedding_dim: int, num_channels: int, num_classes: int):
        super().__init__()
        self.num_channels = num_channels
        self.classifier = nn.Linear(embedding_dim * num_channels, num_classes)

    def forward(self, channel_embeddings: torch.Tensor) -> torch.Tensor:
        if channel_embeddings.shape[1] != self.num_channels:
            channel_embeddings = self._align_channels(channel_embeddings)
        return self.classifier(channel_embeddings.flatten(start_dim=1))

    def _align_channels(self, channel_embeddings: torch.Tensor) -> torch.Tensor:
        channels = channel_embeddings.shape[1]
        if channels > self.num_channels:
            return channel_embeddings[:, : self.num_channels]
        repeat_count = self.num_channels - channels
        padding = channel_embeddings[:, -1:, :].expand(-1, repeat_count, -1)
        return torch.cat([channel_embeddings, padding], dim=1)
