from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SparseGCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, adjacency_norm: torch.Tensor) -> torch.Tensor:
        support = x @ self.weight
        output = torch.sparse.mm(adjacency_norm, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GCNEncoder(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = SparseGCNLayer(in_features, hidden_dim)
        self.conv2 = SparseGCNLayer(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.prelu1 = nn.PReLU(hidden_dim)
        self.prelu2 = nn.PReLU(hidden_dim)

    def forward(self, x: torch.Tensor, adjacency_norm: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h1 = self.prelu1(self.conv1(x, adjacency_norm))
        h1_drop = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.prelu2(self.conv2(h1_drop, adjacency_norm))
        return h2, [h1, h2]


class SupervisedGCN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = GCNEncoder(in_features, hidden_dim, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adjacency_norm: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        z, hidden = self.encoder(x, adjacency_norm)
        logits = self.classifier(F.dropout(z, p=self.dropout, training=self.training))
        return logits, hidden


class DGIModel(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = GCNEncoder(in_features, hidden_dim, dropout=0.0)
        self.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.weight)

    def discriminate(self, z: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        projected = self.weight @ summary
        return z @ projected

    def forward(
        self,
        x: torch.Tensor,
        adjacency_norm: torch.Tensor,
        permutation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        pos_z, hidden = self.encoder(x, adjacency_norm)
        if permutation is None:
            permutation = torch.randperm(x.size(0), device=x.device)
        neg_z, _ = self.encoder(x[permutation], adjacency_norm)
        summary = torch.sigmoid(pos_z.mean(dim=0))
        return pos_z, neg_z, summary, hidden

    def loss(self, pos_z: torch.Tensor, neg_z: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        pos_logits = self.discriminate(pos_z, summary)
        neg_logits = self.discriminate(neg_z, summary)
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        return pos_loss + neg_loss
