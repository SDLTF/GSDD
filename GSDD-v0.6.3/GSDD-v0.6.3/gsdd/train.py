from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .models import DGIModel, SupervisedGCN


@dataclass
class TrainResult:
    best_epoch: int
    best_value: float
    history: list[dict[str, float]]


def accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    count = int(mask.sum().item())
    if count == 0:
        return float("nan")
    predictions = logits[mask].argmax(dim=1)
    return float((predictions == labels[mask]).float().mean().item())


def train_supervised(
    model: SupervisedGCN,
    x: torch.Tensor,
    y: torch.Tensor,
    adjacency_norm: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    verbose: bool,
) -> TrainResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x, adjacency_norm)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(x, adjacency_norm)
            val_loss = float(F.cross_entropy(logits[val_mask], y[val_mask]).item())
            train_acc = accuracy(logits, y, train_mask)
            val_acc = accuracy(logits, y, val_mask)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(loss.item()),
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == epochs):
            print(
                f"[Supervised] epoch={epoch:04d} loss={loss.item():.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
                flush=True,
            )
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    return TrainResult(best_epoch=best_epoch, best_value=best_val_loss, history=history)


def train_dgi(
    model: DGIModel,
    x: torch.Tensor,
    adjacency_norm: torch.Tensor,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    verbose: bool,
) -> TrainResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pos_z, neg_z, summary, _ = model(x, adjacency_norm)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        value = float(loss.item())
        history.append({"epoch": float(epoch), "ssl_loss": value})

        if value < best_loss - 1e-6:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == epochs):
            print(f"[DGI] epoch={epoch:04d} loss={value:.4f}", flush=True)
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    return TrainResult(best_epoch=best_epoch, best_value=best_loss, history=history)


@torch.no_grad()
def extract_supervised_hidden(
    model: SupervisedGCN,
    x: torch.Tensor,
    adjacency_norm: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    model.eval()
    return model(x, adjacency_norm)


@torch.no_grad()
def extract_dgi_hidden(
    model: DGIModel,
    x: torch.Tensor,
    adjacency_norm: torch.Tensor,
) -> list[torch.Tensor]:
    model.eval()
    _, hidden = model.encoder(x, adjacency_norm)
    return hidden
