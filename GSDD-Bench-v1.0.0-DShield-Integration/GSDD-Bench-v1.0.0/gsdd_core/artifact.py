from __future__ import annotations

from collections.abc import Mapping

import torch


_INDEX_KEYS = (
    "train_idx",
    "poison_train_idx",
    "val_idx",
    "clean_test_idx",
    "attack_test_idx",
    "attach_idx",
)

_LABELED_INDEX_KEYS = (
    "train_idx",
    "poison_train_idx",
    "val_idx",
    "clean_test_idx",
    "attack_test_idx",
    "attach_idx",
)


def _as_1d_long(value: torch.Tensor | object, key: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.detach().cpu().long().reshape(-1)
    if value.numel() and int(value.min().item()) < 0:
        raise ValueError(f"{key} contains a negative node index")
    return value


def normalize_attack_bundle(bundle: Mapping[str, object]) -> dict[str, object]:
    """Align node-sized tensors in an official attack artifact.

    Node-injection attacks such as SBA append trigger nodes to ``poison_x`` and
    ``poison_train_edge_index`` while the upstream implementation may keep
    ``poison_y`` at the original graph size.  GSDD uses masks sized to the
    poisoned graph, so labels are padded with ``-1`` for injected, unlabeled
    nodes.  Every supervised index is then checked to ensure it references a
    real label.
    """

    out = dict(bundle)
    if "poison_x" not in out or "poison_y" not in out:
        raise KeyError("artifact must contain poison_x and poison_y")

    poison_x = out["poison_x"]
    poison_y = out["poison_y"]
    if not isinstance(poison_x, torch.Tensor) or poison_x.ndim != 2:
        raise ValueError("poison_x must be a rank-2 tensor")
    if not isinstance(poison_y, torch.Tensor):
        poison_y = torch.as_tensor(poison_y)
    poison_y = poison_y.detach().cpu().long().reshape(-1)

    poison_num_nodes = int(poison_x.shape[0])
    label_num_nodes = int(poison_y.numel())
    if label_num_nodes > poison_num_nodes:
        raise ValueError(
            f"poison_y has {label_num_nodes} rows but poison_x has only "
            f"{poison_num_nodes} nodes"
        )
    if label_num_nodes < poison_num_nodes:
        padded = torch.full((poison_num_nodes,), -1, dtype=torch.long)
        padded[:label_num_nodes] = poison_y
        poison_y = padded

    out["poison_y"] = poison_y
    out["poison_num_nodes"] = poison_num_nodes
    out["label_num_nodes_before_padding"] = label_num_nodes
    out["injected_node_idx"] = torch.arange(
        label_num_nodes, poison_num_nodes, dtype=torch.long
    )

    edge_index = out.get("poison_train_edge_index")
    if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("poison_train_edge_index must have shape [2, num_edges]")
    edge_index = edge_index.detach().cpu().long()
    if edge_index.numel():
        edge_min = int(edge_index.min().item())
        edge_max = int(edge_index.max().item())
        if edge_min < 0 or edge_max >= poison_num_nodes:
            raise ValueError(
                "poison_train_edge_index references a node outside poison_x: "
                f"min={edge_min}, max={edge_max}, nodes={poison_num_nodes}"
            )
    out["poison_train_edge_index"] = edge_index

    for key in _INDEX_KEYS:
        if key not in out:
            continue
        idx = _as_1d_long(out[key], key)
        if idx.numel() and int(idx.max().item()) >= poison_num_nodes:
            raise ValueError(
                f"{key} references node {int(idx.max().item())}, but the "
                f"poisoned graph has {poison_num_nodes} nodes"
            )
        out[key] = idx

    for key in _LABELED_INDEX_KEYS:
        idx = out.get(key)
        if not isinstance(idx, torch.Tensor) or idx.numel() == 0:
            continue
        unlabeled = idx[poison_y[idx] < 0]
        if unlabeled.numel():
            preview = unlabeled[:10].tolist()
            raise ValueError(
                f"{key} contains injected/unlabeled nodes {preview}; injected "
                "trigger nodes must not be used as supervised examples"
            )

    return out
