from __future__ import annotations

import pickle
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


PLANETOID_BASE_URL = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"
PLANETOID_OBJECTS = ("x", "tx", "allx", "y", "ty", "ally", "graph")


@dataclass
class GraphData:
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_original_nodes: int
    poison_mask: torch.Tensor
    trigger_feature_indices: torch.Tensor | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_features(self) -> int:
        return int(self.x.size(1))

    @property
    def num_classes(self) -> int:
        valid = self.y[self.y >= 0]
        return int(valid.max().item() + 1)

    def to(self, device: torch.device) -> "GraphData":
        return replace(
            self,
            x=self.x.to(device),
            y=self.y.to(device),
            edge_index=self.edge_index.to(device),
            train_mask=self.train_mask.to(device),
            val_mask=self.val_mask.to(device),
            test_mask=self.test_mask.to(device),
            poison_mask=self.poison_mask.to(device),
            trigger_feature_indices=(
                None if self.trigger_feature_indices is None else self.trigger_feature_indices.to(device)
            ),
        )


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "GSDD-v0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as output:
        output.write(response.read())


def _parse_index_file(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as handle:
        return [int(line.strip()) for line in handle if line.strip()]


def _row_normalize(features: sp.spmatrix) -> sp.csr_matrix:
    rowsum = np.asarray(features.sum(1)).reshape(-1)
    inv = np.zeros_like(rowsum, dtype=np.float64)
    nonzero = rowsum != 0
    inv[nonzero] = 1.0 / rowsum[nonzero]
    return sp.diags(inv).dot(features).tocsr()


def _edge_index_from_graph_dict(graph: dict[int, list[int]], num_nodes: int) -> torch.Tensor:
    rows: list[int] = []
    cols: list[int] = []
    for source, neighbors in graph.items():
        for target in neighbors:
            rows.append(int(source))
            cols.append(int(target))
    adjacency = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(num_nodes, num_nodes),
    )
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    coo = adjacency.tocoo()
    return torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)


def load_cora(root: str | Path, normalize_features: bool = True) -> GraphData:
    raw_dir = Path(root) / "Planetoid" / "Cora" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    required = [f"ind.cora.{name}" for name in PLANETOID_OBJECTS] + ["ind.cora.test.index"]
    for filename in required:
        path = raw_dir / filename
        if not path.exists():
            _download(f"{PLANETOID_BASE_URL}/{filename}", path)

    objects = []
    for name in PLANETOID_OBJECTS:
        with (raw_dir / f"ind.cora.{name}").open("rb") as handle:
            objects.append(pickle.load(handle, encoding="latin1"))
    x, tx, allx, y, ty, ally, graph = objects

    test_idx_reorder = np.array(_parse_index_file(raw_dir / "ind.cora.test.index"), dtype=np.int64)
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    if normalize_features:
        features = _row_normalize(features)
    features = np.asarray(features.todense(), dtype=np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = labels.argmax(axis=1).astype(np.int64)

    num_nodes = features.shape[0]
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[: len(y)] = True
    val_mask[len(y) : len(y) + 500] = True
    test_mask[test_idx_range] = True

    return GraphData(
        x=torch.from_numpy(features),
        y=torch.from_numpy(labels),
        edge_index=_edge_index_from_graph_dict(graph, num_nodes),
        train_mask=torch.from_numpy(train_mask),
        val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask),
        num_original_nodes=num_nodes,
        poison_mask=torch.zeros(num_nodes, dtype=torch.bool),
    )


def load_synthetic(
    seed: int,
    num_nodes: int = 240,
    num_features: int = 48,
    num_classes: int = 3,
) -> GraphData:
    rng = np.random.default_rng(seed)
    class_sizes = np.full(num_classes, num_nodes // num_classes, dtype=int)
    class_sizes[: num_nodes % num_classes] += 1
    labels = np.concatenate([np.full(size, c, dtype=np.int64) for c, size in enumerate(class_sizes)])
    rng.shuffle(labels)

    features = rng.uniform(0.0, 0.03, size=(num_nodes, num_features)).astype(np.float32)
    block = max(2, num_features // (2 * num_classes))
    for node, label in enumerate(labels):
        start = int(label) * block
        end = min(num_features, start + block)
        features[node, start:end] += rng.uniform(0.8, 1.2, size=end - start)
    features += rng.normal(0.0, 0.01, size=features.shape).astype(np.float32)
    features = np.clip(features, 0.0, None)
    denom = features.sum(axis=1, keepdims=True)
    features = features / np.clip(denom, 1e-8, None)

    rows: list[int] = []
    cols: list[int] = []
    p_in, p_out = 0.16, 0.004
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            probability = p_in if labels[i] == labels[j] else p_out
            if rng.random() < probability:
                rows.extend([i, j])
                cols.extend([j, i])

    degree = np.bincount(rows, minlength=num_nodes)
    for node in np.flatnonzero(degree == 0):
        candidates = np.flatnonzero(labels == labels[node])
        candidates = candidates[candidates != node]
        neighbor = int(rng.choice(candidates))
        rows.extend([node, neighbor])
        cols.extend([neighbor, node])

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    for c in range(num_classes):
        indices = np.flatnonzero(labels == c)
        rng.shuffle(indices)
        n_train = max(10, int(0.25 * len(indices)))
        n_val = max(8, int(0.15 * len(indices)))
        train_mask[indices[:n_train]] = True
        val_mask[indices[n_train : n_train + n_val]] = True
        test_mask[indices[n_train + n_val :]] = True

    return GraphData(
        x=torch.from_numpy(features),
        y=torch.from_numpy(labels),
        edge_index=torch.tensor([rows, cols], dtype=torch.long),
        train_mask=torch.from_numpy(train_mask),
        val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask),
        num_original_nodes=num_nodes,
        poison_mask=torch.zeros(num_nodes, dtype=torch.bool),
    )


def load_dataset(
    name: str,
    root: str | Path,
    normalize_features: bool,
    seed: int,
    synthetic_nodes: int,
    synthetic_features: int,
    synthetic_classes: int,
) -> GraphData:
    normalized = name.lower().strip()
    if normalized == "cora":
        return load_cora(root, normalize_features=normalize_features)
    if normalized == "synthetic":
        return load_synthetic(
            seed=seed,
            num_nodes=synthetic_nodes,
            num_features=synthetic_features,
            num_classes=synthetic_classes,
        )
    raise ValueError(f"Unsupported dataset: {name}. GSDD-v0.1 supports Cora and synthetic.")


def _select_trigger_features(
    data: GraphData,
    target_class: int,
    feature_count: int,
) -> torch.Tensor:
    del target_class  # Trigger signature is intentionally target-independent.
    global_mean = data.x[data.train_mask].mean(dim=0)
    count = min(max(1, feature_count), data.num_features)
    # Rare coordinates create a reproducible shortcut without imitating target semantics.
    return torch.topk(global_mean, k=count, largest=False).indices


def _select_victims(
    data: GraphData,
    target_class: int,
    selection_method: str,
    count: int,
    seed: int,
) -> torch.Tensor:
    if selection_method == "dirty_label":
        candidates = torch.where(data.train_mask & (data.y != target_class))[0]
    elif selection_method == "clean_label":
        candidates = torch.where(data.train_mask & (data.y == target_class))[0]
    else:
        raise ValueError("selection_method must be 'dirty_label' or 'clean_label'")
    if candidates.numel() == 0:
        raise ValueError("No candidate victim nodes are available")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(candidates.numel(), generator=generator)
    return candidates[order[: min(count, candidates.numel())]]


def _make_trigger_feature(
    data: GraphData,
    target_class: int,
    trigger_indices: torch.Tensor,
    value: float,
) -> torch.Tensor:
    del target_class
    prototype = torch.zeros(data.num_features, dtype=data.x.dtype)
    prototype[trigger_indices] = value
    return prototype


def attach_trigger(
    data: GraphData,
    victims: torch.Tensor,
    target_class: int,
    trigger_size: int,
    trigger_feature_indices: torch.Tensor,
    trigger_feature_value: float,
    stamp_victim_features: bool,
    relabel_victims: bool,
    mark_poison: bool,
) -> GraphData:
    if trigger_size < 0:
        raise ValueError("trigger_size must be non-negative")

    x = data.x.clone()
    y = data.y.clone()
    edge_index = data.edge_index.clone()
    train_mask = data.train_mask.clone()
    val_mask = data.val_mask.clone()
    test_mask = data.test_mask.clone()
    poison_mask = data.poison_mask.clone()

    trigger_feature = _make_trigger_feature(
        data,
        target_class=target_class,
        trigger_indices=trigger_feature_indices,
        value=trigger_feature_value,
    )

    new_features: list[torch.Tensor] = []
    new_edges_src: list[int] = []
    new_edges_dst: list[int] = []
    next_node = data.num_nodes

    for victim_tensor in victims:
        victim = int(victim_tensor.item())
        if stamp_victim_features:
            x[victim, trigger_feature_indices] = trigger_feature_value
        if relabel_victims:
            y[victim] = target_class
        if mark_poison:
            poison_mask[victim] = True

        trigger_nodes = list(range(next_node, next_node + trigger_size))
        next_node += trigger_size
        for _ in trigger_nodes:
            new_features.append(trigger_feature.clone())

        for trigger_node in trigger_nodes:
            new_edges_src.extend([victim, trigger_node])
            new_edges_dst.extend([trigger_node, victim])
        for i in range(trigger_size):
            for j in range(i + 1, trigger_size):
                u, v = trigger_nodes[i], trigger_nodes[j]
                new_edges_src.extend([u, v])
                new_edges_dst.extend([v, u])

    if new_features:
        x = torch.cat([x, torch.stack(new_features)], dim=0)
        y = torch.cat([y, torch.full((len(new_features),), -1, dtype=y.dtype)], dim=0)
        train_mask = torch.cat([train_mask, torch.zeros(len(new_features), dtype=torch.bool)])
        val_mask = torch.cat([val_mask, torch.zeros(len(new_features), dtype=torch.bool)])
        test_mask = torch.cat([test_mask, torch.zeros(len(new_features), dtype=torch.bool)])
        poison_mask = torch.cat([poison_mask, torch.zeros(len(new_features), dtype=torch.bool)])
        additional_edges = torch.tensor([new_edges_src, new_edges_dst], dtype=torch.long)
        edge_index = torch.cat([edge_index, additional_edges], dim=1)

    return GraphData(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_original_nodes=data.num_original_nodes,
        poison_mask=poison_mask,
        trigger_feature_indices=trigger_feature_indices.clone(),
    )


def poison_training_graph(
    data: GraphData,
    target_class: int,
    selection_method: str,
    ablation_mode: str,
    poison_count: int,
    trigger_size: int,
    trigger_feature_count: int,
    trigger_feature_value: float,
    stamp_victim_features: bool,
    seed: int,
) -> tuple[GraphData, torch.Tensor]:
    """Create a controlled training-graph intervention.

    The same victim-selection rule is used in all four v0.4 ablation modes:

    - ``full``: attach the trigger and apply dirty-label relabeling when requested
    - ``label_only``: relabel selected nodes but do not alter graph/features
    - ``trigger_only``: attach the trigger but preserve the original labels
    - ``none``: mark the selected nodes for a negative-control audit only

    Marking the selected nodes in ``poison_mask`` for every mode lets the
    diagnostic compare the same kind of selected-node subset even when no
    intervention is applied. In the ``none`` condition, AUROC should be near
    chance; this is a falsification control, not a real poisoning attack.
    """
    mode = ablation_mode.strip().lower()
    allowed = {"full", "label_only", "trigger_only", "none"}
    if mode not in allowed:
        raise ValueError(f"ablation_mode must be one of {sorted(allowed)}, got {ablation_mode!r}")

    trigger_indices = _select_trigger_features(data, target_class, trigger_feature_count)
    victims = _select_victims(data, target_class, selection_method, poison_count, seed)

    use_trigger = mode in {"full", "trigger_only"}
    use_label = mode in {"full", "label_only"} and selection_method == "dirty_label"

    intervened = attach_trigger(
        data=data,
        victims=victims,
        target_class=target_class,
        trigger_size=trigger_size if use_trigger else 0,
        trigger_feature_indices=trigger_indices,
        trigger_feature_value=trigger_feature_value,
        stamp_victim_features=stamp_victim_features if use_trigger else False,
        relabel_victims=use_label,
        mark_poison=True,
    )
    return intervened, victims


def make_triggered_test_graph(
    poisoned_training_graph: GraphData,
    target_class: int,
    trigger_size: int,
    trigger_feature_value: float,
    stamp_victim_features: bool,
    test_victim_count: int,
    seed: int,
) -> tuple[GraphData, torch.Tensor]:
    if poisoned_training_graph.trigger_feature_indices is None:
        raise ValueError("The poisoned graph does not contain trigger feature indices")
    candidates = torch.where(
        poisoned_training_graph.test_mask
        & (poisoned_training_graph.y != target_class)
        & (torch.arange(poisoned_training_graph.num_nodes) < poisoned_training_graph.num_original_nodes)
    )[0]
    generator = torch.Generator().manual_seed(seed + 100003)
    order = torch.randperm(candidates.numel(), generator=generator)
    victims = candidates[order[: min(test_victim_count, candidates.numel())]]
    triggered = attach_trigger(
        data=poisoned_training_graph,
        victims=victims,
        target_class=target_class,
        trigger_size=trigger_size,
        trigger_feature_indices=poisoned_training_graph.trigger_feature_indices,
        trigger_feature_value=trigger_feature_value,
        stamp_victim_features=stamp_victim_features,
        relabel_victims=False,
        mark_poison=False,
    )
    return triggered, victims
