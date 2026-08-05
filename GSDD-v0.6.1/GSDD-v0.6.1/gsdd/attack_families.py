from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .config import Config
from .data import GraphData
from .graph_ops import build_normalized_adjacency
from .models import SupervisedGCN
from .train import accuracy, train_supervised
from .utils import log, set_seed


ATTACK_FAMILIES = (
    "fixed_rare_clique",
    "ugba_style_adaptive",
    "dpgba_style_distribution",
)
MOTIFS = ("clique", "star", "chain", "cycle")


def _normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = torch.clamp(x, min=0.0)
    return x / x.sum(dim=-1, keepdim=True).clamp_min(eps)


def _neighbor_mean(data: GraphData, device: torch.device) -> torch.Tensor:
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    src, dst = edge_index[0], edge_index[1]
    result = torch.zeros_like(x)
    result.index_add_(0, dst, x[src])
    degree = torch.zeros(x.size(0), dtype=x.dtype, device=device)
    degree.index_add_(0, dst, torch.ones(dst.numel(), dtype=x.dtype, device=device))
    isolated = degree == 0
    result = result / degree.clamp_min(1.0).unsqueeze(1)
    result[isolated] = x[isolated]
    return result


def select_victims(
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


def select_test_victims(
    data: GraphData,
    target_class: int,
    count: int,
    seed: int,
) -> torch.Tensor:
    node_ids = torch.arange(data.num_nodes)
    candidates = torch.where(
        data.test_mask
        & (data.y != target_class)
        & (node_ids < data.num_original_nodes)
    )[0]
    if candidates.numel() == 0:
        raise ValueError("No non-target test victims are available")
    generator = torch.Generator().manual_seed(seed + 100003)
    order = torch.randperm(candidates.numel(), generator=generator)
    return candidates[order[: min(count, candidates.numel())]]


def motif_pairs(trigger_size: int, motif: str) -> list[tuple[int, int]]:
    if trigger_size <= 1:
        return []
    motif = motif.lower().strip()
    pairs: set[tuple[int, int]] = set()
    if motif == "clique":
        for i in range(trigger_size):
            for j in range(i + 1, trigger_size):
                pairs.add((i, j))
    elif motif == "star":
        for j in range(1, trigger_size):
            pairs.add((0, j))
    elif motif == "chain":
        for i in range(trigger_size - 1):
            pairs.add((i, i + 1))
    elif motif == "cycle":
        for i in range(trigger_size):
            a, b = i, (i + 1) % trigger_size
            if a != b:
                pairs.add(tuple(sorted((a, b))))
    else:
        raise ValueError(f"Unsupported trigger motif: {motif}")
    return sorted(pairs)


def _attack_edge_index(
    base_edge_index: torch.Tensor,
    base_num_nodes: int,
    victims: torch.Tensor,
    trigger_size: int,
    motif: str,
    device: torch.device,
) -> torch.Tensor:
    src: list[int] = []
    dst: list[int] = []
    pairs = motif_pairs(trigger_size, motif)
    next_node = base_num_nodes
    for victim_tensor in victims:
        victim = int(victim_tensor.item())
        local_nodes = list(range(next_node, next_node + trigger_size))
        next_node += trigger_size
        for node in local_nodes:
            src.extend([victim, node])
            dst.extend([node, victim])
        for a, b in pairs:
            u, v = local_nodes[a], local_nodes[b]
            src.extend([u, v])
            dst.extend([v, u])
    if not src:
        return base_edge_index.to(device)
    additional = torch.tensor([src, dst], dtype=torch.long, device=device)
    return torch.cat([base_edge_index.to(device), additional], dim=1)


def _augment_x_differentiable(
    base_x: torch.Tensor,
    victims: torch.Tensor,
    trigger_features: torch.Tensor,
    stamp_strength: float,
) -> torch.Tensor:
    # trigger_features: [victims, trigger_size, features]
    x = base_x.clone()
    if stamp_strength > 0:
        mean_trigger = trigger_features.mean(dim=1)
        blended = (1.0 - stamp_strength) * x[victims] + stamp_strength * mean_trigger
        x = x.index_copy(0, victims, _normalize_rows(blended))
    return torch.cat([x, trigger_features.reshape(-1, base_x.size(1))], dim=0)


class SparseAdaptiveGenerator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        trigger_size: int,
        topk_features: int,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.trigger_size = trigger_size
        self.topk_features = min(max(1, topk_features), feature_dim)
        self.network = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, trigger_size * feature_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        logits = self.network(context).view(-1, self.trigger_size, self.feature_dim)
        indices = torch.topk(logits, k=self.topk_features, dim=-1).indices
        mask = torch.zeros_like(logits).scatter_(-1, indices, 1.0)
        values = F.softplus(logits) * mask
        return _normalize_rows(values)


class PrototypeMixtureGenerator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        trigger_size: int,
        prototype_count: int,
    ) -> None:
        super().__init__()
        self.trigger_size = trigger_size
        self.prototype_count = prototype_count
        self.network = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, trigger_size * prototype_count),
        )

    def forward(self, context: torch.Tensor, prototype_bank: torch.Tensor) -> torch.Tensor:
        logits = self.network(context).view(-1, self.trigger_size, self.prototype_count)
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("mtp,pd->mtd", weights, prototype_bank)


@dataclass
class AttackPlan:
    family: str
    victims: torch.Tensor
    target_class: int
    trigger_size: int
    motif: str
    stamp_strength: float
    fixed_features: torch.Tensor | None = None
    generator: nn.Module | None = None
    prototype_bank: torch.Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @torch.no_grad()
    def generate(
        self,
        context_graph: GraphData,
        victims: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if self.fixed_features is not None:
            return self.fixed_features.to(device).unsqueeze(0).expand(
                victims.numel(), -1, -1
            ).clone()
        if self.generator is None:
            raise RuntimeError("Attack plan does not contain a trigger generator")
        self.generator.eval()
        neighbor_mean = _neighbor_mean(context_graph, device)
        ids = victims.to(device)
        context = torch.cat(
            [context_graph.x.to(device)[ids], neighbor_mean[ids]], dim=1
        )
        if isinstance(self.generator, PrototypeMixtureGenerator):
            if self.prototype_bank is None:
                raise RuntimeError("Prototype mixture plan is missing its prototype bank")
            return self.generator(context, self.prototype_bank.to(device))
        return self.generator(context)


def apply_attack_plan(
    base_graph: GraphData,
    context_graph: GraphData,
    victims: torch.Tensor,
    plan: AttackPlan,
    relabel_victims: bool,
    mark_poison: bool,
    device: torch.device,
) -> GraphData:
    trigger_features = plan.generate(context_graph, victims, device).detach().cpu()
    x = base_graph.x.clone()
    y = base_graph.y.clone()
    train_mask = base_graph.train_mask.clone()
    val_mask = base_graph.val_mask.clone()
    test_mask = base_graph.test_mask.clone()
    poison_mask = base_graph.poison_mask.clone()

    if plan.stamp_strength > 0:
        mean_trigger = trigger_features.mean(dim=1)
        blended = (
            (1.0 - plan.stamp_strength) * x[victims]
            + plan.stamp_strength * mean_trigger
        )
        x[victims] = _normalize_rows(blended)
    if relabel_victims:
        y[victims] = plan.target_class
    if mark_poison:
        poison_mask[victims] = True

    edge_index = _attack_edge_index(
        base_graph.edge_index,
        base_graph.num_nodes,
        victims,
        plan.trigger_size,
        plan.motif,
        torch.device("cpu"),
    )
    new_count = victims.numel() * plan.trigger_size
    if new_count:
        x = torch.cat([x, trigger_features.reshape(new_count, x.size(1))], dim=0)
        y = torch.cat([y, torch.full((new_count,), -1, dtype=y.dtype)], dim=0)
        train_mask = torch.cat([train_mask, torch.zeros(new_count, dtype=torch.bool)])
        val_mask = torch.cat([val_mask, torch.zeros(new_count, dtype=torch.bool)])
        test_mask = torch.cat([test_mask, torch.zeros(new_count, dtype=torch.bool)])
        poison_mask = torch.cat([poison_mask, torch.zeros(new_count, dtype=torch.bool)])

    return GraphData(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_original_nodes=base_graph.num_original_nodes,
        poison_mask=poison_mask,
        trigger_feature_indices=None,
    )


def _fixed_plan(data: GraphData, cfg: Config, victims: torch.Tensor) -> AttackPlan:
    global_mean = data.x[data.train_mask].mean(dim=0)
    count = min(max(1, cfg.attack.trigger_feature_count), data.num_features)
    rare = torch.topk(global_mean, k=count, largest=False).indices
    prototype = torch.zeros(data.num_features, dtype=data.x.dtype)
    prototype[rare] = cfg.attack.trigger_feature_value
    fixed = prototype.unsqueeze(0).repeat(cfg.attack.trigger_size, 1)
    stamp = 1.0 if cfg.attack.stamp_victim_features else 0.0
    return AttackPlan(
        family="fixed_rare_clique",
        victims=victims,
        target_class=cfg.attack.target_class,
        trigger_size=cfg.attack.trigger_size,
        motif="clique",
        stamp_strength=stamp,
        fixed_features=fixed,
        diagnostics={
            "generator_kind": "fixed_rare_coordinates",
            "rare_feature_count": int(count),
        },
    )


def _initial_generator(
    family: str,
    data: GraphData,
    cfg: Config,
    device: torch.device,
) -> tuple[nn.Module, torch.Tensor | None]:
    if family == "ugba_style_adaptive":
        generator = SparseAdaptiveGenerator(
            data.num_features,
            cfg.attack.generator_hidden_dim,
            cfg.attack.trigger_size,
            cfg.attack.generator_topk_features,
        ).to(device)
        return generator, None
    if family == "dpgba_style_distribution":
        prototypes = data.x[data.train_mask & (data.y == cfg.attack.target_class)]
        if prototypes.numel() == 0:
            raise ValueError("No target-class training prototypes are available")
        count = min(cfg.attack.generator_prototype_count, prototypes.size(0))
        # Deterministic spread through the target-class prototype bank.
        positions = torch.linspace(0, prototypes.size(0) - 1, count).round().long()
        bank = prototypes[positions].to(device)
        generator = PrototypeMixtureGenerator(
            data.num_features,
            cfg.attack.generator_hidden_dim,
            cfg.attack.trigger_size,
            count,
        ).to(device)
        return generator, bank
    raise ValueError(f"Unknown learned attack family: {family}")


def _generate_with_module(
    generator: nn.Module,
    prototype_bank: torch.Tensor | None,
    context: torch.Tensor,
) -> torch.Tensor:
    if isinstance(generator, PrototypeMixtureGenerator):
        if prototype_bank is None:
            raise RuntimeError("Prototype mixture generator requires a bank")
        return generator(context, prototype_bank)
    return generator(context)


def _train_provisional_surrogate(
    clean_graph: GraphData,
    victims: torch.Tensor,
    generator: nn.Module,
    prototype_bank: torch.Tensor | None,
    cfg: Config,
    device: torch.device,
    motif: str,
) -> tuple[SupervisedGCN, dict[str, float]]:
    generator.eval()
    neighbor_mean = _neighbor_mean(clean_graph, device)
    ids = victims.to(device)
    context = torch.cat([clean_graph.x.to(device)[ids], neighbor_mean[ids]], dim=1)
    with torch.no_grad():
        trigger_features = _generate_with_module(generator, prototype_bank, context)
    x_aug = _augment_x_differentiable(
        clean_graph.x.to(device), ids, trigger_features, cfg.attack.victim_stamp_strength
    )
    edge_index = _attack_edge_index(
        clean_graph.edge_index,
        clean_graph.num_nodes,
        victims,
        cfg.attack.trigger_size,
        motif,
        device,
    )
    adjacency = build_normalized_adjacency(
        edge_index, x_aug.size(0), device=device, add_self_loops=True
    )
    y = torch.cat(
        [
            clean_graph.y.to(device).clone(),
            torch.full(
                (victims.numel() * cfg.attack.trigger_size,),
                -1,
                dtype=clean_graph.y.dtype,
                device=device,
            ),
        ]
    )
    if cfg.attack.selection_method == "dirty_label":
        y[ids] = cfg.attack.target_class
    train_mask = torch.cat(
        [
            clean_graph.train_mask.to(device),
            torch.zeros(victims.numel() * cfg.attack.trigger_size, dtype=torch.bool, device=device),
        ]
    )
    val_mask = torch.cat(
        [
            clean_graph.val_mask.to(device),
            torch.zeros(victims.numel() * cfg.attack.trigger_size, dtype=torch.bool, device=device),
        ]
    )
    set_seed(cfg.experiment.seed + 4401)
    surrogate = SupervisedGCN(
        in_features=clean_graph.num_features,
        hidden_dim=cfg.attack.surrogate_hidden_dim,
        num_classes=clean_graph.num_classes,
        dropout=cfg.model.dropout,
    ).to(device)
    result = train_supervised(
        surrogate,
        x_aug,
        y,
        adjacency,
        train_mask,
        val_mask,
        cfg.attack.surrogate_epochs,
        cfg.model.learning_rate,
        cfg.model.weight_decay,
        cfg.attack.surrogate_patience,
        verbose=False,
    )
    surrogate.eval()
    with torch.no_grad():
        logits, _ = surrogate(x_aug, adjacency)
        target_rate = float(
            (logits[ids].argmax(dim=1) == cfg.attack.target_class).float().mean().item()
        )
        clean_test_mask = torch.cat(
            [
                clean_graph.test_mask.to(device),
                torch.zeros(victims.numel() * cfg.attack.trigger_size, dtype=torch.bool, device=device),
            ]
        )
        clean_accuracy = accuracy(logits, y, clean_test_mask)
    return surrogate, {
        "provisional_surrogate_best_epoch": int(result.best_epoch),
        "provisional_surrogate_victim_target_rate": target_rate,
        "provisional_surrogate_test_accuracy": clean_accuracy,
    }


def _distribution_loss(
    generated: torch.Tensor,
    target_features: torch.Tensor,
) -> torch.Tensor:
    flat = generated.reshape(-1, generated.size(-1))
    mean_loss = F.mse_loss(flat.mean(dim=0), target_features.mean(dim=0))
    var_loss = F.mse_loss(flat.var(dim=0, unbiased=False), target_features.var(dim=0, unbiased=False))
    distances = torch.cdist(flat, target_features)
    nearest_loss = distances.min(dim=1).values.pow(2).mean()
    return mean_loss + var_loss + 0.1 * nearest_loss


def _train_generator(
    family: str,
    clean_graph: GraphData,
    victims: torch.Tensor,
    generator: nn.Module,
    prototype_bank: torch.Tensor | None,
    surrogate: SupervisedGCN,
    cfg: Config,
    device: torch.device,
    training_motif: str,
) -> dict[str, Any]:
    ids = victims.to(device)
    base_x = clean_graph.x.to(device)
    neighbor_mean = _neighbor_mean(clean_graph, device)
    context = torch.cat([base_x[ids], neighbor_mean[ids]], dim=1)
    edge_index = _attack_edge_index(
        clean_graph.edge_index,
        clean_graph.num_nodes,
        victims,
        cfg.attack.trigger_size,
        training_motif,
        device,
    )
    total_nodes = clean_graph.num_nodes + victims.numel() * cfg.attack.trigger_size
    adjacency = build_normalized_adjacency(
        edge_index, total_nodes, device=device, add_self_loops=True
    )
    target_features = clean_graph.x[
        clean_graph.train_mask & (clean_graph.y == cfg.attack.target_class)
    ].to(device)

    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    surrogate.eval()
    optimizer = torch.optim.Adam(
        generator.parameters(), lr=cfg.attack.generator_learning_rate
    )
    best_state = copy.deepcopy(generator.state_dict())
    best_value = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.attack.generator_epochs + 1):
        generator.train()
        optimizer.zero_grad(set_to_none=True)
        generated = _generate_with_module(generator, prototype_bank, context)
        x_aug = _augment_x_differentiable(
            base_x, ids, generated, cfg.attack.victim_stamp_strength
        )
        logits, _ = surrogate(x_aug, adjacency)
        target = torch.full(
            (ids.numel(),), cfg.attack.target_class, dtype=torch.long, device=device
        )
        target_loss = F.cross_entropy(logits[ids], target)
        generated_mean = generated.mean(dim=1)
        homophily = (1.0 - F.cosine_similarity(generated_mean, neighbor_mean[ids], dim=1)).mean()
        entropy = -(
            generated.clamp_min(1e-8) * generated.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        dist_loss = _distribution_loss(generated, target_features)
        if family == "ugba_style_adaptive":
            distribution_coefficient = 0.05 * cfg.attack.distribution_weight
        else:
            distribution_coefficient = cfg.attack.distribution_weight
        loss = (
            target_loss
            + cfg.attack.homophily_weight * homophily
            + distribution_coefficient * dist_loss
            + cfg.attack.sparsity_weight * entropy
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
        optimizer.step()

        value = float(loss.item())
        history.append(
            {
                "epoch": float(epoch),
                "loss": value,
                "target_loss": float(target_loss.item()),
                "homophily_loss": float(homophily.item()),
                "distribution_loss": float(dist_loss.item()),
                "entropy": float(entropy.item()),
            }
        )
        if value < best_value - 1e-6:
            best_value = value
            best_state = copy.deepcopy(generator.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if cfg.output.verbose and (epoch == 1 or epoch % 25 == 0):
            log(
                f"[AttackGen:{family}] epoch={epoch:04d} loss={value:.4f} "
                f"target={target_loss.item():.4f} stealth={dist_loss.item():.4f}",
                True,
            )
        if stale >= cfg.attack.generator_patience:
            break

    generator.load_state_dict(best_state)
    generator.eval()
    with torch.no_grad():
        generated = _generate_with_module(generator, prototype_bank, context)
        x_aug = _augment_x_differentiable(
            base_x, ids, generated, cfg.attack.victim_stamp_strength
        )
        logits, _ = surrogate(x_aug, adjacency)
        target_probability = torch.softmax(logits[ids], dim=1)[:, cfg.attack.target_class]
        target_rate = float((logits[ids].argmax(dim=1) == cfg.attack.target_class).float().mean().item())
        cosine_neighbor = float(
            F.cosine_similarity(generated.mean(dim=1), neighbor_mean[ids], dim=1).mean().item()
        )
        distribution_value = float(_distribution_loss(generated, target_features).item())
        nonzero = float((generated > 1e-8).float().sum(dim=-1).mean().item())
    return {
        "generator_best_epoch": int(best_epoch),
        "generator_best_loss": float(best_value),
        "generator_surrogate_target_rate": target_rate,
        "generator_surrogate_target_probability": float(target_probability.mean().item()),
        "generated_neighbor_cosine": cosine_neighbor,
        "generated_distribution_loss": distribution_value,
        "generated_mean_nonzero_features": nonzero,
        "history": history,
    }


def _select_topology(
    clean_graph: GraphData,
    victims: torch.Tensor,
    generator: nn.Module,
    prototype_bank: torch.Tensor | None,
    surrogate: SupervisedGCN,
    cfg: Config,
    device: torch.device,
) -> tuple[str, list[dict[str, float]]]:
    generator.eval()
    ids = victims.to(device)
    neighbor_mean = _neighbor_mean(clean_graph, device)
    context = torch.cat(
        [clean_graph.x.to(device)[ids], neighbor_mean[ids]], dim=1
    )
    with torch.no_grad():
        generated = _generate_with_module(generator, prototype_bank, context)
        x_aug = _augment_x_differentiable(
            clean_graph.x.to(device), ids, generated, cfg.attack.victim_stamp_strength
        )
    rows: list[dict[str, float]] = []
    best_motif = "clique"
    best_score = -float("inf")
    for motif in MOTIFS:
        edge_index = _attack_edge_index(
            clean_graph.edge_index,
            clean_graph.num_nodes,
            victims,
            cfg.attack.trigger_size,
            motif,
            device,
        )
        adjacency = build_normalized_adjacency(
            edge_index, x_aug.size(0), device=device, add_self_loops=True
        )
        with torch.no_grad():
            logits, _ = surrogate(x_aug, adjacency)
            probability = float(
                torch.softmax(logits[ids], dim=1)[:, cfg.attack.target_class].mean().item()
            )
        possible = max(1, cfg.attack.trigger_size * (cfg.attack.trigger_size - 1) // 2)
        density = len(motif_pairs(cfg.attack.trigger_size, motif)) / possible
        score = probability - cfg.attack.topology_density_weight * density
        rows.append(
            {
                "motif": motif,
                "target_probability": probability,
                "internal_density": float(density),
                "selection_score": float(score),
            }
        )
        if score > best_score:
            best_score = score
            best_motif = motif
    return best_motif, rows


def build_attack_plan(
    clean_graph: GraphData,
    cfg: Config,
    device: torch.device,
) -> AttackPlan:
    family = cfg.attack.family.lower().strip()
    if family not in ATTACK_FAMILIES:
        raise ValueError(
            f"attack.family must be one of {ATTACK_FAMILIES}, got {cfg.attack.family!r}"
        )
    victims = select_victims(
        clean_graph,
        cfg.attack.target_class,
        cfg.attack.selection_method,
        cfg.attack.poison_count,
        cfg.experiment.seed,
    )
    if family == "fixed_rare_clique":
        return _fixed_plan(clean_graph, cfg, victims)

    set_seed(cfg.experiment.seed + 3301)
    generator, prototype_bank = _initial_generator(family, clean_graph, cfg, device)
    training_motif = "clique" if family == "ugba_style_adaptive" else "chain"
    surrogate, surrogate_diag = _train_provisional_surrogate(
        clean_graph,
        victims,
        generator,
        prototype_bank,
        cfg,
        device,
        training_motif,
    )
    generator_diag = _train_generator(
        family,
        clean_graph,
        victims,
        generator,
        prototype_bank,
        surrogate,
        cfg,
        device,
        training_motif,
    )
    motif, topology_rows = _select_topology(
        clean_graph,
        victims,
        generator,
        prototype_bank,
        surrogate,
        cfg,
        device,
    )
    stamp = cfg.attack.victim_stamp_strength if cfg.attack.stamp_victim_features else 0.0
    diagnostics = {
        "generator_kind": type(generator).__name__,
        "training_motif": training_motif,
        "selected_motif": motif,
        "topology_search": topology_rows,
        **surrogate_diag,
        **{key: value for key, value in generator_diag.items() if key != "history"},
        "generator_history": generator_diag["history"],
    }
    return AttackPlan(
        family=family,
        victims=victims,
        target_class=cfg.attack.target_class,
        trigger_size=cfg.attack.trigger_size,
        motif=motif,
        stamp_strength=stamp,
        generator=generator,
        prototype_bank=prototype_bank,
        diagnostics=diagnostics,
    )


def build_paired_graphs(
    clean_graph: GraphData,
    plan: AttackPlan,
    cfg: Config,
    device: torch.device,
) -> dict[str, GraphData]:
    victims = plan.victims
    graphs = {
        "none": apply_attack_plan(
            clean_graph,
            clean_graph,
            victims,
            AttackPlan(
                family=plan.family,
                victims=victims,
                target_class=plan.target_class,
                trigger_size=0,
                motif=plan.motif,
                stamp_strength=0.0,
                fixed_features=torch.empty(0, clean_graph.num_features),
            ),
            relabel_victims=False,
            mark_poison=True,
            device=device,
        ),
        "label_only": apply_attack_plan(
            clean_graph,
            clean_graph,
            victims,
            AttackPlan(
                family=plan.family,
                victims=victims,
                target_class=plan.target_class,
                trigger_size=0,
                motif=plan.motif,
                stamp_strength=0.0,
                fixed_features=torch.empty(0, clean_graph.num_features),
            ),
            relabel_victims=(cfg.attack.selection_method == "dirty_label"),
            mark_poison=True,
            device=device,
        ),
        "trigger_only": apply_attack_plan(
            clean_graph,
            clean_graph,
            victims,
            plan,
            relabel_victims=False,
            mark_poison=True,
            device=device,
        ),
        "full": apply_attack_plan(
            clean_graph,
            clean_graph,
            victims,
            plan,
            relabel_victims=(cfg.attack.selection_method == "dirty_label"),
            mark_poison=True,
            device=device,
        ),
    }
    return graphs


def make_triggered_test_graph_from_plan(
    poisoned_training_graph: GraphData,
    clean_context_graph: GraphData,
    plan: AttackPlan,
    cfg: Config,
    device: torch.device,
) -> tuple[GraphData, torch.Tensor]:
    victims = select_test_victims(
        clean_context_graph,
        cfg.attack.target_class,
        cfg.attack.test_victim_count,
        cfg.experiment.seed,
    )
    triggered = apply_attack_plan(
        poisoned_training_graph,
        clean_context_graph,
        victims,
        plan,
        relabel_victims=False,
        mark_poison=False,
        device=device,
    )
    return triggered, victims
