from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .attack_families import (
    AttackPlan,
    MOTIFS,
    PrototypeMixtureGenerator,
    _attack_edge_index,
    _augment_x_differentiable,
    _distribution_loss,
    _fixed_plan,
    _generate_with_module,
    _initial_generator,
    _neighbor_mean,
    apply_attack_plan,
    build_paired_graphs,
    make_triggered_test_graph_from_plan,
    motif_pairs,
    select_victims,
)
from .config import Config
from .data import GraphData
from .graph_ops import build_normalized_adjacency
from .models import SupervisedGCN
from .train import accuracy, train_supervised
from .utils import log, set_seed


ATTACK_FAMILIES_V065 = (
    "fixed_rare_clique",
    "ugba_style_binding_aware",
    "dpgba_style_binding_aware",
)


@dataclass
class SurrogatePair:
    clean: SupervisedGCN
    poisoned: SupervisedGCN
    initial_state: dict[str, torch.Tensor]
    diagnostics: dict[str, Any]


class ContextBlendWrapper(nn.Module):
    """Blend a learned trigger with the local neighbor prototype.

    The wrapper keeps the learned component differentiable while preventing the
    generated nodes from becoming an unconstrained target-class adversarial
    example. Prototype banks are stored inside the module so AttackPlan.generate
    can call the wrapper with context only.
    """

    def __init__(
        self,
        base: nn.Module,
        feature_dim: int,
        raw_blend: float,
        prototype_bank: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.feature_dim = feature_dim
        self.raw_blend = float(raw_blend)
        if prototype_bank is None:
            self.register_buffer("prototype_bank", torch.empty(0))
        else:
            self.register_buffer("prototype_bank", prototype_bank.detach().clone())

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if isinstance(self.base, PrototypeMixtureGenerator):
            if self.prototype_bank.numel() == 0:
                raise RuntimeError("Prototype wrapper has no prototype bank")
            raw = self.base(context, self.prototype_bank)
        else:
            raw = self.base(context)
        neighbor = context[:, self.feature_dim :].unsqueeze(1).expand_as(raw)
        mixed = self.raw_blend * raw + (1.0 - self.raw_blend) * neighbor
        return mixed.clamp_min(0.0) / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _family_base_name(family: str) -> str:
    if family == "ugba_style_binding_aware":
        return "ugba_style_adaptive"
    if family == "dpgba_style_binding_aware":
        return "dpgba_style_distribution"
    return family


def _select_calibration_nodes(
    data: GraphData,
    target_class: int,
    count: int,
    seed: int,
) -> torch.Tensor:
    # Validation nodes are not used to fit either surrogate. They provide a
    # held-out context set for checking whether the learned trigger generalizes.
    candidates = torch.where(data.val_mask & (data.y != target_class))[0]
    if candidates.numel() < count:
        extra = torch.where(data.test_mask & (data.y != target_class))[0]
        candidates = torch.unique(torch.cat([candidates, extra], dim=0))
    if candidates.numel() == 0:
        raise ValueError("No non-target calibration nodes are available")
    generator = torch.Generator().manual_seed(seed + 17011)
    order = torch.randperm(candidates.numel(), generator=generator)
    return candidates[order[: min(count, candidates.numel())]]


def _build_model(
    data: GraphData,
    cfg: Config,
    device: torch.device,
    initial_state: dict[str, torch.Tensor],
) -> SupervisedGCN:
    model = SupervisedGCN(
        in_features=data.num_features,
        hidden_dim=cfg.attack.surrogate_hidden_dim,
        num_classes=data.num_classes,
        dropout=cfg.model.dropout,
    ).to(device)
    model.load_state_dict(initial_state)
    return model


def _train_surrogate_on_graph(
    graph: GraphData,
    cfg: Config,
    device: torch.device,
    initial_state: dict[str, torch.Tensor],
    seed: int,
) -> tuple[SupervisedGCN, Any]:
    graph_device = graph.to(device)
    adjacency = build_normalized_adjacency(
        graph_device.edge_index,
        graph_device.num_nodes,
        device=device,
        add_self_loops=True,
    )
    model = _build_model(graph, cfg, device, initial_state)
    set_seed(seed)
    result = train_supervised(
        model=model,
        x=graph_device.x,
        y=graph_device.y,
        adjacency_norm=adjacency,
        train_mask=graph_device.train_mask,
        val_mask=graph_device.val_mask,
        epochs=cfg.attack.surrogate_epochs,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        patience=cfg.attack.surrogate_patience,
        verbose=False,
    )
    return model, result


def _make_plan(
    family: str,
    victims: torch.Tensor,
    cfg: Config,
    motif: str,
    generator: nn.Module,
    prototype_bank: torch.Tensor | None,
) -> AttackPlan:
    # Learned families deliberately avoid stamping the original victim feature.
    # The trigger is carried by attached nodes, preventing the stamp itself from
    # becoming a direct feature-space evasion perturbation.
    return AttackPlan(
        family=family,
        victims=victims,
        target_class=cfg.attack.target_class,
        trigger_size=cfg.attack.trigger_size,
        motif=motif,
        stamp_strength=0.0,
        generator=generator,
        prototype_bank=prototype_bank,
    )


def _make_label_only_graph(
    clean_graph: GraphData, victims: torch.Tensor, cfg: Config
) -> GraphData:
    y = clean_graph.y.clone()
    if cfg.attack.selection_method == "dirty_label":
        y[victims] = cfg.attack.target_class
    return GraphData(
        x=clean_graph.x.clone(),
        y=y,
        edge_index=clean_graph.edge_index.clone(),
        train_mask=clean_graph.train_mask.clone(),
        val_mask=clean_graph.val_mask.clone(),
        test_mask=clean_graph.test_mask.clone(),
        num_original_nodes=clean_graph.num_original_nodes,
        poison_mask=clean_graph.poison_mask.clone(),
        trigger_feature_indices=clean_graph.trigger_feature_indices,
    )


def _train_control_surrogates(
    clean_graph: GraphData,
    victims: torch.Tensor,
    cfg: Config,
    device: torch.device,
    initial_state: dict[str, torch.Tensor],
) -> tuple[SupervisedGCN, SupervisedGCN, dict[str, Any]]:
    clean_model, clean_result = _train_surrogate_on_graph(
        clean_graph, cfg, device, initial_state, cfg.experiment.seed + 5101
    )
    label_graph = _make_label_only_graph(clean_graph, victims, cfg)
    label_model, label_result = _train_surrogate_on_graph(
        label_graph, cfg, device, initial_state, cfg.experiment.seed + 5101
    )

    diagnostics: dict[str, Any] = {
        "clean_surrogate_best_epoch": int(clean_result.best_epoch),
        "label_only_surrogate_best_epoch": int(label_result.best_epoch),
    }
    for name, model, graph_data in [
        ("clean", clean_model, clean_graph),
        ("label_only", label_model, label_graph),
    ]:
        graph = graph_data.to(device)
        adjacency = build_normalized_adjacency(
            graph.edge_index, graph.num_nodes, device=device, add_self_loops=True
        )
        model.eval()
        with torch.no_grad():
            logits, _ = model(graph.x, adjacency)
        diagnostics[f"{name}_surrogate_test_accuracy"] = accuracy(
            logits, graph.y, graph.test_mask
        )
    return clean_model, label_model, diagnostics


def _train_poisoned_surrogate(
    clean_graph: GraphData,
    plan: AttackPlan,
    cfg: Config,
    device: torch.device,
    initial_state: dict[str, torch.Tensor],
    round_index: int,
) -> tuple[SupervisedGCN, dict[str, Any]]:
    poison_graph = apply_attack_plan(
        clean_graph,
        clean_graph,
        plan.victims,
        plan,
        relabel_victims=(cfg.attack.selection_method == "dirty_label"),
        mark_poison=True,
        device=device,
    )
    model, result = _train_surrogate_on_graph(
        poison_graph,
        cfg,
        device,
        initial_state,
        cfg.experiment.seed + 5201,
    )
    graph = poison_graph.to(device)
    adjacency = build_normalized_adjacency(
        graph.edge_index, graph.num_nodes, device=device, add_self_loops=True
    )
    ids = plan.victims.to(device)
    model.eval()
    with torch.no_grad():
        logits, _ = model(graph.x, adjacency)
        target_rate = float(
            (logits[ids].argmax(dim=1) == cfg.attack.target_class)
            .float()
            .mean()
            .item()
        )
    return model, {
        f"round_{round_index}_poisoned_surrogate_best_epoch": int(result.best_epoch),
        f"round_{round_index}_poisoned_surrogate_train_target_rate": target_rate,
    }


def _forward_on_triggered_nodes(
    model: SupervisedGCN,
    clean_graph: GraphData,
    attack_nodes: torch.Tensor,
    generated: torch.Tensor,
    motif: str,
    cfg: Config,
    device: torch.device,
) -> torch.Tensor:
    ids = attack_nodes.to(device)
    base_x = clean_graph.x.to(device)
    x_aug = _augment_x_differentiable(base_x, ids, generated, 0.0)
    edge_index = _attack_edge_index(
        clean_graph.edge_index,
        clean_graph.num_nodes,
        attack_nodes,
        cfg.attack.trigger_size,
        motif,
        device,
    )
    adjacency = build_normalized_adjacency(
        edge_index, x_aug.size(0), device=device, add_self_loops=True
    )
    logits, _ = model(x_aug, adjacency)
    return logits[ids]


def _binding_components(
    poison_logits: torch.Tensor,
    clean_logits: torch.Tensor,
    label_logits: torch.Tensor,
    original_labels: torch.Tensor,
    target_class: int,
    cfg: Config,
) -> dict[str, torch.Tensor]:
    target = torch.full_like(original_labels, target_class)
    poison_target = F.cross_entropy(poison_logits, target)
    clean_preserve = 0.5 * (
        F.cross_entropy(clean_logits, original_labels)
        + F.cross_entropy(label_logits, original_labels)
    )

    rows = torch.arange(original_labels.numel(), device=original_labels.device)
    poison_target_margin = poison_logits[:, target_class] - poison_logits[rows, original_labels]
    clean_target_margin = clean_logits[:, target_class] - clean_logits[rows, original_labels]
    label_target_margin = label_logits[:, target_class] - label_logits[rows, original_labels]
    worst_control_margin = torch.maximum(clean_target_margin, label_target_margin)
    differential_margin = poison_target_margin - worst_control_margin
    binding_gap = F.relu(cfg.attack.binding_margin - differential_margin).mean()
    clean_evasion = 0.5 * (
        F.relu(clean_target_margin + cfg.attack.clean_preserve_margin).mean()
        + F.relu(label_target_margin + cfg.attack.clean_preserve_margin).mean()
    )
    return {
        "poison_target_loss": poison_target,
        "clean_preserve_loss": clean_preserve,
        "binding_gap_loss": binding_gap,
        "clean_evasion_loss": clean_evasion,
        "poison_target_margin": poison_target_margin.mean(),
        "clean_target_margin": clean_target_margin.mean(),
        "label_target_margin": label_target_margin.mean(),
        "worst_control_margin": worst_control_margin.mean(),
        "differential_margin": differential_margin.mean(),
    }


def _stealth_components(
    family: str,
    generated: torch.Tensor,
    neighbor_features: torch.Tensor,
    target_features: torch.Tensor,
    cfg: Config,
) -> dict[str, torch.Tensor]:
    generated_mean = generated.mean(dim=1)
    homophily = (
        1.0 - F.cosine_similarity(generated_mean, neighbor_features, dim=1)
    ).mean()
    entropy = -(
        generated.clamp_min(1e-8) * generated.clamp_min(1e-8).log()
    ).sum(dim=-1).mean()
    dist_loss = _distribution_loss(generated, target_features)
    distribution_coefficient = (
        0.05 * cfg.attack.distribution_weight
        if family == "ugba_style_binding_aware"
        else cfg.attack.distribution_weight
    )
    stealth = (
        cfg.attack.homophily_weight * homophily
        + distribution_coefficient * dist_loss
        + cfg.attack.sparsity_weight * entropy
    )
    return {
        "stealth_loss": stealth,
        "homophily_loss": homophily,
        "distribution_loss": dist_loss,
        "entropy": entropy,
    }


def _victim_shuffle_generated(generated: torch.Tensor) -> torch.Tensor:
    if generated.size(0) > 1:
        return torch.roll(generated, shifts=1, dims=0)
    if generated.size(1) > 1:
        return torch.flip(generated, dims=(1,))
    return generated.clone()


def _context_relation_loss(generated: torch.Tensor, context: torch.Tensor, feature_dim: int) -> torch.Tensor:
    """Preserve pairwise context geometry in generated trigger means."""
    trigger_mean = F.normalize(generated.mean(dim=1), dim=1)
    victim_feature = F.normalize(context[:, :feature_dim], dim=1)
    if trigger_mean.size(0) <= 1:
        return trigger_mean.new_zeros(())
    trigger_similarity = trigger_mean @ trigger_mean.T
    context_similarity = victim_feature @ victim_feature.T
    mask = ~torch.eye(trigger_mean.size(0), dtype=torch.bool, device=trigger_mean.device)
    return F.mse_loss(trigger_similarity[mask], context_similarity[mask])


def _optimize_generator_round(
    family: str,
    clean_graph: GraphData,
    train_victims: torch.Tensor,
    calibration_nodes: torch.Tensor,
    generator: nn.Module,
    prototype_bank: torch.Tensor | None,
    clean_surrogate: SupervisedGCN,
    label_surrogate: SupervisedGCN,
    poisoned_surrogate: SupervisedGCN,
    cfg: Config,
    device: torch.device,
    motif: str,
    round_index: int,
) -> dict[str, Any]:
    attack_nodes = torch.unique(torch.cat([train_victims, calibration_nodes], dim=0))
    ids = attack_nodes.to(device)
    base_x = clean_graph.x.to(device)
    neighbor_mean = _neighbor_mean(clean_graph, device)
    context = torch.cat([base_x[ids], neighbor_mean[ids]], dim=1)
    original_labels = clean_graph.y.to(device)[ids]
    if family == "dpgba_style_binding_aware":
        target_features = clean_graph.x[clean_graph.train_mask].to(device)
    else:
        target_features = clean_graph.x[
            clean_graph.train_mask & (clean_graph.y == cfg.attack.target_class)
        ].to(device)

    for model in [clean_surrogate, label_surrogate, poisoned_surrogate]:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

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
        poison_logits = _forward_on_triggered_nodes(
            poisoned_surrogate,
            clean_graph,
            attack_nodes,
            generated,
            motif,
            cfg,
            device,
        )
        clean_logits = _forward_on_triggered_nodes(
            clean_surrogate,
            clean_graph,
            attack_nodes,
            generated,
            motif,
            cfg,
            device,
        )
        label_logits = _forward_on_triggered_nodes(
            label_surrogate,
            clean_graph,
            attack_nodes,
            generated,
            motif,
            cfg,
            device,
        )
        binding = _binding_components(
            poison_logits,
            clean_logits,
            label_logits,
            original_labels,
            cfg.attack.target_class,
            cfg,
        )
        stealth = _stealth_components(
            family,
            generated,
            neighbor_mean[ids],
            target_features,
            cfg,
        )

        attack_mode = str(cfg.attack.clean_label_attack_mode).lower().strip()
        mode_components: dict[str, torch.Tensor] = {}
        mode_loss = generated.new_zeros(())
        shuffled_generated = _victim_shuffle_generated(generated)
        if attack_mode == "generic":
            shuffled_poison_logits = _forward_on_triggered_nodes(
                poisoned_surrogate, clean_graph, attack_nodes, shuffled_generated, motif, cfg, device
            )
            shuffled_clean_logits = _forward_on_triggered_nodes(
                clean_surrogate, clean_graph, attack_nodes, shuffled_generated, motif, cfg, device
            )
            shuffled_label_logits = _forward_on_triggered_nodes(
                label_surrogate, clean_graph, attack_nodes, shuffled_generated, motif, cfg, device
            )
            target = torch.full_like(original_labels, cfg.attack.target_class)
            shuffled_target_loss = F.cross_entropy(shuffled_poison_logits, target)

            # Universal triggers should not depend strongly on the specific victim.
            consistency_loss = (
                generated - generated.mean(dim=0, keepdim=True)
            ).pow(2).mean()

            # Clean-label training victims already belong to the target class, so a
            # clean-model target penalty on those nodes is mathematically ill-posed.
            # Selective activation is therefore enforced only on held-out non-target
            # calibration nodes.
            calibration_mask = (
                torch.arange(attack_nodes.numel(), device=device)
                >= train_victims.numel()
            )
            if calibration_mask.any():
                matched_poison_prob = torch.softmax(poison_logits[calibration_mask], dim=1)[:, cfg.attack.target_class]
                shuffled_poison_prob = torch.softmax(shuffled_poison_logits[calibration_mask], dim=1)[:, cfg.attack.target_class]
                matched_clean_prob = torch.maximum(
                    torch.softmax(clean_logits[calibration_mask], dim=1)[:, cfg.attack.target_class],
                    torch.softmax(label_logits[calibration_mask], dim=1)[:, cfg.attack.target_class],
                )
                shuffled_clean_prob = torch.maximum(
                    torch.softmax(shuffled_clean_logits[calibration_mask], dim=1)[:, cfg.attack.target_class],
                    torch.softmax(shuffled_label_logits[calibration_mask], dim=1)[:, cfg.attack.target_class],
                )
            else:
                matched_poison_prob = torch.softmax(poison_logits, dim=1)[:, cfg.attack.target_class]
                shuffled_poison_prob = torch.softmax(shuffled_poison_logits, dim=1)[:, cfg.attack.target_class]
                matched_clean_prob = torch.maximum(
                    torch.softmax(clean_logits, dim=1)[:, cfg.attack.target_class],
                    torch.softmax(label_logits, dim=1)[:, cfg.attack.target_class],
                )
                shuffled_clean_prob = torch.maximum(
                    torch.softmax(shuffled_clean_logits, dim=1)[:, cfg.attack.target_class],
                    torch.softmax(shuffled_label_logits, dim=1)[:, cfg.attack.target_class],
                )

            clean_cap = float(cfg.attack.generic_clean_probability_cap)
            clean_cap_loss = 0.5 * (
                F.relu(matched_clean_prob - clean_cap).pow(2).mean()
                + F.relu(shuffled_clean_prob - clean_cap).pow(2).mean()
            )
            matched_selectivity = matched_poison_prob - matched_clean_prob
            shuffled_selectivity = shuffled_poison_prob - shuffled_clean_prob
            selectivity_loss = 0.5 * (
                F.relu(cfg.attack.generic_selectivity_margin - matched_selectivity).mean()
                + F.relu(cfg.attack.generic_selectivity_margin - shuffled_selectivity).mean()
            )

            generated_mean = F.normalize(generated.mean(dim=1), dim=1)
            target_centroid = clean_graph.x[
                clean_graph.train_mask & (clean_graph.y == cfg.attack.target_class)
            ].to(device).mean(dim=0, keepdim=True)
            target_centroid = F.normalize(target_centroid, dim=1)
            local_context = F.normalize(neighbor_mean[ids], dim=1)
            target_similarity = (generated_mean * target_centroid).sum(dim=1)
            local_similarity = (generated_mean * local_context).sum(dim=1)
            target_similarity_excess = F.relu(
                target_similarity
                - local_similarity
                - cfg.attack.generic_target_similarity_allowance
            ).mean()

            mode_loss = (
                cfg.attack.generic_shuffled_target_weight * shuffled_target_loss
                + cfg.attack.generic_consistency_weight * consistency_loss
                + cfg.attack.generic_clean_cap_weight * clean_cap_loss
                + cfg.attack.generic_selectivity_weight * selectivity_loss
                + cfg.attack.generic_target_similarity_weight * target_similarity_excess
            )
            mode_components = {
                "generic_shuffled_target_loss": shuffled_target_loss,
                "generic_consistency_loss": consistency_loss,
                "generic_clean_cap_loss": clean_cap_loss,
                "generic_selectivity_loss": selectivity_loss,
                "generic_matched_poison_probability": matched_poison_prob.mean(),
                "generic_shuffled_poison_probability": shuffled_poison_prob.mean(),
                "generic_matched_clean_probability": matched_clean_prob.mean(),
                "generic_shuffled_clean_probability": shuffled_clean_prob.mean(),
                "generic_matched_selectivity": matched_selectivity.mean(),
                "generic_shuffled_selectivity": shuffled_selectivity.mean(),
                "generic_target_similarity_excess": target_similarity_excess,
            }
        elif attack_mode == "contextual":
            shuffled_poison_logits = _forward_on_triggered_nodes(
                poisoned_surrogate, clean_graph, attack_nodes, shuffled_generated, motif, cfg, device
            )
            matched_score = poison_logits[:, cfg.attack.target_class]
            shuffled_score = shuffled_poison_logits[:, cfg.attack.target_class]
            calibration_mask = torch.arange(attack_nodes.numel(), device=device) >= train_victims.numel()
            if calibration_mask.any():
                pair_gap = matched_score[calibration_mask] - shuffled_score[calibration_mask]
            else:
                pair_gap = matched_score - shuffled_score
            pair_loss = F.relu(cfg.attack.contextual_pair_margin - pair_gap).mean()
            relation_loss = _context_relation_loss(generated, context, clean_graph.num_features)
            feature_std = generated.mean(dim=1).std(dim=0, unbiased=False).mean()
            diversity_loss = F.relu(cfg.attack.contextual_min_feature_std - feature_std)
            mode_loss = (
                cfg.attack.contextual_pair_weight * pair_loss
                + cfg.attack.contextual_relation_weight * relation_loss
                + cfg.attack.contextual_diversity_weight * diversity_loss
            )
            mode_components = {
                "contextual_pair_loss": pair_loss,
                "contextual_pair_gap": pair_gap.mean(),
                "contextual_relation_loss": relation_loss,
                "contextual_feature_std": feature_std,
                "contextual_diversity_loss": diversity_loss,
            }
        else:
            raise ValueError(
                "clean_label_attack_mode must be 'generic' or 'contextual', "
                f"got {cfg.attack.clean_label_attack_mode!r}"
            )

        loss = (
            cfg.attack.poison_target_weight * binding["poison_target_loss"]
            + cfg.attack.binding_gap_weight * binding["binding_gap_loss"]
            + cfg.attack.clean_preserve_weight * binding["clean_preserve_loss"]
            + cfg.attack.clean_evasion_weight * binding["clean_evasion_loss"]
            + stealth["stealth_loss"]
            + mode_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
        optimizer.step()

        value = float(loss.item())
        row = {
            "outer_round": float(round_index),
            "epoch": float(epoch),
            "loss": value,
        }
        for name, component in {**binding, **stealth, **mode_components}.items():
            row[name] = float(component.detach().item())
        history.append(row)
        if value < best_value - 1e-6:
            best_value = value
            best_state = copy.deepcopy(generator.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if cfg.output.verbose and (epoch == 1 or epoch % 25 == 0):
            log(
                f"[BindingGen:{family}] round={round_index} epoch={epoch:04d} "
                f"loss={value:.4f} gap={binding['differential_margin'].item():.4f} "
                f"clean_margin={binding['clean_target_margin'].item():.4f}",
                True,
            )
        if stale >= cfg.attack.generator_patience:
            break

    generator.load_state_dict(best_state)
    return {
        "round": int(round_index),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_value),
        "history": history,
    }


def _evaluate_surrogate_binding(
    clean_graph: GraphData,
    nodes: torch.Tensor,
    generator: nn.Module,
    prototype_bank: torch.Tensor | None,
    clean_surrogate: SupervisedGCN,
    label_surrogate: SupervisedGCN,
    poisoned_surrogate: SupervisedGCN,
    cfg: Config,
    device: torch.device,
    motif: str,
) -> dict[str, float]:
    ids = nodes.to(device)
    neighbor_mean = _neighbor_mean(clean_graph, device)
    context = torch.cat(
        [clean_graph.x.to(device)[ids], neighbor_mean[ids]], dim=1
    )
    generator.eval()
    clean_surrogate.eval()
    label_surrogate.eval()
    poisoned_surrogate.eval()
    with torch.no_grad():
        generated = _generate_with_module(generator, prototype_bank, context)
        poison_logits = _forward_on_triggered_nodes(
            poisoned_surrogate, clean_graph, nodes, generated, motif, cfg, device
        )
        clean_logits = _forward_on_triggered_nodes(
            clean_surrogate, clean_graph, nodes, generated, motif, cfg, device
        )
        label_logits = _forward_on_triggered_nodes(
            label_surrogate, clean_graph, nodes, generated, motif, cfg, device
        )
        poison_prob = torch.softmax(poison_logits, dim=1)[:, cfg.attack.target_class]
        clean_prob = torch.softmax(clean_logits, dim=1)[:, cfg.attack.target_class]
        label_prob = torch.softmax(label_logits, dim=1)[:, cfg.attack.target_class]
        original = clean_graph.y.to(device)[ids]
        clean_preserve_rate = float(
            (clean_logits.argmax(dim=1) == original).float().mean().item()
        )
        label_preserve_rate = float(
            (label_logits.argmax(dim=1) == original).float().mean().item()
        )
    return {
        "poison_target_rate": float(
            (poison_logits.argmax(dim=1) == cfg.attack.target_class)
            .float()
            .mean()
            .item()
        ),
        "clean_target_rate": float(
            (clean_logits.argmax(dim=1) == cfg.attack.target_class).float().mean().item()
        ),
        "label_only_target_rate": float(
            (label_logits.argmax(dim=1) == cfg.attack.target_class).float().mean().item()
        ),
        "poison_target_probability": float(poison_prob.mean().item()),
        "clean_target_probability": float(clean_prob.mean().item()),
        "label_only_target_probability": float(label_prob.mean().item()),
        "probability_binding_gap": float(
            (poison_prob - torch.maximum(clean_prob, label_prob)).mean().item()
        ),
        "clean_original_prediction_rate": clean_preserve_rate,
        "label_only_original_prediction_rate": label_preserve_rate,
        "generated_neighbor_cosine": float(
            F.cosine_similarity(
                generated.mean(dim=1), neighbor_mean[ids], dim=1
            ).mean().item()
        ),
        "generated_distribution_loss": float(
            _distribution_loss(
                generated,
                clean_graph.x[clean_graph.train_mask].to(device),
            ).item()
        ),
        "generated_mean_nonzero_features": float(
            (generated > 1e-8).float().sum(dim=-1).mean().item()
        ),
    }


def _select_topology_binding_aware(
    clean_graph: GraphData,
    calibration_nodes: torch.Tensor,
    generator: nn.Module,
    prototype_bank: torch.Tensor | None,
    clean_surrogate: SupervisedGCN,
    label_surrogate: SupervisedGCN,
    poisoned_surrogate: SupervisedGCN,
    cfg: Config,
    device: torch.device,
) -> tuple[str, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    best_motif = "chain"
    best_score = -float("inf")
    for motif in MOTIFS:
        metrics = _evaluate_surrogate_binding(
            clean_graph,
            calibration_nodes,
            generator,
            prototype_bank,
            clean_surrogate,
            label_surrogate,
            poisoned_surrogate,
            cfg,
            device,
            motif,
        )
        possible = max(
            1, cfg.attack.trigger_size * (cfg.attack.trigger_size - 1) // 2
        )
        density = len(motif_pairs(cfg.attack.trigger_size, motif)) / possible
        # Reward poisoned-only activation and clean-label preservation. Direct
        # clean-model target activation is explicitly penalized.
        score = (
            metrics["probability_binding_gap"]
            + 0.25 * metrics["clean_original_prediction_rate"]
            - cfg.attack.topology_density_weight * density
        )
        rows.append(
            {
                "motif": motif,
                "selection_score": float(score),
                "internal_density": float(density),
                **metrics,
            }
        )
        if score > best_score:
            best_score = score
            best_motif = motif
    return best_motif, rows


def build_attack_plan_v065(
    clean_graph: GraphData,
    cfg: Config,
    device: torch.device,
) -> AttackPlan:
    family = cfg.attack.family.lower().strip()
    if family not in ATTACK_FAMILIES_V065:
        raise ValueError(
            f"attack.family must be one of {ATTACK_FAMILIES_V065}, got {family!r}"
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

    calibration_nodes = _select_calibration_nodes(
        clean_graph,
        cfg.attack.target_class,
        cfg.attack.generator_calibration_count,
        cfg.experiment.seed,
    )
    base_family = _family_base_name(family)
    set_seed(cfg.experiment.seed + 3301)
    generator, prototype_bank = _initial_generator(
        base_family, clean_graph, cfg, device
    )
    if family == "dpgba_style_binding_aware":
        # Round 6.2 uses a mixed prototype bank. A purely global bank was too
        # weak to bind the generated trigger to the target class, while a
        # target-only bank can become direct targeted evasion. The mixture gives
        # the poisoned surrogate target-related capacity and keeps global
        # distribution support; clean and label-only surrogate penalties remain
        # active during generator optimization.
        train_mask = clean_graph.train_mask
        target_pool = clean_graph.x[train_mask & (clean_graph.y == cfg.attack.target_class)]
        background_pool = clean_graph.x[train_mask & (clean_graph.y != cfg.attack.target_class)]
        total = min(
            cfg.attack.generator_prototype_count,
            target_pool.size(0) + background_pool.size(0),
        )
        target_count = min(
            target_pool.size(0),
            max(1, int(round(total * cfg.attack.distribution_target_prototype_fraction))),
        )
        background_count = min(background_pool.size(0), max(1, total - target_count))
        target_pos = torch.linspace(0, target_pool.size(0) - 1, target_count).round().long()
        background_pos = torch.linspace(
            0, background_pool.size(0) - 1, background_count
        ).round().long()
        prototype_bank = torch.cat(
            [target_pool[target_pos], background_pool[background_pos]], dim=0
        ).to(device)

        # _initial_generator() sizes the DPGBA mixture head using only the
        # target-class prototype pool. Round 6.2 replaces that pool with a
        # larger mixed target/background bank, so the final linear layer must
        # be rebuilt with the actual mixed-bank size. Otherwise the mixture
        # weights have shape [..., target_pool_count] while prototype_bank has
        # shape [mixed_count, feature_dim], causing an einsum p-dimension error.
        generator = PrototypeMixtureGenerator(
            feature_dim=clean_graph.num_features,
            hidden_dim=cfg.attack.generator_hidden_dim,
            trigger_size=cfg.attack.trigger_size,
            prototype_count=int(prototype_bank.size(0)),
        ).to(device)
        raw_blend = cfg.attack.distribution_raw_blend
    else:
        raw_blend = cfg.attack.adaptive_raw_blend
    generator = ContextBlendWrapper(
        generator, clean_graph.num_features, raw_blend, prototype_bank
    ).to(device)
    prototype_bank = None
    training_motif = "chain" if family == "dpgba_style_binding_aware" else "star"

    # The clean and poisoned surrogates share exactly the same initialization.
    set_seed(cfg.experiment.seed + 5001)
    template = SupervisedGCN(
        in_features=clean_graph.num_features,
        hidden_dim=cfg.attack.surrogate_hidden_dim,
        num_classes=clean_graph.num_classes,
        dropout=cfg.model.dropout,
    ).to(device)
    initial_state = copy.deepcopy(template.state_dict())
    del template

    clean_surrogate, label_surrogate, clean_diag = _train_control_surrogates(
        clean_graph, victims, cfg, device, initial_state
    )
    all_history: list[dict[str, float]] = []
    round_diagnostics: list[dict[str, Any]] = []
    poisoned_surrogate: SupervisedGCN | None = None

    for round_index in range(1, cfg.attack.generator_outer_rounds + 1):
        plan = _make_plan(
            family,
            victims,
            cfg,
            training_motif,
            generator,
            prototype_bank,
        )
        poisoned_surrogate, poison_diag = _train_poisoned_surrogate(
            clean_graph,
            plan,
            cfg,
            device,
            initial_state,
            round_index,
        )
        optimization = _optimize_generator_round(
            family,
            clean_graph,
            victims,
            calibration_nodes,
            generator,
            prototype_bank,
            clean_surrogate,
            label_surrogate,
            poisoned_surrogate,
            cfg,
            device,
            training_motif,
            round_index,
        )
        all_history.extend(optimization.pop("history"))
        eval_metrics = _evaluate_surrogate_binding(
            clean_graph,
            calibration_nodes,
            generator,
            prototype_bank,
            clean_surrogate,
            label_surrogate,
            poisoned_surrogate,
            cfg,
            device,
            training_motif,
        )
        round_diagnostics.append(
            {**poison_diag, **optimization, **eval_metrics}
        )

    # Refit the poisoned surrogate on the final generator before topology search.
    final_plan = _make_plan(
        family,
        victims,
        cfg,
        training_motif,
        generator,
        prototype_bank,
    )
    poisoned_surrogate, final_poison_diag = _train_poisoned_surrogate(
        clean_graph,
        final_plan,
        cfg,
        device,
        initial_state,
        cfg.attack.generator_outer_rounds + 1,
    )
    motif, topology_rows = _select_topology_binding_aware(
        clean_graph,
        calibration_nodes,
        generator,
        prototype_bank,
        clean_surrogate,
        label_surrogate,
        poisoned_surrogate,
        cfg,
        device,
    )
    final_binding = _evaluate_surrogate_binding(
        clean_graph,
        calibration_nodes,
        generator,
        prototype_bank,
        clean_surrogate,
        label_surrogate,
        poisoned_surrogate,
        cfg,
        device,
        motif,
    )

    diagnostics: dict[str, Any] = {
        "generator_kind": type(generator).__name__,
        "clean_label_attack_mode": str(cfg.attack.clean_label_attack_mode),
        "attack_objective": "binding_aware_difference_of_models",
        "training_motif": training_motif,
        "selected_motif": motif,
        "calibration_node_count": int(calibration_nodes.numel()),
        "calibration_node_ids": calibration_nodes.tolist(),
        "outer_rounds": int(cfg.attack.generator_outer_rounds),
        "selection_method": cfg.attack.selection_method,
        "target_class": int(cfg.attack.target_class),
        "poison_count": int(victims.numel()),
        "distribution_target_prototype_fraction": float(
            cfg.attack.distribution_target_prototype_fraction
        ),
        "round_diagnostics": round_diagnostics,
        "topology_search": topology_rows,
        **clean_diag,
        **final_poison_diag,
        **{f"final_{key}": value for key, value in final_binding.items()},
        "generator_history": all_history,
    }
    return AttackPlan(
        family=family,
        victims=victims,
        target_class=cfg.attack.target_class,
        trigger_size=cfg.attack.trigger_size,
        motif=motif,
        stamp_strength=0.0,
        generator=generator,
        prototype_bank=prototype_bank,
        diagnostics=diagnostics,
    )


__all__ = [
    "ATTACK_FAMILIES_V065",
    "build_attack_plan_v065",
    "build_paired_graphs",
    "make_triggered_test_graph_from_plan",
]
