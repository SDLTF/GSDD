from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from .attack_families import AttackPlan, _attack_edge_index, select_test_victims
from .data import GraphData
from .diagnostics import (
    aggregate_component,
    degree_ecdf_tail_matrix,
    evaluate_binary_scores,
    rank_normalize,
    raw_feature_report,
)
from .graph_ops import node_degree
from .paired_diagnostics import _permutation_test, _shrinkage_mahalanobis
from .spectral import band_energies, decompose_delta_gain, log_band_gain


FACTORIAL_MODEL_NAMES = ("clean", "poison")
FACTORIAL_TRIGGER_NAMES = ("none", "matched", "shuffled")


def build_graph_with_explicit_trigger(
    base_graph: GraphData,
    victims: torch.Tensor,
    plan: AttackPlan,
    trigger_features: torch.Tensor,
    *,
    mark_poison: bool,
) -> GraphData:
    """Attach explicit per-victim trigger features without changing labels.

    trigger_features must have shape [num_victims, trigger_size, feature_dim].
    This helper is used by the clean-label factorial audit so matched and
    shuffled graphs have exactly the same topology and trigger marginals.
    """
    victims = victims.detach().cpu().long()
    trigger_features = trigger_features.detach().cpu()
    expected = (victims.numel(), plan.trigger_size, base_graph.num_features)
    if tuple(trigger_features.shape) != expected:
        raise ValueError(
            f"trigger_features must have shape {expected}, got {tuple(trigger_features.shape)}"
        )

    x = base_graph.x.clone()
    y = base_graph.y.clone()
    train_mask = base_graph.train_mask.clone()
    val_mask = base_graph.val_mask.clone()
    test_mask = base_graph.test_mask.clone()
    poison_mask = base_graph.poison_mask.clone()
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


def victim_shuffle_trigger_features(features: torch.Tensor, shift: int = 1) -> torch.Tensor:
    """Cyclically reassign complete triggers across victims.

    The operation preserves the exact multiset of trigger tensors and therefore
    preserves all marginal feature statistics. It only breaks the learned
    context-to-trigger pairing. For one victim, trigger-node order is reversed
    as a deterministic fallback.
    """
    if features.ndim != 3:
        raise ValueError("features must have shape [victims, trigger_size, feature_dim]")
    if features.size(0) > 1:
        effective = int(shift) % features.size(0)
        if effective == 0:
            effective = 1
        return torch.roll(features, shifts=effective, dims=0)
    if features.size(1) > 1:
        return torch.flip(features, dims=(1,))
    # This degenerate case is still a valid exact-marginal control, but cannot
    # break a pairing because only one atomic trigger exists.
    return features.clone()


def build_training_factorial_graphs(
    clean_graph: GraphData,
    plan: AttackPlan,
    device: torch.device,
    shuffle_shift: int,
) -> tuple[dict[str, GraphData], dict[str, torch.Tensor]]:
    matched = plan.generate(clean_graph, plan.victims, device).detach().cpu()
    shuffled = victim_shuffle_trigger_features(matched, shuffle_shift)
    graphs = {
        "none": clean_graph,
        "matched": build_graph_with_explicit_trigger(
            clean_graph, plan.victims, plan, matched, mark_poison=True
        ),
        "shuffled": build_graph_with_explicit_trigger(
            clean_graph, plan.victims, plan, shuffled, mark_poison=True
        ),
    }
    return graphs, {"matched": matched, "shuffled": shuffled}


def build_test_factorial_graphs(
    base_eval_graph: GraphData,
    clean_context_graph: GraphData,
    plan: AttackPlan,
    test_victim_count: int,
    seed: int,
    device: torch.device,
    shuffle_shift: int,
) -> tuple[dict[str, GraphData], torch.Tensor, dict[str, torch.Tensor]]:
    test_victims = select_test_victims(
        clean_context_graph, plan.target_class, test_victim_count, seed
    )
    matched = plan.generate(clean_context_graph, test_victims, device).detach().cpu()
    shuffled = victim_shuffle_trigger_features(matched, shuffle_shift)
    graphs = {
        "none": base_eval_graph,
        "matched": build_graph_with_explicit_trigger(
            base_eval_graph, test_victims, plan, matched, mark_poison=False
        ),
        "shuffled": build_graph_with_explicit_trigger(
            base_eval_graph, test_victims, plan, shuffled, mark_poison=False
        ),
    }
    return graphs, test_victims, {"matched": matched, "shuffled": shuffled}


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64, copy=False)


def _save_roc_pr(y_true: np.ndarray, scores: dict[str, np.ndarray], output_dir: Path) -> None:
    selected = {
        key: value
        for key, value in scores.items()
        if key in {
            "cl_did_shape_l2",
            "cl_did_distribution_l2",
            "cl_did_spectral_hybrid",
            "cl_did_target_logit_abs",
        }
    }
    plt.figure(figsize=(7.4, 5.4))
    for name, values in selected.items():
        fpr, tpr, _ = roc_curve(y_true, values)
        plt.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_true, values):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Clean-label matched-vs-shuffled spectral interaction: ROC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_clean_label_factorial.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.4, 5.4))
    for name, values in selected.items():
        precision, recall, _ = precision_recall_curve(y_true, values)
        plt.plot(recall, precision, label=f"{name} ({average_precision_score(y_true, values):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Clean-label matched-vs-shuffled spectral interaction: PR")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_clean_label_factorial.png", dpi=180)
    plt.close()


def _save_band_profiles(
    interaction_layers: list[np.ndarray], y_true: np.ndarray, output_dir: Path
) -> None:
    for layer_index, values in enumerate(interaction_layers, start=1):
        control_mean = values[y_true == 0].mean(axis=0)
        victim_mean = values[y_true == 1].mean(axis=0)
        positions = np.arange(values.shape[1])
        width = 0.38
        plt.figure(figsize=(7.0, 4.5))
        plt.bar(positions - width / 2, control_mean, width=width, label="other target-class nodes")
        plt.bar(positions + width / 2, victim_mean, width=width, label="poisoned target-class nodes")
        plt.axhline(0.0, linestyle="--")
        plt.xticks(positions, [f"B{i}" for i in range(values.shape[1])])
        plt.xlabel("Bernstein graph-frequency band")
        plt.ylabel("Matched-minus-shuffled model interaction")
        plt.title(f"Clean-label spectral interaction, layer {layer_index}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"clean_label_band_profile_layer{layer_index}.png", dpi=180)
        plt.close()


def compute_clean_label_factorial_diagnostics(
    *,
    graph_x: dict[str, torch.Tensor],
    graph_edge_index: dict[str, torch.Tensor],
    graph_laplacian: dict[str, torch.Tensor],
    hidden: dict[str, dict[str, list[torch.Tensor]]],
    logits: dict[str, dict[str, torch.Tensor]],
    candidate_indices: torch.Tensor,
    victim_mask: torch.Tensor,
    clean_labels: torch.Tensor,
    target_class: int,
    num_bands: int,
    epsilon: float,
    global_degree_bins: int,
    minimum_group_size: int,
    mad_epsilon: float,
    score_clip: float,
    topk_fraction: float,
    permutation_repeats: int,
    seed: int,
    output_dir: Path,
    make_plots: bool,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Compute clean-label model × trigger factorial spectral interactions.

    Primary interaction:

        CL-DID = [P(matched)-C(matched)] - [P(shuffled)-C(shuffled)]

    where P and C are poisoned and clean models. The victim shuffle preserves
    trigger topology and the exact trigger-feature multiset, so the interaction
    targets context-specific trigger binding rather than generic anomaly energy.
    A secondary no-trigger interaction is also saved for audit purposes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    input_raw: dict[str, torch.Tensor] = {}
    for condition in FACTORIAL_TRIGGER_NAMES:
        input_raw[condition], _ = band_energies(
            graph_laplacian[condition], graph_x[condition], num_bands, epsilon
        )

    cl_did_layers: list[torch.Tensor] = []
    cl_did_shape_layers: list[torch.Tensor] = []
    cl_did_level_layers: list[torch.Tensor] = []
    cl_did_distribution_layers: list[torch.Tensor] = []
    no_control_layers: list[torch.Tensor] = []
    original_nodes = graph_x["none"].size(0)
    layer_count = len(hidden["clean"]["none"])

    for layer_index in range(layer_count):
        gains: dict[str, dict[str, torch.Tensor]] = {m: {} for m in FACTORIAL_MODEL_NAMES}
        distributions: dict[str, dict[str, torch.Tensor]] = {
            m: {} for m in FACTORIAL_MODEL_NAMES
        }
        for model_name in FACTORIAL_MODEL_NAMES:
            for condition in FACTORIAL_TRIGGER_NAMES:
                values = hidden[model_name][condition][layer_index]
                raw, distribution = band_energies(
                    graph_laplacian[condition], values, num_bands, epsilon
                )
                gains[model_name][condition] = log_band_gain(
                    raw,
                    input_raw[condition],
                    values.size(1),
                    graph_x[condition].size(1),
                    epsilon,
                )
                distributions[model_name][condition] = distribution

        matched_gap = (
            gains["poison"]["matched"][:original_nodes]
            - gains["clean"]["matched"][:original_nodes]
        )
        shuffled_gap = (
            gains["poison"]["shuffled"][:original_nodes]
            - gains["clean"]["shuffled"][:original_nodes]
        )
        none_gap = gains["poison"]["none"] - gains["clean"]["none"]
        interaction = matched_gap - shuffled_gap
        no_control = matched_gap - none_gap
        level, shape, _ = decompose_delta_gain(interaction)
        distribution_interaction = (
            distributions["poison"]["matched"][:original_nodes]
            - distributions["clean"]["matched"][:original_nodes]
            - distributions["poison"]["shuffled"][:original_nodes]
            + distributions["clean"]["shuffled"][:original_nodes]
        )
        cl_did_layers.append(interaction)
        cl_did_level_layers.append(level)
        cl_did_shape_layers.append(shape)
        cl_did_distribution_layers.append(distribution_interaction)
        no_control_layers.append(no_control)

    cl_did_all = torch.cat(cl_did_layers, dim=1)
    cl_shape_all = torch.cat(cl_did_shape_layers, dim=1)
    cl_level_all = torch.cat(cl_did_level_layers, dim=1)
    cl_distribution_all = torch.cat(cl_did_distribution_layers, dim=1)
    no_control_all = torch.cat(no_control_layers, dim=1)

    target_logit_interaction = (
        logits["poison"]["matched"][:original_nodes, target_class]
        - logits["clean"]["matched"][:original_nodes, target_class]
        - logits["poison"]["shuffled"][:original_nodes, target_class]
        + logits["clean"]["shuffled"][:original_nodes, target_class]
    )
    probabilities = {
        model_name: {
            condition: torch.softmax(logits[model_name][condition], dim=1)
            for condition in FACTORIAL_TRIGGER_NAMES
        }
        for model_name in FACTORIAL_MODEL_NAMES
    }
    target_probability_interaction = (
        probabilities["poison"]["matched"][:original_nodes, target_class]
        - probabilities["clean"]["matched"][:original_nodes, target_class]
        - probabilities["poison"]["shuffled"][:original_nodes, target_class]
        + probabilities["clean"]["shuffled"][:original_nodes, target_class]
    )

    candidate = _to_numpy(candidate_indices).astype(np.int64)
    y_true = _to_numpy(victim_mask[candidate_indices]).astype(np.int64)
    degrees = _to_numpy(
        node_degree(graph_edge_index["none"], graph_x["none"].size(0)).to(
            graph_x["none"].device
        )[candidate_indices]
    )
    arrays: dict[str, np.ndarray] = {
        "cl_did_raw_l2": _to_numpy(torch.linalg.vector_norm(cl_did_all, dim=1))[candidate],
        "cl_did_level_l2": _to_numpy(torch.linalg.vector_norm(cl_level_all, dim=1))[candidate],
        "cl_did_shape_l2": _to_numpy(torch.linalg.vector_norm(cl_shape_all, dim=1))[candidate],
        "cl_did_distribution_l2": _to_numpy(
            torch.linalg.vector_norm(cl_distribution_all, dim=1)
        )[candidate],
        "cl_no_control_l2": _to_numpy(torch.linalg.vector_norm(no_control_all, dim=1))[candidate],
        "cl_did_target_logit_abs": np.abs(_to_numpy(target_logit_interaction)[candidate]),
        "cl_did_target_probability_abs": np.abs(
            _to_numpy(target_probability_interaction)[candidate]
        ),
    }
    shape_candidate = _to_numpy(cl_shape_all)[candidate]
    raw_candidate = _to_numpy(cl_did_all)[candidate]
    distribution_candidate = _to_numpy(cl_distribution_all)[candidate]
    arrays["cl_did_shape_mahalanobis"] = _shrinkage_mahalanobis(
        shape_candidate, mad_epsilon
    )
    arrays["cl_did_shape_ecdf"] = aggregate_component(
        degree_ecdf_tail_matrix(
            shape_candidate,
            degrees,
            global_degree_bins,
            minimum_group_size,
            mad_epsilon,
            score_clip,
        )
    )
    arrays["cl_did_raw_ecdf"] = aggregate_component(
        degree_ecdf_tail_matrix(
            raw_candidate,
            degrees,
            global_degree_bins,
            minimum_group_size,
            mad_epsilon,
            score_clip,
        )
    )
    arrays["cl_did_distribution_ecdf"] = aggregate_component(
        degree_ecdf_tail_matrix(
            distribution_candidate,
            degrees,
            global_degree_bins,
            minimum_group_size,
            mad_epsilon,
            score_clip,
        )
    )
    arrays["cl_did_spectral_hybrid"] = rank_normalize(
        np.column_stack(
            [
                arrays["cl_did_shape_l2"],
                arrays["cl_did_distribution_l2"],
                arrays["cl_did_shape_ecdf"],
            ]
        )
    ).mean(axis=1)
    arrays["cl_did_spectral_logit_hybrid"] = rank_normalize(
        np.column_stack(
            [
                arrays["cl_did_spectral_hybrid"],
                arrays["cl_did_target_logit_abs"],
            ]
        )
    ).mean(axis=1)

    score_names = [
        "cl_did_raw_l2",
        "cl_did_level_l2",
        "cl_did_shape_l2",
        "cl_did_distribution_l2",
        "cl_did_shape_mahalanobis",
        "cl_did_shape_ecdf",
        "cl_did_raw_ecdf",
        "cl_did_distribution_ecdf",
        "cl_did_spectral_hybrid",
        "cl_did_target_logit_abs",
        "cl_did_target_probability_abs",
        "cl_did_spectral_logit_hybrid",
        "cl_no_control_l2",
    ]
    score_map = {name: arrays[name] for name in score_names}
    metrics = evaluate_binary_scores(
        y_true,
        score_map,
        topk_fraction=topk_fraction,
        oracle_positive_count=int(y_true.sum()),
    )
    permutation = {
        name: _permutation_test(values, y_true, permutation_repeats, seed + 101 * index)
        for index, (name, values) in enumerate(score_map.items())
    }

    frame = pd.DataFrame(
        {
            "node_id": candidate,
            "clean_label": _to_numpy(clean_labels)[candidate].astype(np.int64),
            "is_selected_victim": y_true,
            **{f"score_{name}": values for name, values in arrays.items()},
        }
    )
    for layer_index, values in enumerate(cl_did_layers, start=1):
        matrix = _to_numpy(values)[candidate]
        for band_index in range(matrix.shape[1]):
            frame[f"layer{layer_index}_cl_did_band{band_index}"] = matrix[:, band_index]
    frame.to_csv(output_dir / "clean_label_factorial_node_scores.csv", index=False)

    raw_report = raw_feature_report(
        y_true,
        {
            "clean_label_spectral_interaction": {
                f"layer{layer_index}_band{band_index}": _to_numpy(values)[candidate, band_index]
                for layer_index, values in enumerate(cl_did_layers, start=1)
                for band_index in range(values.size(1))
            }
        },
    )
    if not raw_report.empty:
        raw_report = raw_report.sort_values(
            "auroc_oriented_diagnostic", ascending=False
        ).head(min(30, len(raw_report)))
    raw_report.to_csv(
        output_dir / "clean_label_factorial_raw_features.csv", index=False
    )
    raw_top = raw_report.to_dict(orient="records")

    if make_plots:
        _save_roc_pr(y_true, score_map, output_dir)
        _save_band_profiles([_to_numpy(v)[candidate] for v in cl_did_layers], y_true, output_dir)

    return frame, metrics, {
        "permutation_tests": permutation,
        "raw_feature_top": raw_top,
        "positive_count": int(y_true.sum()),
        "candidate_count": int(y_true.size),
        "interaction_definition": (
            "[poison(matched)-clean(matched)]-"
            "[poison(shuffled)-clean(shuffled)]"
        ),
    }
