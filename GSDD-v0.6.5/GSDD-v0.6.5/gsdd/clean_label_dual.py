from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .clean_label_factorial import FACTORIAL_MODEL_NAMES, FACTORIAL_TRIGGER_NAMES
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


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64, copy=False)


def compute_generic_clean_label_diagnostics(
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
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Generic clean-label interaction.

    The trigger term averages matched and shuffled conditions because a generic
    trigger is expected to transfer across victim-trigger pairings:

        D_generic = avg_T[P(T)-C(T)] - [P(none)-C(none)]
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    input_raw: dict[str, torch.Tensor] = {}
    for condition in FACTORIAL_TRIGGER_NAMES:
        input_raw[condition], _ = band_energies(
            graph_laplacian[condition], graph_x[condition], num_bands, epsilon
        )

    raw_layers: list[torch.Tensor] = []
    shape_layers: list[torch.Tensor] = []
    level_layers: list[torch.Tensor] = []
    distribution_layers: list[torch.Tensor] = []
    original_nodes = graph_x["none"].size(0)

    for layer_index in range(len(hidden["clean"]["none"])):
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

        matched_gap = gains["poison"]["matched"][:original_nodes] - gains["clean"]["matched"][:original_nodes]
        shuffled_gap = gains["poison"]["shuffled"][:original_nodes] - gains["clean"]["shuffled"][:original_nodes]
        none_gap = gains["poison"]["none"] - gains["clean"]["none"]
        interaction = 0.5 * (matched_gap + shuffled_gap) - none_gap
        level, shape, _ = decompose_delta_gain(interaction)
        distribution_interaction = 0.5 * (
            distributions["poison"]["matched"][:original_nodes]
            - distributions["clean"]["matched"][:original_nodes]
            + distributions["poison"]["shuffled"][:original_nodes]
            - distributions["clean"]["shuffled"][:original_nodes]
        ) - (
            distributions["poison"]["none"] - distributions["clean"]["none"]
        )
        raw_layers.append(interaction)
        level_layers.append(level)
        shape_layers.append(shape)
        distribution_layers.append(distribution_interaction)

    raw_all = torch.cat(raw_layers, dim=1)
    level_all = torch.cat(level_layers, dim=1)
    shape_all = torch.cat(shape_layers, dim=1)
    distribution_all = torch.cat(distribution_layers, dim=1)

    target_logit = 0.5 * (
        logits["poison"]["matched"][:original_nodes, target_class]
        - logits["clean"]["matched"][:original_nodes, target_class]
        + logits["poison"]["shuffled"][:original_nodes, target_class]
        - logits["clean"]["shuffled"][:original_nodes, target_class]
    ) - (
        logits["poison"]["none"][:, target_class]
        - logits["clean"]["none"][:, target_class]
    )

    candidate = _np(candidate_indices).astype(np.int64)
    y_true = _np(victim_mask[candidate_indices]).astype(np.int64)
    degrees = _np(
        node_degree(graph_edge_index["none"], graph_x["none"].size(0)).to(
            graph_x["none"].device
        )[candidate_indices]
    )

    shape_candidate = _np(shape_all)[candidate]
    raw_candidate = _np(raw_all)[candidate]
    distribution_candidate = _np(distribution_all)[candidate]
    arrays: dict[str, np.ndarray] = {
        "cl_generic_raw_l2": _np(torch.linalg.vector_norm(raw_all, dim=1))[candidate],
        "cl_generic_level_l2": _np(torch.linalg.vector_norm(level_all, dim=1))[candidate],
        "cl_generic_shape_l2": _np(torch.linalg.vector_norm(shape_all, dim=1))[candidate],
        "cl_generic_distribution_l2": _np(torch.linalg.vector_norm(distribution_all, dim=1))[candidate],
        "cl_generic_target_logit_abs": np.abs(_np(target_logit)[candidate]),
    }
    arrays["cl_generic_shape_mahalanobis"] = _shrinkage_mahalanobis(
        shape_candidate, mad_epsilon
    )
    arrays["cl_generic_shape_ecdf"] = aggregate_component(
        degree_ecdf_tail_matrix(
            shape_candidate,
            degrees,
            global_degree_bins,
            minimum_group_size,
            mad_epsilon,
            score_clip,
        )
    )
    arrays["cl_generic_distribution_ecdf"] = aggregate_component(
        degree_ecdf_tail_matrix(
            distribution_candidate,
            degrees,
            global_degree_bins,
            minimum_group_size,
            mad_epsilon,
            score_clip,
        )
    )
    arrays["cl_generic_spectral_hybrid"] = rank_normalize(
        np.column_stack(
            [
                arrays["cl_generic_shape_l2"],
                arrays["cl_generic_distribution_l2"],
                arrays["cl_generic_shape_ecdf"],
            ]
        )
    ).mean(axis=1)
    arrays["cl_generic_spectral_logit_hybrid"] = rank_normalize(
        np.column_stack(
            [arrays["cl_generic_spectral_hybrid"], arrays["cl_generic_target_logit_abs"]]
        )
    ).mean(axis=1)

    metrics = evaluate_binary_scores(
        y_true,
        arrays,
        topk_fraction=topk_fraction,
        oracle_positive_count=int(y_true.sum()),
    )
    permutation = {
        name: _permutation_test(values, y_true, permutation_repeats, seed + 211 * index)
        for index, (name, values) in enumerate(arrays.items())
    }
    frame = pd.DataFrame(
        {
            "node_id": candidate,
            "clean_label": _np(clean_labels)[candidate].astype(np.int64),
            "is_selected_victim": y_true,
            **{f"score_{name}": values for name, values in arrays.items()},
        }
    )
    for layer_index, values in enumerate(raw_layers, start=1):
        matrix = _np(values)[candidate]
        for band_index in range(matrix.shape[1]):
            frame[f"layer{layer_index}_cl_generic_band{band_index}"] = matrix[:, band_index]
    frame.to_csv(output_dir / "clean_label_generic_node_scores.csv", index=False)

    raw_report = raw_feature_report(
        y_true,
        {
            "clean_label_generic_interaction": {
                f"layer{li}_band{bi}": _np(values)[candidate, bi]
                for li, values in enumerate(raw_layers, start=1)
                for bi in range(values.size(1))
            }
        },
    )
    if not raw_report.empty:
        raw_report = raw_report.sort_values("auroc_oriented_diagnostic", ascending=False).head(30)
    raw_report.to_csv(output_dir / "clean_label_generic_raw_features.csv", index=False)

    return frame, metrics, {
        "permutation_tests": permutation,
        "raw_feature_top": raw_report.to_dict(orient="records"),
        "positive_count": int(y_true.sum()),
        "candidate_count": int(y_true.size),
        "interaction_definition": "avg_T[poison(T)-clean(T)]-[poison(none)-clean(none)]",
    }
