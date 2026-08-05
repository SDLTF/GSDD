from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .diagnostics import (
    aggregate_component,
    degree_ecdf_tail_matrix,
    evaluate_binary_scores,
    rank_normalize,
    raw_feature_report,
)
from .graph_ops import node_degree
from .spectral import band_energies, decompose_delta_gain, log_band_gain


MODES = ("none", "label_only", "trigger_only", "full")


def _shrinkage_mahalanobis(values: np.ndarray, epsilon: float) -> np.ndarray:
    """Deterministic covariance-aware distance without robust-MCD warnings."""
    values = np.asarray(values, dtype=np.float64)
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center), axis=0)
    scale = np.where(1.4826 * mad > epsilon, 1.4826 * mad, 1.0)
    standardized = (values - center) / scale
    estimator = LedoitWolf(assume_centered=False).fit(standardized)
    return np.sqrt(np.maximum(estimator.mahalanobis(standardized), 0.0))


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64, copy=False)


def _permutation_test(
    scores: np.ndarray,
    y_true: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    victims = np.flatnonzero(y_true == 1)
    controls = np.flatnonzero(y_true == 0)
    if len(victims) == 0 or len(controls) < len(victims):
        return {"observed_victim_mean": float("nan"), "p_value": float("nan")}
    observed = float(scores[victims].mean())
    rng = np.random.default_rng(seed)
    null = np.empty(max(1, repeats), dtype=np.float64)
    for index in range(len(null)):
        sampled = rng.choice(controls, size=len(victims), replace=False)
        null[index] = float(scores[sampled].mean())
    p_value = float((1 + np.sum(null >= observed)) / (len(null) + 1))
    return {
        "observed_victim_mean": observed,
        "control_mean": float(scores[controls].mean()),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)) if len(null) > 1 else 0.0,
        "p_value": p_value,
        "repeats": int(len(null)),
    }


def _save_histogram(values: np.ndarray, y_true: np.ndarray, title: str, path: Path) -> None:
    clean = values[y_true == 0]
    victims = values[y_true == 1]
    bins = min(30, max(10, int(np.sqrt(len(values)))))
    plt.figure(figsize=(7.0, 4.5))
    plt.hist(clean, bins=bins, alpha=0.65, label="other training nodes")
    plt.hist(victims, bins=bins, alpha=0.75, label="selected victims")
    plt.xlabel(title)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _save_roc_pr(y_true: np.ndarray, scores: dict[str, np.ndarray], output_dir: Path) -> None:
    plt.figure(figsize=(7.4, 5.4))
    for name, values in scores.items():
        fpr, tpr, _ = roc_curve(y_true, values)
        plt.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_true, values):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Backdoor-specific paired discrepancy: ROC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_paired_did.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.4, 5.4))
    for name, values in scores.items():
        precision, recall, _ = precision_recall_curve(y_true, values)
        plt.plot(recall, precision, label=f"{name} ({average_precision_score(y_true, values):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Backdoor-specific paired discrepancy: precision-recall")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_paired_did.png", dpi=180)
    plt.close()


def _save_band_profiles(
    did_layers: list[np.ndarray],
    y_true: np.ndarray,
    output_dir: Path,
) -> None:
    for layer_index, values in enumerate(did_layers, start=1):
        clean_mean = values[y_true == 0].mean(axis=0)
        victim_mean = values[y_true == 1].mean(axis=0)
        x = np.arange(values.shape[1])
        width = 0.38
        plt.figure(figsize=(7.0, 4.5))
        plt.bar(x - width / 2, clean_mean, width=width, label="other nodes")
        plt.bar(x + width / 2, victim_mean, width=width, label="selected victims")
        plt.axhline(0.0, linestyle="--")
        plt.xticks(x, [f"B{i}" for i in range(values.shape[1])])
        plt.xlabel("Bernstein graph-frequency band")
        plt.ylabel("Difference-in-differences log gain")
        plt.title(f"Backdoor-specific spectral interaction, layer {layer_index}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"did_band_profile_layer{layer_index}.png", dpi=180)
        plt.close()


def compute_paired_diagnostics(
    clean_x: torch.Tensor,
    trigger_x: torch.Tensor,
    clean_edge_index: torch.Tensor,
    clean_laplacian: torch.Tensor,
    trigger_laplacian: torch.Tensor,
    hidden_by_mode: dict[str, list[torch.Tensor]],
    logits_by_mode: dict[str, torch.Tensor],
    candidate_indices: torch.Tensor,
    victim_mask: torch.Tensor,
    clean_labels: torch.Tensor,
    full_labels: torch.Tensor,
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
    """Compute paired backdoor-specific spectral difference-in-differences.

    The two within-graph contrasts are

        B_T = T_full - T_trigger_only
        B_C = T_label_only - T_none

    and the causal interaction diagnostic is

        DID = B_T - B_C.

    Because each contrast compares models on exactly the same graph and all
    models share the same initialization/training RNG, trigger-input effects
    and ordinary dirty-label effects are removed before victim detection.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_input_raw, _ = band_energies(clean_laplacian, clean_x, num_bands, epsilon)
    trigger_input_raw, _ = band_energies(trigger_laplacian, trigger_x, num_bands, epsilon)

    did_gain_layers: list[torch.Tensor] = []
    did_shape_layers: list[torch.Tensor] = []
    did_level_layers: list[torch.Tensor] = []
    did_distribution_layers: list[torch.Tensor] = []
    trigger_binding_layers: list[torch.Tensor] = []
    clean_binding_layers: list[torch.Tensor] = []

    original_nodes = clean_x.size(0)
    layer_count = len(hidden_by_mode["none"])
    for layer_index in range(layer_count):
        raw: dict[str, torch.Tensor] = {}
        distribution: dict[str, torch.Tensor] = {}
        for mode in MODES:
            laplacian = clean_laplacian if mode in {"none", "label_only"} else trigger_laplacian
            input_raw = clean_input_raw if mode in {"none", "label_only"} else trigger_input_raw
            input_dim = clean_x.size(1) if mode in {"none", "label_only"} else trigger_x.size(1)
            hidden = hidden_by_mode[mode][layer_index]
            hidden_raw, hidden_distribution = band_energies(
                laplacian, hidden, num_bands, epsilon
            )
            raw[mode] = log_band_gain(
                hidden_raw,
                input_raw,
                hidden.size(1),
                input_dim,
                epsilon,
            )
            distribution[mode] = hidden_distribution

        binding_trigger = raw["full"][:original_nodes] - raw["trigger_only"][:original_nodes]
        binding_clean = raw["label_only"] - raw["none"]
        did_gain = binding_trigger - binding_clean
        did_level, did_shape, _ = decompose_delta_gain(did_gain)

        distribution_did = (
            distribution["full"][:original_nodes]
            - distribution["trigger_only"][:original_nodes]
            - distribution["label_only"]
            + distribution["none"]
        )
        trigger_binding_layers.append(binding_trigger)
        clean_binding_layers.append(binding_clean)
        did_gain_layers.append(did_gain)
        did_level_layers.append(did_level)
        did_shape_layers.append(did_shape)
        did_distribution_layers.append(distribution_did)

    did_gain_all = torch.cat(did_gain_layers, dim=1)
    did_shape_all = torch.cat(did_shape_layers, dim=1)
    did_level_all = torch.cat(did_level_layers, dim=1)
    did_distribution_all = torch.cat(did_distribution_layers, dim=1)
    trigger_binding_all = torch.cat(trigger_binding_layers, dim=1)
    clean_binding_all = torch.cat(clean_binding_layers, dim=1)

    target_logit_did = (
        logits_by_mode["full"][:original_nodes, target_class]
        - logits_by_mode["trigger_only"][:original_nodes, target_class]
        - logits_by_mode["label_only"][:, target_class]
        + logits_by_mode["none"][:, target_class]
    )
    probabilities = {mode: torch.softmax(logits, dim=1) for mode, logits in logits_by_mode.items()}
    target_probability_did = (
        probabilities["full"][:original_nodes, target_class]
        - probabilities["trigger_only"][:original_nodes, target_class]
        - probabilities["label_only"][:, target_class]
        + probabilities["none"][:, target_class]
    )

    candidate = _to_numpy(candidate_indices).astype(np.int64)
    y_true = _to_numpy(victim_mask[candidate_indices]).astype(np.int64)
    degrees = _to_numpy(node_degree(clean_edge_index, clean_x.size(0)).to(clean_x.device)[candidate_indices])

    arrays = {
        "did_raw_l2": _to_numpy(torch.linalg.vector_norm(did_gain_all, dim=1))[candidate],
        "did_level_l2": _to_numpy(torch.linalg.vector_norm(did_level_all, dim=1))[candidate],
        "did_shape_l2": _to_numpy(torch.linalg.vector_norm(did_shape_all, dim=1))[candidate],
        "did_distribution_l2": _to_numpy(torch.linalg.vector_norm(did_distribution_all, dim=1))[candidate],
        "did_target_logit_abs": np.abs(_to_numpy(target_logit_did)[candidate]),
        "did_target_probability_abs": np.abs(_to_numpy(target_probability_did)[candidate]),
        "trigger_binding_l2": _to_numpy(torch.linalg.vector_norm(trigger_binding_all, dim=1))[candidate],
        "clean_label_binding_l2": _to_numpy(torch.linalg.vector_norm(clean_binding_all, dim=1))[candidate],
    }

    did_shape_candidate = _to_numpy(did_shape_all)[candidate]
    did_gain_candidate = _to_numpy(did_gain_all)[candidate]
    did_dist_candidate = _to_numpy(did_distribution_all)[candidate]
    arrays["did_shape_mahalanobis"] = _shrinkage_mahalanobis(
        did_shape_candidate, mad_epsilon
    )
    arrays["did_shape_ecdf"] = aggregate_component(
        degree_ecdf_tail_matrix(
            did_shape_candidate,
            degrees,
            global_degree_bins,
            minimum_group_size,
            mad_epsilon,
            score_clip,
        )
    )
    arrays["did_raw_ecdf"] = aggregate_component(
        degree_ecdf_tail_matrix(
            did_gain_candidate,
            degrees,
            global_degree_bins,
            minimum_group_size,
            mad_epsilon,
            score_clip,
        )
    )
    arrays["did_distribution_ecdf"] = aggregate_component(
        degree_ecdf_tail_matrix(
            did_dist_candidate,
            degrees,
            global_degree_bins,
            minimum_group_size,
            mad_epsilon,
            score_clip,
        )
    )
    arrays["did_spectral_hybrid"] = rank_normalize(
        np.column_stack(
            [
                arrays["did_shape_l2"],
                arrays["did_distribution_l2"],
                arrays["did_shape_ecdf"],
            ]
        )
    ).mean(axis=1)
    arrays["did_spectral_logit_hybrid"] = rank_normalize(
        np.column_stack(
            [
                arrays["did_spectral_hybrid"],
                arrays["did_target_logit_abs"],
            ]
        )
    ).mean(axis=1)

    score_names = [
        "did_raw_l2",
        "did_level_l2",
        "did_shape_l2",
        "did_distribution_l2",
        "did_shape_mahalanobis",
        "did_shape_ecdf",
        "did_raw_ecdf",
        "did_spectral_hybrid",
        "did_target_logit_abs",
        "did_target_probability_abs",
        "did_spectral_logit_hybrid",
    ]
    score_map = {name: arrays[name] for name in score_names}
    metrics = evaluate_binary_scores(
        y_true,
        score_map,
        topk_fraction=topk_fraction,
        oracle_positive_count=int(y_true.sum()),
    )

    permutation = {
        name: _permutation_test(values, y_true, permutation_repeats, seed + 97 * index)
        for index, (name, values) in enumerate(score_map.items())
    }

    frame = pd.DataFrame(
        {
            "node_id": candidate,
            "clean_label": _to_numpy(clean_labels)[candidate].astype(np.int64),
            "full_observed_label": _to_numpy(full_labels)[candidate].astype(np.int64),
            "is_selected_victim": y_true,
            **{f"score_{name}": values for name, values in arrays.items()},
        }
    )
    for layer_index, (did, shape, level, dist) in enumerate(
        zip(did_gain_layers, did_shape_layers, did_level_layers, did_distribution_layers),
        start=1,
    ):
        did_np = _to_numpy(did)[candidate]
        shape_np = _to_numpy(shape)[candidate]
        level_np = _to_numpy(level)[candidate, 0]
        dist_np = _to_numpy(dist)[candidate]
        frame[f"did_level_l{layer_index}"] = level_np
        for band in range(num_bands):
            frame[f"did_gain_l{layer_index}_b{band}"] = did_np[:, band]
            frame[f"did_shape_l{layer_index}_b{band}"] = shape_np[:, band]
            frame[f"did_distribution_l{layer_index}_b{band}"] = dist_np[:, band]

    frame.to_csv(output_dir / "paired_node_scores.csv", index=False)
    (output_dir / "paired_detection_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "paired_permutation_tests.json").write_text(
        json.dumps(permutation, indent=2), encoding="utf-8"
    )

    feature_groups: dict[str, dict[str, np.ndarray]] = {
        "did_gain": {},
        "did_shape": {},
        "did_distribution": {},
    }
    for layer_index in range(layer_count):
        did_np = _to_numpy(did_gain_layers[layer_index])[candidate]
        shape_np = _to_numpy(did_shape_layers[layer_index])[candidate]
        dist_np = _to_numpy(did_distribution_layers[layer_index])[candidate]
        for band in range(num_bands):
            feature_groups["did_gain"][f"l{layer_index + 1}_b{band}"] = did_np[:, band]
            feature_groups["did_shape"][f"l{layer_index + 1}_b{band}"] = shape_np[:, band]
            feature_groups["did_distribution"][f"l{layer_index + 1}_b{band}"] = dist_np[:, band]
    feature_groups["non_spectral_anchor"] = {
        "target_logit_did": _to_numpy(target_logit_did)[candidate],
        "target_probability_did": _to_numpy(target_probability_did)[candidate],
    }
    raw_report = raw_feature_report(y_true, feature_groups)
    raw_report.to_csv(output_dir / "paired_raw_feature_metrics.csv", index=False)

    if make_plots and len(np.unique(y_true)) == 2:
        plot_scores = {
            "shape_l2": arrays["did_shape_l2"],
            "distribution_l2": arrays["did_distribution_l2"],
            "spectral_hybrid": arrays["did_spectral_hybrid"],
            "logit_anchor": arrays["did_target_logit_abs"],
        }
        _save_roc_pr(y_true, plot_scores, output_dir)
        _save_histogram(
            arrays["did_shape_l2"], y_true, "DID spectral-shape magnitude", output_dir / "distribution_did_shape_l2.png"
        )
        _save_histogram(
            arrays["did_target_logit_abs"], y_true, "DID target-logit magnitude", output_dir / "distribution_did_target_logit.png"
        )
        _save_band_profiles([_to_numpy(x)[candidate] for x in did_gain_layers], y_true, output_dir)
        plt.figure(figsize=(6.2, 5.2))
        plt.scatter(
            arrays["did_target_logit_abs"][y_true == 0],
            arrays["did_shape_l2"][y_true == 0],
            alpha=0.65,
            label="other nodes",
        )
        plt.scatter(
            arrays["did_target_logit_abs"][y_true == 1],
            arrays["did_shape_l2"][y_true == 1],
            alpha=0.9,
            label="selected victims",
        )
        plt.xlabel("Absolute target-logit DID")
        plt.ylabel("Spectral-shape DID magnitude")
        plt.title("Backdoor association anchor vs spectral interaction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "scatter_logit_vs_spectral_did.png", dpi=180)
        plt.close()

    extra = {
        "permutation_tests": permutation,
        "raw_feature_top": raw_report.head(20).to_dict(orient="records"),
        "victim_count": int(y_true.sum()),
        "candidate_count": int(len(y_true)),
    }
    return frame, metrics, extra
