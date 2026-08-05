from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from .graph_ops import node_degree
from .spectral import band_energies, js_divergence, log_band_gain


def _quantile_bins(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values)
    if bins <= 1 or len(np.unique(values)) <= 1:
        return np.zeros_like(values, dtype=np.int64)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    boundaries = np.unique(np.quantile(values, quantiles))
    if len(boundaries) <= 2:
        return np.zeros_like(values, dtype=np.int64)
    return np.digitize(values, boundaries[1:-1], right=True).astype(np.int64)


def _as_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[:, None] if values.ndim == 1 else values


def robust_z_matrix(
    values: np.ndarray,
    labels: np.ndarray,
    degrees: np.ndarray,
    degree_bins: int,
    minimum_group_size: int,
    epsilon: float,
    clip: float,
) -> np.ndarray:
    """Legacy label/degree-conditioned absolute MAD score."""
    values = _as_matrix(values)
    result = np.zeros_like(values, dtype=np.float64)
    degree_groups = _quantile_bins(degrees, degree_bins)

    for label in np.unique(labels):
        label_mask = labels == label
        label_indices = np.flatnonzero(label_mask)
        for degree_group in np.unique(degree_groups[label_mask]):
            group_mask = label_mask & (degree_groups == degree_group)
            group_indices = np.flatnonzero(group_mask)
            reference = group_indices if len(group_indices) >= minimum_group_size else label_indices
            reference_values = values[reference]
            median = np.median(reference_values, axis=0)
            mad = np.median(np.abs(reference_values - median), axis=0)
            scale = 1.4826 * mad + epsilon
            result[group_indices] = np.abs(values[group_indices] - median) / scale

    return np.clip(result, 0.0, clip)


def degree_global_mad_matrix(
    values: np.ndarray,
    degrees: np.ndarray,
    degree_bins: int,
    minimum_group_size: int,
    epsilon: float,
    clip: float,
) -> np.ndarray:
    """Label-free two-sided robust deviation, optionally conditioned on degree."""
    values = _as_matrix(values)
    pseudo_labels = np.zeros(len(values), dtype=np.int64)
    return robust_z_matrix(
        values,
        pseudo_labels,
        degrees,
        degree_bins,
        minimum_group_size,
        epsilon,
        clip,
    )


def degree_ecdf_tail_matrix(
    values: np.ndarray,
    degrees: np.ndarray,
    degree_bins: int,
    minimum_group_size: int,
    epsilon: float,
    clip: float,
) -> np.ndarray:
    """Label-free two-sided empirical-tail score.

    For every coordinate, values in either tail receive large -log(p) scores.
    This is direction agnostic and therefore can detect both high-band and
    low-band backdoor signatures without choosing a direction from poison labels.
    """
    values = _as_matrix(values)
    result = np.zeros_like(values, dtype=np.float64)
    degree_groups = _quantile_bins(degrees, degree_bins)
    all_indices = np.arange(len(values))

    for degree_group in np.unique(degree_groups):
        group_indices = np.flatnonzero(degree_groups == degree_group)
        reference = group_indices if len(group_indices) >= minimum_group_size else all_indices
        ref_values = values[reference]
        n_ref = len(reference)
        for coordinate in range(values.shape[1]):
            reference_column = ref_values[:, coordinate]
            for index in group_indices:
                value = values[index, coordinate]
                less = float(np.sum(reference_column < value))
                equal = float(np.sum(reference_column == value))
                # Mid-rank empirical CDF with finite-sample smoothing.
                lower = (less + 0.5 * equal + 0.5) / (n_ref + 1.0)
                upper = 1.0 - lower
                two_sided_p = min(1.0, 2.0 * min(lower, upper))
                result[index, coordinate] = -np.log(max(two_sided_p, epsilon))

    return np.clip(result, 0.0, clip)


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Map a vector or each matrix column to [0, 1] by average ranks."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        if len(values) == 0:
            return values.copy()
        return (rankdata(values, method="average") - 0.5) / len(values)
    result = np.zeros_like(values, dtype=np.float64)
    for column in range(values.shape[1]):
        result[:, column] = rank_normalize(values[:, column])
    return result


def trimmed_robust_z_matrix(
    values: np.ndarray,
    labels: np.ndarray,
    degrees: np.ndarray,
    seed_scores: np.ndarray,
    degree_bins: int,
    minimum_group_size: int,
    trim_fraction: float,
    trim_minimum_keep: int,
    epsilon: float,
    clip: float,
) -> np.ndarray:
    """Re-estimate class references after removing globally suspicious nodes.

    No poison labels are used. The label-free seed score decides which samples
    to exclude from each class/degree reference set.
    """
    values = _as_matrix(values)
    seed_scores = np.asarray(seed_scores, dtype=np.float64)
    result = np.zeros_like(values, dtype=np.float64)
    degree_groups = _quantile_bins(degrees, degree_bins)

    if not 0.0 <= trim_fraction < 1.0:
        raise ValueError("trim_fraction must lie in [0, 1)")

    for label in np.unique(labels):
        label_mask = labels == label
        label_indices = np.flatnonzero(label_mask)
        for degree_group in np.unique(degree_groups[label_mask]):
            group_mask = label_mask & (degree_groups == degree_group)
            group_indices = np.flatnonzero(group_mask)
            base_reference = (
                group_indices if len(group_indices) >= minimum_group_size else label_indices
            )
            keep_count = max(
                trim_minimum_keep,
                int(np.ceil((1.0 - trim_fraction) * len(base_reference))),
            )
            keep_count = min(len(base_reference), keep_count)
            ordered = base_reference[np.argsort(seed_scores[base_reference])]
            reference = ordered[:keep_count]
            reference_values = values[reference]
            median = np.median(reference_values, axis=0)
            mad = np.median(np.abs(reference_values - median), axis=0)
            scale = 1.4826 * mad + epsilon
            result[group_indices] = np.abs(values[group_indices] - median) / scale

    return np.clip(result, 0.0, clip)


def aggregate_component(anomaly_values: np.ndarray) -> np.ndarray:
    anomaly_values = np.asarray(anomaly_values, dtype=np.float64)
    if anomaly_values.ndim == 1:
        return anomaly_values
    if anomaly_values.shape[1] == 1:
        return anomaly_values[:, 0]
    sorted_values = np.sort(anomaly_values, axis=1)
    take = min(2, sorted_values.shape[1])
    return sorted_values[:, -take:].mean(axis=1)


def combine_components(component_scores: dict[str, np.ndarray]) -> np.ndarray:
    matrix = np.stack(list(component_scores.values()), axis=1)
    # Rank fusion prevents one family with a numerically larger scale from
    # dominating the diagnostic combination.
    return rank_normalize(matrix).mean(axis=1)


def spectral_relation_discrepancy(
    supervised_distribution: np.ndarray,
    ssl_distribution: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """DShield-style same-label relation contraction in spectral space."""
    scores = np.zeros(len(labels), dtype=np.float64)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) <= 1:
            continue
        supervised = supervised_distribution[indices]
        ssl = ssl_distribution[indices]
        supervised_distance = np.linalg.norm(
            supervised[:, None, :] - supervised[None, :, :], axis=2
        )
        ssl_distance = np.linalg.norm(ssl[:, None, :] - ssl[None, :, :], axis=2)
        contraction = np.maximum(ssl_distance - supervised_distance, 0.0)
        np.fill_diagonal(contraction, 0.0)
        scores[indices] = contraction.sum(axis=1) / max(1, len(indices) - 1)
    return scores


def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = np.flatnonzero(tpr >= target_tpr)
    if len(valid) == 0:
        return 1.0
    return float(fpr[valid[0]])


def evaluate_binary_scores(
    y_true: np.ndarray,
    score_map: dict[str, np.ndarray],
    topk_fraction: float,
    oracle_positive_count: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if len(np.unique(y_true)) < 2:
        return {"warning": "Only one class is present in the diagnostic candidates"}

    for name, scores in score_map.items():
        scores = np.asarray(scores, dtype=np.float64)
        entry: dict[str, float] = {
            "auroc": float(roc_auc_score(y_true, scores)),
            "auprc": float(average_precision_score(y_true, scores)),
            "fpr_at_95_tpr": fpr_at_tpr(y_true, scores),
        }

        k_fraction = max(1, int(round(topk_fraction * len(scores))))
        k_oracle = max(1, min(int(oracle_positive_count), len(scores)))
        for label, k in (("fixed_fraction", k_fraction), ("oracle_k", k_oracle)):
            predicted = np.zeros_like(y_true)
            top_indices = np.argsort(scores)[-k:]
            predicted[top_indices] = 1
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                predicted,
                average="binary",
                zero_division=0,
            )
            entry[f"precision_{label}"] = float(precision)
            entry[f"recall_{label}"] = float(recall)
            entry[f"f1_{label}"] = float(f1)
        metrics[name] = entry
    return metrics


def raw_feature_report(
    y_true: np.ndarray,
    feature_groups: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Diagnostic-only oracle orientation report for primitive coordinates."""
    rows: list[dict[str, Any]] = []
    if len(np.unique(y_true)) < 2:
        return pd.DataFrame(rows)

    for group_name, features in feature_groups.items():
        for feature_name, raw_values in features.items():
            values = np.asarray(raw_values, dtype=np.float64)
            auc_raw = float(roc_auc_score(y_true, values))
            if auc_raw >= 0.5:
                direction = "high"
                oriented = values
                auc_oriented = auc_raw
            else:
                direction = "low"
                oriented = -values
                auc_oriented = 1.0 - auc_raw
            clean = values[y_true == 0]
            poison = values[y_true == 1]
            pooled = np.sqrt(
                0.5 * (np.var(clean, ddof=1) + np.var(poison, ddof=1))
            ) if len(clean) > 1 and len(poison) > 1 else 0.0
            effect = (float(np.mean(poison)) - float(np.mean(clean))) / (pooled + 1e-12)
            rows.append(
                {
                    "group": group_name,
                    "feature": feature_name,
                    "direction_for_poison": direction,
                    "auroc_raw_high_direction": auc_raw,
                    "auroc_oriented_diagnostic": auc_oriented,
                    "auprc_oriented_diagnostic": float(
                        average_precision_score(y_true, oriented)
                    ),
                    "clean_mean": float(np.mean(clean)),
                    "poison_mean": float(np.mean(poison)),
                    "standardized_mean_difference": effect,
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["auroc_oriented_diagnostic", "auprc_oriented_diagnostic"],
            ascending=False,
        ).reset_index(drop=True)
    return frame


def contamination_report(labels: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    rows = []
    for label in np.unique(labels):
        mask = labels == label
        total = int(mask.sum())
        poisoned = int(y_true[mask].sum())
        rows.append(
            {
                "observed_label": int(label),
                "candidate_count": total,
                "poisoned_count_diagnostic": poisoned,
                "poison_fraction_diagnostic": poisoned / max(total, 1),
            }
        )
    return pd.DataFrame(rows)


def _save_histogram(
    clean: np.ndarray,
    poison: np.ndarray,
    title: str,
    xlabel: str,
    path: Path,
) -> None:
    plt.figure(figsize=(7.2, 4.6))
    bins = min(30, max(10, int(np.sqrt(len(clean) + len(poison)))))
    plt.hist(clean, bins=bins, alpha=0.65, label="clean")
    plt.hist(poison, bins=bins, alpha=0.65, label="poisoned")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _save_roc_pr(
    y_true: np.ndarray,
    score_map: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    plt.figure(figsize=(7.4, 5.4))
    for name, scores in score_map.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        plt.plot(fpr, tpr, label=f"{name} ({auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Calibration-family ROC comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_calibration_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.4, 5.4))
    for name, scores in score_map.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(recall, precision, label=f"{name} ({ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Calibration-family precision-recall comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_calibration_comparison.png", dpi=180)
    plt.close()


def _save_band_profile(
    delta_gain_layers: list[np.ndarray],
    y_true: np.ndarray,
    output_dir: Path,
) -> None:
    for layer_index, delta in enumerate(delta_gain_layers, start=1):
        clean_mean = delta[y_true == 0].mean(axis=0)
        poison_mean = delta[y_true == 1].mean(axis=0)
        x_axis = np.arange(delta.shape[1])
        width = 0.38
        plt.figure(figsize=(7.0, 4.5))
        plt.bar(x_axis - width / 2, clean_mean, width=width, label="clean")
        plt.bar(x_axis + width / 2, poison_mean, width=width, label="poisoned")
        plt.xticks(x_axis, [f"B{i}" for i in range(delta.shape[1])])
        plt.xlabel("Bernstein graph-frequency band")
        plt.ylabel("Supervised gain - SSL gain")
        plt.title(f"Cross-model spectral transfer profile, layer {layer_index}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"band_transfer_layer{layer_index}.png", dpi=180)
        plt.close()


def _save_raw_feature_report(frame: pd.DataFrame, output_dir: Path, topk: int) -> None:
    if frame.empty:
        return
    top = frame.head(max(1, topk)).iloc[::-1]
    labels = [f"{row.group}:{row.feature}" for row in top.itertuples()]
    values = top["auroc_oriented_diagnostic"].to_numpy()
    plt.figure(figsize=(9.2, max(5.2, 0.34 * len(top))))
    y_axis = np.arange(len(top))
    plt.barh(y_axis, values)
    plt.yticks(y_axis, labels, fontsize=8)
    plt.axvline(0.5, linestyle="--")
    plt.xlim(0.45, 1.01)
    plt.xlabel("Best-direction AUROC (diagnostic only)")
    plt.title("Primitive spectral coordinates: signal audit")
    plt.tight_layout()
    plt.savefig(output_dir / "raw_feature_signal_audit.png", dpi=180)
    plt.close()


def compute_diagnostics(
    x: torch.Tensor,
    labels: torch.Tensor,
    edge_index: torch.Tensor,
    laplacian: torch.Tensor,
    moments: torch.Tensor,
    supervised_hidden: list[torch.Tensor],
    ssl_hidden: list[torch.Tensor],
    candidate_indices: torch.Tensor,
    poison_mask: torch.Tensor,
    num_bands: int,
    epsilon: float,
    degree_bins: int,
    global_degree_bins: int,
    minimum_group_size: int,
    mad_epsilon: float,
    score_clip: float,
    trim_fraction: float,
    trim_minimum_keep: int,
    raw_feature_topk: int,
    topk_fraction: float,
    output_dir: Path,
    make_plots: bool,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_raw, input_distribution = band_energies(laplacian, x, num_bands, epsilon)

    model_js_layers: list[torch.Tensor] = []
    delta_gain_layers: list[torch.Tensor] = []
    supervised_distributions: list[torch.Tensor] = []
    ssl_distributions: list[torch.Tensor] = []

    for supervised_hidden_layer, ssl_hidden_layer in zip(supervised_hidden, ssl_hidden):
        supervised_raw, supervised_distribution = band_energies(
            laplacian, supervised_hidden_layer, num_bands, epsilon
        )
        ssl_raw, ssl_distribution = band_energies(
            laplacian, ssl_hidden_layer, num_bands, epsilon
        )
        supervised_distributions.append(supervised_distribution)
        ssl_distributions.append(ssl_distribution)
        model_js_layers.append(
            js_divergence(supervised_distribution, ssl_distribution, epsilon)
        )
        supervised_gain = log_band_gain(
            supervised_raw,
            input_raw,
            supervised_hidden_layer.size(1),
            x.size(1),
            epsilon,
        )
        ssl_gain = log_band_gain(
            ssl_raw,
            input_raw,
            ssl_hidden_layer.size(1),
            x.size(1),
            epsilon,
        )
        delta_gain_layers.append(supervised_gain - ssl_gain)

    candidate = candidate_indices.detach().cpu().numpy()
    observed_labels = labels[candidate_indices].detach().cpu().numpy()
    y_true = poison_mask[candidate_indices].to(torch.int64).detach().cpu().numpy()
    degrees_all = node_degree(edge_index, x.size(0)).numpy()
    degrees = degrees_all[candidate]

    moments_np = moments[candidate_indices].detach().cpu().numpy()
    input_distribution_np = input_distribution[candidate_indices].detach().cpu().numpy()
    model_js_np = (
        torch.stack(model_js_layers, dim=1)[candidate_indices].detach().cpu().numpy()
    )
    delta_gain_np = (
        torch.cat(delta_gain_layers, dim=1)[candidate_indices].detach().cpu().numpy()
    )
    relation_layers = []
    for supervised_distribution, ssl_distribution in zip(
        supervised_distributions, ssl_distributions
    ):
        relation_layers.append(
            spectral_relation_discrepancy(
                supervised_distribution[candidate_indices].detach().cpu().numpy(),
                ssl_distribution[candidate_indices].detach().cpu().numpy(),
                observed_labels,
            )
        )
    relation_np = np.stack(relation_layers, axis=1)

    raw_components: dict[str, np.ndarray] = {
        "structure": np.log1p(np.maximum(moments_np, 0.0)),
        "input_spectrum": input_distribution_np,
        "model_js": model_js_np,
        "spectral_relation": relation_np,
        "transfer": delta_gain_np,
    }

    legacy_components: dict[str, np.ndarray] = {}
    global_mad_components: dict[str, np.ndarray] = {}
    global_ecdf_components: dict[str, np.ndarray] = {}
    trimmed_components: dict[str, np.ndarray] = {}
    hybrid_components: dict[str, np.ndarray] = {}

    for name, values in raw_components.items():
        legacy_matrix = robust_z_matrix(
            values,
            observed_labels,
            degrees,
            degree_bins,
            minimum_group_size,
            mad_epsilon,
            score_clip,
        )
        global_mad_matrix = degree_global_mad_matrix(
            values,
            degrees,
            global_degree_bins,
            minimum_group_size,
            mad_epsilon,
            score_clip,
        )
        global_ecdf_matrix = degree_ecdf_tail_matrix(
            values,
            degrees,
            global_degree_bins,
            minimum_group_size,
            mad_epsilon,
            score_clip,
        )

        legacy_score = aggregate_component(legacy_matrix)
        global_mad_score = aggregate_component(global_mad_matrix)
        global_ecdf_score = aggregate_component(global_ecdf_matrix)
        seed_score = 0.5 * rank_normalize(global_mad_score) + 0.5 * rank_normalize(
            global_ecdf_score
        )
        trimmed_matrix = trimmed_robust_z_matrix(
            values,
            observed_labels,
            degrees,
            seed_score,
            degree_bins,
            minimum_group_size,
            trim_fraction,
            trim_minimum_keep,
            mad_epsilon,
            score_clip,
        )
        trimmed_score = aggregate_component(trimmed_matrix)
        hybrid_score = (
            rank_normalize(global_mad_score)
            + rank_normalize(global_ecdf_score)
            + rank_normalize(trimmed_score)
        ) / 3.0

        legacy_components[name] = legacy_score
        global_mad_components[name] = global_mad_score
        global_ecdf_components[name] = global_ecdf_score
        trimmed_components[name] = trimmed_score
        hybrid_components[name] = hybrid_score

    # Retain the exact v0.1-style second-stage label calibration for comparison.
    legacy_component_matrix = np.stack(list(legacy_components.values()), axis=1)
    legacy_combined = aggregate_component(
        robust_z_matrix(
            legacy_component_matrix,
            observed_labels,
            degrees,
            degree_bins,
            minimum_group_size,
            mad_epsilon,
            score_clip,
        )
    )

    combined_scores = {
        "legacy_combined": legacy_combined,
        "global_mad_combined": combine_components(global_mad_components),
        "global_ecdf_combined": combine_components(global_ecdf_components),
        "trimmed_combined": combine_components(trimmed_components),
        "hybrid_combined": combine_components(hybrid_components),
    }

    score_map: dict[str, np.ndarray] = dict(combined_scores)
    for family_name, family in (
        ("legacy", legacy_components),
        ("global_mad", global_mad_components),
        ("global_ecdf", global_ecdf_components),
        ("trimmed", trimmed_components),
        ("hybrid", hybrid_components),
    ):
        for component_name, scores in family.items():
            score_map[f"{family_name}_{component_name}"] = scores

    metrics = evaluate_binary_scores(
        y_true=y_true,
        score_map=score_map,
        topk_fraction=topk_fraction,
        oracle_positive_count=int(y_true.sum()),
    )

    frame_data: dict[str, np.ndarray] = {
        "node_id": candidate,
        "observed_label": observed_labels,
        "degree": degrees,
        "is_poisoned": y_true,
    }
    for family_name, family in (
        ("legacy", legacy_components),
        ("global_mad", global_mad_components),
        ("global_ecdf", global_ecdf_components),
        ("trimmed", trimmed_components),
        ("hybrid", hybrid_components),
    ):
        for component_name, scores in family.items():
            frame_data[f"score_{family_name}_{component_name}"] = scores
    for name, scores in combined_scores.items():
        frame_data[f"score_{name}"] = scores

    frame = pd.DataFrame(frame_data)
    for index in range(moments_np.shape[1]):
        frame[f"moment_{index}"] = moments_np[:, index]
    for band in range(num_bands):
        frame[f"input_band_{band}"] = input_distribution_np[:, band]
    for layer, js_values in enumerate(model_js_np.T, start=1):
        frame[f"model_js_layer_{layer}"] = js_values
    for layer, relation_values in enumerate(relation_np.T, start=1):
        frame[f"spectral_relation_layer_{layer}"] = relation_values
    delta_gain_candidates = []
    for layer, delta in enumerate(delta_gain_layers, start=1):
        delta_np = delta[candidate_indices].detach().cpu().numpy()
        delta_gain_candidates.append(delta_np)
        for band in range(num_bands):
            frame[f"delta_gain_l{layer}_b{band}"] = delta_np[:, band]

    primitive_groups: dict[str, dict[str, np.ndarray]] = {
        "structure": {
            f"log_moment_{index}": raw_components["structure"][:, index]
            for index in range(raw_components["structure"].shape[1])
        },
        "input_spectrum": {
            f"input_band_{band}": input_distribution_np[:, band]
            for band in range(num_bands)
        },
        "model_js": {
            f"model_js_layer_{layer}": model_js_np[:, layer - 1]
            for layer in range(1, model_js_np.shape[1] + 1)
        },
        "spectral_relation": {
            f"spectral_relation_layer_{layer}": relation_np[:, layer - 1]
            for layer in range(1, relation_np.shape[1] + 1)
        },
        "transfer": {
            f"delta_gain_l{layer}_b{band}": delta_gain_candidates[layer - 1][:, band]
            for layer in range(1, len(delta_gain_candidates) + 1)
            for band in range(num_bands)
        },
    }
    raw_report = raw_feature_report(y_true, primitive_groups)
    contamination = contamination_report(observed_labels, y_true)

    frame.to_csv(output_dir / "node_scores.csv", index=False)
    raw_report.to_csv(output_dir / "raw_feature_metrics.csv", index=False)
    contamination.to_csv(output_dir / "label_contamination_diagnostic.csv", index=False)
    with (output_dir / "detection_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    if make_plots and len(np.unique(y_true)) == 2:
        comparison_scores = {
            name: combined_scores[name]
            for name in (
                "legacy_combined",
                "global_mad_combined",
                "global_ecdf_combined",
                "trimmed_combined",
                "hybrid_combined",
            )
        }
        _save_roc_pr(y_true, comparison_scores, output_dir)
        for name, scores in comparison_scores.items():
            _save_histogram(
                scores[y_true == 0],
                scores[y_true == 1],
                title=f"{name} score distribution",
                xlabel=name,
                path=output_dir / f"distribution_{name}.png",
            )
        _save_band_profile(delta_gain_candidates, y_true, output_dir)
        _save_raw_feature_report(raw_report, output_dir, raw_feature_topk)

    max_label_contamination = (
        float(contamination["poison_fraction_diagnostic"].max())
        if not contamination.empty
        else 0.0
    )
    warnings: list[str] = []
    if max_label_contamination >= 0.5:
        warnings.append(
            "At least one observed-label group is >=50% poisoned; median/MAD class calibration can reverse direction."
        )
    if int(y_true.sum()) == 0:
        warnings.append("No poisoned diagnostic candidate is present.")

    extra = {
        "y_true": y_true,
        "candidate_indices": candidate,
        "delta_gain_layers": delta_gain_candidates,
        "raw_feature_top": raw_report.head(raw_feature_topk).to_dict(orient="records"),
        "max_label_contamination_diagnostic": max_label_contamination,
        "diagnostic_warnings": warnings,
    }
    return frame, metrics, extra
