from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
    if bins <= 1 or len(np.unique(values)) <= 1:
        return np.zeros_like(values, dtype=np.int64)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    boundaries = np.unique(np.quantile(values, quantiles))
    if len(boundaries) <= 2:
        return np.zeros_like(values, dtype=np.int64)
    return np.digitize(values, boundaries[1:-1], right=True).astype(np.int64)


def robust_z_matrix(
    values: np.ndarray,
    labels: np.ndarray,
    degrees: np.ndarray,
    degree_bins: int,
    minimum_group_size: int,
    epsilon: float,
    clip: float,
) -> np.ndarray:
    if values.ndim == 1:
        values = values[:, None]
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


def aggregate_component(z_values: np.ndarray) -> np.ndarray:
    if z_values.ndim == 1:
        return z_values
    if z_values.shape[1] == 1:
        return z_values[:, 0]
    # Mean of the two strongest coordinates keeps sensitivity without one noisy maximum dominating.
    sorted_values = np.sort(z_values, axis=1)
    take = min(2, sorted_values.shape[1])
    return sorted_values[:, -take:].mean(axis=1)


def spectral_relation_discrepancy(
    supervised_distribution: np.ndarray,
    ssl_distribution: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """DShield-style same-label relation contraction in spectral-signature space."""
    scores = np.zeros(len(labels), dtype=np.float64)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) <= 1:
            continue
        sup = supervised_distribution[indices]
        ssl = ssl_distribution[indices]
        sup_distance = np.linalg.norm(sup[:, None, :] - sup[None, :, :], axis=2)
        ssl_distance = np.linalg.norm(ssl[:, None, :] - ssl[None, :, :], axis=2)
        contraction = np.maximum(ssl_distance - sup_distance, 0.0)
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
    plt.figure(figsize=(6.8, 5.2))
    for name, scores in score_map.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        plt.plot(fpr, tpr, label=f"{name} ({auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Poison-node detection ROC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.8, 5.2))
    for name, scores in score_map.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(recall, precision, label=f"{name} ({ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Poison-node detection precision-recall")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curves.png", dpi=180)
    plt.close()


def _save_band_profile(
    delta_gain_layers: list[np.ndarray],
    y_true: np.ndarray,
    output_dir: Path,
) -> None:
    for layer_index, delta in enumerate(delta_gain_layers, start=1):
        clean_mean = delta[y_true == 0].mean(axis=0)
        poison_mean = delta[y_true == 1].mean(axis=0)
        x = np.arange(delta.shape[1])
        width = 0.38
        plt.figure(figsize=(7.0, 4.5))
        plt.bar(x - width / 2, clean_mean, width=width, label="clean")
        plt.bar(x + width / 2, poison_mean, width=width, label="poisoned")
        plt.xticks(x, [f"B{i}" for i in range(delta.shape[1])])
        plt.xlabel("Bernstein graph-frequency band")
        plt.ylabel("Supervised gain - SSL gain")
        plt.title(f"Cross-model spectral transfer profile, layer {layer_index}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"band_transfer_layer{layer_index}.png", dpi=180)
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
    minimum_group_size: int,
    mad_epsilon: float,
    score_clip: float,
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

    for sup_h, ssl_h in zip(supervised_hidden, ssl_hidden):
        sup_raw, sup_distribution = band_energies(laplacian, sup_h, num_bands, epsilon)
        ssl_raw, ssl_distribution = band_energies(laplacian, ssl_h, num_bands, epsilon)
        supervised_distributions.append(sup_distribution)
        ssl_distributions.append(ssl_distribution)
        model_js_layers.append(js_divergence(sup_distribution, ssl_distribution, epsilon))
        sup_gain = log_band_gain(sup_raw, input_raw, sup_h.size(1), x.size(1), epsilon)
        ssl_gain = log_band_gain(ssl_raw, input_raw, ssl_h.size(1), x.size(1), epsilon)
        delta_gain_layers.append(sup_gain - ssl_gain)

    candidate = candidate_indices.detach().cpu().numpy()
    observed_labels = labels[candidate_indices].detach().cpu().numpy()
    y_true = poison_mask[candidate_indices].to(torch.int64).detach().cpu().numpy()
    degrees_all = node_degree(edge_index, x.size(0)).numpy()
    degrees = degrees_all[candidate]

    moments_np = moments[candidate_indices].detach().cpu().numpy()
    input_distribution_np = input_distribution[candidate_indices].detach().cpu().numpy()
    model_js_np = torch.stack(model_js_layers, dim=1)[candidate_indices].detach().cpu().numpy()
    delta_gain_np = torch.cat(delta_gain_layers, dim=1)[candidate_indices].detach().cpu().numpy()
    relation_layers = []
    for sup_distribution, ssl_distribution in zip(supervised_distributions, ssl_distributions):
        relation_layers.append(
            spectral_relation_discrepancy(
                sup_distribution[candidate_indices].detach().cpu().numpy(),
                ssl_distribution[candidate_indices].detach().cpu().numpy(),
                observed_labels,
            )
        )
    relation_np = np.stack(relation_layers, axis=1)

    structure_z = robust_z_matrix(
        np.log1p(moments_np), observed_labels, degrees, degree_bins,
        minimum_group_size, mad_epsilon, score_clip,
    )
    input_z = robust_z_matrix(
        input_distribution_np, observed_labels, degrees, degree_bins,
        minimum_group_size, mad_epsilon, score_clip,
    )
    model_z = robust_z_matrix(
        model_js_np, observed_labels, degrees, degree_bins,
        minimum_group_size, mad_epsilon, score_clip,
    )
    relation_z = robust_z_matrix(
        relation_np, observed_labels, degrees, degree_bins,
        minimum_group_size, mad_epsilon, score_clip,
    )
    transfer_z = robust_z_matrix(
        delta_gain_np, observed_labels, degrees, degree_bins,
        minimum_group_size, mad_epsilon, score_clip,
    )

    structure_score = aggregate_component(structure_z)
    input_score = aggregate_component(input_z)
    model_score = aggregate_component(model_z)
    relation_score = aggregate_component(relation_z)
    transfer_score = aggregate_component(transfer_z)

    component_matrix = np.stack(
        [structure_score, input_score, model_score, relation_score, transfer_score], axis=1
    )
    component_z = robust_z_matrix(
        component_matrix, observed_labels, degrees, degree_bins,
        minimum_group_size, mad_epsilon, score_clip,
    )
    combined_score = component_z.mean(axis=1)

    score_map = {
        "structure": structure_score,
        "input_spectrum": input_score,
        "model_js": model_score,
        "spectral_relation": relation_score,
        "transfer": transfer_score,
        "combined": combined_score,
    }
    metrics = evaluate_binary_scores(
        y_true=y_true,
        score_map=score_map,
        topk_fraction=topk_fraction,
        oracle_positive_count=int(y_true.sum()),
    )

    frame = pd.DataFrame(
        {
            "node_id": candidate,
            "observed_label": observed_labels,
            "degree": degrees,
            "is_poisoned": y_true,
            "score_structure": structure_score,
            "score_input_spectrum": input_score,
            "score_model_js": model_score,
            "score_spectral_relation": relation_score,
            "score_transfer": transfer_score,
            "score_combined": combined_score,
        }
    )
    for index in range(moments_np.shape[1]):
        frame[f"moment_{index}"] = moments_np[:, index]
    for band in range(num_bands):
        frame[f"input_band_{band}"] = input_distribution_np[:, band]
    for layer, js_values in enumerate(model_js_np.T, start=1):
        frame[f"model_js_layer_{layer}"] = js_values
    for layer, relation_values in enumerate(relation_np.T, start=1):
        frame[f"spectral_relation_layer_{layer}"] = relation_values
    for layer, delta in enumerate(delta_gain_layers, start=1):
        delta_np = delta[candidate_indices].detach().cpu().numpy()
        for band in range(num_bands):
            frame[f"delta_gain_l{layer}_b{band}"] = delta_np[:, band]

    frame.to_csv(output_dir / "node_scores.csv", index=False)
    with (output_dir / "detection_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    if make_plots and len(np.unique(y_true)) == 2:
        for name, scores in score_map.items():
            _save_histogram(
                scores[y_true == 0],
                scores[y_true == 1],
                title=f"{name} score distribution",
                xlabel=name,
                path=output_dir / f"distribution_{name}.png",
            )
        _save_roc_pr(y_true, score_map, output_dir)
        _save_band_profile(
            [delta[candidate_indices].detach().cpu().numpy() for delta in delta_gain_layers],
            y_true,
            output_dir,
        )

    extra = {
        "y_true": y_true,
        "candidate_indices": candidate,
        "delta_gain_layers": [
            delta[candidate_indices].detach().cpu().numpy() for delta in delta_gain_layers
        ],
    }
    return frame, metrics, extra
