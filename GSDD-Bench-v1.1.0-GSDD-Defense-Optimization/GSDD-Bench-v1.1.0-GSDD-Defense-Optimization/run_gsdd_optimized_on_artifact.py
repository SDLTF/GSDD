from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from gsdd_core.artifact import normalize_attack_bundle
from gsdd_core.calibration import (
    fuse_cauchy,
    fuse_fisher,
    fuse_max,
    robust_two_sided_calibration,
    tail_soft_weights,
)
from gsdd_core.graph_ops import build_normalized_adjacency, build_normalized_laplacian, node_degree
from gsdd_core.models import DGIModel, SupervisedGCN
from gsdd_core.spectral import band_energies, decompose_delta_gain, log_band_gain
from gsdd_core.train import extract_dgi_hidden, extract_supervised_hidden, train_dgi, train_supervised


FEATURE_NAMES = (
    "raw_l1",
    "shape_l1",
    "distribution_l1",
    "raw_l2",
    "shape_l2",
    "distribution_l2",
)
FUSION_NAMES = ("robust_max", "fisher", "cauchy")


def _safe_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    score = np.asarray(score, dtype=np.float64)
    if not np.isfinite(score).all():
        finite = score[np.isfinite(score)]
        replacement = float(np.median(finite)) if len(finite) else 0.0
        score = np.nan_to_num(score, nan=replacement, posinf=replacement, neginf=replacement)
    fpr, tpr, _ = roc_curve(y_true, score)
    at_95 = np.flatnonzero(tpr >= 0.95)
    order = np.argsort(score)[::-1]
    result = {
        "auroc": float(roc_auc_score(y_true, score)),
        "auprc": float(average_precision_score(y_true, score)),
        "fpr_at_95_tpr": float(fpr[at_95[0]]) if len(at_95) else 1.0,
    }
    for fraction in (0.005, 0.01, 0.02, 0.05):
        k = max(1, int(math.ceil(len(y_true) * fraction)))
        result[f"recall_at_top_{fraction:g}"] = float(
            y_true[order[:k]].sum() / max(1, y_true.sum())
        )
    return result


def _rank01(values: np.ndarray) -> np.ndarray:
    return (rankdata(values, method="average") - 0.5) / len(values)


def _budget_code(value: float) -> str:
    return f"b{int(round(value * 1000)):03d}"


def _save_operational_variants(
    output: Path,
    candidates: np.ndarray,
    num_nodes: int,
    score_map: dict[str, np.ndarray],
    budgets: list[float],
    soft_strength: float,
) -> list[dict[str, object]]:
    hard_dir = output / "hard_indices"
    soft_dir = output / "soft_weights"
    hard_dir.mkdir(parents=True, exist_ok=True)
    soft_dir.mkdir(parents=True, exist_ok=True)
    variants: list[dict[str, object]] = []

    for method, score in score_map.items():
        ordering = np.argsort(score)[::-1]
        for budget in budgets:
            code = _budget_code(budget)
            count = max(1, int(math.ceil(len(candidates) * budget)))
            removed = candidates[ordering[:count]]
            filtered = np.setdiff1d(candidates, removed, assume_unique=False)
            hard_path = hard_dir / f"{method}_{code}_train_idx.pt"
            torch.save(torch.as_tensor(filtered, dtype=torch.long), hard_path)
            variants.append(
                {
                    "mode": "hard",
                    "method": method,
                    "budget": budget,
                    "count": count,
                    "path": str(hard_path.resolve()),
                }
            )

            candidate_weights = tail_soft_weights(score, budget, strength=soft_strength)
            full_weights = torch.ones(num_nodes, dtype=torch.float32)
            full_weights[torch.as_tensor(candidates, dtype=torch.long)] = torch.as_tensor(
                candidate_weights, dtype=torch.float32
            )
            soft_path = soft_dir / f"{method}_{code}_node_weights.pt"
            torch.save(full_weights, soft_path)
            variants.append(
                {
                    "mode": "soft",
                    "method": method,
                    "budget": budget,
                    "count": count,
                    "path": str(soft_path.resolve()),
                    "minimum_weight": float(candidate_weights.min()),
                    "mean_weight": float(candidate_weights.mean()),
                }
            )
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1027)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--bands", type=int, default=5)
    parser.add_argument("--supervised-epochs", type=int, default=200)
    parser.add_argument("--ssl-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--degree-bins", type=int, default=4)
    parser.add_argument("--min-group-size", type=int, default=20)
    parser.add_argument("--budgets", default="0.005,0.01,0.02,0.05")
    parser.add_argument("--soft-strength", type=float, default=6.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; CPU fallback is disabled")
    budgets = [float(value) for value in args.budgets.split(",") if value.strip()]
    if not budgets or any(not 0.0 < value < 1.0 for value in budgets):
        raise SystemExit("--budgets must contain fractions in (0,1)")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    artifact = Path(args.artifact).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    bundle = normalize_attack_bundle(
        torch.load(artifact / "artifact.pt", map_location="cpu", weights_only=False)
    )
    x = bundle["poison_x"].float().to(device)
    y = bundle["poison_y"].long().to(device)
    edge_index = bundle["poison_train_edge_index"].long().to(device)
    num_nodes = int(x.shape[0])
    train_idx = bundle["poison_train_idx"].long().to(device)
    val_idx = bundle["val_idx"].long().to(device)
    attach_idx = bundle["attach_idx"].long().cpu().numpy()
    if y.shape[0] != num_nodes:
        raise RuntimeError(
            f"Artifact normalization failed: poison_x has {num_nodes} nodes but poison_y has {y.shape[0]} labels"
        )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask[val_idx] = True
    adjacency = build_normalized_adjacency(edge_index, num_nodes, device)
    laplacian = build_normalized_laplacian(edge_index, num_nodes, device)

    num_classes = int(y[y >= 0].max().item() + 1)
    supervised = SupervisedGCN(x.shape[1], args.hidden_dim, num_classes, 0.5).to(device)
    self_supervised = DGIModel(x.shape[1], args.hidden_dim).to(device)

    started = time.time()
    supervised_history = train_supervised(
        supervised,
        x,
        y,
        adjacency,
        train_mask,
        val_mask,
        args.supervised_epochs,
        args.lr,
        args.weight_decay,
        args.patience,
        False,
    )
    ssl_history = train_dgi(
        self_supervised,
        x,
        adjacency,
        args.ssl_epochs,
        args.lr,
        args.weight_decay,
        args.patience,
        False,
    )
    _, supervised_hidden = extract_supervised_hidden(supervised, x, adjacency)
    ssl_hidden = extract_dgi_hidden(self_supervised, x, adjacency)

    input_raw, _ = band_energies(laplacian, x, args.bands, 1e-9)
    full_features: dict[str, np.ndarray] = {}
    for layer, (supervised_layer, ssl_layer) in enumerate(
        zip(supervised_hidden, ssl_hidden), start=1
    ):
        supervised_raw, supervised_distribution = band_energies(
            laplacian, supervised_layer, args.bands, 1e-9
        )
        ssl_raw, ssl_distribution = band_energies(
            laplacian, ssl_layer, args.bands, 1e-9
        )
        supervised_gain = log_band_gain(
            supervised_raw, input_raw, supervised_layer.shape[1], x.shape[1], 1e-9
        )
        ssl_gain = log_band_gain(
            ssl_raw, input_raw, ssl_layer.shape[1], x.shape[1], 1e-9
        )
        delta = supervised_gain - ssl_gain
        _, shape, _ = decompose_delta_gain(delta)
        full_features[f"raw_l{layer}"] = (
            torch.linalg.vector_norm(delta, dim=1).detach().cpu().numpy()
        )
        full_features[f"shape_l{layer}"] = (
            torch.linalg.vector_norm(shape, dim=1).detach().cpu().numpy()
        )
        full_features[f"distribution_l{layer}"] = (
            torch.linalg.vector_norm(
                supervised_distribution - ssl_distribution, dim=1
            )
            .detach()
            .cpu()
            .numpy()
        )

    candidates = np.unique(bundle["poison_train_idx"].long().cpu().numpy())
    truth = np.isin(candidates, attach_idx).astype(np.int64)
    if truth.sum() == 0 or truth.sum() == len(truth):
        raise SystemExit("Detection candidates must contain both clean and poisoned nodes")

    candidate_labels = y.detach().cpu().numpy()[candidates]
    candidate_degree = node_degree(edge_index, num_nodes).detach().cpu().numpy()[candidates]
    score_table: dict[str, object] = {
        "node_id": candidates,
        "is_poison": truth,
        "observed_label": candidate_labels,
        "degree": candidate_degree,
    }
    feature_metrics: dict[str, dict[str, float]] = {}
    abs_z_columns: list[np.ndarray] = []
    p_columns: list[np.ndarray] = []

    for name in FEATURE_NAMES:
        raw_score = full_features[name][candidates]
        calibration = robust_two_sided_calibration(
            raw_score,
            candidate_labels,
            candidate_degree,
            degree_bins=args.degree_bins,
            min_group_size=args.min_group_size,
        )
        score_table[name] = raw_score
        score_table[f"{name}_abs_z"] = calibration.abs_z
        score_table[f"{name}_two_sided_p"] = calibration.two_sided_p
        score_table[f"{name}_center"] = calibration.centers
        score_table[f"{name}_scale"] = calibration.scales
        score_table[f"{name}_group_size"] = calibration.group_size
        score_table[f"{name}_degree_bin"] = calibration.degree_bin
        abs_z_columns.append(calibration.abs_z)
        p_columns.append(calibration.two_sided_p)
        feature_metrics[name] = {
            "one_sided": _safe_metrics(truth, raw_score),
            "two_sided": _safe_metrics(truth, calibration.abs_z),
        }

    legacy_full = np.stack(
        [_rank01(full_features[name]) for name in FEATURE_NAMES], axis=1
    ).mean(axis=1)
    legacy_candidate = legacy_full[candidates]
    score_table["legacy_spectral_hybrid"] = legacy_candidate

    abs_z_matrix = np.stack(abs_z_columns, axis=1)
    p_matrix = np.stack(p_columns, axis=1)
    fusion_scores = {
        "robust_max": fuse_max(abs_z_matrix),
        "fisher": fuse_fisher(p_matrix),
        "cauchy": fuse_cauchy(p_matrix),
    }
    fusion_metrics: dict[str, dict[str, float]] = {}
    for name, score in fusion_scores.items():
        score_table[name] = score
        score_table[f"{name}_rank"] = _rank01(score)
        fusion_metrics[name] = _safe_metrics(truth, score)

    pd.DataFrame(score_table).to_csv(output / "node_scores_optimized.csv", index=False)
    variants = _save_operational_variants(
        output,
        candidates,
        num_nodes,
        fusion_scores,
        budgets,
        args.soft_strength,
    )

    summary = {
        "version": "1.1.0",
        "artifact": str(artifact),
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "candidate_count": int(len(candidates)),
        "poison_count": int(truth.sum()),
        "feature_names": list(FEATURE_NAMES),
        "fusion_names": list(FUSION_NAMES),
        "budgets": budgets,
        "soft_strength": args.soft_strength,
        "degree_bins": args.degree_bins,
        "min_group_size": args.min_group_size,
        "supervised_best_epoch": supervised_history.best_epoch,
        "ssl_best_epoch": ssl_history.best_epoch,
        "detection_seconds": time.time() - started,
        "feature_metrics": feature_metrics,
        "legacy_spectral_hybrid_metrics": _safe_metrics(truth, legacy_candidate),
        "fusion_metrics": fusion_metrics,
        "variants": variants,
    }
    (output / "optimization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# GSDD v1.1.0 Optimization Detection Summary",
        "",
        f"- Artifact: `{artifact}`",
        f"- CUDA device: `{summary['device']}`",
        f"- Candidate training nodes: `{len(candidates)}`",
        f"- Known poisoned nodes (evaluation only): `{truth.sum()}`",
        "- Operational methods never use poison labels",
        "",
        "## Legacy one-sided baseline",
        "",
        f"- Legacy spectral-hybrid AUROC: `{summary['legacy_spectral_hybrid_metrics']['auroc']:.4f}`",
        f"- Legacy spectral-hybrid AUPRC: `{summary['legacy_spectral_hybrid_metrics']['auprc']:.4f}`",
        "",
        "## Unsupervised fusion scores",
        "",
        "| Fusion | AUROC | AUPRC | FPR@95TPR | Recall@0.5% | Recall@1% | Recall@2% | Recall@5% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metric in fusion_metrics.items():
        lines.append(
            f"| {name} | {metric['auroc']:.4f} | {metric['auprc']:.4f} | "
            f"{metric['fpr_at_95_tpr']:.4f} | {metric['recall_at_top_0.005']:.4f} | "
            f"{metric['recall_at_top_0.01']:.4f} | {metric['recall_at_top_0.02']:.4f} | "
            f"{metric['recall_at_top_0.05']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Two-sided feature audit",
            "",
            "| Feature | One-sided AUROC | Two-sided AUROC | Two-sided AUPRC |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, metric in feature_metrics.items():
        lines.append(
            f"| {name} | {metric['one_sided']['auroc']:.4f} | "
            f"{metric['two_sided']['auroc']:.4f} | {metric['two_sided']['auprc']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Generated `{len(variants)}` operational hard-filter/soft-weight variants.",
        ]
    )
    (output / "OPTIMIZATION_DETECTION_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(output / "optimization_summary.json")


if __name__ == "__main__":
    main()
