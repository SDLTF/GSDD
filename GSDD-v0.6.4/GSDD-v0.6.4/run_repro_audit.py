from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from gsdd.config import Config, load_config, save_config
from gsdd.paired_diagnostics import MODES
from gsdd.reproducibility import configure_reproducibility
from gsdd.utils import environment_info, resolve_device, save_json
from run_gsdd_v05 import run as run_paired_did

PRIMARY_SCORES = [
    "did_shape_l2",
    "did_distribution_l2",
    "did_spectral_hybrid",
    "did_shape_mahalanobis",
]
ALL_SCORES = PRIMARY_SCORES + [
    "did_target_logit_abs",
    "did_spectral_logit_hybrid",
    "did_raw_l2",
    "did_level_l2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GSDD-v0.5.1 deterministic paired-DID reproducibility audit"
    )
    parser.add_argument("--config", default="configs/cora_repro_audit.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--replicas", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.seed is not None:
        cfg.experiment.seed = args.seed
    if args.device is not None:
        cfg.experiment.device = args.device
    if args.name is not None:
        cfg.experiment.name = args.name
    if args.output_root is not None:
        cfg.experiment.output_root = args.output_root
    if args.replicas is not None:
        cfg.reproducibility.replicas = args.replicas
    if args.strict:
        cfg.reproducibility.warn_only = False
    return cfg


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Expected a state_dict in {path}")
    return value


def compare_states(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> dict[str, float]:
    maximum = 0.0
    squared_difference = 0.0
    squared_reference = 0.0
    element_count = 0
    for key in sorted(first):
        if key not in second:
            raise KeyError(f"Missing parameter in replica 2: {key}")
        a = first[key].detach().to(torch.float64)
        b = second[key].detach().to(torch.float64)
        if a.shape != b.shape:
            raise ValueError(f"Parameter shape mismatch for {key}: {a.shape} vs {b.shape}")
        delta = a - b
        maximum = max(maximum, float(delta.abs().max().item()))
        squared_difference += float(delta.square().sum().item())
        squared_reference += float(a.square().sum().item())
        element_count += a.numel()
    return {
        "max_abs_difference": maximum,
        "rmse": math.sqrt(squared_difference / max(1, element_count)),
        "relative_l2_difference": math.sqrt(squared_difference)
        / max(math.sqrt(squared_reference), 1e-30),
    }


def safe_correlation(first: np.ndarray, second: np.ndarray, kind: str) -> float:
    if first.size < 2:
        return float("nan")
    if np.array_equal(first, second):
        return 1.0
    if np.all(first == first[0]) or np.all(second == second[0]):
        return float("nan")
    if kind == "pearson":
        return float(pearsonr(first, second).statistic)
    return float(spearmanr(first, second).statistic)


def topk_set(values: np.ndarray, node_ids: np.ndarray, k: int) -> set[int]:
    order = np.argsort(-values, kind="mergesort")[:k]
    return set(int(node_ids[index]) for index in order)


def compare_replicas(
    run_dirs: list[Path],
    cfg: Config,
    audit_dir: Path,
) -> dict[str, Any]:
    if len(run_dirs) != 2:
        raise ValueError("v0.5.1 currently requires exactly two replicas")
    first_dir, second_dir = run_dirs
    first_summary = load_json(first_dir / "summary.json")
    second_summary = load_json(second_dir / "summary.json")
    if first_summary.get("status") != "success" or second_summary.get("status") != "success":
        raise RuntimeError("Both replica runs must complete successfully")

    first_nodes = pd.read_csv(first_dir / "paired_node_scores.csv").sort_values("node_id")
    second_nodes = pd.read_csv(second_dir / "paired_node_scores.csv").sort_values("node_id")
    invariant_columns = ["node_id", "clean_label", "full_observed_label", "is_selected_victim"]
    for column in invariant_columns:
        if not np.array_equal(first_nodes[column].to_numpy(), second_nodes[column].to_numpy()):
            raise AssertionError(f"Replica node invariant failed for column: {column}")

    first_metrics = load_json(first_dir / "paired_detection_metrics.json")
    second_metrics = load_json(second_dir / "paired_detection_metrics.json")
    node_ids = first_nodes["node_id"].to_numpy(dtype=np.int64)
    victim_count = int(first_nodes["is_selected_victim"].sum())
    operational_k = max(1, int(math.ceil(len(node_ids) * cfg.reproducibility.operational_topk_fraction)))
    oracle_k = max(1, victim_count)

    score_rows: list[dict[str, Any]] = []
    difference_frame = pd.DataFrame({
        "node_id": node_ids,
        "is_selected_victim": first_nodes["is_selected_victim"].to_numpy(dtype=np.int64),
    })
    for score in ALL_SCORES:
        column = f"score_{score}"
        if column not in first_nodes or column not in second_nodes:
            continue
        first = first_nodes[column].to_numpy(dtype=float)
        second = second_nodes[column].to_numpy(dtype=float)
        delta = second - first
        first_oracle = topk_set(first, node_ids, oracle_k)
        second_oracle = topk_set(second, node_ids, oracle_k)
        first_operational = topk_set(first, node_ids, operational_k)
        second_operational = topk_set(second, node_ids, operational_k)
        metric_first = first_metrics.get(score, {})
        metric_second = second_metrics.get(score, {})
        row = {
            "score": score,
            "pearson": safe_correlation(first, second, "pearson"),
            "spearman": safe_correlation(first, second, "spearman"),
            "mean_abs_difference": float(np.mean(np.abs(delta))),
            "max_abs_difference": float(np.max(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta**2))),
            "oracle_k": oracle_k,
            "oracle_topk_overlap": len(first_oracle & second_oracle) / oracle_k,
            "operational_k": operational_k,
            "operational_topk_overlap": len(first_operational & second_operational) / operational_k,
            "auroc_replica1": metric_first.get("auroc"),
            "auroc_replica2": metric_second.get("auroc"),
            "auroc_abs_delta": abs(float(metric_first.get("auroc", float("nan"))) - float(metric_second.get("auroc", float("nan")))),
            "auprc_replica1": metric_first.get("auprc"),
            "auprc_replica2": metric_second.get("auprc"),
            "auprc_abs_delta": abs(float(metric_first.get("auprc", float("nan"))) - float(metric_second.get("auprc", float("nan")))),
        }
        row["passes"] = bool(
            (np.isnan(row["spearman"]) or row["spearman"] >= cfg.reproducibility.spearman_min)
            and (np.isnan(row["pearson"]) or row["pearson"] >= cfg.reproducibility.pearson_min)
            and row["operational_topk_overlap"] >= cfg.reproducibility.topk_overlap_min
            and row["auroc_abs_delta"] <= cfg.reproducibility.auroc_delta_max
            and row["auprc_abs_delta"] <= cfg.reproducibility.auprc_delta_max
        )
        score_rows.append(row)
        if score in PRIMARY_SCORES:
            difference_frame[f"{score}_replica1"] = first
            difference_frame[f"{score}_replica2"] = second
            difference_frame[f"{score}_abs_difference"] = np.abs(delta)
            plt.figure(figsize=(5.6, 5.2))
            plt.scatter(first, second, alpha=0.7)
            low = min(float(first.min()), float(second.min()))
            high = max(float(first.max()), float(second.max()))
            plt.plot([low, high], [low, high], linestyle="--")
            plt.xlabel("Replica 1 score")
            plt.ylabel("Replica 2 score")
            plt.title(f"Reproducibility: {score}")
            plt.tight_layout()
            plt.savefig(audit_dir / f"repro_scatter_{score}.png", dpi=180)
            plt.close()

    score_frame = pd.DataFrame(score_rows)
    score_frame.to_csv(audit_dir / "score_reproducibility.csv", index=False)
    difference_frame.to_csv(audit_dir / "node_score_reproducibility.csv", index=False)

    parameter_rows: list[dict[str, Any]] = []
    for mode in MODES:
        first_state = load_state(first_dir / f"supervised_{mode}.pt")
        second_state = load_state(second_dir / f"supervised_{mode}.pt")
        parameter_rows.append({"mode": mode, **compare_states(first_state, second_state)})
    parameter_frame = pd.DataFrame(parameter_rows)
    parameter_frame.to_csv(audit_dir / "model_parameter_reproducibility.csv", index=False)

    behavior_rows: list[dict[str, Any]] = []
    for mode in MODES:
        first_behavior = first_summary["model_behavior"][mode]
        second_behavior = second_summary["model_behavior"][mode]
        behavior_rows.append(
            {
                "mode": mode,
                "clean_accuracy_replica1": first_behavior["clean_accuracy"],
                "clean_accuracy_replica2": second_behavior["clean_accuracy"],
                "clean_accuracy_abs_delta": abs(first_behavior["clean_accuracy"] - second_behavior["clean_accuracy"]),
                "asr_replica1": first_behavior["triggered_asr"],
                "asr_replica2": second_behavior["triggered_asr"],
                "asr_abs_delta": abs(first_behavior["triggered_asr"] - second_behavior["triggered_asr"]),
                "best_epoch_replica1": first_behavior["best_epoch"],
                "best_epoch_replica2": second_behavior["best_epoch"],
                "best_epoch_abs_delta": abs(first_behavior["best_epoch"] - second_behavior["best_epoch"]),
            }
        )
    behavior_frame = pd.DataFrame(behavior_rows)
    behavior_frame.to_csv(audit_dir / "model_behavior_reproducibility.csv", index=False)

    primary = score_frame[score_frame["score"].isin(PRIMARY_SCORES)]
    result = {
        "status": "success",
        "seed": int(cfg.experiment.seed),
        "device": str(cfg.experiment.device),
        "replica_dirs": [str(path) for path in run_dirs],
        "initialization_hash_match": first_summary.get("initialization_sha256") == second_summary.get("initialization_sha256"),
        "victim_count": victim_count,
        "candidate_count": len(node_ids),
        "operational_topk_fraction": cfg.reproducibility.operational_topk_fraction,
        "all_primary_scores_pass": bool(primary["passes"].all()),
        "primary_score_pass_count": int(primary["passes"].sum()),
        "primary_score_total": int(len(primary)),
        "maximum_parameter_abs_difference": float(parameter_frame["max_abs_difference"].max()),
        "maximum_clean_accuracy_abs_delta": float(behavior_frame["clean_accuracy_abs_delta"].max()),
        "maximum_asr_abs_delta": float(behavior_frame["asr_abs_delta"].max()),
        "score_reproducibility": score_rows,
        "parameter_reproducibility": parameter_rows,
        "behavior_reproducibility": behavior_rows,
    }
    return result


def make_markdown(summary: dict[str, Any], path: Path) -> None:
    score_rows = summary["score_reproducibility"]
    lines = [
        "# GSDD-v0.5.1 deterministic reproducibility audit",
        "",
        f"- Seed: `{summary['seed']}`",
        f"- Device request: `{summary['device']}`",
        f"- Initialization hash match: `{summary['initialization_hash_match']}`",
        f"- Primary scores passing all stability criteria: `{summary['primary_score_pass_count']}/{summary['primary_score_total']}`",
        f"- Maximum parameter absolute difference: `{summary['maximum_parameter_abs_difference']:.6g}`",
        f"- Maximum clean-accuracy difference: `{summary['maximum_clean_accuracy_abs_delta']:.6g}`",
        f"- Maximum ASR difference: `{summary['maximum_asr_abs_delta']:.6g}`",
        "",
        "## Node-score stability",
        "",
        "| Score | Pearson | Spearman | Top-k overlap | AUROC Δ | AUPRC Δ | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in score_rows:
        lines.append(
            f"| {row['score']} | {row['pearson']:.4f} | {row['spearman']:.4f} | "
            f"{row['operational_topk_overlap']:.4f} | {row['auroc_abs_delta']:.4f} | "
            f"{row['auprc_abs_delta']:.4f} | {row['passes']} |"
        )
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "The paired spectral mechanism is considered numerically reproducible only when the scale-invariant primary scores preserve node ranking, operational top-k candidates, and AUROC/AUPRC across two identical runs.",
            "",
            "See `score_reproducibility.csv`, `node_score_reproducibility.csv`, `model_parameter_reproducibility.csv`, and `model_behavior_reproducibility.csv` for complete values.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    source_config = Path(args.config)
    cfg = apply_overrides(load_config(source_config), args)
    if cfg.reproducibility.replicas != 2:
        raise ValueError("Set reproducibility.replicas to exactly 2 for v0.5.1")
    cfg.reproducibility.enabled = True
    cfg.reproducibility.deterministic_algorithms = True
    cfg.paired.repeat_full_control = False
    cfg.output.save_models = True
    settings = configure_reproducibility(cfg.reproducibility)
    device = resolve_device(cfg.experiment.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_dir = Path(cfg.experiment.output_root) / f"{cfg.experiment.name}_audit_{timestamp}_seed{cfg.experiment.seed}"
    audit_dir.mkdir(parents=True, exist_ok=False)
    save_config(cfg, audit_dir / "config_resolved.yaml")
    shutil.copy2(source_config, audit_dir / "config_source.yaml")
    save_json({**environment_info(device), "reproducibility": settings}, audit_dir / "environment.json")
    progress: dict[str, Any] = {"status": "running", "seed": cfg.experiment.seed}
    save_json(progress, audit_dir / "summary.json")

    try:
        run_dirs: list[Path] = []
        base_name = cfg.experiment.name
        for replica in range(1, cfg.reproducibility.replicas + 1):
            replica_cfg = copy.deepcopy(cfg)
            replica_cfg.experiment.name = f"{base_name}_replica{replica:02d}"
            replica_dir = run_paired_did(replica_cfg, source_config)
            run_dirs.append(replica_dir)
        summary = compare_replicas(run_dirs, cfg, audit_dir)
        summary["reproducibility_settings"] = settings
        save_json(summary, audit_dir / "summary.json")
        make_markdown(summary, audit_dir / "REPRODUCIBILITY_AUDIT.md")
        print(f"AUDIT_DIR={audit_dir}", flush=True)
        return 0
    except Exception as exc:
        progress.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        save_json(progress, audit_dir / "summary.json")
        (audit_dir / "ERROR.txt").write_text(progress["traceback"], encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
