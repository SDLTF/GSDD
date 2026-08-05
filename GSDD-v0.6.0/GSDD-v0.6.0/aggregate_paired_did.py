from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCORES = [
    "did_shape_l2",
    "did_distribution_l2",
    "did_spectral_hybrid",
    "did_target_logit_abs",
    "did_spectral_logit_hybrid",
    "did_raw_l2",
    "did_level_l2",
    "did_shape_mahalanobis",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--prefix", default="gsdd_v05_cora_paired_did")
    parser.add_argument("--output-dir", default="results/paired_did_aggregate")
    return parser.parse_args()


def complete_run(path: Path) -> bool:
    required = [
        path / "summary.json",
        path / "paired_node_scores.csv",
        path / "paired_detection_metrics.json",
        path / "model_behavior.csv",
    ]
    if not all(item.exists() for item in required):
        return False
    try:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return summary.get("status") == "success"


def flatten_run(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_dir": str(path),
        "seed": int(summary["seed"]),
        "device": summary.get("device"),
        "victim_count": summary.get("victim_count"),
        "repeat_parameter_max_abs": summary.get("repeat_control", {}).get(
            "parameter_max_abs_difference"
        ),
        "repeat_logit_max_abs": summary.get("repeat_control", {}).get(
            "training_logit_max_abs_difference"
        ),
        "repeat_best_epoch_difference": summary.get("repeat_control", {}).get(
            "best_epoch_difference"
        ),
    }
    behavior = summary.get("model_behavior", {})
    for mode in ["none", "label_only", "trigger_only", "full"]:
        item = behavior.get(mode, {})
        row[f"{mode}_clean_accuracy"] = item.get("clean_accuracy")
        row[f"{mode}_asr"] = item.get("triggered_asr")
        row[f"{mode}_best_epoch"] = item.get("best_epoch")

    detection = summary.get("paired_detection", {})
    permutation = summary.get("permutation_tests", {})
    for score in SCORES:
        item = detection.get(score, {})
        row[f"{score}_auroc"] = item.get("auroc")
        row[f"{score}_auprc"] = item.get("auprc")
        row[f"{score}_fpr95"] = item.get("fpr_at_95_tpr")
        row[f"{score}_f1_oracle"] = item.get("f1_oracle_k")
        row[f"{score}_permutation_p"] = permutation.get(score, {}).get("p_value")
    return row


def main() -> int:
    args = parse_args()
    root = Path(args.results_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    for path in sorted(root.glob(f"{args.prefix}*")):
        if not path.is_dir() or not complete_run(path):
            continue
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        candidates.append(
            {
                "run_dir": str(path),
                "seed": int(summary["seed"]),
                "mtime": path.stat().st_mtime,
                "summary": summary,
            }
        )

    if not candidates:
        raise RuntimeError(f"No complete runs match prefix {args.prefix!r}")

    candidate_frame = pd.DataFrame(
        [{k: v for k, v in item.items() if k != "summary"} for item in candidates]
    ).sort_values(["seed", "mtime"])
    candidate_frame.to_csv(output / "paired_did_run_candidates.csv", index=False)

    selected: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for seed, group in candidate_frame.groupby("seed"):
        keep_index = group["mtime"].idxmax()
        keep_path = group.loc[keep_index, "run_dir"]
        for index, row in group.iterrows():
            if index != keep_index:
                duplicates.append(
                    {
                        "seed": int(seed),
                        "ignored_run_dir": row["run_dir"],
                        "kept_run_dir": keep_path,
                    }
                )
        item = next(candidate for candidate in candidates if candidate["run_dir"] == keep_path)
        selected.append(flatten_run(Path(keep_path), item["summary"]))

    pd.DataFrame(duplicates).to_csv(output / "duplicate_run_candidates.csv", index=False)
    frame = pd.DataFrame(selected).sort_values("seed").reset_index(drop=True)
    frame.to_csv(output / "paired_did_runs.csv", index=False)

    numeric = [column for column in frame.select_dtypes(include="number").columns if column != "seed"]
    stats = frame[numeric].agg(["mean", "std", "min", "max"]).T
    stats.to_csv(output / "paired_did_summary_stats.csv")

    criteria = []
    for row in frame.itertuples():
        criteria.append(
            {
                "seed": row.seed,
                "functional_backdoor": bool(
                    row.full_asr >= 0.80
                    and row.trigger_only_asr <= 0.20
                    and row.label_only_asr <= 0.20
                    and row.none_asr <= 0.20
                ),
                "spectral_shape_signal": bool(
                    row.did_shape_l2_auroc >= 0.70
                    and row.did_shape_l2_permutation_p <= 0.05
                ),
                "distribution_signal": bool(
                    row.did_distribution_l2_auroc >= 0.70
                    and row.did_distribution_l2_permutation_p <= 0.05
                ),
                "repeat_stable": bool(
                    (pd.isna(row.repeat_logit_max_abs) or row.repeat_logit_max_abs <= 1e-4)
                ),
            }
        )
    criteria_frame = pd.DataFrame(criteria)
    criteria_frame.to_csv(output / "paired_did_success_criteria.csv", index=False)

    def mean(column: str) -> float:
        return float(frame[column].mean())

    lines = [
        "# GSDD-v0.5 paired backdoor-specific DID aggregate",
        "",
        f"Unique seeds: {len(frame)}",
        f"Complete candidate runs: {len(candidate_frame)}",
        f"Ignored duplicate runs: {len(duplicates)}",
        "",
        "## Functional controls",
        "",
        "| Mode | Mean clean accuracy | Mean triggered ASR |",
        "|---|---:|---:|",
    ]
    for mode in ["none", "label_only", "trigger_only", "full"]:
        lines.append(
            f"| {mode} | {mean(f'{mode}_clean_accuracy'):.4f} | {mean(f'{mode}_asr'):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Backdoor-specific paired discrepancy",
            "",
            "| Score | AUROC mean ± std | AUPRC mean ± std | Mean permutation p |",
            "|---|---:|---:|---:|",
        ]
    )
    for score in SCORES:
        auc = frame[f"{score}_auroc"]
        ap = frame[f"{score}_auprc"]
        p = frame[f"{score}_permutation_p"]
        lines.append(
            f"| {score} | {auc.mean():.4f} ± {auc.std(ddof=1):.4f} | "
            f"{ap.mean():.4f} ± {ap.std(ddof=1):.4f} | {p.mean():.4g} |"
        )

    lines.extend(
        [
            "",
            "## Numerical repeat control",
            "",
            f"- Mean parameter max-absolute difference: `{mean('repeat_parameter_max_abs'):.6g}`",
            f"- Mean training-logit max-absolute difference: `{mean('repeat_logit_max_abs'):.6g}`",
            f"- Mean best-epoch difference: `{mean('repeat_best_epoch_difference'):.4f}`",
            "",
            "## Decision rule",
            "",
            "A backdoor-specific spectral mechanism is supported only if the `full` model alone has high ASR and the scale-invariant paired scores (`did_shape_l2` or `did_distribution_l2`) remain predictive with small permutation p-values across seeds. A strong logit DID with weak spectral DID means that the label-trigger association exists, but the proposed frequency mechanism is not yet supported.",
            "",
            "See `paired_did_runs.csv`, `paired_did_summary_stats.csv`, and `paired_did_success_criteria.csv` for all values.",
        ]
    )
    (output / "PAIRED_DID_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(frame)} unique paired-DID runs to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
