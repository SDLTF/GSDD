from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


FAMILIES = [
    "fixed_rare_clique",
    "ugba_style_adaptive",
    "dpgba_style_distribution",
]
SCORES = [
    "did_shape_l2",
    "did_distribution_l2",
    "did_spectral_hybrid",
    "did_shape_mahalanobis",
    "did_target_logit_abs",
    "did_raw_l2",
    "did_level_l2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--prefix", default="gsdd_v06_cora_attack_generalization")
    parser.add_argument("--output-dir", default="results/attack_generalization_aggregate")
    return parser.parse_args()


def complete_run(path: Path) -> bool:
    required = [
        path / "summary.json",
        path / "paired_node_scores.csv",
        path / "paired_detection_metrics.json",
        path / "model_behavior.csv",
        path / "attack_diagnostics.json",
    ]
    if not all(item.exists() for item in required):
        return False
    try:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return summary.get("status") == "success" and summary.get("attack_family") in FAMILIES


def flatten_run(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_dir": str(path),
        "seed": int(summary["seed"]),
        "attack_family": summary.get("attack_family"),
        "trigger_motif": summary.get("trigger_motif"),
        "device": summary.get("device"),
        "victim_count": summary.get("victim_count"),
    }
    behavior = summary.get("model_behavior", {})
    for mode in ["none", "label_only", "trigger_only", "full"]:
        item = behavior.get(mode, {})
        row[f"{mode}_clean_accuracy"] = item.get("clean_accuracy")
        row[f"{mode}_asr"] = item.get("triggered_asr")
        row[f"{mode}_best_epoch"] = item.get("best_epoch")

    diagnostics = summary.get("attack_diagnostics", {})
    for key in [
        "provisional_surrogate_victim_target_rate",
        "provisional_surrogate_test_accuracy",
        "generator_surrogate_target_rate",
        "generator_surrogate_target_probability",
        "generated_neighbor_cosine",
        "generated_distribution_loss",
        "generated_mean_nonzero_features",
    ]:
        row[f"attack_{key}"] = diagnostics.get(key)

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
                "attack_family": summary["attack_family"],
                "mtime": path.stat().st_mtime,
                "summary": summary,
            }
        )
    if not candidates:
        raise RuntimeError(f"No complete runs match prefix {args.prefix!r}")

    candidate_frame = pd.DataFrame(
        [{k: v for k, v in item.items() if k != "summary"} for item in candidates]
    ).sort_values(["attack_family", "seed", "mtime"])
    candidate_frame.to_csv(output / "attack_generalization_run_candidates.csv", index=False)

    selected: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for (family, seed), group in candidate_frame.groupby(["attack_family", "seed"]):
        keep_index = group["mtime"].idxmax()
        keep_path = group.loc[keep_index, "run_dir"]
        for index, row in group.iterrows():
            if index != keep_index:
                duplicates.append(
                    {
                        "attack_family": family,
                        "seed": int(seed),
                        "ignored_run_dir": row["run_dir"],
                        "kept_run_dir": keep_path,
                    }
                )
        item = next(candidate for candidate in candidates if candidate["run_dir"] == keep_path)
        selected.append(flatten_run(Path(keep_path), item["summary"]))

    pd.DataFrame(duplicates).to_csv(output / "duplicate_run_candidates.csv", index=False)
    frame = pd.DataFrame(selected).sort_values(["attack_family", "seed"]).reset_index(drop=True)
    frame.to_csv(output / "attack_generalization_runs.csv", index=False)

    numeric = [c for c in frame.select_dtypes(include="number").columns if c != "seed"]
    group_stats = frame.groupby("attack_family")[numeric].agg(["mean", "std", "min", "max"])
    group_stats.to_csv(output / "attack_generalization_group_stats.csv")

    criteria_rows: list[dict[str, Any]] = []
    for row in frame.itertuples():
        criteria_rows.append(
            {
                "attack_family": row.attack_family,
                "seed": row.seed,
                "functional_backdoor": bool(
                    row.full_asr >= 0.80
                    and row.trigger_only_asr <= 0.30
                    and row.label_only_asr <= 0.20
                    and row.none_asr <= 0.20
                ),
                "shape_generalizes": bool(
                    row.did_shape_l2_auroc >= 0.70
                    and row.did_shape_l2_permutation_p <= 0.05
                ),
                "distribution_generalizes": bool(
                    row.did_distribution_l2_auroc >= 0.70
                    and row.did_distribution_l2_permutation_p <= 0.05
                ),
                "spectral_hybrid_generalizes": bool(
                    row.did_spectral_hybrid_auroc >= 0.75
                    and row.did_spectral_hybrid_permutation_p <= 0.05
                ),
            }
        )
    criteria = pd.DataFrame(criteria_rows)
    criteria.to_csv(output / "attack_generalization_success_criteria.csv", index=False)

    family_criteria = (
        criteria.groupby("attack_family")[
            [
                "functional_backdoor",
                "shape_generalizes",
                "distribution_generalizes",
                "spectral_hybrid_generalizes",
            ]
        ]
        .mean()
        .reset_index()
    )
    family_criteria.to_csv(output / "attack_generalization_family_pass_rates.csv", index=False)

    lines = [
        "# GSDD-v0.6 attack-family generalization aggregate",
        "",
        f"Unique family/seed runs: {len(frame)}",
        f"Complete candidate runs: {len(candidate_frame)}",
        f"Ignored duplicate runs: {len(duplicates)}",
        "",
        "## Attack behavior and spectral DID",
        "",
        "| Attack family | Full ASR | Trigger-only ASR | Clean ACC | Shape AUROC | Distribution AUROC | Spectral hybrid AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        group = frame[frame["attack_family"] == family]
        if group.empty:
            lines.append(f"| {family} | NA | NA | NA | NA | NA | NA |")
            continue
        lines.append(
            f"| {family} | {group['full_asr'].mean():.4f} ± {group['full_asr'].std():.4f} | "
            f"{group['trigger_only_asr'].mean():.4f} ± {group['trigger_only_asr'].std():.4f} | "
            f"{group['full_clean_accuracy'].mean():.4f} ± {group['full_clean_accuracy'].std():.4f} | "
            f"{group['did_shape_l2_auroc'].mean():.4f} ± {group['did_shape_l2_auroc'].std():.4f} | "
            f"{group['did_distribution_l2_auroc'].mean():.4f} ± {group['did_distribution_l2_auroc'].std():.4f} | "
            f"{group['did_spectral_hybrid_auroc'].mean():.4f} ± {group['did_spectral_hybrid_auroc'].std():.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "The spectral mechanism generalizes only when an attack forms a functional backdoor (`full` ASR high and control ASRs low) and the scale-invariant DID scores remain predictive with significant permutation tests. A high trigger-only ASR indicates that the generated trigger also behaves as a direct evasion perturbation; such a run must not be interpreted as a pure learned-backdoor mechanism.",
            "",
            "The adaptive and distribution-preserving families in this package are self-contained mechanism-faithful adapters inspired by UGBA and DPGBA. They are not verbatim executions of the original repositories and must be reported with the `_style_` names used here.",
            "",
            "See `attack_generalization_runs.csv`, `attack_generalization_group_stats.csv`, and `attack_generalization_success_criteria.csv` for all values.",
        ]
    )
    (output / "ATTACK_GENERALIZATION_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"Wrote {len(frame)} unique attack-family runs to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
