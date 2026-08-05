from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


FAMILIES = [
    "fixed_rare_clique",
    "ugba_style_binding_aware",
    "dpgba_style_binding_aware",
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
    parser.add_argument("--prefix", default="gsdd_v061_cora_attack_validity_repair")
    parser.add_argument("--output-dir", default="results/attack_validity_repair_aggregate")
    return parser.parse_args()


def complete_run(path: Path) -> bool:
    required = [
        path / "summary.json",
        path / "attack_validity.json",
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
    validity = summary.get("attack_validity", {})
    row: dict[str, Any] = {
        "run_dir": str(path),
        "seed": int(summary["seed"]),
        "attack_family": summary.get("attack_family"),
        "trigger_motif": summary.get("trigger_motif"),
        "device": summary.get("device"),
        "victim_count": summary.get("victim_count"),
        "valid_attack": bool(validity.get("is_valid", False)),
        "validity_status": validity.get("status"),
        "validity_reasons": ";".join(validity.get("reasons", [])),
        "control_asr_max": validity.get("control_asr_max"),
        "full_minus_trigger_only_gap": validity.get("full_minus_trigger_only_gap"),
    }
    behavior = summary.get("model_behavior", {})
    for mode in ["none", "label_only", "trigger_only", "full"]:
        item = behavior.get(mode, {})
        row[f"{mode}_clean_accuracy"] = item.get("clean_accuracy")
        row[f"{mode}_asr"] = item.get("triggered_asr")
        row[f"{mode}_best_epoch"] = item.get("best_epoch")

    diagnostics = summary.get("attack_diagnostics", {})
    for key in [
        "clean_surrogate_test_accuracy",
        "final_poison_target_rate",
        "final_clean_target_rate",
        "final_poison_target_probability",
        "final_clean_target_probability",
        "final_probability_binding_gap",
        "final_clean_original_prediction_rate",
        "final_generated_neighbor_cosine",
        "final_generated_distribution_loss",
        "final_generated_mean_nonzero_features",
    ]:
        row[f"attack_{key}"] = diagnostics.get(key)

    detection = summary.get("paired_detection", {})
    permutation = summary.get("permutation_tests", {})
    for score in SCORES:
        item = detection.get(score, {})
        row[f"{score}_auroc"] = item.get("auroc")
        row[f"{score}_auprc"] = item.get("auprc")
        row[f"{score}_fpr95"] = item.get("fpr_at_95_tpr")
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
        candidates.append({
            "run_dir": str(path),
            "seed": int(summary["seed"]),
            "attack_family": summary["attack_family"],
            "mtime": path.stat().st_mtime,
            "summary": summary,
        })
    if not candidates:
        raise RuntimeError(f"No complete runs match prefix {args.prefix!r}")

    candidate_frame = pd.DataFrame(
        [{k: v for k, v in item.items() if k != "summary"} for item in candidates]
    ).sort_values(["attack_family", "seed", "mtime"])
    candidate_frame.to_csv(output / "attack_validity_run_candidates.csv", index=False)

    selected: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for (family, seed), group in candidate_frame.groupby(["attack_family", "seed"]):
        keep_index = group["mtime"].idxmax()
        keep_path = group.loc[keep_index, "run_dir"]
        for index, row in group.iterrows():
            if index != keep_index:
                duplicates.append({
                    "attack_family": family,
                    "seed": int(seed),
                    "ignored_run_dir": row["run_dir"],
                    "kept_run_dir": keep_path,
                })
        item = next(candidate for candidate in candidates if candidate["run_dir"] == keep_path)
        selected.append(flatten_run(Path(keep_path), item["summary"]))

    pd.DataFrame(duplicates).to_csv(output / "duplicate_run_candidates.csv", index=False)
    frame = pd.DataFrame(selected).sort_values(["attack_family", "seed"]).reset_index(drop=True)
    frame.to_csv(output / "attack_validity_runs.csv", index=False)

    numeric = [column for column in frame.select_dtypes(include="number").columns if column != "seed"]
    frame.groupby("attack_family")[numeric].agg(["mean", "std", "min", "max"]).to_csv(
        output / "attack_validity_group_stats_all.csv"
    )
    valid = frame[frame["valid_attack"]].copy()
    if not valid.empty:
        valid.groupby("attack_family")[numeric].agg(["mean", "std", "min", "max"]).to_csv(
            output / "attack_validity_group_stats_valid_only.csv"
        )
    else:
        pd.DataFrame().to_csv(output / "attack_validity_group_stats_valid_only.csv")

    criteria_rows: list[dict[str, Any]] = []
    for row in frame.itertuples():
        criteria_rows.append({
            "attack_family": row.attack_family,
            "seed": row.seed,
            "functional_backdoor": bool(row.valid_attack),
            "shape_generalizes_if_valid": bool(
                row.valid_attack
                and row.did_shape_l2_auroc >= 0.70
                and row.did_shape_l2_permutation_p <= 0.05
            ),
            "distribution_generalizes_if_valid": bool(
                row.valid_attack
                and row.did_distribution_l2_auroc >= 0.70
                and row.did_distribution_l2_permutation_p <= 0.05
            ),
            "spectral_hybrid_generalizes_if_valid": bool(
                row.valid_attack
                and row.did_spectral_hybrid_auroc >= 0.75
                and row.did_spectral_hybrid_permutation_p <= 0.05
            ),
        })
    criteria = pd.DataFrame(criteria_rows)
    criteria.to_csv(output / "attack_validity_success_criteria.csv", index=False)
    criteria.groupby("attack_family")[
        [
            "functional_backdoor",
            "shape_generalizes_if_valid",
            "distribution_generalizes_if_valid",
            "spectral_hybrid_generalizes_if_valid",
        ]
    ].mean().reset_index().to_csv(output / "attack_validity_family_pass_rates.csv", index=False)

    lines = [
        "# GSDD-v0.6.1 binding-aware attack-validity repair",
        "",
        f"Unique family/seed runs: {len(frame)}",
        f"Valid functional attacks: {int(frame['valid_attack'].sum())}/{len(frame)}",
        f"Ignored duplicate runs: {len(duplicates)}",
        "",
        "## Functional controls and spectral DID",
        "",
        "| Attack family | Valid runs | Full ASR | Control max ASR | Binding gap | Spectral hybrid AUROC (valid only) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        group = frame[frame["attack_family"] == family]
        valid_group = group[group["valid_attack"]]
        if group.empty:
            lines.append(f"| {family} | 0/0 | NA | NA | NA | NA |")
            continue
        hybrid = "NA" if valid_group.empty else f"{valid_group['did_spectral_hybrid_auroc'].mean():.4f}"
        lines.append(
            f"| {family} | {int(group['valid_attack'].sum())}/{len(group)} | "
            f"{group['full_asr'].mean():.4f} | {group['control_asr_max'].mean():.4f} | "
            f"{group['full_minus_trigger_only_gap'].mean():.4f} | {hybrid} |"
        )
    lines.extend([
        "",
        "## Admission rule",
        "",
        "A run is a valid functional backdoor only when `full ASR >= 0.80`, every control ASR is at most `0.10`, and `full ASR - trigger-only ASR >= 0.60`. Spectral generalization is reported only for admitted runs. Invalid attacks remain in the audit tables but are excluded from the method claim.",
        "",
        "The learned families are self-contained binding-aware adapters, not verbatim official UGBA or DPGBA executions.",
    ])
    (output / "ATTACK_VALIDITY_REPAIR_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(frame)} unique runs to {output}; valid={int(frame['valid_attack'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
