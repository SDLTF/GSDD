from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

FAMILIES = [
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
    parser.add_argument("--prefix", default="gsdd_v062_repair2")
    parser.add_argument(
        "--output-dir", default="results/attack_validity_repair_round2_aggregate"
    )
    return parser.parse_args()


def complete_run(path: Path) -> bool:
    required = [
        path / "summary.json",
        path / "attack_validity.json",
        path / "paired_node_scores.csv",
        path / "paired_detection_metrics.json",
        path / "model_behavior.csv",
        path / "attack_diagnostics.json",
        path / "config_resolved.yaml",
    ]
    if not all(item.exists() for item in required):
        return False
    try:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return summary.get("status") == "success" and summary.get("attack_family") in FAMILIES


def read_config(path: Path) -> dict[str, Any]:
    with (path / "config_resolved.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def flatten_run(path: Path, summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    validity = summary.get("attack_validity", {})
    attack_cfg = config.get("attack", {})
    behavior = summary.get("model_behavior", {})
    row: dict[str, Any] = {
        "run_dir": str(path),
        "run_name": path.name,
        "seed": int(summary["seed"]),
        "attack_family": summary.get("attack_family"),
        "target_class": int(summary.get("target_class", attack_cfg.get("target_class", 0))),
        "selection_method": summary.get(
            "selection_method", attack_cfg.get("selection_method", "dirty_label")
        ),
        "poison_count": int(summary.get("victim_count", attack_cfg.get("poison_count", 0))),
        "trigger_size": int(attack_cfg.get("trigger_size", 0)),
        "trigger_motif": summary.get("trigger_motif"),
        "valid_attack": bool(validity.get("is_valid", False)),
        "validity_status": validity.get("status"),
        "validity_reasons": ";".join(validity.get("reasons", [])),
        "control_asr_max": float(validity.get("control_asr_max", float("nan"))),
        "full_minus_trigger_only_gap": float(
            validity.get("full_minus_trigger_only_gap", float("nan"))
        ),
    }
    for mode in ["none", "label_only", "trigger_only", "full"]:
        item = behavior.get(mode, {})
        row[f"{mode}_clean_accuracy"] = item.get("clean_accuracy")
        row[f"{mode}_asr"] = item.get("triggered_asr")
        row[f"{mode}_best_epoch"] = item.get("best_epoch")

    diagnostics = summary.get("attack_diagnostics", {})
    for key in [
        "final_poison_target_rate",
        "final_clean_target_rate",
        "final_label_only_target_rate",
        "final_poison_target_probability",
        "final_clean_target_probability",
        "final_label_only_target_probability",
        "final_probability_binding_gap",
        "final_clean_original_prediction_rate",
        "final_label_only_original_prediction_rate",
        "final_generated_neighbor_cosine",
        "final_generated_distribution_loss",
        "final_generated_mean_nonzero_features",
        "distribution_target_prototype_fraction",
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

    full_asr = float(row.get("full_asr") or 0.0)
    control = float(row.get("control_asr_max") or 0.0)
    gap = float(row.get("full_minus_trigger_only_gap") or 0.0)
    row["admission_distance"] = (
        max(0.0, 0.80 - full_asr)
        + max(0.0, control - 0.10)
        + max(0.0, 0.60 - gap)
    )
    # Used only for ranking repair candidates, never as a replacement for the
    # hard admission gate.
    row["repair_priority_score"] = (
        4.0 * float(row["valid_attack"])
        + full_asr
        + gap
        - 2.0 * control
        - row["admission_distance"]
    )
    return row


def variant_key(row: pd.Series | dict[str, Any]) -> str:
    return (
        f"{row['attack_family']}|t{int(row['target_class'])}|"
        f"{row['selection_method']}|pc{int(row['poison_count'])}|"
        f"ts{int(row['trigger_size'])}"
    )


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
        config = read_config(path)
        row = flatten_run(path, summary, config)
        row["mtime"] = path.stat().st_mtime
        row["variant_key"] = variant_key(row)
        candidates.append(row)
    if not candidates:
        raise RuntimeError(f"No complete runs match prefix {args.prefix!r}")

    candidate_frame = pd.DataFrame(candidates).sort_values(
        ["attack_family", "target_class", "selection_method", "poison_count", "seed", "mtime"]
    )
    candidate_frame.to_csv(output / "repair2_run_candidates.csv", index=False)

    selected_rows: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    group_cols = [
        "attack_family",
        "target_class",
        "selection_method",
        "poison_count",
        "trigger_size",
        "seed",
    ]
    for _, group in candidate_frame.groupby(group_cols, dropna=False):
        keep_index = group["mtime"].idxmax()
        keep = group.loc[keep_index].to_dict()
        selected_rows.append(keep)
        for index, row in group.iterrows():
            if index != keep_index:
                duplicates.append(
                    {
                        "ignored_run_dir": row["run_dir"],
                        "kept_run_dir": keep["run_dir"],
                        "variant_key": keep["variant_key"],
                        "seed": int(keep["seed"]),
                    }
                )

    frame = pd.DataFrame(selected_rows).drop(columns=["mtime"]).sort_values(
        ["attack_family", "target_class", "selection_method", "poison_count", "seed"]
    )
    frame.to_csv(output / "repair2_runs.csv", index=False)
    pd.DataFrame(duplicates).to_csv(output / "duplicate_run_candidates.csv", index=False)

    numeric = [c for c in frame.select_dtypes(include="number").columns if c != "seed"]
    frame.groupby(["attack_family", "target_class", "selection_method", "poison_count"])[
        numeric
    ].agg(["mean", "std", "min", "max"]).to_csv(output / "repair2_group_stats.csv")

    ranking = (
        frame.groupby(
            ["attack_family", "target_class", "selection_method", "poison_count", "trigger_size"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "count"),
            valid_rate=("valid_attack", "mean"),
            full_asr=("full_asr", "mean"),
            control_asr_max=("control_asr_max", "mean"),
            binding_gap=("full_minus_trigger_only_gap", "mean"),
            admission_distance=("admission_distance", "mean"),
            repair_priority_score=("repair_priority_score", "mean"),
            spectral_hybrid_auroc=("did_spectral_hybrid_auroc", "mean"),
            full_clean_accuracy=("full_clean_accuracy", "mean"),
        )
        .sort_values(
            ["attack_family", "valid_rate", "admission_distance", "repair_priority_score"],
            ascending=[True, False, True, False],
        )
    )
    ranking["variant_key"] = ranking.apply(variant_key, axis=1)
    ranking.to_csv(output / "repair2_candidate_ranking.csv", index=False)

    chosen: list[dict[str, Any]] = []
    for family in FAMILIES:
        group = ranking[ranking["attack_family"] == family]
        if group.empty:
            continue
        valid = group[group["valid_rate"] > 0]
        best = (valid if not valid.empty else group).iloc[0]
        chosen.append(
            {
                "attack_family": family,
                "target_class": int(best["target_class"]),
                "selection_method": str(best["selection_method"]),
                "poison_count": int(best["poison_count"]),
                "trigger_size": int(best["trigger_size"]),
                "variant_key": str(best["variant_key"]),
                "pilot_valid": bool(best["valid_rate"] > 0),
                "pilot_valid_rate": float(best["valid_rate"]),
                "pilot_full_asr": float(best["full_asr"]),
                "pilot_control_asr_max": float(best["control_asr_max"]),
                "pilot_binding_gap": float(best["binding_gap"]),
                "pilot_admission_distance": float(best["admission_distance"]),
            }
        )
    (output / "selected_candidates.json").write_text(
        json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    criteria = frame[
        [
            "attack_family",
            "target_class",
            "selection_method",
            "poison_count",
            "seed",
            "valid_attack",
            "full_asr",
            "control_asr_max",
            "full_minus_trigger_only_gap",
            "did_spectral_hybrid_auroc",
            "did_spectral_hybrid_permutation_p",
        ]
    ].copy()
    criteria["spectral_generalizes_if_valid"] = (
        criteria["valid_attack"]
        & (criteria["did_spectral_hybrid_auroc"] >= 0.75)
        & (criteria["did_spectral_hybrid_permutation_p"] <= 0.05)
    )
    criteria.to_csv(output / "repair2_success_criteria.csv", index=False)

    lines = [
        "# GSDD-v0.6.2 Attack Validity Repair Round 2",
        "",
        f"Unique candidate/seed runs: {len(frame)}",
        f"Valid functional attacks: {int(frame['valid_attack'].sum())}/{len(frame)}",
        f"Ignored duplicate runs: {len(duplicates)}",
        "",
        "## Candidate ranking",
        "",
        "| Family | Target | Selection | Poison count | Seeds | Valid rate | Full ASR | Control max | Binding gap | Admission distance |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranking.itertuples():
        lines.append(
            f"| {row.attack_family} | {row.target_class} | {row.selection_method} | "
            f"{row.poison_count} | {row.seeds} | {row.valid_rate:.3f} | "
            f"{row.full_asr:.3f} | {row.control_asr_max:.3f} | "
            f"{row.binding_gap:.3f} | {row.admission_distance:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Automatic expansion rule",
            "",
            "Only a pilot candidate that passes the hard attack-validity gate is expanded to additional seeds. Near-valid candidates are ranked and reported, but are not promoted automatically.",
            "",
            "UGBA candidates reduce dirty-label poison count and vary target class to control label-only leakage. DPGBA candidates use clean-label target-class victims and a mixed target/global prototype bank to strengthen trigger-target binding without relying on relabeling.",
        ]
    )
    (output / "ATTACK_VALIDITY_REPAIR_ROUND2_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        f"Wrote {len(frame)} runs to {output}; valid={int(frame['valid_attack'].sum())}; "
        f"selected={len(chosen)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
