from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate GSDD-v0.6.5 generic selective-activation repair")
    p.add_argument("--results-root", default="results")
    p.add_argument("--prefix", default="gsdd_v065_generic_selective")
    p.add_argument("--output-dir", default="results/generic_selective_repair_aggregate")
    p.add_argument("--pilot-seed", type=int, default=1027)
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: dict[str, Any], name: str, field: str) -> float:
    try:
        return float(summary.get("clean_label_detection", {}).get(name, {}).get(field, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def row_from_summary(run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    validity = summary.get("attack_validity", {})
    behavior = summary.get("factorial_behavior", {})
    clean = behavior.get("clean", {})
    poison = behavior.get("poison", {})
    params = summary.get("generic_repair_parameters", {})
    return {
        "run_dir": str(run_dir),
        "seed": int(summary.get("seed", -1)),
        "target_class": int(summary.get("target_class", -1)),
        "poison_count": int(summary.get("victim_count", -1)),
        "trigger_size": int(summary.get("trigger_size", -1)),
        "trigger_motif": summary.get("trigger_motif", ""),
        "is_valid": bool(validity.get("is_valid", False)),
        "validity_status": validity.get("status", "unknown"),
        "validity_reasons": json.dumps(validity.get("reasons", []), ensure_ascii=False),
        "full_asr": float(validity.get("full_asr", float("nan"))),
        "control_asr_max": float(validity.get("control_asr_max", float("nan"))),
        "generic_did": float(validity.get("generic_did", float("nan"))),
        "clean_none_asr": float(clean.get("none", {}).get("asr", float("nan"))),
        "clean_matched_asr": float(clean.get("matched", {}).get("asr", float("nan"))),
        "clean_shuffled_asr": float(clean.get("shuffled", {}).get("asr", float("nan"))),
        "poison_none_asr": float(poison.get("none", {}).get("asr", float("nan"))),
        "poison_matched_asr": float(poison.get("matched", {}).get("asr", float("nan"))),
        "poison_shuffled_asr": float(poison.get("shuffled", {}).get("asr", float("nan"))),
        "clean_accuracy": float(poison.get("none", {}).get("clean_accuracy", float("nan"))),
        "clean_cap_weight": float(params.get("clean_cap_weight", float("nan"))),
        "clean_probability_cap": float(params.get("clean_probability_cap", float("nan"))),
        "selectivity_weight": float(params.get("selectivity_weight", float("nan"))),
        "selectivity_margin": float(params.get("selectivity_margin", float("nan"))),
        "target_similarity_weight": float(params.get("target_similarity_weight", float("nan"))),
        "target_similarity_allowance": float(params.get("target_similarity_allowance", float("nan"))),
        "raw_blend": float(params.get("raw_blend", float("nan"))),
        "target_prototype_fraction": float(params.get("target_prototype_fraction", float("nan"))),
        "outer_rounds": int(params.get("outer_rounds", -1)),
        "poison_target_weight": float(params.get("poison_target_weight", float("nan"))),
        "shuffled_target_weight": float(params.get("shuffled_target_weight", float("nan"))),
        "spectral_auroc": metric(summary, "cl_generic_spectral_hybrid", "auroc"),
        "spectral_auprc": metric(summary, "cl_generic_spectral_hybrid", "auprc"),
        "spectral_fpr95": metric(summary, "cl_generic_spectral_hybrid", "fpr_at_95_tpr"),
        "spectral_permutation_p": float(
            summary.get("permutation_tests", {})
            .get("cl_generic_spectral_hybrid", {})
            .get("p_value", float("nan"))
        ),
    }


def admission_distance(row: pd.Series) -> float:
    return float(
        max(0.0, 0.80 - row.full_asr)
        + max(0.0, row.control_asr_max - 0.20)
        + max(0.0, 0.40 - row.generic_did)
    )


def main() -> int:
    args = parse_args()
    root = Path(args.results_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], tuple[Path, float]] = {}

    parameter_keys = (
        "trigger_size", "clean_cap_weight", "clean_probability_cap",
        "selectivity_weight", "selectivity_margin", "target_similarity_weight",
        "target_similarity_allowance", "raw_blend", "target_prototype_fraction",
        "outer_rounds", "poison_target_weight", "shuffled_target_weight",
    )
    for run_dir in sorted(root.glob(f"{args.prefix}*seed*")):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        if summary.get("status") != "success":
            continue
        row = row_from_summary(run_dir, summary)
        key = tuple(round(row[k], 8) if isinstance(row[k], float) else row[k] for k in parameter_keys) + (row["seed"],)
        modified = summary_path.stat().st_mtime
        if key in seen:
            old_path, old_modified = seen[key]
            if modified <= old_modified:
                duplicates.append({"ignored": str(run_dir), "kept": str(old_path), "key": str(key)})
                continue
            rows = [r for r in rows if r["run_dir"] != str(old_path)]
            duplicates.append({"ignored": str(old_path), "kept": str(run_dir), "key": str(key)})
        seen[key] = (run_dir, modified)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "generic_selective_repair_runs.csv", index=False)
    pd.DataFrame(duplicates).to_csv(output / "duplicate_run_candidates.csv", index=False)
    if frame.empty:
        (output / "GENERIC_SELECTIVE_REPAIR_SUMMARY.md").write_text(
            "# GSDD-v0.6.5 Generic Selective-activation Repair\n\nNo successful runs found.\n",
            encoding="utf-8",
        )
        (output / "selected_candidate.json").write_text("{}\n", encoding="utf-8")
        return 0

    pilots = frame[frame["seed"] == args.pilot_seed].copy()
    pilots["admission_distance"] = pilots.apply(admission_distance, axis=1)
    ordered = pilots.sort_values(
        ["is_valid", "admission_distance", "full_asr", "generic_did", "spectral_auroc"],
        ascending=[False, True, False, False, False],
    )
    best = ordered.iloc[0]
    selected = {k: (bool(best[k]) if k == "is_valid" else float(best[k]) if isinstance(best[k], float) else int(best[k]) if k in {"seed", "target_class", "poison_count", "trigger_size", "outer_rounds"} else best[k]) for k in []}
    selected = {
        "pilot_valid": bool(best.is_valid),
        "pilot_seed": int(best.seed),
        "target_class": int(best.target_class),
        "poison_count": int(best.poison_count),
        "trigger_size": int(best.trigger_size),
        "clean_cap_weight": float(best.clean_cap_weight),
        "clean_probability_cap": float(best.clean_probability_cap),
        "selectivity_weight": float(best.selectivity_weight),
        "selectivity_margin": float(best.selectivity_margin),
        "target_similarity_weight": float(best.target_similarity_weight),
        "target_similarity_allowance": float(best.target_similarity_allowance),
        "raw_blend": float(best.raw_blend),
        "target_prototype_fraction": float(best.target_prototype_fraction),
        "outer_rounds": int(best.outer_rounds),
        "poison_target_weight": float(best.poison_target_weight),
        "shuffled_target_weight": float(best.shuffled_target_weight),
        "pilot_full_asr": float(best.full_asr),
        "pilot_control_asr_max": float(best.control_asr_max),
        "pilot_generic_did": float(best.generic_did),
        "pilot_spectral_auroc": float(best.spectral_auroc),
        "pilot_admission_distance": float(best.admission_distance),
    }
    (output / "selected_candidate.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    group_cols = list(parameter_keys)
    stats: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, dropna=False):
        valid = group[group["is_valid"]]
        item = dict(zip(group_cols, key))
        item.update({
            "run_count": int(len(group)),
            "valid_count": int(len(valid)),
            "valid_rate": float(len(valid) / len(group)),
        })
        for col in ("full_asr", "control_asr_max", "generic_did", "clean_accuracy", "spectral_auroc", "spectral_auprc", "spectral_fpr95"):
            values = pd.to_numeric(valid[col], errors="coerce").dropna()
            item[f"{col}_mean_valid"] = float(values.mean()) if len(values) else float("nan")
            item[f"{col}_std_valid"] = float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else float("nan"))
        stats.append(item)
    pd.DataFrame(stats).to_csv(output / "generic_selective_repair_group_stats.csv", index=False)

    pd.DataFrame([
        {
            "total_runs": int(len(frame)),
            "valid_runs": int(frame["is_valid"].sum()),
            "valid_rate": float(frame["is_valid"].mean()),
            "valid_spectral_auroc_mean": float(frame.loc[frame["is_valid"], "spectral_auroc"].mean()) if frame["is_valid"].any() else float("nan"),
            "valid_permutation_pass_rate": float((frame.loc[frame["is_valid"], "spectral_permutation_p"] < 0.05).mean()) if frame["is_valid"].any() else float("nan"),
        }
    ]).to_csv(output / "generic_selective_repair_success_criteria.csv", index=False)

    lines = [
        "# GSDD-v0.6.5 Generic Selective-activation Repair",
        "",
        f"- Unique successful runs: `{len(frame)}`",
        f"- Valid attacks: `{int(frame['is_valid'].sum())}`",
        "",
        "## Pilot candidates",
        "",
        "| Trigger | Cap weight | Cap | Selectivity weight | Margin | Target-sim weight | Raw blend | Target-prototype fraction | Full ASR | Control | DiD | Valid |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in pilots.sort_values("admission_distance").iterrows():
        lines.append(
            f"| {int(row.trigger_size)} | {row.clean_cap_weight:.1f} | {row.clean_probability_cap:.2f} | "
            f"{row.selectivity_weight:.1f} | {row.selectivity_margin:.2f} | {row.target_similarity_weight:.1f} | "
            f"{row.raw_blend:.2f} | {row.target_prototype_fraction:.2f} | {row.full_asr:.3f} | "
            f"{row.control_asr_max:.3f} | {row.generic_did:.3f} | {bool(row.is_valid)} |"
        )
    lines.extend([
        "",
        "## Selected candidate",
        "",
        f"- Pilot valid: `{selected['pilot_valid']}`",
        f"- Full ASR: `{selected['pilot_full_asr']:.4f}`",
        f"- Maximum control ASR: `{selected['pilot_control_asr_max']:.4f}`",
        f"- Generic DiD: `{selected['pilot_generic_did']:.4f}`",
        f"- Admission distance: `{selected['pilot_admission_distance']:.4f}`",
        "",
        "The experiment isolates clean-model trigger activation as the repair target. Detection metrics count as defense evidence only for functionally valid attacks.",
    ])
    (output / "GENERIC_SELECTIVE_REPAIR_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
