from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate GSDD-v0.6.4 dual clean-label audit")
    p.add_argument("--results-root", default="results")
    p.add_argument("--prefix", default="gsdd_v064_dual_cl")
    p.add_argument("--output-dir", default="results/dual_clean_label_aggregate")
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
    mode = str(summary.get("attack_mode", validity.get("attack_mode", "unknown")))
    main_score = "cl_generic_spectral_hybrid" if mode == "generic" else "cl_did_spectral_hybrid"
    return {
        "run_dir": str(run_dir),
        "seed": int(summary.get("seed", -1)),
        "attack_mode": mode,
        "target_class": int(summary.get("target_class", -1)),
        "poison_count": int(summary.get("victim_count", -1)),
        "trigger_size": int(summary.get("trigger_size", -1)),
        "pair_weight": float(summary.get("contextual_pair_weight", float("nan"))),
        "attack_family": summary.get("attack_family", ""),
        "trigger_motif": summary.get("trigger_motif", ""),
        "is_valid": bool(validity.get("is_valid", False)),
        "validity_status": validity.get("status", "unknown"),
        "validity_reasons": json.dumps(validity.get("reasons", []), ensure_ascii=False),
        "full_asr": float(validity.get("full_asr", float("nan"))),
        "control_asr_max": float(validity.get("control_asr_max", float("nan"))),
        "admission_gap": float(validity.get("binding_gap", float("nan"))),
        "contextual_gap": float(validity.get("contextual_binding_gap", float("nan"))),
        "generic_did": float(validity.get("generic_did", float("nan"))),
        "clean_none_asr": float(clean.get("none", {}).get("asr", float("nan"))),
        "clean_matched_asr": float(clean.get("matched", {}).get("asr", float("nan"))),
        "clean_shuffled_asr": float(clean.get("shuffled", {}).get("asr", float("nan"))),
        "poison_none_asr": float(poison.get("none", {}).get("asr", float("nan"))),
        "poison_matched_asr": float(poison.get("matched", {}).get("asr", float("nan"))),
        "poison_shuffled_asr": float(poison.get("shuffled", {}).get("asr", float("nan"))),
        "clean_accuracy": float(poison.get("none", {}).get("clean_accuracy", float("nan"))),
        "main_score_name": main_score,
        "spectral_auroc": metric(summary, main_score, "auroc"),
        "spectral_auprc": metric(summary, main_score, "auprc"),
        "spectral_fpr95": metric(summary, main_score, "fpr_at_95_tpr"),
        "spectral_permutation_p": float(
            summary.get("permutation_tests", {}).get(main_score, {}).get("p_value", float("nan"))
        ),
    }


def admission_distance(row: pd.Series) -> float:
    if row.attack_mode == "generic":
        return float(
            max(0.0, 0.80 - row.full_asr)
            + max(0.0, row.control_asr_max - 0.20)
            + max(0.0, 0.40 - row.generic_did)
        )
    return float(
        max(0.0, 0.80 - row.full_asr)
        + max(0.0, row.control_asr_max - 0.20)
        + max(0.0, 0.40 - row.contextual_gap)
    )


def main() -> int:
    args = parse_args()
    root = Path(args.results_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], tuple[Path, float]] = {}
    for run_dir in sorted(root.glob(f"{args.prefix}*seed*")):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        if summary.get("status") != "success":
            continue
        row = row_from_summary(run_dir, summary)
        key = (
            row["attack_mode"], row["target_class"], row["poison_count"],
            row["trigger_size"], round(row["pair_weight"], 6), row["seed"],
        )
        modified = summary_path.stat().st_mtime
        if key in seen:
            old_path, old_modified = seen[key]
            if modified <= old_modified:
                duplicates.append({"ignored": str(run_dir), "kept": str(old_path), "key": str(key)})
                continue
            rows = [r for r in rows if not (
                r["attack_mode"] == key[0] and r["target_class"] == key[1]
                and r["poison_count"] == key[2] and r["trigger_size"] == key[3]
                and round(r["pair_weight"], 6) == key[4] and r["seed"] == key[5]
            )]
            duplicates.append({"ignored": str(old_path), "kept": str(run_dir), "key": str(key)})
        seen[key] = (run_dir, modified)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "dual_clean_label_runs.csv", index=False)
    pd.DataFrame(duplicates).to_csv(output / "duplicate_run_candidates.csv", index=False)
    if frame.empty:
        (output / "DUAL_CLEAN_LABEL_SUMMARY.md").write_text(
            "# GSDD-v0.6.4 Dual Clean-label Audit\n\nNo successful runs found.\n",
            encoding="utf-8",
        )
        (output / "selected_candidates.json").write_text("[]\n", encoding="utf-8")
        return 0

    group_cols = ["attack_mode", "target_class", "poison_count", "trigger_size", "pair_weight"]
    groups: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, dropna=False):
        valid = group[group["is_valid"]]
        row = dict(zip(group_cols, key))
        row.update({
            "run_count": int(len(group)),
            "valid_count": int(len(valid)),
            "valid_rate": float(len(valid) / len(group)),
        })
        for col in ["full_asr", "control_asr_max", "admission_gap", "clean_accuracy", "spectral_auroc", "spectral_auprc", "spectral_fpr95"]:
            vals = pd.to_numeric(valid[col], errors="coerce").dropna()
            row[f"{col}_mean_valid"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std_valid"] = float(vals.std(ddof=1)) if len(vals) > 1 else (0.0 if len(vals) else float("nan"))
        groups.append(row)
    group_frame = pd.DataFrame(groups).sort_values(
        ["attack_mode", "valid_rate", "full_asr_mean_valid", "admission_gap_mean_valid"],
        ascending=[True, False, False, False],
    )
    group_frame.to_csv(output / "dual_clean_label_group_stats.csv", index=False)

    pilots = frame[frame["seed"] == args.pilot_seed].copy()
    pilots["admission_distance"] = pilots.apply(admission_distance, axis=1)
    selected: list[dict[str, Any]] = []
    for mode, mode_rows in pilots.groupby("attack_mode"):
        ordered = mode_rows.sort_values(
            ["is_valid", "admission_distance", "full_asr", "admission_gap", "spectral_auroc"],
            ascending=[False, True, False, False, False],
        )
        best = ordered.iloc[0]
        selected.append({
            "attack_mode": str(mode),
            "target_class": int(best.target_class),
            "poison_count": int(best.poison_count),
            "trigger_size": int(best.trigger_size),
            "pair_weight": float(best.pair_weight),
            "pilot_seed": int(best.seed),
            "pilot_valid": bool(best.is_valid),
            "pilot_full_asr": float(best.full_asr),
            "pilot_control_asr_max": float(best.control_asr_max),
            "pilot_admission_gap": float(best.admission_gap),
            "pilot_spectral_auroc": float(best.spectral_auroc),
            "pilot_admission_distance": float(best.admission_distance),
        })
    (output / "selected_candidates.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    success_rows = []
    for mode, mode_rows in frame.groupby("attack_mode"):
        valid = mode_rows[mode_rows["is_valid"]]
        success_rows.append({
            "attack_mode": mode,
            "total_runs": int(len(mode_rows)),
            "valid_runs": int(len(valid)),
            "valid_rate": float(len(valid) / len(mode_rows)),
            "valid_spectral_auroc_mean": float(valid["spectral_auroc"].mean()) if len(valid) else float("nan"),
            "valid_permutation_pass_rate": float((valid["spectral_permutation_p"] < 0.05).mean()) if len(valid) else float("nan"),
        })
    pd.DataFrame(success_rows).to_csv(output / "dual_clean_label_success_criteria.csv", index=False)

    lines = [
        "# GSDD-v0.6.4 Dual Clean-label Aggregate",
        "",
        f"- Unique successful runs: `{len(frame)}`",
        f"- Valid attacks: `{int(frame['is_valid'].sum())}`",
        "",
        "## Candidate groups",
        "",
        "| Mode | Target | Poison | Trigger size | Pair weight | Runs | Valid | Full ASR | Control | Gap | Spectral AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in group_frame.iterrows():
        lines.append(
            f"| {r.attack_mode} | {int(r.target_class)} | {int(r.poison_count)} | {int(r.trigger_size)} | "
            f"{r.pair_weight:.2f} | {int(r.run_count)} | {int(r.valid_count)} | "
            f"{r.full_asr_mean_valid:.4f} | {r.control_asr_max_mean_valid:.4f} | "
            f"{r.admission_gap_mean_valid:.4f} | {r.spectral_auroc_mean_valid:.4f} |"
        )
    lines.extend([
        "",
        "Generic and contextual attacks use separate admission rules. Detection metrics are summarized as defense evidence only for runs that pass the corresponding functional gate.",
    ])
    (output / "DUAL_CLEAN_LABEL_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote aggregate to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
