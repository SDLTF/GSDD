from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate GSDD-v0.6.3 clean-label factorial runs")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--prefix", default="gsdd_v063_clean_label")
    parser.add_argument("--output-dir", default="results/clean_label_factorial_aggregate")
    parser.add_argument("--pilot-seed", type=int, default=1027)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_metric(summary: dict[str, Any], name: str, field: str) -> float:
    item = summary.get("clean_label_detection", {}).get(name, {})
    try:
        return float(item.get(field, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def row_from_summary(run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    validity = summary.get("attack_validity", {})
    behavior = summary.get("factorial_behavior", {})
    clean = behavior.get("clean", {})
    poison = behavior.get("poison", {})
    return {
        "run_dir": str(run_dir),
        "seed": int(summary.get("seed", -1)),
        "target_class": int(summary.get("target_class", -1)),
        "poison_count": int(summary.get("victim_count", -1)),
        "attack_family": summary.get("attack_family", ""),
        "trigger_motif": summary.get("trigger_motif", ""),
        "is_valid": bool(validity.get("is_valid", False)),
        "validity_status": validity.get("status", "unknown"),
        "validity_reasons": json.dumps(validity.get("reasons", []), ensure_ascii=False),
        "full_asr": float(validity.get("full_asr", float("nan"))),
        "control_asr_max": float(validity.get("control_asr_max", float("nan"))),
        "binding_gap": float(validity.get("binding_gap", float("nan"))),
        "clean_none_asr": float(clean.get("none", {}).get("asr", float("nan"))),
        "clean_matched_asr": float(clean.get("matched", {}).get("asr", float("nan"))),
        "clean_shuffled_asr": float(clean.get("shuffled", {}).get("asr", float("nan"))),
        "poison_none_asr": float(poison.get("none", {}).get("asr", float("nan"))),
        "poison_matched_asr": float(poison.get("matched", {}).get("asr", float("nan"))),
        "poison_shuffled_asr": float(poison.get("shuffled", {}).get("asr", float("nan"))),
        "clean_accuracy": float(poison.get("none", {}).get("clean_accuracy", float("nan"))),
        "spectral_auroc": scalar_metric(summary, "cl_did_spectral_hybrid", "auroc"),
        "spectral_auprc": scalar_metric(summary, "cl_did_spectral_hybrid", "auprc"),
        "spectral_fpr95": scalar_metric(summary, "cl_did_spectral_hybrid", "fpr_at_95_tpr"),
        "shape_auroc": scalar_metric(summary, "cl_did_shape_l2", "auroc"),
        "distribution_auroc": scalar_metric(summary, "cl_did_distribution_l2", "auroc"),
        "logit_auroc": scalar_metric(summary, "cl_did_target_logit_abs", "auroc"),
        "spectral_permutation_p": float(
            summary.get("permutation_tests", {})
            .get("cl_did_spectral_hybrid", {})
            .get("p_value", float("nan"))
        ),
    }


def main() -> int:
    args = parse_args()
    root = Path(args.results_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    seen: dict[tuple[int, int, int], tuple[Path, float]] = {}
    for run_dir in sorted(root.glob(f"{args.prefix}*seed*")):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        if summary.get("status") != "success":
            continue
        row = row_from_summary(run_dir, summary)
        key = (row["target_class"], row["poison_count"], row["seed"])
        modified = summary_path.stat().st_mtime
        if key in seen:
            old_path, old_modified = seen[key]
            if modified <= old_modified:
                duplicate_rows.append({"ignored": str(run_dir), "kept": str(old_path), "key": str(key)})
                continue
            rows = [item for item in rows if not (
                item["target_class"] == key[0]
                and item["poison_count"] == key[1]
                and item["seed"] == key[2]
            )]
            duplicate_rows.append({"ignored": str(old_path), "kept": str(run_dir), "key": str(key)})
        seen[key] = (run_dir, modified)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "clean_label_factorial_runs.csv", index=False)
    pd.DataFrame(duplicate_rows).to_csv(output / "duplicate_run_candidates.csv", index=False)
    if frame.empty:
        (output / "CLEAN_LABEL_FACTORIAL_SUMMARY.md").write_text(
            "# GSDD-v0.6.3 Clean-label Factorial Audit\n\nNo successful runs found.\n",
            encoding="utf-8",
        )
        (output / "selected_candidates.json").write_text("[]\n", encoding="utf-8")
        print(f"No runs found under {root}")
        return 0

    group_rows: list[dict[str, Any]] = []
    for (target, poison), group in frame.groupby(["target_class", "poison_count"]):
        valid = group[group["is_valid"]]
        row: dict[str, Any] = {
            "target_class": int(target),
            "poison_count": int(poison),
            "run_count": int(len(group)),
            "valid_count": int(len(valid)),
            "valid_rate": float(len(valid) / len(group)),
        }
        for column in [
            "full_asr",
            "control_asr_max",
            "binding_gap",
            "clean_accuracy",
            "spectral_auroc",
            "spectral_auprc",
            "spectral_fpr95",
        ]:
            values = pd.to_numeric(valid[column], errors="coerce").dropna()
            row[f"{column}_mean_valid"] = float(values.mean()) if len(values) else float("nan")
            row[f"{column}_std_valid"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else float("nan")
        group_rows.append(row)
    group_frame = pd.DataFrame(group_rows).sort_values(
        ["valid_rate", "full_asr_mean_valid", "binding_gap_mean_valid"], ascending=False
    )
    group_frame.to_csv(output / "clean_label_factorial_group_stats.csv", index=False)

    pilots = frame[frame["seed"] == args.pilot_seed].copy()
    selected: list[dict[str, Any]] = []
    if not pilots.empty:
        pilots["admission_distance"] = (
            np.maximum(0.0, 0.80 - pilots["full_asr"].to_numpy())
            + np.maximum(0.0, pilots["control_asr_max"].to_numpy() - 0.20)
            + np.maximum(0.0, 0.40 - pilots["binding_gap"].to_numpy())
        )
        pilots = pilots.sort_values(
            ["is_valid", "admission_distance", "full_asr", "binding_gap", "spectral_auroc"],
            ascending=[False, True, False, False, False],
        )
        best = pilots.iloc[0]
        selected.append(
            {
                "attack_family": str(best["attack_family"]),
                "target_class": int(best["target_class"]),
                "poison_count": int(best["poison_count"]),
                "selection_method": "clean_label",
                "pilot_seed": int(best["seed"]),
                "pilot_valid": bool(best["is_valid"]),
                "pilot_full_asr": float(best["full_asr"]),
                "pilot_control_asr_max": float(best["control_asr_max"]),
                "pilot_binding_gap": float(best["binding_gap"]),
                "pilot_spectral_auroc": float(best["spectral_auroc"]),
                "pilot_admission_distance": float(best["admission_distance"]),
            }
        )
    (output / "selected_candidates.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    valid_frame = frame[frame["is_valid"]].copy()
    success = {
        "total_runs": int(len(frame)),
        "valid_runs": int(len(valid_frame)),
        "valid_rate": float(len(valid_frame) / len(frame)),
        "valid_spectral_auroc_mean": float(valid_frame["spectral_auroc"].mean()) if len(valid_frame) else float("nan"),
        "valid_spectral_permutation_pass_rate": float(
            (valid_frame["spectral_permutation_p"] < 0.05).mean()
        ) if len(valid_frame) else float("nan"),
    }
    pd.DataFrame([success]).to_csv(output / "clean_label_factorial_success_criteria.csv", index=False)

    lines = [
        "# GSDD-v0.6.3 Clean-label Factorial Aggregate",
        "",
        f"- Unique successful run directories: `{len(frame)}`",
        f"- Valid clean-label attacks: `{len(valid_frame)}`",
        f"- Overall valid rate: `{len(valid_frame) / len(frame):.3f}`",
        "",
        "## Candidate groups",
        "",
        "| Target | Poison | Runs | Valid | Valid rate | Full ASR | Control max | Binding gap | Spectral AUROC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in group_frame.iterrows():
        lines.append(
            f"| {int(row.target_class)} | {int(row.poison_count)} | {int(row.run_count)} | "
            f"{int(row.valid_count)} | {row.valid_rate:.3f} | "
            f"{row.full_asr_mean_valid:.4f} | {row.control_asr_max_mean_valid:.4f} | "
            f"{row.binding_gap_mean_valid:.4f} | {row.spectral_auroc_mean_valid:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "The clean-label functional binding score is",
            "",
            "$$",
            r"\Delta_{\mathrm{CL}}=\operatorname{ASR}(M_p,T)-\max\{\operatorname{ASR}(M_c,T),\operatorname{ASR}(M_p,T_{\mathrm{shuffle}}),\operatorname{ASR}(M_p,\varnothing),\operatorname{ASR}(M_c,\varnothing)\}",
            "$$",
            "",
            "Detection metrics are summarized only for attacks that pass the functional validity gate.",
        ]
    )
    (output / "CLEAN_LABEL_FACTORIAL_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"Wrote aggregate to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
