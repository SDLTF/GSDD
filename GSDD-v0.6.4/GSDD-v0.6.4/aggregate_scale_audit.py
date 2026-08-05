from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--prefix", default="gsdd_v03_cora_scale_audit_multiseed")
    parser.add_argument("--output-dir", default="results/scale_audit_aggregate")
    return parser.parse_args()


def metric(summary: dict, name: str, key: str) -> float | None:
    value = summary.get("detection", {}).get(name, {})
    return value.get(key) if isinstance(value, dict) else None


def best_group_auc(raw: pd.DataFrame, group: str) -> float | None:
    subset = raw.loc[raw["group"] == group, "auroc_oriented_diagnostic"]
    return float(subset.max()) if len(subset) else None


def main() -> int:
    args = parse_args()
    root = Path(args.results_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for run_dir in sorted(root.glob(f"{args.prefix}*")):
        summary_path = run_dir / "summary.json"
        raw_path = run_dir / "raw_feature_metrics.csv"
        if not summary_path.exists() or not raw_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        raw = pd.read_csv(raw_path)
        row = {
            "run_dir": str(run_dir),
            "seed": summary.get("seed"),
            "status": summary.get("status"),
            "clean_accuracy": summary.get("clean_test_accuracy"),
            "asr": summary.get("triggered_test_asr"),
            "ecdf_transfer_auroc": metric(summary, "global_ecdf_transfer", "auroc"),
            "ecdf_transfer_auprc": metric(summary, "global_ecdf_transfer", "auprc"),
            "ecdf_transfer_level_auroc": metric(summary, "global_ecdf_transfer_level", "auroc"),
            "ecdf_transfer_level_auprc": metric(summary, "global_ecdf_transfer_level", "auprc"),
            "ecdf_transfer_shape_auroc": metric(summary, "global_ecdf_transfer_shape", "auroc"),
            "ecdf_transfer_shape_auprc": metric(summary, "global_ecdf_transfer_shape", "auprc"),
            "ecdf_transfer_shape_norm_auroc": metric(summary, "global_ecdf_transfer_shape_norm", "auroc"),
            "ecdf_transfer_shape_norm_auprc": metric(summary, "global_ecdf_transfer_shape_norm", "auprc"),
            "ecdf_scale_invariant_auroc": metric(summary, "global_ecdf_scale_invariant", "auroc"),
            "ecdf_scale_invariant_auprc": metric(summary, "global_ecdf_scale_invariant", "auprc"),
            "hybrid_scale_invariant_auroc": metric(summary, "hybrid_scale_invariant", "auroc"),
            "hybrid_scale_invariant_auprc": metric(summary, "hybrid_scale_invariant", "auprc"),
            "best_raw_transfer_auc": best_group_auc(raw, "transfer"),
            "best_level_auc": best_group_auc(raw, "transfer_level"),
            "best_shape_auc": best_group_auc(raw, "transfer_shape"),
            "best_shape_norm_auc": best_group_auc(raw, "transfer_shape_norm"),
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "scale_audit_runs.csv", index=False)
    numeric = frame.select_dtypes(include="number")
    stats = pd.DataFrame({"mean": numeric.mean(), "std": numeric.std(ddof=1), "min": numeric.min(), "max": numeric.max()})
    stats.to_csv(output / "scale_audit_summary_stats.csv")

    lines = [
        "# GSDD-v0.3 multi-seed scale audit",
        "",
        f"Runs found: {len(frame)}",
        "",
        "The decisive comparison is raw/level transfer versus scale-invariant transfer shape.",
        "If level remains strong while shape stays near chance, v0.2 H4 was mainly a representation-scale signal rather than a frequency-selective transfer effect.",
        "",
        "## Mean metrics",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ]
    for name in [
        "clean_accuracy",
        "asr",
        "ecdf_transfer_auroc",
        "ecdf_transfer_level_auroc",
        "ecdf_transfer_shape_auroc",
        "ecdf_transfer_shape_norm_auroc",
        "ecdf_scale_invariant_auroc",
        "hybrid_scale_invariant_auroc",
        "best_level_auc",
        "best_shape_auc",
        "best_shape_norm_auc",
    ]:
        if name in stats.index:
            lines.append(f"| {name} | {stats.loc[name, 'mean']:.4f} | {stats.loc[name, 'std']:.4f} |")
    (output / "SCALE_AUDIT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(frame)} scale-audit runs to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
