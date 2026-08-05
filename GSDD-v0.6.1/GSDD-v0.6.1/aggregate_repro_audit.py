from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

PRIMARY_SCORES = [
    "did_shape_l2",
    "did_distribution_l2",
    "did_spectral_hybrid",
    "did_shape_mahalanobis",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--prefix", default="gsdd_v051_cora_repro")
    parser.add_argument("--output-dir", default="results/repro_audit_aggregate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.results_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for path in sorted(root.glob(f"{args.prefix}_audit_*")):
        summary_path = path / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        candidate_rows.append(
            {
                "run_dir": str(path),
                "seed": summary.get("seed"),
                "status": summary.get("status"),
                "mtime": path.stat().st_mtime,
            }
        )
        if summary.get("status") != "success":
            continue
        base = {
            "run_dir": str(path),
            "seed": int(summary["seed"]),
            "initialization_hash_match": summary["initialization_hash_match"],
            "all_primary_scores_pass": summary["all_primary_scores_pass"],
            "maximum_parameter_abs_difference": summary["maximum_parameter_abs_difference"],
            "maximum_clean_accuracy_abs_delta": summary["maximum_clean_accuracy_abs_delta"],
            "maximum_asr_abs_delta": summary["maximum_asr_abs_delta"],
        }
        score_map = {item["score"]: item for item in summary["score_reproducibility"]}
        for score in PRIMARY_SCORES:
            item = score_map[score]
            for key in [
                "pearson",
                "spearman",
                "operational_topk_overlap",
                "auroc_abs_delta",
                "auprc_abs_delta",
                "passes",
            ]:
                base[f"{score}_{key}"] = item[key]
        records.append(base)

    pd.DataFrame(candidate_rows).to_csv(output / "repro_audit_candidates.csv", index=False)
    if not records:
        raise RuntimeError(f"No successful audit runs match prefix {args.prefix!r}")
    frame = pd.DataFrame(records).sort_values("seed").drop_duplicates("seed", keep="last")
    frame.to_csv(output / "repro_audit_runs.csv", index=False)
    numeric = [column for column in frame.select_dtypes(include="number").columns if column != "seed"]
    frame[numeric].agg(["mean", "std", "min", "max"]).T.to_csv(
        output / "repro_audit_summary_stats.csv"
    )

    lines = [
        "# GSDD-v0.5.1 reproducibility aggregate",
        "",
        f"Unique seeds: {len(frame)}",
        f"All primary criteria passed for every seed: `{bool(frame['all_primary_scores_pass'].all())}`",
        f"Initialization hashes matched for every seed: `{bool(frame['initialization_hash_match'].all())}`",
        "",
        "| Score | Spearman mean ± std | Top-k overlap mean ± std | AUROC Δ mean ± std | Pass rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for score in PRIMARY_SCORES:
        spearman = frame[f"{score}_spearman"]
        overlap = frame[f"{score}_operational_topk_overlap"]
        auc_delta = frame[f"{score}_auroc_abs_delta"]
        pass_rate = frame[f"{score}_passes"].astype(float).mean()
        lines.append(
            f"| {score} | {spearman.mean():.4f} ± {spearman.std(ddof=1):.4f} | "
            f"{overlap.mean():.4f} ± {overlap.std(ddof=1):.4f} | "
            f"{auc_delta.mean():.4f} ± {auc_delta.std(ddof=1):.4f} | {pass_rate:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Maximum parameter difference observed: `{frame['maximum_parameter_abs_difference'].max():.6g}`",
            "",
            "The next attack-generalization stage should begin only if the scale-invariant score rankings and operational top-k sets are stable across repeated identical runs.",
        ]
    )
    (output / "REPRO_AUDIT_AGGREGATE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(frame)} reproducibility audits to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
