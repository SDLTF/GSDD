from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODES = ["none", "label_only", "trigger_only", "full"]
KEY_COLUMNS = ["seed", "ablation_mode"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--prefix", default="gsdd_v04_cora_causal_ablation")
    parser.add_argument("--output-dir", default="results/causal_ablation_aggregate")
    return parser.parse_args()


def metric(summary: dict, name: str, key: str) -> float | None:
    value = summary.get("detection", {}).get(name, {})
    return value.get(key) if isinstance(value, dict) else None


def selected_contrast(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns or "is_poisoned" not in frame.columns:
        return None
    selected = frame.loc[frame["is_poisoned"] == 1, column]
    reference = frame.loc[frame["is_poisoned"] == 0, column]
    if len(selected) == 0 or len(reference) == 0:
        return None
    return float(selected.mean() - reference.mean())


def choose_latest_complete_runs(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Keep only the latest complete run for each seed/mode pair.

    Re-running a condition leaves multiple timestamped result directories.  The
    previous aggregator used ``set_index(...).loc[...]`` and therefore received
    a Series rather than a scalar whenever duplicates existed.  We retain the
    newest complete directory and write an audit CSV for transparency.
    """
    if frame.empty:
        return frame

    duplicate_mask = frame.duplicated(KEY_COLUMNS, keep=False)
    duplicates = frame.loc[duplicate_mask].copy()
    if not duplicates.empty:
        duplicates = duplicates.sort_values(
            KEY_COLUMNS + ["run_mtime_ns", "run_dir"],
            kind="stable",
        )
        duplicates.to_csv(output / "duplicate_run_candidates.csv", index=False)
        print(
            "[Aggregate] duplicate seed/mode runs detected; "
            "keeping the latest complete run for each pair. "
            f"Audit: {output / 'duplicate_run_candidates.csv'}"
        )

    deduplicated = (
        frame.sort_values(
            KEY_COLUMNS + ["run_mtime_ns", "run_dir"],
            kind="stable",
        )
        .drop_duplicates(KEY_COLUMNS, keep="last")
        .reset_index(drop=True)
    )
    return deduplicated


def safe_float(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else float("nan")


def main() -> int:
    args = parse_args()
    root = Path(args.results_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for run_dir in sorted(root.glob(f"{args.prefix}*")):
        summary_path = run_dir / "summary.json"
        node_path = run_dir / "node_scores.csv"
        if not summary_path.exists() or not node_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        node = pd.read_csv(node_path)
        row = {
            "run_dir": str(run_dir),
            "run_mtime_ns": run_dir.stat().st_mtime_ns,
            "seed": summary.get("seed"),
            "ablation_mode": summary.get("ablation_mode", "unknown"),
            "status": summary.get("status"),
            "clean_accuracy": summary.get("clean_test_accuracy"),
            "asr": summary.get("triggered_test_asr"),
            "shape_norm_auroc": metric(summary, "global_ecdf_transfer_shape_norm", "auroc"),
            "shape_norm_auprc": metric(summary, "global_ecdf_transfer_shape_norm", "auprc"),
            "shape_geometry_auroc": metric(summary, "shape_geometry_mahalanobis", "auroc"),
            "shape_geometry_auprc": metric(summary, "shape_geometry_mahalanobis", "auprc"),
            "scale_invariant_auroc": metric(summary, "global_ecdf_scale_invariant", "auroc"),
            "scale_invariant_auprc": metric(summary, "global_ecdf_scale_invariant", "auprc"),
            "raw_transfer_auroc": metric(summary, "global_ecdf_transfer", "auroc"),
            "level_auroc": metric(summary, "global_ecdf_transfer_level", "auroc"),
            "input_spectrum_auroc": metric(summary, "global_ecdf_input_spectrum", "auroc"),
            "model_js_auroc": metric(summary, "global_ecdf_model_js", "auroc"),
            "contrast_shape_geometry": selected_contrast(node, "score_shape_geometry_mahalanobis"),
            "contrast_shape_norm_l1": selected_contrast(node, "gain_shape_norm_l1"),
            "contrast_shape_norm_l2": selected_contrast(node, "gain_shape_norm_l2"),
        }
        rows.append(row)

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise RuntimeError(
            f"No complete runs matching prefix '{args.prefix}' were found under {root}."
        )

    candidates.to_csv(output / "causal_ablation_run_candidates.csv", index=False)
    frame = choose_latest_complete_runs(candidates, output)
    frame.to_csv(output / "causal_ablation_runs.csv", index=False)

    numeric_cols = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in {"seed", "run_mtime_ns"}
    ]
    grouped = (
        frame.groupby("ablation_mode")[numeric_cols]
        .agg(["mean", "std", "min", "max"])
        .reindex(MODES)
    )
    grouped.to_csv(output / "causal_ablation_group_stats.csv")

    interaction_rows = []
    for seed, seed_frame in frame.groupby("seed", dropna=False):
        # Deduplication above guarantees at most one row per mode.  Reindex also
        # makes missing conditions explicit instead of silently broadcasting.
        by_mode = seed_frame.set_index("ablation_mode").reindex(MODES)
        if by_mode["run_dir"].isna().any():
            continue
        entry = {"seed": seed}
        for metric_name in [
            "contrast_shape_geometry",
            "contrast_shape_norm_l1",
            "contrast_shape_norm_l2",
            "shape_geometry_auroc",
            "shape_norm_auroc",
        ]:
            values = {
                mode: safe_float(by_mode.at[mode, metric_name])
                for mode in MODES
            }
            if any(not np.isfinite(value) for value in values.values()):
                entry[f"{metric_name}_trigger_main"] = np.nan
                entry[f"{metric_name}_label_main"] = np.nan
                entry[f"{metric_name}_interaction"] = np.nan
                continue
            entry[f"{metric_name}_trigger_main"] = 0.5 * (
                values["trigger_only"]
                + values["full"]
                - values["none"]
                - values["label_only"]
            )
            entry[f"{metric_name}_label_main"] = 0.5 * (
                values["label_only"]
                + values["full"]
                - values["none"]
                - values["trigger_only"]
            )
            entry[f"{metric_name}_interaction"] = (
                values["full"]
                - values["label_only"]
                - values["trigger_only"]
                + values["none"]
            )
        interaction_rows.append(entry)
    interaction = pd.DataFrame(interaction_rows)
    interaction.to_csv(output / "causal_ablation_factorial_effects.csv", index=False)

    candidate_count = len(candidates)
    used_count = len(frame)
    duplicate_count = candidate_count - used_count
    lines = [
        "# GSDD-v0.4 causal ablation summary",
        "",
        f"Complete run directories found: {candidate_count}",
        f"Unique seed/mode runs used: {used_count}",
        f"Older duplicate runs ignored: {duplicate_count}",
        "",
        "The same seeded non-target training nodes are audited under four interventions:",
        "",
        "- `none`: selected-node negative control",
        "- `label_only`: dirty-label conflict without a trigger",
        "- `trigger_only`: graph/feature trigger without relabeling",
        "- `full`: trigger plus dirty-label relabeling",
        "",
        "A genuine trigger-dependent spectral-shape mechanism should be near chance in `none`,",
        "not be explained entirely by `label_only`, and become stronger in `trigger_only` or `full`.",
        "",
        "## Mean results by condition",
        "",
        "| Mode | Clean acc. | ASR | Shape-norm AUROC | Shape-geometry AUROC | Scale-invariant AUROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        subset = frame[frame["ablation_mode"] == mode]
        if subset.empty:
            continue

        def mean(name: str) -> float:
            return float(subset[name].mean())

        lines.append(
            f"| {mode} | {mean('clean_accuracy'):.4f} | {mean('asr'):.4f} | "
            f"{mean('shape_norm_auroc'):.4f} | {mean('shape_geometry_auroc'):.4f} | "
            f"{mean('scale_invariant_auroc'):.4f} |"
        )

    if not interaction.empty:
        lines.extend(
            [
                "",
                "## Factorial contrasts",
                "",
                "For a scalar diagnostic contrast, the interaction is",
                "",
                "$$",
                "I = S_{\\mathrm{full}}-S_{\\mathrm{label}}-S_{\\mathrm{trigger}}+S_{\\mathrm{none}}",
                "$$",
                "",
                "Positive interaction means trigger and label conflict reinforce each other beyond additive main effects.",
                "",
                "| Quantity | Mean trigger main | Mean label main | Mean interaction |",
                "|---|---:|---:|---:|",
            ]
        )
        for base in [
            "contrast_shape_geometry",
            "contrast_shape_norm_l1",
            "contrast_shape_norm_l2",
        ]:
            lines.append(
                f"| {base} | "
                f"{interaction[f'{base}_trigger_main'].mean():.4f} | "
                f"{interaction[f'{base}_label_main'].mean():.4f} | "
                f"{interaction[f'{base}_interaction'].mean():.4f} |"
            )

    lines.extend(
        [
            "",
            "See `causal_ablation_runs.csv`, `causal_ablation_group_stats.csv`, and",
            "`causal_ablation_factorial_effects.csv` for complete values.",
            "",
            "If duplicate runs were found, `duplicate_run_candidates.csv` lists every candidate.",
        ]
    )
    (output / "CAUSAL_ABLATION_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        f"Wrote {used_count} unique runs to {output} "
        f"({duplicate_count} older duplicate runs ignored)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
