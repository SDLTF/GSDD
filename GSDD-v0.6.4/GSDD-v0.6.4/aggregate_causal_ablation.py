from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODES = ["none", "label_only", "trigger_only", "full"]


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

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "causal_ablation_runs.csv", index=False)

    numeric_cols = [
        column for column in frame.select_dtypes(include="number").columns if column != "seed"
    ]
    grouped = (
        frame.groupby("ablation_mode")[numeric_cols]
        .agg(["mean", "std", "min", "max"])
        .reindex(MODES)
    )
    grouped.to_csv(output / "causal_ablation_group_stats.csv")

    interaction_rows = []
    for seed, seed_frame in frame.groupby("seed"):
        by_mode = seed_frame.set_index("ablation_mode")
        if not all(mode in by_mode.index for mode in MODES):
            continue
        entry = {"seed": seed}
        for metric_name in [
            "contrast_shape_geometry",
            "contrast_shape_norm_l1",
            "contrast_shape_norm_l2",
            "shape_geometry_auroc",
            "shape_norm_auroc",
        ]:
            values = {mode: float(by_mode.loc[mode, metric_name]) for mode in MODES}
            entry[f"{metric_name}_trigger_main"] = 0.5 * (
                values["trigger_only"] + values["full"]
                - values["none"] - values["label_only"]
            )
            entry[f"{metric_name}_label_main"] = 0.5 * (
                values["label_only"] + values["full"]
                - values["none"] - values["trigger_only"]
            )
            entry[f"{metric_name}_interaction"] = (
                values["full"] - values["label_only"]
                - values["trigger_only"] + values["none"]
            )
        interaction_rows.append(entry)
    interaction = pd.DataFrame(interaction_rows)
    interaction.to_csv(output / "causal_ablation_factorial_effects.csv", index=False)

    lines = [
        "# GSDD-v0.4 causal ablation summary",
        "",
        f"Runs found: {len(frame)}",
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
        ]
    )
    (output / "CAUSAL_ABLATION_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(frame)} runs to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
