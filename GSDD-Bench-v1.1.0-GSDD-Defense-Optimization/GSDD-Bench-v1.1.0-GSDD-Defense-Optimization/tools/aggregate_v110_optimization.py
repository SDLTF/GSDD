from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ATTACKS = ("SBA", "UGBA", "GCBA")
BASELINES = (
    ("None", "none.log"),
    ("DShield", "dshield.log"),
    ("GSDD_v1.0", "gsdd_defense.log"),
)
VARIANT_PATTERN = re.compile(
    r"^GSDD2_(Hard|Soft)_(robust_max|fisher|cauchy)_(b\d{3})$"
)


def read_log(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.count(b"\x00") > max(8, len(data) // 10):
        return data.decode("utf-16le", errors="replace").lstrip("\ufeff")
    return data.decode("utf-8-sig", errors="replace")


def last_float(pattern: str, text: str) -> float | None:
    values = re.findall(pattern, text, flags=re.IGNORECASE)
    return float(values[-1]) if values else None


def parse_log(path: Path, dataset: str, attack: str, defense: str, seed: int) -> dict[str, object]:
    text = read_log(path)
    row = {
        "dataset": dataset,
        "attack": attack,
        "defense": defense,
        "seed": seed,
        "asr": last_float(r"ASR:\s*([0-9.]+)", text),
        "clean_accuracy": last_float(
            r"(?:Accuracy on clean test nodes|Accuracy):\s*([0-9.]+)", text
        ),
        "defense_seconds": last_float(r"Defense Time\s*=\s*([0-9.]+)s", text),
        "log": str(path.resolve()),
    }
    if row["asr"] is None or row["clean_accuracy"] is None:
        raise RuntimeError(f"Failed to parse official metrics from {path}")
    return row


def budget_from_code(code: str) -> float:
    return int(code[1:]) / 1000.0


def protocol_columns(defense: str) -> tuple[str | None, str | None, float | None]:
    match = VARIANT_PATTERN.match(defense)
    if not match:
        return None, None, None
    mode, method, code = match.groups()
    return mode.lower(), method, budget_from_code(code)


def write_markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1027)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    result_root = project / "results"
    optimization_root = project / "results_optimization"
    output = project / "artifacts" / "gsdd_v110_optimization_aggregate"
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []

    for attack in ATTACKS:
        run_id = f"Pubmed_{attack}_seed{args.seed}"
        baseline_root = result_root / run_id
        for defense, filename in BASELINES:
            log_path = baseline_root / filename
            if not log_path.exists():
                raise SystemExit(
                    f"Missing Stage-1 baseline log: {log_path}. Run the PubMed pilot before v1.1.0."
                )
            rows.append(parse_log(log_path, "Pubmed", attack, defense, args.seed))

        case_root = optimization_root / run_id
        detection_path = case_root / "Detection" / "optimization_summary.json"
        if not detection_path.exists():
            raise SystemExit(f"Missing optimized detection summary: {detection_path}")
        detection = json.loads(detection_path.read_text(encoding="utf-8"))
        legacy_metric = detection.get("legacy_spectral_hybrid_metrics")
        if legacy_metric:
            detection_rows.append(
                {
                    "dataset": "Pubmed",
                    "attack": attack,
                    "seed": args.seed,
                    "method": "legacy_spectral_hybrid",
                    "candidate_count": detection["candidate_count"],
                    "poison_count": detection["poison_count"],
                    "detection_seconds": detection["detection_seconds"],
                    **legacy_metric,
                }
            )
        for method, metric in detection["fusion_metrics"].items():
            detection_rows.append(
                {
                    "dataset": "Pubmed",
                    "attack": attack,
                    "seed": args.seed,
                    "method": method,
                    "candidate_count": detection["candidate_count"],
                    "poison_count": detection["poison_count"],
                    "detection_seconds": detection["detection_seconds"],
                    **metric,
                }
            )

        for metric_path in sorted(case_root.glob("GSDD2_*/official_metrics.json")):
            row = json.loads(metric_path.read_text(encoding="utf-8"))
            if row.get("asr") is None or row.get("clean_accuracy") is None:
                raise RuntimeError(f"Incomplete optimized metric file: {metric_path}")
            rows.append(row)

    frame = pd.DataFrame(rows)
    mode_method_budget = frame["defense"].apply(protocol_columns)
    frame[["mode", "method", "budget"]] = pd.DataFrame(
        mode_method_budget.tolist(), index=frame.index
    )

    none = (
        frame[frame.defense == "None"]
        .set_index(["dataset", "attack", "seed"])[["asr", "clean_accuracy"]]
        .rename(columns={"asr": "none_asr", "clean_accuracy": "none_clean_accuracy"})
    )
    frame = frame.join(none, on=["dataset", "attack", "seed"])
    frame["asr_reduction_vs_none"] = frame["none_asr"] - frame["asr"]
    frame["relative_asr_reduction_vs_none"] = frame["asr_reduction_vs_none"] / frame[
        "none_asr"
    ].replace(0, np.nan)
    frame["clean_accuracy_delta_vs_none"] = (
        frame["clean_accuracy"] - frame["none_clean_accuracy"]
    )
    frame.to_csv(output / "optimization_runs.csv", index=False, encoding="utf-8-sig")

    detection_frame = pd.DataFrame(detection_rows)
    detection_frame.to_csv(
        output / "detection_fusion_metrics.csv", index=False, encoding="utf-8-sig"
    )

    variants = frame[frame["method"].notna()].copy()
    protocol = (
        variants.groupby(["mode", "method", "budget"], dropna=False)
        .agg(
            attacks=("attack", "count"),
            mean_asr=("asr", "mean"),
            worst_asr=("asr", "max"),
            mean_asr_reduction=("asr_reduction_vs_none", "mean"),
            mean_clean_accuracy=("clean_accuracy", "mean"),
            mean_clean_accuracy_delta=("clean_accuracy_delta_vs_none", "mean"),
            mean_defense_seconds=("defense_seconds", "mean"),
        )
        .reset_index()
    )
    protocol["eligible_ca"] = protocol["mean_clean_accuracy_delta"] >= -0.02
    protocol["selection_objective"] = protocol["mean_asr"] + 5.0 * np.maximum(
        0.0, -0.02 - protocol["mean_clean_accuracy_delta"]
    )
    protocol = protocol.sort_values(
        ["eligible_ca", "selection_objective", "worst_asr"],
        ascending=[False, True, True],
    )
    protocol.to_csv(
        output / "cross_attack_protocol_summary.csv", index=False, encoding="utf-8-sig"
    )

    best_rows: list[pd.Series] = []
    for attack, attack_frame in variants.groupby("attack"):
        eligible = attack_frame[attack_frame.clean_accuracy_delta_vs_none >= -0.02]
        pool = eligible if len(eligible) else attack_frame
        chosen = pool.sort_values(["asr", "clean_accuracy"], ascending=[True, False]).iloc[0]
        best_rows.append(chosen)
    best_per_attack = pd.DataFrame(best_rows)
    best_per_attack.to_csv(
        output / "best_variant_per_attack.csv", index=False, encoding="utf-8-sig"
    )

    baseline_summary = frame[frame.defense.isin(["None", "DShield", "GSDD_v1.0"])][
        [
            "attack",
            "defense",
            "asr",
            "clean_accuracy",
            "asr_reduction_vs_none",
            "clean_accuracy_delta_vs_none",
            "defense_seconds",
        ]
    ].sort_values(["attack", "defense"])

    best_protocol = protocol.iloc[0]
    selected_protocol = {
        "version": "1.1.0",
        "dataset": "Pubmed",
        "pilot_seed": args.seed,
        "mode": str(best_protocol["mode"]),
        "method": str(best_protocol["method"]),
        "budget": float(best_protocol["budget"]),
        "mean_asr": float(best_protocol["mean_asr"]),
        "worst_asr": float(best_protocol["worst_asr"]),
        "mean_clean_accuracy_delta": float(best_protocol["mean_clean_accuracy_delta"]),
        "status": "pilot_selection_requires_multiseed_validation",
    }
    (output / "selected_protocol.json").write_text(
        json.dumps(selected_protocol, indent=2), encoding="utf-8"
    )
    lines = [
        "# GSDD-Bench v1.1.0 Defense Optimization Summary",
        "",
        f"- Dataset: `PubMed`",
        f"- Pilot seed: `{args.seed}`",
        "- Official attack artifacts reused without retraining",
        "- Operational scores use no poison labels",
        "- Pilot selection criterion: minimize mean ASR while keeping mean clean-accuracy delta at least -0.02",
        "",
        "## Stage-1 baselines",
        "",
    ]
    lines.extend(
        write_markdown_table(
            baseline_summary,
            [
                "attack",
                "defense",
                "asr",
                "clean_accuracy",
                "asr_reduction_vs_none",
                "clean_accuracy_delta_vs_none",
                "defense_seconds",
            ],
            ["Attack", "Defense", "ASR", "CA", "ASR reduction", "CA delta", "Seconds"],
        )
    )
    lines.extend(["", "## Best optimized variant per attack", ""])
    lines.extend(
        write_markdown_table(
            best_per_attack.sort_values("attack"),
            [
                "attack",
                "mode",
                "method",
                "budget",
                "asr",
                "clean_accuracy",
                "asr_reduction_vs_none",
                "clean_accuracy_delta_vs_none",
            ],
            ["Attack", "Mode", "Fusion", "Budget", "ASR", "CA", "ASR reduction", "CA delta"],
        )
    )
    lines.extend(["", "## Cross-attack protocol ranking", ""])
    lines.extend(
        write_markdown_table(
            protocol.head(12),
            [
                "mode",
                "method",
                "budget",
                "mean_asr",
                "worst_asr",
                "mean_asr_reduction",
                "mean_clean_accuracy_delta",
                "mean_defense_seconds",
            ],
            ["Mode", "Fusion", "Budget", "Mean ASR", "Worst ASR", "Mean ASR reduction", "Mean CA delta", "Seconds"],
        )
    )
    lines.extend(
        [
            "",
            "## Selected pilot protocol",
            "",
            f"- Mode: `{best_protocol['mode']}`",
            f"- Fusion: `{best_protocol['method']}`",
            f"- Budget: `{float(best_protocol['budget']):.3%}`",
            f"- Mean ASR: `{float(best_protocol['mean_asr']):.4f}`",
            f"- Worst-attack ASR: `{float(best_protocol['worst_asr']):.4f}`",
            f"- Mean clean-accuracy delta: `{float(best_protocol['mean_clean_accuracy_delta']):.4f}`",
            "",
            "This is a seed-1027 pilot selection, not a final claim. The selected protocol must be frozen and rerun on seeds 2026 and 3407 before OGBN-Arxiv evaluation.",
        ]
    )
    (output / "V110_OPTIMIZATION_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    archive = project / "artifacts" / "gsdd_v110_optimization_aggregate.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zip_file.write(path, arcname=path.relative_to(output))
    print(archive)


if __name__ == "__main__":
    main()
