from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="results/aggregate_results.csv")
    return parser.parse_args()


def flatten(summary: dict, run_dir: Path) -> dict:
    row = {
        "run_dir": str(run_dir),
        "status": summary.get("status"),
        "dataset": summary.get("dataset"),
        "seed": summary.get("seed"),
        "device": summary.get("device"),
        "clean_test_accuracy": summary.get("clean_test_accuracy"),
        "triggered_test_asr": summary.get("triggered_test_asr"),
        "poisoned_training_victims": summary.get("poisoned_training_victims"),
    }
    for score_name, values in summary.get("detection", {}).items():
        if isinstance(values, dict):
            for metric_name, metric_value in values.items():
                row[f"{score_name}_{metric_name}"] = metric_value
    return row


def main() -> int:
    args = parse_args()
    root = Path(args.results_root)
    rows = []
    for summary_path in sorted(root.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(flatten(summary, summary_path.parent))
        except Exception as exc:
            rows.append(
                {
                    "run_dir": str(summary_path.parent),
                    "status": "unreadable",
                    "error": str(exc),
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} runs to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
