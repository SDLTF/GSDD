from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    candidates = []
    for path in Path(args.results_root).glob(f"{args.prefix}*"):
        summary_path = path / "summary.json"
        validity_path = path / "attack_validity.json"
        if not summary_path.exists() or not validity_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            validity = json.loads(validity_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if int(summary.get("seed", -1)) != args.seed or summary.get("status") != "success":
            continue
        candidates.append((path.stat().st_mtime, bool(validity.get("is_valid", False))))
    print("true" if candidates and max(candidates)[1] else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
