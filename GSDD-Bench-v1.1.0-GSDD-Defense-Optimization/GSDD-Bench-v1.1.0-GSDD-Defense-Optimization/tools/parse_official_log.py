from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read_log(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.count(b"\x00") > max(8, len(data) // 10):
        return data.decode("utf-16le", errors="replace").lstrip("\ufeff")
    return data.decode("utf-8-sig", errors="replace")


def last(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return float(matches[-1]) if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--attack", required=True)
    parser.add_argument("--defense", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    log_path = Path(args.log)
    text = read_log(log_path)
    row = {
        "dataset": args.dataset,
        "attack": args.attack,
        "defense": args.defense,
        "seed": args.seed,
        "asr": last(r"ASR:\s*([0-9.]+)", text),
        "clean_accuracy": last(r"(?:Accuracy on clean test nodes|Accuracy):\s*([0-9.]+)", text),
        "defense_seconds": last(r"Defense Time\s*=\s*([0-9.]+)s", text),
        "log": str(log_path.resolve()),
        "encoding_detected": "utf-16le" if data_is_utf16(log_path) else "utf-8",
    }
    if row["asr"] is None or row["clean_accuracy"] is None:
        raise SystemExit(
            f"Could not parse ASR/clean accuracy from {log_path}; inspect the full log before aggregating"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row))


def data_is_utf16(path: Path) -> bool:
    data = path.read_bytes()
    return data.startswith(b"\xff\xfe") or data.count(b"\x00") > max(8, len(data) // 10)


if __name__ == "__main__":
    main()
