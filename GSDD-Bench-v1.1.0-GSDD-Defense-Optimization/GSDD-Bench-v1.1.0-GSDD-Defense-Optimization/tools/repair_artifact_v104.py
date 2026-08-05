from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Running a script by path puts only the tools directory on sys.path.
# Add the project root explicitly so the sibling gsdd_core package is importable
# regardless of the caller's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from gsdd_core.artifact import normalize_attack_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact).resolve()
    artifact_path = artifact_dir / "artifact.pt"
    manifest_path = artifact_dir / "manifest.json"
    if not artifact_path.exists():
        raise SystemExit(f"Missing artifact: {artifact_path}")

    original = torch.load(artifact_path, map_location="cpu", weights_only=False)
    old_label_count = int(
        original.get(
            "label_num_nodes_before_padding",
            torch.as_tensor(original["poison_y"]).numel(),
        )
    )
    poison_count = int(torch.as_tensor(original["poison_x"]).shape[0])
    normalized = normalize_attack_bundle(original)

    backup_path = artifact_dir / "artifact.pt.pre_v104.bak"
    if not backup_path.exists():
        shutil.copy2(artifact_path, backup_path)
    torch.save(normalized, artifact_path)

    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = max(2, int(manifest.get("format_version", 1)))
    manifest["poison_num_nodes"] = poison_count
    manifest["label_num_nodes_before_padding"] = old_label_count
    manifest["injected_node_count"] = int(normalized["injected_node_idx"].numel())
    repair_record = {
        "version": "1.0.4",
        "repair": "pad poison_y to poison_x node count with -1 for injected unlabeled nodes",
    }
    repairs = manifest.setdefault("compatibility_repairs", [])
    if repair_record not in repairs:
        repairs.append(repair_record)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "artifact": str(artifact_dir),
                "poison_num_nodes": poison_count,
                "labels_before": old_label_count,
                "labels_after": int(normalized["poison_y"].numel()),
                "injected_node_count": int(normalized["injected_node_idx"].numel()),
                "backup": str(backup_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
