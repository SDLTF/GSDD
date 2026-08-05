from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
args = parser.parse_args()
repo = Path(args.repo)
node = repo / "NodeClassificationTasks"
required = [
    node / "main.py",
    node / "gsdd_bench_export.py",
    node / "gsdd_bench_compat.py",
    node / "defense/dshield.py",
    node / "utils.py",
    node / "models/construct.py",
    node / "models/GCN.py",
]
for path in required:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
for path in [
    node / "main.py",
    node / "gsdd_bench_export.py",
    node / "gsdd_bench_compat.py",
    node / "heuristic_selection.py",
    node / "utils.py",
    node / "models/construct.py",
    node / "models/GCN.py",
]:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
main_text = (node / "main.py").read_text(encoding="utf-8")
utils_text = (node / "utils.py").read_text(encoding="utf-8")
construct_text = (node / "models/construct.py").read_text(encoding="utf-8")
gcn_text = (node / "models/GCN.py").read_text(encoding="utf-8")
checks = {
    "main_patch": "GSDD_BENCH_PATCH_V1" in main_text,
    "artifact_cli": "--gsdd_export_dir" in main_text and "--gsdd_load_artifact" in main_text,
    "cuda_only": "CPU fallback is disabled" in main_text,
    "no_eager_torch_scatter": "import torch_scatter" not in utils_text,
    "pyg_scatter_compat": "GSDD_BENCH_PYG_SCATTER_COMPAT_V103" in utils_text and "dim_size=num_labels" in utils_text,
    "optional_robustgcn": "GSDD_BENCH_OPTIONAL_ROBUSTGCN_V103" in construct_text,
    "weighted_training_cli": "--gsdd_train_weight_override" in main_text,
    "weighted_gcn": "GSDD_BENCH_NODE_WEIGHTS_V110" in gcn_text and "reduction='none'" in gcn_text,
}
print(json.dumps(checks, indent=2))
if not all(checks.values()):
    raise SystemExit("DShield compatibility patch validation failed")
