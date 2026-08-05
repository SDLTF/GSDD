from __future__ import annotations
import argparse, ast, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True)
args = ap.parse_args()
repo = Path(args.repo)
node = repo / 'NodeClassificationTasks'
required = [
    node / 'main.py',
    node / 'gsdd_bench_export.py',
    node / 'gsdd_bench_compat.py',
    node / 'defense/dshield.py',
    node / 'utils.py',
    node / 'models/construct.py',
]
for p in required:
    if not p.exists():
        raise SystemExit(f'Missing required file: {p}')
for p in [
    node / 'main.py',
    node / 'gsdd_bench_export.py',
    node / 'gsdd_bench_compat.py',
    node / 'heuristic_selection.py',
    node / 'utils.py',
    node / 'models/construct.py',
]:
    ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
main_text = (node / 'main.py').read_text(encoding='utf-8')
utils_text = (node / 'utils.py').read_text(encoding='utf-8')
construct_text = (node / 'models/construct.py').read_text(encoding='utf-8')
checks = {
    'main_patch': 'GSDD_BENCH_PATCH_V1' in main_text,
    'artifact_cli': '--gsdd_export_dir' in main_text and '--gsdd_load_artifact' in main_text,
    'cuda_only': 'CPU fallback is disabled' in main_text,
    'no_eager_torch_scatter': 'import torch_scatter' not in utils_text,
    'pyg_scatter_compat': 'GSDD_BENCH_PYG_SCATTER_COMPAT_V103' in utils_text and 'dim_size=num_labels' in utils_text,
    'optional_robustgcn': 'GSDD_BENCH_OPTIONAL_ROBUSTGCN_V103' in construct_text,
}
print(json.dumps(checks, indent=2))
if not all(checks.values()):
    raise SystemExit('DShield compatibility patch validation failed')
