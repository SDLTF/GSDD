from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ATTACK_IMPORTS = [
    ("attack.explain_backdoor", ["ExplainBackdoor"]),
    ("attack.gcba", ["GCBA"]),
    ("attack.gta", ["GTA"]),
    ("attack.nba", ["LGCBackdoor", "FGBackdoor"]),
    ("attack.sba", ["SBA"]),
    ("attack.adada", ["AdaDA"]),
    ("attack.adaca", ["AdaCA"]),
    ("attack.percba", ["PerCBA"]),
    ("attack.target_node_attack", ["TargetNodeAttack"]),
    ("attack.trap", ["TRAP"]),
    ("attack.ugba", ["UGBA"]),
    ("attack.dpgba", ["DPGBA"]),
    ("attack.mlgb", ["MLGB"]),
]


def optional_import_block() -> str:
    lines = [
        "# GSDD_BENCH_OPTIONAL_IMPORTS_BEGIN",
        "import importlib",
        "class _UnavailableComponent:",
        "    _error = None",
        "    def __init__(self, *args, **kwargs):",
        "        raise RuntimeError(f'Optional DShield component is unavailable: {self._error}')",
        "def _optional_component(module_name, symbol):",
        "    try:",
        "        return getattr(importlib.import_module(module_name), symbol)",
        "    except Exception as exc:",
        "        return type(symbol, (_UnavailableComponent,), {'_error': f'{module_name}.{symbol}: {exc!r}'})",
    ]
    for module, names in ATTACK_IMPORTS:
        for name in names:
            lines.append(f"{name} = _optional_component({module!r}, {name!r})")
    lines.append("# GSDD_BENCH_OPTIONAL_IMPORTS_END")
    return "\n".join(lines)


def patch_main(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "GSDD_BENCH_PATCH_V1" in text:
        return
    original = text

    for module, names in ATTACK_IMPORTS:
        pattern = rf"^from {re.escape(module)} import {', '.join(names)}\s*$"
        text = re.sub(pattern, "", text, flags=re.MULTILINE)

    anchor = "from torch_geometric.utils import to_undirected, k_hop_subgraph\n"
    if anchor not in text:
        raise RuntimeError("main.py import anchor not found")
    text = text.replace(anchor, anchor + optional_import_block() + "\n", 1)

    cli = """    # GSDD_BENCH_PATCH_V1
    parser.add_argument('--gsdd_export_dir', type=str, default='')
    parser.add_argument('--gsdd_run_id', type=str, default='')
    parser.add_argument('--gsdd_load_artifact', type=str, default='')
    parser.add_argument('--gsdd_train_idx_override', type=str, default='')
"""
    anchor = "    args = parser.parse_known_args()[0]\n"
    if anchor not in text:
        raise RuntimeError("argument parse anchor not found")
    text = text.replace(anchor, cli + anchor, 1)

    old = (
        "    args.cuda = not args.no_cuda and torch.cuda.is_available()\n"
        "    device = ('cuda:{}' if torch.cuda.is_available() and args.cuda else 'cpu').format(args.device_id)\n"
    )
    new = (
        "    if args.no_cuda or not torch.cuda.is_available():\n"
        "        raise RuntimeError('GSDD-Bench formal runs require CUDA; CPU fallback is disabled')\n"
        "    args.cuda = True\n"
        "    device = ('cuda:{}').format(args.device_id)\n"
    )
    if old not in text:
        raise RuntimeError("CUDA anchor not found")
    text = text.replace(old, new, 1)

    anchor = "    mask_edge_index = data.edge_index[:, torch.bitwise_not(edge_mask)]\n"
    if anchor not in text:
        raise RuntimeError("clean edge anchor not found")
    text = text.replace(anchor, anchor + "    clean_train_edge_index = train_edge_index.clone()\n", 1)

    anchor = "    if args.attack_method == 'none':\n"
    loader = """    if args.gsdd_load_artifact:
        from gsdd_bench_export import load_attack_artifact
        _bundle, backdoor_gen_model, _manifest = load_attack_artifact(args.gsdd_load_artifact, device)
        feat = _bundle['poison_x'].to(device)
        train_edge_index = _bundle['poison_train_edge_index'].to(device)
        train_edge_weight = _bundle['poison_train_edge_weight'].to(device)
        labels = _bundle['poison_y'].to(device)
        train_node_idx = _bundle['poison_train_idx'].to(device)
        attach_idx = _bundle['attach_idx'].to(device)
        args.target_class = _manifest['target_class']
    elif args.attack_method == 'none':
"""
    if anchor not in text:
        raise RuntimeError("attack branch anchor not found")
    text = text.replace(anchor, loader, 1)

    anchor = "    # Defense\n"
    block = """    if args.gsdd_export_dir and not args.gsdd_load_artifact:
        from gsdd_bench_export import export_attack_artifact
        _gsdd_artifact_path = export_attack_artifact(
            output_root=args.gsdd_export_dir, run_id=args.gsdd_run_id, args=args, data=data,
            clean_train_edge_index=clean_train_edge_index, mask_edge_index=mask_edge_index,
            feat=feat, train_edge_index=train_edge_index, train_edge_weight=train_edge_weight, labels=labels,
            train_idx=train_idx, val_idx=val_idx, clean_test_idx=clean_test_idx, atk_idx=atk_idx,
            attach_idx=attach_idx, train_node_idx=train_node_idx, backdoor_gen_model=backdoor_gen_model)
    elif args.gsdd_load_artifact:
        _gsdd_artifact_path = args.gsdd_load_artifact
    else:
        _gsdd_artifact_path = ''
    if args.gsdd_train_idx_override:
        train_node_idx = torch.load(args.gsdd_train_idx_override, map_location=device, weights_only=True).long().to(device)
        logger.info('[GSDD-Bench] using train-index override with %d nodes', int(train_node_idx.numel()))

"""
    if anchor not in text:
        raise RuntimeError("defense anchor not found")
    text = text.replace(anchor, block + anchor, 1)

    anchor = '    logger.info("Accuracy: {:.4f}".format(overall_ca))\n'
    tail = """    if _gsdd_artifact_path:
        from gsdd_bench_export import update_metrics
        update_metrics(_gsdd_artifact_path, args.defense_method if not args.gsdd_train_idx_override else 'GSDD',
                       overall_ca, overall_asr, end_time - begin_time)
"""
    if anchor not in text:
        raise RuntimeError("metric anchor not found")
    text = text.replace(anchor, anchor + tail, 1)

    path.with_suffix(".py.gsddbench.bak").write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")


def patch_heuristic(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "from gsdd_bench_compat import cluster" in text:
        return
    old = "from sklearn_extra import cluster\n"
    if old in text:
        text = text.replace(
            old,
            "try:\n    from sklearn_extra import cluster\nexcept Exception:\n    from gsdd_bench_compat import cluster\n",
            1,
        )
        path.write_text(text, encoding="utf-8")


def patch_utils(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "GSDD_BENCH_PYG_SCATTER_COMPAT_V103"
    if marker in text:
        return
    original = text

    if "import torch_scatter\n" not in text:
        raise RuntimeError("utils.py torch_scatter import anchor not found")
    text = text.replace(
        "import torch_scatter\n",
        f"# {marker}: optional torch_scatter replaced by PyG/PyTorch scatter\n",
        1,
    )

    old_import = "from torch_geometric.utils import degree, to_undirected\n"
    new_import = "from torch_geometric.utils import degree, to_undirected, scatter\n"
    if old_import not in text:
        raise RuntimeError("utils.py torch_geometric import anchor not found")
    text = text.replace(old_import, new_import, 1)

    old_call = "label_degree_cnt = torch_scatter.scatter(node_degrees, labels, reduce='sum')"
    new_call = "label_degree_cnt = scatter(node_degrees, labels, dim=0, dim_size=num_labels, reduce='sum')"
    if old_call not in text:
        raise RuntimeError("utils.py torch_scatter call anchor not found")
    text = text.replace(old_call, new_call, 1)

    path.with_suffix(".py.gsddbench.v103.bak").write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")


def patch_model_construct(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "GSDD_BENCH_OPTIONAL_ROBUSTGCN_V103"
    if marker in text:
        return
    old = "from models.RobustGCN import RobustGCN\n"
    if old not in text:
        raise RuntimeError("models/construct.py RobustGCN import anchor not found")
    replacement = """# GSDD_BENCH_OPTIONAL_ROBUSTGCN_V103
try:
    from models.RobustGCN import RobustGCN
except Exception as _gsdd_robustgcn_import_error:
    # RobustGCN needs the optional compiled torch_sparse extension.
    # Standard GCN/GAT/GraphSAGE experiments must remain importable without it.
    RobustGCN = None
"""
    original = text
    text = text.replace(old, replacement, 1)
    path.with_suffix(".py.gsddbench.v103.bak").write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    node = repo / "NodeClassificationTasks"
    if not (node / "main.py").exists():
        raise SystemExit(f"Invalid DShield checkout: {repo}")

    shutil.copy2(Path(__file__).with_name("gsdd_bench_export.py"), node / "gsdd_bench_export.py")
    shutil.copy2(Path(__file__).with_name("gsdd_bench_compat.py"), node / "gsdd_bench_compat.py")

    patch_main(node / "main.py")
    patch_heuristic(node / "heuristic_selection.py")
    patch_utils(node / "utils.py")
    patch_model_construct(node / "models/construct.py")
    print(f"Patched DShield checkout for GSDD-Bench: {repo}")


if __name__ == "__main__":
    main()
