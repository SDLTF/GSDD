from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
from typing import Any
import torch


def _cpu(value):
    return value.detach().cpu() if isinstance(value, torch.Tensor) else value


def _jsonable(value: Any):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)




def _normalize_bundle(bundle):
    out = dict(bundle)
    poison_x = out["poison_x"]
    poison_y = out["poison_y"]
    if not isinstance(poison_x, torch.Tensor) or poison_x.ndim != 2:
        raise ValueError("poison_x must be a rank-2 tensor")
    if not isinstance(poison_y, torch.Tensor):
        poison_y = torch.as_tensor(poison_y)
    poison_y = poison_y.detach().cpu().long().reshape(-1)
    poison_num_nodes = int(poison_x.shape[0])
    label_num_nodes = int(poison_y.numel())
    if label_num_nodes > poison_num_nodes:
        raise ValueError(f"poison_y has {label_num_nodes} rows but poison_x has {poison_num_nodes} nodes")
    if label_num_nodes < poison_num_nodes:
        padded = torch.full((poison_num_nodes,), -1, dtype=torch.long)
        padded[:label_num_nodes] = poison_y
        poison_y = padded
    out["poison_y"] = poison_y
    out["poison_num_nodes"] = poison_num_nodes
    out["label_num_nodes_before_padding"] = label_num_nodes
    out["injected_node_idx"] = torch.arange(label_num_nodes, poison_num_nodes, dtype=torch.long)
    edge = out["poison_train_edge_index"].detach().cpu().long()
    if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= poison_num_nodes):
        raise ValueError("poison_train_edge_index references a node outside poison_x")
    out["poison_train_edge_index"] = edge
    for key in ("train_idx", "poison_train_idx", "val_idx", "clean_test_idx", "attack_test_idx", "attach_idx"):
        if key not in out:
            continue
        idx = out[key]
        if not isinstance(idx, torch.Tensor):
            idx = torch.as_tensor(idx)
        idx = idx.detach().cpu().long().reshape(-1)
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= poison_num_nodes):
            raise ValueError(f"{key} references a node outside poison_x")
        if idx.numel() and bool((poison_y[idx] < 0).any()):
            raise ValueError(f"{key} contains injected/unlabeled trigger nodes")
        out[key] = idx
    return out

def _git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def export_attack_artifact(*, output_root, run_id, args, data, clean_train_edge_index,
                           mask_edge_index, feat, train_edge_index, train_edge_weight,
                           labels, train_idx, val_idx, clean_test_idx, atk_idx,
                           attach_idx, train_node_idx, backdoor_gen_model):
    output_root = Path(output_root)
    run_id = run_id or f"{args.dataset}_{args.attack_method}_seed{args.seed}"
    out = output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    if train_edge_weight is None:
        train_edge_weight = torch.ones(train_edge_index.shape[1], dtype=torch.float32, device=train_edge_index.device)
    bundle = {
        "format_version": 2,
        "clean_x": _cpu(data.x),
        "clean_y": _cpu(data.y),
        "clean_full_edge_index": _cpu(data.edge_index),
        "clean_train_edge_index": _cpu(clean_train_edge_index),
        "heldout_edge_index": _cpu(mask_edge_index),
        "poison_x": _cpu(feat),
        "poison_y": _cpu(labels),
        "poison_train_edge_index": _cpu(train_edge_index),
        "poison_train_edge_weight": _cpu(train_edge_weight),
        "train_idx": _cpu(train_idx),
        "poison_train_idx": _cpu(train_node_idx),
        "val_idx": _cpu(val_idx),
        "clean_test_idx": _cpu(clean_test_idx),
        "attack_test_idx": _cpu(atk_idx),
        "attach_idx": _cpu(attach_idx) if attach_idx is not None else torch.empty(0, dtype=torch.long),
    }
    bundle = _normalize_bundle(bundle)
    torch.save(bundle, out / "artifact.pt")
    trigger_saved = False
    trigger_error = None
    try:
        torch.save(backdoor_gen_model, out / "trigger_model.pt")
        trigger_saved = True
    except Exception as exc:
        trigger_error = repr(exc)
    repo_dir = Path(__file__).resolve().parents[1]
    manifest = {
        "format_version": 2,
        "status": "generated",
        "dataset": args.dataset,
        "attack": args.attack_method,
        "seed": int(args.seed),
        "target_class": _jsonable(args.target_class),
        "trigger_size": int(args.trigger_size),
        "poison_count": int(len(bundle["attach_idx"])),
        "clean_num_nodes": int(data.x.shape[0]),
        "poison_num_nodes": int(bundle["poison_x"].shape[0]),
        "injected_node_count": int(bundle["injected_node_idx"].numel()),
        "victim_model": args.model,
        "dshield_commit": _git_commit(repo_dir),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "trigger_model_saved": trigger_saved,
        "trigger_model_error": trigger_error,
        "arguments": _jsonable(vars(args)),
        "metrics": {},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "READY").write_text("artifact generated\n", encoding="utf-8")
    print(f"[GSDD-Bench] exported official attack artifact: {out}", flush=True)
    return out


def load_attack_artifact(path, device):
    path = Path(path)
    bundle = _normalize_bundle(torch.load(path / "artifact.pt", map_location="cpu", weights_only=False))
    bundle = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in bundle.items()}
    trigger_path = path / "trigger_model.pt"
    trigger_model = torch.load(trigger_path, map_location=device, weights_only=False) if trigger_path.exists() else None
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return bundle, trigger_model, manifest


def update_metrics(path, defense_method, clean_accuracy, asr, defense_seconds):
    path = Path(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("metrics", {})[str(defense_method)] = {
        "clean_accuracy": float(clean_accuracy),
        "asr": float(asr),
        "defense_seconds": float(defense_seconds),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
