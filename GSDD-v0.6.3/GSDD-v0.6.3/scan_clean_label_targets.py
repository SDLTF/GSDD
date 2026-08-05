from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from gsdd.config import load_config
from gsdd.data import load_dataset
from gsdd.graph_ops import build_normalized_adjacency
from gsdd.models import SupervisedGCN
from gsdd.reproducibility import configure_reproducibility
from gsdd.train import accuracy, train_supervised
from gsdd.utils import resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan low-baseline target classes for v0.6.3")
    parser.add_argument("--config", default="configs/cora_clean_label_factorial.yaml")
    parser.add_argument("--seed", type=int, default=1027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/clean_label_target_scan.json")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.experiment.seed = args.seed
    cfg.experiment.device = args.device
    if not args.allow_cpu and not args.device.lower().startswith("cuda"):
        raise RuntimeError("Formal target scan is CUDA-only")
    if args.device.lower().startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    configure_reproducibility(cfg.reproducibility)
    set_seed(args.seed)
    device = resolve_device(args.device)
    graph = load_dataset(
        cfg.dataset.name,
        cfg.dataset.root,
        cfg.dataset.normalize_features,
        args.seed,
        cfg.dataset.synthetic_nodes,
        cfg.dataset.synthetic_features,
        cfg.dataset.synthetic_classes,
    ).to(device)
    adjacency = build_normalized_adjacency(
        graph.edge_index, graph.num_nodes, device=device, add_self_loops=True
    )
    set_seed(args.seed + 271)
    model = SupervisedGCN(
        graph.num_features,
        cfg.model.hidden_dim,
        graph.num_classes,
        cfg.model.dropout,
    ).to(device)
    initial_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(initial_state)
    set_seed(args.seed + cfg.paired.training_seed_offset)
    result = train_supervised(
        model,
        graph.x,
        graph.y,
        adjacency,
        graph.train_mask,
        graph.val_mask,
        cfg.model.supervised_epochs,
        cfg.model.learning_rate,
        cfg.model.weight_decay,
        cfg.model.patience,
        verbose=False,
    )
    model.eval()
    with torch.no_grad():
        logits, _ = model(graph.x, adjacency)
        predictions = logits.argmax(dim=1)
    rows = []
    node_ids = torch.arange(graph.num_nodes, device=device)
    for target in range(graph.num_classes):
        non_target_test = graph.test_mask & (graph.y != target) & (node_ids < graph.num_original_nodes)
        count = int(non_target_test.sum().item())
        asr = float((predictions[non_target_test] == target).float().mean().item()) if count else float("nan")
        target_train_count = int((graph.train_mask & (graph.y == target)).sum().item())
        rows.append(
            {
                "target_class": target,
                "clean_no_trigger_asr": asr,
                "non_target_test_count": count,
                "target_train_count": target_train_count,
                "eligible": bool(
                    asr <= cfg.attack.clean_label_target_scan_max
                    and target_train_count >= 2
                ),
            }
        )
    ordered = sorted(rows, key=lambda item: (not item["eligible"], item["clean_no_trigger_asr"]))
    selected = [item["target_class"] for item in ordered if item["eligible"]][
        : cfg.attack.clean_label_max_targets
    ]
    if not selected:
        selected = [item["target_class"] for item in ordered[: cfg.attack.clean_label_max_targets]]
    payload = {
        "dataset": cfg.dataset.name,
        "seed": args.seed,
        "device": str(device),
        "best_epoch": int(result.best_epoch),
        "clean_accuracy": accuracy(logits, graph.y, graph.test_mask),
        "threshold": cfg.attack.clean_label_target_scan_max,
        "max_targets": cfg.attack.clean_label_max_targets,
        "targets": rows,
        "selected_target_classes": selected,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(payload, output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
