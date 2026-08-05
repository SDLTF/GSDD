from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gsdd.config import Config, load_config, save_config
from gsdd.data import (
    GraphData,
    load_dataset,
    make_triggered_test_graph,
    poison_training_graph,
)
from gsdd.diagnostics import compute_diagnostics
from gsdd.graph_ops import build_normalized_adjacency, build_normalized_laplacian
from gsdd.models import DGIModel, SupervisedGCN
from gsdd.spectral import estimate_local_spectral_moments
from gsdd.train import (
    accuracy,
    extract_dgi_hidden,
    extract_supervised_hidden,
    train_dgi,
    train_supervised,
)
from gsdd.utils import (
    environment_info,
    log,
    make_run_dir,
    resolve_device,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GSDD-v0.4 causal-ablation and spectral-shape diagnostic experiment"
    )
    parser.add_argument("--config", type=str, default="configs/cora_low_poison.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--poison-count", type=int, default=None)
    parser.add_argument("--selection-method", choices=["dirty_label", "clean_label"], default=None)
    parser.add_argument("--ablation-mode", choices=["full", "label_only", "trigger_only", "none"], default=None)
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.seed is not None:
        cfg.experiment.seed = args.seed
    if args.device is not None:
        cfg.experiment.device = args.device
    if args.output_root is not None:
        cfg.experiment.output_root = args.output_root
    if args.name is not None:
        cfg.experiment.name = args.name
    if args.poison_count is not None:
        cfg.attack.poison_count = args.poison_count
    if args.selection_method is not None:
        cfg.attack.selection_method = args.selection_method
    if args.ablation_mode is not None:
        cfg.attack.ablation_mode = args.ablation_mode
    return cfg


def save_history(history: list[dict[str, float]], path: Path) -> None:
    pd.DataFrame(history).to_csv(path, index=False)


@torch.no_grad()
def evaluate_model(
    model: SupervisedGCN,
    graph: GraphData,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    graph_device = graph.to(device)
    adjacency = build_normalized_adjacency(
        graph_device.edge_index,
        graph_device.num_nodes,
        device=device,
        add_self_loops=True,
    )
    model.eval()
    logits, _ = model(graph_device.x, adjacency)
    test_acc = accuracy(logits, graph_device.y, graph_device.test_mask)
    return test_acc, logits


def make_summary_markdown(summary: dict, path: Path) -> None:
    detection = summary.get("detection", {})
    main_scores = [
        "legacy_combined",
        "global_mad_combined",
        "global_ecdf_combined",
        "trimmed_combined",
        "hybrid_combined",
        "global_ecdf_scale_invariant",
        "hybrid_scale_invariant",
        "shape_geometry_mahalanobis",
    ]
    lines = [
        "# GSDD-v0.4 Causal-Ablation Diagnostic Summary",
        "",
        "## Run status",
        "",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Dataset: `{summary.get('dataset')}`",
        f"- Seed: `{summary.get('seed')}`",
        f"- Device: `{summary.get('device')}`",
        f"- Selection method: `{summary.get('selection_method')}`",
        f"- Causal ablation mode: `{summary.get('ablation_mode')}`",
        f"- Poisoned training victims: `{summary.get('poisoned_training_victims')}`",
        f"- Clean test accuracy: `{summary.get('clean_test_accuracy', float('nan')):.4f}`",
        f"- Triggered test ASR: `{summary.get('triggered_test_asr', float('nan')):.4f}`",
        f"- Maximum observed-label contamination (diagnostic): `{summary.get('max_label_contamination_diagnostic', float('nan')):.4f}`",
        "",
        "## Calibration comparison",
        "",
        "| Score | AUROC | AUPRC | FPR@95TPR | F1 (oracle-k) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in main_scores:
        metrics = detection.get(name)
        if not isinstance(metrics, dict) or "auroc" not in metrics:
            continue
        lines.append(
            f"| {name} | {metrics['auroc']:.4f} | {metrics['auprc']:.4f} | "
            f"{metrics['fpr_at_95_tpr']:.4f} | {metrics['f1_oracle_k']:.4f} |"
        )

    warnings = summary.get("diagnostic_warnings", [])
    if warnings:
        lines.extend(["", "## Diagnostic warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings])

    top_features = summary.get("raw_feature_top", [])[:10]
    if top_features:
        lines.extend(
            [
                "",
                "## Strongest primitive spectral coordinates",
                "",
                "Ground-truth poison labels are used only in this section to audit whether a raw coordinate contains signal. These directions are not used to construct operational scores.",
                "",
                "| Group | Feature | Poison direction | Oriented AUROC | Clean mean | Poison mean |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for item in top_features:
            lines.append(
                f"| {item['group']} | {item['feature']} | {item['direction_for_poison']} | "
                f"{item['auroc_oriented_diagnostic']:.4f} | {item['clean_mean']:.6g} | "
                f"{item['poison_mean']:.6g} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `legacy_combined` reproduces the v0.1 label/degree-conditioned MAD calibration.",
            "- `global_mad_combined` uses no observed labels and therefore cannot be reversed by a majority-poisoned target class.",
            "- `global_ecdf_combined` is a direction-free empirical two-tail score.",
            "- `trimmed_combined` removes globally suspicious nodes before rebuilding class references.",
            "- `hybrid_combined` rank-fuses global MAD, global ECDF, and trimmed calibration.",
            "- `global_ecdf_scale_invariant` excludes the scale-sensitive raw transfer level and uses centered transfer shape.",
            "- `hybrid_scale_invariant` rank-fuses the scale-invariant component family.",
            "- `raw_feature_metrics.csv` is a diagnostic signal audit, not a deployable detector, because it uses poison ground truth to select the better direction.",
            "- Compare `transfer_level` against `transfer_shape` before claiming a genuine graph-frequency mechanism.",
            "- `shape_geometry_mahalanobis` is a label-free multivariate distance in the joint zero-sum transfer-shape space.",
            "- The four ablation modes separate label conflict, trigger perturbation, and their interaction.",
            "",
            "See `node_scores.csv`, `detection_metrics.json`, `raw_feature_metrics.csv`, and the generated comparison plots.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config, source_config: Path) -> Path:
    set_seed(cfg.experiment.seed)
    device = resolve_device(cfg.experiment.device)
    run_dir = make_run_dir(
        cfg.experiment.output_root,
        cfg.experiment.name,
        cfg.experiment.seed,
    )
    save_config(cfg, run_dir / "config_resolved.yaml")
    shutil.copy2(source_config, run_dir / "config_source.yaml")
    save_json(environment_info(device), run_dir / "environment.json")

    verbose = cfg.output.verbose
    log(f"[GSDD] run_dir={run_dir}", verbose)
    log(f"[GSDD] device={device}", verbose)

    summary: dict = {
        "status": "running",
        "dataset": cfg.dataset.name,
        "seed": cfg.experiment.seed,
        "device": str(device),
    }
    save_json(summary, run_dir / "summary.json")

    try:
        clean_graph = load_dataset(
            name=cfg.dataset.name,
            root=cfg.dataset.root,
            normalize_features=cfg.dataset.normalize_features,
            seed=cfg.experiment.seed,
            synthetic_nodes=cfg.dataset.synthetic_nodes,
            synthetic_features=cfg.dataset.synthetic_features,
            synthetic_classes=cfg.dataset.synthetic_classes,
        )
        log(
            f"[Data] nodes={clean_graph.num_nodes} edges={clean_graph.edge_index.size(1)} "
            f"features={clean_graph.num_features} classes={clean_graph.num_classes}",
            verbose,
        )

        if cfg.attack.enabled:
            graph, poison_victims = poison_training_graph(
                clean_graph,
                target_class=cfg.attack.target_class,
                selection_method=cfg.attack.selection_method,
                ablation_mode=cfg.attack.ablation_mode,
                poison_count=cfg.attack.poison_count,
                trigger_size=cfg.attack.trigger_size,
                trigger_feature_count=cfg.attack.trigger_feature_count,
                trigger_feature_value=cfg.attack.trigger_feature_value,
                stamp_victim_features=cfg.attack.stamp_victim_features,
                seed=cfg.experiment.seed,
            )
        else:
            graph = clean_graph
            poison_victims = torch.empty(0, dtype=torch.long)

        log(
            f"[Attack] poisoned_victims={len(poison_victims)} total_nodes_after_trigger={graph.num_nodes}",
            verbose,
        )

        graph_device = graph.to(device)
        adjacency_norm = build_normalized_adjacency(
            graph_device.edge_index,
            graph_device.num_nodes,
            device=device,
            add_self_loops=True,
        )
        laplacian = build_normalized_laplacian(
            graph_device.edge_index,
            graph_device.num_nodes,
            device=device,
        )

        supervised = SupervisedGCN(
            in_features=graph_device.num_features,
            hidden_dim=cfg.model.hidden_dim,
            num_classes=graph_device.num_classes,
            dropout=cfg.model.dropout,
        ).to(device)
        supervised_result = train_supervised(
            model=supervised,
            x=graph_device.x,
            y=graph_device.y,
            adjacency_norm=adjacency_norm,
            train_mask=graph_device.train_mask,
            val_mask=graph_device.val_mask,
            epochs=cfg.model.supervised_epochs,
            learning_rate=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay,
            patience=cfg.model.patience,
            verbose=verbose,
        )
        save_history(supervised_result.history, run_dir / "history_supervised.csv")

        dgi = DGIModel(
            in_features=graph_device.num_features,
            hidden_dim=cfg.model.hidden_dim,
        ).to(device)
        dgi_result = train_dgi(
            model=dgi,
            x=graph_device.x,
            adjacency_norm=adjacency_norm,
            epochs=cfg.model.ssl_epochs,
            learning_rate=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay,
            patience=cfg.model.patience,
            verbose=verbose,
        )
        save_history(dgi_result.history, run_dir / "history_ssl.csv")

        clean_test_accuracy, clean_logits = evaluate_model(supervised, graph, device)
        if cfg.attack.enabled:
            triggered_test_graph, test_victims = make_triggered_test_graph(
                graph,
                target_class=cfg.attack.target_class,
                trigger_size=cfg.attack.trigger_size,
                trigger_feature_value=cfg.attack.trigger_feature_value,
                stamp_victim_features=cfg.attack.stamp_victim_features,
                test_victim_count=cfg.attack.test_victim_count,
                seed=cfg.experiment.seed,
            )
            _, triggered_logits = evaluate_model(supervised, triggered_test_graph, device)
            victim_device = test_victims.to(device)
            triggered_asr = float(
                (triggered_logits[victim_device].argmax(dim=1) == cfg.attack.target_class)
                .float()
                .mean()
                .item()
            ) if test_victims.numel() else float("nan")
        else:
            test_victims = torch.empty(0, dtype=torch.long)
            triggered_asr = float("nan")

        _, supervised_hidden = extract_supervised_hidden(
            supervised, graph_device.x, adjacency_norm
        )
        ssl_hidden = extract_dgi_hidden(dgi, graph_device.x, adjacency_norm)

        selected_layers = [index - 1 for index in cfg.spectral.layers]
        if any(index < 0 or index >= len(supervised_hidden) for index in selected_layers):
            raise ValueError(
                f"spectral.layers={cfg.spectral.layers} is incompatible with "
                f"the {len(supervised_hidden)} available hidden layers"
            )
        supervised_hidden = [supervised_hidden[index] for index in selected_layers]
        ssl_hidden = [ssl_hidden[index] for index in selected_layers]

        moments = estimate_local_spectral_moments(
            laplacian=laplacian,
            orders=cfg.spectral.moment_orders,
            probes=cfg.spectral.hutchinson_probes,
            seed=cfg.experiment.seed + 17,
        )

        candidate_indices = torch.where(
            graph_device.train_mask
            & (torch.arange(graph_device.num_nodes, device=device) < graph_device.num_original_nodes)
        )[0]
        _, detection_metrics, diagnostic_extra = compute_diagnostics(
            x=graph_device.x,
            labels=graph_device.y,
            edge_index=graph_device.edge_index,
            laplacian=laplacian,
            moments=moments,
            supervised_hidden=supervised_hidden,
            ssl_hidden=ssl_hidden,
            candidate_indices=candidate_indices,
            poison_mask=graph_device.poison_mask,
            num_bands=cfg.spectral.num_bands,
            epsilon=cfg.spectral.epsilon,
            degree_bins=cfg.detection.degree_bins,
            global_degree_bins=cfg.detection.global_degree_bins,
            minimum_group_size=cfg.detection.minimum_group_size,
            mad_epsilon=cfg.detection.mad_epsilon,
            score_clip=cfg.detection.score_clip,
            trim_fraction=cfg.detection.trim_fraction,
            trim_minimum_keep=cfg.detection.trim_minimum_keep,
            raw_feature_topk=cfg.detection.raw_feature_topk,
            topk_fraction=cfg.detection.topk_fraction,
            output_dir=run_dir,
            make_plots=cfg.output.make_plots,
        )

        if cfg.output.save_models:
            torch.save(supervised.state_dict(), run_dir / "supervised_gcn.pt")
            torch.save(dgi.state_dict(), run_dir / "dgi_encoder.pt")

        summary.update(
            {
                "status": "success",
                "num_nodes_clean": clean_graph.num_nodes,
                "num_nodes_poisoned_graph": graph.num_nodes,
                "num_edges_poisoned_graph": int(graph.edge_index.size(1)),
                "poisoned_training_victims": int(poison_victims.numel()),
                "triggered_test_victims": int(test_victims.numel()),
                "target_class": cfg.attack.target_class,
                "selection_method": cfg.attack.selection_method,
                "ablation_mode": cfg.attack.ablation_mode,
                "clean_test_accuracy": clean_test_accuracy,
                "triggered_test_asr": triggered_asr,
                "supervised_best_epoch": supervised_result.best_epoch,
                "ssl_best_epoch": dgi_result.best_epoch,
                "detection": detection_metrics,
                "diagnostic_warnings": diagnostic_extra.get("diagnostic_warnings", []),
                "max_label_contamination_diagnostic": diagnostic_extra.get("max_label_contamination_diagnostic"),
                "raw_feature_top": diagnostic_extra.get("raw_feature_top", []),
            }
        )
        save_json(summary, run_dir / "summary.json")
        make_summary_markdown(summary, run_dir / "SUMMARY.md")
        log(
            f"[Done] clean_acc={clean_test_accuracy:.4f} ASR={triggered_asr:.4f}",
            verbose,
        )
        if isinstance(detection_metrics.get("hybrid_combined"), dict):
            log(
                f"[Calibration] hybrid AUROC={detection_metrics['hybrid_combined']['auroc']:.4f} "
                f"AUPRC={detection_metrics['hybrid_combined']['auprc']:.4f}",
                verbose,
            )
        if diagnostic_extra.get("diagnostic_warnings"):
            for warning in diagnostic_extra["diagnostic_warnings"]:
                log(f"[Warning] {warning}", verbose)
        return run_dir
    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        save_json(summary, run_dir / "summary.json")
        (run_dir / "ERROR.txt").write_text(summary["traceback"], encoding="utf-8")
        raise


def main() -> int:
    args = parse_args()
    source_config = Path(args.config).resolve()
    cfg = apply_overrides(load_config(source_config), args)
    run_dir = run(cfg, source_config)
    print(f"RESULT_DIR={run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
