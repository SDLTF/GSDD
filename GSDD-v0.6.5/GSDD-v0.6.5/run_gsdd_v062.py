from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from gsdd.config import Config, load_config, save_config
from gsdd.data import GraphData, load_dataset
from gsdd.attack_families_v062 import (
    ATTACK_FAMILIES_V062,
    build_attack_plan_v062,
    build_paired_graphs,
    make_triggered_test_graph_from_plan,
)
from gsdd.graph_ops import build_normalized_adjacency, build_normalized_laplacian
from gsdd.models import SupervisedGCN
from gsdd.paired_diagnostics import MODES, compute_paired_diagnostics
from gsdd.reproducibility import configure_reproducibility
from gsdd.train import accuracy, extract_supervised_hidden, train_supervised
from gsdd.utils import environment_info, log, make_run_dir, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GSDD-v0.6.2 CUDA-only binding-aware attack-validity audit"
    )
    parser.add_argument("--config", default="configs/cora_attack_generalization.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="Only cuda/cuda:0 are accepted in v0.6.2")
    parser.add_argument("--attack-family", choices=ATTACK_FAMILIES_V062, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--poison-count", type=int, default=None)
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument(
        "--selection-method", choices=("dirty_label", "clean_label"), default=None
    )
    parser.add_argument("--no-repeat-control", action="store_true")
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.seed is not None:
        cfg.experiment.seed = args.seed
    if args.device is not None and not args.device.lower().startswith("cuda"):
        raise ValueError("GSDD-v0.6.2 is CUDA-only; --device must start with 'cuda'")
    cfg.experiment.device = "cuda" if args.device is None else args.device
    if args.attack_family is not None:
        cfg.attack.family = args.attack_family
    if args.name is not None:
        cfg.experiment.name = args.name
    if args.output_root is not None:
        cfg.experiment.output_root = args.output_root
    if args.poison_count is not None:
        cfg.attack.poison_count = args.poison_count
    if args.target_class is not None:
        cfg.attack.target_class = args.target_class
    if args.selection_method is not None:
        cfg.attack.selection_method = args.selection_method
    if args.no_repeat_control:
        cfg.paired.repeat_full_control = False
    return cfg


def save_history(history: list[dict[str, float]], path: Path) -> None:
    pd.DataFrame(history).to_csv(path, index=False)


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_max_abs_difference(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> float:
    value = 0.0
    for key in first:
        value = max(
            value,
            float((first[key].detach().cpu() - second[key].detach().cpu()).abs().max().item()),
        )
    return value


def assert_graph_equal(first: GraphData, second: GraphData, compare_labels: bool) -> None:
    fields = ["x", "edge_index", "train_mask", "val_mask", "test_mask"]
    if compare_labels:
        fields.append("y")
    for field in fields:
        if not torch.equal(getattr(first, field), getattr(second, field)):
            raise AssertionError(f"Paired graph invariant failed for field: {field}")


@torch.no_grad()
def model_forward(
    model: SupervisedGCN,
    graph: GraphData,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor], float]:
    graph_device = graph.to(device)
    adjacency = build_normalized_adjacency(
        graph_device.edge_index,
        graph_device.num_nodes,
        device=device,
        add_self_loops=True,
    )
    model.eval()
    logits, hidden = model(graph_device.x, adjacency)
    test_accuracy = accuracy(logits, graph_device.y, graph_device.test_mask)
    return logits, hidden, test_accuracy


def train_one(
    mode: str,
    graph: GraphData,
    cfg: Config,
    initial_state: dict[str, torch.Tensor],
    training_seed: int,
    device: torch.device,
    run_dir: Path,
    history_suffix: str = "",
) -> tuple[SupervisedGCN, Any]:
    graph_device = graph.to(device)
    adjacency = build_normalized_adjacency(
        graph_device.edge_index,
        graph_device.num_nodes,
        device=device,
        add_self_loops=True,
    )
    model = SupervisedGCN(
        in_features=graph_device.num_features,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=graph_device.num_classes,
        dropout=cfg.model.dropout,
    ).to(device)
    model.load_state_dict(initial_state)
    set_seed(training_seed)
    log(f"[Paired] training mode={mode} shared_seed={training_seed}", cfg.output.verbose)
    result = train_supervised(
        model=model,
        x=graph_device.x,
        y=graph_device.y,
        adjacency_norm=adjacency,
        train_mask=graph_device.train_mask,
        val_mask=graph_device.val_mask,
        epochs=cfg.model.supervised_epochs,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        patience=cfg.model.patience,
        verbose=cfg.output.verbose,
    )
    save_history(result.history, run_dir / f"history_{mode}{history_suffix}.csv")
    return model, result


def make_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    behavior = summary.get("model_behavior", {})
    detection = summary.get("paired_detection", {})
    permutation = summary.get("permutation_tests", {})
    score_order = [
        "did_shape_l2",
        "did_distribution_l2",
        "did_spectral_hybrid",
        "did_target_logit_abs",
        "did_spectral_logit_hybrid",
        "did_raw_l2",
        "did_level_l2",
    ]
    lines = [
        "# GSDD-v0.6.2 Binding-Aware Attack Validity Summary",
        "",
        f"- Attack family: `{summary.get('attack_family')}`",
        f"- Trigger motif: `{summary.get('trigger_motif')}`",
        "",
        "## Run status",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Dataset: `{summary.get('dataset')}`",
        f"- Seed: `{summary.get('seed')}`",
        f"- Device: `{summary.get('device')}`",
        f"- Selected victims: `{summary.get('victim_count')}`",
        f"- Shared initialization SHA-256: `{summary.get('initialization_sha256')}`",
        f"- Attack validity: `{summary.get('attack_validity', {}).get('status', 'unknown')}`",
        f"- Validity reasons: `{summary.get('attack_validity', {}).get('reasons', [])}`",
        "",
        "## Model behavior",
        "",
        "| Mode | Clean accuracy | Triggered ASR | Best epoch |",
        "|---|---:|---:|---:|",
    ]
    for mode in MODES:
        item = behavior.get(mode, {})
        lines.append(
            f"| {mode} | {item.get('clean_accuracy', float('nan')):.4f} | "
            f"{item.get('triggered_asr', float('nan')):.4f} | {item.get('best_epoch', 'NA')} |"
        )

    lines.extend(
        [
            "",
            "## Paired difference-in-differences detection",
            "",
            "| Score | AUROC | AUPRC | FPR@95TPR | Permutation p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in score_order:
        item = detection.get(name)
        if not isinstance(item, dict):
            continue
        p_value = permutation.get(name, {}).get("p_value", float("nan"))
        lines.append(
            f"| {name} | {item['auroc']:.4f} | {item['auprc']:.4f} | "
            f"{item['fpr_at_95_tpr']:.4f} | {p_value:.4g} |"
        )

    repeat = summary.get("repeat_control", {})
    lines.extend(
        [
            "",
            "## Numerical repeat control",
            "",
            f"- Enabled: `{repeat.get('enabled', False)}`",
            f"- Parameter max absolute difference: `{repeat.get('parameter_max_abs_difference', float('nan')):.6g}`",
            f"- Training-logit max absolute difference: `{repeat.get('training_logit_max_abs_difference', float('nan')):.6g}`",
            f"- Best-epoch difference: `{repeat.get('best_epoch_difference', 'NA')}`",
            "",
            "## Interpretation",
            "",
            "The primary spectral interaction is",
            "",
            "$$",
            r"D_{\mathrm{DID}}=[T_{\mathrm{full}}-T_{\mathrm{trigger-only}}]-[T_{\mathrm{label-only}}-T_{\mathrm{none}}]",
            "$$",
            "",
            "The first bracket isolates label binding on the same trigger-bearing graph. The second bracket estimates ordinary dirty-label learning on the unchanged graph. Their difference removes the input trigger signature and the additive label-conflict effect.",
            "",
            "`did_target_logit_abs` is a non-spectral positive-control anchor. The central research test is whether `did_shape_l2` and `did_distribution_l2` remain predictive across seeds without relying on this logit anchor.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config, source_config: Path) -> Path:
    if cfg.experiment.device.lower() != "cuda" and not cfg.experiment.device.lower().startswith("cuda:"):
        raise RuntimeError("GSDD-v0.6.2 formal experiments are CUDA-only")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GSDD-v0.6.2 requires CUDA, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build before running the experiment."
        )
    reproducibility_settings = configure_reproducibility(cfg.reproducibility)
    set_seed(cfg.experiment.seed)
    device = resolve_device(cfg.experiment.device)
    run_dir = make_run_dir(cfg.experiment.output_root, cfg.experiment.name, cfg.experiment.seed)
    save_config(cfg, run_dir / "config_resolved.yaml")
    shutil.copy2(source_config, run_dir / "config_source.yaml")
    save_json(environment_info(device), run_dir / "environment.json")

    summary: dict[str, Any] = {
        "status": "running",
        "dataset": cfg.dataset.name,
        "seed": cfg.experiment.seed,
        "device": str(device),
        "reproducibility": reproducibility_settings,
    }
    save_json(summary, run_dir / "summary.json")
    log(f"[GSDD-v0.6.2] run_dir={run_dir}", cfg.output.verbose)
    log(f"[GSDD-v0.6.2] device={device} attack_family={cfg.attack.family}", cfg.output.verbose)

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
        attack_plan = build_attack_plan_v062(clean_graph, cfg, device)
        victims = attack_plan.victims
        graphs = build_paired_graphs(clean_graph, attack_plan, cfg, device)
        assert_graph_equal(graphs["none"], graphs["label_only"], compare_labels=False)
        assert_graph_equal(graphs["trigger_only"], graphs["full"], compare_labels=False)
        if not torch.equal(graphs["none"].y, clean_graph.y):
            raise AssertionError("none labels differ from clean labels")
        if not torch.equal(graphs["trigger_only"].y[: clean_graph.num_nodes], clean_graph.y):
            raise AssertionError("trigger_only labels differ from clean labels")
        attack_diagnostics = dict(attack_plan.diagnostics)
        generator_history = attack_diagnostics.pop("generator_history", None)
        if generator_history:
            pd.DataFrame(generator_history).to_csv(
                run_dir / "attack_generator_history.csv", index=False
            )
        save_json(attack_diagnostics, run_dir / "attack_diagnostics.json")
        trigger_features = attack_plan.generate(clean_graph, victims, device).detach().cpu()
        torch.save(
            {
                "family": attack_plan.family,
                "motif": attack_plan.motif,
                "stamp_strength": attack_plan.stamp_strength,
                "victims": victims,
                "trigger_features": trigger_features,
                "generator_state_dict": (
                    None
                    if attack_plan.generator is None
                    else {k: v.detach().cpu() for k, v in attack_plan.generator.state_dict().items()}
                ),
                "prototype_bank": (
                    None if attack_plan.prototype_bank is None else attack_plan.prototype_bank.detach().cpu()
                ),
            },
            run_dir / "attack_plan.pt",
        )
        log(
            f"[Data] nodes={clean_graph.num_nodes} features={clean_graph.num_features} "
            f"victims={victims.numel()} trigger_nodes={graphs['full'].num_nodes-clean_graph.num_nodes} "
            f"family={attack_plan.family} motif={attack_plan.motif}",
            cfg.output.verbose,
        )

        # Create one initialization and clone it into every intervention model.
        set_seed(cfg.experiment.seed + 271)
        template = SupervisedGCN(
            in_features=clean_graph.num_features,
            hidden_dim=cfg.model.hidden_dim,
            num_classes=clean_graph.num_classes,
            dropout=cfg.model.dropout,
        ).to(device)
        initial_state = copy.deepcopy(template.state_dict())
        initialization_sha256 = state_hash(initial_state)
        del template

        shared_training_seed = cfg.experiment.seed + cfg.paired.training_seed_offset
        models: dict[str, SupervisedGCN] = {}
        train_results: dict[str, Any] = {}
        for mode in MODES:
            models[mode], train_results[mode] = train_one(
                mode,
                graphs[mode],
                cfg,
                initial_state,
                shared_training_seed,
                device,
                run_dir,
            )

        repeat_control: dict[str, Any] = {"enabled": bool(cfg.paired.repeat_full_control)}
        repeat_model: SupervisedGCN | None = None
        repeat_result = None
        if cfg.paired.repeat_full_control:
            repeat_model, repeat_result = train_one(
                "full",
                graphs["full"],
                cfg,
                initial_state,
                shared_training_seed,
                device,
                run_dir,
                history_suffix="_repeat",
            )

        # A single inference graph is used for all ASR measurements.
        triggered_test_graph, test_victims = make_triggered_test_graph_from_plan(
            graphs["full"], clean_graph, attack_plan, cfg, device
        )

        logits_by_mode: dict[str, torch.Tensor] = {}
        hidden_by_mode: dict[str, list[torch.Tensor]] = {}
        behavior_rows: list[dict[str, Any]] = []
        behavior: dict[str, Any] = {}
        selected_layers = [index - 1 for index in cfg.spectral.layers]
        for mode in MODES:
            logits, hidden, clean_accuracy = model_forward(models[mode], graphs[mode], device)
            logits_by_mode[mode] = logits
            hidden_by_mode[mode] = [hidden[index] for index in selected_layers]
            triggered_logits, _, _ = model_forward(models[mode], triggered_test_graph, device)
            victim_device = test_victims.to(device)
            asr = float(
                (triggered_logits[victim_device].argmax(dim=1) == cfg.attack.target_class)
                .float()
                .mean()
                .item()
            )
            entry = {
                "mode": mode,
                "clean_accuracy": clean_accuracy,
                "triggered_asr": asr,
                "best_epoch": int(train_results[mode].best_epoch),
                "best_val_loss": float(train_results[mode].best_value),
            }
            behavior_rows.append(entry)
            behavior[mode] = entry
        pd.DataFrame(behavior_rows).to_csv(run_dir / "model_behavior.csv", index=False)

        control_asr_max = max(
            behavior["none"]["triggered_asr"],
            behavior["label_only"]["triggered_asr"],
            behavior["trigger_only"]["triggered_asr"],
        )
        functional_gap = (
            behavior["full"]["triggered_asr"]
            - behavior["trigger_only"]["triggered_asr"]
        )
        validity_reasons: list[str] = []
        if behavior["full"]["triggered_asr"] < cfg.attack.validity_full_asr_min:
            validity_reasons.append("full_asr_below_minimum")
        if control_asr_max > cfg.attack.validity_control_asr_max:
            validity_reasons.append("control_asr_above_maximum")
        if functional_gap < cfg.attack.validity_binding_gap_min:
            validity_reasons.append("full_minus_trigger_only_gap_below_minimum")
        attack_validity = {
            "is_valid": len(validity_reasons) == 0,
            "status": "valid_attack" if not validity_reasons else "invalid_attack",
            "reasons": validity_reasons,
            "full_asr": behavior["full"]["triggered_asr"],
            "none_asr": behavior["none"]["triggered_asr"],
            "label_only_asr": behavior["label_only"]["triggered_asr"],
            "trigger_only_asr": behavior["trigger_only"]["triggered_asr"],
            "control_asr_max": control_asr_max,
            "full_minus_trigger_only_gap": functional_gap,
            "thresholds": {
                "full_asr_min": cfg.attack.validity_full_asr_min,
                "control_asr_max": cfg.attack.validity_control_asr_max,
                "binding_gap_min": cfg.attack.validity_binding_gap_min,
            },
        }
        save_json(attack_validity, run_dir / "attack_validity.json")
        log(
            f"[AttackValidity] status={attack_validity['status']} "
            f"full={behavior['full']['triggered_asr']:.4f} "
            f"control_max={control_asr_max:.4f} gap={functional_gap:.4f}",
            cfg.output.verbose,
        )

        if repeat_model is not None and repeat_result is not None:
            repeat_logits, _, _ = model_forward(repeat_model, graphs["full"], device)
            repeat_control.update(
                {
                    "parameter_max_abs_difference": state_max_abs_difference(
                        models["full"].state_dict(), repeat_model.state_dict()
                    ),
                    "training_logit_max_abs_difference": float(
                        (logits_by_mode["full"] - repeat_logits).abs().max().item()
                    ),
                    "best_epoch_difference": int(
                        abs(train_results["full"].best_epoch - repeat_result.best_epoch)
                    ),
                    "repeat_best_epoch": int(repeat_result.best_epoch),
                }
            )
        save_json(repeat_control, run_dir / "repeat_control.json")

        clean_device = graphs["none"].to(device)
        trigger_device = graphs["full"].to(device)
        clean_laplacian = build_normalized_laplacian(
            clean_device.edge_index, clean_device.num_nodes, device=device
        )
        trigger_laplacian = build_normalized_laplacian(
            trigger_device.edge_index, trigger_device.num_nodes, device=device
        )
        candidate_indices = torch.where(clean_device.train_mask)[0]

        _, paired_metrics, extra = compute_paired_diagnostics(
            clean_x=clean_device.x,
            trigger_x=trigger_device.x,
            clean_edge_index=clean_device.edge_index,
            clean_laplacian=clean_laplacian,
            trigger_laplacian=trigger_laplacian,
            hidden_by_mode=hidden_by_mode,
            logits_by_mode=logits_by_mode,
            candidate_indices=candidate_indices,
            victim_mask=clean_device.poison_mask,
            clean_labels=clean_device.y,
            full_labels=trigger_device.y,
            target_class=cfg.attack.target_class,
            num_bands=cfg.spectral.num_bands,
            epsilon=cfg.spectral.epsilon,
            global_degree_bins=cfg.detection.global_degree_bins,
            minimum_group_size=cfg.detection.minimum_group_size,
            mad_epsilon=cfg.detection.mad_epsilon,
            score_clip=cfg.detection.score_clip,
            topk_fraction=cfg.detection.topk_fraction,
            permutation_repeats=cfg.paired.permutation_repeats,
            seed=cfg.experiment.seed,
            output_dir=run_dir,
            make_plots=cfg.output.make_plots,
        )

        graph_checks = {
            "none_label_graph_same_input": True,
            "trigger_full_graph_same_input": True,
            "selected_victim_ids": victims.tolist(),
            "full_label_changes": int((graphs["full"].y[: clean_graph.num_nodes] != clean_graph.y).sum().item()),
            "label_only_label_changes": int((graphs["label_only"].y != clean_graph.y).sum().item()),
        }
        save_json(graph_checks, run_dir / "paired_graph_checks.json")

        if cfg.output.save_models:
            for mode, model in models.items():
                torch.save(model.state_dict(), run_dir / f"supervised_{mode}.pt")

        summary.update(
            {
                "status": "success",
                "victim_count": int(victims.numel()),
                "test_victim_count": int(test_victims.numel()),
                "target_class": cfg.attack.target_class,
                "selection_method": cfg.attack.selection_method,
                "attack_family": attack_plan.family,
                "trigger_motif": attack_plan.motif,
                "attack_diagnostics": attack_diagnostics,
                "attack_validity": attack_validity,
                "initialization_sha256": initialization_sha256,
                "shared_training_seed": shared_training_seed,
                "model_behavior": behavior,
                "paired_detection": paired_metrics,
                "permutation_tests": extra["permutation_tests"],
                "raw_feature_top": extra["raw_feature_top"],
                "repeat_control": repeat_control,
                "graph_checks": graph_checks,
            }
        )
        save_json(summary, run_dir / "summary.json")
        make_summary_markdown(summary, run_dir / "SUMMARY.md")
        main = paired_metrics.get("did_shape_l2", {})
        log(
            f"[Done:{attack_plan.family}] validity={attack_validity['status']} full_ASR={behavior['full']['triggered_asr']:.4f} "
            f"trigger_only_ASR={behavior['trigger_only']['triggered_asr']:.4f} "
            f"DID_shape_AUROC={main.get('auroc', float('nan')):.4f}",
            cfg.output.verbose,
        )
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
