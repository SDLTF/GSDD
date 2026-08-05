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

from gsdd.attack_families_v065 import build_attack_plan_v065
from gsdd.clean_label_factorial import (
    FACTORIAL_MODEL_NAMES,
    FACTORIAL_TRIGGER_NAMES,
    build_test_factorial_graphs,
    build_training_factorial_graphs,
    compute_clean_label_factorial_diagnostics,
)
from gsdd.clean_label_dual import compute_generic_clean_label_diagnostics
from gsdd.config import Config, load_config, save_config
from gsdd.data import GraphData, load_dataset
from gsdd.graph_ops import build_normalized_adjacency, build_normalized_laplacian
from gsdd.models import SupervisedGCN
from gsdd.reproducibility import configure_reproducibility
from gsdd.train import accuracy, train_supervised
from gsdd.utils import environment_info, log, make_run_dir, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GSDD-v0.6.5 clean-label model x trigger factorial audit"
    )
    parser.add_argument("--config", default="configs/cora_clean_label_factorial.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--poison-count", type=int, default=None)
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument("--attack-mode", choices=["generic", "contextual"], default=None)
    parser.add_argument("--pair-weight", type=float, default=None)
    parser.add_argument("--trigger-size", type=int, default=None)
    parser.add_argument("--clean-cap-weight", type=float, default=None)
    parser.add_argument("--clean-probability-cap", type=float, default=None)
    parser.add_argument("--selectivity-weight", type=float, default=None)
    parser.add_argument("--selectivity-margin", type=float, default=None)
    parser.add_argument("--target-similarity-weight", type=float, default=None)
    parser.add_argument("--target-similarity-allowance", type=float, default=None)
    parser.add_argument("--raw-blend", type=float, default=None)
    parser.add_argument("--target-prototype-fraction", type=float, default=None)
    parser.add_argument("--outer-rounds", type=int, default=None)
    parser.add_argument("--poison-target-weight", type=float, default=None)
    parser.add_argument("--shuffled-target-weight", type=float, default=None)
    parser.add_argument("--allow-cpu", action="store_true", help="Smoke-test only")
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.seed is not None:
        cfg.experiment.seed = args.seed
    if args.device is not None:
        cfg.experiment.device = args.device
    if args.name is not None:
        cfg.experiment.name = args.name
    if args.output_root is not None:
        cfg.experiment.output_root = args.output_root
    if args.poison_count is not None:
        cfg.attack.poison_count = args.poison_count
    if args.target_class is not None:
        cfg.attack.target_class = args.target_class
    if args.attack_mode is not None:
        cfg.attack.clean_label_attack_mode = args.attack_mode
    if args.pair_weight is not None:
        cfg.attack.contextual_pair_weight = args.pair_weight
    if args.trigger_size is not None:
        cfg.attack.trigger_size = args.trigger_size
    if args.clean_cap_weight is not None:
        cfg.attack.generic_clean_cap_weight = args.clean_cap_weight
    if args.clean_probability_cap is not None:
        cfg.attack.generic_clean_probability_cap = args.clean_probability_cap
    if args.selectivity_weight is not None:
        cfg.attack.generic_selectivity_weight = args.selectivity_weight
    if args.selectivity_margin is not None:
        cfg.attack.generic_selectivity_margin = args.selectivity_margin
    if args.target_similarity_weight is not None:
        cfg.attack.generic_target_similarity_weight = args.target_similarity_weight
    if args.target_similarity_allowance is not None:
        cfg.attack.generic_target_similarity_allowance = args.target_similarity_allowance
    if args.raw_blend is not None:
        cfg.attack.distribution_raw_blend = args.raw_blend
    if args.target_prototype_fraction is not None:
        cfg.attack.distribution_target_prototype_fraction = args.target_prototype_fraction
    if args.outer_rounds is not None:
        cfg.attack.generator_outer_rounds = args.outer_rounds
    if args.poison_target_weight is not None:
        cfg.attack.poison_target_weight = args.poison_target_weight
    if args.shuffled_target_weight is not None:
        cfg.attack.generic_shuffled_target_weight = args.shuffled_target_weight
    cfg.attack.selection_method = "clean_label"
    cfg.attack.family = "dpgba_style_binding_aware"
    cfg._allow_cpu_v065 = bool(args.allow_cpu)  # type: ignore[attr-defined]
    return cfg


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_max_abs_difference(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> float:
    return max(
        float((first[key].detach().cpu() - second[key].detach().cpu()).abs().max().item())
        for key in first
    )


def save_history(history: list[dict[str, float]], path: Path) -> None:
    pd.DataFrame(history).to_csv(path, index=False)


def train_model(
    *,
    name: str,
    graph: GraphData,
    cfg: Config,
    initial_state: dict[str, torch.Tensor],
    training_seed: int,
    device: torch.device,
    run_dir: Path,
    suffix: str = "",
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
    log(f"[Factorial] training model={name} shared_seed={training_seed}", cfg.output.verbose)
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
    save_history(result.history, run_dir / f"history_{name}{suffix}.csv")
    return model, result


@torch.no_grad()
def forward_model(
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
    return logits, hidden, accuracy(logits, graph_device.y, graph_device.test_mask)


def make_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    behavior = summary.get("factorial_behavior", {})
    validity = summary.get("attack_validity", {})
    detection = summary.get("clean_label_detection", {})
    permutation = summary.get("permutation_tests", {})
    attack_mode = str(summary.get("attack_mode", validity.get("attack_mode", "unknown")))
    lines = [
        "# GSDD-v0.6.5 Generic Selective-activation Repair",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Attack mode: `{attack_mode}`",
        f"- Dataset: `{summary.get('dataset')}`",
        f"- Seed: `{summary.get('seed')}`",
        f"- Target class: `{summary.get('target_class')}`",
        f"- Poison count: `{summary.get('victim_count')}`",
        f"- Motif: `{summary.get('trigger_motif')}`",
        "",
        "## Factorial behavior",
        "",
        "| Model | Test trigger | Clean accuracy | ASR |",
        "|---|---|---:|---:|",
    ]
    for model_name in FACTORIAL_MODEL_NAMES:
        for condition in FACTORIAL_TRIGGER_NAMES:
            item = behavior.get(model_name, {}).get(condition, {})
            lines.append(
                f"| {model_name} | {condition} | "
                f"{item.get('clean_accuracy', float('nan')):.4f} | "
                f"{item.get('asr', float('nan')):.4f} |"
            )
    lines.extend(
        [
            "",
            "## Attack validity",
            "",
            f"- Status: `{validity.get('status')}`",
            f"- Reasons: `{validity.get('reasons', [])}`",
            f"- Admission ASR: `{validity.get('full_asr', float('nan')):.4f}`",
            f"- Maximum control ASR: `{validity.get('control_asr_max', float('nan')):.4f}`",
            f"- Admission gap: `{validity.get('binding_gap', float('nan')):.4f}`",
            f"- Contextual binding gap: `{validity.get('contextual_binding_gap', float('nan')):.4f}`",
            f"- Generic trigger DiD: `{validity.get('generic_did', float('nan')):.4f}`",
            "",
        ]
    )
    if attack_mode == "generic":
        lines.extend(
            [
                "Generic mode requires both matched and shuffled triggers to transfer:",
                "",
                "$$",
                r"\Delta_{\mathrm{generic}}=\min\{\operatorname{ASR}(M_p,T),\operatorname{ASR}(M_p,T_{\mathrm{shuffle}})\}-\max\{\operatorname{ASR}(M_c,T),\operatorname{ASR}(M_c,T_{\mathrm{shuffle}}),\operatorname{ASR}(M_p,\varnothing),\operatorname{ASR}(M_c,\varnothing)\}",
                "$$",
                "",
            ]
        )
        score_order = [
            "cl_generic_shape_l2",
            "cl_generic_distribution_l2",
            "cl_generic_spectral_hybrid",
            "cl_generic_target_logit_abs",
            "cl_generic_spectral_logit_hybrid",
            "cl_generic_raw_l2",
        ]
    else:
        lines.extend(
            [
                "Contextual mode requires matched triggers to outperform shuffled controls:",
                "",
                "$$",
                r"\Delta_{\mathrm{context}}=\operatorname{ASR}(M_p,T)-\max\{\operatorname{ASR}(M_c,T),\operatorname{ASR}(M_p,T_{\mathrm{shuffle}}),\operatorname{ASR}(M_p,\varnothing),\operatorname{ASR}(M_c,\varnothing)\}",
                "$$",
                "",
            ]
        )
        score_order = [
            "cl_did_shape_l2",
            "cl_did_distribution_l2",
            "cl_did_spectral_hybrid",
            "cl_did_target_logit_abs",
            "cl_did_spectral_logit_hybrid",
            "cl_did_raw_l2",
            "cl_no_control_l2",
        ]
    lines.extend(
        [
            "## Spectral interaction",
            "",
            "| Score | AUROC | AUPRC | FPR@95TPR | Permutation $p$ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in score_order:
        item = detection.get(name)
        if not isinstance(item, dict):
            continue
        p_value = permutation.get(name, {}).get("p_value", float("nan"))
        lines.append(
            f"| {name} | {item.get('auroc', float('nan')):.4f} | "
            f"{item.get('auprc', float('nan')):.4f} | "
            f"{item.get('fpr_at_95_tpr', float('nan')):.4f} | {p_value:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "Detection metrics count as defense evidence only after the corresponding generic or contextual attack passes its own functional validity gate.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config, source_config: Path) -> Path:
    allow_cpu = bool(getattr(cfg, "_allow_cpu_v065", False))
    if not allow_cpu and not cfg.experiment.device.lower().startswith("cuda"):
        raise RuntimeError("GSDD-v0.6.5 formal experiments are CUDA-only")
    if cfg.experiment.device.lower().startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "GSDD-v0.6.5 requires CUDA for formal experiments, but "
            "torch.cuda.is_available() is False"
        )
    if cfg.attack.selection_method != "clean_label":
        raise ValueError("GSDD-v0.6.5 only audits clean-label attacks")
    if str(cfg.attack.clean_label_attack_mode).lower().strip() != "generic":
        raise ValueError("GSDD-v0.6.5 is a generic-only selective-activation repair")

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
        "protocol": "generic_selective_activation_repair",
        "attack_mode": str(cfg.attack.clean_label_attack_mode),
        "generic_repair_parameters": {
            "clean_cap_weight": float(cfg.attack.generic_clean_cap_weight),
            "clean_probability_cap": float(cfg.attack.generic_clean_probability_cap),
            "selectivity_weight": float(cfg.attack.generic_selectivity_weight),
            "selectivity_margin": float(cfg.attack.generic_selectivity_margin),
            "target_similarity_weight": float(cfg.attack.generic_target_similarity_weight),
            "target_similarity_allowance": float(cfg.attack.generic_target_similarity_allowance),
            "raw_blend": float(cfg.attack.distribution_raw_blend),
            "target_prototype_fraction": float(cfg.attack.distribution_target_prototype_fraction),
            "outer_rounds": int(cfg.attack.generator_outer_rounds),
            "poison_target_weight": float(cfg.attack.poison_target_weight),
            "shuffled_target_weight": float(cfg.attack.generic_shuffled_target_weight),
        },
    }
    save_json(summary, run_dir / "summary.json")
    log(f"[GSDD-v0.6.5] run_dir={run_dir}", cfg.output.verbose)

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
        plan = build_attack_plan_v065(clean_graph, cfg, device)
        if plan.family != "dpgba_style_binding_aware":
            raise AssertionError("v0.6.5 formal protocol expects DPGBA-style plan")
        victims = plan.victims
        if not torch.all(clean_graph.y[victims] == cfg.attack.target_class):
            raise AssertionError("Clean-label victims must already belong to target class")

        training_graphs, training_features = build_training_factorial_graphs(
            clean_graph,
            plan,
            device,
            cfg.attack.clean_label_shuffle_shift,
        )
        if not torch.equal(training_graphs["matched"].y, training_graphs["shuffled"].y):
            raise AssertionError("Matched and shuffled clean-label graphs must share labels")
        if not torch.equal(
            training_graphs["matched"].edge_index, training_graphs["shuffled"].edge_index
        ):
            raise AssertionError("Matched and shuffled graphs must share topology")
        if not torch.allclose(
            training_features["matched"].sort(dim=0).values,
            training_features["shuffled"].sort(dim=0).values,
        ):
            raise AssertionError("Victim shuffling must preserve trigger marginals")

        attack_diagnostics = dict(plan.diagnostics)
        generator_history = attack_diagnostics.pop("generator_history", None)
        if generator_history:
            pd.DataFrame(generator_history).to_csv(
                run_dir / "attack_generator_history.csv", index=False
            )
        save_json(attack_diagnostics, run_dir / "attack_diagnostics.json")
        torch.save(
            {
                "family": plan.family,
                "motif": plan.motif,
                "victims": victims,
                "matched_training_trigger_features": training_features["matched"],
                "shuffled_training_trigger_features": training_features["shuffled"],
                "generator_state_dict": (
                    None
                    if plan.generator is None
                    else {key: value.detach().cpu() for key, value in plan.generator.state_dict().items()}
                ),
            },
            run_dir / "attack_plan_v064.pt",
        )

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
        shared_seed = cfg.experiment.seed + cfg.paired.training_seed_offset

        models: dict[str, SupervisedGCN] = {}
        train_results: dict[str, Any] = {}
        models["clean"], train_results["clean"] = train_model(
            name="clean",
            graph=training_graphs["none"],
            cfg=cfg,
            initial_state=initial_state,
            training_seed=shared_seed,
            device=device,
            run_dir=run_dir,
        )
        models["poison"], train_results["poison"] = train_model(
            name="poison",
            graph=training_graphs["matched"],
            cfg=cfg,
            initial_state=initial_state,
            training_seed=shared_seed,
            device=device,
            run_dir=run_dir,
        )

        repeat_control: dict[str, Any] = {"enabled": bool(cfg.paired.repeat_full_control)}
        if cfg.paired.repeat_full_control:
            repeat_model, repeat_result = train_model(
                name="poison",
                graph=training_graphs["matched"],
                cfg=cfg,
                initial_state=initial_state,
                training_seed=shared_seed,
                device=device,
                run_dir=run_dir,
                suffix="_repeat",
            )
            repeat_control.update(
                {
                    "parameter_max_abs_difference": state_max_abs_difference(
                        models["poison"].state_dict(), repeat_model.state_dict()
                    ),
                    "best_epoch_difference": int(
                        abs(train_results["poison"].best_epoch - repeat_result.best_epoch)
                    ),
                }
            )
        save_json(repeat_control, run_dir / "repeat_control.json")

        # Training-graph factorial inference for node-level spectral diagnosis.
        logits_factorial: dict[str, dict[str, torch.Tensor]] = {
            model_name: {} for model_name in FACTORIAL_MODEL_NAMES
        }
        hidden_factorial: dict[str, dict[str, list[torch.Tensor]]] = {
            model_name: {} for model_name in FACTORIAL_MODEL_NAMES
        }
        selected_layers = [index - 1 for index in cfg.spectral.layers]
        for model_name in FACTORIAL_MODEL_NAMES:
            for condition in FACTORIAL_TRIGGER_NAMES:
                logits_value, hidden_value, _ = forward_model(
                    models[model_name], training_graphs[condition], device
                )
                logits_factorial[model_name][condition] = logits_value
                hidden_factorial[model_name][condition] = [
                    hidden_value[index] for index in selected_layers
                ]

        # At test time both models are evaluated on identical graphs that retain
        # the clean-label training triggers. Only the test-trigger condition varies.
        test_graphs, test_victims, test_features = build_test_factorial_graphs(
            training_graphs["matched"],
            clean_graph,
            plan,
            cfg.attack.test_victim_count,
            cfg.experiment.seed,
            device,
            cfg.attack.clean_label_shuffle_shift,
        )
        victim_device = test_victims.to(device)
        behavior: dict[str, dict[str, Any]] = {
            model_name: {} for model_name in FACTORIAL_MODEL_NAMES
        }
        behavior_rows: list[dict[str, Any]] = []
        for model_name in FACTORIAL_MODEL_NAMES:
            for condition in FACTORIAL_TRIGGER_NAMES:
                logits_value, _, clean_accuracy = forward_model(
                    models[model_name], test_graphs[condition], device
                )
                asr = float(
                    (logits_value[victim_device].argmax(dim=1) == cfg.attack.target_class)
                    .float()
                    .mean()
                    .item()
                )
                entry = {
                    "model": model_name,
                    "trigger_condition": condition,
                    "clean_accuracy": clean_accuracy,
                    "asr": asr,
                    "best_epoch": int(train_results[model_name].best_epoch),
                }
                behavior[model_name][condition] = entry
                behavior_rows.append(entry)
        pd.DataFrame(behavior_rows).to_csv(
            run_dir / "clean_label_factorial_behavior.csv", index=False
        )

        attack_mode = str(cfg.attack.clean_label_attack_mode).lower().strip()
        poison_matched_asr = behavior["poison"]["matched"]["asr"]
        poison_shuffled_asr = behavior["poison"]["shuffled"]["asr"]
        baseline_asr = behavior["clean"]["none"]["asr"]
        clean_matched_asr = behavior["clean"]["matched"]["asr"]
        clean_shuffled_asr = behavior["clean"]["shuffled"]["asr"]
        poison_none_asr = behavior["poison"]["none"]["asr"]

        contextual_reference = max(
            baseline_asr, clean_matched_asr, poison_none_asr, poison_shuffled_asr
        )
        contextual_gap = poison_matched_asr - contextual_reference
        generic_trigger_floor = min(poison_matched_asr, poison_shuffled_asr)
        generic_clean_trigger_ceiling = max(clean_matched_asr, clean_shuffled_asr)
        generic_reference = max(
            baseline_asr, generic_clean_trigger_ceiling, poison_none_asr
        )
        generic_did = generic_trigger_floor - generic_reference

        validity_reasons: list[str] = []
        if attack_mode == "generic":
            full_asr = generic_trigger_floor
            control_asr_max = generic_reference
            binding_gap = generic_did
            if baseline_asr > cfg.attack.clean_label_baseline_asr_max:
                validity_reasons.append("clean_no_trigger_asr_above_maximum")
            if generic_clean_trigger_ceiling > cfg.attack.generic_clean_trigger_asr_max:
                validity_reasons.append("clean_model_generic_trigger_asr_above_maximum")
            if poison_none_asr > cfg.attack.generic_poison_no_trigger_asr_max:
                validity_reasons.append("poison_model_no_trigger_asr_above_maximum")
            if generic_trigger_floor < cfg.attack.generic_full_asr_min:
                validity_reasons.append("poison_model_generic_trigger_asr_below_minimum")
            if generic_did < cfg.attack.generic_did_min:
                validity_reasons.append("generic_trigger_did_below_minimum")
            valid_status = "valid_generic_clean_label_attack"
        elif attack_mode == "contextual":
            full_asr = poison_matched_asr
            control_asr_max = max(
                baseline_asr, clean_matched_asr, poison_none_asr,
                poison_shuffled_asr, clean_shuffled_asr
            )
            binding_gap = contextual_gap
            if baseline_asr > cfg.attack.clean_label_baseline_asr_max:
                validity_reasons.append("clean_no_trigger_asr_above_maximum")
            if clean_matched_asr > cfg.attack.clean_label_clean_trigger_asr_max:
                validity_reasons.append("clean_model_matched_trigger_asr_above_maximum")
            if poison_none_asr > cfg.attack.clean_label_poison_no_trigger_asr_max:
                validity_reasons.append("poison_model_no_trigger_asr_above_maximum")
            if poison_shuffled_asr > cfg.attack.clean_label_shuffled_asr_max:
                validity_reasons.append("poison_model_shuffled_trigger_asr_above_maximum")
            if poison_matched_asr < cfg.attack.clean_label_full_asr_min:
                validity_reasons.append("poison_model_matched_trigger_asr_below_minimum")
            if contextual_gap < cfg.attack.clean_label_binding_gap_min:
                validity_reasons.append("contextual_binding_gap_below_minimum")
            valid_status = "valid_contextual_clean_label_attack"
        else:
            raise ValueError(
                "clean_label_attack_mode must be 'generic' or 'contextual', "
                f"got {cfg.attack.clean_label_attack_mode!r}"
            )

        attack_validity = {
            "is_valid": not validity_reasons,
            "status": valid_status if not validity_reasons else f"invalid_{attack_mode}_clean_label_attack",
            "attack_mode": attack_mode,
            "reasons": validity_reasons,
            "full_asr": full_asr,
            "clean_no_trigger_asr": baseline_asr,
            "clean_matched_trigger_asr": clean_matched_asr,
            "clean_shuffled_trigger_asr": clean_shuffled_asr,
            "poison_no_trigger_asr": poison_none_asr,
            "poison_matched_trigger_asr": poison_matched_asr,
            "poison_shuffled_trigger_asr": poison_shuffled_asr,
            "control_asr_max": control_asr_max,
            "binding_gap": binding_gap,
            "contextual_binding_gap": contextual_gap,
            "generic_trigger_floor": generic_trigger_floor,
            "generic_clean_trigger_ceiling": generic_clean_trigger_ceiling,
            "generic_reference": generic_reference,
            "generic_did": generic_did,
            "thresholds": {
                "contextual_full_asr_min": cfg.attack.clean_label_full_asr_min,
                "contextual_binding_gap_min": cfg.attack.clean_label_binding_gap_min,
                "generic_full_asr_min": cfg.attack.generic_full_asr_min,
                "generic_did_min": cfg.attack.generic_did_min,
                "generic_clean_trigger_asr_max": cfg.attack.generic_clean_trigger_asr_max,
                "generic_poison_no_trigger_asr_max": cfg.attack.generic_poison_no_trigger_asr_max,
            },
        }
        save_json(attack_validity, run_dir / "clean_label_attack_validity.json")
        log(
            f"[CleanLabelValidity:{attack_mode}] status={attack_validity['status']} "
            f"full={full_asr:.4f} control_max={control_asr_max:.4f} gap={binding_gap:.4f}",
            cfg.output.verbose,
        )

        graph_x: dict[str, torch.Tensor] = {}
        graph_edge_index: dict[str, torch.Tensor] = {}
        graph_laplacian: dict[str, torch.Tensor] = {}
        for condition in FACTORIAL_TRIGGER_NAMES:
            graph_device = training_graphs[condition].to(device)
            graph_x[condition] = graph_device.x
            graph_edge_index[condition] = graph_device.edge_index
            graph_laplacian[condition] = build_normalized_laplacian(
                graph_device.edge_index, graph_device.num_nodes, device=device
            )

        candidate_indices = torch.where(
            training_graphs["none"].train_mask
            & (training_graphs["none"].y == cfg.attack.target_class)
        )[0].to(device)
        _, detection_metrics, extra = compute_clean_label_factorial_diagnostics(
            graph_x=graph_x,
            graph_edge_index=graph_edge_index,
            graph_laplacian=graph_laplacian,
            hidden=hidden_factorial,
            logits=logits_factorial,
            candidate_indices=candidate_indices,
            victim_mask=training_graphs["matched"].poison_mask.to(device),
            clean_labels=training_graphs["none"].y.to(device),
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

        _, generic_detection_metrics, generic_extra = compute_generic_clean_label_diagnostics(
            graph_x=graph_x,
            graph_edge_index=graph_edge_index,
            graph_laplacian=graph_laplacian,
            hidden=hidden_factorial,
            logits=logits_factorial,
            candidate_indices=candidate_indices,
            victim_mask=training_graphs["matched"].poison_mask.to(device),
            clean_labels=training_graphs["none"].y.to(device),
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
        )
        all_detection_metrics = {**detection_metrics, **generic_detection_metrics}
        all_permutation_tests = {
            **extra["permutation_tests"],
            **generic_extra["permutation_tests"],
        }

        graph_checks = {
            "selection_method": cfg.attack.selection_method,
            "victims_are_target_class": bool(
                torch.all(clean_graph.y[victims] == cfg.attack.target_class).item()
            ),
            "matched_shuffled_same_topology": bool(
                torch.equal(
                    training_graphs["matched"].edge_index,
                    training_graphs["shuffled"].edge_index,
                )
            ),
            "matched_shuffled_same_labels": bool(
                torch.equal(training_graphs["matched"].y, training_graphs["shuffled"].y)
            ),
            "label_changes": int(
                (training_graphs["matched"].y[: clean_graph.num_nodes] != clean_graph.y)
                .sum()
                .item()
            ),
            "candidate_count_target_class": int(candidate_indices.numel()),
            "selected_victim_ids": victims.tolist(),
            "test_victim_ids": test_victims.tolist(),
            "training_trigger_marginal_max_abs_difference": float(
                (
                    training_features["matched"].reshape(-1, clean_graph.num_features).sort(dim=0).values
                    - training_features["shuffled"].reshape(-1, clean_graph.num_features).sort(dim=0).values
                )
                .abs()
                .max()
                .item()
            ),
            "test_trigger_marginal_max_abs_difference": float(
                (
                    test_features["matched"].reshape(-1, clean_graph.num_features).sort(dim=0).values
                    - test_features["shuffled"].reshape(-1, clean_graph.num_features).sort(dim=0).values
                )
                .abs()
                .max()
                .item()
            ),
        }
        save_json(graph_checks, run_dir / "clean_label_factorial_checks.json")

        if cfg.output.save_models:
            for name, model in models.items():
                torch.save(model.state_dict(), run_dir / f"supervised_{name}.pt")

        summary.update(
            {
                "status": "success",
                "target_class": cfg.attack.target_class,
                "victim_count": int(victims.numel()),
                "test_victim_count": int(test_victims.numel()),
                "attack_family": plan.family,
                "trigger_motif": plan.motif,
                "attack_diagnostics": attack_diagnostics,
                "attack_validity": attack_validity,
                "factorial_behavior": behavior,
                "clean_label_detection": all_detection_metrics,
                "permutation_tests": all_permutation_tests,
                "raw_feature_top": {
                    "contextual": extra["raw_feature_top"],
                    "generic": generic_extra["raw_feature_top"],
                },
                "attack_mode": attack_mode,
                "generic_repair_parameters": summary["generic_repair_parameters"],
                "trigger_size": int(cfg.attack.trigger_size),
                "contextual_pair_weight": float(cfg.attack.contextual_pair_weight),
                "generic_consistency_weight": float(cfg.attack.generic_consistency_weight),
                "initialization_sha256": initialization_sha256,
                "shared_training_seed": shared_seed,
                "repeat_control": repeat_control,
                "graph_checks": graph_checks,
            }
        )
        save_json(summary, run_dir / "summary.json")
        make_summary_markdown(summary, run_dir / "SUMMARY.md")
        main_score_name = (
            "cl_generic_spectral_hybrid"
            if attack_mode == "generic"
            else "cl_did_spectral_hybrid"
        )
        main = all_detection_metrics.get(main_score_name, {})
        log(
            f"[Done:v0.6.5] validity={attack_validity['status']} "
            f"full_ASR={full_asr:.4f} binding_gap={binding_gap:.4f} "
            f"spectral_AUROC={main.get('auroc', float('nan')):.4f}",
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
