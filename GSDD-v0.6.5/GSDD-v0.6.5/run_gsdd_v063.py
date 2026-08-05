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

from gsdd.attack_families_v062 import build_attack_plan_v062
from gsdd.clean_label_factorial import (
    FACTORIAL_MODEL_NAMES,
    FACTORIAL_TRIGGER_NAMES,
    build_test_factorial_graphs,
    build_training_factorial_graphs,
    compute_clean_label_factorial_diagnostics,
)
from gsdd.config import Config, load_config, save_config
from gsdd.data import GraphData, load_dataset
from gsdd.graph_ops import build_normalized_adjacency, build_normalized_laplacian
from gsdd.models import SupervisedGCN
from gsdd.reproducibility import configure_reproducibility
from gsdd.train import accuracy, train_supervised
from gsdd.utils import environment_info, log, make_run_dir, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GSDD-v0.6.3 clean-label model x trigger factorial audit"
    )
    parser.add_argument("--config", default="configs/cora_clean_label_factorial.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--poison-count", type=int, default=None)
    parser.add_argument("--target-class", type=int, default=None)
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
    cfg.attack.selection_method = "clean_label"
    cfg.attack.family = "dpgba_style_binding_aware"
    cfg._allow_cpu_v063 = bool(args.allow_cpu)  # type: ignore[attr-defined]
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
    lines = [
        "# GSDD-v0.6.3 Clean-label Factorial Audit",
        "",
        f"- Status: `{summary.get('status')}`",
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
            "## Clean-label validity",
            "",
            f"- Status: `{validity.get('status')}`",
            f"- Reasons: `{validity.get('reasons', [])}`",
            f"- Full ASR $\\operatorname{{ASR}}(M_p,T)$: `{validity.get('full_asr', float('nan')):.4f}`",
            f"- Maximum control ASR: `{validity.get('control_asr_max', float('nan')):.4f}`",
            f"- Clean-label binding gap: `{validity.get('binding_gap', float('nan')):.4f}`",
            "",
            "The binding gap is",
            "",
            "$$",
            r"\Delta_{\mathrm{CL}}=\operatorname{ASR}(M_p,T)-\max\{\operatorname{ASR}(M_c,T),\operatorname{ASR}(M_p,T_{\mathrm{shuffle}}),\operatorname{ASR}(M_p,\varnothing),\operatorname{ASR}(M_c,\varnothing)\}",
            "$$",
            "",
            "## Clean-label spectral interaction",
            "",
            "$$",
            r"D_{\mathrm{CL}}=[S(M_p,T)-S(M_c,T)]-[S(M_p,T_{\mathrm{shuffle}})-S(M_c,T_{\mathrm{shuffle}})]",
            "$$",
            "",
            "| Score | AUROC | AUPRC | FPR@95TPR | Permutation $p$ |",
            "|---|---:|---:|---:|---:|",
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
            "Detection metrics are research evidence only when the clean-label attack passes the factorial validity gate. Invalid attacks remain useful for debugging, but must not be counted as successful defense cases.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config, source_config: Path) -> Path:
    allow_cpu = bool(getattr(cfg, "_allow_cpu_v063", False))
    if not allow_cpu and not cfg.experiment.device.lower().startswith("cuda"):
        raise RuntimeError("GSDD-v0.6.3 formal experiments are CUDA-only")
    if cfg.experiment.device.lower().startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "GSDD-v0.6.3 requires CUDA for formal experiments, but "
            "torch.cuda.is_available() is False"
        )
    if cfg.attack.selection_method != "clean_label":
        raise ValueError("GSDD-v0.6.3 only audits clean-label attacks")

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
        "protocol": "clean_label_model_by_trigger_factorial",
    }
    save_json(summary, run_dir / "summary.json")
    log(f"[GSDD-v0.6.3] run_dir={run_dir}", cfg.output.verbose)

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
        plan = build_attack_plan_v062(clean_graph, cfg, device)
        if plan.family != "dpgba_style_binding_aware":
            raise AssertionError("v0.6.3 formal protocol expects DPGBA-style plan")
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
            run_dir / "attack_plan_v063.pt",
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

        full_asr = behavior["poison"]["matched"]["asr"]
        baseline_asr = behavior["clean"]["none"]["asr"]
        clean_trigger_asr = behavior["clean"]["matched"]["asr"]
        poison_none_asr = behavior["poison"]["none"]["asr"]
        poison_shuffled_asr = behavior["poison"]["shuffled"]["asr"]
        clean_shuffled_asr = behavior["clean"]["shuffled"]["asr"]
        control_asr_max = max(
            baseline_asr,
            clean_trigger_asr,
            poison_none_asr,
            poison_shuffled_asr,
            clean_shuffled_asr,
        )
        binding_reference = max(
            baseline_asr, clean_trigger_asr, poison_none_asr, poison_shuffled_asr
        )
        binding_gap = full_asr - binding_reference
        validity_reasons: list[str] = []
        if baseline_asr > cfg.attack.clean_label_baseline_asr_max:
            validity_reasons.append("clean_no_trigger_asr_above_maximum")
        if clean_trigger_asr > cfg.attack.clean_label_clean_trigger_asr_max:
            validity_reasons.append("clean_model_matched_trigger_asr_above_maximum")
        if poison_none_asr > cfg.attack.clean_label_poison_no_trigger_asr_max:
            validity_reasons.append("poison_model_no_trigger_asr_above_maximum")
        if poison_shuffled_asr > cfg.attack.clean_label_shuffled_asr_max:
            validity_reasons.append("poison_model_shuffled_trigger_asr_above_maximum")
        if full_asr < cfg.attack.clean_label_full_asr_min:
            validity_reasons.append("poison_model_matched_trigger_asr_below_minimum")
        if binding_gap < cfg.attack.clean_label_binding_gap_min:
            validity_reasons.append("clean_label_binding_gap_below_minimum")
        attack_validity = {
            "is_valid": not validity_reasons,
            "status": "valid_clean_label_attack" if not validity_reasons else "invalid_clean_label_attack",
            "reasons": validity_reasons,
            "full_asr": full_asr,
            "clean_no_trigger_asr": baseline_asr,
            "clean_matched_trigger_asr": clean_trigger_asr,
            "clean_shuffled_trigger_asr": clean_shuffled_asr,
            "poison_no_trigger_asr": poison_none_asr,
            "poison_shuffled_trigger_asr": poison_shuffled_asr,
            "control_asr_max": control_asr_max,
            "binding_reference": binding_reference,
            "binding_gap": binding_gap,
            "thresholds": {
                "clean_no_trigger_asr_max": cfg.attack.clean_label_baseline_asr_max,
                "clean_matched_trigger_asr_max": cfg.attack.clean_label_clean_trigger_asr_max,
                "poison_no_trigger_asr_max": cfg.attack.clean_label_poison_no_trigger_asr_max,
                "poison_shuffled_trigger_asr_max": cfg.attack.clean_label_shuffled_asr_max,
                "full_asr_min": cfg.attack.clean_label_full_asr_min,
                "binding_gap_min": cfg.attack.clean_label_binding_gap_min,
            },
        }
        save_json(attack_validity, run_dir / "clean_label_attack_validity.json")
        log(
            f"[CleanLabelValidity] status={attack_validity['status']} "
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
                "clean_label_detection": detection_metrics,
                "permutation_tests": extra["permutation_tests"],
                "raw_feature_top": extra["raw_feature_top"],
                "initialization_sha256": initialization_sha256,
                "shared_training_seed": shared_seed,
                "repeat_control": repeat_control,
                "graph_checks": graph_checks,
            }
        )
        save_json(summary, run_dir / "summary.json")
        make_summary_markdown(summary, run_dir / "SUMMARY.md")
        main = detection_metrics.get("cl_did_spectral_hybrid", {})
        log(
            f"[Done:v0.6.3] validity={attack_validity['status']} "
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
