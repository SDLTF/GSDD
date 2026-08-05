from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    name: str = "gsdd_v02"
    seed: int = 1027
    output_root: str = "results"
    device: str = "auto"


@dataclass
class DatasetConfig:
    name: str = "Cora"
    root: str = "data"
    normalize_features: bool = True
    synthetic_nodes: int = 240
    synthetic_features: int = 48
    synthetic_classes: int = 3


@dataclass
class AttackConfig:
    enabled: bool = True
    target_class: int = 0
    selection_method: str = "dirty_label"
    # v0.4 causal ablation: full, label_only, trigger_only, or none.
    ablation_mode: str = "full"
    poison_count: int = 4
    trigger_size: int = 3
    trigger_feature_count: int = 40
    trigger_feature_value: float = 1.0
    stamp_victim_features: bool = True
    test_victim_count: int = 100


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    dropout: float = 0.5
    supervised_epochs: int = 300
    ssl_epochs: int = 300
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    patience: int = 60


@dataclass
class SpectralConfig:
    num_bands: int = 4
    moment_orders: list[int] = field(default_factory=lambda: [2, 3, 4, 5])
    hutchinson_probes: int = 48
    epsilon: float = 1e-8
    layers: list[int] = field(default_factory=lambda: [1, 2])


@dataclass
class DetectionConfig:
    # Legacy class/degree-conditioned calibration retained as a baseline.
    degree_bins: int = 3
    minimum_group_size: int = 8
    mad_epsilon: float = 1e-6
    topk_fraction: float = 0.10
    score_clip: float = 10.0

    # New label-free calibration. Keeping one global degree bin is deliberate:
    # the previous Cora run showed that fine degree stratification could dilute
    # the very strong low-frequency trigger signal.
    global_degree_bins: int = 1

    # Iterative calibration: first use label-free anomaly scores, then trim the
    # most suspicious fraction before recomputing class-conditional medians.
    trim_fraction: float = 0.25
    trim_minimum_keep: int = 6

    # Diagnostic-only report. Ground-truth poison labels are used only to tell
    # us whether a raw coordinate contains signal and in which direction.
    raw_feature_topk: int = 20


@dataclass
class PairedConfig:
    # Reset the RNG to this seed offset before training every paired model.
    # Models also share exactly the same initial state_dict.
    training_seed_offset: int = 10000
    repeat_full_control: bool = True
    permutation_repeats: int = 2000


@dataclass
class ReproducibilityConfig:
    enabled: bool = False
    deterministic_algorithms: bool = True
    # CUDA sparse kernels may only support warning mode on some PyTorch builds.
    warn_only: bool = True
    allow_tf32: bool = False
    matmul_precision: str = "highest"
    cublas_workspace_config: str = ":4096:8"
    replicas: int = 2
    operational_topk_fraction: float = 0.05
    spearman_min: float = 0.95
    pearson_min: float = 0.95
    topk_overlap_min: float = 0.80
    auroc_delta_max: float = 0.02
    auprc_delta_max: float = 0.05


@dataclass
class OutputConfig:
    save_models: bool = True
    make_plots: bool = True
    verbose: bool = True


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    paired: PairedConfig = field(default_factory=PairedConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, values: dict[str, Any] | None) -> Any:
    if not values:
        return instance
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown configuration key: {type(instance).__name__}.{key}")
        setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    cfg = Config()
    _merge_dataclass(cfg.experiment, raw.get("experiment"))
    _merge_dataclass(cfg.dataset, raw.get("dataset"))
    _merge_dataclass(cfg.attack, raw.get("attack"))
    _merge_dataclass(cfg.model, raw.get("model"))
    _merge_dataclass(cfg.spectral, raw.get("spectral"))
    _merge_dataclass(cfg.detection, raw.get("detection"))
    _merge_dataclass(cfg.paired, raw.get("paired"))
    _merge_dataclass(cfg.reproducibility, raw.get("reproducibility"))
    _merge_dataclass(cfg.output, raw.get("output"))
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, allow_unicode=True, sort_keys=False)
