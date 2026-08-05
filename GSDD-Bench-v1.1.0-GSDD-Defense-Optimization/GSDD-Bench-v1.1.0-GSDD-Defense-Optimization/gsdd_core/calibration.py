from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import norm, rankdata


@dataclass(frozen=True)
class CalibrationResult:
    abs_z: np.ndarray
    two_sided_p: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    group_size: np.ndarray
    degree_bin: np.ndarray


def _mad(values: np.ndarray, center: float, epsilon: float) -> float:
    raw = float(np.median(np.abs(values - center)))
    # 1.4826 makes MAD comparable with standard deviation under a Gaussian model.
    return max(1.4826 * raw, epsilon)


def _degree_bins(degree: np.ndarray, bins: int) -> np.ndarray:
    if bins < 1:
        raise ValueError("bins must be positive")
    log_degree = np.log1p(np.asarray(degree, dtype=np.float64))
    quantiles = np.quantile(log_degree, np.linspace(0.0, 1.0, bins + 1))
    inner = np.unique(quantiles[1:-1])
    return np.searchsorted(inner, log_degree, side="right").astype(np.int64)


def robust_two_sided_calibration(
    score: np.ndarray,
    labels: np.ndarray,
    degree: np.ndarray,
    *,
    degree_bins: int = 4,
    min_group_size: int = 20,
    epsilon: float = 1e-6,
    z_clip: float = 12.0,
) -> CalibrationResult:
    """Condition a score on observed label and degree, then take both tails.

    The primary reference group is ``(observed label, degree quantile)``.  If
    that group is too small, calibration falls back to the label-only group,
    and finally to all candidate training nodes.  This is unsupervised with
    respect to poison labels.
    """
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    degree = np.asarray(degree, dtype=np.float64).reshape(-1)
    if not (len(score) == len(labels) == len(degree)):
        raise ValueError("score, labels, and degree must have equal length")
    if len(score) < 2:
        raise ValueError("at least two candidate nodes are required")

    bins = _degree_bins(degree, degree_bins)
    centers = np.empty_like(score)
    scales = np.empty_like(score)
    sizes = np.empty(len(score), dtype=np.int64)

    global_center = float(np.median(score))
    global_scale = _mad(score, global_center, epsilon)

    for index in range(len(score)):
        fine = (labels == labels[index]) & (bins == bins[index])
        coarse = labels == labels[index]
        if int(fine.sum()) >= min_group_size:
            mask = fine
        elif int(coarse.sum()) >= min_group_size:
            mask = coarse
        else:
            mask = np.ones(len(score), dtype=bool)

        values = score[mask]
        center = float(np.median(values)) if len(values) else global_center
        scale = _mad(values, center, epsilon) if len(values) else global_scale
        centers[index] = center
        scales[index] = scale
        sizes[index] = int(mask.sum())

    z = np.clip((score - centers) / scales, -z_clip, z_clip)
    abs_z = np.abs(z)
    p = np.maximum(2.0 * norm.sf(abs_z), np.finfo(np.float64).tiny)
    return CalibrationResult(
        abs_z=abs_z,
        two_sided_p=p,
        centers=centers,
        scales=scales,
        group_size=sizes,
        degree_bin=bins,
    )


def fuse_max(abs_z_matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(abs_z_matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("abs_z_matrix must be rank 2")
    return values.max(axis=1)


def fuse_fisher(p_matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(p_matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("p_matrix must be rank 2")
    values = np.clip(values, np.finfo(np.float64).tiny, 1.0)
    return -2.0 * np.log(values).sum(axis=1)


def fuse_cauchy(p_matrix: np.ndarray) -> np.ndarray:
    """Cauchy combination statistic; larger values are more anomalous."""
    values = np.asarray(p_matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("p_matrix must be rank 2")
    values = np.clip(values, 1e-12, 1.0 - 1e-12)
    statistic = np.tan((0.5 - values) * math.pi).mean(axis=1)
    # Rank normalization avoids numerical domination by a single p≈0 value,
    # while preserving the ordering needed by filtering and soft weighting.
    return (rankdata(statistic, method="average") - 0.5) / len(statistic)


def tail_soft_weights(score: np.ndarray, budget: float, strength: float = 6.0) -> np.ndarray:
    """Smoothly down-weight only the top ``budget`` fraction of a score.

    Nodes below the selected tail retain weight 1.  Within the tail, weights
    decay continuously to exp(-strength) for the most anomalous node.
    """
    if not 0.0 < budget < 1.0:
        raise ValueError("budget must lie in (0, 1)")
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    quantile = (rankdata(score, method="average") - 0.5) / len(score)
    start = 1.0 - budget
    tail = np.clip((quantile - start) / budget, 0.0, 1.0)
    return np.exp(-strength * tail ** 2)
