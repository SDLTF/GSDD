from __future__ import annotations

import math

import torch


def sparse_mm(matrix: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(matrix, dense)


def scaled_laplacian_apply(laplacian: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
    return 0.5 * sparse_mm(laplacian, signal)


def one_minus_scaled_laplacian_apply(laplacian: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
    return signal - scaled_laplacian_apply(laplacian, signal)


def bernstein_filter_signal(
    laplacian: torch.Tensor,
    signal: torch.Tensor,
    band: int,
    num_bands: int,
) -> torch.Tensor:
    """Apply Bernstein basis filter B_{band,n}(L/2), n=num_bands-1."""
    if num_bands < 2:
        raise ValueError("num_bands must be at least 2")
    degree = num_bands - 1
    if not 0 <= band <= degree:
        raise ValueError("band index is out of range")

    output = signal
    for _ in range(band):
        output = scaled_laplacian_apply(laplacian, output)
    for _ in range(degree - band):
        output = one_minus_scaled_laplacian_apply(laplacian, output)
    return float(math.comb(degree, band)) * output


@torch.no_grad()
def band_energies(
    laplacian: torch.Tensor,
    signal: torch.Tensor,
    num_bands: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw row-wise energy and normalized band distribution [N, B]."""
    energies: list[torch.Tensor] = []
    for band in range(num_bands):
        filtered = bernstein_filter_signal(laplacian, signal, band, num_bands)
        energy = filtered.square().sum(dim=1)
        energies.append(energy)
    raw = torch.stack(energies, dim=1)
    distribution = (raw + epsilon) / (raw.sum(dim=1, keepdim=True) + num_bands * epsilon)
    return raw, distribution


def js_divergence(p: torch.Tensor, q: torch.Tensor, epsilon: float) -> torch.Tensor:
    p = p.clamp_min(epsilon)
    q = q.clamp_min(epsilon)
    p = p / p.sum(dim=1, keepdim=True)
    q = q / q.sum(dim=1, keepdim=True)
    midpoint = 0.5 * (p + q)
    kl_pm = (p * (p.log() - midpoint.log())).sum(dim=1)
    kl_qm = (q * (q.log() - midpoint.log())).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


@torch.no_grad()
def estimate_local_spectral_moments(
    laplacian: torch.Tensor,
    orders: list[int],
    probes: int,
    seed: int,
) -> torch.Tensor:
    """Hutchinson estimate of diag(L^k) for each requested order."""
    if not orders:
        raise ValueError("At least one moment order is required")
    if min(orders) < 1:
        raise ValueError("Moment orders must be positive")
    num_nodes = laplacian.size(0)
    generator = torch.Generator(device=laplacian.device)
    generator.manual_seed(seed)
    signs = torch.randint(
        low=0,
        high=2,
        size=(num_nodes, probes),
        generator=generator,
        device=laplacian.device,
        dtype=torch.int64,
    ).to(torch.float32)
    signs = signs.mul_(2.0).sub_(1.0)

    requested = set(orders)
    current = signs
    estimates: dict[int, torch.Tensor] = {}
    for order in range(1, max(orders) + 1):
        current = sparse_mm(laplacian, current)
        if order in requested:
            estimate = (signs * current).mean(dim=1)
            estimates[order] = estimate.clamp_min(0.0)
    return torch.stack([estimates[order] for order in orders], dim=1)


def log_band_gain(
    hidden_raw: torch.Tensor,
    input_raw: torch.Tensor,
    hidden_dim: int,
    input_dim: int,
    epsilon: float,
) -> torch.Tensor:
    hidden_per_channel = hidden_raw / max(1, hidden_dim)
    input_per_channel = input_raw / max(1, input_dim)
    return torch.log(hidden_per_channel + epsilon) - torch.log(input_per_channel + epsilon)
