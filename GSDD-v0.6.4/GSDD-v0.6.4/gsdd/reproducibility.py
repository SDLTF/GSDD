from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

import torch


def configure_reproducibility(config: Any) -> dict[str, Any]:
    """Configure PyTorch reproducibility without silently changing the device.

    `CUBLAS_WORKSPACE_CONFIG` is best set before Python starts. The PowerShell
    launchers do that; setting a default here is a defensive fallback.
    """
    if not bool(getattr(config, "enabled", False)):
        return {
            "enabled": False,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        }

    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        str(getattr(config, "cublas_workspace_config", ":4096:8")),
    )
    deterministic = bool(getattr(config, "deterministic_algorithms", True))
    warn_only = bool(getattr(config, "warn_only", True))
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = bool(getattr(config, "allow_tf32", False))
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(getattr(config, "allow_tf32", False))
    try:
        torch.set_float32_matmul_precision(str(getattr(config, "matmul_precision", "highest")))
    except Exception:
        pass

    details = asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config.__dict__)
    details.update(
        {
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only_enabled": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cublas_workspace_config_runtime": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
            "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
        }
    )
    return details
