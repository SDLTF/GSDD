from __future__ import annotations

import importlib
import json
import platform
import sys

REQUIRED = ["torch", "numpy", "scipy", "sklearn", "pandas", "matplotlib", "yaml"]


def main() -> int:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "modules": {},
    }
    failed = False
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            result["modules"][name] = {
                "ok": True,
                "version": getattr(module, "__version__", "unknown"),
            }
        except Exception as exc:
            failed = True
            result["modules"][name] = {"ok": False, "error": str(exc)}
    if result["modules"].get("torch", {}).get("ok"):
        import torch

        result["torch_cuda_available"] = torch.cuda.is_available()
        result["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            result["gpu_name"] = torch.cuda.get_device_name(0)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
