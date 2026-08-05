from __future__ import annotations

import json
import sys

import torch


def main() -> int:
    payload = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        payload["gpu_name"] = torch.cuda.get_device_name(0)
        payload["capability"] = torch.cuda.get_device_capability(0)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not torch.cuda.is_available():
        print(
            "GSDD-v0.6 requires a CUDA-enabled PyTorch build. CPU fallback is disabled.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
