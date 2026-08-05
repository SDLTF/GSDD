from __future__ import annotations
import json, sys
import torch

info = {
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "torch_cuda": torch.version.cuda,
    "device_count": torch.cuda.device_count(),
}
if torch.cuda.is_available():
    info["device_name"] = torch.cuda.get_device_name(0)
    info["capability"] = list(torch.cuda.get_device_capability(0))
print(json.dumps(info, indent=2, ensure_ascii=False))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required. This package never falls back to CPU.")
# Execute a real kernel rather than trusting only driver discovery.
x = torch.randn(256, 256, device="cuda")
y = x @ x
if not torch.isfinite(y).all():
    raise SystemExit("CUDA kernel smoke test produced non-finite values")
print("CUDA kernel smoke test: PASS")
