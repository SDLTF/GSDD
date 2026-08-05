from __future__ import annotations
import importlib, json, platform, sys
mods = ["torch", "torch_geometric", "ogb", "numpy", "scipy", "pandas", "sklearn", "numexpr", "umap", "yaml"]
out = {"python": sys.version, "platform": platform.platform(), "modules": {}}
for name in mods:
    try:
        module = importlib.import_module(name)
        out["modules"][name] = {"ok": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        out["modules"][name] = {"ok": False, "error": repr(exc)}
print(json.dumps(out, indent=2, ensure_ascii=False))
failed = [k for k,v in out["modules"].items() if not v["ok"]]
if failed:
    raise SystemExit("Environment imports failed: " + ", ".join(failed))
