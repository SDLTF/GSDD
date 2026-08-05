from __future__ import annotations
import json, platform, sys, sysconfig

version_ok = sys.version_info[:2] == (3, 13)
free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED") or 0)
info = {
    "python": sys.version,
    "implementation": platform.python_implementation(),
    "executable": sys.executable,
    "version_ok": version_ok,
    "free_threaded": free_threaded,
}
print(json.dumps(info, indent=2, ensure_ascii=False))
if not version_ok:
    raise SystemExit("CPython 3.13 is required")
if platform.python_implementation() != "CPython":
    raise SystemExit("Standard CPython is required")
if free_threaded:
    raise SystemExit("CPython 3.13t/free-threaded is not supported; install standard CPython 3.13")
