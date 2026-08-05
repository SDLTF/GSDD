# Apply v1.0.2

Overwrite `scripts\common.ps1` in the current GSDD-Bench directory.

Your Python 3.13, PyTorch 2.13.0+cu130, RTX 5060 and CUDA kernel test have
already passed. Do not delete `.venv`.

Run the same command again. If the official DShield program still fails,
upload `results\Cora_SBA_seed1027\none.log`; it will now contain the actual
exception rather than only the first `Traceback` line.
