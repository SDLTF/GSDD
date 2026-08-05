# GSDD-Bench v1.0.4 hotfix

## Fixed failure

Node-injection attacks may return a poisoned feature matrix with more rows than the original label vector. For the Cora SBA smoke run:

- `poison_x.shape[0] = 2714`
- `poison_y.shape[0] = 2708`
- injected trigger nodes = 6

GSDD correctly created masks for 2714 nodes, but indexing the 2708-row label tensor with those masks raised a shape mismatch.

## Repair

- Pad `poison_y` to the poisoned graph size with label `-1`
- Never include padded/injected nodes in supervised train, validation, clean-test, attack-test, or attach indices
- Validate all node indices and poisoned edges against `poison_x`
- Upgrade new exported artifacts to format version 2
- Repair existing format-version-1 artifacts in place while preserving `artifact.pt.pre_v104.bak`

## Resume the interrupted Cora SBA smoke run

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\apply_v104_hotfix.ps1
powershell.exe -ExecutionPolicy Bypass -File .\scripts\resume_cora_sba_v104.ps1
```

The official SBA attack and DShield baseline do not need to be regenerated.
