# GSDD-v0.6.2.1 hotfix

This hotfix repairs the DPGBA mixed-prototype generator dimension mismatch in Round 6.2.

## Root cause

`_initial_generator()` initialized `PrototypeMixtureGenerator` with the number of target-class Cora training prototypes (20), while Round 6.2 subsequently replaced the bank with 32 mixed target/background prototypes. The generated mixture weights therefore had a prototype axis of 20, but the bank had a prototype axis of 32.

## Fix

- Rebuild the DPGBA `PrototypeMixtureGenerator` after constructing the mixed prototype bank, using the bank's actual row count.
- Add explicit shape and prototype-count validation before `einsum`.
- Verified with a 32-prototype regression test producing output shape `(8, 3, 1433)`.

## Resume without rerunning UGBA pilots

After applying the hotfix, run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_attack_validity_repair_round2_resume_dpgba.ps1
```
