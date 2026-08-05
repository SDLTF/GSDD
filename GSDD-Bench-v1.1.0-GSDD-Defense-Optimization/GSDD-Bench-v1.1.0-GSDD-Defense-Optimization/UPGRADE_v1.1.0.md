# Upgrade to v1.1.0

适用起点：已经完成 v1.0.4/v1.0.4.1 修复，并且本地已有 PubMed SBA、UGBA、GCBA 官方 artifact 与 Stage 1 baseline 日志。

1. 将升级补丁复制到项目根目录并覆盖同名文件
2. 应用 DShield 加权训练补丁

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\apply_v110_hotfix.ps1
```

3. 可选 smoke test

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_smoke_v110.ps1
```

4. 运行 PubMed 优化矩阵

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

完成后上传：

```text
artifacts\gsdd_v110_optimization_aggregate.zip
```

本升级不会删除或重写现有攻击 artifact、Stage 1 结果与 DShield baseline。
