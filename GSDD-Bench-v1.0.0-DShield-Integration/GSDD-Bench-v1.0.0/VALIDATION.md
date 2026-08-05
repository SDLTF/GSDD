# Validation

构建时完成：

- 所有本包 Python 文件通过 `compileall`
- `patch_dshield_py313.py` 在 DShield 官方 raw `main.py` 快照与 `heuristic_selection.py` 兼容性夹具上通过
- 补丁重复执行保持幂等
- 补丁后的 `main.py` 可被 Python AST 解析
- 标准 artifact 导出/载入模块通过合成张量测试
- GSDD official-artifact runner 的训练、频带计算、指标与训练节点过滤逻辑通过小型合成图测试
- PowerShell 脚本通过静态括号与关键参数检查

未在构建环境完成：

- 正式 CPython 3.13 + CUDA wheel 安装
- DShield 官方仓库完整 clone
- Cora/PubMed 正式 GPU 训练

这些步骤由用户机器上的 setup、bootstrap 和 smoke 脚本完成。失败时不会静默回退 CPU。
