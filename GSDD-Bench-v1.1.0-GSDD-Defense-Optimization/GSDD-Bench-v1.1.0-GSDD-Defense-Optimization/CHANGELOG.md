# Changelog

## 1.1.0

- 新增按观测标签和度数分位条件化的双侧 median/MAD 校准
- 新增 `robust_max`、`fisher`、`cauchy` 三种无监督谱特征融合
- 新增 0.5%、1%、2%、5% 四档硬过滤预算
- 新增与预算匹配的平滑软降权策略
- 为 DShield 官方 GCN 增加逐节点加权交叉熵入口
- 新增 PubMed SBA/UGBA/GCBA 的 72 变体断点续跑脚本
- 新增 UTF-8/UTF-16LE 自动日志解码，修复 Stage 1 空指标问题
- 新增跨攻击协议排序、单攻击最优变体和检测指标汇总
- 修复 v1.0.4 artifact 修复脚本从 `tools` 目录运行时的项目根路径导入
- `run_next_experiment.ps1` 现指向 GSDD 防御优化阶段，不再重新生成攻击
