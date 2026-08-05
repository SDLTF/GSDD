# GSDD-Bench v1.1.0 — GSDD Defense Optimization

本版本不重新训练攻击。它直接复用 Stage 1 已经生成的 PubMed 官方攻击产物：

```text
artifacts/official_attacks/Pubmed_SBA_seed1027
artifacts/official_attacks/Pubmed_UGBA_seed1027
artifacts/official_attacks/Pubmed_GCBA_seed1027
```

研究目标从“接入官方攻击”转为“优化 GSDD 的检测分数和防御训练”。

## 本轮整合的四类改进

### 1. 双侧稳健校准

旧版默认认为谱差异越大越可疑，但 Stage 1 中 UGBA 的投毒节点在部分谱特征上反而更小。v1.1.0 对每个谱特征计算双侧稳健分数：

$$
Z_j(v)=\frac{S_j(v)-\operatorname{median}(S_j\mid y_v,\deg v)}{\operatorname{MAD}(S_j\mid y_v,\deg v)+\varepsilon}
$$

$$
A_j(v)=|Z_j(v)|
$$

参考组优先使用“观测标签 + 度数分位组”；样本不足时依次回退到标签组和全局训练节点。

### 2. 无监督多指标融合

所有融合方法只使用训练图和谱分数，不使用投毒标签：

- `robust_max`：六个双侧稳健分数的最大值
- `fisher`：六个双侧 $p$ 值的 Fisher 合并统计量
- `cauchy`：相关检验下更稳健的 Cauchy 合并排序

投毒标签仅用于最终计算 AUROC、AUPRC 和 Recall@Budget，不参与节点排序。

### 3. 多过滤预算

统一测试四档审计预算：

```text
0.5%
1%
2%
5%
```

每个融合方法都生成四份硬过滤训练索引。

### 4. 软降权训练

每个融合方法和预算同时生成逐节点权重。预算之外的训练节点权重保持 1；预算尾部内平滑衰减，最异常节点权重约为 $\exp(-6)$。

DShield 官方 GCN 被最小化补丁扩展为逐节点加权交叉熵：

$$
\mathcal L=\frac{\sum_{v\in V_{\mathrm{train}}}w_v\operatorname{CE}(f(v),y_v)}{\sum_{v\in V_{\mathrm{train}}}w_v}
$$

这不是重复采样近似，硬过滤与软降权使用相同模型和训练轮数。

## 实验矩阵

- 数据集：PubMed
- 官方攻击：SBA、UGBA、GCBA
- Victim model：GCN
- Pilot seed：1027
- 融合：3 种
- 预算：4 档
- 训练策略：硬过滤、软降权

总计：

$$
3\times4\times2\times3=72
$$

个优化防御评估。攻击和谱特征分别只生成/计算一次。

## 从现有 v1.0.4 项目升级

将补丁包内容复制到当前项目根目录并覆盖，然后执行：

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\apply_v110_hotfix.ps1
```

无需：

- 删除 `.venv`
- 重新安装 PyTorch
- 重新克隆 DShield
- 重新生成官方攻击
- 重新运行 DShield baseline

## 建议先跑 smoke test

需要已经存在 `Cora_SBA_seed1027` artifact：

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_smoke_v110.ps1
```

该 smoke test 只验证：

- 双侧校准和融合
- 硬过滤文件
- 软权重文件
- DShield 官方 GCN 加权训练入口

Cora smoke 数值不进入正式结果。

## 运行完整 PubMed 优化实验

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_next_experiment.ps1
```

等价命令：

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_pubmed_gsdd_optimization.ps1
```

脚本可断点续跑：已有的 detection summary 和 `official_metrics.json` 会被跳过。强制重跑全部变体：

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_pubmed_gsdd_optimization.ps1 -Force
```

## 输出

检测阶段：

```text
results_optimization/<RunId>/Detection/
├── node_scores_optimized.csv
├── optimization_summary.json
├── OPTIMIZATION_DETECTION_SUMMARY.md
├── hard_indices/
└── soft_weights/
```

每个硬过滤/软降权变体都会生成官方 DShield 评估日志和指标 JSON。

最终优先上传：

```text
artifacts\gsdd_v110_optimization_aggregate.zip
```

需要排查单个变体时，再上传：

```text
results_optimization\<RunId>\
```

## 结果选择规则

v1.1.0 是 seed 1027 的优化 pilot。聚合器会在平均干净准确率下降不超过 0.02 的方案中，优先选择平均 ASR 最低、最坏攻击 ASR 更低的固定协议。

该协议只能用于确定下一轮候选。正式结论必须：

1. 冻结融合方法、预算和训练策略
2. 在 seeds 2026、3407 上复现
3. 与 DShield 使用相同官方攻击 artifact 比较
4. 通过后再进入 OGBN-Arxiv

## 环境约束

- 标准 CPython 3.13，不支持 3.13t
- 正式运行强制 CUDA，不允许 CPU 回退
- 当前软降权补丁仅支持 DShield 官方 `model=GCN`
- 官方攻击代码和参数保持不变
