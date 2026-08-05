# Validation — GSDD-Bench v1.1.0

已完成：

- 全部 Python 文件 `compileall`
- 双侧条件校准合成数据测试
- Max/Fisher/Cauchy 融合有限值和排序测试
- 四档预算软权重范围、尾部衰减和非尾部权重保持测试
- DShield v1.0 compatibility patch 后继续应用 v1.1 weighted-training patch
- v1.1 DShield patch 重复执行幂等性测试
- patched `main.py` 与 `models/GCN.py` 语法编译
- `--gsdd_train_weight_override` 长度、有限值、正值守卫
- GCN 加权交叉熵使用逐节点 unreduced loss 并按权重归一化
- Stage 1 UTF-16LE 日志解析回归
- 聚合器 CSV/Markdown/ZIP 输出静态测试

构建环境没有 NVIDIA GPU，因此没有执行正式 PubMed CUDA 训练。正式结果以用户本机 RTX 5060 Laptop GPU 运行结果为准。包内所有正式入口均在 CUDA 不可用时直接终止。
