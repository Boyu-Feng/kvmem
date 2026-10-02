# KVMem / StepKV

面向多轮工具调用推理的 KV cache 压缩实验。原 StepKV 代码、论文草稿与历史说明保留在原路径；本轮 rebuttal 使用恢复后的原评测引擎，新增 baseline 与原论文表格分开报告。

## 当前实验入口

- **[三 seed 扩展实验最新已完成结果](reports/multiseed_20261002_1308_UTC/RESULTS.md)**：36/72 组完成，18,000 条逐题评测；其余组继续运行，阶段分数未列入。
- **[实验结果总表：数据集 × 方法](reports/rebuttal_20261001_053953_UTC/RESULTS.md)**：27/27 组完成，13,500 条逐题评测，包含 20%/50% 的 EM/F1。
- **[Rebuttal 当前进展与结果](docs/rebuttal/STATUS.md)**：上传时的静态快照、完成情况及已知阻塞。
- **[三 seed 扩展实验](docs/rebuttal/MULTISEED_20261001.md)**：新增 seed 42/3407 和 ThinKV 仅淘汰适配的运行范围、状态入口。
- **[后续执行计划](docs/rebuttal/NEXT_STEPS.md)**：主实验完成记录、待修复的效率测量、ThinKV 单类分支和统计核验。
- **[代码与复现说明](docs/rebuttal/REPRODUCING.md)**：冻结适配代码、环境、资产版本和启动方式。
- **[协议边界](docs/rebuttal/PROTOCOL.md)**：预算、模型、数据划分、计时与对照范围。
- [原论文表格参考](ORIGINAL_RESULTS_REFERENCE.md)：作者提供表格的转录，尚未从原始逐题结果重新计算。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `models/`、`kv_cache/`、`token_tracker.py` | 原模型封装与 KV 管理/评分实现 |
| `run_rebuttal_legacy*.py`、`run_rebuttal_multidataset.py` | 本轮固定协议的评测入口 |
| `scripts/` | 队列、SideQuest、ThinkKV 校准、效率测量、汇总与快照导出 |
| `rebuttal/adapters/` | FlowKV-style、LazyEviction、H2O-step、TOVA-step 的冻结适配增量、测试、来源与许可证 |
| `config/` | 本轮环境记录和模型/数据 revision |
| `reports/` | 可提交的结果快照、逐题指标、源文件哈希、校准和失败记录 |
| `docs/rebuttal/` | 当前状态、协议、后续计划与复现说明 |
| `tests/` | CPU 回归与审计测试 |
| `results/`、`.runtime/` | 本地完整运行数据与环境，Git 忽略 |

模型权重、Wikipedia 索引、数据集正文、完整运行日志不随 Git 分发。`reports/` 提供本轮逐题分数和证据绑定；`results/` 的历史路径不是 clone 后自动存在的结果。

## 查看与导出

在拥有本地运行产物的机器上：

```bash
.runtime/rebuttal/bin/python scripts/rebuttal_status.py
python3 scripts/export_rebuttal_snapshot.py --output reports/rebuttal_YYYYMMDD_HHMMSS_UTC
```

导出器只读取运行目录，输出必须使用新目录；不会重跑任务或覆盖旧快照。新增快照后更新 `docs/rebuttal/STATUS.md` 中的链接。

最初 27 组主实验固定模型为 Qwen2.5-7B-Instruct、抽样 seed233；扩展实验新增 seed42、3407。固定预算补实验使用名义 20%/50% 轨迹 token 保留档位；SideQuest 使用独立解码器和自适应预算。跨协议结果不能直接当作等显存或等计算量的算法排名。

根目录 `REBUTTAL_*.md` 保留各阶段详细记录，其中历史“正在运行”状态不代表当前进度；以带时间戳的结果快照为准。
