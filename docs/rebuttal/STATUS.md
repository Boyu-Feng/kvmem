# Rebuttal 当前进展

快照时间：**2026-09-30 05:16:36–05:16:39 UTC（北京时间 13:16）**。这是本次 GitHub 发布的固定时间截面；线上队列继续运行。

**已完成 20/27 组主实验，保存 10,206/13,500 次题目评测。** 总数为 24 组固定预算补实验和 3 组 SideQuest 适配；不含 smoke、20 题 pilot、ThinKV 候选或效率补测。

- **[完整结果表与进度快照](../../reports/rebuttal_20260930_051636_UTC/README.md)**
- [机器可读汇总](../../reports/rebuttal_20260930_051636_UTC/summary.json)、[逐题指标](../../reports/rebuttal_20260930_051636_UTC/per_question/)、[源文件 SHA256](../../reports/rebuttal_20260930_051636_UTC/source_bindings.json)
- [上一份快照（9 月 29 日）](../../reports/rebuttal_20260929_030950_UTC/README.md)
- [下一步及验收条件](NEXT_STEPS.md)、[代码与复现](REPRODUCING.md)、[发布验证](VALIDATION.md)

## 已完成和正在运行

| 实验范围 | 进展 | 当前情况 |
|---|---|---|
| HotpotQA 固定预算 | 8/8 组，4,000 次评测 | FlowKV-style、LazyEviction、H2O-step、TOVA-step，各 20%/50% |
| 2Wiki 固定预算 | 8/8 组，4,000 次评测 | 四种方法各 20%/50% 全部完成 |
| MuSiQue 固定预算 | 3/8 组，1,706/4,000 次评测 | FlowKV-style20、H2O-step20/50 完成；FlowKV-style50 为 39/500；TOVA-step20 为 167/500 |
| SideQuest 未微调适配 | 1/3 组，500/1,500 次评测 | HotpotQA 完成；另两集等待 `multidataset_baselines` 整条队列结束 |
| SideQuest 配对 pilot | 20/20 题完成 | 独立于 500 题主实验；有同 decoder FullKV 对照 |
| ThinKV / ThinkKV | 校准 20/20，主实验 0/6 | 三类 thought 预检没有共同三峰层；普通 LLM 单类分支尚未实现 |
| 原范围效率补测 | 0/7，失败 | FullKV 阶段检测到同卡其他计算进程，未产生有效效率结论 |

已完成结果均重新从逐题分数计算，并核对 checkpoint、最终 results 和登记 ID；运行中分数只保存在机器可读快照中，标为 `partial_not_ranked`。

HotpotQA 的 LazyEviction50 为 **EM 23.60 / F1 33.59**；SideQuest 为 **EM 24.00 / F1 33.52**。它们的预算和 decoder 不同，不能据此作等预算排名。2Wiki 的 LazyEviction20/50 分别为 **EM 13.40 / F1 19.84** 和 **EM 16.60 / F1 22.81**。MuSiQue 的 FlowKV-style20 为 **EM 1.60 / F1 2.72**，H2O-step20/50 分别为 **EM 1.20 / F1 2.44** 和 **EM 2.60 / F1 7.75**；这些完成组均为 N=500。完整方法表见上方快照。

## 本轮需要写清楚的结论边界

1. **LazyEviction 的 token 比例不是显存比例。** 当前实现将原生 4 个 KV 头展开到 28 个 query heads，同等缓存 token 数下 K/V payload 为原生 GQA 的 7 倍；还需计算状态、副本与临时工作区。实际字节和独占时间补测列为优先事项。
2. **ThinKV 当前是三类校准未通过。** [论文附录 E.10](https://arxiv.org/html/2510.01290v2#A5.SS10) 给普通 LLM 的单类 thought / 统一 4-bit / 预算触发淘汰分支不依赖三峰 gate；后续应先实现并验证此分支。当前没有可汇报的 ThinKV 压缩结果。
3. **SideQuest 不是官方训练权重复现。** 本轮是独立解码器上的未微调适配。500 题中辅助调用 308 次、无效命令 68 次，151 题实际删除，共删除 38,183 tokens。[500 题保留率汇总](../../reports/rebuttal_20260930_051636_UTC/evidence/sidequest_adaptive_v1/hotpotqa/main/sidequest_untrained/RETENTION_SUMMARY.json)已完成：排除初始 prompt 的终态轨迹 KV 保留率，逐题等权平均为 **90.05%**，合并 token 计数为 **86.62%**；已记录步末的先题内再题间平均为 **97.04%**。这些指标仅统计主分支，不含辅助缓存，也不代表峰值显存或全程时间加权平均；500 题匹配 FullKV 对照仍是待办。
4. **原论文表与新实验分开。** [原表参考](../../ORIGINAL_RESULTS_REFERENCE.md) 是作者提供表格的转录，原逐题文件及精确环境尚待溯源；不得与本轮 seed233 结果混作配对统计。

## 接下来按此顺序推进

先收齐已排队的 2Wiki/MuSiQue 和 SideQuest；随后恢复独占效率测量并补实际字节对照；再推进 ThinKV 单类分支、SideQuest 分析和同题统计。详细依赖、协议和验收标准见 [后续计划](NEXT_STEPS.md)。

R-KV 保持取消，IntentKV 暂缓；更大模型、新 seed 和历史完整规划中的其他消融属于备选。本次整理没有启动新的 GPU 实验。
