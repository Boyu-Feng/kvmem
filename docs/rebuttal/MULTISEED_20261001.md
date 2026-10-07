# 三 seed 扩展实验（已完成）

本文件记录 2026-10-01 启动的扩展实验。此前 [seed233 结果快照](../../reports/rebuttal_20261001_053953_UTC/RESULTS.md)保留为独立的静态结果；本轮扩展结果另行归档。

截至 **2026-10-07 14:55 北京时间**，[完整结果快照](../../reports/multiseed_20261007_065533_UTC/RESULTS.md)已核对并归档 **72/72 组、36,000 条逐题结果**：SideQuest 6/6、ThinkKV 仅淘汰 18/18、两个新 seed 的固定预算实验各 24/24。[机器可读汇总](../../reports/multiseed_20261007_065533_UTC/summary.json)、[逐题分数](../../reports/multiseed_20261007_065533_UTC/per_question/)和[源文件哈希](../../reports/multiseed_20261007_065533_UTC/source_bindings.json)均已保存；原始运行目录保持 Git 忽略。[10 月 2 日的 36/72 中间快照](../../reports/multiseed_20261002_1308_UTC/RESULTS.md)仍保留。

| 方法 | 数据集 | seed | 预算 | 每组主实验 |
|---|---|---|---|---:|
| H2O-step、TOVA-step、FlowKV-style、LazyEviction | HotpotQA、2Wiki、MuSiQue | 新增 42、3407；233 已完成 | 20%、50% | 500 题 |
| SideQuest 未微调适配 | HotpotQA、2Wiki、MuSiQue | 新增 42、3407；233 已完成 | 方法自身的自适应预算 | 500 题 |
| ThinKV/ThinkKV **仅淘汰适配** | HotpotQA、2Wiki、MuSiQue | 233、42、3407 | 20%、50% 名义轨迹 token 预算 | 500 题 |

新增队列总计 72 组主实验、36,000 次题目评测；每组先用抽样后第 501 题起的独立 smoke 验证多步推理和真实淘汰。seed 决定题目抽样顺序，**不同 seed 的 500 题集合不保证相同**；贪心解码本身没有重复抽样。每组最多 7 个 agent 步骤。

用户指定的 ThinKV 版本**不量化**：prompt 受保护；保留的 K/V 均维持模型原生 BF16；当预算触发时，按原始 token 位置划分的 128-token 已完成片段从旧到新逐级减半，最少保留 4 个 token。代表 token 通过最后一层 post-RoPE key 的 K-means medoid 选择，所有层共享所选 token 索引。当前正在处理的片段不能淘汰，因此 20%/50% 是名义目标，实际终态保留率和字节数须由逐题 audit 核验。这是论文附录 E.10 的**仅淘汰 QA 适配消融**，不是论文完整的量化加淘汰实现，也没有专门的 CT/PagedAttention kernel。此前 4-bit 候选试跑保留在独立目录，不纳入主结果或三 seed 汇总。

本机运行目录为 `results/rebuttal_multiseed_20261001/`，Git 忽略原始 checkpoint。查看实时状态：

```bash
python scripts/rebuttal_multiseed_status.py
```

完整快照的导出程序已对 72 组逐一核对 `complete` 状态、500 个唯一题目 ID、checkpoint 与最终结果的逐题 EM/F1，并从逐题结果重算汇总分数。机制和效率结论仍须按各方法的审计与测量口径单独判断。对于 ThinkKV，仅淘汰和其余方法可能有不同的实际 KV 字节、解码器和峰值暂存；对比时须同时报告 EM/F1、实际保留 token、KV 字节和时间，不能把同一名义比例直接当作等显存对照。
