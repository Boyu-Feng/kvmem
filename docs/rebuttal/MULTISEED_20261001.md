# 三 seed 扩展实验（运行中）

本文件记录 2026-10-01 新启动的扩展实验。此前 [seed233 结果快照](../../reports/rebuttal_20261001_053953_UTC/RESULTS.md)保留为已完成的静态结果；本轮运行中的 checkpoint 不计入该快照。

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

只有 `complete` 且题目 ID、checkpoint、最终结果和机制审计一致的组，才能进入下一次可提交的结果快照。对于 ThinKV，仅淘汰和其余方法可能有不同的实际 KV 字节、解码器和峰值暂存；对比时须同时报告 EM/F1、实际保留 token、KV 字节和时间，不能把同一名义比例直接当作等显存对照。
