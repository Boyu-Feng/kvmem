# Rebuttal 首批实验运行说明

**2026-09-28 最新状态：作者要求恢复原评测模式。本文件下述 controlled_v2 队列已停止，结果仅保留审计。当前入口、原结果保留方式与新 baseline 队列见 [REBUTTAL_LEGACY_MODE.md](REBUTTAL_LEGACY_MODE.md)。**

## 当前批次

2026-09-28 执行范围：只跑 Qwen、seed=233；大模型 Qwen2.5-32B 暂不启动。新增 FlowKV-style、R-KV、LazyEviction 使用独立队列，详见 [REBUTTAL_EXTENSIONS.md](REBUTTAL_EXTENSIONS.md)。下文描述原 controlled_v2 核心对照队列。

- 运行目录：`results/rebuttal_20260927/controlled_v2/`。
- 入口：`run_rebuttal_schedule.py`；持久队列：`scripts/run_rebuttal_queue.py`。
- 独立 Python 环境：`.runtime/rebuttal/bin/python`；完整依赖版本保存在运行目录的 `pip_freeze.txt`。
- 模型：Qwen2.5-7B-Instruct，模型 revision 和数据 revision 写入每组 `manifest.json`。
- 数据：HotpotQA distractor validation，全文 BM25 检索 5,233,235 篇 Wikipedia，沿用原仓库索引与 agent prompt。
- GPU：本机 GPU 2，A100 PCIe 40GB；未终止其他任务。同卡存在其他进程，因此本轮耗时只用于诊断，不用于论文最终速度结论。

## 队列与审稿意见的对应

先运行每个条件 2 题的 smoke 检查，再执行以下条件，每组 500 题、最多 7 步、每个生成块最多 256 tokens。按作者 2026-09-28 的决定，本批只使用抽样种子 233，不运行 seed=42 或 3407；先跑 20%，再跑 50%。FullKV 只跑一次，供两个预算共用。

| 条件名 | 剪枝时机 | 用途 |
|---|---|---|
| `fullkv` | 不剪枝 | R1/R3/R4：相同增量引擎的未压缩参照 |
| `h2o_token` / `h2o_step` | 每 token / 完整生成块结束 | R1：H2O selector 的时机控制 |
| `tova_token` / `tova_step` | 每 token / 完整生成块结束 | R1：TOVA selector 的时机控制 |
| `stepkv` | 完整生成块结束 | R1/R3/R4/R5：完整 step utility 与 token 分数融合 |
| `stepkv_no_step_score` | 与 StepKV 相同 | R1/R3/R4/R5：beta=0，保留相同 step floor 的 utility 消融 |

这是 13 个主实验任务，共 6,500 个 question-condition evaluations，另有 14 个 smoke evaluations。队列最晚于 10 月 12 日 18:00 UTC 停止启动新任务；已有任务会继续保存检查点。本批完成范围为 seed=233 的两个预算。Smoke 使用 seed=233 排序后的第 501–502 题，不纳入该 seed 的主结果；不根据 smoke 准确率调参。

R2 要求的新任务/更大模型，以及 R1/R2/R5 的 FlowKV-style、LazyEviction 等新 baseline，不在这一批运行中。首批先回答争议最大的时机与 utility 归因问题。完整补实验规划见 `REBUTTAL_PLAN_2026-10-13.md`。

## 必须写进论文的协议边界

2026-09-28 源码核查更正：实际执行改动不止剪枝时机；完整差异及送审溯源限制见 [REBUTTAL_PROTOCOL_AUDIT_2026-09-28.md](REBUTTAL_PROTOCOL_AUDIT_2026-09-28.md)。此前“初始 generation 沿用原程序不剪枝”对旧 H2O/TOVA 不成立。

1. **本批是新增的 estimator-controlled 对照，不是送审表格的复现。** 需要评分的剪枝方法统一使用 full-cache EOS attention probe，H2O/StepKV 聚合最后三层，TOVA 使用最后一层；所有方法使用同一 greedy 循环。本批额外前向开销必须计入，不能用它证明“无需额外 forward”。这里的 H2O/TOVA 是仓库 selector 的适配版本，不宣称为官方实现复现。修改前本地代码快照保存在 `results/rebuttal_20260927/source_snapshot/before_changes.tar`，尚未证明该快照就是生成送审表格的精确版本。
2. step schedule 在后续完整生成块结束时剪枝；token schedule 在后续 observation prefill 后和解码 token 后检查预算。**新引擎所有方法的首轮都不剪枝：这与旧 StepKV 一致，但改变了旧 H2O/TOVA 在首轮生成结束后立即执行预算剪枝的行为。** 新的 h2o_token/tova_token 也受此变化影响。H2O 的近期保护窗口为 32，TOVA/StepKV 为 0；同一 baseline 的 token/step 对照保持窗口一致。因此跨方法差距不能全部归因于 utility，新旧结果差距也不能全部归因于后续剪枝时机。
3. 预算分母是累计 logical non-prompt tokens，保护初始 prompt。候选只包括当前存活 KV 和新写入 KV。H2O 最近保护窗口可能形成预算下限；旧 selector 还存在至少保留一个候选及浮点截断的 1-token 误差，运行时明确容忍不超过 1 token 并记录实际值。
4. StepKV-no-step-score 保留原来的每 step 10% 当前存活 token floor；预算不足时仍按总预算裁剪。它不等价于去掉所有结构机制的 TOVA-step。首批保留原 step utility 实现，包括自身 observation overlap 的现有行为；该问题的修正需要另起标记，不混入本批。
5. 运行时显式使用逻辑位置进行 RoPE，并让最后生成 token 真正写入缓存。Stop marker 涉及的 token 从缓存中移除；共享 BPE token 中位于 marker 之前的有效字符重新编码入缓存，保留 Action 右括号。每次检查各层长度、全局 ID 映射、预算；EOS probe 不修改 live KV。这些修正统一用于所有实验臂。
6. `peak_cache_bytes` 统计 live K+V 张量、包含 prompt 和剪枝前的步内峰值，不包括临时评分张量或原 runner 持有的 KV 副本。`peak_memory_mb` 是同进程 CUDA allocator 峰值减参数量，包含其他推理分配。二者不可混写成同一指标。
7. seed=233 决定题目子集，生成使用原始 logits 的 argmax，不采样，不启用 temperature/top-p/top-k 或 repetition penalty（有效 penalty=1.0）。Qwen 自带 generation_config 中的 penalty=1.05 不应用于本循环；这是与原 `generate()` 路径的另一处明确差异。EOS 沿用模型的 `[151645, 151643]`。本批为单 seed，不报告跨 seed 标准差；按题配对 bootstrap 保留。每题保存 trajectory、预测、EM/F1、预算事件及时间。错误直接报失败，不作为答错样本计入。

## 查看进度与产物

```bash
cat results/rebuttal_20260927/controlled_v2/queue_status.json
tail -n 40 results/rebuttal_20260927/logs/queue_v2.log
tail -n 40 results/rebuttal_20260927/controlled_v2/smoke/fullkv/run.log
```

每个条件目录都有 `manifest.json`、`status.json`、`run.log`、逐题原子保存的 `checkpoint.json`，完成后生成 `results.json`。manifest 固定样本 ID、源代码 SHA256、模型与数据 revision、全部关键设置。运行源码变更时队列停止；配置不匹配时拒绝覆盖旧结果。

预算核对应以逐题 `runtime_audit.cache_measurements` 中的 `logical_tokens`、`target_tokens`、`before_tokens`、`after_tokens` 为准。原 runner 的部分控制台 `[STEP SUMMARY]` 使用旧的局部计数和 selector 字段，在逐 token 路径可能显示不一致的预算；这些控制台派生数不用于本批预算结论。

每个主实验完成后自动更新 `SUMMARY.md` 与 `paired_statistics.json`：在相同题目 ID 上计算 EM/F1 差值、10,000 次配对 bootstrap 的 95% CI，以及明确标记为未做多重比较校正的 McNemar exact p 值。无完整配对时不生成差值结论，smoke 不进入结果汇总。

`seed_aggregates.json` 保存各配置已完成种子的结果；本批仅有 seed=233，样本标准差为 null，汇总显示 `SD not applicable (single seed)`。按题配对 bootstrap 与其他分析保持不变。`analysis_metadata.json` 记录分析脚本 SHA256。配对对照同时包含 H2O-step 减 H2O-token、TOVA-step 减 TOVA-token，以及 StepKV 减各对照。

简洁进度命令：`.runtime/rebuttal/bin/python scripts/rebuttal_status.py`。

重新汇总：

```bash
.runtime/rebuttal/bin/python scripts/summarize_rebuttal.py --root results/rebuttal_20260927/controlled_v2
```

发生错误后先检查对应日志；配置/源码未变时可原命令续跑，逐题恢复。源码或协议改变时使用新的 `--output` 目录，保留旧 manifest 与失败记录。后台任务独立于本次聊天存活，但不意味着聊天助手会在无人触发时自动修复错误。

首轮 `controlled_v1` smoke 发现 stop-marker 共享 BPE token 的截断问题，队列已停止，没有启动主实验。该目录保留用于审计，不计入结果。修复后的 `controlled_v2` 重新从 smoke 开始，相关回归测试已通过。
