# 新 baseline 执行记录（2026-09-28）

**最新状态：下述 controlled 新引擎队列已停止。按作者要求，新 baseline 改接原评测引擎，当前路径与验证要求见 [REBUTTAL_LEGACY_MODE.md](REBUTTAL_LEGACY_MODE.md)。本文件其余内容保留为先前执行记录。**

作者确认：当前只使用 Qwen2.5-7B-Instruct、seed=233，不跑 Llama 或额外 seed。大模型候选已确定为 Qwen2.5-32B，但本轮暂不启动。

原 `controlled_v2` 队列在 GPU 2 继续完成 13 个主任务（6,500 次题目×条件评测）。新增 baseline 使用独立源码 worktree、独立结果和 GPU 0，与原队列并行；GPU 0 上的新 baseline 按顺序执行，避免多个模型同时抢占剩余显存。

| 方法 | 执行次序 | 实现范围 |
|---|---:|---|
| `flowkv_style` | 1 | FlowKV Appendix D 的一次性压缩机制；冻结压缩后的历史，仅给新完成块分配新增预算；与原 TOVA-step 相同的 EOS probe、最后层/最后 query/head mean selector。明确标为 agent-step adaptation。 |
| `rkv` | 2 | 基于官方 serving-port 全局共享 token selector 的多轮适配；19 项 CPU 测试和独立审查通过，已冻结并排队，GPU smoke 待执行。 |
| `lazyeviction` | 3 | 基于官方真实 query recurrence、逐头延迟淘汰的多轮适配；19 项 CPU 测试和独立审查通过，已冻结并排队，GPU smoke 待执行。 |

每种方法先做 20% 和 50% 两组 smoke，各 2 题；两组通过后再跑两个预算各 500 题。均使用 seed233 第 501–502 题做 smoke，第 1–500 题做主评测；不根据 smoke EM/F1 调参。新增计划共 6 个主任务（3,000 次题目×条件评测），另有 12 次 smoke。方法尚未通过实现验证时，队列会等待该方法的 readiness 文件；不能把排队当作实验已完成。

FlowKV-style 已通过新旧 runtime 共 18 项 CPU 测试，2026-09-28 在 GPU 0 完成两个预算的 smoke，各 2 题；均确认实际淘汰、全局预算及冻结历史保留，随后自动进入 20% 主实验。其正式结果可和原 controlled_v2 按相同题目 ID 配对，汇总前检查共同引擎源码、模型/数据 revision、解码协议和软件版本；`CONTROL_COMPARISONS.md` 与 `paired_control_statistics.json` 单独记录这些比较。

R-KV/LazyEviction 需要真实 query 信息，其 attention/预算/暂存窗口与现有 EOS-probe 控制不同，必须保留各自 manifest 的 adaptation 说明和实际峰值内存；不能把方法名当成官方端到端复现。LazyEviction 若采用 query-head 展开的 KV，还必须计入展开后实际字节数与 recurrence/回滚状态。

## 目录与查看命令

- 新队列入口：`scripts/run_rebuttal_extensions.py`。
- 结果：`results/rebuttal_20260928/new_baselines/`，包括 `queue_plan.json`、`queue_status.json`、`queue_process.json`、`queue.log`。
- 隔离源码父目录：`/data/experiment/fengboyu/stepkv/rebuttal_extensions_20260928/`；FlowKV 使用 `source/`，R-KV 使用 `rkv_source/`，LazyEviction 使用 `lazy_source/`。
- 每个 baseline 的 readiness 文件记录测试与冻结源码哈希。源码变化后拒绝混合续跑；修正需要新结果目录与新记录。
- `source_snapshots/` 保存三个 baseline 的冻结源码、测试、适配说明与来源许可证；每个方法的 `source_pin.json` 固定从第一项任务起使用的源码。
- 当前为共享 A100 GPU；所有耗时只用于调度和诊断，不作为论文独占硬件速度结论。

```bash
.runtime/rebuttal/bin/python scripts/rebuttal_status.py
.runtime/rebuttal/bin/python scripts/rebuttal_status.py --root results/rebuttal_20260928/new_baselines
tail -n 30 results/rebuttal_20260928/new_baselines/queue.log
```

断点恢复新队列：

```bash
.runtime/rebuttal/bin/python -u scripts/run_rebuttal_extensions.py --gpu 0
```

脚本使用进程锁防止重复调度，已完成任务保留；接管仍存活的相同 worker 时核对完整命令。停止时间沿用 2026-10-12 18:00 UTC，届时停止启动新任务。
