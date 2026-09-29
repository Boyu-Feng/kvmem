# 恢复原评测模式

**2026-09-28 后续更新：用户已取消全部R-KV任务，要求替换为SideQuest。两个新增数据集的执行计划见 [REBUTTAL_MULTIDATASET.md](REBUTTAL_MULTIDATASET.md)，SideQuest当前为独立原型验证，见 [SIDEQUEST_PILOT.md](SIDEQUEST_PILOT.md)。下文R-KV接入描述保留为历史记录。**

2026-09-28，按作者“改回之前的评测模式、使用之前的结果”的要求执行。

## 当前执行约定

- 原表结果继续作为主结果，不用 controlled_v2 数字替换。参考值见 [ORIGINAL_RESULTS_REFERENCE.md](ORIGINAL_RESULTS_REFERENCE.md)。原方法不重新排 500 题主实验。
- 继续只使用 Qwen2.5-7B-Instruct、seed=233；新增 baseline 跑 20% 和 50%，不启动 Llama 或 Qwen2.5-32B。
- 原评测入口仍为 `run_all_wiki_experiments_v2.py`。`run_rebuttal_legacy.py` 负责本机资产路径、选题、输出隔离和 manifest；不重写原解码、position_ids、stop/crop、首轮预算或各方法默认评分来源。
- 原脚本 `run_qwen25_7b_wiki_experiments.sh` 保持原样供溯源，其中含多 seed 旧配置；本轮使用固定 seed=233 的包装入口，不运行该多 seed 脚本。
- 新增 baseline 通过独立 `LegacyBaselineLLM` 接入必要的选择/淘汰逻辑，不改公共旧引擎。方法适配范围记录在各自 manifest 和 `LEGACY_VALIDATION.json`。
- Agent IntentKV（arXiv:2606.09916）的公开代码及训练权重未找到，作者也确认手头没有；随后明确选择“暂缓 IntentKV，继续其他 baseline”。因此 IntentKV 不进入当前运行队列，也不启动 Phase1 或重新训练。证据保留在 `/data/experiment/fengboyu/stepkv/rebuttal_legacy_20260928/intentkv_source/ARTIFACT_STATUS.md`。

## 已恢复及保留内容

`run_all_wiki_experiments_v2.py`、`kv_cache/pruning_strategy.py` 已恢复为修改前 `before_changes.tar` 的逐字版本。包括模型、tracker、selector、检索器在内的 9 个核心文件均在每次新运行前与快照核对；不一致就拒绝启动。

原 StepKV 使用 piggyback attention；原 H2O/TOVA 首轮生成后剪枝；原 generate/token-loop、默认 repetition penalty、位置与停止逻辑均保留。

原 controlled 队列及 worker 已停止。新引擎入口和队列入口已添加停用提示，避免误启动。其源码备份、停止记录和恢复哈希保存在 `results/rebuttal_20260928/legacy_restore/`；原数值、manifest、checkpoint 均保留，状态文件标记为因协议恢复而停止。

## 新增 baseline 队列

入口：`scripts/run_rebuttal_legacy_queue.py`，结果目录：`results/rebuttal_20260928/legacy_baselines/`。

各方法先通过 CPU 检查，再分别运行 20% / 50% 的 2 题 GPU smoke；必须实际多步执行和发生 KV 淘汰，日志无运行错误，才能进入两个预算各 500 题的主实验。Smoke 使用 seed233 第 501–502 题，主实验使用第 1–500 题。不会根据 smoke EM/F1 调整方法。

原 StepKV、H2O 各完成 2 题 GPU 检查，确认真实多步执行与剪枝。FlowKV-style 的两个预算共 4 题 GPU smoke 均通过，20% 主实验已启动；R-KV/LazyEviction 的 CPU 验证已完成，等待队列中的 GPU smoke。三种适配共 36 项 CPU 测试通过。

主实验完成后也核对逐题 ID、sidecar 完整性和错误日志；原引擎吞掉异常后生成的答错记录不会被静默纳入新 baseline 汇总。该检查在推理结束后执行，不改变原生成路径。Sidecar 写盘包含在本轮 sample_time 内，因此耗时仅作诊断，不能直接与旧表比较速度。

已接入队列顺序为 FlowKV-style → R-KV → LazyEviction。每个方法的源码从第一次 smoke 起固定；修改源码必须使用新输出目录，不能续接不同实现的检查点。队列只汇总本目录已完成的新 baseline，排除已停止的新引擎结果。

另有 GPU 0 的 H2O-step / TOVA-step 原代码对照队列：`scripts/run_rebuttal_legacy_step_queue.py`，输出 `results/rebuttal_20260928/legacy_step_controls/`。首轮保持原 H2O/TOVA 的生成后剪枝；仅将后续 observation/token 剪枝延迟到完整生成及原 stop 裁剪之后，继续使用原 manual token 解码、EOS scorer 和 32/0 近期窗口。每项仍先检查两个预算各 2 题，再做两个预算各 500 题。独立入口 `run_rebuttal_legacy_step.py` 避免改动正在运行的其他 baseline 源码指纹。

查看进度：

```bash
.runtime/rebuttal/bin/python scripts/rebuttal_status.py
```

## 结果解释范围

恢复的是本次修改前的本地代码快照。送审表格的逐题结果、实际模型/数据 revision 和环境记录尚待对应，不能仅凭恢复代码就宣称精确复现了旧表。新增 baseline 为 seed233 单次结果，原表为多 seed 均值与标准差；保留各自统计口径，不将两者拼成配对显著性结论。

## W3 效率补测

用户要求补充效率证据。独立入口为 `scripts/run_rebuttal_efficiency.py`，队列为 `scripts/run_rebuttal_efficiency_queue.py`，输出为 `results/rebuttal_20260928/efficiency_legacy_v1/`。仍使用九个原始核心文件，单张 GPU 0（A100-SXM4-40GB）、Qwen2.5-7B、seed233，固定前20题；第501–502题预热，20题各两次主计时和一次诊断重放。比较 FullKV、StepKV20/50、H2O20/50、TOVA20/50，共7条件。

当前 GPU0 的准确率 worker 自然完成后，maintenance gate 暂缓启动下一项，等待 GPU 无其他进程再开始测量；补测完成或失败后自动恢复准确率队列。仅重新启动 controller 以接入 gate，当前 inference worker 的 PID 与运行状态保持不变。GPU2 的 baseline 队列继续运行。

主延迟与细粒度诊断分开记录。主延迟覆盖原 episode 的检索、解析、评分、step utility 和 KV 管理；CUDA 同步，结果写盘移出测量区。原 reset 原位执行后重置 allocator peak，避免上一题驻留缓存污染本题峰值。诊断读取真实 K/V 的 tokens 和 tensor payload MiB，含保护区总量与 trajectory 量分列，覆盖逐 forward 及剪枝前后；其他副本及工作张量由 allocator peak 单列。每题强制 clean repeats 与诊断重放的输出、轨迹及最终 KV 字节校验；失败不计入汇总。

测量模块12项CPU验证与maintenance机制13项CPU验证通过；实际GPU一致性将在运行中逐题检查。该子集只用于新的效率证据，不替换原表准确率。回复草稿与统计口径见 `REBUTTAL_W3_REPLY_ZH.md`；送审 Figure4/5 的原始数据及原硬件仍未溯源，新补测不能冒充历史图的原始值。
