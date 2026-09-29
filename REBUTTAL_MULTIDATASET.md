# 2Wiki / MuSiQue 后续补实验

2026-09-28 用户要求“后面把其他两个数据集的也测了”。按仓库原有三个 QA 数据集扩展至 2WikiMultihopQA 和 MuSiQue。

每个数据集运行 FlowKV-style、LazyEviction、H2O-step、TOVA-step；每种方法 20% / 50% 两个预算、500 题、seed233、Qwen2.5-7B-Instruct、最多7步。共16个主实验，8,000次题目评测；另有16个2题smoke任务，共32次检查。用户随后要求取消R-KV并改为SideQuest，R-KV在三个数据集的未启动任务均已移除，旧计划保存在各队列的`rkv_cancellation.json`。SideQuest先执行独立20题原型验证，再按未微调适配版各补一组500题自适应预算实验，详见 `SIDEQUEST_PILOT.md`；因此两个新增数据集为16组legacy补实验加2组SideQuest探索性补测，共18组，不能将不同解码协议混为正式对照。原表方法不重新排主实验，原结果继续保留。

数据来自固定 revision 的开发集，沿用仓库数据归一化与抽样函数：2Wiki 12,576题，MuSiQue 2,417题。每项主实验使用 seed233 排序后的前500个不同ID；smoke 使用第501–502题。下载文件、归一化文件的 SHA256、来源 revision、选题ID保存在 `/data/experiment/fengboyu/stepkv/datasets/{2wiki,musique}/ready.json` 和 `results/rebuttal_20260928/multidataset_preflight/`。

本次为数据集扩展：固定当前 HotpotQA legacy adapter 的推理和评分参数，只替换题目与指标标签。历史 MuSiQue CLI 对部分方法默认使用 piggyback attention，本次不会自动套用该覆盖项。该差异写入每个 manifest；新增单 seed 数字不作为历史多 seed 表格的精确复现或配对显著性证据。

入口：`run_rebuttal_multidataset.py`；队列：`scripts/run_rebuttal_multidataset_queue.py`。

| 队列目录 | GPU | 方法 | 启动条件 |
|---|---:|---|---|
| `results/rebuttal_20260928/multidataset_baselines/` | 2 | FlowKV-style、LazyEviction | `legacy_baselines` 队列成功完成 |
| `results/rebuttal_20260928/multidataset_step_controls/` | 0 | H2O-step、TOVA-step | `legacy_step_controls` 队列成功完成 |

两条 controller 已独立持久启动。前序失败时显示 `blocked_by_previous_queue`，不跳过失败；前序成功后依次运行2Wiki、MuSiQue。每种方法先通过两个预算的smoke，再执行主实验。每题原子检查点、核心源码与adapter哈希、样本ID、sidecar及错误日志均有校验。当前 HotpotQA runner 未改写。取消R-KV时仅重启baseline controller并接管原FlowKV worker，worker PID保持2234799；step control队列未重启。

GPU3 仅用于先行检查两数据集的 FlowKV-style 20% smoke，结果仍写入对应后续队列，完成后供队列验证复用。

查看全部进度：

```bash
.runtime/rebuttal/bin/python scripts/rebuttal_status.py
```

结果完成后自动写入各目录 `SUMMARY.md`。本批耗时仍只作为运行诊断，不是独占 GPU 效率实验。

2026-09-28 后续新增 ThinkKV / ThinKV 候选：两个数据集各登记 20% / 50% 两档、500 题、seed233，连同 HotpotQA 共 6 组候选。当前先做独立的 thought 类型校准预检，尚未接入正式推理 runner/controller，因此不计入上面已可运行的 18 组。预算与校准前提见 `THINKKV_REBUTTAL.md`，状态入口已包含这项候选。
