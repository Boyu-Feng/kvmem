# 两模型三 seed baseline 补跑

用户要求：Llama-3.1-8B 补齐此前六个 baseline 的三个 seed；IntentKV agent
版本在 Qwen2.5-7B 和 Llama-3.1-8B 上补齐三个 seed。完整已有组校验后复用。

## 固定范围

| 模型 / 方法 | 正式组数 | 已有并复用 | 新增正式组数 |
|---|---:|---:|---:|
| Llama-3.1-8B：FlowKV-style、LazyEviction、H2O-step、TOVA-step、ThinkKV 仅淘汰 | 90 | 0 | 90 |
| Llama-3.1-8B：SideQuest 未微调适配 | 9 | 0 | 9 |
| Llama-3.1-8B：IntentKV agent 规则适配 | 18 | 0 | 18 |
| Qwen2.5-7B：IntentKV agent 规则适配 | 18 | 6 | 12 |
| 合计 | 135 | 6 | 129 |

公共抽样 seed 为 **233、42、3407**；每个 seed 对应 HotpotQA、2WikiMultihopQA、
MuSiQue 的 500 道正式题，所有方法和两模型共用相同内容、ID 和顺序。
九份题单逐一对照 GitHub 完整快照的 ID 与原始索引；不同 seed 的题目集合不同。
每个新条件另有抽样位置 501/502 的两题 smoke，与正式题无重叠。
共补跑 **64,500 次正式评测和 258 次 smoke 评测**。

每题最多七个 ReAct 步骤、每生成块 256 token，原始提示、检索、动作解析及
EM/F1 计分保持原版本。采用同一套完整 5,233,235 篇 Wikipedia 检索资产。
SideQuest 使用原自适应预算，其余方法为名义非 prompt token 保留比例 50%、20%。
IntentKV 在每个模型和 seed 内，必须先完成三个数据集的 50% 正式组，才能进入 20%。

Llama revision 为 `0e9e39f249a16976918f6564b8830bc894c89659`；
Qwen revision 为 `a09a35458c702b33eeacc393d103063234e8bc28`。
两模型均为原生 BF16。准备阶段重新哈希全部权重分片，并校验 metadata、tokenizer、
数据载荷及运行依赖；plan 保存来源和源码哈希。实际模型加载后检查原生拓扑、
全参数/缓冲区设备、BF16 及 generation config。

## 方法边界

IntentKV 指《IntentKV: Cross-Turn Intent-Aware KV Cache Pruning for Agent Inference》
的现有 `intentkv_rule` agent 适配，沿用 QueryMemory、候选归一化、全层全 query-head
求和和 forced-set 规则，不包含训练残差头。Qwen 使用原类；Llama 只新增对真实
`LlamaAttention` 的 post-RoPE query 捕获，保持相同选择公式。

LazyEviction 使用原逐 query-head recurrence、窗口和选择算法，Llama 仅适配原生
Llama rotary helper 与 attention 安装。其 KV 从 8 个原生头扩展为 32 个 query 头，
实际字节单独保存，不能把同名义比例解释成等字节预算。

保留历史 SideQuestPilot 和 ThinkKV 仅淘汰版的独立 greedy decoder / logical RoPE；
其余五个方法保留历史 HF generate 后裁剪路径。这是按原适配补齐模型/seed 的实验，
不同 decoder 的结果单列，不能据此作严格同运行时或同显存排名。
本轮准确率运行中的时间仅为诊断，不作为独占硬件延迟测量。

## 复用与恢复

Qwen7B IntentKV seed233 六个正式组来自：
`results/alfworld_h200_20260929/intentkv_rule_qa_7b_seed233_20261005_v5/`。
复用前验证完整 500 题、manifest/checkpoint/final 一致性、每题 audit 哈希、
原计分器结果、题目内容/顺序、模型 revision 和旧源码绑定。
其他三 seed Qwen baseline 已在 GitHub 完整快照，本任务不重跑。
已完成 Llama StepKV 组件消融属于另一组方法，不冒充本批六个 baseline。

新入口位于 `scripts/baseline_multiseed/`，不修改历史冻结 runner 或 adapter。
每题先持久化 audit，再原子提交 checkpoint；恢复只接受完整有序前缀，
源码/资产/题目不匹配时拒绝恢复。运行错误保留，不转换为零分；失败依赖被阻止，
其他独立条件继续。队列重启拒绝重复启动仍存活的 worker。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  /mnt/checkpoints/bruli/fengboyu/kvmem_rebuttal_20260929/venv/bin/python -B \
  -m scripts.baseline_multiseed.queue --prepare --devices 4,5,6

CUDA_VISIBLE_DEVICES='' \
  /mnt/checkpoints/bruli/fengboyu/kvmem_rebuttal_20260929/venv/bin/python -B -u \
  -m scripts.baseline_multiseed.queue --execute
```

输出默认目录：
`results/alfworld_h200_20260929/qa_baselines_two_models_three_seeds_20261013_v1/`。
`plan.json` 是冻结计划，`preflight.json` 记录准备检查；`queue_status.json` 与
各条件 `status.json` 记录进度；完成后 `SUMMARY.json` 汇总全部 135 组并标明复用来源。
仅准备计划或 smoke 通过，不表示 500 题正式组完成。

CPU 验证：`tests/test_baseline_multiseed.py` 共 **11 passed**，包括真实小型 Llama
的 decoder/KV 对照、实际淘汰、QueryMemory 强制保留、三 seed 题单、先后依赖、
损坏 checkpoint/audit 拒绝和不完整结果拒绝终结。

## 启动验收

2026-10-13 15:56 北京时间，后台 controller **1659033** 已在 GPU **4、5、6**
运行，八个模型—方法启动 smoke 共 **16/16 题**验证通过。
首个正式条件 `llama31_8b/seed233/hotpotqa/flowkv_style/r50/main` 已提交 **4/500**
题，checkpoint、逐题内容绑定和 audit 哈希检查通过；当时无 worker 失败。
这只是启动截面，不表示新增 129 个正式组已完成。

冻结 plan SHA256：`7a65e7ee2ef144ddedd91fecc4c9ba3acd8dab18db23da16d2d2e0e22585b487`。
输出中的 `source_snapshot/` 保存与登记哈希一致的源文件副本。

- [实时队列状态](../../results/alfworld_h200_20260929/qa_baselines_two_models_three_seeds_20261013_v1/queue_status.json)
- [启动核验记录](../../results/alfworld_h200_20260929/qa_baselines_two_models_three_seeds_20261013_v1/launch_verified.json)
- [完整冻结计划](../../results/alfworld_h200_20260929/qa_baselines_two_models_three_seeds_20261013_v1/plan.json)

## 2026-10-14 资源扩容

用户要求启用其他空闲 GPU，并允许单卡多进程。资源配额扩为 **GPU 0–7、每卡最多
两个 worker，总并发最多 16**。每进程原有的 48 GiB CUDA allocator 上限保持不变，
启动要求该卡至少有 52 GiB 空闲显存；发现本实验之外的计算进程时不再向该卡追加任务。
调度按模型／方法／seed 分散，优先启动尚缺的 IntentKV 和 TOVA；原有 smoke 和
IntentKV 50%→20% 依赖继续强制执行。

原始 `plan.json`、所有已冻结源码、逐题顺序和评分指纹保持原哈希。
`scripts/baseline_capacity.py` 用独立资源修订文件
`capacity_20261014.json` 记录新增 GPU UUID、并发、资源入口源码哈希和用户授权，
没有改写原计划中的设备字段。新 worker 的评测主体与原 worker 经完整 AST 对照，
差异仅限资源准入及资源来源记录；推理、预算、存储、计分和错误处理保持一致。
准确率运行时间继续仅作诊断，不作为独占硬件延迟结果。

2026-10-14 11:12 北京时间启动新 controller **3019743**，仅替换旧 controller
**1659033**。原有 H2O worker **2942520、2962114、2973519** 按原 PID 和启动时间接管，
没有停止或重新评测。`capacity_handoff_*.json` 保留接管前状态及进程身份。
新调度器复核完整结果后跳过；失败记录保留，不自动重跑。

资源修订 SHA256：`c9c916082f5f933279051a633e5b18ded4eb8fbf4f416d421f8e79714539db6b`。
新入口源码副本位于输出目录 `capacity_source_snapshot/`。原始测试和扩容测试合计
**18 passed**，覆盖完整评测主体不变、显存／并发限制、外部 PID 和 PID 重用拒绝、
跨模型／seed 调度、前置依赖、完成／失败跳过及仅替换调度进程。

```bash
# 本次初次接管已执行；不要重复启动仍然存活的 controller。
CUDA_VISIBLE_DEVICES='' \
  /mnt/checkpoints/bruli/fengboyu/kvmem_rebuttal_20260929/venv/bin/python -B -u \
  -m scripts.baseline_capacity --execute
```

以上命令仅用于 controller 退出后的恢复，存活 worker 会按 PID／启动时间被接管。
`queue_status.json` 保持为统一进度入口，`active` 键改为 `GPU:slot`；每个新条件的
`capacity.json` 绑定本次资源修订。后台日志为 `capacity_controller.log`。

11:15 后的扩容验收确认：**16 个存活 worker、每卡两个位置、零失败**，原三个
H2O 进程身份保持不变且继续提交。Llama IntentKV、Qwen IntentKV、Llama TOVA
三个新增正式组分别已有 **3、5、3** 条提交，题单前缀、指纹及所有已提交 audit
哈希检查通过。这是启动验收截面，完整结果仍以实时队列状态为准。

- [扩容启动验收](../../results/alfworld_h200_20260929/qa_baselines_two_models_three_seeds_20261013_v1/capacity_launch_verified.json)
