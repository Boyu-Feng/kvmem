# ThinkKV / ThinKV rebuttal 候选实验

**2026-09-29 补充：以下未通过的是三类 thought 校准预检，不代表 ThinKV 所有路线不可用。论文附录 E.10 针对普通 LLM 给出了单类 thought、统一 4-bit、达到预算后淘汰的分支，不要求三峰分类。目前该分支尚未实现，TBQ/TBE runner 与 controller 也未接入；下一步优先按当前 Qwen2.5-7B-Instruct setting 验证这个分支。详见 [后续计划](docs/rebuttal/NEXT_STEPS.md)。**

用户于 2026-09-28 要求将 ThinkKV 加入 rebuttal 候选，并使用 20% / 50% 两档保留比例。论文为 [ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models](https://arxiv.org/html/2510.01290v2)；文中也使用 ThinkKV 名称。

## 实验矩阵与预算

| 数据集 | 模型 | 目标轨迹 token 保留比例 | 每档主实验题数 | seed |
|---|---|---|---:|---:|
| HotpotQA | Qwen2.5-7B-Instruct | 20%、50% | 500 | 233 |
| 2WikiMultihopQA | Qwen2.5-7B-Instruct | 20%、50% | 500 | 233 |
| MuSiQue | Qwen2.5-7B-Instruct | 20%、50% | 500 | 233 |

共 6 组候选主实验。20% 指保留 20%，不是删除 20%。初始 prompt 单列保护，可压缩轨迹包含生成内容和后续工具 observation。分段淘汰可能低于目标；活跃段和最小段保留量也可能形成高于目标的下限，必须报告实际比例。动态比例预算是本 QA 实验的适配，论文的固定 token 容量不是同一设置。

**20 题校准预检已完成，当前配置未通过：共同三峰层为 0，所需层数为 4。6 组主实验尚未启动，也未配置自动启动推理的 controller。** 机器可读候选矩阵见 `results/rebuttal_20260928/thinkkv_candidate_v1/experiment_plan.json`，完整逐题预检结果见该目录的 `SUMMARY.md`。不能将“登记候选”表述为“已完成复现”或“主实验正在运行”。

## 实现前提与当前验证

在论文页面、作者 GitHub 和 GitHub 仓库搜索中尚未找到作者的公开实现；搜索结果及论文快照保存在 `results/rebuttal_20260928/thinkkv_preflight/`。因此尚未有可声称为官方 ThinkKV 的运行结果。

第一步检查论文 Algorithm 1 的 thought 类型校准能否迁移到当前 QA 模型。校准器 `scripts/thinkkv_calibration.py` 使用原 FullKV 生成与评分路径，逐层观察真实 decode query。它不改注意力输出、token、位置、stop/crop 行为；CPU 小模型测试覆盖 prefill、连续 decode、工具块 prefill 和 stop 裁剪后的 logits/KV 精确一致性。

校准使用 HotpotQA seed233 第 523–542 题（0-based sample_start=522），与主实验前 500 题、smoke 第 501–502 题和 SideQuest pilot 第 503–522 题均不重合。这是 **20 题 QA 兼容性预检**；论文使用 100 条 s1K 提示和推理模型，本预检不能代替论文设置的复现。

统计依论文 C.2：GQA 组内 query logits 做 max pooling，再 softmax，并在 KV groups 间平均；以行最大注意力的 1% 为阈值计算稀疏度。校准对每题、每层做 Gaussian KDE，要求共同层在全部校准题中都出现三个峰，取四个共同层、平均谷值作为两道阈值。KDE 使用 Scott 带宽（论文未给出带宽），多于四个候选层时按层号选取，未使用 EM/F1 调参。没有共同层时，明确报告预检失败，不以关键词或分位数强行制造 thought 标签。

校准数据：`results/rebuttal_20260928/thinkkv_preflight/calibration_qa20/`，逐题原子检查点、原始源码哈希、模型/数据资产、独立样本 ID 和计算结果均保留。

本次 20 题全部正常结束，无 runtime error；未得到可供推理使用的分类阈值，因此没有运行 TBQ/TBE 压缩。对前 13 题额外检查了固定 KDE 带宽 0.1/0.2/0.3/0.5、Scott 和 Silverman，共同三峰层也均为空（`bandwidth_sensitivity.json`，仅诊断，未用于推理）。这不是对论文方法普遍适用性的否定；校准数据、模型以及未公开的实现细节都与论文设置存在差异。后续需要可用的作者实现/校准配置，或明确验证另一套校准设置，才能继续实现并运行主实验。

## 完整适配需要保留的机制

- 基于校准的 attention sparsity thought 分类，刷新间隔 128。
- TBQ：R/E 用 4-bit NVFP4、T 用 2-bit ternary，组大小 16；keys 按 channel、values 按 token 分组，保留未填满的高精度 buffer。
- TBE：transition 结束时渐进淘汰先前段；超预算时选择最不重要类型中最老的可压缩段。每段保留阶梯 64/32/16/8/4，基于 post-RoPE keys 做段内聚类和代表选择。
- 论文的 CT/PagedAttention kernel 与软件实现必须区分；不能将 fake quantization 的理论 bit 数当作实际显存下降或宣称复现论文吞吐。

正式推理接入还需完成上述适配、无压缩控制的一致性测试、真实量化/淘汰 smoke。若校准预检不成立，先记录原因并保持这六组为候选，不能用不同算法挂 ThinkKV 名字产出比较表。

## 最终应保存的指标

EM、F1、题目数；逐题等权平均终态轨迹 token 保留率；合并 token 后的保留率；明确采样位置的过程平均；包含/排除初始 prompt 的两种口径；实际 KV 存储字节（含量化 scales/buffers）、临时反量化工作区及峰值显存；量化格式、淘汰次数和预算下限。时间统计须注明是否使用论文 CT kernel。

这些候选用于用户最后选择合适 baseline。R-KV 的取消保持有效，现有 baseline/SideQuest 的队列与推理源码不因本预检改动。
