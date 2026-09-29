**StepKV 审稿总结与 rebuttal 执行方案**

**2026-09-28 最新决定：恢复修改前评测模式，保留作者原表结果；controlled_v2 及其扩展队列均已停止。新增 baseline 在原引擎中继续，另核查 Agent IntentKV。当前执行说明见 [REBUTTAL_LEGACY_MODE.md](REBUTTAL_LEGACY_MODE.md)，以下先前计划不覆盖此决定。**

2026-09-28 最新执行范围：作者要求当前只跑 Qwen，固定 seed=233；先启动新增 baseline，大模型 Qwen2.5-32B 暂缓，Llama 不启动。新增 baseline 的实际队列与适配边界见 [REBUTTAL_EXTENSIONS.md](REBUTTAL_EXTENSIONS.md)。下文保留完整证据规划，不能据此默认扩大当前运行范围。

2026-09-28 协议核查：本轮实际采用新增 RebuttalLLM，除剪枝时机外，还改变了首轮预算执行、StepKV attention 来源、RoPE 位置、解码与 stop/cache 处理。下文“仅改变时机”是原定最小对照目标，不能用来描述本轮与送审结果之间的差别。当前结果应独立标为 controlled_v2，不直接替换送审表格；详见 [源码差异核查](REBUTTAL_PROTOCOL_AUDIT_2026-09-28.md)。

整理日期：2026-09-27。截止日期：2026-10-13。已知资源：可使用 H200、A100，卡数与具体可用时段尚未确定。字数限制未提供，因此本文件先组织证据与实验，不假定最终回复篇幅。

2026-09-28 执行范围更新：作者决定当前 rebuttal 批次只跑 **seed=233**，保留 **20% 和 50%** 两个预算，不执行 seed=42、3407 或其他多 seed 重复。以下原计划中“三组匹配”“三组抽样”等新增实验建议不适用于本批；原始送审结果的溯源仍按实际运行记录核对。本批报告单 seed 结果，不报告跨 seed 标准差；保留相同题目 ID 上的配对 bootstrap、95% CI 和其他现有分析。实际队列范围见 `REBUTTAL_RUNBOOK.md`。

本文件依据五份审稿意见、当前仓库的静态检查，以及相关工作的原论文/官方资料。没有启动模型实验。下面所有“补实验”“预期交付”都是计划，不是已有结果。

2026-09-27 更新：作者明确“完整 step 后再剪枝”是方法组件。保留这一设计及原生 baseline 比较，新增 H2O/TOVA 在相同 step 边界剪枝的对照。SideQuest 按当前公开 artifact 不可获得处理；SnapKV 从必跑项降为需要显式 online 适配的可选项。近期方法优先核查 LazyEviction，结构控制优先 FlowKV-style，ChunkKV 与额外候选 R-KV 按接入成本选做。

**一、总体判断**

五位审稿人都认可问题重要、方法训练成本低，以及紧预算下的正向结果。核心阻力集中在：实验协议是否公平且一致、贡献能否与简单结构保护和延迟剪枝区分、启发式信号是否可靠、效率与泛化证据是否充分。

优先处理实验可信度，再强化方法贡献。单纯增加更多数据集或更大的模型，不能消除剪枝时机、FullKV/ReAct 参照组、表格数值来源的疑点。

| 审稿人 | 当前评价特征 | 最关心的问题 | 回应重点 |
|---|---|---|---|
| R1 | 问题相关性 4；技术与复现 2；具体且可操作 | 时机矛盾、在线候选集、效率缺项、复现与消融不一致、近邻方法 | 用协议表、代码语义、同条件对照和补充测量逐项回答 |
| R2 | 技术与复现 2；主要质疑覆盖范围 | 近期 agent baseline、更多任务形态、更强模型 | 一个直接近邻比较，加一个较强模型和一个真实不同的交互设置 |
| R3 | 技术质量 3；主要认可方法有效 | 启发式与词面相似、陈旧步骤累积、运行开销、超参 | 长轨迹诊断、语义相似度替换、敏感性与开销分解 |
| R4 | 正面评价较多，但技术 2、置信度 2 | 相对近期工作的技术区别、隐式语义依赖、非 ReAct 结构、超参 | 清楚解释 token/step 两类信号，给适用边界和机制证据 |
| R5 | 技术 2；对实验可信度追问最全面 | baseline、测试集调参、词面脆弱、FullKV/ReAct 异常、统计、泛化 | 对齐参照实现与统计，补强 baseline 和稳健性，不先替异常结果编解释 |

五位的 novelty 都是 3；四位的 technical quality 是 2，只有 R3 是 3。所贴内容没有总体接收分，不能据此判断接收概率。

建议把贡献表述收敛为：**在可观测的 agent 交互边界上，用工具有效性、证据增益、动作重复和后续重用代理信号估计 step utility，再与模型注意力得到的 token saliency 融合，在线分配有限 KV。** “结构化保留”“延迟淘汰”“信息后来重新重要”本身都不应宣称为首次提出。[FlowKV](https://arxiv.org/abs/2505.15347)、[LazyEviction](https://arxiv.org/abs/2506.15969v3) 和 [SideQuest](https://arxiv.org/abs/2602.22603v2) 已覆盖其中不同部分。

**二、当前仓库已经能确认什么**

这些事实描述当前 checkout，不自动等于送审实验。第一项工作必须是匹配送审 PDF、运行 commit、配置和结果。

| 发现 | 代码证据 | 对 rebuttal 的影响 |
|---|---|---|
| 本地 tex 是另一版本：HotpotQA StepKV 写 26.50，而审稿意见引用 23.23/18.07 | [论文草稿](/home/fengboyu/kvmem/stepkv_acl_draft.tex:367) | 不能用旧稿直接解释送审 Table 1–3 |
| 当前 StepKV 在完整 decode 与截断后触发剪枝；H2O/TOVA 等走逐 token 剪枝路径 | [StepKV 触发点](/home/fengboyu/kvmem/models/QwenLLMWithKVCache.py:982)、[token 路径](/home/fengboyu/kvmem/models/QwenLLMWithKVCache.py:1236) | 作者确认时机是方法设计；保留原比较，追加同 schedule 对照，并按 Table 1 实际协议纠正附录的“全部相同”表述 |
| StepKV 还有每个 step 的最低保留机制；poolwise 分支当前比例为 10%，总预算不足时会裁掉 floor 中的 token | [两阶段选择](/home/fengboyu/kvmem/kv_cache/pruning_strategy.py:956) | 需要分离 timing、floor、utility；不要把所有收益归给 step score |
| `no_step` 消融把 beta 设为 0，保留 StepKV 策略入口 | [消融定义](/home/fengboyu/kvmem/run_stepkv_component_ablation.py:78) | 它适合回答同流程去掉 step score 的问题；还需确认实际运行是否保留相同 floor、预算和样本 |
| 消融默认 500 题、seed=42、ratio=0.5，Full 默认从主实验 run2 导入 | [消融参数](/home/fengboyu/kvmem/run_stepkv_component_ablation.py:722)、[Full 导入](/home/fengboyu/kvmem/run_stepkv_component_ablation.py:885) | “21.60 是单次 run2，23.23 是三次均值”是合理待验证假设，尚不是结论 |
| 主实验运行脚本记录原 run1=233，另跑 42、3407；seed 控制题目抽样，生成采用 greedy | [三组抽样](/home/fengboyu/kvmem/run_qwen25_7b_wiki_experiments.sh:22)、[抽样函数](/home/fengboyu/kvmem/run_all_wiki_experiments_v2.py:350)、[解码配置](/home/fengboyu/kvmem/models/QwenLLMWithKVCache.py:1283) | 必须说明标准差来自抽样、随机生成还是硬件重复；不能笼统说 decoding variance |
| ReAct 路径把完整 prompt 重新套 chat template，并额外做一次观测用 prefill 后再 generate；FullKV 使用增量 KV 路径 | [ReAct 包装与测量](/home/fengboyu/kvmem/models/QwenLLM.py:79) | 不能把两者当成完全相同的未压缩参照，延迟差异也含实现因素 |
| 当前 `snapkv` 分支对 KV 做 pooling，未使用官方 SnapKV 的 attention 选择机制；调用的 `snapkv_pooler` 在当前构造函数未见初始化 | [SnapKV 分支](/home/fengboyu/kvmem/kv_cache/pruning_strategy.py:1165)、[构造函数](/home/fengboyu/kvmem/kv_cache/pruning_strategy.py:88) | 不能直接运行该入口后称官方 SnapKV 对照；需要忠实移植并验证 |
| step score 有 log1p、上限 8、每次 reuse 增量上限 1，但没有时间衰减 | [score 映射](/home/fengboyu/kvmem/run_all_wiki_experiments_v2.py:1529)、[reuse 更新](/home/fengboyu/kvmem/run_all_wiki_experiments_v2.py:2452) | 能解释增长受抑制，不能声称解决陈旧 step 占位问题 |
| token saliency 在选定层上对 head/query 求和、对层求平均；该函数没有额外 min-max/z-score 归一化 | [attention 聚合](/home/fengboyu/kvmem/kv_cache/h2o_scorer.py:47) | 回答 R1 的归一化问题，并检查不同 observation 长度是否改变 token/step 相对权重 |
| 未找到送审表格的原始结果 JSON/CSV；结果目录在忽略列表中 | [忽略规则](/home/fengboyu/kvmem/.gitignore:8) | 脚本存在不能证明实验完成；表中均值、标准差、显著性还不能复算 |

当前 StepKV 对应 `step_aware_h2o`，不要误把旧的 `kv_cache/ours.py` 当成论文方法。

还有两个应先做小轨迹验证的实现疑点：新 step 注册后，reuse 扫描是否包含该 step 自身的 observation；prefill attention 覆盖范围与 decode 后候选 KV 长度是否一致。这些是当前实现的待验证风险，不能直接推断送审结果有错。若确认且影响结果，需要保留原版本记录，明确更正并重跑受影响比较。

**三、先完成 P0：协议与结果溯源**

1. 建立逐表逐行的结果清单：表号、模型版本、代码 commit、结果文件、样本 ID、seed 的用途、ratio、保护区、prompt、工具后端、解码参数、最大 steps/tokens、硬件。
2. 找回 Table 1–3、Figures 4–5 的运行结果，复算均值与标准差。特别核对 21.60、23.23、18.07。若消融确实只跑单 seed，承认原表未遵循统一报告规则，并补三组匹配实验；不要把单次结果包装成三次均值。
3. 导出每种方法的真正配置，核对 schedule、saliency 来源、最近窗口、floor 和预算分母。论文正文与附录按真实协议统一。
4. 检查 FullKV/ReAct 等价性：固定同一 transcript、相同 chat 模板/特殊 token、相同位置编码与 stop 条件，对比重新 prefill 和增量 KV 的 next-token logits/输出。若输入序列相同，先解释实现差异；若协议不同，明确各自角色。核心压缩比较以同一增量引擎关闭剪枝的 FullKV 为参照。
5. 做小规模 trace 检查：保留原始 token ID、物理 KV slot、step ID 的映射；记录每次候选数、目标预算、实际保留数；检查删掉的 KV 不会无说明地重新出现、新生成 token 不会因 attention 长度错位漏选、reuse 不会把自身重合算成 delayed reuse。

关于在线语义，可在核对后用下述形式补充算法。设历史累计轨迹 token 为 G_t，上次剪枝后的存活集为 R_(t-1)，新写入 token 为 A_t，则候选集为 C_t = R_(t-1) ∪ A_t；保护前缀另外保留。已驱逐 KV 不在 C_t 中，历史文本/anchor 元数据仍可用于评分，但不等于 KV 可恢复。预算若定义为累计轨迹比例，写成 B_t = min(|C_t|, max(1, floor(rho × |G_t|)))；另有保护后缀时要明确它消耗多少轨迹预算。

当前管理器有把累计 logical token 预算换算为当前候选集有效 ratio 的逻辑，因此不能只看到选择器里的 `int(n * keep_ratio)` 就判断为反复对 resident cache 乘 rho。最终要以管理器到 scorer 的完整路径和 trace 为准。[预算换算](/home/fengboyu/kvmem/kv_cache/kv_cache_manager.py:449)

复现说明还需明确：当前 cue 按 Observation 优先、Thought 补充提取；action redundancy 是规范化 action argument 的精确重复，并不是语义相似度。StepKV 默认复用 prefill attention，H2O 默认另做 scoring forward，二者还存在 estimator 与开销差异。[默认配置](/home/fengboyu/kvmem/run_all_wiki_experiments_v2.py:1219) 这些细节也必须在同条件对照中控制。

**四、补实验按证据目标组织**

**E1：分离剪枝时机、最低保留和动态 utility——最高优先级。**

作者确认后的最小新增实验为 **H2O-step 与 TOVA-step**：各自保持原 scorer 和保护规则，仅将剪枝触发改为 StepKV 使用的完整 step 边界。与原生 H2O/TOVA 的差值衡量该时机的影响，与 StepKV 的差距衡量整体策略的额外收益。注意后者还可能包含 scorer/floor 的差异，因此保留 `StepKV w/o step score` 作为单独验证 utility 的控制。无需把“完整 step 后剪枝”从方法贡献中删除；正文和附录仍须一致描述各表的协议。

最小报告表：FullKV、H2O-original、H2O-step、TOVA-original、TOVA-step、StepKV w/o step score、StepKV。先用同一模型、同一数据集、20%/50% 预算完成；同步记录完整 step 内的瞬时峰值。

先在 HotpotQA、Qwen2.5-7B、20% 预算完成全部核心控制，再把主要比较扩到 50%、第二模型或 MuSiQue。最终报告使用预先固定的样本与配对配置，尽可能对齐原文三组抽样；开发调参样本单独划分。下面为建议实验，不是当前已实现的完整矩阵。

| 实验臂 | 剪枝条件 | floor | utility | 要回答的问题 |
|---|---|---|---|---|
| FullKV | 同一 agent 引擎，关闭压缩 | 无 | 无 | 正确的未压缩参照 |
| Token-original | 原基线时机 | 无或按原法披露 | 无 | 保留原方法参考 |
| Token-boundary | 与 StepKV 相同的完成边界 | 关闭 | 无 | 时机改变带来多少收益 |
| Token-boundary+floor | 相同完成边界 | 与 StepKV 相同 | 无 | 简单结构保护带来多少收益；应与 `no_step` 对齐 |
| StepKV-static | 相同完成边界 | 相同 | 初始化 utility，不做后续 reuse 更新 | delayed reuse 的独立贡献 |
| StepKV-full | 相同完成边界 | 相同 | 完整 | 相对上面各控制的增益 |
| FlowKV-style | 相同完成边界，已压历史不再压缩 | 明确匹配规则 | 与所比较方法相同的 scorer 定义 | 一次性压缩是否已能解释收益 |

另加一个可选但有价值的 `StepKV-full, floor=0`，区分 utility 与 floor 的交互。固定大小 chunk 与真实 step 边界的比较可作为结构消融；若只是简化 chunk 控制，不称为官方 ChunkKV 复现。

需要同时匹配保护 prefix、近期窗口、KV dtype、saliency 定义、解码引擎、工具输出和实际总预算。不同 schedule 的步内峰值可能不同，因此同时报 boundary 后 cache 和全程 peak，不把相同 rho 自动解释成相同峰值内存。延迟比较若仍分别用手写逐 token 循环和 `generate()`，不能单独归因于剪枝算法。

FlowKV-style 的实现要公布历史冻结与新步配额。累计比例预算下可以只使用预算新增空间压缩新步；若使用固定绝对上限，必须说明历史占满后的处理，不能暗中重新压历史却仍称一次性压缩。[FlowKV 原论文](https://arxiv.org/abs/2505.15347)

**E2：近期 baseline——优先可获得实现的在线方法和直接机制控制。**

以下为截至 2026-09-27 的原论文与公开代码核查；“未找到可核验 artifact”不等于证明不存在任何作者内部或未公开实现。未运行任何外部代码。

| 方法 | 机制与 artifact 核查 | 新优先级 |
|---|---|---|
| LazyEviction | 官方仓库有 MRI/recurrence 追踪、实际 KV 选择与 gather，以及 Qwen/Llama 运行脚本；属于 decode 期间的延迟淘汰 | **首选近期实测 baseline**；需先适配多 token 工具输出续写 |
| FlowKV-style | 冻结已压缩历史，只压新完成 turn；原论文给出动态预算算法；未找到可核验的独立官方实现入口 | **必做机制控制**；按论文实现并标明 adaptation |
| ChunkKV | 官方链接指向 NVIDIA/kvpress；存在 ChunkKVPress，但默认 hook 只在 initial prefill 执行 | **第二优先级结构比较**；在 step 边界调用 selector 并明确 online 适配 |
| ThinKV | thought 类型相关的混合精度与淘汰；本次未核实可下载且可运行的官方 artifact | 补技术定位，暂不列必跑；完整复现涉及校准、量化与系统改动 |
| agent IntentKV | 2606.09916；Appendix I 仍写代码与 trained residual-head checkpoints 将发布；本次未找到对应官方仓库/checkpoint | 补技术区别与 artifact 限制，不承诺完整复现 |
| SideQuest | 作者确认未开源；公开论文/检索也未定位官方代码、训练权重入口 | 从必跑移除，保留相关工作和区别讨论 |
| SnapKV | 原本用于 prefill；可以显式适配为周期性/step-boundary decode 压缩 | 可选，不能用原生 prefill 结果直接充当在线等预算比较 |
| Quest | 官方代码公开，选择 attention 访问的 KV pages，不等于永久释放常驻 KV | 不优先放入等常驻内存的淘汰比较；若做则另报访问量与延迟 |

**LazyEviction 的价值与接入条件。** 它直接检验“延迟保留、观察未来注意力重现”是否已经足以获得 StepKV 的收益，回应 R1 的近期工作和 delayed-eviction 问题，也回应 R3 的历史重用疑问。[原论文](https://arxiv.org/abs/2506.15969v3)、[官方仓库](https://github.com/Halo-949/LazyEviction) 官方实现基于单次 prefill 加逐 token decode：多 token Observation 带历史 KV 续写、跨工具调用的 recurrence 状态，以及不同 GQA 展开方式都要验证；不能只改 `pruning_mode` 就宣称接入完成。默认环境版本也与本仓库 transformers 5.0 不同，建议移植策略并检查等价性，而非全局替换模型实现。保持原 token 级 recurrence 语义；如果另做 step-boundary 变体，应与原机制明确区分。实际 KV bytes、暂存窗口和额外状态都计入比较。

**FlowKV-style 的最小实现。** 在与 StepKV 相同的完成边界，保留旧的已压缩 KV；仅对新完成 step 选取 token。新步额度为 `B_global(t) - frozen_history_size`，并裁到合法区间；保护 prompt 一致。用相同 scorer 比较 repeated-selection 与 one-time-isolation，再比较完整 StepKV。原文 Appendix D 给出了这一预算原则，可据此复现，不应拿同名 KV 传输系统替代。[方法和预算算法](https://arxiv.org/html/2505.15347v2)

**ChunkKV 的适配边界。** 它能回答“保留普通连续块是否已足够”，但与 SnapKV 一样不能被描述成开箱即用的 agent online 方法。[ChunkKVPress](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/chunkkv_press.py) 的默认调用来自 [BasePress](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/base_press.py) 的 initial-prefill hook；[DecodingPress](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/decoding_press.py) 当前类型接口也不能直接包装 ChunkKVPress。复用 selector、增加 step 触发并披露 scorer/chunk size，标为 agent-step adaptation；若只借用连续块规则，标为 ChunkKV-style control。chunk 应基于原始 global token IDs 保持连续性，不能把剪枝后相邻 slot 无说明地当作原始连续文本。整块保留的预算取整要报告实际使用量，库的 compression ratio 与本仓库 keep ratio 含义相反。

**补充候选 R-KV：审稿人没有直接点名，但原稿相关工作已引用，适合作为开源 reasoning baseline。** 它在 decode 期间按生成 buffer 周期压缩，综合 attention importance 与 key-vector redundancy。官方 HuggingFace 路径提供 Llama/Qwen2 适配，因而比依赖未发布训练权重的方案更可落实。[论文](https://arxiv.org/html/2505.24133v3)、[官方仓库](https://github.com/Zefan-Cai/R-KV)、[模型接口](https://github.com/Zefan-Cai/R-KV/blob/main/HuggingFace/rkv/monkeypatch.py) 仍须测试多 token Observation 续写；若把固定 token interval 换成完整 step，命名 R-KV-step adaptation。预算需包含暂存生成 buffer，且不能把原论文按平均生成长度计算的 ratio 直接等同本仓库逐步累计预算。

**SnapKV 的回应不能只写“prefill 所以不能比”。** R-KV 原论文 §4.1 已明确把 SnapKV 改成每 128 个 decode tokens 压缩一次。因此可以把它降为次优先级，并说明需要 online adaptation；若剩余时间允许，也可做 SnapKV-step 作为附加结果。[已有适配先例](https://arxiv.org/html/2505.24133v3)

**IntentKV 同名问题。** [agent IntentKV](https://arxiv.org/html/2606.09916v1) 的 QueryMemory 与学习残差方法，不是 [ACL IntentKV](https://aclanthology.org/2026.acl-long.1250/)；后者虽有 [公开代码](https://github.com/linkezh/IntentKV)，研究的是 decode 前 prompt cache 压缩。不能跑后者就声称已经比较 R2 所指 agent 方法。缺少 learned head 时，自己实现的 QueryMemory 规则只能叫 Phase-1 adaptation，不能冒充完整 IntentKV。

**本轮确定的执行顺序：H2O/TOVA-step → FlowKV-style → LazyEviction；之后按可运行性从 ChunkKV-step 与 R-KV 中选一项。** 新增的 reasoning/结构比较可覆盖多位审稿人的竞争性疑问，但不能写成已实测完整 SideQuest/agent IntentKV。对 [ThinKV](https://arxiv.org/abs/2510.01290)、[SideQuest](https://arxiv.org/abs/2602.22603v2) 和 [Quest](https://github.com/mit-han-lab/Quest) 给清楚的资源口径与方法区别。

**E3：词面信号稳健性、复用可靠性和超参。**

先修正或确认 self-reuse 等语义，再开展稳健性验证。建议在独立开发集上固定超参，用 held-out 题目完成最终比较。

| 子实验 | 设计 | 报告内容 |
|---|---|---|
| 语义相似度替换 | lexical / embedding / 二者结合；证据新颖性与 reuse 用缓存 embedding，动作相似度保持 tool name 与参数结构可区分 | EM/F1、step 排序相关性、保留 token 重合率、额外时间与显存 |
| 噪声与释义 | 相同题目构造事实保真的 observation 释义、无关噪声、重复检索；验证答案仍可恢复 | 相对干净条件的退化与配对差值 |
| 有用复用 vs 重复检索 | 区分重复文档/同 query 与推进证据链的重用；在可用 supporting facts 或固定人工样本上核验 | 两类步骤的 reuse 分数、预算占比、下游正确率；不能把词面重合直接称为因果证据 |
| 长轨迹陈旧占位 | 选择确实有更长轨迹的任务，按实际完成步数分组；原式对比无 reuse 与带衰减变体 | 旧步骤占 cache 比例、最后一次有效复用距今步数、EM/F1；仅提高 max_steps 不等于得到长轨迹 |
| 参数敏感性 | 覆盖 token/step 混合、reward/reuse 比例、repeat penalty、reuse scale、cue cap、score cap、floor；围绕默认值取少量预先确定水平 | 均值/CI、跨数据集使用同一组参数；明确开发集选择过程 |

衰减可以作为新增变体，例如 c_k(t) = gamma × c_k(t−1) + overlap_increment；不能把这个新机制写成原始 StepKV 已有。

现有 [component ablation](/home/fengboyu/kvmem/run_stepkv_component_ablation.py:47) 和 [weight sweep](/home/fengboyu/kvmem/run_stepkv_stepscore_weight_sweep.py:66) 可以复用，但 weight sweep 默认 alpha=1−beta，未自动包含 beta=0，且和主配置 alpha=beta=0.8 的口径需统一。补端点、报告实际参数及 floor 状态，避免图注与实现不一致。

R4 关于隐式依赖的回应应准确：注意力 token saliency 提供不同于词面 overlap 的信号，可以帮助保留词面不显著的 token；它并不保证识别全部隐式依赖。用 token-only、step-only、融合及释义实验验证，避免把这一互补性写成保证。

**E4：效率与统计——和核心实验同时采集。**

每个方法至少交付：EM/F1、实际 episode 长度、trajectory KV tokens/bytes、含保护前缀的总 KV、剪枝前/后长度、整个运行的峰值显存、prefill/decode 时间、step utility 时间、saliency 收集时间、ranking 与 cache gather 时间、端到端时间。

GPU 测量使用适当同步或 CUDA events，做 warmup，记录设备与软件版本。关闭会影响测量的大量 trace 输出，诊断运行另存。总延迟包含 scoring/cache management；工具等待单列，并同时给含工具与不含工具的口径。不把“没有额外 forward”解释成“没有 attention materialization 或管理开销”。

现有首次生成的 prefill/decode 细分使用总时间的 30%/70% 估算，不能直接作为测量值写进 rebuttal。[当前计时](/home/fengboyu/kvmem/models/QwenLLMWithKVCache.py:485) 当前诊断快照还会把注意力矩阵转成 CPU list，应单独控制这个开销。[快照逻辑](/home/fengboyu/kvmem/kv_cache/pruning_strategy.py:19) 复杂度分析至少包括每轮 O(S×64) 的历史 cue 比较和 O(N log N) 全局排序，另计 metadata 映射、attention 收集及 cache gather；64-cue cap 不会让随轨迹增长的总成本变成常数。

Figure 4 若归一化，明确 `value(method) / value(FullKV)`，附绝对值与单位。Figure 5 需要真正补 latency 面板或修正文案，标明模型、GPU、样本集、单位、统计范围。峰值 GPU 显存减权重不能直接当 KV bytes；缓存张量大小与全程工作显存分开报告。

在相同硬件上完成方法间速度比较；H200 与 A100 可以分担不同实验组，不混用设备得出一个方法的 speedup。固定 transcript 的回放测量适合隔离系统成本，但最终准确率必须来自完整 agent rollout；两类结果分别标注。

统计上复用共同样本 ID 进行配对分析，报告关键差值与 95% CI。F1 可用按题配对 bootstrap，单次确定性 EM 可补 McNemar；多组抽样存在题目重叠时不能当成完全独立重复。明确三组 seed 是抽样重复还是生成重复，不以重复跑三遍 greedy 输出来制造独立证据。关键对照预先确定，CI 不支持的比较如实降低结论强度。

**E5：更强模型与非 ReAct 交互。**

H200 优先用于一个 32B 级模型的兼容性与显存检查，随后运行 FullKV、StepKV、最强可比 baseline；预算覆盖 20% 和 50%，优先一个主 QA 集加一个已有长轨迹任务。若 32B 适配不能及时完成，14B 可作备选。是否“更强”由 FullKV 的实际任务表现支撑，不能仅凭参数量判断。

非 ReAct 验证优先做 function-calling agent：用运行时真实 tool-call/tool-return 事件定义一个完整交互单元，至少有两种工具，并包含噪声工具结果。除去显式 Thought/Action/Observation 字样后，agent 仍可有可观测的工具边界，因此这一实验验证的是对显式文本标记的依赖，并不证明支持任意无边界隐式推理。

尽可能选择真实不同的任务环境，而不是仅换 QA prompt 的格式。当前 WebArena runner 明确复用 QA/ReAct 与本地 BM25，并默认限制 wikipedia 及可 string-match 的任务，不能作为原生浏览器 WebArena 结果。[runner 定义](/home/fengboyu/kvmem/run_all_webarena_experiments_v2.py:1)、[任务过滤](/home/fengboyu/kvmem/run_all_webarena_experiments_v2.py:48) 完全无外部交互边界的纯推理模型应明确为当前适用范围之外，分段方法只能作为后续适配。

**五、10 月 13 日倒排安排**

按实验依赖排期，吞吐和卡数未知，不预先承诺具体 GPU 小时。先用少量非测试样本测单 episode 成本，再估计总耗时。

| 日期 | 主要交付 | 资源安排 |
|---|---|---|
| 9/27–9/29 | P0 协议清单、找回原始结果、表格复算、小轨迹与 FullKV 一致性检查；LazyEviction 多轮适配判断 | CPU 整理，A100 小规模验证；H200 检查大模型适配 |
| 9/30–10/3 | H2O/TOVA-step、FlowKV-style、LazyEviction 首轮；同时采集 E4；有余力选 ChunkKV/R-KV | A100 跑 7B/8B 核心实验，H200 跑较强模型首轮 |
| 10/4–10/7 | E3 稳健性与参数，E5 function-calling/多工具任务，完成较强模型比较 | 两类 GPU 按实验组独立排队 |
| 10/8–10/10 | 必要重复、paired CI、长轨迹与异常样本分析，生成最终表图 | 保留时间重跑失败或协议不一致的关键组 |
| 10/11–10/12 | 按五位 reviewer 写英文回应、核对每个数字来源、压缩到正式字数限制 | 以写作和独立核查为主，停止无关扩展 |
| 10/13 | 最终检查并提交 | 留提交缓冲，不依赖当天新实验 |

最低交付应包括：协议与数值来源修复、H2O/TOVA 同 schedule 与 utility 对照、FlowKV-style 控制、至少一个新增可靠 reasoning baseline、完整效率口径、关键配对统计、主要参数和词面稳健性结果。目标交付再加 ChunkKV/R-KV、较强模型和 function-calling 实验。若资源紧张，先缩减横向数据集组合，保护能够区分贡献的控制实验；不把没有公开 artifact 的完整 agent 方法列为提交依赖。

**六、对应五份审稿意见的逐项回复框架**

| Reviewer / 原问题 | 应如何回答 | 对应证据 / 工作 | 不能提前声称 |
|---|---|---|---|
| R1 W1：schedule 矛盾；no-step 是否纯去掉 step term | 明确完整 step 后剪枝是方法组件，给出 Table 1 的实际时机并纠正文案；新增 H2O/TOVA-step，说明 no-step 保留的条件 | P0、E1；一张方法协议表与 timing 控制结果 | 未核实就说“所有方法原本同 schedule”；把设计本身当作错误 |
| R1 W2：删掉的 KV 能否回来 | 明确存活 KV + 新 token 的候选集，原始 ID 到物理 slot 映射，元数据与 KV 的区别 | P0 在线语义与 trace | 能借 delayed reuse 恢复已经完全丢失的 KV |
| R1 W3：Figure 4/5 与效率边界 | 提供模型/GPU/单位、绝对值、归一化公式，包含步内峰值和全部管理开销 | E4 | post-pruning cache 就是全程 peak |
| R1 W4：复现细节、标准差、21.60 来源 | 列 regex/cue、overlap、attention 聚合、数据/解码/steps，逐表给配置与结果来源 | P0、E4；三组匹配消融 | 尚未追溯就确定 21.60 的成因 |
| R1 W5：新颖性、FlowKV-style、重复检索 | 缩小首创主张，补近邻定位，同 scorer 的一次压缩控制，区分复用代理与真正推进 | E1、E2、E3 | overlap 是真实因果依赖 |
| R2 Q1：SideQuest/IntentKV | 说明公开 artifact/权重不可获得的核查结果及方法区别；补可复现的近期 reasoning baseline 与结构控制 | E2 | 用 ACL 同名方法冒充 agent IntentKV；称已完成完整 SideQuest/IntentKV 对照 |
| R2 Q2：任务多样性、噪声、无清晰边界 | function-calling、多工具与噪声验证；对完全无可观测边界的场景明确限制 | E3、E5 | 增加一个多跳 QA 就等于跨任务泛化 |
| R2 Q3：较强模型 | 一个更大模型的 FullKV/StepKV/强 baseline，报告原始能力与压缩退化 | E5 | 参数更多必然 agent 更强 |
| R3 Q1：旧步骤累积 | 原式无衰减；解释 log/cap 只抑制增长，补长轨迹占位与 decay 变体 | E3 | 有上限就不会保留陈旧信息 |
| R3 Q2：额外运行代价 | 给每步 metadata/similarity、排序、cache gather 的实测与规模变化 | E4 | 无训练/无额外 forward 等于零开销 |
| R3 Q3：参数敏感性 | 覆盖初始化、reuse、融合、floor，而非仅 alpha/beta | E3 | 只展示最优参数或把 test 当开发集 |
| R4 Q1：技术区别 | 聚焦 agent-field utility + 在线更新 + token 融合，分别用 static、no-step、结构控制验证 | E1、E2 | 结构意识或 delayed reuse 本身首次出现 |
| R4 Q2：隐式语义可靠性 | 说明 attention 是补充信号但无保证，用 embedding/释义及 token-only/step-only 对照 | E3 | token saliency 能必然恢复所有隐式依赖 |
| R4 Q3：其他 agent 框架 | 工具事件定义步骤，不依赖文本字段名；说明纯隐式推理范围 | E5 | function-call 边界实验证明完全无边界场景 |
| R4 Q4：各类超参 | 给同一默认配置跨条件表现和敏感性图 | E3 | 把新增 tuning 结果当原实验预设参数 |
| R5 Q1：SnapKV/Quest/agent baselines | 补近期 reasoning/结构近邻，解释 SnapKV online 适配及 Quest 资源口径；SnapKV-step 有余力再做 | E2 | “prefill 所以完全不能比较”；以相同“token budget”混比不同资源 |
| R5 Q2：FullKV/ReAct 崩溃与 schedule | 给两个入口的 prompt/引擎/解码差异和一致性检查；用同引擎 FullKV 重建核心比较 | P0、E1 | 没有检查就归因于模型能力 |
| R5 Q3：全超参与 test tuning | 公开哪些数据用于选参，独立开发与最终评估；如果原先用过测试指标，要如实纠正并重新独立验证 | P0、E3 | 无记录却承诺从未用 test 调参 |
| R5 Q4：embedding 替换与步骤排序 | 报任务质量、ranking/retention 变化和成本，允许出现无收益结果 | E3 | 语义相似度一定提升 |
| R5 Q5：压缩超过 FullKV | 给 paired 的“仅 StepKV 正确/仅 FullKV 正确/都正确/都错误”计数和差值 CI；固定抽样分析证据与干扰 | E4、E3 | “去掉干扰”在没有干预证据时就是原因 |
| R5 Q6：free-form/code agent | 提供实际新框架结果，区分工具边界适配与真正无边界轨迹；未覆盖代码任务则明示 | E5 | 将检索型 WebArena runner 称真实 code/browser agent |
| R5 Q7：扫描开销、runs、显著性 | 给每步开销、真实 run/seed 定义、题目配对 CI，紧预算结果单列 | E4 | 只看误差条是否重叠就断言显著 |

R5 summary 中把 StepKV 描述为“以整步作为 keep/drop 单位”，与 token-level 最终选择有偏差。回复开头可友好澄清：step 是 utility 估计单位，KV 淘汰仍在 token 粒度；当前实现另有 floor，需一并说明。

正式回复每一条建议采用“直接回答 → 证据/数字 → 论文更正位置 → 适用限制”的顺序。新增结果未完成前使用明确占位符，不能写成已经完成的英文实验结论。共性问题复用相同实验编号，避免五份回复出现不同配置或不同数字。

仓库 `custom.bib` 中 LazyEviction、ThinKV、SideQuest 的标题/作者存在旧占位或与正式记录不一致的内容，需按原论文核正；本次仅在方案中提示，未修改引用或方法代码。[当前引用](/home/fengboyu/kvmem/custom.bib:92)
