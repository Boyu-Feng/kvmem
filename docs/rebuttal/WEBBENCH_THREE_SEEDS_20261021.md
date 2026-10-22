# BrowseComp-Plus / WebWalkerQA 三 seed 共同评测

**当前运行入口（2026-10-21 格式修复后）：** BrowseComp 保留
`comparison_20261021_r01`；WebWalker 的全部模型 / 方法统一改用
`comparison_20261021_webwalker_r04`。两份冻结计划由 **seed_order_v4** 共同
调度，seed233 的两数据集演员与评分都完成后才进入 seed42。旧 WebWalker
smoke 保留，不计入新版资格；旧计划的 WebWalker pending 行已被新版取代。
本页较早 v1/v2/v3 启动记录是历史，不能据其恢复旧队列。

本轮按用户要求，在 BrowseComp-Plus 和 WebWalkerQA 上评测 StepKV 与已选
baseline，两个数据集均为每个 seed 100 题、三个 seed。使用现有
Qwen2.5-7B-Instruct 和 Llama-3.1-8B-Instruct，两模型分别比较。全部条件为
新运行，重跑共同 FullKV；不混入历史 QA、ALFWorld 或独立 decoder 的结果。

准备时已阅读 CL 的 `EXPERIMENTS.md` 第 4 节和
`ttcl/universal_experience/SHARED_EVALUATION_20261012.md`。本轮的共同题单、
顺序、模型、推理实现、动作接口和预算写入独立 `plan.json`；方法配置与实际
神经输入另存审计。准备完成、浏览器预检通过和 smoke 通过均不等于正式结果完成。

## 固定范围

| 方法 | 名义非 prompt KV 保留比例 | 每个数据集 / 模型 / seed 的条件数 |
|---|---|---:|
| FullKV | 100% | 1 |
| StepKV | 50%、20% | 2 |
| FlowKV-style | 50%、20% | 2 |
| LazyEviction | 50%、20% | 2 |
| H2O-step | 50%、20% | 2 |
| TOVA-step | 50%、20% | 2 |
| ThinKV 仅淘汰适配（`thinkkv_eviction_only`） | 50%、20% | 2 |
| IntentKV agent 规则适配（`intentkv_rule`） | 50%、20% | 2 |
| SideQuest 未微调适配（`sidequest_untrained`） | 原自适应预算 | 1 |

公共 seed 为 **233、42、3407**。每个数据集 / 模型 / seed 共 16 条件：
**2 数据集 × 2 模型 × 3 seed × 16 = 192 个正式条件，19,200 次正式评测**。
正式运行前，seed233 的完整方法矩阵执行每条件两题 smoke：
**2 × 2 × 16 = 64 个 smoke 条件，128 次 smoke 评测**。
64 个条件都必须完成真实神经推理，全部审计通过后才允许任何正式条件启动。
没有推理的不可访问根页面记录不能使 smoke 通过。

用户随后要求“先跑一个seed全部的 然后再接着跑其他seed”。正式条件因此按
**seed233 → seed42 → seed3407** 分批执行；每个 seed 的 64 个正式条件全部
完成演员执行，且 **64 份官方评分输出全部通过验证** 后，才允许下一个 seed
启动，先形成该 seed 的完整结果表。上述 64 条件 smoke 共同验收仍先于全部
正式条件，题单、预算和评分协议不变。正式条件完成全部请求题目、且将环境
不可用记录在评分中审计为缺失后，可完成调度阶段；这不等于获得完整 accuracy。
演员基础设施失败或评分失败会阻止后续 seed，不自动重跑。

50%、20% 是保留比例，并非减少比例，也不代表各方法实际 KV 字节相等。
SideQuest 单独标为 adaptive，不能当作固定 50% 或 20%。IntentKV 为现有规则
适配，不含训练残差头；SideQuest 不增加微调。各方法的适配边界、辅助打分和
真实 KV 结构保存在冻结方法配置及逐题 adapter audit 中。

## 共同题单与数据来源

抽样仅依据公开数据行位置：`random.Random(seed).shuffle(indices)` 后取前
100 题，顺序也按该排列冻结。每个 benchmark 内所有模型和方法使用同一份
对应 seed 的内容、ID 和顺序。不同 seed 允许有交集，不把它们宣称为独立的
300 道不重复题。各 seed 的两道 smoke 从余下排列中选取，且必须在全部三个
正式集合的并集之外；当前矩阵仅运行 seed233 的 smoke。

| 资产 | 固定来源 / revision | 使用范围 |
|---|---|---|
| BrowseComp-Plus 题目 | `Tevatron/browsecomp-plus` / `144cff8e35b5eaef7e526346aa60774a9deb941f` | test 共 830 题 |
| BrowseComp-Plus 语料 | `Tevatron/browsecomp-plus-corpus` / `b27b02bc3e45511b8b82a13e6f90ce761df726f6` | 全部 100,195 文档 |
| 官方 BM25 索引 | `Tevatron/browsecomp-plus-indexes` / `b3f37f70c33829eb09d04784a54277a31871fd63` | 完整 Lucene 索引 |
| WebWalkerQA | `callanwu/WebWalkerQA` / `c7ab3329a2a4a35ca569fdca369b61a3d1787e5b` | main 共 680 题；不使用 silver 训练集 |

BrowseComp-Plus 三个正式集合并集为 **263** 题；WebWalkerQA 为 **261** 题。
WebWalkerQA 的 seed233/42、233/3407、42/3407 两两交集分别为 11、18、13，
三者交集为 3。原始数据、题单、输入与评分目标分别绑定哈希；不存在按方法
重新抽题、根据答案筛语料、根据可访问性换正式题或复用旧 history ID 标签。

演员只接收公开题目，以及 WebWalkerQA 的公开根 URL。答案、golden path、
`source_website` 注释和 BrowseComp 证据 / negative doc ID 均留在评分侧。
BrowseComp 检索保留官方原始 doc ID、完整语料和来源 URL。

## 共同演员与预算

Qwen revision 为 `a09a35458c702b33eeacc393d103063234e8bc28`；Llama revision
为 `0e9e39f249a16976918f6564b8830bc894c89659`。两模型均使用原生 BF16。
全部方法共用 Hugging Face greedy generate 后原停止裁剪路径；SideQuest 和
ThinKV 使用已有 unified adapter 接入该路径，不沿用历史独立 decoder。
StepKV 原 episode 的 utility / span 记账保留，提示与工具统一适配新 benchmark。

| 项目 | 所有方法共同设置 |
|---|---|
| 单题尝试 | 1 次；失败不自动重抽、重跑 |
| ReAct 步数 | 最多 40 |
| 每生成块 | 最多 256 token |
| 逻辑上下文总预算 | 32,768 token |
| 工具 observation | 每次最多 3,072 tokenizer token，包括 WebWalker 初始根页面 |
| BrowseComp Search | 官方 BM25 top 5，每篇最多 512 tokenizer token |
| BrowseComp Lookup | 仅能查先前返回 doc ID，每篇最多 512 tokenizer token |
| WebWalker Visit | 只能访问已交付给演员的完整可见链接，或返回根页面 |
| WebWalker Search | 搜索已交付的链接文字 / URL；不调用外部全网搜索 |

上下文记账包含实际 chat template 的初始 prefill、增量输入，以及每个已执行
生成块完整预留的 256 token。预算不足时，在下一次神经调用前终止；不能因
KV 淘汰释放缓存而获得更多逻辑上下文。预留量包括停止裁剪丢弃的生成部分，
不等于声称每块实际生成满 256 token。真实输入、生成调用和预算用量均另存。
正常 `finish`、`step_budget`、`context_budget`、`invalid_action` 保留为模型
执行结果；模型异常和基础设施错误单独失败，不能伪装成预算耗尽或零分。

模型权重、tokenizer、chat template、generation config、库版本与源码哈希在
准备和实际加载时检查。每题记录真实神经 token、动作、工具 observation、
峰值显存、实际 KV 和方法辅助计算。各方法学习 / 辅助成本单列；同评测预算
不代表总计算量相等。共享 GPU 环境的用时仅作诊断，不能作为独占延迟排名。

## WebWalker 浏览环境

浏览使用真实 Crawl4AI **0.7.4**、Playwright **1.64.0**、Chromium
**156.0.8078.4**，依赖在独立 Python 3.12 环境冻结。网页采用共同、只追加的
渲染缓存：每个 URL 的首次响应或失败、HTML / 解析结果哈希、时间和 renderer
身份永久保留，后续方法共用该记录。缓存损坏直接拒绝，不重新抓取覆盖。
这是一套共同的单演员 ReAct 浏览适配；不声称复现原 WebWalker 多智能体算法。

根页面只允许依据真实观测到的响应重定向采用规范 URL：路径、query 不变，
同域名或单个 `www.` 别名，协议不变或 HTTP 升级 HTTPS。其他跨域 / 路径
重定向不暴露内容。每个事件记录原 root、effective root、final URL 和快照
哈希。observation 截断后才登记可见链接；藏在截断外或被截断的 URL 不能
通过 Search 恢复或 Visit 猜到。详细边界见
[WebWalker 环境说明](WEBWALKER_ENVIRONMENT_20261021.md)。

2026-10-21 浏览器预检已验证真实 JS 渲染、JS 新增链接的后续访问，并分别
渲染 seed233 两道 smoke 的正式根 URL：`webwalkerqa_00553` 的
`https://www.ironhidegames.com/`、`webwalkerqa_00420` 的
`http://ac.dragonest.com/` 均可访问。该预检有 **0 次演员 / 裁判调用，0 次
正式评测**，不替代方法 smoke。

- 浏览器运行清单：`assets/webwalkerqa/provenance/crawler_runtime.json`，
  SHA256 `49319890d88ae6ad13c2e5cec9876a490615c117cde1e757fc4b4426cf1dba8f`。
- JS / 导航预检：`assets/webwalkerqa/preflight/browser_20261021_075018_540218.json`，
  SHA256 `286b3bd6111ddd4e001f09737d464d924ddb082b5918e6912c82b5786900fb96`。
- 两根页面预检：`assets/webwalkerqa/preflight/seed233_roots_20261021_075156_070945.json`，
  SHA256 `73428516bf48166ddbe6184d49667539a6aac30c4dbc99d401194d3ecade523e`。

以上路径均相对于 `/mnt/checkpoints/bruli/fengboyu/kvmem_webbench_20261021/`。

## 评分与缺失结果

BrowseComp-Plus 保留 vendored 官方 `evaluate_run.py` 的 grader prompt、
parser 和引用指标代码，源码 SHA256 为
`1a21233937c377ab6323c98ff9af67742756a57fbacab4ebf9bc30852eae530a`。
裁判使用 **Qwen/Qwen3-32B** revision
`9216db5781bf21249d130ec9da846c4624c16137`；采样参数按官方设置：
temperature 0.7、top_p 0.8、top_k 20、max_tokens 4096。独立裁判 runtime
与每个输入、提示、原始返回和解析结果均绑定保存。

WebWalkerQA 保留上游 `evaluate.py` 的 `cot_qa` 评分方式，固定 LangChain
0.3.27 的原模板 / parser；上游未固定 LangChain 版本，因此不声称复现其
历史依赖环境。原公开默认裁判为 GPT-4，seed42、temperature0。用户已明确授权
“用 GPT-4.1 评分并明确标注”。本轮因此采用 **GPT-4.1 裁判，实际返回模型
`gpt-4.1-2025-04-14`**，请求模型 `gpt-4.1`，仍固定 seed42、temperature0。
这属于经授权的裁判变更，不声称与公开默认 GPT-4 评分完全相同。

授权原文与时间保存在 `assets/judge/webwalker_authorization.json`；独立合成
qualification 已通过，记录在 `assets/judge/webwalker_probe.json`，不计入正式
实验、未调用 actor。评分前校验 qualification，每个评分 contract 绑定授权内容和实际模型；
`SUMMARY.json`、`RESULTS.md` 明确显示 GPT-4.1 与
`user_authorized_gpt41_changed_judge` 标记。实际返回模型变化时停止评分。

不可访问的 WebWalker 根页面保留 `environment_failure`，其 prediction / score
为缺失，并继续后续题目；不替换题目、不记零。基础设施和裁判错误保留原始
失败与尝试记录，不通过自动重跑选择更有利结果。已完成的裁判返回不可重采样；
WebWalker 网络 / 服务失败的预先声明调用上限为 10，BrowseComp 裁判为 1。

每个正式条件只有完整 100 题均有效计分才报告正式 accuracy；有环境缺失时
单列覆盖率和已评分子集准确率，不据此排名。仅在三个 seed 全部完整时报告
三个 accuracy 的等权均值和样本标准差（分母 n−1）；缺失条件不填零。

## 启动、恢复和记录

运行输出：
`/mnt/checkpoints/bruli/fengboyu/kvmem_webbench_20261021/runs/comparison_20261021_r01`。
所有大资产、环境与结果位于数据盘，不加入代码仓库。计划与源快照冻结后不能
原地修改科学协议；需要变更时建立新版本并保留旧运行与失败。

本轮共同计划和调度器已准备并冻结。下面仅列当前控制器退出后的恢复入口，
已有存活 controller 时不得重复启动。一个演员协调器、两个评分控制器分别
在独立终端执行，每个终端先设置路径并切换到对应冻结源码目录。
v4 依赖两份已有的 `score_queue_contract.json`。实际设备和并发以冻结计划 / 资源准入记录为准；演员与本地裁判
共用容量锁，等待资源不计作任务执行。

```bash
WEBBENCH_PY=/mnt/checkpoints/bruli/fengboyu/kvmem_rebuttal_20260929/venv/bin/python
WEBBENCH_RUN=/mnt/checkpoints/bruli/fengboyu/kvmem_webbench_20261021/runs/comparison_20261021_r01
WEBWALKER_RUN=/mnt/checkpoints/bruli/fengboyu/kvmem_webbench_20261021/runs/comparison_20261021_webwalker_r04
cd "$WEBBENCH_RUN/source_snapshot"

# 评分 controller；在一个独立终端中执行。
CUDA_VISIBLE_DEVICES='' "$WEBBENCH_PY" -B -u \
  -m scripts.webbench_score_queue_v1 queue --plan "$WEBBENCH_RUN/plan.json"

# WebWalker API 评分 controller；独立终端设置同样变量后执行。
cd "$WEBWALKER_RUN/source_snapshot"
CUDA_VISIBLE_DEVICES='' "$WEBBENCH_PY" -B -u \
  -m scripts.webbench_score_queue_v1 queue --plan "$WEBWALKER_RUN/plan.json" \
  --endpoint http://127.0.0.1:8313/v1

# 统一演员调度 controller；独立终端设置同样变量后执行。
cd "$WEBBENCH_RUN/source_snapshot"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$WEBBENCH_PY" -B -u \
  "$WEBBENCH_RUN/scheduler/seed_order_v4/webbench_seed_queue_v4.py" \
  queue --old-plan "$WEBBENCH_RUN/plan.json" --webwalker-plan "$WEBWALKER_RUN/plan.json"

# 只读状态查询。
"$WEBBENCH_PY" -B -m scripts.webbench_v1 status --output "$WEBBENCH_RUN"
```

`plan.json`、`preflight.json` 和 `source_snapshot/` 保存共同协议及源码；
`queue_status.json` 为演员进度入口，`scoring_status.json` 为评分进度入口。
每个条件位于 `jobs/<benchmark>/<model>/<method>/<budget>/seed<seed>/<purpose>/`，
包含 `status.json`、`actual_model.json`、`checkpoint.json`、`predictions.json`、
逐题 `audits/`、`attempts/`、`worker.log`、`launch.json` 及存在时的
`failure.json`。评分位于条件的 `scoring/`，保存 contract、逐题输入绑定、
调用记录、模型身份与 scores。`SUMMARY.json` 和 `RESULTS.md` 展示完成状态
与三 seed 汇总，未完成明确保留 missing。

恢复核验完整有序前缀、task / input / target 内容哈希与 audit 哈希。存活 worker
按 PID 与进程启动时间接管；有失败或未提交的先前尝试时不静默重跑。
正式执行进度以对应输出中的状态文件为准。

## 2026-10-21 启动记录

本轮已冻结 `comparison_20261021_r01`，计划内容哈希为
`a79227d003e92b3736ad0fd64b041e4bf14c15243d47b44d252fc35ecde769ec`；
`plan.json` 文件 SHA256 为
`423749050c7d76236c7891974bdee0c8f94529d5f2cc45c18a02782522627b68`。
`cpu_snapshot_verification.json` 记录从 `source_snapshot/` 实际导入 runner 后，
200 个源码文件、64 个 smoke 方法配置、全部共同题单及正式 / smoke 零交集的
CPU 核验通过；该核验本身没有调用 GPU 演员。

北京时间 **15:59** 已启动演员 controller **PID 3684992**，进程启动标识
`start_ticks=416728126`，工作目录为本轮 `source_snapshot/`。
`actor_controller_launch.json` 保存命令、目录、PID / 启动标识和计划哈希；
`actor_controller.log` 保存控制器日志。启动截面为 **0/64 个 smoke 条件完成、
0/192 个正式条件完成**，首批演员尚在加载；不能据此宣称 GPU 推理或完整
smoke 已验证。GPU7 当时由独立裁判合成预检持有 reservation，资源记录为
`scorer_reservation.json`，预检不计正式评分。

以上记录路径均相对于本轮输出目录。实时演员进度见 `queue_status.json`，
实时评分进度见 `scoring_status.json`；完成统计必须以已核验的逐题提交为依据。

用户追加 seed 分批顺序后，仅停止旧演员 controller **3684992**，保留并接管
当时的六个存活 smoke worker；此时尚无正式条件启动。
`seed_order_change_20261021.json` 保存用户原文、顺序、旧进程身份与接管策略。
独立调度器约束跨 seed 的启动顺序，不修改已冻结 `plan.json` 或演员源码。
v1 接管记录如下，现已保留为历史，不作为当前恢复入口。

调度器源快照 SHA256 为
`9e7899dc00eeb539fd3903a45f7968e9025d27651dbf100e282a9f0a32f26fcc`；
相邻 `scheduler/seed_order_v1/contract.json` 的内容哈希为
`bd4625929465140cacd3d4d564249bf1903b91a74d9d5bca571e58d34d375f7a`。
其 7 项 CPU 测试通过，阶段 gate、接管与失败记录遵循同一 controller 锁；
准备调度器本身没有新增演员调用。v1 调度器曾启动为 **PID 3705828**、
`start_ticks=416778841`；`actor_seed_controller_launch.json` 和
`actor_seed_controller.log` 保存新启动记录与日志。接管首个状态为 smoke 阶段，
9/64 条件完成、0 个正式条件完成、无失败；仍等待完整 smoke 矩阵验收。

评分 controller 已独立启动为 **PID 3698483**、`start_ticks=416760790`，
从同一 `source_snapshot/` 工作；`scoring_controller_launch.json` 和
`scoring_controller.log` 保存启动与日志。该 controller 启动本身不构成正式
评分完成，继续以 `scoring_status.json` 和逐题评分记录为准。

格式修复前采用 **v2**：每个 seed 同时等待 64 个演员条件和 64 份官方评分结果验证，
之后才进入下一个 seed。仅停止 v1 controller **3705828**，保留全部存活
worker；演员、评分源码和科学协议均未改变。当前调度器为 **PID 3716315**、
`start_ticks=416802699`，启动与日志分别为
`actor_seed_scored_controller_launch.json`、`actor_seed_scored_controller.log`。
这份 v2 启动记录现已保留为历史；格式修复后必须使用上方 v4 入口。

`scheduler/seed_order_v2/webbench_seed_queue_v2.py` 的 SHA256 为
`2ec09ecc14414035adc686d4f1ac9d004e2907444bf6dafdcc8e3e0889e2640a`；
相邻 `contract.json` 的内容哈希为
`416bcc9671903e9f013336d743bd05b201683b4853bbbabebb2569cd35c2db7a`。
北京时间 **16:12** 状态为 **12/64 个 smoke 条件完成、0 个正式条件完成、
0 份正式评分完成**，演员与评分失败记录均为空，WebWalker 方法 smoke 尚未
开始。该状态是启动进度截面，不代表正式实验或完整 smoke 已完成。

## 2026-10-21 WebWalker 动作格式修复

用户明确要求修复后继续启动。旧 Llama 的 16 条件 × 两道 smoke 均输出
`Action N: Visit URL`，漏写本轮 ReAct 适配要求的方括号；原始神经输出完整，
不是 stop-crop 丢失。两个 URL 均已交付给演员。重复 BOS 属于既有运行器行为，
没有证据证明它造成此问题，本次没有更改 tokenizer / BOS。

最终采用 `webwalker_action_format_v3`：严格语法优先且保持原行为；严格解析
失败时，接受唯一明确的裸 Visit / Search / Finish 动作。Visit 和 Search 的参数
占一行；Finish 的答案可位于同一行或后续多行。仅去掉参数首尾的格式空白，
保留内部文本、引号、标点和所有原始模型输出，不补猜 URL、不修补答案、
不增加演员调用。多个动作或额外 Observation / Question 等段落不能通过新增
兼容规则。页面可见链接与根站边界继续由同一环境强制检查。

所有 WebWalker 方法、保留率和两个模型共用此版本；题单、题序、模型、提示、
预算、浏览器缓存与 GPT-4.1 官方提示评分均保留。动作执行和审计使用同一
版本解析器，审计同时保存原始命令、规范动作和解析来源，并在验收时复算。

最终计划位于 `comparison_20261021_webwalker_r04/plan.json`，内容 SHA256 为
`1a0afdde2dd1a2069175937ad5602b95541506592d3646914c914f0d3b525e36`。
解析器源码 SHA256 为
`ea76f9b4a145c06f97db259dfd62468ba9f86de58f1ee4f234a3cab73a7db35a`。
计划中 change 的简略文字“single-line”指动作命令行；实际冻结实现允许 Finish
答案跨行，完整语法以上述规则和绑定源码为准，原冻结计划不作原地文字改写。

新计划包含 32 个 smoke 条件和 96 个正式条件；旧 smoke 不能代替新资格。
每条 smoke 的初始动作必须有效，每个条件两题中至少一次成功访问站内页面，
环境失败或越界重定向不能充当访问成功；后续模型失败仍按真实结果保留。
`shared_evaluation_check.json` 保存共同条件核验。解析、执行审计、预算、
真实双快照加载和调度回归共 **206 项测试通过**；CPU 检查不替代真实神经预检。

开发记录完整保留：r02 仅准备、零模型调用；r03 的 Visit-only 修复运行一组
Llama FullKV 的两道 held-out smoke，成功访问后暴露 bare Search 与多行 Finish
也缺括号，两题各达到 40 步后结束。r03 的 367.13 秒诊断推理及完整输出单列
为开发成本，两个中间版本均有 `SUPERSEDED.json`。两者都没有正式评测或
正式评分，没有任何输出或资格复用进 r04。

仅切换协调器，不停止原有 BrowseComp worker。原 v2 PID 3716315、过渡 v3
PID 3880795 和过渡 r03 API 评分 PID 3880302 均已停止并留存启动 / 切换记录。
当前统一 v4 controller 为 **3894616**，r04 API 评分 controller 为 **3894253**；
旧 BrowseComp 评分 controller **3698483** 继续服务既有结果。
调度器源码 SHA256 为
`823b34103d134f207fd358e2037a81eb06d4acf0ccc4a4e725d311200322ac8a`，
调度合同 SHA256 为
`a8e63c8ebf39cc423f690b4b771fd0cf37cbc92970d7e416441a44cf8cce11f2`。

v4 共用原 GPU 4–7 的容量锁，优先安排 Llama 新 smoke，再安排 Qwen 新 smoke；
BrowseComp 正在执行的任务保持原冻结协议。之后两 benchmark 共同遵守
seed233 → seed42 → seed3407，等待每 seed 的 64 份演员和评分结果验证。
当前状态以 r01 的 `queue_status.json` 为准，其中 `route_plans` 标明两个版本；
WebWalker 结果与评分看 r04 的 `RESULTS.md` / `SUMMARY.json`，旧 r01 中的
WebWalker pending 行属于已取代的旧路由，不能当作新版本缺失条件汇总。

## 2026-10-22 中断恢复

用户要求“kvmem实验现在中断了 从之前的断点启动接着跑”。北京时间
**00:03:39** 已重新启动原冻结 v4 协调器 **PID 71294**，
`start_ticks=419633173`。原协调器与八个演员进程均已退出；两条评分
controller 及仍在执行的 WebWalker 评分进程保持运行。

八个中断条件的已提交题数为：BrowseComp Qwen LazyEviction 50% / 20%
分别 **89 / 85**；WebWalker Qwen LazyEviction 50% / 20% 为 **32 / 38**，
H2O-step50 为 **87**，TOVA-step50 / 20 为 **67 / 33**，ThinKV50 为 **4**。
共 **435** 条前缀的题序、内容、plan/job/task 绑定和 audit 哈希均通过核验。
每个条件恰有一条中断且未提交的 attempt，无孤立 audit。

此次按用户恢复指令登记一次性例外：保留并重新执行上述八道未提交题目，
不重新执行已提交答案，不重新采样已完成评分。现有断点仅到题目边界，
不能恢复题内 KV；这一例外单独记录，不声称中断题仅有一次进程尝试。
原 attempt、launch、日志、status、actual_model、checkpoint 副本及哈希
在变更前归档；仅移走原位的八份未提交 attempt 和八份旧 launch，
冻结计划、源码、原 checkpoint、已提交 audit 与共同网页缓存保持原样。

两条 BrowseComp LazyEviction 日志记录了 `unknown parameter type`，
随后出现 `CUDACachingAllocator.cpp:3902 invalid device pointer` 原生崩溃；
其余六条没有终态记录。根因尚未确定，不将这些错误解释为模型零分。
恢复前 GPU 4–7 的 BF16 CUDA 矩阵运算检查通过，原资源准入及 seed
233 → 42 → 3407 的演员、评分共同门控继续生效。

完整恢复证据位于 r01 输出的 `recovery/resume_20261021_1602/`：
`recover.py`、`manifest.json`、`before/`、`jobs/`、
`controller_launch.json` 和 `controller.log`。恢复脚本只供此次使用，
已经登记 manifest 后拒绝再次执行；后续恢复须先核对实际进程与新断点。

北京时间 **00:11** 启动核验：八个原中断条件均由新进程在 GPU 4–7
运行，435 条原前缀保持一致，已新增 **15 条提交**（12 条真实推理结果、
3 条按原协议保留的环境缺失）。两条 BrowseComp LazyEviction 已继续多步
生成，尚未提交当前题；恢复后的日志未复现上述运行时错误。演员与评分
失败记录为空，当前仍为 seed233，正式演员条件完成 **16/64**、评分完成
**14/64**；完整核验见该恢复目录的 `launch_verified.json`。

## 2026-10-22 第二次中断恢复

用户再次要求“又断了 接着从断点启动”。八个演员日志停止于北京时间
**00:51:07–11**；旧协调器 PID 71294 也已退出。最新完整前缀为：
BrowseComp LazyEviction50 / 20 **99 / 94**、ThinKV20 **18**；
WebWalker LazyEviction50 / 20 **52 / 60**、TOVA20 **93**、
ThinKV50 / 20 **55 / 35**。共 **506** 条已提交结果绑定及 audit 哈希
核验通过，无孤立 audit，每个条件恰有一道未提交题目。

两个 BrowseComp LazyEviction 再次出现 `unknown parameter type` 和
`invalid device pointer`；BrowseComp ThinKV20 另记录
`malloc(): unsorted double linked list corrupted`。其余五个 WebWalker
进程在生成期间退出。根因仍未确定，原始错误和前次恢复证据继续保留。
本次 GPU 4–7 的独立 BF16 CUDA 检查通过。

WebWalker FlowKV20 / H2O20 的评分另外因 HTTP 503 停止，分别已有
**83 / 35** 条 items（含 **13 / 4** 条环境缺失），各只有下一题一次
失败调用，尚余原定 10 次预算中的 9 次。全部已完成 items 和原始 calls
保留并复用；只归档阻止重新入队的旧 queue_failure、queue_launch 和
failures 目录，失败调用仍计入原预算。独立合成请求已收到固定
`gpt-4.1-2025-04-14` 的正常返回，未改写共同 judge probe 或评分协议。

本次独立证据目录为 r01 的 `recovery/resume_20261021_1725/`；
演员的八道未提交题目按本次明确恢复指令从题首执行，已提交结果、
冻结计划、源码、网页缓存及 seed233 → seed42 → seed3407 门控保留。

北京时间 **01:30:22** 启动新的 WebWalker 评分 controller **111430**
（`start_ticks=420153502`），**01:31:11** 启动原冻结 v4 演员协调器
**111750**（`start_ticks=420158355`）。BrowseComp 评分 controller
**3698483** 保持运行。启动记录分别为本次恢复目录的
`scoring_controller_launch.json` 和 `controller_launch.json`；原评分
controller 及两个失败条件的完整评分目录均已归档，旧 calls 中的失败
请求仍计入原尝试预算。

北京时间 **01:36:49** 恢复核验：八个 worker 均在 GPU 4–7 存活并继续
生成；原 **506** 条结果前缀及已提交题目的 attempts 保持一致，八道中断
题目的旧 attempts 已原样归档，新 attempts 与原 task 精确绑定。已新增
**3 条真实推理提交**（WebWalker ThinKV50 / 20、BrowseComp ThinKV20
各一条），新增 audit 哈希和 plan/job/task 绑定通过核验。恢复后追加
日志未发现新的原生运行时错误；两项 BrowseComp LazyEviction 当前题仍
在生成。实时队列的演员、评分失败记录均为空，仍为 seed233，正式演员
完成 **18/64**、评分完成 **16/64**。

WebWalker FlowKV20 评分已由 **83** 推进至 **98** 条 items；H2O20 保留
**35** 条，等待串行评分。独立评分复核确认原 **118** 个 items 和 **103**
个 calls 字节哈希不变，既有裁判结果没有重采样。完整启动核验及评分
证据分别见本次恢复目录的 `launch_verified.json` 和
`scoring_resume_verified.json`。实验继续运行，此核验不代表全部条件完成
或先前崩溃根因已修复。

## 2026-10-22 跨服务器迁移快照

用户要求上传最新结果并在另一台不能访问原共享目录的服务器续跑。
北京时间 20:46 的独立审计确认原推理、评分进程均已退出，最新原始
日志停止于 01:52:43–55。正式结果共 **2,415** 条（BrowseComp 1,239、
WebWalker 1,176），**20/192** 个推理条件完成、**18/192** 个评分条件
完成；64 个 smoke 条件全部完成，共 128 条。八个中断条件的完整前缀为
BC LazyEviction20 / ThinKV20 / IntentKV50 **96 / 28 / 15**，
WW LazyEviction50 / 20、ThinKV50 / 20、IntentKV50 **60 / 68 / 82 / 59 / 7**。

全部 2,543 条推理记录、1,826 条评分 items、1,115 份裁判调用的绑定
核验通过。WebWalker TOVA20 评分保留 26 个 items 及下一题已请求但
尚未返回的调用；迁移保持原调用预算。完整审计、加密恢复分卷及新服务器
操作步骤见 [迁移说明](WEBBENCH_MIGRATION_20261022.md)，公开的数字结果
见 [最新结果快照](../../reports/latest_20261022/README.md)。本机原始结果
和冻结计划保持原样，目标机的 A100 调度与 CPU 卸载另行登记，不把旧机
状态中的 `running` 当作存活证据。
