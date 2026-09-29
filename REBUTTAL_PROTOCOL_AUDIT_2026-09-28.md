# controlled_v2 与修改前本地代码的差异核查

核查日期：2026-09-28。用户指出送审表格中 Qwen2.5-7B、HotpotQA、20% 的 H2O 为 EM 1.53±1.21 / F1 3.52±1.01，而本轮 H2O-step 为 25.20 / 35.19。

**结论：确实改变了执行代码与实验协议，不只是增加 step 剪枝触发。当前数字不是送审 H2O 的直接复现，不能把全部差距归因于剪枝时机，也不能据此认定旧结果错误。** 此前对结果的说明没有充分交代这些差异；“沿用原程序首轮不剪枝”的说明对 H2O/TOVA 有误。

## 核查依据

比较 `results/rebuttal_20260927/source_snapshot/before_changes.tar` 与当前实际入口。快照中的 `models/QwenLLMWithKVCache.py`、`kv_cache/kv_cache_manager.py`、`kv_cache/h2o_scorer.py`、`token_tracker.py` 与当前同名文件逐字相同；`pruning_strategy.py` 仅增加关闭调试分数快照的开关。旧类 SHA256：`2b0403e9f4eab546a44d7a52d84c1ad7ed2e654a17e7781b348798b47e69c5ad`。

但是本轮新增了 `models/RebuttalLLM.py` 和 `run_rebuttal_schedule.py`，并通过 `run_all_wiki_experiments_v2.py` 的 `llm_class` 参数切换到新类，FullKV 也使用新类。因此“旧类文件没改”不等于“实验执行代码没改”。实际注入入口见 [run_rebuttal_schedule.py](run_rebuttal_schedule.py#L139)。

## 已确认的执行差异

| 项目 | 修改前本地代码 | controlled_v2 |
|---|---|---|
| H2O/TOVA 首轮预算 | 首轮生成结束后立即循环剪枝到预算 | 所有方法首轮均跳过剪枝；包括新的 token 对照 |
| 后续 H2O 剪枝 | observation 后及逐 token 检查预算 | H2O-step 改为完整生成块结束批量剪枝；H2O-token 保留后续逐 token 检查，但使用新引擎 |
| StepKV attention 来源 | 默认 piggyback，复用 observation prefill attention；不可用时可能 fallback | 强制 scoring_forward，另做 full-cache EOS probe |
| RoPE 位置 | 相关 forward 未显式指定逻辑 position_ids，generate 使用当前物理 cache 长度 | 显式按累计 logical token 位置指定 position_ids |
| 解码 | 首轮所有方法及后续 StepKV/FullKV 使用 model.generate；H2O/TOVA 后续使用手写循环 | 全部使用 raw-logit argmax；不应用 repetition penalty |
| Stop 与缓存 | 先生成完整块再截断文本和缓存 | 每 token 检测 stop、按逻辑 ID 回退、重新写入共享 BPE token 中有效字符 |
| 最后输出 token | generate 路径按返回序列更新长度，最后选中 token 可能尚未 forward 入 KV | 每个保留输出 token 都 forward 写入 KV，并检查实际张量长度 |

首轮差异见 [旧首轮执行](models/QwenLLMWithKVCache.py#L553) 与 [新首轮跳过](models/RebuttalLLM.py#L126)。StepKV attention 差异见 [旧默认配置](run_all_wiki_experiments_v2.py#L1221) 与 [新覆盖配置](run_rebuttal_schedule.py#L45)。新位置与停止逻辑见 [RebuttalLLM](models/RebuttalLLM.py#L71)。

当前下载的 Qwen 权重 generation_config 中 repetition_penalty=1.05；旧 generate 路径没有覆盖该值，新手写循环有效值为 1.0。旧 generate 已明确 do_sample=False，不能将差异描述为“旧版随机采样、新版贪心”。尚未确认送审时使用的模型 revision/config。

H2O/TOVA 在修改前已使用 scoring_forward，旧 EOS probe 已 deepcopy 缓存以避免修改 live KV；不能将其表述为本次首次加入 EOS probe 或首次避免 probe 污染。

## 没有改变的主要设置

- 本轮该组 cache_ratio 仍为 0.2，不是改成 0.5。预算分母按累计 logical nonprompt tokens、保护初始 prompt 的原则在快照中已存在。
- H2O 的近期保护窗口为 32，TOVA/StepKV 为 0，沿用快照默认值。近期保护可形成预算下限；20% 不表示整个运行期间的每个 token 边界都只有总 KV 的 20%。
- StepKV alpha=beta=0.8，reward/citation/repeat 权重为 .85/.15/.3，poolwise 分支实际最低保留比例为各 step 当前存活 token 的 10%，均沿用原实现。
- 新增的 self-reuse 排除开关在本批为 False，没有启用。
- H2O/StepKV saliency 使用最后三层，TOVA 使用最后一层。现有 manifest 的 estimator 简写“last three layers”对 TOVA 不精确；为保留运行记录，不原地重写 manifest。

## 结果来源与解释边界

本轮完整结果是同一 seed=233、同一组 500 题的新协议对照；送审截图为多 seed 均值与标准差。截图 Qwen HotpotQA 的 FullKV 为 26.40 / 36.16、StepKV 20% 为 18.07 / 24.29；本轮分别为 21.60 / 32.17、25.40 / 33.47。结果变化不仅出现在 H2O，不能用 seed 差别或单项代码变化替代溯源。

before_changes.tar 是本轮动手前的本地 checkout 快照，尚未证明它就是生成送审表格的代码版本。未找到旧表对应的逐题 sample IDs、模型/数据 revision 与实际依赖 manifest；旧下载脚本未固定 revision。当前 manifest 的 Git commit 只是 checkout 基点，实际新增代码以 source_hashes 为准。旧 requirements 也不能作为送审实际运行环境的证明。

要解释各项改动的因果影响，需要先对齐原结果的代码、环境和样本，再在同一 seed=233、同一批题目上逐项改变条件。当前核查是静态源码与运行记录比较，没有完成这些逐项配对实验，因此不能声称某项修改解释了多少 EM/F1。

本次核查只更正文档并保留证据；没有改写已运行任务的源码、manifest 或结果文件。
