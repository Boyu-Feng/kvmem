# 可复现性问题：中文回复与核对记录

核对对象：用户提供的 [StepKV arXiv v2](https://arxiv.org/html/2609.22158v2)，以及本地恢复的 before_changes 代码。论文表号已对应；本地旧 tex 不是本次论文的准确配置依据。arXiv源码包中的三次运行汇总表已找到；原逐题输出和完整manifest仍未找回。

## 先明确表格对应

- Table 2：MuSiQue 上 step score 子信号消融，Qwen Full StepKV 为 7.00 / 12.49；Llama 为 4.80 / 8.93。其他行是相对 Full 的百分点变化，不是绝对 EM/F1。
- Table 3：三个数据集上的 token score / step score 消融；Qwen Full 为 Hotpot 21.60 / 29.96、2Wiki 23.60 / 26.85、MuSiQue 7.00 / 12.49。
- D.1：明确写了三个 seeds（233、42、3407），且所有结果 mean ± std。
- 两个消融表没有给 std，也没有写确切 rho。源码包的 `three_run_time_result.tex` 显示：Qwen 的全部消融 Full 行等于 run2 / 50% 行；Llama 也对应 run2 / 50%，但 Table3 的 2Wiki F1 写23.90，run2表写23.91。主表 StepKV 的均值/样本SD可由三次汇总值重算吻合。
- 本地运行脚本将run2映射到seed42；component ablation也默认seed42、rho=.50，并导入run2 Full。这是单seed消融口径的强来源线索，不能据此宣称其余消融变体的三seed结果都已找到。
- 因此差异解释应围绕“run2单次Full vs三次汇总”，而不是声称表2/3使用不同keep ratio导致差异；仍需核对消融变体与导入Full的完整配置是否匹配。

源码位置：`results/rebuttal_20260928/provenance/arxiv_2609.22158v2_source/source/three_run_time_result.tex:59`（Qwen）及`:71`（Llama）。这些是文稿汇总数值，不是原始逐题实验日志。

## 可写入回复的技术说明（以当前恢复代码为依据）

感谢审稿人指出可复现性描述中的缺失。我们将补充 cue 提取和评分的完整实现规则。Cue 由确定性词法匹配得到：先将文本转为小写，再按 `[a-z][a-z0-9_-]{3,}|\d+` 从左到右扫描，移除固定停用词、按首次出现去重，并保留前 64 个不同 cue。数字也计入这 64 个名额。步骤 cue 从 Observation 正文与 Thought 的拼接中提取，Observation 在前；不使用额外的实体识别模型或语义编码器。

设当前步骤 cue 为 \(A_s\)，此前步骤 cue 的并集为 \(U_{<s}\)，新颖性为

\[
\mathrm{nov}_s=|A_s\setminus U_{<s}|/\max(1,|A_s|).
\]

重复度为经过小写化、首尾去空白和内部空白合并后的 action argument 在先前步骤中出现的次数。它不是 action 字段的两两 overlap 最大值。用于 reuse 更新的 overlap 为

\[
\mathrm{overlap}(A,R)=\frac{|A\cap R|}{\max(1,\min(|A|,|R|))},\qquad
\Delta c_s=\min(1,1.5\,\mathrm{overlap}(A_s,R_t)).
\]

这里 \(R_t\) 来自本轮待输入的 Observation 及 Thought 前缀。无交集时不更新；这不是 Jaccard 相似度。当前循环也包含刚注册的步骤，因此不能描述成严格排除自身的跨步骤引用。正文和附录中的证据增益、重复度以及 reuse 定义需要按实际产生结果的实现统一。

Token saliency 使用该次评分调用提供的 attention，对 query 维和 head 维求和、对最后三层求平均：

\[
T_i=\frac{1}{3}\sum_{\ell\in\mathcal L_{\mathrm{last3}}}\sum_h\sum_{q\in Q_{\mathrm{provided}}}A^{(\ell)}_{h,q,i}.
\]

实现没有再对 \(T_i\) 做 min–max、z-score、按 query 数除法或按 step/pool 归一化。默认优先复用 observation-prefill attention；不可用时沿用额外 EOS scoring forward。不能声称每次都只有同一种 query 来源。Step utility 采用 reward clipping、\(\log(1+c_s)\) 和外层 clipping 控制尺度，而不是与 token 分数一起归一化。最终按 \(P_i=\alpha T_i+\beta S_{s(i)}\) 组合。

我们还将逐表列出数据版本、split、实际完成样本数、抽样 seed、模型与生成配置、最大步数以及精确 trajectory keep ratio。Table 2 和 Table 3 的 Full 行可溯源到 \(\rho=0.50\) 的run2结果；其与Table 1三次汇总的统计口径不同。各消融变体的预算、样本和完整配置将逐项核对，并在表注中明确。表中消融变化应定义为同一配置下 \(\Delta=M_{\mathrm{ablated}}-M_{\mathrm{full}}\)，单位是百分点。

## 标准差段落：只能按实际记录选一种

**若 Tables 2–3 确有三个 seed 的完整结果：**

“我们将在 Tables 2–3 补充三个 seed 的均值与标准差，并报告每个消融项在同 seed 下相对 Full StepKV 的差值。对于差值的不确定性，我们先计算每个 seed 的配对差值，再跨 seed 汇总，不通过两个标准差相减得到。具体数值为【待补】。”

**若 Tables 2–3 实际只有一个 seed：**

“Tables 2–3 为固定 seed【待确认】的消融结果；D.1 对所有实验均为三 seed 的概括不准确，我们将修正其适用范围，并在表注中明确单 seed、实际样本数及 keep ratio。单 seed 不能估计跨 seed 标准差，因此不补写虚假的 ± 值。Table 1 的多 seed 汇总与消融表的单 seed Full 结果将明确区分。”

用户当前只授权一个 seed；本次没有启动任何额外 seed。不能把当前 seed233 补测与原表结果拼接成三 seed，也不能用题目级 bootstrap 误充跨 seed 标准差。

## 配置证据与未解决差异（内部核对，不直接当成原运行事实）

| 项目 | 当前可验证事实 | 原表状态 |
|---|---|---|
| HotpotQA | distractor validation，7405 中 seed shuffle 去重取500；7步 | 当前恢复运行可验证；其他旧 seed 原输出未找到 |
| 2Wiki | 默认本地 dev.json，500题、7步；下载fallback可选其他split | 原实际split/样本ID待记录确认 |
| MuSiQue 主实验 | 默认 answerable validation/dev，500题、12步 | 原实际参数待记录确认 |
| MuSiQue 消融 | component脚本调用Hotpot基类，未设MAX_STEPS，默认7步 | 是否实际使用默认值未确认 |
| BrowseComp | runner默认100题、40步，具体split由文件/HF参数决定 | 不能泛写500题或统一7步 |
| 消融配置 | component默认seed42、rho=.50、N500；Full默认从run2导入 | 源文件run2/50%行已匹配；变体逐题记录未恢复 |
| 导入Full | 不核验原Full的seed、sample IDs、完整config | 必须核对来源，不能假设天然匹配 |
| 权重 | 论文Table5为(.2,.8)，主runner为(.8,.8) | 须确定哪个对应原表；不要为迎合文字而改正在跑的代码 |
| 解码 | batch1、do_sample=False、每次Thought/Action最多256新token；按EOS停止，stop strings在生成后裁剪 | 原模型generation_config/revision待对应 |
| 当前Qwen生成配置 | generate路径repetition_penalty=1.05；H2O/TOVA后续manual循环直接argmax，没有该processor | 不应笼统写所有方法完整生成参数完全相同 |
| Saliency | head/query sum、last3 layer mean；无额外归一化 | Appendix C.1 Eq16的归一化假设不能当作已实现操作 |
| 新颖/重复/reuse | union novelty、argument重复次数、formatted输入overlap且含新注册步骤 | 与论文§5.1中pairwise最大相似度、纯previous-observation更新不完全一致 |

另需核对Table1的TokenSkipping标准差：例如源码中Qwen/50%/Hotpot EM三次为10.40、8.60、7.40，均值8.80，样本标准差约1.51，而样本方差为2.28；主表写8.80±2.28。不能擅自据文稿数值修改原实验结果，但应回查计算脚本，避免把方差标成标准差。这与“表2/3缺标准差”是不同问题。

代码入口：`kv_cache/anchor_utils.py`；`run_all_wiki_experiments_v2.py` 的 1524、2395、2452 附近；`kv_cache/h2o_scorer.py:47`；`kv_cache/pruning_strategy.py:867`；`run_stepkv_component_ablation.py:103,441,723`。

固定停用词（52个）：

```text
thought action observation search lookup find information invalid could
would should about which what where when from with this that then than
into have has been were also there their them they because movie film
series director actor american british the and for was are who after
before had not but finish
```
