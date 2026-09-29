# Rebuttal 协议与结果解释

本轮为 `restored_original_evaluator_v1`：恢复 2026-09-27 修改前本地代码快照中的九个核心文件。冻结哈希见 [评测模式记录](../../REBUTTAL_EVALUATION_MODE.json)，恢复不等于已证明该快照就是产生送审表格的精确环境。

## 模型、数据与预算

- Qwen2.5-7B-Instruct，抽样 seed233；每个主实验 500 题，最多 7 个 agent 步骤。沿用各入口已有生成与 stop/crop 规则。
- HotpotQA 主实验使用 seed233 排序后的前 500 题；第 501–502 题用于 smoke；SideQuest pilot 第 503–522 题；ThinkKV 预检第 523–542 题。
- 2Wiki 与 MuSiQue 为固定 revision 的开发集：12,576 / 2,417 题，经原仓库 loader 归一化；各取 500 道不同 ID。revision、文件哈希见 [资产记录](../../config/rebuttal_assets_20260928.json)。
- 每个预算的方法先通过对应 2 题 smoke，再进入主实验；smoke 不计入主结果。不以主实验分数调参。
- 固定预算档位是名义的非 prompt 轨迹 token 保留率 20% / 50%；prompt 单独保护。窗口下限、首轮策略、不同头布局会改变实际字节量与峰值。

## 方法范围

| 方法 | 本轮实际实现 | 必须保留的限制 |
|---|---|---|
| FlowKV-style | 冻结已压历史，只对新完成块选择 token | 机制适配，不声称官方实现复现 |
| H2O-step / TOVA-step | 保留原首轮行为，仅延迟后续剪枝到生成与 stop 裁剪完成后 | scorer、窗口、decoder 与其他方法仍有差异 |
| LazyEviction | 真 query recurrence、逐层逐 query-head 选择；W=175、alpha=1e-4 | KV 由 4 头展开至 28 头；同 slots 为原生 GQA 的 7 倍 K/V payload；另计元数据、副本、暂存窗口 |
| SideQuest-inspired | 未微调 Qwen，独立 decoder，每 4 次工具交互请求辅助删除决策 | 自适应预算；不使用官方训练权重；辅助分支开销与无效命令必须报告 |
| ThinKV / ThinkKV | 仅完成三类 thought 校准预检 | 无压缩主实验；普通 LLM 单类分支尚未实现，校准未通过不等于整个方法不适用 |

此前显示的 LazyEviction 20%/50% 准确率是 **slot 档位**下的结果，不应直接用于“相同显存胜过其他方法”的结论。需要补实际 byte 预算对照。

## 统计和效率

原表为作者提供的多 seed 汇总，本轮新增结果为 seed233 单次运行。原表逐题文件与环境未溯源前，不将两者拼成配对显著性比较；也不把停止的 `controlled_v2` 数字替换原表。

只有状态为 `complete` 且样本数量、ID、checkpoint/results 一致的条件进入最终表。运行中条件只作进度记录。基础设施失败、无效运行和待测条件保留状态，不补零分。

现有队列时间是诊断值：GPU 非独占、轨迹长度不同、日志和 sidecar I/O 均可能影响计时。独立效率补测因同卡其他进程而失败，当前没有有效效率结论。`scoring_time=0` 不代表 LazyEviction 无评分成本；其 recurrence/selection 在 attention 的 prefill/decode 内执行。

KV 张量 payload、跟踪元数据、返回副本、临时 attention/反量化工作区和 CUDA allocator 峰值需分开报告。减去模型参数量后的峰值显存不等于 KV 大小，辅助分支内存也不能忽略。

## 历史与暂停范围

R-KV 已被用户取消并由 SideQuest 适配替代；IntentKV 暂缓；更大模型与额外 seed 当前不在运行范围。详见 [后续计划](NEXT_STEPS.md)。冻结文件、失败记录和源哈希均保留，未来改变推理机制需另用输出目录和协议标识。
