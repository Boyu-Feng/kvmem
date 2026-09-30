# Rebuttal 后续事项

更新日期：2026-09-30。这里区分当前排队任务和待开发事项；列入计划不代表已经实现或自动启动。最近发布进度见 [状态入口](STATUS.md)。

## 1. 完成已经排队的准确率补实验

1. 继续两条 2Wiki → MuSiQue 队列：FlowKV-style / LazyEviction，以及 H2O-step / TOVA-step。每种方法分别完成 20% / 50%、500 题；新两集共 16 组、8,000 次题目评测。
2. `multidataset_baselines` 整条队列完成后，SideQuest controller 依次运行 2Wiki、MuSiQue 的 smoke 与各 500 题主实验。它当前依赖前序队列，不只是 GPU 是否空闲；若调整依赖需明确登记新调度配置。
3. 每组结束核对完整 ID、逐题分数、真实 KV 淘汰记录与错误日志；再次导出带时间戳的快照，不覆盖已发布快照。

验收：已登记的 24 组 legacy 补实验与 3 组 SideQuest 适配各有明确终态，完整结果和失败分别报告。原 StepKV/FullKV 等原方法不自动重排 500 题，保留用户要求的原表参考。

## 2. 恢复有效的效率证据，并补等字节口径

- 已登记的 7 条件效率队列在 FullKV 阶段因同 GPU 出现其他计算进程失败，完成数 0/7。先定位 maintenance gate 与外部作业的冲突，再安排真正独占的窗口；保留此次失败。
- 原效率范围是 FullKV、StepKV20/50、H2O20/50、TOVA20/50。**它不包含 LazyEviction**；若要回答新 baseline 效率问题，需另登记相应 profiling 条件。
- LazyEviction 当前 28 query-head KV 对比原生 4 KV-head，不能把名义 token 率视作物理显存率。补充 live KV bytes、跟踪状态、返回副本、步内峰值及窗口下限；设计等实际字节预算对照。
- 记录同步后的端到端延迟、prefill/decode、额外 attention/选择/搬运开销。自由生成结果与固定轨迹回放分开，后者用于隔离系统成本。

验收：所有比较在相同硬件与明确工作量下进行；报告实际字节和绝对延迟，不能用队列墙钟时间或名义 20%/50% 宣称效率优势。

## 3. ThinKV / ThinkKV：接入适合普通 LLM 的路线

已有事实：20 道独立 QA 的三类 thought 预检正常完成，跨所有题的共同三峰层为 0；没有分类阈值、TBQ/TBE 主 runner 或 controller，6 组仅为候选。原三类预检及带宽诊断保留。

2026-09-29 复核发现：[论文附录 E.10](https://arxiv.org/html/2510.01290v2#A5.SS10) 针对普通 LLM 使用 **单类 thought、统一 4-bit、达到预算后淘汰**。因此，不应把三峰预检失败解释为所有 ThinKV 路线都不可运行。

下一步优先实现论文单类分支的明确 QA 适配：

1. 写清分段、保护 prompt、动态 token 预算与量化格式，单独命名协议；不将三类失败记录改成“通过”。
2. 实现真实 packed KV、scales/buffers、分段淘汰与解码接口；验证预算、位置、stop/crop 和不压缩控制的一致性。
3. 在独立 smoke 题确认真实存储下降、真实淘汰和可继续生成，再决定是否启动原登记的 6 组候选。
4. 使用软件量化/反量化时，分别计其存储和工作区；未实现作者 CT/PagedAttention kernel 就不声称复现其吞吐。

验收：独立协议、通过机制检查、有实际字节记录。当前没有 ThinKV EM/F1/压缩率结论，也没有自动接入此分支。

## 4. SideQuest 的结果解释与补充分析

- HotpotQA 500 题主实验的主分支 token 保留率、实际删除和无效命令已汇总并保存至 [最新快照](STATUS.md)。仍需补充辅助分支 KV 和时间；不把 20 题 pilot 的保留率当成 500 题结果，也不把终态比例视作全程或峰值显存比例。
- 当前 pilot 有同 decoder 的 20 题 FullKV 对照；500 题扩展只有 SideQuest 单臂。若需要因果性的准确率/速度收益，应另登记同 decoder 的匹配 FullKV 对照。
- 单列 adaptive/untrained 结果，不并入固定 20%/50% 等预算排名。

## 5. 完成 rebuttal 的统计与文字

- 找回并绑定原论文逐题结果、抽样 seed、模型/data revision 和运行环境。找不到时保留为历史表格转录，不制造重复测量或配对证据。
- 对确实同题同协议的比较计算 EM/F1 差值与配对区间；单 seed 不报告跨 seed SD。原表与新增结果分栏。
- 对照审稿意见更新方法定位、预算分母、首轮剪枝、scorer 和效率图注；保留所有负结果和协议限制。
- 词面稳健性、utility/floor 解耦、较强模型等见 [完整规划](../../REBUTTAL_PLAN_2026-10-13.md)，属于后续备选，不能写成当前队列已经覆盖。

R-KV 保持取消、IntentKV 保持暂缓；当前不自动扩展 Llama、32B 模型或新 seed。优先收齐已运行实验与可信的效率/预算证据。
