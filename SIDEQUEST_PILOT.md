# SideQuest 部分机制复现

论文：[SideQuest: Model-Driven KV Cache Management for Long-Horizon Agentic Reasoning](https://arxiv.org/abs/2602.22603v2)。2026-09-28再次核查论文入口、GitHub和Hugging Face，未定位对应官方实现及训练权重；这不是不存在任何私有实现的断言。

本轮实现推理机制的未微调原型：每4次工具交互触发辅助分支；分支读取同一时刻的上下文及可用cursor，输出删除ID；在主分支的生成边界应用删除。辅助分支的提示及输出不加入主上下文。实际移除所选工具调用与响应的K/V，保留prompt和其他token，并继续使用逻辑位置解码。参见论文Algorithm 1及第3节。

论文采用专门微调的gpt-oss模型、训练数据合成和SGLang服务。本原型使用现有Qwen2.5-7B及显式提示，没有复现论文训练、权重、检索环境及生产服务效率。Python线程的并发任务不代表已经实现论文的低开销GPU并行。

实现：`scripts/sidequest_runtime.py`；入口：`scripts/run_sidequest_pilot.py`。独立pilot的两个实验臂使用相同greedy解码器、逻辑RoPE和cursor标记，分别启用/禁用辅助管理。复用 `models/RebuttalLLM.py` 中的解码基础，不启动已停用的controlled队列。它与原legacy评测在解码细节上不同，两个体系的结果不得合并。

默认在HotpotQA seed233的第503–522题运行20题配对，共40个episode；与500题主实验及501–502题smoke不重叠。模型输入没有gold答案或未来轨迹。结果写入 `results/rebuttal_20260928/sidequest_pilot_v1/`，保存逐题轨迹、每次辅助原文、JSON解析状态、删除ID、K/V删除前后字节数、主分支峰值和辅助分支峰值。无效命令保守处理为空删除，并单独计数；推理错误直接失败。

FullKV与原型均允许自由生成，报告小样本EM/F1及终止状态；main KV不含辅助分支，不能用main KV下降宣称整体峰值下降。episode时间包括等待未完成的辅助分支，仍只用于诊断。

验证包括：拒绝不存在/重复/非整数cursor；prompt保护；fork共享前缀及分支追加隔离；按全局ID删除后保留其他K/V；真实小型Qwen模型删除中间缓存后继续forward。运行：

```bash
.runtime/rebuttal/bin/python -m pytest -q tests/test_sidequest_pilot.py
```

本结果应标记为 `SideQuest-inspired, untrained Qwen pilot`，不能作为官方SideQuest准确率或效率复现结果。

用户随后要求“取消rkv，改成sidequest”。R-KV已从三个数据集的所有待运行任务删除。替代补测按未微调适配版准备，单列在 `results/rebuttal_20260928/sidequest_adaptive_v1/`：HotpotQA、2Wiki、MuSiQue各500题、自适应预算，共3个主实验及各2题的smoke。原20%/50%预算不强加到SideQuest上。

持久队列为 `scripts/run_sidequest_queue.py`，worker为 `scripts/run_sidequest_baseline.py`。队列先要求20题pilot成功完成且发生真实cursor删除，再等待原baseline队列完成。HotpotQA接在 `legacy_baselines` 后，另外两个数据集接在 `multidataset_baselines` 后；使用GPU3并检查可用显存。失败直接记录，不自动混合不同源码检查点。

这3组仍使用独立原型解码器，不能作为与legacy逐项对齐的官方SideQuest baseline。队列、manifest、摘要和目录名均保留 `untrained` / `adaptive` 标记。若改为先训练，需另设训练与评测协议、输出目录及对照。
