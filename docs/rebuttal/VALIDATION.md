# 发布验证记录

## 2026-10-01 完整主实验结果

验证对象：[05:39:53 UTC 快照与横向结果表](../../reports/rebuttal_20261001_053953_UTC/RESULTS.md)。本次更新导出器以生成数据集为行、方法为列的 Markdown 总表，并归档全部主实验指标；原始实验输出保持不变。

- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .runtime/rebuttal/bin/python -m pytest tests/test_rebuttal_publication.py -q`：**3 passed**。
- **27/27 组、13,500 条记录**均已完成；每组为 500 个唯一 ID，与 manifest 登记集合完全一致。
- 从逐题 JSON 独立重算全部 EM/F1，与机器可读汇总一致；逐一核对总表 15 个方法–数据集单元格中的预算与分数，覆盖全部 27 组。
- 导出时核对 checkpoint 与最终 results 的 ID/EM/F1；导出后复核 **242 份源文件 SHA256 和文件大小**，均未变化。导出器代码哈希与快照登记值一致。
- 快照包含 **114 个文件、12,894,626 bytes**，最大单文件 610,109 bytes；JSON 解析、相关 Markdown 相对链接及 `git diff --check` 通过。
- 保存 SideQuest 三个数据集的最终汇总，保留 ThinkKV 校准未通过和效率测量失败状态；未启动新的 GPU 实验。

## 2026-09-30 结果快照

验证对象：[05:16:36 UTC 快照](../../reports/rebuttal_20260930_051636_UTC/README.md)。本次只读取运行结果，生成新快照并更新发布文档；实验 runner、adapter、队列配置和原始输出均未修改。

- 禁用 CUDA 运行 `pytest tests/test_rebuttal_publication.py -q`：**3 passed**。
- 从快照逐题 JSON 独立重算 **10,206 条记录**的 EM/F1，与汇总一致；**20/27 组完成**，完成组均为 500 个唯一 ID，且与 manifest 登记集合完全一致。
- 导出时逐组核对已完成 checkpoint 与 results 的 ID/EM/F1；未完成结果保留 `partial_not_ranked` 标识。
- 新增保存 HotpotQA 500 题 SideQuest 保留率汇总；其 provenance 中 results 和 manifest 的 SHA256 与本次捕获的源文件哈希一致。
- 快照共 **95 个文件、10,015,505 bytes**，最大单文件 610,109 bytes；`git diff --check` 通过。

以下为上一次发布的历史验证记录，未在本次重复执行其全部测试。

## 2026-09-29 代码与结果发布

验证日期：2026-09-29。验证对象为本轮新增 rebuttal 代码、冻结 adapter 包和 [03:09:50 UTC 结果快照](../../reports/rebuttal_20260929_030950_UTC/README.md)。全部测试禁止 CUDA，未占用 GPU 启动模型实验。

## CPU 测试

使用本轮 `.runtime/rebuttal/bin/python`，`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`：

| 对象 | 命令范围 | 结果 |
|---|---|---:|
| 原有本轮测试 | 仓库根目录 `pytest tests -q`，新增发布测试写入前 | 41 passed |
| 新增发布测试 | `pytest tests/test_rebuttal_publication.py -q` | 3 passed |
| FlowKV-style | 重建目录 `flowkv_source/` 内 `pytest tests -q` | 9 passed |
| LazyEviction | 重建目录 `lazy_source/` 内 `pytest tests -q` | 26 passed |
| H2O-step | 重建目录 `h2o_step_source/` 内 `pytest tests -q` | 9 passed |
| TOVA-step | 重建目录 `tova_step_source/` 内 `pytest tests -q` | 9 passed |

共 **97 次测试执行通过**，其中隔离 adapter 包包含部分共同测试，不表示 97 个互不重复的场景。覆盖已有 runtime、校准、SideQuest、效率统计，以及新增的 ID/分数审计、最终与阶段分数隔离、源文件不修改、拒绝覆盖快照、冻结源码篡改检测。

## 源码与结果审计

- 使用 `scripts/prepare_rebuttal_sources.py --destination .runtime/publication_check_20260929 --prepare-reference` 从仓库包重建四个隔离源码目录；历史参考归档只读核验。
- 独立重算重建目录中 **99 个文件项**的 SHA256（含共享文件的各方法副本、协议说明、测试和许可证），全部与 `rebuttal/adapters/manifest.json` 相符。
- 从导出的逐题 JSON 独立重算 **5,208 条题目–条件记录**的 EM/F1，全部与汇总一致；27 组条件中 10 组完成，完成组均为 500 个唯一 ID。
- 导出器逐组核对登记 ID；已完成组的 checkpoint 与 results 中 ID/EM/F1 序列一致。原始输入的哈希、大小和读取时间保存在 `source_bindings.json`。
- 保留效率失败 traceback、ThinKV 校准负结果、R-KV 取消记录和先前停止队列的状态。未将失败补成零分，未把运行中指标列入最终结果表。
- 46 个新增 Python 文件语法解析、70 个新增 JSON 文件解析及新文档相对链接检查通过；发布文件未发现常见密钥格式。LazyEviction 上游参考文件保留原始尾部空白以维持冻结哈希，`.gitattributes` 仅对这两份 vendored Python 文件关闭空白检查与换行转换。

这些验证确认代码包和本次快照的一致性；没有宣称重新完成所有 GPU 实验、跨机器精确复现，或证明原论文表格的来源。历史硬编码路径、运行环境与比较边界见 [复现说明](REPRODUCING.md) 和 [协议](PROTOCOL.md)。
