# 本次发布验证

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
