# 代码整理与复现说明

## 冻结源码

本轮在线 worker/controller 有源码哈希检查。因此本次整理采用新增文档、导出器和 adapter 包的方式，原推理核心与正在运行的 controller/worker 文件保持原字节。

[`rebuttal/adapters/manifest.json`](../../rebuttal/adapters/manifest.json) 记录每个隔离实现所需源码的 SHA256。与仓库相同的文件复用根目录版本，差异文件按方法收录：

- `flowkv_style/models/LegacyBaseline.py`
- `lazyeviction/models/LegacyBaseline.py` 与 `LazyEvictionLLM.py`
- `h2o_step/models/LegacyBaseline.py`
- `tova_step/models/LegacyBaseline.py`

各目录保留原 CPU 验证记录、测试和协议说明。LazyEviction 官方参考代码来自 `Halo-949/LazyEviction` 的 `051d663990cd1628d72db3e0e849bb20d4bd0e0c`，对应 Apache-2.0 LICENSE 随文件分发。没有下载模型、数据或依赖嵌套 Git 仓库。

在新目录重建隔离源码：

```bash
python3 scripts/prepare_rebuttal_sources.py \
  --destination .runtime/rebuttal_sources \
  --prepare-reference
```

此命令逐文件核对哈希，拒绝覆盖已有源码目录。若历史 `before_changes.tar` 不存在，只从已核验的九个核心文件重建 runner 所需参考归档；重建归档不是历史原 tar，不包含也不生成实验结果。

## 环境与公开资产

本轮 Python 3.12，PyTorch 2.7.0+cu126、Transformers 5.0.0。完整安装状态保存在 [freeze](../../config/environments/rebuttal_20260928.freeze.txt)；它包含共享底层环境的额外包，是来源记录，不保证全部包在任意平台一次解析安装成功。根目录旧 `requirements.txt` 的 torch 版本不同，不应直接称作本轮运行环境。

模型、HotpotQA、LongRAG、2Wiki、MuSiQue 的 revision 与归一化 SHA256 见 [资产记录](../../config/rebuttal_assets_20260928.json)。完整 Wiki 索引使用 5,233,235 篇文档。模型、索引和数据不随 Git 上传。

`scripts/prepare_rebuttal_assets.py` 是当时下载入口，但会在新调用时解析远端当前 revision；精确复现应使用上述记录中的 revision 调用 `huggingface_hub.snapshot_download`，不能把新解析版本当成历史版本。2Wiki/MuSiQue 的下载 revision 已固定在 `scripts/prepare_rebuttal_multidataset.py`。归一化和索引构建沿用原 loader / `build_wiki_index.py`。

## 单组运行

在准备好 pinned assets、`model_ready.json` / `data_ready.json` 后，新输出目录的 HotpotQA 示例：

```bash
export KVMEM_PYTHON="$PWD/.runtime/rebuttal/bin/python"
export KVMEM_ASSETS="/path/to/stepkv-assets"
CUDA_VISIBLE_DEVICES=0 "$KVMEM_PYTHON" run_rebuttal_legacy.py \
  --source "$PWD/.runtime/rebuttal_sources/lazy_source" \
  --assets "$KVMEM_ASSETS" --method lazyeviction --ratio 0.2 \
  --seed 233 --samples 2 --sample-start 500 --purpose smoke \
  --output "$PWD/results/new_run/lazy_smoke_r20"
```

先确认 smoke 多步执行、实际淘汰且无运行错误，再另起 500 题主实验；不能根据 smoke 分数选择配置。H2O-step / TOVA-step 使用 `run_rebuttal_legacy_step.py`；2Wiki/MuSiQue 使用 `run_rebuttal_multidataset.py --dataset ...`。所有改变配置或源码的运行都使用新目录。

历史 controller、SideQuest/校准入口及部分原测试仍包含本机 `/data/experiment/fengboyu/stepkv`、`/home/fengboyu/kvmem` 路径，属于记录下来的部署限制。换机时先做路径适配与新的哈希登记，不能直接续接旧 controller 或宣称逐字环境一致。本次未改这些在线入口以免触发正在运行队列的源码变更保护。

## 测试、发布与状态

主目录 CPU 回归测试：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 \
  .runtime/rebuttal/bin/python -m pytest tests -q
```

隔离 adapter 测试在 materialize 后的对应目录运行。历史测试里的 reference tar 绝对路径需要在新服务器显式适配；线上机器可直接核验。发布验证记录见 [验证报告](VALIDATION.md)。

`reports/` 的主实验逐题导出保留 ID、EM/F1、诊断耗时和紧凑缓存统计；省略题目正文、标准答案、完整轨迹及权重。`source_bindings.json` 将每个导出绑定到实际读取的 checkpoint、manifest、sidecar 和状态文件。运行中指标在机器可读文件中标为 `partial_not_ranked`，最终表只列已完成结果。

实时状态需在有本地 `results/` 的机器运行 `scripts/rebuttal_status.py`。GitHub 上的是明确时间范围的静态快照；后续更新使用新的 `reports/` 子目录。
