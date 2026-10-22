# WebBench 跨服务器恢复

本说明用于从 GitHub 上的加密恢复包继续现有实验。运行继续使用
BrowseComp `comparison_20261021_r01` 与 WebWalker
`comparison_20261021_webwalker_r04` 的冻结计划，以及 r01 的
`scheduler/seed_order_v4` 的条件路由与 seed 门控。为目标 **3–4 张 A100、每张
约 30 GiB 空闲显存**增加独立主机资源适配器：改变设备分配、并发、显存上限
与本地裁判 CPU offload，原计划、演员源码、实验预算、题序、精度、
已提交答案和已有裁判返回保持原样；实际资源配置单独留下合同和执行证据。

实验演员仍是 **Qwen2.5-7B 与 Llama-3.1-8B**。**Qwen3-32B 只负责
BrowseComp-Plus 的答案评分**，不运行网页任务；WebWalkerQA 评分调用冻结的
GPT-4.1 API。三个模型权重都需要恢复。

## 目标服务器

- Linux x86_64，建议与原机一致的 Ubuntu 24.04；三个隔离环境要求
  **Python 3.12.3**。包中的 venv 解释器链接到 `/usr/bin/python3.12`，
  复制 venv 本身不能替代系统 Python。
- NVIDIA 驱动与 Container Toolkit（使用容器时）能运行打包的 CUDA 12.x
  PyTorch/vLLM；原机为 H200、驱动 `595.91.07`。目标 GPU 型号、UUID、
  驱动会写入迁移记录；不同硬件不保证数值逐位相同。
- 使用下文 A100 profile，`--devices 0,1,2` 选择三张，四张则用
  `--devices 0,1,2,3`。**每卡最多一个演员或本地裁判**，不会清退别人的进程。
  每卡准入要求至少 **28 GiB 空闲**：26 GiB 显存预算和 2 GiB 余量。
  演员 allocator 上限为 26 GiB；上下文长度、KV 保留比例、生成预算不变。
- 本地 32B 裁判保留 BF16 和 TP=1，采用 **48 GiB CPU 权重 offload**。
  目标必须至少有 **96 GiB 当前可用主机内存（MemAvailable）**。实际 vLLM
  显存比例为 `min(0.65, 26 / 显卡总GiB)`：例如 80 GiB 卡约为 0.325。
  冻结评分合同中的名义 0.65 保留，实际比例、offload 和GPU身份另记
  `host_execution.json`；不能只改 offload 而仍让 vLLM 按80GiB的65%申请。
- 原机是 H200，尚未在目标 A100 上验证所有实验分支的峰值容量。启动时先
  在每张选定 GPU 上运行无基准题目的裁判加载/单 token 探针，成功后才启动
  正式工作。此探针只验证裁判容量；演员真实长任务仍可能 OOM。失败保留原始
  记录并停止对应条件，不自动降低精度、缩短上下文、减少预算或重试。
- 目标需访问公开网页；WebWalker 裁判依赖本地
  `http://127.0.0.1:8313/v1` 服务。服务必须接收 `gpt-4.1` 请求并返回
  `gpt-4.1-2025-04-14`；原 seed 42、temperature 0 保留。代理的凭证在目标
  单独配置，不能上传到公开仓库。也可用SSH把原机已配置的8313端口转发到
  目标的同一端口。恢复预检只检测端口，不发额外API裁判请求；
  冻结评分器对每次实际返回的模型身份继续检查。
- 磁盘需容纳约14.7 GiB加密分卷、49.36 GB恢复内容、96.8 GB模型权重，
  以及解包暂存、迁移前证据归档和后续审计；建议至少250 GiB可用空间。

## 必须保持的路径

```text
/mnt/checkpoints/bruli/fengboyu/kvmem_webbench_20261021/
/mnt/checkpoints/bruli/fengboyu/kvmem_rebuttal_20260929/
/home/bruli/fengboyu/kvmem/
```

计划、source hashes、Lucene 资产、浏览器 runtime、评分 contract 和 v4 路由
合同都包含绝对路径。使用真实目录或 bind mount 保留这些路径。不要用指向
其他位置的目录 symlink：冻结路由检查 `Path.resolve()`，而合同保留原路径。
不要批量替换 JSON 路径或重新计算原计划的 hash。

如果磁盘实际挂载在其他位置，可把解包后的两个目录树 bind mount 到上述
路径。容器也可以这样暴露原目录，示例中的宿主路径按实际位置替换：

```bash
docker run --rm -it --gpus all --ipc=host --network host \
  --mount type=bind,src=/data/webbench_restore/mnt/checkpoints/bruli/fengboyu,dst=/mnt/checkpoints/bruli/fengboyu \
  --mount type=bind,src=/data/webbench_restore/home/bruli/fengboyu/kvmem,dst=/home/bruli/fengboyu/kvmem \
  ubuntu:24.04 bash
```

此示例只说明挂载，镜像还需安装 Python 3.12.3 和原浏览器所需系统动态库。
通过容器 `nvidia-smi` 确认设备编号，再传给主机适配器的 `--devices`。不要
修改冻结 `plan.json` 的4–7编号。

**系统 Python、NVIDIA 驱动和普通浏览器系统动态库是外部依赖，不在恢复包中。**
包内包含Chromium二进制和已冻结的libasound安装树；恢复预检会实际执行两个
浏览器的 `--version`，缺动态库会直接报错。Ubuntu 24.04的最小镜像通常还需：

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv age zstd git curl \
  libglib2.0-0t64 libnss3 libnspr4 libatk1.0-0t64 libatk-bridge2.0-0t64 \
  libdbus-1-3 libcups2t64 libxkbcommon0 libgbm1 libx11-6 libxcomposite1 \
  libxdamage1 libxext6 libxfixes3 libxrandr2 libpango-1.0-0 libcairo2 libatspi2.0-0t64
/usr/bin/python3.12 --version
```

最后一条必须输出 `Python 3.12.3`；若发行版提供其他小版本，需要另外配置
3.12.3，不能修改原环境manifest绕过检查。不要在三个打包venv中执行pip升级。

## 下载与复原

1. 克隆发布版本的代码，下载
   [kvmem-migration-20261022](https://github.com/Boyu-Feng/kvmem/releases/tag/kvmem-migration-20261022)
   的全部16个加密分卷和校验清单；使用单独交付的 `restore.agekey` 解密。
   密钥源机保存在迁移目录的 `private/restore.agekey`，不会上传GitHub。
   题目、答案、神经输出、网页快照和裁判记录只存在于加密包；不要
   将解密目录提交到公开 Git。
2. 原路径下恢复全部 r01–r04 运行树、共享 `assets`、`retrieval_runtime`、
   `browsers`、`webwalker_runtime`、演员 venv、裁判 venv，以及原工作区中
   被 manifest 引用的 BrowseComp 源码/qrels。旧 r02/r03 和历史 recovery
   是开发/恢复证据，不作为当前正式条件重新启动。
3. 运行仓库提供的 `scripts/webbench_migrate_models.py`，按其帮助中的
   `--download` 选项补齐三个固定 revision 的模型分片，再逐文件核对
   `config/environments/webbench_migration_models_20261022.json`。Llama 下载
   需要目标机器拥有对应 Hugging Face 访问权限。不得用模型名称的最新
   revision 替换已冻结分片。
4. 按原版本提供系统 Python 和浏览器动态库，启动目标自己的 GPT-4.1
   本地代理。不要运行实验 `prepare`、重新下载网页缓存、重写原 qualification
   或重建模型/资产 ready manifest。

在Ubuntu 24.04安装 `age`、`zstd`，并确保当前用户可以创建原绝对路径后：

```bash
git clone --branch kvmem-migration-20261022 https://github.com/Boyu-Feng/kvmem.git /home/bruli/fengboyu/kvmem
cd /home/bruli/fengboyu/kvmem
python3 -B scripts/webbench_migrate_bundle.py download --directory /data/kvmem_release
python3 -B scripts/webbench_migrate_bundle.py inspect --directory /data/kvmem_release --identity /secure/restore.agekey
python3 -B scripts/webbench_migrate_bundle.py restore --directory /data/kvmem_release --identity /secure/restore.agekey --destination /
curl -fL https://github.com/Boyu-Feng/kvmem/releases/download/kvmem-migration-20261022/handoff.json -o /data/kvmem_release/handoff.json
python3 -B scripts/webbench_migrate_models.py --download
```

密钥路径按安全传输后的实际位置填写。Llama需目标自己的已授权Hugging Face
令牌，工具读取 `HF_TOKEN` 或目标已有token缓存，不复制源机凭据。模型下载支持
断点续传，已有文件先验hash，不覆盖不匹配文件。

如借用原机已配置的WebWalker代理，在目标另开一个持久在线终端：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:8313:127.0.0.1:8313 USER@SOURCE_HOST
```

转发只提供原模型服务，不能更换裁判模型或在新请求里改采样参数。

模型固定 revision：

| 模型 | revision |
| --- | --- |
| Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` |
| meta-llama/Meta-Llama-3.1-8B-Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` |
| Qwen/Qwen3-32B | `9216db5781bf21249d130ec9da846c4624c16137` |

恢复包必须包含所有共享 URL JSON 记录及其 `objects/*.html`，包括原本抓取失败的
记录。后续方法必须看到相同的首次网页响应；丢失缓存后重新抓取无法恢复原协议。
所有 `audits`、`attempts`、checkpoint、predictions、scoring `items` / `calls`
和历史 failure/recovery 也需要保留。仅有结果表或 checkpoint 无法验收恢复。

演员固定核心版本为 torch 2.10.0、transformers 5.0.0、tokenizers 0.22.2、
numpy 1.26.4、safetensors 0.8.0、accelerate 1.15.0。裁判独立环境为 vLLM
0.11.0、torch 2.8.0、transformers 4.57.3；评分器会精确比较 **全部已安装
distribution 的名称与版本**，因此不要在这个环境新增工具包或升级 pip。
浏览器为 Crawl4AI 0.7.4、Playwright 1.64.0、Chromium revision 1248
（156.0.8078.4），相关可执行文件和 libasound 有字节哈希绑定。
Lucene 的 JRE、pyserini 和 pyjnius 安装树包含已绑定的 dist-info 文件，直接
复原打包内容比通过 pip 重装更可靠。

## 只读验收与显式启动

恢复工具默认不启动进程，也不移走 attempt 或 launch。模型尚未下载时，可先
检查其余内容；这个模式不能用于启动：

```bash
cd /home/bruli/fengboyu/kvmem
python3 -B scripts/webbench_migrate_resume.py --skip-model-weights \
  --host-profile config/environments/webbench_migration_a100_30gib.json --devices 0,1,2
```

模型齐备后完整验收：

```bash
python3 -B scripts/webbench_migrate_resume.py --verify-only \
  --host-profile config/environments/webbench_migration_a100_30gib.json --devices 0,1,2
```

工具检查冻结计划及源文件、所有模型文件、检索资产/运行库、不可变网页缓存、
checkpoint 有序前缀、audit 哈希、attempt 绑定、已完成预测/评分、未完成评分的
输入与已耗用请求次数、三个隔离环境和 GPU 条件。缺文件、hash 变化或版本不符
都会给出具体错误并返回非零退出码。检查大文件需要时间。

源服务器必须已停止这轮实验的所有写入者，且恢复包包含最后提交的状态。
发布包的 `handoff.json` 应含如下字段（实际值由导出时记录）：

```json
{
  "schema": "webbench_migration_handoff_v1",
  "source_host": {"hostname": "SOURCE_HOST", "boot_id": "SOURCE_BOOT_ID"},
  "captured_at_utc": "EXPORT_TIMESTAMP",
  "source_writers_stopped": true,
  "route_plan_sha256": {
    "browsecomp_plus": "a79227d003e92b3736ad0fd64b041e4bf14c15243d47b44d252fc35ecde769ec",
    "webwalkerqa": "1a0afdde2dd1a2069175937ad5602b95541506592d3646914c914f0d3b525e36"
  }
}
```

不要为了通过检查把一个仍在写入的复制过程标成 finalized。源端和目标端不能
各自从同一断点继续后再合并结果。验收通过后，在新服务器明确启动：

```bash
python3 -B scripts/webbench_migrate_resume.py --start --handoff /data/kvmem_release/handoff.json \
  --host-profile config/environments/webbench_migration_a100_30gib.json --devices 0,1,2
```

工具会再次完整验收，并拒绝在源机器同一 boot 上启动。它先取得 actor/scorer
控制器和 worker 的锁，将原主机 launch、active/reservation 视图及中断任务的
证据归档到 r01 的 `migration/host_<目标 boot ID>/`。原主机 PID 不能用来接管
目标主机进程，所以旧 actor/scorer launch 和活动视图会移出活动命名空间。
对于未完成演员，每个条件只移出一个绑定正确的未提交 attempt，再从已提交
题目边界继续。中断题从头执行，当前实现不保存题内 KV 状态。

已完成答案、checkpoint、已提交 attempt、audit、评分 `items` 和 `calls` 保持
原样。已返回的裁判输出复用；requested/failed 调用仍计入原上限。BrowseComp
单题裁判上限为 1，WebWalker 为 10。若中断评分已耗尽上限，或已有正式 failure
记录，工具停止并指出条件，不删除失败来换取重试。

随后冻结 `host_resources.json`，逐卡执行裁判容量探针并保存
`judge_probe_gpu<N>.json` / `.log`。全部通过后启动单个主机协调器；它调用
原演员、原评分器和 v4 条件/seed门控，继续遵守 seed233 → seed42 → seed3407，
必须完成当前 seed 的演员及评分验证才进入下一 seed。每卡一个工作进程；
最多一个本地32B裁判和一个API评分进程，先为已有答案安排评分再分配演员。
新进程及硬件写入迁移 manifest，实际配置写入每条件 `host_execution.json`
及两路 `HOST_EXECUTION.json`；日志保存在同一迁移目录。
启动返回只表示控制器存活，继续查看 r01 `queue_status.json`、两路
`scoring_status.json` 和新 checkpoint 提交来确认实际推进。

同一个目标 boot 的迁移不能盲目重复执行。如果网络、磁盘或进程启动在中间
失败，先查看 manifest 的阶段和已归档原文件，保留证据后按具体阶段恢复。
工具不会自动反复重跑中断题。

不带 `--host-profile` 的入口保留原H200控制器，要求物理4–7卡和每卡至少
100 GiB总显存，仅供匹配原硬件布局的目标使用；上述A100命令必须保留profile。
