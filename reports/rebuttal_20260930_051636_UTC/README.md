# Rebuttal 实验快照

读取时间：2026-09-30T05:16:36.299000+00:00 至 2026-09-30T05:16:39.320542+00:00。
这是上传时的静态快照；在线任务继续运行。各源文件的读取时间和 SHA256 见 `source_bindings.json`。

已完成 **20/27 组**可运行主实验；保存 **10206/13500 次题目评测**。
主实验总数包含 24 组 legacy 补实验和 3 组 SideQuest 适配；不含 smoke、pilot、ThinkKV 候选和效率补测。

## 队列进展

| 队列 | 状态 | 完成主实验 | 保存题数 |
|---|---|---:|---:|
| legacy_baselines | complete | 4/4 | 2000/2000 |
| legacy_step_controls | complete | 4/4 | 2000/2000 |
| multidataset_baselines | running | 5/8 | 2539/4000 |
| multidataset_step_controls | running | 6/8 | 3167/4000 |
| sidequest_adaptive_v1 | waiting_for_previous_queue | 1/3 | 500/1500 |

## 已完成的固定预算实验

Qwen2.5-7B-Instruct，seed233。20%/50% 是名义轨迹 token 保留档位，不能当作等物理显存。

| 数据集 | 方法 | 名义保留比例 | N | EM (%) | F1 (%) |
|---|---|---:|---:|---:|---:|
| hotpotqa | flowkv_style | 20% | 500 | 8.20 | 11.85 |
| hotpotqa | flowkv_style | 50% | 500 | 16.40 | 25.50 |
| hotpotqa | lazyeviction | 20% | 500 | 18.80 | 28.54 |
| hotpotqa | lazyeviction | 50% | 500 | 23.60 | 33.59 |
| hotpotqa | h2o_step | 20% | 500 | 10.80 | 15.97 |
| hotpotqa | h2o_step | 50% | 500 | 14.80 | 24.08 |
| hotpotqa | tova_step | 20% | 500 | 11.60 | 16.87 |
| hotpotqa | tova_step | 50% | 500 | 15.80 | 25.52 |
| 2wiki | flowkv_style | 20% | 500 | 4.20 | 6.43 |
| 2wiki | flowkv_style | 50% | 500 | 9.60 | 14.94 |
| 2wiki | lazyeviction | 20% | 500 | 13.40 | 19.84 |
| 2wiki | lazyeviction | 50% | 500 | 16.60 | 22.81 |
| musique | flowkv_style | 20% | 500 | 1.60 | 2.72 |
| 2wiki | h2o_step | 20% | 500 | 5.40 | 6.98 |
| 2wiki | h2o_step | 50% | 500 | 7.20 | 12.85 |
| 2wiki | tova_step | 20% | 500 | 3.40 | 4.92 |
| 2wiki | tova_step | 50% | 500 | 9.00 | 13.96 |
| musique | h2o_step | 20% | 500 | 1.20 | 2.44 |
| musique | h2o_step | 50% | 500 | 2.60 | 7.75 |

## SideQuest 未微调适配

独立解码协议、自适应预算；不能作为官方实现复现或与 legacy 等预算排名。

| 数据集 | 状态 | N | EM (%) | F1 (%) |
|---|---|---:|---:|---:|
| hotpotqa | complete | 500/500 | 24.00 | 33.52 |
| 2wiki | pending | 0/500 | — | — |
| musique | pending | 0/500 | — | — |

hotpotqa：辅助调用 308 次，无效命令 68 次；151 题实际删除 KV，共 38183 tokens。

## 尚未完成

| 数据集 | 方法 | 预算 | 状态 | 保存题数 |
|---|---|---|---|---:|
| musique | flowkv_style | 0.5 | running | 39/500 |
| musique | lazyeviction | 0.2 | pending | 0/500 |
| musique | lazyeviction | 0.5 | pending | 0/500 |
| musique | tova_step | 0.2 | running | 167/500 |
| musique | tova_step | 0.5 | pending | 0/500 |
| 2wiki | sidequest_untrained | adaptive | pending | 0/500 |
| musique | sidequest_untrained | adaptive | pending | 0/500 |

## 已知阻塞与后续

- SideQuest 后两集等待 `multidataset_baselines` 整条队列完成。
- ThinkKV：三类 thought 校准 20/20 完成，共同三峰层为 0，主实验 0/6；普通 LLM 单类分支尚未实现，不能据此认定方法不适用。
- 效率补测：0/7，因同卡出现其他计算进程而失败；没有有效独占效率结论。
- LazyEviction：4 个原生 KV 头展开至 28 个，须报告实际 KV 字节、跟踪状态及副本，不能把 slot 比例当成显存比例。
- 原表逐题结果与原环境仍待溯源；原表参考值不与本轮单 seed 拼接为配对统计。

详细执行顺序与验收条件见 [后续计划](../../docs/rebuttal/NEXT_STEPS.md)。

## 产物

- `summary.json`：完整进度、最终/阶段指标和统计口径。
- `per_question/`：已保存主实验的逐题 EM/F1、诊断计时及紧凑缓存审计；不复制题目正文、答案或权重。
- `evidence/`：运行 manifest、队列计划、失败状态、校准与 pilot 结果。
- `source_bindings.json`：所有读取的本地源文件 SHA256。
