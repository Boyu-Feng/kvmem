# 三 seed 扩展实验结果快照

捕获时间：2026-10-07T06:55:38.250906+00:00。
已完成 **72/72 组**，共 **36,000 条逐题评测**。
运行中的组只列状态，不发布阶段分数。

| 数据集 | 方法 | seed | 名义预算 | N | EM (%) | F1 (%) |
|---|---|---:|---:|---:|---:|---:|
| hotpotqa | sidequest_untrained | 42 | 自适应 | 500 | 22.60 | 31.55 |
| 2wiki | sidequest_untrained | 42 | 自适应 | 500 | 19.40 | 24.53 |
| musique | sidequest_untrained | 42 | 自适应 | 500 | 4.60 | 6.58 |
| hotpotqa | sidequest_untrained | 3407 | 自适应 | 500 | 25.00 | 33.88 |
| 2wiki | sidequest_untrained | 3407 | 自适应 | 500 | 21.00 | 26.65 |
| musique | sidequest_untrained | 3407 | 自适应 | 500 | 5.20 | 8.15 |
| hotpotqa | thinkkv_eviction_only | 233 | 20% | 500 | 21.40 | 29.23 |
| hotpotqa | thinkkv_eviction_only | 233 | 50% | 500 | 20.00 | 28.69 |
| 2wiki | thinkkv_eviction_only | 233 | 20% | 500 | 17.40 | 20.31 |
| 2wiki | thinkkv_eviction_only | 233 | 50% | 500 | 18.60 | 23.81 |
| musique | thinkkv_eviction_only | 233 | 20% | 500 | 2.00 | 4.76 |
| musique | thinkkv_eviction_only | 233 | 50% | 500 | 2.00 | 5.85 |
| hotpotqa | thinkkv_eviction_only | 42 | 20% | 500 | 18.00 | 25.01 |
| hotpotqa | thinkkv_eviction_only | 42 | 50% | 500 | 24.40 | 30.78 |
| 2wiki | thinkkv_eviction_only | 42 | 20% | 500 | 16.80 | 19.43 |
| 2wiki | thinkkv_eviction_only | 42 | 50% | 500 | 19.80 | 23.75 |
| musique | thinkkv_eviction_only | 42 | 20% | 500 | 1.60 | 3.75 |
| musique | thinkkv_eviction_only | 42 | 50% | 500 | 2.20 | 4.90 |
| hotpotqa | thinkkv_eviction_only | 3407 | 20% | 500 | 17.40 | 24.38 |
| hotpotqa | thinkkv_eviction_only | 3407 | 50% | 500 | 22.20 | 30.43 |
| 2wiki | thinkkv_eviction_only | 3407 | 20% | 500 | 15.80 | 18.76 |
| 2wiki | thinkkv_eviction_only | 3407 | 50% | 500 | 18.00 | 23.81 |
| musique | thinkkv_eviction_only | 3407 | 20% | 500 | 2.00 | 4.95 |
| musique | thinkkv_eviction_only | 3407 | 50% | 500 | 2.40 | 6.03 |
| hotpotqa | h2o_step | 42 | 20% | 500 | 8.60 | 11.05 |
| hotpotqa | h2o_step | 42 | 50% | 500 | 14.80 | 23.25 |
| hotpotqa | tova_step | 42 | 20% | 500 | 10.00 | 15.01 |
| hotpotqa | tova_step | 42 | 50% | 500 | 14.00 | 23.11 |
| hotpotqa | flowkv_style | 42 | 20% | 500 | 8.80 | 13.03 |
| hotpotqa | flowkv_style | 42 | 50% | 500 | 15.60 | 22.51 |
| hotpotqa | lazyeviction | 42 | 20% | 500 | 17.60 | 26.83 |
| hotpotqa | lazyeviction | 42 | 50% | 500 | 24.20 | 34.04 |
| 2wiki | h2o_step | 42 | 20% | 500 | 2.60 | 4.25 |
| 2wiki | h2o_step | 42 | 50% | 500 | 9.00 | 15.34 |
| 2wiki | tova_step | 42 | 20% | 500 | 4.20 | 5.54 |
| 2wiki | tova_step | 42 | 50% | 500 | 9.60 | 14.98 |
| 2wiki | flowkv_style | 42 | 20% | 500 | 3.40 | 5.00 |
| 2wiki | flowkv_style | 42 | 50% | 500 | 9.40 | 14.69 |
| 2wiki | lazyeviction | 42 | 20% | 500 | 16.40 | 23.11 |
| 2wiki | lazyeviction | 42 | 50% | 500 | 19.20 | 25.31 |
| musique | h2o_step | 42 | 20% | 500 | 0.40 | 1.66 |
| musique | h2o_step | 42 | 50% | 500 | 3.80 | 8.26 |
| musique | tova_step | 42 | 20% | 500 | 1.00 | 3.30 |
| musique | tova_step | 42 | 50% | 500 | 2.00 | 6.17 |
| musique | flowkv_style | 42 | 20% | 500 | 1.20 | 2.47 |
| musique | flowkv_style | 42 | 50% | 500 | 2.20 | 4.91 |
| musique | lazyeviction | 42 | 20% | 500 | 2.60 | 7.70 |
| musique | lazyeviction | 42 | 50% | 500 | 4.60 | 8.96 |
| hotpotqa | h2o_step | 3407 | 20% | 500 | 8.20 | 13.54 |
| hotpotqa | h2o_step | 3407 | 50% | 500 | 12.80 | 23.77 |
| hotpotqa | tova_step | 3407 | 20% | 500 | 10.80 | 14.89 |
| hotpotqa | tova_step | 3407 | 50% | 500 | 15.00 | 23.83 |
| hotpotqa | flowkv_style | 3407 | 20% | 500 | 8.00 | 12.32 |
| hotpotqa | flowkv_style | 3407 | 50% | 500 | 16.20 | 24.82 |
| hotpotqa | lazyeviction | 3407 | 20% | 500 | 20.00 | 29.22 |
| hotpotqa | lazyeviction | 3407 | 50% | 500 | 26.40 | 35.91 |
| 2wiki | h2o_step | 3407 | 20% | 500 | 3.60 | 5.48 |
| 2wiki | h2o_step | 3407 | 50% | 500 | 10.60 | 16.59 |
| 2wiki | tova_step | 3407 | 20% | 500 | 4.00 | 6.01 |
| 2wiki | tova_step | 3407 | 50% | 500 | 6.60 | 12.76 |
| 2wiki | flowkv_style | 3407 | 20% | 500 | 4.60 | 6.60 |
| 2wiki | flowkv_style | 3407 | 50% | 500 | 7.60 | 12.75 |
| 2wiki | lazyeviction | 3407 | 20% | 500 | 17.00 | 25.00 |
| 2wiki | lazyeviction | 3407 | 50% | 500 | 20.00 | 26.87 |
| musique | h2o_step | 3407 | 20% | 500 | 1.40 | 3.27 |
| musique | h2o_step | 3407 | 50% | 500 | 2.20 | 7.71 |
| musique | tova_step | 3407 | 20% | 500 | 1.40 | 2.93 |
| musique | tova_step | 3407 | 50% | 500 | 2.00 | 6.26 |
| musique | flowkv_style | 3407 | 20% | 500 | 1.00 | 2.32 |
| musique | flowkv_style | 3407 | 50% | 500 | 2.60 | 6.04 |
| musique | lazyeviction | 3407 | 20% | 500 | 3.20 | 9.70 |
| musique | lazyeviction | 3407 | 50% | 500 | 5.20 | 10.51 |

## 解读限制

- 只有已完成组发布分数；运行中和待运行组只列状态。
- seed 42、3407 与 seed 233 的题目集合不保证相同。
- SideQuest 使用独立的未微调解码器和自适应预算。
- ThinkKV 是仅按片段淘汰的 QA 适配，没有 KV 量化。
- 名义 token 保留比例不等于相同 KV 字节数或延迟。

逐组状态、完整精度指标及机制汇总见 [summary.json](summary.json)；[逐题分数](per_question/)只包含 ID、索引和 EM/F1；[源文件哈希](source_bindings.json)用于核对本地原始结果。
