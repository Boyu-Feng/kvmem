# 三 seed 扩展实验结果快照

捕获时间：2026-10-02T13:09:28.671899+00:00。
已完成 **36/72 组**，共 **18,000 条逐题评测**。
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
| hotpotqa | h2o_step | 3407 | 20% | 500 | 8.20 | 13.54 |
| hotpotqa | h2o_step | 3407 | 50% | 500 | 12.80 | 23.77 |
| hotpotqa | tova_step | 3407 | 20% | 500 | 10.80 | 14.89 |
| hotpotqa | tova_step | 3407 | 50% | 500 | 15.00 | 23.83 |
| hotpotqa | flowkv_style | 3407 | 20% | 500 | 8.00 | 12.32 |
| hotpotqa | flowkv_style | 3407 | 50% | 500 | 16.20 | 24.82 |

## 解读限制

- 只有已完成组发布分数；运行中和待运行组只列状态。
- seed 42、3407 与 seed 233 的题目集合不保证相同。
- SideQuest 使用独立的未微调解码器和自适应预算。
- ThinkKV 是仅按片段淘汰的 QA 适配，没有 KV 量化。
- 名义 token 保留比例不等于相同 KV 字节数或延迟。

逐组状态、完整精度指标及机制汇总见 [summary.json](summary.json)；[逐题分数](per_question/)只包含 ID、索引和 EM/F1；[源文件哈希](source_bindings.json)用于核对本地原始结果。
