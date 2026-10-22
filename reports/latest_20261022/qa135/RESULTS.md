# Completed 135-condition QA results

Three seeds (233, 42, 3407), 500 questions per condition. SD uses n−1. Different method decoders and adaptive budgets remain distinct; timing is diagnostic.

| Model | Dataset | Method | Budget | EM mean ± SD | F1 mean ± SD |
|---|---|---|---|---|---|
| llama31_8b | 2wiki | flowkv_style | 20% | 4.60 ± 0.87 | 5.75 ± 0.96 |
| llama31_8b | 2wiki | flowkv_style | 50% | 12.13 ± 1.60 | 14.39 ± 1.66 |
| llama31_8b | 2wiki | h2o_step | 20% | 8.00 ± 0.72 | 9.49 ± 0.68 |
| llama31_8b | 2wiki | h2o_step | 50% | 13.47 ± 0.90 | 16.24 ± 0.97 |
| llama31_8b | 2wiki | intentkv_rule | 20% | 12.47 ± 1.30 | 14.10 ± 1.96 |
| llama31_8b | 2wiki | intentkv_rule | 50% | 18.20 ± 0.53 | 21.18 ± 1.39 |
| llama31_8b | 2wiki | lazyeviction | 20% | 17.73 ± 1.50 | 20.71 ± 1.95 |
| llama31_8b | 2wiki | lazyeviction | 50% | 21.07 ± 2.08 | 24.65 ± 1.85 |
| llama31_8b | 2wiki | sidequest_untrained | adaptive | 22.40 ± 0.80 | 25.39 ± 1.38 |
| llama31_8b | 2wiki | thinkkv_eviction_only | 20% | 15.87 ± 2.00 | 17.54 ± 1.79 |
| llama31_8b | 2wiki | thinkkv_eviction_only | 50% | 17.47 ± 1.60 | 19.43 ± 1.64 |
| llama31_8b | 2wiki | tova_step | 20% | 6.40 ± 1.25 | 7.83 ± 1.48 |
| llama31_8b | 2wiki | tova_step | 50% | 13.47 ± 0.64 | 15.25 ± 0.94 |
| llama31_8b | hotpotqa | flowkv_style | 20% | 12.40 ± 1.04 | 16.73 ± 1.03 |
| llama31_8b | hotpotqa | flowkv_style | 50% | 17.13 ± 0.70 | 23.96 ± 2.21 |
| llama31_8b | hotpotqa | h2o_step | 20% | 14.67 ± 0.92 | 19.52 ± 1.87 |
| llama31_8b | hotpotqa | h2o_step | 50% | 19.20 ± 0.87 | 26.30 ± 1.21 |
| llama31_8b | hotpotqa | intentkv_rule | 20% | 19.93 ± 0.83 | 26.33 ± 1.42 |
| llama31_8b | hotpotqa | intentkv_rule | 50% | 23.20 ± 0.20 | 30.46 ± 0.97 |
| llama31_8b | hotpotqa | lazyeviction | 20% | 19.07 ± 1.86 | 26.14 ± 2.61 |
| llama31_8b | hotpotqa | lazyeviction | 50% | 25.13 ± 1.63 | 34.10 ± 2.42 |
| llama31_8b | hotpotqa | sidequest_untrained | adaptive | 25.20 ± 0.53 | 32.81 ± 1.85 |
| llama31_8b | hotpotqa | thinkkv_eviction_only | 20% | 17.93 ± 1.85 | 24.11 ± 2.02 |
| llama31_8b | hotpotqa | thinkkv_eviction_only | 50% | 21.33 ± 2.25 | 28.55 ± 2.59 |
| llama31_8b | hotpotqa | tova_step | 20% | 12.13 ± 1.62 | 16.35 ± 1.62 |
| llama31_8b | hotpotqa | tova_step | 50% | 15.07 ± 0.99 | 21.26 ± 0.30 |
| llama31_8b | musique | flowkv_style | 20% | 1.20 ± 0.60 | 2.58 ± 0.49 |
| llama31_8b | musique | flowkv_style | 50% | 3.07 ± 1.15 | 5.75 ± 1.30 |
| llama31_8b | musique | h2o_step | 20% | 1.60 ± 0.53 | 3.72 ± 0.92 |
| llama31_8b | musique | h2o_step | 50% | 1.93 ± 0.12 | 4.87 ± 0.31 |
| llama31_8b | musique | intentkv_rule | 20% | 1.80 ± 0.87 | 3.86 ± 1.24 |
| llama31_8b | musique | intentkv_rule | 50% | 3.07 ± 0.83 | 6.67 ± 1.75 |
| llama31_8b | musique | lazyeviction | 20% | 4.13 ± 0.95 | 8.24 ± 1.43 |
| llama31_8b | musique | lazyeviction | 50% | 4.67 ± 1.36 | 8.34 ± 1.92 |
| llama31_8b | musique | sidequest_untrained | adaptive | 3.00 ± 0.35 | 5.51 ± 0.74 |
| llama31_8b | musique | thinkkv_eviction_only | 20% | 0.87 ± 0.31 | 3.08 ± 0.65 |
| llama31_8b | musique | thinkkv_eviction_only | 50% | 1.80 ± 0.72 | 3.86 ± 0.88 |
| llama31_8b | musique | tova_step | 20% | 1.80 ± 0.53 | 3.62 ± 0.47 |
| llama31_8b | musique | tova_step | 50% | 3.20 ± 1.06 | 6.06 ± 1.36 |
| qwen25_7b | 2wiki | intentkv_rule | 20% | 13.73 ± 0.50 | 17.17 ± 0.96 |
| qwen25_7b | 2wiki | intentkv_rule | 50% | 11.00 ± 1.44 | 16.05 ± 1.21 |
| qwen25_7b | hotpotqa | intentkv_rule | 20% | 18.13 ± 0.42 | 25.33 ± 0.64 |
| qwen25_7b | hotpotqa | intentkv_rule | 50% | 19.13 ± 1.80 | 27.75 ± 3.06 |
| qwen25_7b | musique | intentkv_rule | 20% | 1.87 ± 0.64 | 4.96 ± 0.55 |
| qwen25_7b | musique | intentkv_rule | 50% | 2.33 ± 0.12 | 6.44 ± 0.98 |
