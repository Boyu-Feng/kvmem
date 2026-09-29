# LazyEviction on the restored legacy evaluator

Use `models.LegacyBaseline.LegacyBaselineLLM`, `LEGACY_CONFIG_OVERRIDE`,
`LEGACY_METHOD == "lazyeviction"`, and `LEGACY_SCOPE`. The exported class inherits
the original `QwenLLMWithKVCache`. The override uses `pruning_mode="none"` to
disable the original KV manager; LazyEviction alone owns its attention hooks.
The runner overrides `cache_ratio` with 0.2 or 0.5.

`run_all_wiki_experiments_v2.py` and `kv_cache/pruning_strategy.py` were restored
byte-for-byte from `results/rebuttal_20260927/source_snapshot/before_changes.tar`.
The model wrapper already matched that snapshot exactly. All nine core source
files are checked using `run_rebuttal_legacy.verify_original_sources` before
validation. No original generation, scorer, or evaluator code was edited.

## Preserved old behavior

The adapter delegates `generate_first`, `generate_incremental`, `_decode`, and
`truncate_cache` to the original methods. It preserves the original HF
`model.generate` path, inherited generation processors/repetition penalty,
chat template/tokenization, batched observation prefill, supplied positions,
post-generation string truncation and physical-prefix cache crop. It preserves
the old last-emitted-token-not-in-KV behavior and resulting reported/actual cache
length discrepancy. No absolute logical RoPE override or transactional stop
rollback from the newer controlled runtime is used.

The first generation remains unpruned. Real-query recurrence is recorded during
that generation, but no initial-turn eviction is permitted.

## Baseline-specific hook changes

Official source: <https://github.com/Halo-949/LazyEviction>, commit
`051d663990cd1628d72db3e0e849bb20d4bd0e0c`. The exact source files used in CPU
equivalence tests remain under `vendor/lazyeviction_reference` with their
Apache-2.0 license.

- Attention keeps separate resident positions for every layer/query head,
  preserving official head-wise selection. It uses each real query, the
  attention-threshold recurrence update, maximum recurrence interval, and the
  official-code sigmoid importance formula. It does not use an EOS estimator.
- Batched observations remain batched. Their actual causal attention rows update
  recurrence in order. If a batch crosses a W boundary, a single selection is
  deferred to the batch end, scored at that ending query clock. This is an
  explicit schedule adaptation required to preserve the original prefill path.
  Single-token decode forwards retain the ordinary W-query selection schedule.
- The initial prompt is protected. The target slots per head are
  `prompt + max(min(W, processed_nonprompt_queries), floor(ratio *
  processed_nonprompt_queries))`. The clock counts processed input queries,
  including temporary generated continuation later cropped by legacy stop
  handling; it does not count the final emitted token when HF has not forwarded
  that token. This denominator and the W floor are disclosed adaptations, not
  an assertion of identical effective memory budgets to prior StepKV rows.
- Legacy stop handling crops recurrence/ID arrays by the same physical prefix
  as the cache. Prior attention history and query-clock progression remain;
  earlier eviction decisions are not undone. This preserves the original
  generator and stop rule instead of applying the newer runtime's rollback.
- Timestamps/MRI use int64; the official-code sigmoid score is evaluated in
  float32. This avoids cache-dtype timestamp rounding. W=175 and alpha=0.0001
  come unchanged from the official Qwen/MATH example and are not tuned from
  smoke or main accuracy.
- The hook subtracts only actually evicted slots from the legacy reported
  cache counter. It leaves the original reported-versus-materialized length
  discrepancy intact. This necessary bookkeeping prevents an additional drift
  from baseline evictions without changing the old no-eviction engine.

Qwen2.5-7B stores 28 query heads instead of 4 native GQA KV heads in this canonical
layout: **7 times as many heads**. Slot ratios are not physical memory ratios
against native FullKV. `legacy_audit()` reports actual storage bytes, ID/MRI/
timestamp bytes, returned first-turn KV copies, their peaks, every window event,
stop crop, and both legacy-reported and actual cache lengths. CUDA kernel
workspace/temporary attention allocations are outside these cache-specific
counts and remain covered by the original evaluator's GPU memory measurement.

## Verification

`tests/test_legacy_lazy.py` compares the complete first and two incremental
generations against the original CPU Qwen2 engine with eviction disabled:
text, logits, forwarded tokens, position IDs, cache positions, masks, repetition
penalty, and missing-final-KV accounting. Other tests cover legacy stop crop
parity, per-head metadata alignment, unpruned first generation, real later
eviction for both ratios, deferred prefill boundaries, selector equivalence,
memory accounting, and episode reset. The companion selector tests compare
recurrence and selected K/V tensors exactly to the official source.

The uniform runner is external to this source directory:

```bash
/home/fengboyu/kvmem/.runtime/rebuttal/bin/python /home/fengboyu/kvmem/run_rebuttal_legacy.py --source /data/experiment/fengboyu/stepkv/rebuttal_legacy_20260928/lazy_source --method lazyeviction --ratio 0.2 --seed 233 --samples 2 --sample-start 500 --purpose smoke --output OUTPUT_DIRECTORY
```

Only CPU verification was performed during migration; GPU smoke and its actual
eviction gate must pass before running the 500-question main condition.
