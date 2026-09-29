# FlowKV-style on the restored evaluator

This source directory was extracted directly from
`results/rebuttal_20260927/source_snapshot/before_changes.tar`.
All original source files are unchanged, including the old runner, inference
class, attention scorer, cache manager, token tracker and pruning strategy.
The only implementation addition is `models/LegacyBaseline.py`.

The adapter exports the interface consumed by `run_rebuttal_legacy.py`:

- `LEGACY_METHOD = "flowkv_style"`
- `LEGACY_CONFIG_OVERRIDE`: mode `flowkv_style`, original EOS scoring forward,
  full prompt protection, no recent-token window, no step utility/floor. The
  unified runner supplies `cache_ratio` (0.2 or 0.5).
- `LEGACY_SCOPE`: complete manifest description of this labeled adaptation.
- `LegacyBaselineLLM`: directly inherits the unchanged
  `QwenLLMWithKVCache`; constructs an original `TokenTracker` when the old
  runner supplies `None` for the new mode.
- `legacy_audit()`: JSON-serializable per-episode sidecar data, including all
  freeze/compression events. The original evaluator's return values are intact.

Flow policy matches the previously implemented FlowKV-style agent adaptation:
the initial generation is unpruned and unfrozen, joining the first subsequent
block's candidates. After each later complete generated block and the legacy
stop truncation, preserve all previously frozen history. Allocate the remaining
global budget to the current uncompressed resident segment using the existing
TOVA score (last layer, last query, mean across attention heads), and freeze the
retained result. Global target is the legacy prompt-protected, cumulative
logical non-prompt ratio budget. New blocks may receive zero tokens. Previously
evicted tokens cannot be recovered. This is not an official FlowKV reproduction.

The old inference mechanism is deliberately preserved:

- The actual `generate_first`, `generate_incremental` and `_decode` execution
  remains the legacy model code. Its `model.generate` behavior and inherited
  generation-config repetition penalty are not replaced by the controlled raw
  argmax loop.
- No new absolute `position_ids`, per-token stop checking, BPE boundary repair,
  or final-token materialization is introduced. EOS scoring is the exact old
  deep-copy probe with its implicit physical-cache position.
- The public mode name avoids entering the old H2O/TOVA per-token decode path.
  A TOVA manager is used only for legacy bookkeeping and cache construction.
  Its automatic prefill prune hook is suppressed; Flow-specific compression
  happens after the inherited incremental call returns.
- The old StepKV selector and scorer remain untouched, and the runner does not
  calculate StepKV utility for this new mode.

The legacy `generate()` path may count one output token that has no KV tensor
because it was selected but never forwarded. This behavior is not repaired.
Flow selection can only select actual resident tensors; audit events record
both resident and bookkeeping counts, including `counted_but_nonresident_before`.
When actual eviction happens, the selected resident KV define the resulting
physical cache/map; there is no attempt to synthesize the missing token's KV.
When no resident eviction is needed, even the existing bookkeeping discrepancy
is preserved. Token logs and stop cropping remain inherited legacy behavior.
Audit token IDs are the inherited legacy tracker's identities, not a newly
corrected canonical transcript.

`boundary_peak_cache_bytes` measures KV at the observed method boundaries only.
It must not be described as an exact within-generation peak. The restored
runner's existing general CUDA peak metric is still available as before.

CPU validation (no model download or GPU):

```bash
OMP_NUM_THREADS=2 /home/fengboyu/kvmem/.runtime/rebuttal/bin/python -m pytest -q tests/test_legacy_flowkv.py
```

Tests compare outputs and all KV tensors against the original engine when no
eviction is needed, including post-generation stopping, penalty=1.05 and the
unmaterialized final token. They also check source byte identity, multiple
frozen blocks at both ratios, exact agreement with the repository TOVA selector,
independence from step utility, zero new quota, missing-token audit, unchanged
probe positions/cache behavior, and episode reset.
