# Legacy H2O-step

This source is an independent extraction of `before_changes.tar`. The nine
core files checked by the runner remain byte-identical to that snapshot.
`models/LegacyBaseline.py` is the only new runtime file.

The adapter retains `pruning_mode=h2o` and the original first `model.generate`,
manual incremental token decoder, generation configuration, physical cache
positions, post-generation stop-string crop, token bookkeeping and EOS scoring
probe. It does not switch to the controlled rebuttal runtime.

The sole algorithmic change is pruning time. Initial generation executes its
original post-stop pruning loop without suppression. On later calls, the
adapter defers prefill and per-token pruning until the inherited generation
and stop crop finish. It then runs the original prefill batch operation and
original cached-attention budget top-up with the original no-progress break.
A fresh EOS probe is obtained from the cropped cache. The now-unused prefill
probe is skipped. Decoder and selector implementations are inherited.

The original recent window is 32, with three scoring layers configured.
H2O uses the repository's independent EOS-probe score, not a newly introduced
cross-step accumulated-attention scorer; TOVA uses the original last-layer,
last-query, head-mean score. Step utility is not introduced. Prompt protection
and the cumulative logical nonprompt ratio retain the original semantics.
The selector's minimum-token and protected-window floors remain applicable,
so actual retention can exceed the nominal ratio on a short trajectory.

`legacy_audit()` reports individual original pruning calls in
`cache_measurements`, with `phase=initial_completed_block` or
`incremental_completed_block`, resident/bookkeeping lengths, KV bytes,
nominal budget and retained over-budget amount. `step_boundaries` also records
blocks requiring no eviction. Memory is sampled at boundaries, not a claim of
within-generation or EOS-probe-copy peak. Existing evaluator GPU peak metrics
remain untouched. Diagnostic timing includes sidecar overhead, and original
prefill allocator cleanup remains in place; the moved boundary also releases
attention and conditionally calls `torch.cuda.empty_cache()` as the old block did.

CPU validation:

```sh
OMP_NUM_THREADS=2 /home/fengboyu/kvmem/.runtime/rebuttal/bin/python -m pytest -q tests/test_legacy_step.py
```

Validation checks the nine original files, inherited decoder/probe/crop,
no-eviction output and KV parity, first-round output/KV/pruning-history parity
at ratios 0.2 and 0.5, post-stop-only later scoring, original selector parity,
empty response, failure cleanup, and audit/reset. No GPU run is performed here.
