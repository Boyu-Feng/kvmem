# Frozen rebuttal adapters

These are the exact method-specific overlays used in the 2026-09-28 legacy
experiment. `manifest.json` identifies every common and overlay source file by
SHA256. Materialize them with `scripts/prepare_rebuttal_sources.py`; do not copy
one method's `LegacyBaseline.py` over another live experiment.

The `LEGACY_VALIDATION.json` records are historical validation evidence, not a
claim that new hardware/environment has already passed the same checks. Tests
and scope documents remain alongside each overlay. LazyEviction reference files
retain their Apache-2.0 license and upstream commit attribution.

R-KV is not included as an active adapter because its queued experiments were
cancelled. SideQuest and ThinkKV calibration live in the root `scripts/` tree.

See [reproduction notes](../../docs/rebuttal/REPRODUCING.md) and
[protocol limitations](../../docs/rebuttal/PROTOCOL.md).
