# KVMem latest public snapshot — 2026-10-22

Export window: **2026-10-22T12:50:36.438541+00:00 to 2026-10-22T12:51:07.327987+00:00 (UTC)**. This is a static public result export.

- **QA: 135/135 complete conditions, 67,500 evaluations.** Llama-3.1-8B contributes 117 conditions across seven methods; Qwen2.5-7B contributes 18 IntentKV-rule conditions. Six of the 135 reuse the completed 2026-10-05 IntentKV run; 129 were newly executed.
- **WebBench: 2415/19,200 committed main evaluations, 20/192 actor conditions complete, 18/192 terminal scoring conditions.** Only r01 BrowseComp-Plus and r04 WebWalkerQA are included. Smoke and superseded WebWalker runs are excluded.
- **Additional completed work:** Qwen/Llama component ablations (42 + 42 groups), sensitivity (120), historical Qwen2.5-32B QA (33 existing-method + 12 recent-method groups; outside the current 7B/8B WebBench continuation), ALFWorld, semantic reuse pilot, latency, and metadata measurements have compact numeric summaries. These heterogeneous collections are not combined into one evaluation total.

[QA tables](qa135/RESULTS.md) · [QA summary](qa135/summary.json) · [WebBench status/results](webbench/summary.json) · [Other collections](completed_summaries/index.json) · [Source hashes](source_bindings.json)

## Coverage and interpretation

The previously published Qwen2.5-7B six-method **99 groups / 49,500 evaluations** already exist in commit `0ab40124f25752bbecf62cda48acc6ca9abc6279` and are not exported or counted again here. The six IntentKV seed233 groups appear both as provenance summaries and as reused members of QA135; count them once.

QA per-question exports retain IDs, EM/F1 and compact diagnostics. Ordered IDs, full checkpoint/final equality and recomputed EM/F1 were checked. This export does not rerun models, judges, or the complete historical trajectory audits. WebBench exports check task order/content bindings, scored-item/source-row hashes, terminal item equality and summary counts.

WebWalkerQA uses the user-authorized GPT-4.1 judge (`gpt-4.1-2025-04-14`), distinct from the public default GPT-4. Each completed seed233 arm has 17 unavailable-root cases and 83 valid scores. Missing environment cases are not zeros; no full accuracy or across-method ranking is claimed from these subsets. Partial scorer progress is recorded without presenting partial accuracy.

Component ablations contain seeds233/3407 only; their 42+42 groups are not a verified three-seed table. `qwen25_32b_schedules_v1` is explicitly incomplete (2/6 main conditions) and its failed status remains visible. Older StepKV ALFWorld results and the later scoring-aligned reruns are separate protocols. The semantic-reuse pilot is five 50-question conditions, not the 500-question QA benchmark.

Shared-GPU times are diagnostic. Latency results retain separate matched-decoder controls and their original measurement scopes; memory inventory categories can overlap. No heterogeneous aggregate, unverified significance claim, or partial-result ranking is generated.

## Public contents and resuming elsewhere

Public files omit question text, gold/predicted answers, trajectories, browser pages, prompts, raw judge responses, credentials, weights and environments. Other collections use a numeric/metadata projection, so their source JSON hashes refer to the full local artifact, not an identical copied JSON. Full resumable checkpoints belong in the separate encrypted migration archive; these public files alone cannot resume inference.

Cross-server scoring must retain the frozen source snapshots, original absolute paths, `score_queue_contract.json`, per-condition contracts, `items/`, all `calls/` (including failed/requested attempts), prediction inputs and audits. Recreate the local `http://127.0.0.1:8313/v1` endpoint with the same authorized actual judge model. Preserve the original authorization and probe bytes; a new readiness probe must use a separate file. BrowseComp local judging additionally requires its pinned Qwen3-32B weights and judge runtime. Returned judgments are immutable and must not be resampled.

Live source files are atomically read over the export interval. Counts reflect those recorded reads, not a simultaneous checkpoint of every process. Source hashes and read times are recorded in `source_bindings.json`.
