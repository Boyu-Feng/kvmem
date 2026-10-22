# WebWalkerQA shared environment

`scripts/webwalker_environment_v1.py` prepares the official 680-question main
split at revision `c7ab3329a2a4a35ca569fdca369b61a3d1787e5b`. Seeds 233, 42 and
3407 each use 100 questions; the two smoke questions per seed are held out from
the union of all formal selections. Gold answers and annotation metadata stay
on the scoring side. The environment retains only task ID, question and root.

All methods use the same rendered Crawl4AI 0.7.4 / Chromium environment and
append-only URL snapshots, including HTTP failures. The isolated Python and
browser dependencies are recorded in `config/environments/webwalker_20261021.freeze.txt`
and the ignored asset `provenance/crawler_runtime.json`. There is no static HTML
fallback. This is a shared single-actor ReAct adaptation, not a reproduction of
the upstream multi-agent WebWalker algorithm. Scoring separately preserves the
upstream cot_qa prompt/parser. The user explicitly authorized GPT-4.1 for this
comparison; the qualified actual judge is `gpt-4.1-2025-04-14`, with seed 42 and
temperature 0. Reports identify this change from the upstream default GPT-4.

Navigation starts at the public dataset root. When the actual rendered response
redirects that root, the environment may adopt the observed canonical URL only
when the root path and query are unchanged, the hostname is unchanged or differs
by one `www.` prefix, and the protocol is unchanged or upgrades HTTP to HTTPS.
Default ports are allowed. A different hostname, changed root path/query,
nondefault port or HTTPS downgrade cannot establish a canonical root. No target
is inferred from answers or golden paths. Every root/page event records
`original_root`, `effective_root`, `final_url` and its immutable snapshot hash;
the actor's original task identity is never rewritten.

After canonicalization, Visit accepts links observed on rendered pages within
the effective root, plus backtracking to the original/effective root. Search
filters previously observed link labels/URLs; it is not a web search engine.
The worker must call `acknowledge_observation(text)` after its tokenizer token
cap, on the initial and every subsequent observation. Only complete emitted
link lines present in that delivered text enter the cumulative visible-link
set. Links hidden beyond truncation cannot be guessed through Visit or recovered
through Search; full raw snapshots remain available only for audit.
Unrelated redirects expose no outside content. Unavailable roots raise
`EnvironmentFailure` and must remain explicit unscored records rather than
incorrect answers. Cache corruption and browser dependency failures are runtime
errors, not model errors or website failures.

`python3 scripts/webwalker_environment_v1.py --browser-preflight` checks real
JavaScript execution, navigation through a JS-created link, and one official
root from the global smoke holdout. The reports and snapshots are separate from
formal results and contain zero actor/judge calls.
