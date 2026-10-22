"""Export compact public results without modifying any frozen experiment.

Uses only Python's standard library. Raw checkpoints, prompts, answers, browser
content, judge responses, trajectories and credentials are never copied.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

BASE = Path('/mnt/checkpoints/bruli/fengboyu')
QA = BASE / 'kvmem_rebuttal_20260929/results'
WEB = BASE / 'kvmem_webbench_20261021'
ROOT = Path(__file__).resolve().parents[1]
PRIOR_COMMIT = '0ab40124f25752bbecf62cda48acc6ca9abc6279'
STRING_KEYS = {
    'id', 'name', 'group', 'model', 'method', 'dataset', 'benchmark', 'purpose',
    'variant', 'case', 'condition', 'status', 'score_status', 'termination',
    'protocol', 'schema', 'unit', 'timing_policy', 'statistics', 'design',
    'cautions', 'scope', 'normalization', 'speedup', 'timing_warning',
    'memory_warning', 'lazy_warning', 'diagnostic_semantics', 'pairing',
    'paired_comparison_status', 'sidequest_budget', 'execution_order_version',
    'accuracy_scope', 'kv_measurement', 'cache_mean_definition',
    'preprune_definition', 'timing_scope', 'model_repo', 'model_revision',
}
DROP_KEYS = {
    'audit', 'audits', 'debug', 'trajectory', 'trajectories', 'question', 'query',
    'gold_answer', 'answer', 'reference', 'predicted_answer', 'prediction',
    'response', 'judge_response', 'judge_prompt', 'judge_result', 'prompt',
    'messages', 'raw', 'text', 'content', 'html', 'markdown', 'token_ids',
    'sample_ids', 'task_ids', 'main_execution_order', 'display_condition_order',
    'smoke_completed', 'source', 'source_summary', 'output', 'path', 'command',
    'argv', 'traceback', 'error', 'error_message', 'hardware_attempts',
}
SUMMARY_RUNS = [
    'component_ablation_seeds233_3407_20260930_v1',
    'shared_accuracy/llama_component_ablation_seeds233_3407_v1',
    'sensitivity_shared/qwen7b_seed233_v1',
    'qa32_hotpotqa_v1', 'qa32_2wiki_v1', 'qa32_musique_v1',
    'qa32_recent_methods_seed233_r50_20261006_v2',
    'qwen25_7b_v2', 'qwen25_32b_v2', 'qwen25_7b_schedules_v1',
    'qwen25_32b_schedules_v1',
    'qwen25_7b_stepkv_qa_scoring_20260930',
    'qwen25_32b_stepkv_qa_scoring_20260930',
    'alfworld_7b_recent_methods_20261003_v1',
    'intentkv_rule_qa_7b_seed233_20261005_v5',
    'reuse_semantic_hotpotqa50_r20_seed233_20261003_v11',
    'latency_qa7b_h200_gpu7_20260930_v2',
    'latency_qa7b_2wiki_musique_h200_gpu7_recovery_v4',
    'latency_sidequest_qa7b_h200_gpu7_20261002_v2',
    'thinkkv_latency_immediate_20261003_v1',
    'latency_unified_stepkv_decoder_20261003_v1',
]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compact(value, key='', depth=0):
    """Project numeric statistics and explicitly named descriptive metadata."""
    if key in DROP_KEYS or depth > 8:
        return None
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float):
            require(math.isfinite(value), 'Nonfinite public statistic')
        return value
    if isinstance(value, str):
        return value if key in STRING_KEYS or re.fullmatch(r'[a-f0-9]{64}', value) else None
    if isinstance(value, dict):
        result = {}
        for name, item in value.items():
            if name in DROP_KEYS:
                continue
            selected = compact(item, name, depth + 1)
            if selected is not None or item is None:
                result[name] = selected
        return result or None
    if isinstance(value, list):
        if len(value) > 1000:
            return None
        selected = [compact(item, key, depth + 1) for item in value]
        return [item for item in selected if item is not None] or None
    return None


class Export:
    def __init__(self, output):
        require(not output.exists(), 'Snapshot already exists; choose a new directory')
        self.output, self.bindings = output, {}
        self.started = datetime.now(timezone.utc).isoformat()
        output.mkdir(parents=True)

    def label(self, path):
        path = Path(path)
        for base, label in [(QA, 'rebuttal_results'), (WEB, 'webbench'), (ROOT, 'repository')]:
            if path.is_relative_to(base):
                return str(Path(label) / path.relative_to(base))
        raise ValueError('Unexpected source outside approved roots: ' + str(path))

    def read(self, path):
        path = Path(path)
        raw = path.read_bytes()
        binding = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                       read_at_utc=datetime.now(timezone.utc).isoformat())
        label = self.label(path)
        if label in self.bindings:
            require(self.bindings[label]['sha256'] == binding['sha256'], 'Source changed on reread: ' + label)
        self.bindings[label] = binding
        return json.loads(raw)

    def write(self, relative, value):
        path = self.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    def qa(self):
        base = QA / 'qa_baselines_two_models_three_seeds_20261013_v1'
        source = self.read(base / 'SUMMARY.json')
        status = self.read(base / 'queue_status.json')
        require(source['complete_main'] == source['requested_main'] == 135 and status['status'] == 'complete',
                'The 135-condition QA experiment is not complete')
        groups, seed_groups = [], defaultdict(list)
        for job in source['results']:
            directory = Path(job['source'])
            final = self.read(directory / 'results.json')
            checkpoint = self.read(directory / 'checkpoint.json')
            manifest = self.read(directory / 'manifest.json')
            state = self.read(directory / 'status.json')
            rows = final['results']
            ids = [row['id'] for row in rows]
            require(state['status'] == 'complete' and len(rows) == job['samples'] == 500, 'Incomplete QA group: ' + job['name'])
            require(ids == manifest['fingerprint']['sample_ids'] and len(set(ids)) == 500, 'QA ordered sample IDs differ')
            require(checkpoint['results'] == rows, 'QA final/checkpoint content differs: ' + job['name'])
            require(manifest['fingerprint_sha256'] == final['fingerprint_sha256'] == checkpoint['fingerprint_sha256'],
                    'QA checkpoint/final fingerprint differs')
            public_rows = []
            for row in rows:
                require(row['em'] in (0, 1, False, True) and 0 <= row['f1'] <= 1, 'Invalid QA score scale')
                public = {key: row[key] for key in ('id', 'index', 'em', 'f1')}
                for key in ('seconds', 'num_steps', 'peak_cuda_allocated_bytes'):
                    if key in row:
                        public[key] = compact(row[key], key)
                audit = row.get('audit')
                if isinstance(audit, dict) and 'sha256' in audit:
                    public['audit_sha256'] = audit['sha256']
                public_rows.append(public)
            em = 100 * statistics.mean(float(row['em']) for row in rows)
            f1 = 100 * statistics.mean(row['f1'] for row in rows)
            expected = job['summary']
            require(abs(em - expected.get('em_percent', expected.get('exact_match'))) < 1e-8
                    and abs(f1 - expected.get('f1_percent', expected.get('f1_score'))) < 1e-8,
                    'QA summary does not reproduce per-question scores')
            relative = 'qa135/per_question/' + job['name'] + '.json'
            self.write(relative, public_rows)
            record = {key: job[key] for key in ('name', 'model', 'dataset', 'method', 'seed', 'ratio', 'samples', 'reused')}
            record.update(status='complete', em_percent=em, f1_percent=f1, per_question=relative,
                          source=self.label(directory), ordered_ids_sha256=digest(ids),
                          fingerprint_sha256=manifest['fingerprint_sha256'])
            groups.append(record)
            seed_groups[(job['model'], job['dataset'], job['method'], job['ratio'])].append(record)
        aggregated = []
        for (model, dataset, method, ratio), rows in sorted(seed_groups.items(), key=lambda item: str(item[0])):
            require({row['seed'] for row in rows} == {233, 42, 3407}, 'Missing QA seed')
            aggregated.append(dict(model=model, dataset=dataset, method=method, ratio=ratio,
                seeds=[233, 42, 3407], samples_per_seed=500,
                em_mean_percent=statistics.mean(row['em_percent'] for row in rows),
                em_sample_sd=statistics.stdev(row['em_percent'] for row in rows),
                f1_mean_percent=statistics.mean(row['f1_percent'] for row in rows),
                f1_sample_sd=statistics.stdev(row['f1_percent'] for row in rows)))
        result = dict(plan_sha256=source['plan_sha256'], complete_main=135, requested_main=135,
            evaluations=67500, reused_conditions=6, newly_executed_conditions=129,
            coverage='Llama-3.1-8B seven methods: 117 conditions; Qwen2.5-7B IntentKV-rule: 18 conditions.',
            previous_published_groups_not_reexported=99, previous_published_commit=PRIOR_COMMIT,
            validation='Ordered manifest IDs, checkpoint/final equality and recomputed EM/F1; no model or judge calls.',
            timing_policy='Shared GPU elapsed times are diagnostic, not latency measurements.',
            conditions=groups, three_seed_summary=aggregated)
        self.write('qa135/summary.json', result)
        self.write('qa135/status.json', compact(status))
        return result

    def webbench(self):
        conditions, route_states = [], {}
        routes = [('browsecomp_plus', 'comparison_20261021_r01'), ('webwalkerqa', 'comparison_20261021_webwalker_r04')]
        for benchmark, run_name in routes:
            base = WEB / 'runs' / run_name
            plan = self.read(base / 'plan.json')
            queue = self.read(base / 'queue_status.json')
            score_queue = self.read(base / 'scoring_status.json')
            route_states[benchmark] = dict(run=run_name, plan_sha256=plan['sha256'],
                queue_updated=queue['updated'], scoring_updated=score_queue['updated_unix'],
                seed_order=queue['seed_order'], current_stage=queue['current_stage'],
                queue_status=queue['status'], scoring_status=score_queue['status'])
            for job in plan['jobs']:
                if job['benchmark'] != benchmark or job['purpose'] != 'main':
                    continue
                directory = base / 'jobs' / job['name']
                checkpoint_path = directory / 'checkpoint.json'
                final_path = directory / 'predictions.json'
                source_path = final_path if final_path.exists() else checkpoint_path
                stored = self.read(source_path) if source_path.exists() else None
                rows = stored['results'] if stored else []
                tasks = plan['tasks'][benchmark][str(job['seed'])]['main']
                require(len(rows) <= len(tasks) == 100, 'WebBench row count outside plan')
                if stored:
                    require(stored['plan_sha256'] == plan['sha256'] and stored['job'] == job, 'WebBench checkpoint route differs')
                for row, task in zip(rows, tasks):
                    require(row['id'] == task['id'] and row['task_sha256'] == task['task_sha256']
                            and row['question'] == task['question'] and row['gold_answer'] == task['answer'],
                            'WebBench ordered task/content binding differs')
                if final_path.exists():
                    require(len(rows) == 100, 'Final prediction file is incomplete')
                scoring = directory / 'scoring'
                contract = self.read(scoring / 'contract.json') if (scoring / 'contract.json').exists() else None
                final_scores = self.read(scoring / 'scores.json') if (scoring / 'scores.json').exists() else None
                if contract:
                    require(contract['predictions_sha256'] == self.bindings[self.label(final_path)]['sha256'], 'Scoring input hash differs')
                    require(contract['scorer_sha256'] == plan['source_hashes']['scripts/webbench_scoring_v1.py'], 'Scorer source differs')
                public_rows, scored, correct, missing_env = [], 0, 0, 0
                for index, row in enumerate(rows):
                    item_path = scoring / 'items' / f'{index:04d}.json'
                    item = self.read(item_path) if item_path.exists() else None
                    environment_failed = row.get('status') == 'environment_failure'
                    if item:
                        require(contract is not None and item['id'] == str(row['id'])
                                and item['source_row_sha256'] == digest(row)
                                and item['contract_sha256'] == digest(contract), 'Scoring item binding differs')
                        require((environment_failed and item['score'] is None) or
                                (not environment_failed and item['score'] in (0, 1)), 'Invalid score/missingness')
                    if item and item['score'] is not None:
                        scored += 1
                        correct += item['score']
                    missing_env += environment_failed
                    public = {key: row[key] for key in ('id', 'task_sha256', 'termination', 'status', 'audit_sha256') if key in row}
                    public.update(score=item['score'] if item else None, score_recorded=item is not None,
                                  environment_failed=environment_failed)
                    for key in ('seconds', 'logical_budget_used', 'tool_call_counts', 'peak_cuda_allocated_bytes'):
                        if key in row:
                            public[key] = compact(row[key], key)
                    if item:
                        public['source_row_sha256'] = item['source_row_sha256']
                        public['contract_sha256'] = item['contract_sha256']
                    public_rows.append(public)
                terminal = final_scores is not None
                if terminal:
                    require(len(rows) == len(final_scores['results']) == 100 and final_scores['scored'] == scored
                            and final_scores['correct'] == correct and final_scores['environment_failed'] == missing_env
                            and final_scores['contract_sha256'] == digest(contract), 'Terminal score summary differs')
                    for index, item in enumerate(final_scores['results']):
                        require(item == self.read(scoring / 'items' / f'{index:04d}.json'), 'Final score/item content differs')
                relative = 'webbench/per_question/' + job['name'] + '.json'
                if rows:
                    self.write(relative, public_rows)
                record = {key: job[key] for key in ('name', 'benchmark', 'model', 'method', 'ratio', 'seed')}
                record.update(run=run_name, requested=100, committed=len(rows), inferred=len(rows)-missing_env,
                    environment_failed=missing_env, scored=scored, correct=correct,
                    actor_complete=final_path.exists(), scoring_terminal=terminal,
                    status=final_scores['status'] if terminal else 'partial_not_ranked' if rows else 'pending',
                    accuracy=correct / 100 if terminal and scored == 100 else None,
                    scored_subset_accuracy=correct / scored if terminal and scored else None,
                    per_question=relative if rows else None)
                conditions.append(record)
        require(len(conditions) == 192, 'Incorrect routed main-condition denominator')
        result = dict(routes=route_states, conditions=conditions, requested_conditions=192,
            requested_evaluations=19200, committed_evaluations=sum(row['committed'] for row in conditions),
            actor_complete_conditions=sum(row['actor_complete'] for row in conditions),
            scoring_terminal_conditions=sum(row['scoring_terminal'] for row in conditions),
            valid_scored_evaluations=sum(row['scored'] for row in conditions),
            environment_failure_evaluations=sum(row['environment_failed'] for row in conditions),
            active_scope='r01 BrowseComp-Plus and r04 WebWalkerQA only; obsolete r01/r02/r03 WebWalker arms excluded.',
            aggregation='No partial rankings. Full accuracy requires 100 valid scores; environment failures remain missing.',
            judges=dict(browsecomp_plus='Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137',
                        webwalkerqa='gpt-4.1-2025-04-14; user_authorized_gpt41_changed_judge'),
            snapshot_semantics='Atomic per-file reads over the recorded export window, not a simultaneous cluster checkpoint.')
        self.write('webbench/summary.json', result)
        return result

    def other(self):
        records = []
        for name in SUMMARY_RUNS:
            directory = QA / name
            files = []
            candidates = [directory / 'SUMMARY.json', directory / 'SUMMARY.md', directory / 'queue_status.json']
            candidates += sorted((directory / 'report').rglob('SUMMARY.json'))
            candidates += sorted((directory / 'report').rglob('DATASET_SUMMARIES.json'))
            for path in candidates:
                if not path.is_file() or path.suffix != '.json':
                    continue
                value = self.read(path)
                public = compact(value)
                target = 'completed_summaries/' + name + '/' + str(path.relative_to(directory))
                self.write(target, public)
                files.append(target)
            records.append(dict(run=name, files=files,
                overlap_note='Six conditions are reused in qa135; do not add them again.' if name.startswith('intentkv_rule_qa_') else None,
                completeness_note='Only 2 of 6 main conditions completed; failed queue is retained.' if name=='qwen25_32b_schedules_v1' else None))
        for name in ['shared_memory/stepkv_metadata_storage_20261004_v3', 'shared_memory/stepkv_hotpotqa_seed233_step_score_v1']:
            directory = QA / name
            paths = sorted(directory.glob('main/**/status.json')) if 'metadata' in name else sorted(directory.glob('r*/status.json'))
            for path in paths:
                value = self.read(path)
                require(value['status'] == 'complete', 'Memory measurement is incomplete: ' + str(path))
                self.write('completed_summaries/' + name + '/' + str(path.relative_to(directory)), compact(value))
            records.append(dict(run=name, complete_conditions=len(paths), scope='Memory measurements; separate from QA accuracy.'))
        self.write('completed_summaries/index.json', records)
        return records

    def finish(self, qa, web, other):
        finished = datetime.now(timezone.utc).isoformat()
        self.write('source_bindings.json', self.bindings)
        self.write('snapshot.json', dict(schema='kvmem_public_snapshot_20261022_v1',
            started_utc=self.started, finished_utc=finished,
            export_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            prior_published_commit=PRIOR_COMMIT, qa_conditions=135, qa_evaluations=67500,
            webbench_committed=web['committed_evaluations'], webbench_actor_complete=web['actor_complete_conditions'],
            webbench_scoring_terminal=web['scoring_terminal_conditions'], other_summary_collections=len(other),
            source_files=len(self.bindings),
            full_checkpoint_policy='Full resumable state is transferred separately in an encrypted migration archive.'))
        table = ['| Model | Dataset | Method | Budget | EM mean ± SD | F1 mean ± SD |',
                 '|---|---|---|---|---|---|']
        for row in qa['three_seed_summary']:
            budget = 'adaptive' if row['ratio'] is None else f"{100*row['ratio']:g}%"
            table.append(f"| {row['model']} | {row['dataset']} | {row['method']} | {budget} | "
                         f"{row['em_mean_percent']:.2f} ± {row['em_sample_sd']:.2f} | "
                         f"{row['f1_mean_percent']:.2f} ± {row['f1_sample_sd']:.2f} |")
        (self.output / 'qa135/RESULTS.md').write_text('# Completed 135-condition QA results\n\n'
            'Three seeds (233, 42, 3407), 500 questions per condition. SD uses n−1. '
            'Different method decoders and adaptive budgets remain distinct; timing is diagnostic.\n\n'
            + '\n'.join(table) + '\n')
        lines = ['# KVMem latest public snapshot — 2026-10-22', '',
            f'Export window: **{self.started} to {finished} (UTC)**. This is a static public result export.', '',
            f'- **QA: 135/135 complete conditions, 67,500 evaluations.** Llama-3.1-8B contributes 117 conditions across seven methods; Qwen2.5-7B contributes 18 IntentKV-rule conditions. Six of the 135 reuse the completed 2026-10-05 IntentKV run; 129 were newly executed.',
            f'- **WebBench: {web["committed_evaluations"]}/19,200 committed main evaluations, {web["actor_complete_conditions"]}/192 actor conditions complete, {web["scoring_terminal_conditions"]}/192 terminal scoring conditions.** Only r01 BrowseComp-Plus and r04 WebWalkerQA are included. Smoke and superseded WebWalker runs are excluded.',
            '- **Additional completed work:** Qwen/Llama component ablations (42 + 42 groups), sensitivity (120), historical Qwen2.5-32B QA (33 existing-method + 12 recent-method groups; outside the current 7B/8B WebBench continuation), ALFWorld, semantic reuse pilot, latency, and metadata measurements have compact numeric summaries. These heterogeneous collections are not combined into one evaluation total.', '',
            '[QA tables](qa135/RESULTS.md) · [QA summary](qa135/summary.json) · [WebBench status/results](webbench/summary.json) · [Other collections](completed_summaries/index.json) · [Source hashes](source_bindings.json)', '',
            '## Coverage and interpretation', '',
            f'The previously published Qwen2.5-7B six-method **99 groups / 49,500 evaluations** already exist in commit `{PRIOR_COMMIT}` and are not exported or counted again here. The six IntentKV seed233 groups appear both as provenance summaries and as reused members of QA135; count them once.', '',
            'QA per-question exports retain IDs, EM/F1 and compact diagnostics. Ordered IDs, full checkpoint/final equality and recomputed EM/F1 were checked. This export does not rerun models, judges, or the complete historical trajectory audits. WebBench exports check task order/content bindings, scored-item/source-row hashes, terminal item equality and summary counts.', '',
            'WebWalkerQA uses the user-authorized GPT-4.1 judge (`gpt-4.1-2025-04-14`), distinct from the public default GPT-4. Each completed seed233 arm has 17 unavailable-root cases and 83 valid scores. Missing environment cases are not zeros; no full accuracy or across-method ranking is claimed from these subsets. Partial scorer progress is recorded without presenting partial accuracy.', '',
            'Component ablations contain seeds233/3407 only; their 42+42 groups are not a verified three-seed table. `qwen25_32b_schedules_v1` is explicitly incomplete (2/6 main conditions) and its failed status remains visible. Older StepKV ALFWorld results and the later scoring-aligned reruns are separate protocols. The semantic-reuse pilot is five 50-question conditions, not the 500-question QA benchmark.', '',
            'Shared-GPU times are diagnostic. Latency results retain separate matched-decoder controls and their original measurement scopes; memory inventory categories can overlap. No heterogeneous aggregate, unverified significance claim, or partial-result ranking is generated.', '',
            '## Public contents and resuming elsewhere', '',
            'Public files omit question text, gold/predicted answers, trajectories, browser pages, prompts, raw judge responses, credentials, weights and environments. Other collections use a numeric/metadata projection, so their source JSON hashes refer to the full local artifact, not an identical copied JSON. Full resumable checkpoints belong in the separate encrypted migration archive; these public files alone cannot resume inference.', '',
            'Cross-server scoring must retain the frozen source snapshots, original absolute paths, `score_queue_contract.json`, per-condition contracts, `items/`, all `calls/` (including failed/requested attempts), prediction inputs and audits. Recreate the local `http://127.0.0.1:8313/v1` endpoint with the same authorized actual judge model. Preserve the original authorization and probe bytes; a new readiness probe must use a separate file. BrowseComp local judging additionally requires its pinned Qwen3-32B weights and judge runtime. Returned judgments are immutable and must not be resampled.', '',
            'Live source files are atomically read over the export interval. Counts reflect those recorded reads, not a simultaneous checkpoint of every process. Source hashes and read times are recorded in `source_bindings.json`.', '']
        (self.output / 'README.md').write_text('\n'.join(lines))
        size = sum(path.stat().st_size for path in self.output.rglob('*') if path.is_file())
        require(size < 50_000_000, f'Public export unexpectedly large: {size}')
        print(json.dumps(dict(output=str(self.output), bytes=size, qa_conditions=135,
            webbench_committed=web['committed_evaluations'], source_files=len(self.bindings)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/latest_20261022')
    args = parser.parse_args()
    export = Export(args.output)
    qa = export.qa()
    web = export.webbench()
    other = export.other()
    export.finish(qa, web, other)


if __name__ == '__main__':
    main()
