"""Deferred SideQuest-inspired replacement for cancelled R-KV jobs."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_legacy import atomic_json, verify_original_sources
from scripts.run_rebuttal_queue import existing_worker, wait_for_existing_worker, process_matches
from scripts.run_rebuttal_efficiency_queue import maintenance_blockers, maintenance_launch_lock, maintenance_path

BASE = ROOT / 'results/rebuttal_20260928'
OUTPUT = BASE / 'sidequest_adaptive_v1'
RUNNER = ROOT / 'scripts/run_sidequest_baseline.py'
GPU = 3


def validate(directory, job):
    status = json.loads((directory / 'status.json').read_text())
    result = json.loads((directory / 'results.json').read_text())
    manifest = json.loads((directory / 'manifest.json').read_text())
    rows = result['results']
    if (status['status'] != 'complete' or len(rows) != job['samples']
            or {r['id'] for r in rows} != set(manifest['ids'])):
        raise RuntimeError(f'Incomplete SideQuest result: {directory}')
    if any(r['method'] != 'sidequest_untrained' or not r['audit']['enabled'] for r in rows):
        raise RuntimeError('Unexpected method in SideQuest output')
    return result['summary']


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code = ['scripts/run_sidequest_baseline.py', 'scripts/sidequest_runtime.py',
            'models/RebuttalLLM.py', 'token_tracker.py', 'scripts/run_sidequest_queue.py']

    def hashes():
        return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in code}

    pinned = hashes()
    jobs = [dict(dataset=dataset, method='sidequest_untrained', budget='adaptive', purpose=purpose,
                 samples=n, sample_start=start, name=f'{dataset}/{purpose}/sidequest_untrained')
            for dataset in ('hotpotqa', '2wiki', 'musique')
            for purpose, n, start in (('smoke', 2, 500), ('main', 500, 0))]
    plan = dict(jobs=jobs, gpu=GPU, source_hashes=pinned, replaces='rkv', trained=False,
                original_results='retained', main_jobs=3, main_evaluations=1500,
                protocol='independent untrained adaptation; not matched-runtime legacy comparison')
    pp = OUTPUT / 'queue_plan.json'
    if pp.exists() and json.loads(pp.read_text()) != plan:
        raise RuntimeError('Changed SideQuest queue plan: use a new output directory')
    atomic_json(pp, plan)

    def state(status, **extra):
        atomic_json(OUTPUT / 'queue_status.json', dict(status=status, supervisor_pid=os.getpid(),
                    updated=time.time(), total_jobs=len(jobs), **extra))

    def summary():
        lines = ['# SideQuest-inspired 未微调适配：自适应预算', '',
                 '单独解码协议，不能当作与legacy严格对齐的官方SideQuest对照；原论文表格保留。', '',
                 '| 数据集 | N | EM (%) | F1 (%) | 辅助调用 | 实际删除事件 |', '|---|---:|---:|---:|---:|---:|']
        for job in jobs:
            path = OUTPUT / job['name']
            if job['purpose'] == 'main' and (path / 'status.json').exists():
                if json.loads((path / 'status.json').read_text())['status'] == 'complete':
                    s = validate(path, job)
                    lines.append(f"| {job['dataset']} | {s['samples']} | {s['exact_match']:.2f} | {s['f1_score']:.2f} | {s['auxiliary_calls']} | {s['applied_deletion_events']} |")
        (OUTPUT / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_PROGRESS_BARS='1')
    job = None
    try:
        while True:
            sp = BASE / 'sidequest_pilot_v1/status.json'
            pilot = json.loads(sp.read_text()) if sp.exists() else {}
            if pilot.get('status') == 'failed':
                raise RuntimeError('SideQuest pilot failed; do not start expanded runs')
            if pilot.get('status') == 'complete':
                report = json.loads((sp.parent / 'SUMMARY.json').read_text())
                arm = report['arms']['sidequest_untrained']
                if report['paired_samples'] != 20 or arm['applied_deletion_events'] < 1:
                    raise RuntimeError('Pilot did not complete 20 pairs with actual cursor eviction')
                break
            state('waiting_for_pilot', completed_jobs=0)
            time.sleep(15)
        summary()
        for ordinal, job in enumerate(jobs):
            # User requested these after the other rebuttal runs. Different GPU
            # prevents competing with the existing accuracy controllers.
            predecessor = BASE / ('legacy_baselines' if job['dataset'] == 'hotpotqa' else 'multidataset_baselines')
            while True:
                previous = json.loads((predecessor / 'queue_status.json').read_text())
                if previous['status'] == 'complete':
                    break
                state('blocked_by_previous_queue' if previous['status'] == 'failed' else 'waiting_for_previous_queue',
                      job=job, completed_jobs=ordinal, predecessor=str(predecessor), predecessor_status=previous['status'])
                time.sleep(15)
            if hashes() != pinned:
                raise RuntimeError('SideQuest source changed before launch')
            verify_original_sources(ROOT)
            directory = OUTPUT / job['name']
            directory.mkdir(parents=True, exist_ok=True)
            st = directory / 'status.json'
            if st.exists() and json.loads(st.read_text())['status'] == 'complete':
                validate(directory, job)
                continue
            cmd = [sys.executable, '-u', str(RUNNER), '--dataset', job['dataset'], '--purpose', job['purpose'],
                   '--samples', str(job['samples']), '--sample-start', str(job['sample_start']), '--output', str(directory)]
            adopted = existing_worker(st, cmd)
            if adopted is None and st.exists():
                previous_worker = json.loads(st.read_text()).get('pid')
                if previous_worker and process_matches(previous_worker, cmd):
                    adopted = previous_worker
                elif previous_worker and (Path('/proc') / str(previous_worker) / 'cmdline').exists():
                    raise RuntimeError('Live worker PID does not match requested SideQuest command')
            if adopted:
                state('running', job=job, completed_jobs=ordinal, child_pid=adopted, adopted_worker=True)
                wait_for_existing_worker(adopted, cmd, st)
            else:
                while True:
                    with maintenance_launch_lock(GPU):
                        blockers = maintenance_blockers(maintenance_path(GPU))
                        free = int(subprocess.check_output(['nvidia-smi', f'--id={GPU}', '--query-gpu=memory.free',
                                                           '--format=csv,noheader,nounits'], text=True).strip())
                        if not blockers and free >= 28000:
                            if hashes() != pinned:
                                raise RuntimeError('SideQuest source changed while waiting')
                            with (directory / 'run.log').open('a') as log:
                                child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                                         stdout=log, stderr=subprocess.STDOUT)
                            break
                        state('waiting_for_gpu_memory', job=job, completed_jobs=ordinal, free_mb=free)
                    time.sleep(15)
                state('running', job=job, completed_jobs=ordinal, child_pid=child.pid)
                if child.wait():
                    raise RuntimeError(f'SideQuest worker failed: {directory}')
            validate(directory, job)
            summary()
        state('complete', completed_jobs=len(jobs))
    except BaseException as exc:
        state('failed', job=job, error=str(exc))
        raise


if __name__ == '__main__':
    main()
