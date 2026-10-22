"""Append two datasets after an existing batch, without interrupting it or restoring load between batches."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_legacy import atomic_json
from scripts.run_idle_gpu_load_guard import GPU7_UUID
from scripts.run_rebuttal_efficiency_queue import identity_alive, process_identity


def hold_main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-pid", type=int, required=True)
    parser.add_argument("--owner-start-ticks", type=int, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args(argv)
    owner = {"pid": args.owner_pid, "start_ticks": args.owner_start_ticks}
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU7_UUID:
        raise RuntimeError("The continuation reservation is scoped only to the authorized GPU 7")
    atomic_json(args.status, {"owner": owner, "holder": process_identity(os.getpid()), "gpu": 7,
                             "purpose": "Assigned continuation reservation; no CUDA import or context"})
    # The existing guard checks assigned Python experiment processes as well as
    # compute PIDs. This stdlib-only holder closes the gap between two queues.
    while identity_alive(owner):
        time.sleep(1)


def jobs():
    from scripts.latency_datasets import ADDITIONAL_DATASETS, legacy_jobs

    return [(method, ratio, f"{dataset}/{name}")
            for dataset in ADDITIONAL_DATASETS for method, ratio, name in legacy_jobs()]


def worker_command(command, output):
    from scripts.latency_datasets import ADDITIONAL_DATASETS

    if not isinstance(command, (list, tuple)):
        return command
    command = list(command)
    worker = str(ROOT / "scripts/run_latency_experiment.py")
    if worker not in command:
        return command
    index = command.index(worker)
    arguments = command[index + 1:]
    path = Path(arguments[arguments.index("--output") + 1]).resolve()
    dataset = path.relative_to(Path(output).resolve()).parts[0]
    if dataset not in ADDITIONAL_DATASETS:
        raise RuntimeError("Unexpected dataset in additional latency worker output")
    command[index] = str(ROOT / "scripts/run_multidataset_latency.py")
    return command + ["--dataset", dataset]


def summarize_datasets(directories, output, reporter=None):
    from scripts.latency_datasets import ADDITIONAL_DATASETS
    from scripts.latency_runtime import read_json
    if reporter is None:
        from scripts.summarize_latency_experiment import summarize_runs
        reporter = summarize_runs
    grouped = {dataset: [] for dataset in ADDITIONAL_DATASETS}
    for directory in directories:
        dataset = read_json(directory / "manifest.json")["protocol"]["dataset"]
        if dataset not in grouped:
            raise RuntimeError(f"Unexpected dataset in additional latency report: {dataset}")
        grouped[dataset].append(directory)
    summaries = {}
    for dataset, runs in grouped.items():
        if len(runs) != 15:
            raise RuntimeError(f"Incomplete condition matrix for {dataset}")
        summaries[dataset] = reporter(runs, Path(output) / dataset)
    atomic_json(Path(output) / "DATASET_SUMMARIES.json", summaries)
    return summaries


def main(argv=None):
    if argv is None and sys.argv[1:2] == ["--hold-reservation"]:
        return hold_main(sys.argv[2:])
    from scripts import latency_runtime as runtime
    from scripts import run_latency_experiment_queue as queue
    from scripts.latency_datasets import ADDITIONAL_DATASETS, experiment_hashes, normalized_assets, select_samples

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-ready", type=Path, required=True)
    parser.add_argument("--data-ready", type=Path, required=True)
    parser.add_argument("--load-script", type=Path, required=True)
    parser.add_argument("--load-node", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("The continuation controller must be CPU-only")
    args.output, args.after_output = args.output.resolve(), args.after_output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    owner = process_identity(os.getpid())
    holder = None
    hashes = experiment_hashes()
    predecessor = runtime.read_json(args.after_output / "queue_status.json")
    predecessor_plan = runtime.read_json(args.after_output / "queue_plan.json")
    if predecessor_plan["source_hashes"] != runtime.experiment_hashes():
        raise RuntimeError("The current HotpotQA source binding changed; refusing continuation")
    queue.load_assets(args.model_ready, args.data_ready)
    import run_all_wiki_experiments_v2 as base
    samples = {}
    for dataset in ADDITIONAL_DATASETS:
        _, rows, _ = normalized_assets(args.data_ready, dataset)
        selected, warmups = select_samples(base, rows, 20, dataset)
        samples[dataset] = {"sample_ids": [row["id"] for _, row in selected],
                            "warmup_ids": [row["id"] for _, row in warmups]}

    def state(status, **extra):
        atomic_json(args.output / "continuation_status.json", {
            "status": status, "owner": owner, "holder": process_identity(holder.pid) if holder else None,
            "predecessor": str(args.after_output), "datasets": list(ADDITIONAL_DATASETS),
            "total_conditions": 30, "updated": time.time(), **extra,
        })

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (args.output.parent / "latency_additional_gpu7.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if (args.output / "continuation_plan.json").exists():
                raise RuntimeError("Continuation output was already used; choose a new directory")
            atomic_json(args.output / "continuation_plan.json", {
                "source_hashes": hashes, "predecessor": str(args.after_output),
                "predecessor_owner": predecessor["owner"], "jobs": jobs(),
                "samples_per_condition": 20, "clean_repeats": 2, "diagnostic_repeats": 1,
                "datasets": samples, "gpu": 7,
            })
            with (args.output / "reservation_holder.log").open("a") as log:
                holder = subprocess.Popen([
                    sys.executable, "-u", str(Path(__file__).resolve()), "--hold-reservation",
                    "--owner-pid", str(owner["pid"]), "--owner-start-ticks", str(owner["start_ticks"]),
                    "--status", str(args.output / "gpu7_continuation_reservation.json"),
                ], env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPU7_UUID),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state("waiting_for_hotpotqa")
            while identity_alive(predecessor["owner"]) or identity_alive(predecessor.get("worker")):
                if holder.poll() is not None:
                    raise RuntimeError("Continuation reservation holder exited unexpectedly")
                current = runtime.read_json(args.after_output / "queue_status.json")
                if current.get("owner") != predecessor["owner"]:
                    raise RuntimeError("Predecessor owner changed while waiting")
                if current["status"] == "failed":
                    raise RuntimeError("HotpotQA failed; additional datasets blocked, not silently skipped")
                if experiment_hashes() != hashes:
                    raise RuntimeError("Additional latency source changed while waiting")
                state("waiting_for_hotpotqa", predecessor_job=current.get("job"),
                      predecessor_completed_conditions=current["completed_conditions"])
                time.sleep(5)
            current = runtime.read_json(args.after_output / "queue_status.json")
            if current["status"] != "complete" or current["completed_conditions"] != 15:
                raise RuntimeError("Predecessor exited without completing its 15 conditions")
            if experiment_hashes() != hashes or holder.poll() is not None:
                raise RuntimeError("Source or continuation reservation changed before launch")
            original_popen, original_atomic = queue.subprocess.Popen, queue.atomic_json
            original_assets = queue.load_assets

            def popen(command, *positional, **kwargs):
                return original_popen(worker_command(command, args.output), *positional, **kwargs)

            def write(path, record):
                if Path(path).name == "queue_status.json" and "command" in record:
                    record = {**record, "command": worker_command(record["command"], args.output),
                              "dataset": record["job"].split("/", 1)[0]}
                original_atomic(path, record)

            def assets(model_ready, data_ready):
                result = original_assets(model_ready, data_ready)
                for dataset in ADDITIONAL_DATASETS:
                    normalized_assets(data_ready, dataset)
                return result

            def summarize(directories, output):
                queue.terminate_child(holder)
                state("profiling_complete_restoring_load")
                summarize_datasets(directories, output)

            guard = runtime.read_json(args.load_node / "idle-load-guard.status.json")["owner"]
            state("running_additional_datasets")
            with contextlib.ExitStack() as scope:
                for obj, name, value in (
                    (queue, "jobs", jobs), (queue, "experiment_hashes", experiment_hashes),
                    (queue, "load_assets", assets), (queue, "summarize_runs", summarize),
                    (queue, "atomic_json", write), (queue.subprocess, "Popen", popen),
                ):
                    scope.enter_context(runtime.patch_attribute(obj, name, value))
                queue.main([
                    "--output", str(args.output), "--model-ready", str(args.model_ready),
                    "--data-ready", str(args.data_ready), "--load-script", str(args.load_script),
                    "--load-node", str(args.load_node), "--guard-pid", str(guard["pid"]),
                ])
            state("complete")
        except BaseException:
            state("failed", traceback=traceback.format_exc())
            raise
        finally:
            queue.terminate_child(holder)


if __name__ == "__main__":
    main()
