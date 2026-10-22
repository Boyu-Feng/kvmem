"""Run all latency conditions on GPU 7, reserved until inference finishes."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_legacy import atomic_json
from scripts.latency_runtime import METHODS, experiment_hashes, read_json
from scripts.run_idle_gpu_load_guard import gpu_samples, inspect_process, signal_owned
from scripts.run_latency_experiment import load_assets
from scripts.run_rebuttal_efficiency_queue import (
    claim_maintenance, identity_alive, process_identity, register_worker,
    release_maintenance,
)
from scripts.summarize_latency_experiment import summarize_runs, validate_run


def jobs():
    return [("fullkv", .5, "fullkv/r100")] + [
        (method, ratio, f"{method}/r{int(ratio * 100)}")
        for method in METHODS if method != "fullkv" for ratio in (.2, .5)
    ]


def terminate_child(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def main(argv=None):
    if argv is None and "--bootstrap" in sys.argv[1:]:
        return bootstrap(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-ready", type=Path, required=True)
    parser.add_argument("--data-ready", type=Path, required=True)
    parser.add_argument("--load-script", type=Path, required=True)
    parser.add_argument("--load-node", type=Path, required=True)
    manager = parser.add_mutually_exclusive_group(required=True)
    manager.add_argument("--load-supervisor-pid", type=int)
    manager.add_argument("--guard-pid", type=int)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("The CPU-only controller requires CUDA_VISIBLE_DEVICES=''")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    owner = process_identity(os.getpid())
    original = inspect_process(args.load_supervisor_pid)["identity"] if args.load_supervisor_pid else None
    hashes = experiment_hashes()
    load_assets(args.model_ready, args.data_ready)
    child, guardian, worker_identity = None, None, None
    reservation = args.load_node / "latency-reservation-gpu7.json"
    guard_state = args.load_node / "idle-load-guard.status.json"
    completed, directories = 0, []
    if (args.output / "queue_plan.json").exists():
        raise RuntimeError("Queue output was already used; choose a new directory")
    if args.guard_pid:
        info = inspect_process(args.guard_pid)
        if info["argv"][1:3] != ["-u", str(ROOT / "scripts/run_idle_gpu_load_guard.py")]:
            raise RuntimeError("Unexpected existing idle-load guard command")
        record = read_json(guard_state)
        if record["owner"] != info["identity"] or record["status"] != "running":
            raise RuntimeError("Existing idle-load guard is not healthy")
        guard_identity = info["identity"]
        guardian = SimpleNamespace(pid=args.guard_pid,
                                   poll=lambda: None if identity_alive(guard_identity) else 1)

    def state(status, **extra):
        atomic_json(args.output / "queue_status.json", {
            "status": status, "owner": owner, "updated": time.time(),
            "worker": worker_identity, "guardian": process_identity(guardian.pid) if guardian else None,
            "completed_conditions": completed, "total_conditions": len(jobs()), "gpu": 7, **extra,
        })

    def reserve():
        atomic_json(reservation, {"gpu": 7, "owner": owner, "child": worker_identity,
                                  "output": str(args.output), "updated": time.time()})

    def launch_guard(resume=False):
        command = [
            sys.executable, "-u", str(ROOT / "scripts/run_idle_gpu_load_guard.py"),
            "--load-script", str(args.load_script), "--node", str(args.load_node),
            "--reservation", str(reservation),
        ]
        if resume:
            command += ["--resume-record", str(args.load_node / "idle-load-handover.json")]
        else:
            command += ["--adopt-pid", str(original["pid"]), "--adopt-start-ticks", str(original["start_ticks"])]
        with (args.output / "idle-load-guard.log").open("a") as log:
            return subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (args.output / "queue.lock").open("a") as lock, (args.output.parent / "alfworld_gpu7.lock").open("a") as gpu_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if reservation.exists():
            prior = read_json(reservation)
            if identity_alive(prior.get("owner")) or identity_alive(prior.get("child")):
                raise RuntimeError("GPU 7 already has a live efficiency reservation")
        claim_maintenance(7, args.output, owner)
        reserve()
        try:
            atomic_json(args.output / "queue_plan.json", {
                "jobs": jobs(), "gpu": 7, "samples": 20, "repeats": 2,
                "source_hashes": hashes, "load_supervisor_original": original,
                "restoration": "original load when no reservation/experiment/compute PID and utilization=0",
            })
            if guardian is None:
                guardian = launch_guard()
            deadline = time.monotonic() + 90
            while True:
                if guardian.poll() is not None:
                    raise RuntimeError("Idle-load guard failed before GPU handover; inspect guard log")
                if guard_state.exists():
                    record = read_json(guard_state)
                    if record["owner"] == process_identity(guardian.pid) and record["status"] == "running":
                        snapshot = gpu_samples()[7]
                        if not snapshot["pids"] and snapshot["utilization"] == 0:
                            break
                if time.monotonic() > deadline:
                    raise RuntimeError("GPU 7 did not become exclusively idle after handover")
                state("waiting_for_exclusive_gpu7")
                time.sleep(1)
            gpu_uuid = gpu_samples()[7]["uuid"]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid,
                               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
            for method, ratio, name in jobs():
                if experiment_hashes() != hashes:
                    raise RuntimeError("Latency source changed while the queue was running")
                output = args.output / name
                command = [
                    sys.executable, "-u", str(ROOT / "scripts/run_latency_experiment.py"),
                    "--execute", "--method", method, "--ratio", str(ratio),
                    "--model-ready", str(args.model_ready), "--data-ready", str(args.data_ready),
                    "--output", str(output), "--samples", "20", "--repeats", "2",
                ]
                logs = args.output / "logs"
                logs.mkdir(exist_ok=True)
                with (logs / (name.replace("/", "__") + ".log")).open("a") as log:
                    bootstrap_command = [
                        sys.executable, "-u", str(Path(__file__).resolve()), "--bootstrap",
                        "--reservation", str(reservation), "--owner-pid", str(owner["pid"]),
                        "--owner-start-ticks", str(owner["start_ticks"]), "--", *command,
                    ]
                    child = subprocess.Popen(bootstrap_command, cwd=ROOT, env=environment,
                                             stdout=log, stderr=subprocess.STDOUT,
                                             pass_fds=(gpu_lock.fileno(),), start_new_session=True)
                    worker_identity = process_identity(child.pid)
                    if worker_identity is None:
                        raise RuntimeError("Latency worker exited before registration")
                    reserve()
                    while child.poll() is None:
                        if guardian.poll() is not None:
                            raise RuntimeError("Idle-load guard exited; aborting measurement safely")
                        if read_json(guard_state)["status"] != "running":
                            raise RuntimeError("Idle-load guard is unhealthy; aborting measurement safely")
                        state("running", job=name, command=command)
                        time.sleep(1)
                if child.returncode:
                    raise RuntimeError(f"Latency condition {name} exited {child.returncode}; inspect logs")
                validate_run(output)
                directories.append(output)
                completed += 1
                child, worker_identity = None, None
                reserve()
            reservation.unlink()
            release_maintenance(7, owner)
            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
            state("profiling_complete_restoring_load")
            summarize_runs(directories, args.output / "report")
            state("complete", report=str(args.output / "report"))
        except BaseException:
            terminate_child(child)
            state("failed", traceback=traceback.format_exc())
            raise
        finally:
            if reservation.exists():
                prior = read_json(reservation)
                if prior.get("owner") == owner and not identity_alive(prior.get("child")):
                    reservation.unlink()
            release_maintenance(7, owner)
            if original is not None and guardian is not None and guardian.poll() is not None:
                if identity_alive(original):
                    signal_owned(original, signal.SIGCONT)
                elif (args.load_node / "idle-load-handover.json").exists():
                    guardian = launch_guard(resume=True)
                    time.sleep(2)
                    if guardian.poll() is not None:
                        raise RuntimeError("Load guard recovery failed; inspect guard log")


def bootstrap(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--owner-pid", type=int, required=True)
    parser.add_argument("--owner-start-ticks", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    owner = {"pid": args.owner_pid, "start_ticks": args.owner_start_ticks}
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Worker command required")
    register_worker(7, owner, command)
    record = read_json(args.reservation)
    if record["owner"] != owner or not identity_alive(owner):
        raise RuntimeError("Controller died before child registration")
    atomic_json(args.reservation, {**record, "child": process_identity(os.getpid()), "updated": time.time()})
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
