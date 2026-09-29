"""Schedule the seven legacy W3 measurements after existing accuracy work drains.

Maintenance is scoped to one GPU. A live owner OR its registered profiling
child holds the gate; PID start ticks prevent a reused PID from holding it.
This module is also imported by the accuracy controller immediately before
new launches. It never interrupts an existing accuracy worker.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_legacy import atomic_json

DEFAULT_OUTPUT = ROOT / "results/rebuttal_20260928/efficiency_legacy_v1"
WORKER = ROOT / "scripts/run_rebuttal_efficiency.py"
PROC = Path("/proc")


def maintenance_path(gpu):
    return ROOT / f"results/rebuttal_20260928/maintenance_gpu{gpu}.json"


@contextlib.contextmanager
def maintenance_launch_lock(gpu):
    """Serialize maintenance publication with every new accuracy launch."""
    path = maintenance_path(gpu).with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def process_identity(pid, proc_root=PROC):
    """Return a live Linux process identity; zombies cannot hold maintenance."""
    try:
        pid = int(pid)
        if pid <= 0:
            return None
        # comm can contain spaces and ')'; fields after its final ')' start
        # at field 3 (state), so starttime (field 22) has index 19 here.
        fields = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X", "x"):
            return None
        return {"pid": pid, "start_ticks": int(fields[19])}
    except (OSError, ValueError, TypeError, IndexError):
        return None


def identity_alive(identity, proc_root=PROC):
    if not isinstance(identity, dict):
        return False
    try:
        expected = {"pid": int(identity["pid"]), "start_ticks": int(identity["start_ticks"])}
    except (KeyError, ValueError, TypeError):
        return False
    return process_identity(expected["pid"], proc_root) == expected


def read_maintenance(path):
    try:
        record = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    if not isinstance(record, dict) or record.get("version") != 1:
        raise RuntimeError(f"Invalid maintenance record: {path}")
    return record


def maintenance_blockers(path, proc_root=PROC):
    record = read_maintenance(path)
    if record is None:
        return {}
    # A dead owner's live child must continue to block accuracy launches.
    return {role: record[role] for role in ("owner", "child")
            if identity_alive(record.get(role), proc_root)}


def claim_maintenance(gpu, output, owner):
    path = maintenance_path(gpu)
    with maintenance_launch_lock(gpu):
        blockers = maintenance_blockers(path)
        if blockers:
            raise RuntimeError(f"Another efficiency process holds maintenance: {blockers}")
        atomic_json(path, {"version": 1, "gpu": gpu, "owner": owner, "child": None,
                          "output": str(output), "updated": time.time()})


def release_maintenance(gpu, owner):
    """Never remove another owner's lease or expose a still-running child."""
    path = maintenance_path(gpu)
    with maintenance_launch_lock(gpu):
        record = read_maintenance(path)
        if record is None:
            return True
        if record.get("owner") != owner:
            return False
        if identity_alive(record.get("child")):
            return False
        path.unlink(missing_ok=True)
        return True


def register_worker(gpu, owner, command):
    """Child-side registration before exec closes the Popen/owner-death race."""
    path = maintenance_path(gpu)
    with maintenance_launch_lock(gpu):
        record = read_maintenance(path)
        if record is None or record.get("owner") != owner or not identity_alive(owner):
            raise RuntimeError("Efficiency owner exited before child registration; aborting launch")
        if identity_alive(record.get("child")):
            raise RuntimeError("A profiling worker is already registered")
        child = process_identity(os.getpid())
        if child is None:
            raise RuntimeError("Cannot identify profiling child")
        record.update(child=child, child_command=command, updated=time.time())
        atomic_json(path, record)
    return child


def bootstrap_main(argv):
    parser = argparse.ArgumentParser(description="Internal profiling child registration")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--owner-pid", type=int, required=True)
    parser.add_argument("--owner-start-ticks", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Worker command required")
    owner = {"pid": args.owner_pid, "start_ticks": args.owner_start_ticks}
    register_worker(args.gpu, owner, command)
    os.execvpe(command[0], command, os.environ)


def accuracy_workers(gpu):
    """Include workers that exist but have not opened a CUDA context yet."""
    names = {str(ROOT / "run_rebuttal_legacy.py"), str(ROOT / "run_rebuttal_legacy_step.py")}
    workers = []
    for directory in PROC.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            command = [s.decode() for s in (directory / "cmdline").read_bytes().split(b"\0") if s]
            if not names.intersection(command):
                continue
            environment = (directory / "environ").read_bytes().split(b"\0")
            visible = next((s.split(b"=", 1)[1].decode() for s in environment
                            if s.startswith(b"CUDA_VISIBLE_DEVICES=")), None)
            if visible is not None and str(gpu) not in visible.split(","):
                continue
            identity = process_identity(directory.name)
            if identity:
                workers.append(identity)
        except (OSError, UnicodeError):
            continue
    return workers


def gpu_snapshot(gpu):
    output = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True, timeout=15).strip().splitlines()
    if len(output) != 1:
        raise RuntimeError("Expected exactly one profiling GPU")
    uuid, free = [s.strip() for s in output[0].split(",")]
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, timeout=15)
    peers = []
    for line in processes.splitlines():
        fields = [s.strip() for s in line.split(",")]
        if len(fields) == 2 and fields[0] == uuid:
            peers.append(int(fields[1]))
    return {"uuid": uuid, "free_mib": int(free), "compute_pids": peers}


def jobs():
    return [dict(method="fullkv", ratio=.5, display_budget=1.0, name="fullkv/r100")] + [
        dict(method=method, ratio=ratio, display_budget=ratio,
             name=f"{method}/r{int(ratio * 100)}")
        for method in ("stepkv", "h2o", "tova") for ratio in (.2, .5)]


def validate_completed(directory, job, expected_ids=None):
    status = json.loads((directory / "status.json").read_text())
    protocol = json.loads((directory / "manifest.json").read_text())["protocol"]
    rows = json.loads((directory / "measurements.json").read_text())["rows"]
    if status.get("status") != "complete" or len(rows) != 20:
        raise RuntimeError(f"Incomplete profiling output: {directory}")
    if any(protocol.get(k) != v for k, v in dict(method=job["method"], ratio=job["ratio"],
                                                seed=233, repeats=2).items()):
        raise RuntimeError(f"Profiling protocol mismatch: {directory}")
    sample_ids = protocol.get("sample_ids", [])
    if len(sample_ids) != 20 or len(set(sample_ids)) != 20 or None in sample_ids:
        raise RuntimeError(f"Profiling sample count mismatch: {directory}")
    if [row.get("id") for row in rows] != sample_ids:
        raise RuntimeError(f"Profiling result IDs differ from the manifest: {directory}")
    if expected_ids is not None and sample_ids != expected_ids:
        raise RuntimeError(f"Profiling conditions use different sample IDs: {directory}")
    if any(not row.get("parity") or len(row.get("clean_elapsed_seconds", [])) != 2 for row in rows):
        raise RuntimeError(f"Missing clean repeats or output/KV parity: {directory}")
    return sample_ids


def update_summary(output):
    subprocess.run([sys.executable, str(ROOT / "scripts/summarize_rebuttal_efficiency.py"),
                    "--root", str(output)], cwd=ROOT, check=True)


def stop_own_child(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=[0], default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-free-mb", type=int, default=30000)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.min_free_mb < 30000 or not 5 <= args.poll_seconds <= 10:
        parser.error("Require at least 30000 MiB free and a 5–10 second polling interval")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = process_identity(os.getpid())
    if owner is None:
        raise RuntimeError("Cannot identify efficiency supervisor")
    plan = dict(version="legacy_efficiency_v1", gpu=args.gpu, seed=233,
                model="Qwen2.5-7B-Instruct", samples=20, repeats=2, jobs=jobs(),
                minimum_free_mib=args.min_free_mb, worker=str(WORKER),
                maintenance=str(maintenance_path(args.gpu)))
    plan_path = args.output / "queue_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Efficiency queue plan changed; use a new output directory")
    atomic_json(plan_path, plan)

    def state(status, **extra):
        atomic_json(args.output / "queue_status.json", dict(status=status, owner=owner,
            supervisor_pid=owner["pid"], updated=time.time(), total_jobs=len(plan["jobs"]), **extra))

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    child, job, claimed = None, None, False
    completed, sample_ids = 0, None
    try:
        claim_maintenance(args.gpu, args.output, owner)
        claimed = True
        for ordinal, job in enumerate(plan["jobs"]):
            directory = args.output / job["name"]
            status_path = directory / "status.json"
            if status_path.exists() and json.loads(status_path.read_text()).get("status") == "complete":
                sample_ids = validate_completed(directory, job, sample_ids)
                completed = ordinal + 1
                update_summary(args.output)
                continue
            while True:
                active = accuracy_workers(args.gpu)
                if active:
                    state("waiting_for_accuracy_worker", job=job, completed_jobs=completed,
                          accuracy_workers=active)
                else:
                    gpu = gpu_snapshot(args.gpu)
                    if not gpu["compute_pids"] and gpu["free_mib"] >= args.min_free_mb:
                        break
                    state("waiting_for_exclusive_gpu", job=job, completed_jobs=completed, gpu=gpu)
                time.sleep(args.poll_seconds)
            directory.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", str(WORKER), "--method", job["method"],
                       "--ratio", str(job["ratio"]), "--samples", "20", "--repeats", "2",
                       "--output", str(directory)]
            bootstrap = [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker-bootstrap",
                         "--gpu", str(args.gpu), "--owner-pid", str(owner["pid"]),
                         "--owner-start-ticks", str(owner["start_ticks"]), "--", *command]
            with (directory / "run.log").open("a") as log:
                child = subprocess.Popen(bootstrap, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while child.poll() is None:
                    state("running", job=job, completed_jobs=completed, child_pid=child.pid,
                          child=process_identity(child.pid))
                    time.sleep(args.poll_seconds)
                if child.returncode:
                    raise RuntimeError(f"Profiling worker exited {child.returncode}: {directory / 'run.log'}")
            sample_ids = validate_completed(directory, job, sample_ids)
            completed = ordinal + 1
            child = None
            update_summary(args.output)
        released = release_maintenance(args.gpu, owner)
        claimed = not released
        state("complete", completed_jobs=completed, maintenance_released=released)
    except BaseException as exc:
        try:
            stop_own_child(child)
        finally:
            released = release_maintenance(args.gpu, owner) if claimed else True
            claimed = claimed and not released
            state("failed", job=job, completed_jobs=completed, error=str(exc),
                  maintenance_released=released,
                  child=process_identity(child.pid) if child is not None else None)
        raise
    finally:
        if claimed:
            release_maintenance(args.gpu, owner)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--_worker-bootstrap"]:
        bootstrap_main(sys.argv[2:])
    else:
        main()
