"""Adopt existing own load workers; start missing loads only on unreserved idle GPUs."""
from __future__ import annotations

import argparse
import fcntl
import json
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
from scripts.run_rebuttal_efficiency_queue import identity_alive, process_identity

STOP = False
GPU7_UUID = "GPU-b4756b2a-638d-5f58-3f10-995be24f3f4e"


def inspect_process(pid):
    path = Path("/proc") / str(pid)
    if path.stat().st_uid != os.getuid():
        raise PermissionError(f"Refusing another user's process: {pid}")
    identity = process_identity(pid)
    if identity is None:
        raise RuntimeError(f"Process is not alive: {pid}")
    return {
        "identity": identity,
        "argv": [os.fsdecode(item) for item in (path / "cmdline").read_bytes().split(b"\0") if item],
        "cwd": str((path / "cwd").resolve()),
    }


def signal_owned(identity, signum):
    if not identity_alive(identity):
        raise RuntimeError(f"Process identity changed before signal: {identity}")
    fd = os.pidfd_open(identity["pid"])
    try:
        if not identity_alive(identity):
            raise RuntimeError("Process identity changed after opening pidfd")
        if (Path("/proc") / str(identity["pid"])).stat().st_uid != os.getuid():
            raise PermissionError("Refusing another user's process")
        signal.pidfd_send_signal(fd, signum)
    finally:
        os.close(fd)


def stop_worker(identity):
    if not identity_alive(identity):
        return
    signal_owned(identity, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while identity_alive(identity) and time.monotonic() < deadline:
        time.sleep(.1)
    if identity_alive(identity):
        signal_owned(identity, signal.SIGKILL)
    deadline = time.monotonic() + 10
    while identity_alive(identity) and time.monotonic() < deadline:
        time.sleep(.1)
    if identity_alive(identity):
        raise RuntimeError("Own load worker did not exit")


def gpu_samples():
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=15)
    samples = {}
    for line in raw.splitlines():
        index, uuid, util, free = [value.strip() for value in line.split(",")]
        samples[int(index)] = {"uuid": uuid, "utilization": int(util), "free_mib": int(free), "pids": []}
    raw = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], text=True, timeout=15)
    by_uuid = {sample["uuid"]: sample for sample in samples.values()}
    for line in raw.splitlines():
        uuid, pid = [value.strip() for value in line.split(",")]
        if uuid in by_uuid:
            by_uuid[uuid]["pids"].append(int(pid))
    return samples


def reservation_active(path):
    if not path.exists():
        return False
    if path.stat().st_uid != os.getuid():
        raise PermissionError("Reservation file belongs to another user")
    record = json.loads(path.read_text())
    if record.get("gpu") != 7:
        raise RuntimeError("This guard only permits stopping the explicitly authorized GPU 7 load")
    return identity_alive(record.get("owner")) or identity_alive(record.get("child"))


def experiment_bindings(samples, load_script):
    assigned = set()
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid() or process_identity(path.name) is None:
                continue
            argv = [os.fsdecode(item) for item in (path / "cmdline").read_bytes().split(b"\0") if item]
            if str(load_script) in argv:
                continue
            if not argv or "python" not in Path(argv[0]).name:
                continue
            environment = (path / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        visible = next((os.fsdecode(item.split(b"=", 1)[1]) for item in environment
                        if item.startswith(b"CUDA_VISIBLE_DEVICES=")), "")
        selectors = set(visible.split(","))
        for gpu, sample in samples.items():
            if str(gpu) in selectors or sample["uuid"] in selectors:
                assigned.add(gpu)
    return assigned


def can_start_load(sample, reserved, assigned):
    return not reserved and not assigned and not sample["pids"] and sample["utilization"] == 0


def adopt_supervisor(identity, script, checkpoint):
    supervisor = inspect_process(identity["pid"])
    if supervisor["identity"] != identity or supervisor["argv"][1:] != ["-B", str(script), "supervisor"]:
        raise RuntimeError("Unexpected load supervisor identity/command")
    if ".service" in (Path("/proc") / str(identity["pid"]) / "cgroup").read_text():
        raise RuntimeError("Refusing direct systemd handover: configure the node-local service first; "
                           "control-group cleanup would interrupt all workers")
    environment = {
        os.fsdecode(item.split(b"=", 1)[0]): os.fsdecode(item.split(b"=", 1)[1])
        for item in (Path("/proc") / str(identity["pid"]) / "environ").read_bytes().split(b"\0") if b"=" in item
    }
    signal_owned(identity, signal.SIGSTOP)
    committed = False
    try:
        workers = {}
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                if path.stat().st_uid != os.getuid():
                    continue
                fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[1]) != identity["pid"]:
                    continue
                worker = inspect_process(int(path.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
            args = worker["argv"]
            if len(args) != 6 or args[1:5] != ["-B", str(script), "worker", "--gpu"]:
                raise RuntimeError("Unexpected child of the load supervisor; refuse handover")
            gpu = int(args[5])
            if gpu in workers:
                raise RuntimeError("Duplicate existing GPU load workers")
            workers[gpu] = worker
        if set(workers) != set(range(8)):
            raise RuntimeError("Expected eight live own load workers for a no-interruption handover")
        samples = gpu_samples()
        if samples[7]["uuid"] != GPU7_UUID or set(samples[7]["pids"]) != {workers[7]["identity"]["pid"]}:
            raise RuntimeError("GPU 7 is no longer load-only; refuse to stop anything")
        atomic_json(checkpoint, {"original": identity, "script": str(script), "workers": workers})
        # Graceful exit runs the old supervisor's finally block and kills every
        # worker. Retire only its control process, preserving all adopted workers.
        signal_owned(identity, signal.SIGKILL)
        committed = True
        return workers, environment
    finally:
        if not committed and identity_alive(identity):
            signal_owned(identity, signal.SIGCONT)


def resume_workers(checkpoint, script):
    if checkpoint.stat().st_uid != os.getuid():
        raise PermissionError("Handover record must belong to the current user")
    record = json.loads(checkpoint.read_text())
    if record["script"] != str(script) or identity_alive(record["original"]):
        raise RuntimeError("Cannot resume while the original supervisor exists or program differs")
    workers = {int(gpu): worker for gpu, worker in record["workers"].items()}
    if set(workers) != set(range(8)):
        raise RuntimeError("Incomplete handover checkpoint")
    environment = None
    for gpu, worker in workers.items():
        if len(worker["argv"]) != 6 or worker["argv"][1:] != ["-B", str(script), "worker", "--gpu", str(gpu)]:
            raise RuntimeError("Invalid recorded worker command")
        if identity_alive(worker["identity"]):
            actual = inspect_process(worker["identity"]["pid"])
            if actual != worker:
                raise RuntimeError("Recorded worker identity/command/cwd changed")
            if environment is None:
                raw = (Path("/proc") / str(worker["identity"]["pid"]) / "environ").read_bytes()
                environment = {os.fsdecode(item.split(b"=", 1)[0]): os.fsdecode(item.split(b"=", 1)[1])
                               for item in raw.split(b"\0") if b"=" in item}
    if environment is None:
        raise RuntimeError("No surviving own load worker from which to recover the private environment")
    return workers, environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-script", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--adopt-pid", type=int)
    parser.add_argument("--adopt-start-ticks", type=int)
    parser.add_argument("--resume-record", type=Path)
    parser.add_argument("--reservation", type=Path, required=True)
    args = parser.parse_args(argv)
    script, node = args.load_script.resolve(), args.node.resolve()
    if script.stat().st_uid != os.getuid() or node.stat().st_uid != os.getuid():
        raise PermissionError("Load program/storage must belong to the current user")
    args.reservation = args.reservation.resolve()
    if not args.resume_record and not reservation_active(args.reservation):
        raise RuntimeError("A live GPU 7 efficiency reservation is required before handover")
    if not args.resume_record and (args.adopt_pid is None or args.adopt_start_ticks is None):
        parser.error("Supply supervisor PID/start ticks or --resume-record")
    owner = process_identity(os.getpid())
    original = {"pid": args.adopt_pid, "start_ticks": args.adopt_start_ticks}
    state_path = node / "idle-load-guard.status.json"
    workers, children = {}, {}

    def state(status, **extra):
        atomic_json(state_path, {"status": status, "owner": owner, "updated": time.time(),
                                "workers": {str(gpu): worker["identity"] for gpu, worker in workers.items()},
                                "reservation": str(args.reservation), **extra})

    def stop(signum, frame):
        global STOP
        STOP = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (node / "idle-load-guard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            checkpoint = node / "idle-load-handover.json"
            if args.resume_record:
                workers, environment = resume_workers(args.resume_record, script)
            else:
                workers, environment = adopt_supervisor(original, script, checkpoint)
            state("adopted", preserved_other_gpu_workers=True)
            while not STOP:
                try:
                    tick(workers, children, args.reservation, script, node, environment, state)
                except (OSError, subprocess.SubprocessError, ValueError, RuntimeError):
                    error = traceback.format_exc()
                    print(error, flush=True)
                    state("monitor_failed_retrying", traceback=error)
                time.sleep(1)
            state("stopped_workers_left_running")
        except BaseException:
            state("failed", traceback=traceback.format_exc())
            raise


def tick(workers, children, reservation, script, node, environment, state):
    for gpu, child in list(children.items()):
        code = child.poll()
        if code is not None:
            if code:
                print(f"Load worker GPU {gpu} exited {code}; see original gpu-{gpu}.log", flush=True)
            del children[gpu]
    reserved = reservation_active(reservation)
    if reserved and 7 in workers and identity_alive(workers[7]["identity"]):
        stop_worker(workers[7]["identity"])
        state("gpu7_reserved", stopped_only_gpu=7)
    samples = gpu_samples()
    assigned = experiment_bindings(samples, script)
    for gpu, sample in samples.items():
        if gpu in workers and identity_alive(workers[gpu]["identity"]):
            continue
        if can_start_load(sample, gpu == 7 and reserved, gpu in assigned):
            command = [workers[gpu]["argv"][0], "-B", str(script), "worker", "--gpu", str(gpu)]
            with (node / f"gpu-{gpu}.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(command, cwd=node, env=environment, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            identity = process_identity(child.pid)
            if identity is None:
                raise RuntimeError("New load worker exited before identity registration")
            children[gpu] = child
            workers[gpu] = {"identity": identity, "argv": command, "cwd": str(node)}
            checkpoint = node / "idle-load-handover.json"
            record = json.loads(checkpoint.read_text())
            atomic_json(checkpoint, {**record, "workers": workers})
            print(f"Restored original load on idle GPU {gpu}, pid={child.pid}", flush=True)
    state("running", gpu7_reserved=reserved, assigned_experiment_gpus=sorted(assigned),
          samples=samples, policy="existing workers preserved; missing loads start only at "
                                 "utilization=0 with no compute PIDs, assigned experiments or reservation")


if __name__ == "__main__":
    main()
