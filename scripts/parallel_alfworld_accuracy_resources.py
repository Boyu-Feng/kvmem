"""Exact-identity admission and recovery for the separate ALFWorld sidecar."""
from __future__ import annotations

import contextlib
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts import run_deadline_gpu_load_guard as guard
from scripts.reuse_semantic_resources import script as python_entry
from scripts.run_alfworld_rebuttal import ROOT, atomic_json, read_json
from scripts.run_rebuttal_efficiency_queue import (
    identity_alive, maintenance_launch_lock, maintenance_path, process_identity,
    read_maintenance, register_worker, release_maintenance,
)

CAPACITY = {0: 3, 1: 2}
CAP_GIB = 28
MIN_FREE_GIB = 32


@contextlib.contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def write_ownership(sidecar, state):
    path = sidecar / "ownership.json"
    atomic_json(path, state)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    directory = os.open(sidecar, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def first_python_entry(argv):
    if not argv or not Path(argv[0]).name.startswith("python"):
        raise RuntimeError("Not a Python entry point")
    for arg in argv[1:]:
        if arg in ("-B", "-u"):
            continue
        if arg.startswith("-"):
            raise RuntimeError("Unapproved Python invocation")
        return str(Path(arg).resolve())
    raise RuntimeError("Missing Python entry point")


def exact_info(expected, commands=None):
    token = expected["identity"]
    if not identity_alive(token):
        return None
    try:
        actual = guard.inspect(token["pid"])
    except (FileNotFoundError, ProcessLookupError, RuntimeError):
        if not identity_alive(token):
            return None
        raise
    for key in ("identity", "uid", "cwd"):
        if actual[key] != expected[key]:
            raise RuntimeError(f"Owned process {key} changed; refusing to signal")
    if actual["uid"] != os.getuid() or actual["cwd"] != str(ROOT):
        raise RuntimeError("Process is not owned in the exact repository")
    allowed = [expected["argv"]] if commands is None else commands
    if actual["argv"] not in allowed:
        raise RuntimeError("Owned process argv changed; refusing to signal")
    return actual


def validate_controller(info):
    actual = exact_info(info)
    if actual is None:
        raise RuntimeError("Original CPU controller exited")
    if first_python_entry(actual["argv"]) != str(ROOT / "scripts/run_alfworld_recent_queue.py"):
        raise RuntimeError("Not the original ALFWorld serial controller")
    if guard.gpu_selector(info["identity"]["pid"]) != "":
        raise RuntimeError("Original controller is not CPU-only")
    if guard.stopped(info["identity"]["pid"]):
        raise RuntimeError("Original controller already stopped; ownership is unknown")
    return actual


def suspend_controller(sidecar, info):
    with file_lock(sidecar / "cleanup.lock"):
        validate_controller(info)
        state = read_json(sidecar / "ownership.json")
        if not state.get("recovery_armed"):
            raise RuntimeError("Recovery must be armed before suspension")
        state.update(suspend_intent=True, originally_running=True, controller=info)
        write_ownership(sidecar, state)
        guard.signal_exact(info, signal.SIGSTOP)
        until = time.monotonic() + 5
        while not guard.stopped(info["identity"]["pid"]):
            if exact_info(info) is None or time.monotonic() >= until:
                raise RuntimeError("Original CPU controller did not stop")
            time.sleep(.01)
        state["suspended"] = True
        write_ownership(sidecar, state)


def resume_controller(sidecar, state):
    if state.get("resumed") or not state.get("suspend_intent"):
        return
    if not state.get("originally_running"):
        raise RuntimeError("Refusing to resume an originally stopped controller")
    info = exact_info(state["controller"])
    if info is None:
        raise RuntimeError("Original controller disappeared before resume")
    if first_python_entry(info["argv"]) != str(ROOT / "scripts/run_alfworld_recent_queue.py"):
        raise RuntimeError("Unexpected controller entry point before resume")
    # A recovery crash after SIGCONT but before journal publication must not
    # issue another SIGCONT; the running state is an idempotent completion.
    if guard.stopped(info["identity"]["pid"]):
        guard.signal_exact(info, signal.SIGCONT)
    state["resumed"] = True
    write_ownership(sidecar, state)


def owned_marker(marker):
    return exact_info(marker["info"], [marker["bootstrap"], marker["command"]])


def gpu_snapshot():
    from scripts.run_alfworld_rebuttal_queue import gpu_snapshot as sample
    devices = {d["index"]: {**d, "pids": []} for d in sample([0, 1])}
    text = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=20)
    for line in text.splitlines():
        pid, uuid = [part.strip() for part in line.split(",")]
        for device in devices.values():
            if device["uuid"] == uuid:
                device["pids"].append(int(pid))
    return devices


def pending_cuda_candidate(argv):
    entry = python_entry(argv)
    if entry is None:
        return False
    workers = set(guard.WORKERS.values())
    workers.update({
        "run_alfworld_rebuttal.py", "run_alfworld_recent_rebuttal.py",
        "run_alfworld_schedule_rebuttal.py", "run_latency_experiment.py",
        "run_multidataset_latency.py", "run_unified_latency.py",
        "run_sidequest_latency.py", "run_thinkkv_latency.py", "gpu_load.py",
    })
    if Path(entry).name in workers:
        return True
    if entry == "-m":
        module = argv[argv.index("-m") + 1] if argv.index("-m") + 1 < len(argv) else ""
        if module in {"torch.distributed.run", "vllm.entrypoints.openai.api_server",
                      "sglang.launch_server", "accelerate.commands.launch"}:
            return True
    if entry == "-c":
        code = argv[argv.index("-c") + 1] if argv.index("-c") + 1 < len(argv) else ""
        if any(token in code for token in ("torch.cuda", ".cuda(", "cupy", "numba.cuda")):
            return True
    for index, arg in enumerate(argv[1:], start=1):
        if arg in {"--gpu", "--gpus", "--gpu-id", "--gpu-ids", "--cuda-device"}:
            return True
        if arg.startswith(("--gpu=", "--gpus=", "--gpu-id=", "--gpu-ids=", "--cuda-device=")):
            return True
        if arg.startswith(("--device=cuda", "--device-map=cuda")):
            return True
        if arg in {"--device", "--device-map"} and index + 1 < len(argv):
            if argv[index + 1].startswith("cuda"):
                return True
    return False


def pending_assignments(devices, proc=Path("/proc")):
    pending = []
    for directory in proc.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            # Inherited CUDA visibility is not evidence of a pending GPU job.
            if directory.stat().st_uid != os.getuid():
                continue
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] in ("Z", "X", "x"):
                continue
            argv = [os.fsdecode(v) for v in (directory / "cmdline").read_bytes().split(b"\0") if v]
            if not pending_cuda_candidate(argv):
                continue
            values = (directory / "environ").read_bytes().split(b"\0")
            visible = next((v.split(b"=", 1)[1] for v in values
                            if v.startswith(b"CUDA_VISIBLE_DEVICES=")), None)
            selectors = os.fsdecode(visible).split(",") if visible is not None else []
            # Unset means every device is potentially visible. CPU-only
            # controllers/bootstrap/recovery explicitly use the empty string.
            targets = list(CAPACITY) if visible is None else [
                g for g in CAPACITY if str(g) in selectors or devices[g]["uuid"] in selectors
            ]
            pending.extend((int(directory.name), gpu) for gpu in targets)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise RuntimeError(f"Unreadable GPU launch candidate: {directory.name}") from error
    return pending


def assert_admission(gpu, plan, devices, markers, pending, *, launching=True):
    if gpu not in CAPACITY or set(devices) != set(CAPACITY):
        raise RuntimeError("Parallel admission is limited to physical GPUs 0/1")
    if plan["capacity"] != {"0": 3, "1": 2} or plan["memory_limit_gib"] != CAP_GIB:
        raise RuntimeError("Unexpected per-GPU capacity/allocator cap")
    for index in CAPACITY:
        device = devices[index]
        if device["uuid"] != plan["gpu_uuids"][str(index)]:
            raise RuntimeError("Physical GPU UUID changed")
        if device["total_mib"] < CAPACITY[index] * CAP_GIB * 1024:
            raise RuntimeError("Aggregate allocator caps exceed physical VRAM")
        known = {}
        for marker in markers:
            if marker["gpu"] == index:
                actual = owned_marker(marker)
                if actual is not None:
                    known[actual["identity"]["pid"]] = marker
        unknown = set(device["pids"]) - set(known)
        if unknown:
            raise RuntimeError(f"Unknown CUDA processes on GPU{index}: {sorted(unknown)}")
        unknown_pending = [pid for pid, assigned in pending if assigned == index and pid not in known]
        if unknown_pending:
            raise RuntimeError(f"Unknown/pending GPU{index} candidates: {unknown_pending}")
        if len(known) > CAPACITY[index]:
            raise RuntimeError("Owned worker limit exceeded")
        if index == gpu and launching:
            if len(known) >= CAPACITY[index]:
                raise RuntimeError("No parallel worker slot available")
            if device["free_mib"] < MIN_FREE_GIB * 1024:
                raise RuntimeError("Less than 32 GiB free at admission")


def markers(sidecar):
    path = sidecar / "workers.json"
    return read_json(path)["workers"] if path.exists() else []


def publish_worker(sidecar, plan, owner, marker):
    gpu = marker["gpu"]
    with maintenance_launch_lock(gpu):
        record = read_maintenance(maintenance_path(gpu))
        if record is None or record.get("owner") != owner or not identity_alive(owner):
            raise RuntimeError("Sidecar owner/maintenance exited before CPU registration")
        if not identity_alive(record.get("child")):
            raise RuntimeError("Independent recovery sentinel is not alive")
        with file_lock(sidecar / "workers.lock"):
            values = markers(sidecar)
            if any(m["info"]["identity"] == marker["info"]["identity"] for m in values):
                raise RuntimeError("Duplicate child registration")
            if sum(m["gpu"] == gpu and identity_alive(m["info"]["identity"])
                   for m in values) >= CAPACITY[gpu]:
                raise RuntimeError("Worker capacity exceeded before CUDA")
            if exact_info(marker["info"]) is None:
                raise RuntimeError("CPU bootstrap exited before registration")
            values.append(marker)
            # Persist the complete list before allowing bootstrap to expose CUDA.
            atomic_json(sidecar / "workers.json", {"workers": values})
            record["parallel_children"] = [m for m in values if m["gpu"] == gpu]
            atomic_json(maintenance_path(gpu), record)


def arm_recovery(sidecar, owner):
    with file_lock(sidecar / "cleanup.lock"):
        state = read_json(sidecar / "ownership.json")
        if state.get("cleaned"):
            return False
        state["recovery"] = guard.inspect(os.getpid())
        write_ownership(sidecar, state)
    until = time.monotonic() + 60
    while True:
        if read_json(sidecar / "ownership.json").get("cleaned"):
            return False
        if not identity_alive(owner):
            raise RuntimeError("Owner exited before recovery could arm")
        records = [read_maintenance(maintenance_path(gpu)) for gpu in CAPACITY]
        if all(record is not None and record.get("owner") == owner for record in records):
            break
        if time.monotonic() >= until:
            raise RuntimeError("Owned leases were not published for independent recovery")
        time.sleep(.05)
    for gpu in CAPACITY:
        register_worker(gpu, owner, list(guard.inspect(os.getpid())["argv"]))
    with file_lock(sidecar / "cleanup.lock"):
        state = read_json(sidecar / "ownership.json")
        if state["owner"] != owner or not identity_alive(owner):
            raise RuntimeError("Owner exited while arming independent recovery")
        state.update(recovery_armed=True, recovery=guard.inspect(os.getpid()))
        write_ownership(sidecar, state)
    return True


def stop_marker(marker, timeout=120):
    info = owned_marker(marker)
    if info is None:
        return
    guard.signal_exact(info, signal.SIGTERM)
    until = time.monotonic() + timeout
    while identity_alive(info["identity"]):
        if time.monotonic() >= until:
            raise RuntimeError("Owned child has not exited; leases/controller remain held")
        time.sleep(.2)


def unregistered_bootstraps(sidecar, plan, owner, proc=Path("/proc")):
    values = []
    original = read_json(Path(plan["output"]) / "queue_plan.json")
    script = str(ROOT / "scripts/run_parallel_alfworld_accuracy.py")
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            argv = [os.fsdecode(v) for v in (directory / "cmdline").read_bytes().split(b"\0") if v]
            if (not argv or not Path(argv[0]).name.startswith("python")
                    or "--bootstrap" not in argv or script not in argv):
                continue
            if first_python_entry(argv) != script:
                continue
            for job in plan["jobs"]:
                for gpu in CAPACITY:
                    expected = [
                        original["bootstrap_prefix"][0], "-B", "-u", script,
                        "--bootstrap", "--sidecar", str(sidecar), "--job", job["name"],
                        "--gpu", str(gpu), "--owner-pid", str(owner["pid"]),
                        "--owner-start-ticks", str(owner["start_ticks"]),
                    ]
                    if argv == expected:
                        info = guard.inspect(int(directory.name))
                        if info["cwd"] != str(ROOT) or info["argv"] != expected:
                            raise RuntimeError("Pending owned CPU bootstrap identity changed")
                        values.append({"info": info, "bootstrap": expected, "command": expected})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except RuntimeError:
            if process_identity(int(directory.name)) is None:
                continue
            raise
    return values


def cleanup(sidecar, owner):
    with file_lock(sidecar / "cleanup.lock"):
        state = read_json(sidecar / "ownership.json")
        if state["owner"] != owner:
            raise RuntimeError("Refusing cleanup for another sidecar owner")
        if state.get("cleaned"):
            return
        plan = read_json(sidecar / "sidecar_plan.json")
        values = markers(sidecar)
        # A Popen/owner-death gap may precede parent publication. Such a child
        # is still CPU-only, but must also exit before the serial queue resumes.
        pending = unregistered_bootstraps(sidecar, plan, owner)
        known = {m["info"]["identity"]["pid"] for m in values}
        values.extend(m for m in pending if m["info"]["identity"]["pid"] not in known)
        for marker in values:
            stop_marker(marker)
        if any(identity_alive(m["info"]["identity"]) for m in values):
            raise RuntimeError("Owned workers remain alive; cannot release/resume")
        for gpu in CAPACITY:
            with maintenance_launch_lock(gpu):
                record = read_maintenance(maintenance_path(gpu))
                if record is None:
                    continue
                if record.get("owner") != owner:
                    if gpu in state.get("claimed_gpus", []):
                        raise RuntimeError("GPU0/1 lease ownership changed; refusing mutation")
                    continue
                child = record.get("child")
                recovery = state.get("recovery")
                if identity_alive(child) and (recovery is None or child != recovery["identity"]):
                    raise RuntimeError("Unexpected maintenance child; refusing release")
                record.update(child=None, parallel_children=[])
                atomic_json(maintenance_path(gpu), record)
            if not release_maintenance(gpu, owner):
                raise RuntimeError(f"Could not release owned GPU{gpu} maintenance")
        resume_controller(sidecar, state)
        state["cleaned"] = True
        write_ownership(sidecar, state)


def recover(sidecar, owner):
    # Arming failures are fatal and visible; even a partially armed recovery
    # must clean a dead owner's claims rather than abandon the lease.
    try:
        if arm_recovery(sidecar, owner) is False:
            return
        while identity_alive(owner):
            if read_json(sidecar / "ownership.json").get("cleaned"):
                return
            time.sleep(1)
        cleanup(sidecar, owner)
    except BaseException:
        import traceback
        atomic_json(sidecar / "recovery_error.json", {"traceback": traceback.format_exc()})
        if not identity_alive(owner):
            cleanup(sidecar, owner)
        raise
