"""Own-only GPU1 pilot with a standard new-Llama-admission maintenance gate."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_deadline_gpu_load_guard as inspect_api
from scripts import run_rebuttal_efficiency_queue as maintenance

BASE = Path("/mnt/checkpoints/bruli/fengboyu/kvmem_rebuttal_20260929")
GPU = 1
UUID = "GPU-0941ad2b-1a4b-1c62-22b6-284f5359de15"
LOCK = BASE / "results/reuse_semantic_gpu1.lock"
LLAMA = ROOT / "scripts/run_component_ablation_rebuttal.py"
LLAMA_SCOPE = BASE / "results/shared_accuracy/llama_component_ablation_seeds233_3407_v1"
LLAMA_READY = BASE / "ablation_assets/llama31_8b_shared_ready.json"
SELF = Path(__file__).resolve()
POLICY = (
    "Standard maintenance1 temporarily blocks NEW Llama GPU1 admission only. "
    "Existing Llama work continues. QA GPU3/4, GPU2 and GPU7 are untouched. "
    "Release only after the exact own registered pilot child has exited."
)


class Waiting(RuntimeError):
    """An explicit prerequisite/admission wait, not a corrupt-artifact error."""


def read(path):
    return inspect_api.read(Path(path))


def write(path, value):
    inspect_api.write(Path(path), dict(value))


def exact(info):
    try:
        return inspect_api.inspect(info["identity"]["pid"]) == info
    except (OSError, RuntimeError, KeyError):
        return False


def flag(argv, name):
    if argv.count(name) != 1:
        raise RuntimeError(f"Expected exactly one {name}")
    return argv[argv.index(name) + 1]


def script(argv):
    if not argv or not Path(argv[0]).name.startswith("python"):
        return None
    index = 1
    while index < len(argv) and argv[index] in ("-B", "-u"):
        index += 1
    return argv[index] if index < len(argv) else None


def snapshot():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.free",
        "--format=csv,noheader,nounits"], text=True, timeout=20)
    devices = {}
    for row in csv.reader(output.splitlines()):
        index, uuid, name, free = [part.strip() for part in row]
        devices[int(index)] = dict(uuid=uuid, name=name, free_mib=int(free), pids=[])
    output = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True, timeout=20)
    for row in csv.reader(output.splitlines()):
        uuid, pid = [part.strip() for part in row]
        matches = [d for d in devices.values() if d["uuid"] == uuid]
        if len(matches) != 1:
            raise RuntimeError("Unknown GPU UUID in CUDA accounting")
        matches[0]["pids"].append(int(pid))
    return devices


def known_llama(info, source_sha256):
    """Only exact loaded original Llama-scope commands, never QA or arbitrary peers."""
    if (not exact(info) or info["uid"] != os.getuid() or info["cwd"] != str(ROOT)
            or script(info["argv"]) != str(LLAMA)
            or source_sha256 != inspect_api.digest(LLAMA)):
        return False
    argv = info["argv"]
    if inspect_api.gpu_selector(info["identity"]["pid"]) not in ("1", UUID):
        return False
    try:
        from scripts import run_component_ablation_rebuttal as original
        directory = Path(flag(argv, "--output"))
        relative = directory.relative_to(LLAMA_SCOPE)
        if ".." in directory.parts or flag(argv, "--model-ready") != str(LLAMA_READY):
            return False
        plan = read(LLAMA_SCOPE / "queue_plan.json")
        if (plan["model_readys"].get("llama31_8b") != str(LLAMA_READY)
                or plan["data_ready"] != str(BASE / "qa32_assets/data_ready.json")):
            return False
        matches = [job for job in plan["jobs"] if job["name"] == str(relative)
                   and job["model"] == "llama31_8b"]
        if len(matches) != 1:
            return False
        job = matches[0]
        expected = [argv[0], "-u", str(LLAMA), "--execute",
                    "--model-ready", str(LLAMA_READY), "--data-ready", plan["data_ready"],
                    "--output", str(directory), "--dataset", job["dataset"],
                    "--seed", str(job["seed"]), "--variant", job["variant"],
                    "--purpose", job["purpose"]]
        if argv != expected:
            return False
        manifest = read(directory / "manifest.json")
        status = read(directory / "status.json")
        fp = manifest["fingerprint"]
        ready = read(LLAMA_READY)
        return (status.get("status") in ("running", "complete")
                and status.get("pid", info["identity"]["pid"]) == info["identity"]["pid"]
                and fp["source_hashes"] == original.source_hashes()
                and manifest["fingerprint_sha256"] == original.digest(fp)
                and fp["assets"]["model_ready_sha256"] == inspect_api.digest(LLAMA_READY)
                and ready["repo"] == "meta-llama/Meta-Llama-3.1-8B-Instruct"
                and ready["revision"] == original.MODELS[ready["repo"]]["revision"]
                and all(fp["protocol"][key] == job[key] for key in
                        ("dataset", "seed", "variant", "purpose", "samples", "ratio", "max_steps")))
    except (OSError, KeyError, ValueError, RuntimeError):
        return False


def assignment_blockers(known_pids, proc=Path("/proc")):
    """Inspect cwd only for live candidates actually targeting GPU1."""
    def dead(pid):
        try:
            return inspect_api.identity(pid) is None
        except PermissionError:
            return False

    blockers = []
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        pid = int(directory.name)
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            token = inspect_api.identity(pid)
            if token is None:
                continue
            argv = [os.fsdecode(v) for v in (directory / "cmdline").read_bytes().split(b"\0") if v]
            cpu_service = (
                len(argv) == 1 and (argv[0].startswith("sshd: ") or argv[0] == "(sd-pam)")
                or argv == ["/usr/lib/systemd/systemd", "--user"]
            )
            try:
                visible = (inspect_api.gpu_selector(pid) or "").split(",")
            except PermissionError:
                if dead(pid):
                    continue
                if cpu_service and inspect_api.identity(pid) == token:
                    current = [os.fsdecode(v) for v in
                               (directory / "cmdline").read_bytes().split(b"\0") if v]
                    if current == argv:
                        continue
                blockers.append({"pid": pid, "reason": "unreadable live GPU-capable assignment"})
                continue
            if "1" not in visible and UUID not in visible:
                continue
            info = inspect_api.inspect(pid)
            if info["identity"] != token or info["argv"] != argv:
                blockers.append({"pid": pid, "reason": "changing live GPU1 candidate"})
                continue
            expected = known_pids.get(pid) if isinstance(known_pids, dict) else None
            if expected != info:
                blockers.append({"pid": pid, "reason": "unknown/pending GPU1 assignment"})
        except (FileNotFoundError, ProcessLookupError, PermissionError) as error:
            if not dead(pid):
                blockers.append({"pid": pid, "reason": "unreadable/changing live GPU1 candidate",
                                 "error_type": type(error).__name__})
        except RuntimeError as error:
            if dead(pid):
                continue
            if str(error).startswith(("Process exited:", "Process changed during inspection")):
                blockers.append({"pid": pid, "reason": "changing live GPU1 candidate"})
            else:
                raise
    return blockers


def admit(foreign_records, devices=None, blockers=None):
    devices = snapshot() if devices is None else devices
    device = devices.get(GPU)
    if not device or device["uuid"] != UUID or device["name"] != "NVIDIA H200":
        raise RuntimeError("GPU1 H200 UUID/model mismatch; GPU2/GPU4/GPU7 forbidden")
    if len(foreign_records) > 2:
        raise RuntimeError("At most two captured known Llama contexts")
    approved = {}
    for record in foreign_records:
        info = record["info"]
        pid = info["identity"]["pid"]
        if pid in approved:
            raise RuntimeError("Duplicate Llama context snapshot")
        if exact(info):
            if not known_llama(info, record["source_sha256"]):
                raise Waiting("Captured Llama is not an exact fully loaded frozen peer")
            approved[pid] = info
    foreign = set(device["pids"])
    if len(foreign) > 2 or not foreign.issubset(approved):
        raise Waiting("Unknown/unapproved or more than two foreign CUDA contexts")
    if device["free_mib"] < 64 * 1024:
        raise Waiting("GPU1 free memory below 64GiB")
    problems = (assignment_blockers(approved)
                if blockers is None else blockers)
    if problems:
        raise Waiting(json.dumps(problems, sort_keys=True))
    return {"gpu": GPU, **device, "foreign_contexts": len(foreign),
            "allocator_limit_gib": 44, "scope": "shared accuracy, never latency"}


def claim_and_capture(state):
    """Publish standard maintenance1 under its launch lock before GPU inspection."""
    if not exact(state["recovery"]) or not exact(state["owner"]):
        raise RuntimeError("Live CPU controller and recovery required before claim")
    owner = state["maintenance_owner"]
    if owner != state["recovery"]["identity"]:
        raise RuntimeError("Maintenance must be owned by the CPU recovery identity")
    maintenance.claim_maintenance(GPU, state["output"], owner)
    with maintenance.maintenance_launch_lock(GPU):
        verify_maintenance(state)
        devices = snapshot()
        device = devices.get(GPU)
        if not device or device["uuid"] != UUID or device["name"] != "NVIDIA H200":
            raise RuntimeError("GPU1 H200 UUID/model mismatch")
        records = [{"info": inspect_api.inspect(pid), "source_sha256": inspect_api.digest(LLAMA)}
                   for pid in device["pids"]]
        admission = admit(records, devices)
    return records, admission


def verify_maintenance(state):
    record = maintenance.read_maintenance(maintenance.maintenance_path(GPU))
    if (not record or record.get("owner") != state["maintenance_owner"]
            or record.get("output") != state["output"] or record.get("gpu") != GPU):
        raise RuntimeError("Standard maintenance1 identity/output mismatch")
    child = record.get("child")
    expected = state.get("child")
    if maintenance.identity_alive(child) and (not expected or child != expected["bootstrap"]["identity"]):
        raise RuntimeError("Unknown child in standard maintenance1; never signal it")
    return record


def cleanup(state):
    """Do not release either resource while an own/unknown registered child lives."""
    child = state.get("child")
    if child and inspect_api.alive(child["bootstrap"]["identity"]):
        return False
    record = maintenance.read_maintenance(maintenance.maintenance_path(GPU))
    if record is None:
        return True
    if record.get("owner") != state["maintenance_owner"]:
        # Another owner is not ours to release.
        return True
    verify_maintenance(state)
    return maintenance.release_maintenance(GPU, state["maintenance_owner"])

@contextmanager
def locked(path=LOCK):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if os.fstat(fd).st_uid != os.getuid():
            raise PermissionError("Not an owned pilot lock")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Waiting("Another pilot controller/recovery owns the GPU1 lock") from error
        yield fd
    finally:
        os.close(fd)


def child_info(state):
    child = state.get("child")
    if not child:
        return None
    for info in (child["bootstrap"], child["execution"]):
        if exact(info):
            return info
    return None


def stop_owned(state):
    """Use pidfd identity checks; never act on a foreign/reused PID."""
    sent = False
    for _ in range(2):
        info = child_info(state)
        if not info:
            return
        try:
            inspect_api.signal_exact(info, signal.SIGTERM)
            sent = True
            break
        except (OSError, RuntimeError):
            # Bootstrap may exec between inspection and the pidfd check.
            continue
    if sent:
        deadline = time.monotonic() + 30
        while child_info(state) and time.monotonic() < deadline:
            time.sleep(.2)
        info = child_info(state)
        if info:
            try:
                inspect_api.signal_exact(info, signal.SIGKILL)
            except (OSError, RuntimeError):
                pass


def own_memory(state):
    info = child_info(state)
    if info is None:
        return {"status": "own_child_exited"}
    pid = info["identity"]["pid"]
    try:
        status = (Path("/proc") / str(pid) / "status").read_text()
    except FileNotFoundError:
        return {"status": "own_child_exited"}
    host = {}
    for line in status.splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, value, unit = line.split()
            if unit != "kB":
                raise RuntimeError("Unexpected own host memory unit")
            host[name.rstrip(":") + "_bytes"] = int(value) * 1024
    raw = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True, timeout=20)
    cuda = []
    for row in csv.reader(raw.splitlines()):
        uuid, process, memory = [part.strip() for part in row]
        if int(process) == pid:
            if uuid != UUID:
                raise RuntimeError("Own pilot CUDA context appeared on forbidden GPU")
            cuda.append(int(memory) if memory.isdigit() else None)
    if not exact(info):
        return {"status": "own_child_changed_during_sample"}
    return {"at": time.time(), "pid": pid, **host, "driver_cuda_used_mib": cuda}


def recovery(path):
    checked = False
    while True:
        state = read(path)
        owner_dead = not exact(state["owner"])
        if owner_dead or state.get("status") in ("failed", "finished"):
            if owner_dead or state.get("status") == "failed":
                stop_owned(state)
            try:
                released = cleanup(state)
            except RuntimeError as error:
                released = False
                write(path, {**state, "cleanup_error": str(error)})
            if released:
                status = "owner_dead_released" if owner_dead else state["status"] + "_released"
                write(path, {**state, "status": status})
                return
        elif not checked and state.get("maintenance_claimed"):
            try:
                from scripts import run_reuse_semantic_queue as queue
                queue.assert_plan(read(state["plan"]), state["approval"])
                verify_maintenance(state)
                checked = True
            except Exception as error:
                write(path, {**state, "status": "failed", "error": str(error)})
        time.sleep(.5)


def bootstrap(path):
    """Wait CUDA-empty until exact registration and a live CPU recovery exist."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Bootstrap must start CUDA-empty")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        state = read(path)
        if not exact(state["owner"]):
            raise RuntimeError("Controller died before bootstrap admission")
        child = state.get("child")
        if child and state.get("registered"):
            if (child["bootstrap"]["identity"]["pid"] != os.getpid()
                    or not exact(child["bootstrap"]) or not exact(state["recovery"])):
                raise RuntimeError("Registration/recovery identity mismatch")
            from scripts import run_reuse_semantic_queue as queue
            queue.assert_plan(read(state["plan"]), state["approval"])
            verify_maintenance(state)
            registered = maintenance.register_worker(GPU, state["maintenance_owner"], state["command"])
            if registered != child["bootstrap"]["identity"]:
                raise RuntimeError("Standard registered child identity mismatch")
            admitted = admit(state["foreign_contexts"])
            write(path, {**state, "startup_admission": admitted, "status": "executing"})
            env = queue.worker_environment(Path(state["output"]))
            os.execvpe(state["command"][0], state["command"], env)
        time.sleep(.1)
    raise RuntimeError("Controller did not register child within 60 seconds")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bootstrap", "recovery"), required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Resource processes must be CPU-empty")
    from scripts import run_reuse_semantic_queue as queue
    if args.state.resolve() != (queue.OUTPUT / "resource_state.json").resolve():
        parser.error("Only the fixed v11 resource state namespace is supported")
    if args.mode == "recovery":
        if args.lock_fd is None or os.fstat(args.lock_fd).st_uid != os.getuid():
            parser.error("Recovery requires inherited owned resource lock")
        recovery(args.state)
    else:
        bootstrap(args.state)


if __name__ == "__main__":
    main()
