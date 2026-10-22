"""Suspend the exact owned idle guard; release load VRAM and restore its profile."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOAD = Path("/mnt/3fs/models/bruli/gpu-utilization-20260925/gpu_load.py")
NODE = LOAD.parent / "nodes/dev-NDH200-023"
SELF = Path(__file__).resolve()
ENV_ALLOWLIST = {
    "PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "HF_HOME",
    "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "TORCH_HOME",
    "CUDA_CACHE_PATH", "PYTHONPYCACHEPREFIX",
}
RESOURCE_LOCK_FD = None
DEFAULT_RESULTS = Path("/mnt/checkpoints/bruli/fengboyu/kvmem_rebuttal_20260929/results")
DATASETS = ("hotpotqa", "2wiki", "musique")
SCOPES = {
    "qwen": ("component_ablation_seeds233_3407_20260930_v1", "qwen25_7b", 500),
    "llama": ("shared_accuracy/llama_component_ablation_seeds233_3407_v1", "llama31_8b", 500),
    "sensitivity": ("sensitivity_shared/qwen7b_seed233_v1", "qwen25_7b", 100),
    **{f"qa32_{d}": (f"qa32_{d}_v1", None, 500) for d in DATASETS},
}
CONTROLLERS = {
    "qwen": ("run_component_ablation_queue.py", "run_parallel_qwen_ablation.py"),
    "llama": ("run_shared_llama_ablation.py", "run_parallel_llama_ablation.py"),
    "sensitivity": ("run_stepkv_sensitivity_queue.py", "run_parallel_sensitivity_queue.py"),
    **{f"qa32_{d}": ("run_qa32_rebuttal_queue.py", "run_parallel_qa32_queue.py") for d in DATASETS},
}
WORKERS = {
    "qwen": "run_component_ablation_rebuttal.py",
    "llama": "run_component_ablation_rebuttal.py",
    "sensitivity": "run_stepkv_sensitivity.py",
    **{f"qa32_{d}": "run_qa32_rebuttal.py" for d in DATASETS},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(record):
    return bool(record) and identity(record["pid"]) == record


def inspect(pid):
    directory = Path("/proc") / str(pid)
    if directory.stat().st_uid != os.getuid():
        raise PermissionError(f"Not an owned process: {pid}")
    token = identity(pid)
    if token is None:
        raise RuntimeError(f"Process exited: {pid}")
    result = {"identity": token, "uid": os.getuid(),
              "argv": [os.fsdecode(v) for v in (directory / "cmdline").read_bytes().split(b"\0") if v],
              "cwd": str((directory / "cwd").resolve())}
    if identity(pid) != token:
        raise RuntimeError("Process changed during inspection")
    return result


def gpu_selector(pid):
    # Only the GPU selector is retained; no environment is persisted or logged.
    for value in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0"):
        if value.startswith(b"CUDA_VISIBLE_DEVICES="):
            return os.fsdecode(value.split(b"=", 1)[1])
    return None


def safe_environment(pid):
    if (Path("/proc") / str(pid)).stat().st_uid != os.getuid():
        raise PermissionError("Foreign environment")
    result = {}
    for item in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0"):
        key, separator, value = item.partition(b"=")
        name = os.fsdecode(key)
        if separator and name in ENV_ALLOWLIST:
            text = os.fsdecode(value)
            if "://" in text or "\n" in text or "\r" in text:
                raise RuntimeError(f"Not a safe runtime path/selector: {name}")
            result[name] = text
    return result


def stopped(pid):
    fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return fields[0] in ("T", "t")


def signal_exact(info, signum):
    if inspect(info["identity"]["pid"]) != info:
        raise RuntimeError("Owned command/cwd/UID/start ticks changed before signal")
    fd = os.pidfd_open(info["identity"]["pid"])
    try:
        if inspect(info["identity"]["pid"]) != info:
            raise RuntimeError("Identity changed after opening pidfd")
        signal.pidfd_send_signal(fd, signum)
    finally:
        os.close(fd)


def gpu_samples():
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.free",
         "--format=csv,noheader,nounits"], text=True, timeout=15)
    samples = {}
    for line in raw.splitlines():
        index, uuid, name, util, free = [v.strip() for v in line.split(",")]
        samples[index] = {"uuid": uuid, "name": name, "utilization": int(util),
                          "free_mib": int(free), "pids": []}
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, timeout=15)
    for line in raw.splitlines():
        uuid, pid = [v.strip() for v in line.split(",")]
        for sample in samples.values():
            if sample["uuid"] == uuid:
                sample["pids"].append(int(pid))
    return samples


def validate_hardware(record, samples):
    if set(samples) != {str(g) for g in range(8)}:
        raise RuntimeError("Expected all eight original GPUs")
    for gpu, sample in samples.items():
        if "H200" not in sample["name"] or sample["uuid"] != record["gpu_uuids"][gpu]:
            raise RuntimeError(f"GPU{gpu} UUID/model changed")


def assigned_gpus(samples):
    assigned = set()
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid() or identity(directory.name) is None:
                continue
            args = [os.fsdecode(v) for v in (directory / "cmdline").read_bytes().split(b"\0") if v]
            if not args or "python" not in Path(args[0]).name or str(LOAD) in args:
                continue
            visible = set((gpu_selector(int(directory.name)) or "").split(","))
            assigned.update(g for g, sample in samples.items() if g in visible or sample["uuid"] in visible)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return assigned


def read(path):
    path = Path(path)
    if path.stat().st_uid != os.getuid():
        raise PermissionError(f"Not an owned record: {path}")
    return json.loads(path.read_text())


def write(path, record):
    record["updated"] = time.time()
    path = Path(path)
    stage = path.with_suffix(".writing")
    fd = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(record, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stage, path)


@contextmanager
def locked(path, nonblocking=False):
    fd = os.open(Path(path).with_suffix(".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as stream:
        if os.fstat(stream.fileno()).st_uid != os.getuid():
            raise PermissionError("Foreign lock")
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield stream


def validate_load(info, gpu, expected=None):
    if info["uid"] != os.getuid():
        raise PermissionError("Foreign load UID")
    if (gpu not in range(8) or info["argv"] !=
            ["/usr/bin/python3", "-B", str(LOAD), "worker", "--gpu", str(gpu)]
            or info["cwd"] != str(NODE)):
        raise RuntimeError(f"GPU{gpu}: original load configuration differs")
    if expected is not None and info != expected:
        raise RuntimeError("Original load identity changed")


def expected_names(scope):
    if scope.startswith("qa32_"):
        dataset = scope.removeprefix("qa32_")
        conditions = [("fullkv", 100)] + [(m, r) for m in
                      ("stepkv", "h2o", "tova", "h2o_step", "tova_step") for r in (20, 50)]
        return {f"{dataset}/main/{m}/r{r}" for m, r in conditions}
    model = SCOPES[scope][1]
    if scope == "sensitivity":
        levels = {"beta": (.4, 1.6), "reward_share": (.7, .95),
                  "repeat_penalty": (.15, .6), "reuse_scale": (.75, 3.),
                  "citation_increment_cap": (.5, 2.), "score_cap": (4., 16.),
                  "reward_cap": (1., 4.), "floor_ratio": (0., .05, .2)}
        cases = {"default", "token_only", "step_only"} | {
            f"{factor}_{value:g}" for factor, values in levels.items() for value in values}
        return {f"{model}/seed233/{d}/r{r}/main/{c}"
                for d in DATASETS for r in (20, 50) for c in cases}
    variants = ("full", "no_repeat", "no_novelty", "no_success", "no_cite", "no_token", "no_step")
    return {f"{model}/seed{s}/{d}/main/{v}" for s in (233, 3407) for d in DATASETS for v in variants}


def scope_complete(scope, summary):
    rows = summary.get("results", [])
    if SCOPES[scope][1]:
        rows = [row for row in rows if row.get("model") == SCOPES[scope][1]]
    names = [row.get("name") for row in rows]
    expected = expected_names(scope)
    if len(names) != len(set(names)) or not set(names).issubset(expected):
        raise RuntimeError(f"{scope}: unexpected or duplicate formal condition")
    if any(row.get("completed_samples") != SCOPES[scope][2] for row in rows):
        raise RuntimeError(f"{scope}: invalid formal sample count")
    return set(names) == expected, len(names)


def entry(info):
    args = list(info["argv"][1:])
    while args and args[0] in ("-B", "-u"):
        args.pop(0)
    return (Path(info["cwd"]) / args[0]).resolve() if args and not args[0].startswith("-") else None


def argument_path(info, option):
    args = info["argv"]
    if option not in args:
        return None
    index = args.index(option) + 1
    if index >= len(args):
        raise RuntimeError(f"Malformed {option} for {info['identity']}")
    return (Path(info["cwd"]) / args[index]).resolve()


def processes(results, proc=Path("/proc")):
    live = {scope: {"controllers": [], "workers": []} for scope in SCOPES}
    scripts = set(WORKERS.values()) | {v for names in CONTROLLERS.values() for v in names}
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            raw = (directory / "cmdline").read_bytes()
            if not any(name.encode() in raw for name in scripts) or identity(directory.name) is None:
                continue
            info = inspect(int(directory.name))
            script = entry(info)
            output = argument_path(info, "--output")
            global_results = argument_path(info, "--results")
            for scope, (relative, _, _) in SCOPES.items():
                root = (results / relative).resolve()
                if script == ROOT / "scripts" / WORKERS[scope] and output and output != root and output.is_relative_to(root):
                    live[scope]["workers"].append(info["identity"])
                if script in {ROOT / "scripts" / name for name in CONTROLLERS[scope]}:
                    if output == root or (script.name == "run_parallel_qa32_queue.py" and
                                          ((global_results or DEFAULT_RESULTS) == results or output == results)):
                        live[scope]["controllers"].append(info["identity"])
        except (FileNotFoundError, ProcessLookupError):
            continue
    return live


def work_status(results):
    live = processes(results)
    status = {}
    for scope, (relative, _, _) in SCOPES.items():
        error = None
        try:
            complete, count = scope_complete(scope, read(results / relative / "SUMMARY.json"))
        except FileNotFoundError:
            complete, count = False, 0
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            complete, count, error = False, 0, str(exc)
        workers, controllers = live[scope]["workers"], live[scope]["controllers"]
        # Formal completion plus worker exit is sufficient. In particular, the
        # old Qwen controller may wait forever on a transferred Llama placeholder.
        active = bool(workers or (controllers and not complete))
        status[scope] = {"complete": complete, "completed": count, "expected": len(expected_names(scope)),
                         "active": active, "error": error, **live[scope]}
    return status


def restore_decision(work):
    if any(item["active"] for item in work.values()):
        return None
    return "success" if all(item["complete"] and not item["error"] for item in work.values()) else "experiments_failed"


def validate_idle(record, require_fresh=False):
    actual = inspect(record["idle_guard"]["identity"]["pid"])
    if actual != record["idle_guard"]:
        raise RuntimeError("Preserved idle guard identity/configuration changed")
    if digest(ROOT / "scripts/run_idle_gpu_load_guard.py") != record["idle_guard_sha256"]:
        raise RuntimeError("Idle guard source changed")
    if not require_fresh:
        return actual
    status = read(NODE / "idle-load-guard.status.json")
    if status["owner"] != actual["identity"] or status["status"] not in ("running", "adopted", "gpu7_reserved"):
        raise RuntimeError("Idle guard unhealthy")
    if time.time() - status.get("updated", 0) > 30:
        raise RuntimeError("Idle guard heartbeat stale")
    return status


def find_loads(record):
    found = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid() or identity(directory.name) is None:
                continue
            raw = (directory / "cmdline").read_bytes()
            if str(LOAD).encode() not in raw:
                continue
            info = inspect(int(directory.name))
            args = info["argv"]
            if "worker" not in args:
                continue
            if "--gpu" not in args or args.index("--gpu") + 1 >= len(args):
                raise RuntimeError("Malformed owned load worker")
            gpu = int(args[args.index("--gpu") + 1])
            validate_load(info, gpu)
            if str(gpu) in found:
                raise RuntimeError(f"Duplicate owned load workers on GPU{gpu}")
            if safe_environment(info["identity"]["pid"]) != record["loads"][str(gpu)]["environment"]:
                raise RuntimeError(f"GPU{gpu}: original allowlisted runtime environment changed")
            found[str(gpu)] = info
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def verify_restored(record):
    if digest(LOAD) != record["load_sha256"] or digest(SELF) != record["guard_sha256"]:
        raise RuntimeError("Original load/deadline guard source changed")
    if alive(record["idle_guard"]["identity"]):
        if stopped(record["idle_guard"]["identity"]["pid"]):
            raise RuntimeError("Idle guard is still suspended, not restored")
        validate_idle(record, require_fresh=True)
    samples = gpu_samples()
    validate_hardware(record, samples)
    found = find_loads(record)
    if set(found) != {str(g) for g in range(8)}:
        raise RuntimeError(f"Missing original-profile loads: {sorted(set(samples) - set(found))}")
    for gpu, actual in found.items():
        if stopped(actual["identity"]["pid"]):
            raise RuntimeError(f"GPU{gpu}: load is stopped, not restored")
        sample = samples[gpu]
        if actual["identity"]["pid"] not in sample.get("pids", []) or sample.get("utilization", 0) <= 0:
            raise RuntimeError(f"GPU{gpu}: load has not returned to GPU compute")
    return found


def restore_idle(record):
    if digest(LOAD) != record["load_sha256"] or digest(SELF) != record["guard_sha256"]:
        raise RuntimeError("Restoration refused: original/deadline source changed")
    if LOAD.stat().st_uid != os.getuid() or NODE.stat().st_uid != os.getuid():
        raise PermissionError("Restoration refused: source/node ownership changed")
    samples = gpu_samples()
    validate_hardware(record, samples)
    if alive(record["idle_guard"]["identity"]):
        actual = validate_idle(record)
        if stopped(actual["identity"]["pid"]):
            signal_exact(actual, signal.SIGCONT)
        record["restoration_mode"] = "same_idle_guard_resumed"
        return
    # The preserved private environment is lost. Do not reconstruct credentials
    # or restart another guard; use only the recorded nonsecret runtime paths.
    if any(item["active"] for item in work_status(Path(record["results"])).values()):
        raise RuntimeError("Fallback refused: experiment/controller still active")
    found = find_loads(record)
    assigned = assigned_gpus(samples)
    record["restoration_mode"] = "idle_guard_dead_safe_environment_fallback"
    record["idle_guard_dead_warning"] = True
    for gpu in map(str, range(8)):
        if gpu in found:
            continue
        sample = samples[gpu]
        if gpu in assigned or sample["pids"] or sample["utilization"] != 0:
            raise RuntimeError(f"GPU{gpu}: fallback waits for unoccupied idle hardware")
        load = record["loads"][gpu]
        environment = load["environment"]
        if not set(environment).issubset(ENV_ALLOWLIST):
            raise RuntimeError("Unexpected fallback environment key")
        with (NODE / f"gpu-{gpu}.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(load["worker"]["argv"], cwd=NODE, env=environment,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        token = identity(child.pid)
        if token is None:
            raise RuntimeError(f"GPU{gpu} fallback worker exited during launch")
        record.setdefault("fallback_workers", {})[gpu] = token


def prepare(args):
    if not args.release_authorized:
        raise RuntimeError("--release-authorized is mandatory")
    deadline = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
    if deadline.tzinfo is None or deadline.timestamp() <= time.time():
        raise RuntimeError("Supply a future timezone-qualified deadline")
    if LOAD.stat().st_uid != os.getuid() or NODE.stat().st_uid != os.getuid():
        raise PermissionError("Foreign load source/node")
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = args.output.resolve() / "status.json"
    with locked(path):
        if path.exists():
            raise RuntimeError("Record already exists; use a fresh output")
        idle = inspect(args.idle_guard_pid)
        if entry(idle) != ROOT / "scripts/run_idle_gpu_load_guard.py":
            raise RuntimeError("Not the owned idle load guard")
        if argument_path(idle, "--load-script") != LOAD or argument_path(idle, "--node") != NODE:
            raise RuntimeError("Idle guard load configuration differs")
        loads = {}
        for gpu, pid in enumerate(args.load_pids):
            info = inspect(pid)
            validate_load(info, gpu)
            loads[str(gpu)] = {"worker": info, "environment": safe_environment(pid)}
        samples = gpu_samples()
        if set(samples) != {str(g) for g in range(8)} or any("H200" not in s["name"] for s in samples.values()):
            raise RuntimeError("Release requires the eight H200 GPUs")
        record = {"version": 2, "status": "prepared", "release_authorized": True,
                  "record_path": str(path),
                  "deadline": deadline.timestamp(), "results": str(args.results.resolve()),
                  "load_sha256": digest(LOAD), "guard_sha256": digest(SELF),
                  "idle_guard_sha256": digest(ROOT / "scripts/run_idle_gpu_load_guard.py"),
                  "idle_guard": idle, "loads": loads, "prepared": time.time(),
                  "gpu_uuids": {g: s["uuid"] for g, s in samples.items()},
                  "poll_seconds": args.poll_seconds, "quiet_seconds": args.quiet_seconds,
                  "restore_timeout": args.restore_timeout,
                  "mode": "SIGSTOP owned idle guard; parent releases loads; SIGCONT same guard",
                  "environment_policy": "only allowlisted nonsecret paths/selectors for dead-guard fallback"}
        validate_idle(record, require_fresh=True)
        if stopped(idle["identity"]["pid"]):
            raise RuntimeError("Idle guard already suspended; refuse overlapping ownership")
        write(path, record)
    print(path, flush=True)


def spawn(mode, path):
    owner = read(path)["owner"]
    inherited = [] if RESOURCE_LOCK_FD is None else ["--resource-lock-fd", str(RESOURCE_LOCK_FD)]
    with (path.parent / f"{mode}.log").open("a") as log:
        child = subprocess.Popen([sys.executable, "-B", "-u", str(SELF), f"--{mode}",
                                  "--record", str(path), "--launch-owner-pid", str(owner["pid"]),
                                  "--launch-owner-start-ticks", str(owner["start_ticks"]), *inherited], cwd=ROOT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES="",
                                          PYTHONDONTWRITEBYTECODE="1"),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                 pass_fds=() if RESOURCE_LOCK_FD is None else (RESOURCE_LOCK_FD,))
    token = identity(child.pid)
    if token is None:
        raise RuntimeError(f"{mode} exited during launch")
    return token


def monitor_valid(record, role):
    if not alive(record.get(role)):
        return False
    info = inspect(record[role]["pid"])
    args = list(info["argv"][1:])
    if len(args) >= 2 and args[-2] == "--resource-lock-fd" and args[-1].isdigit():
        args = args[:-2]
    expected = ["-B", "-u", str(SELF), f"--{role}", "--record", record["record_path"],
                "--launch-owner-pid", str(record["owner"]["pid"]),
                "--launch-owner-start-ticks", str(record["owner"]["start_ticks"])]
    return (info["argv"][0] == sys.executable and args == expected
            and gpu_selector(info["identity"]["pid"]) == "")


def tick(path, role):
    with locked(path):
        record = read(path)
        if record.get(role) != identity(os.getpid()):
            raise RuntimeError(f"Unregistered {role} identity; refusing record update")
        if record["status"] in ("restored", "restored_after_experiment_failure"):
            return True
        record[f"{role}_heartbeat"] = time.time()
        if role == "failsafe" and (alive(record.get("owner")) or alive(record.get("rescue"))):
            write(path, record)
            return False
        if role == "rescue" and alive(record.get("owner")):
            write(path, record)
            return False
        work = work_status(Path(record["results"]))
        record["experiments"] = work
        record["monitor_role"] = role
        record["deadline_expired_warning"] = time.time() >= record["deadline"]
        outcome = restore_decision(work)
        if outcome is None:
            record.pop("quiet_since", None)
        else:
            record.setdefault("quiet_since", time.time())
        quiet = outcome is not None and time.time() - record["quiet_since"] >= record["quiet_seconds"]
        errors = []
        try:
            source_safe = digest(SELF) == record["guard_sha256"]
        except OSError:
            source_safe = False
        try:
            if digest(LOAD) != record["load_sha256"] or not source_safe:
                raise RuntimeError("Load/deadline guard source changed")
            if alive(record["idle_guard"]["identity"]):
                validate_idle(record)
            else:
                record["idle_guard_dead_warning"] = True
                errors.append("Idle guard died: safe allowlisted fallback required after experiments exit")
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            errors.append(str(exc))
        original_alive = {}
        for gpu, load in record["loads"].items():
            token = load["worker"]["identity"]
            original_alive[gpu] = alive(token)
            if original_alive[gpu]:
                try:
                    validate_load(inspect(token["pid"]), int(gpu), load["worker"])
                except (OSError, RuntimeError, PermissionError) as exc:
                    errors.append(str(exc))
        record["original_loads_alive"] = original_alive
        released = [gpu for gpu, exists in original_alive.items() if not exists]
        record["released_gpus"] = released
        if not record.get("restore_started"):
            if source_safe and role != "failsafe":
                for helper in ("rescue", "failsafe") if role == "owner" else ("failsafe",):
                    if not alive(record.get(helper)):
                        record[helper] = spawn(helper, path)
                        record.pop(f"{helper}_heartbeat", None)
            ready = (monitor_valid(record, "rescue") and monitor_valid(record, "failsafe")
                     and time.time() - record.get("rescue_heartbeat", 0) < 30
                     and time.time() - record.get("failsafe_heartbeat", 0) < 30)
            if (ready or record.get("suspend_planned")) and not errors and alive(record["idle_guard"]["identity"]):
                try:
                    idle = validate_idle(record)
                    if not stopped(idle["identity"]["pid"]):
                        # Commit the intent before the exact-identity signal.
                        record["suspend_planned"] = time.time()
                        write(path, record)
                        signal_exact(idle, signal.SIGSTOP)
                    record["idle_guard_suspended"] = stopped(idle["identity"]["pid"])
                except (OSError, RuntimeError, ValueError) as exc:
                    errors.append(str(exc))
            record["release_ready"] = ready and not errors and record.get("idle_guard_suspended", False)
            record["status"] = ("awaiting_release" if record["release_ready"] else "starting") if not released else (
                "released" if len(released) == 8 else "partial_release")
            if record["release_ready"]:
                record.setdefault("release_planned", time.time())
        if quiet:
            record["experiment_outcome"] = outcome
            record.setdefault("restore_started", time.time())
            record["status"] = "restoring"
            record["release_ready"] = False
        if record.get("restore_started") and outcome is None:
            errors.append("New experiment appeared during restoration; re-suspending idle guard")
            try:
                if alive(record["idle_guard"]["identity"]):
                    idle = validate_idle(record)
                    if not stopped(idle["identity"]["pid"]):
                        signal_exact(idle, signal.SIGSTOP)
                record.pop("restore_started", None)
            except (OSError, RuntimeError) as exc:
                errors.append(str(exc))
        if record.get("restore_started") and outcome is not None:
            try:
                # Retry restoration even if a prior SIGCONT was interrupted.
                record["resume_planned"] = time.time()
                write(path, record)
                restore_idle(record)
                record["restored_loads"] = verify_restored(record)
                record["status"] = "restored" if outcome == "success" else "restored_after_experiment_failure"
                record["restored_at"] = time.time()
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                errors.append(str(exc))
                if time.time() - record["restore_started"] > record["restore_timeout"]:
                    record["status"] = "restore_failed_retrying"
        record["errors"] = errors
        if errors:
            print("; ".join(errors), file=sys.stderr, flush=True)
        write(path, record)
        return record["status"] in ("restored", "restored_after_experiment_failure")


def monitor(path, role, launch_owner=None):
    # No monitor reserves GPU selectors: frozen admission must see no holder.
    if launch_owner is not None:
        with locked(path):
            record = read(path)
            if record.get("owner") != launch_owner or digest(SELF) != record["guard_sha256"]:
                raise RuntimeError("Detached launch owner/source mismatch")
            if not alive(record.get(role)):
                # Cover SIGKILL between Popen and the parent's registration write.
                record[role] = identity(os.getpid())
                write(path, record)
            elif record[role] != identity(os.getpid()):
                raise RuntimeError("A different registered monitor is already alive")
    while True:
        try:
            if tick(path, role):
                return
        except Exception as exc:
            print(f"{role}: monitoring failed (idle suspension retained): {exc}", file=sys.stderr, flush=True)
        try:
            interval = read(path)["poll_seconds"]
        except Exception:
            interval = 2
        time.sleep(interval)


def execute(path):
    global RESOURCE_LOCK_FD
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("Deadline controller must not advertise any CUDA selectors")
    # Helpers inherit this same locked open-file description, so ownership
    # survives owner SIGKILL and prevents a second output adopting the resource.
    with locked(NODE / "deadline-release-resource.json", nonblocking=True) as resource, \
            locked(path.with_name("owner.json"), nonblocking=True):
        RESOURCE_LOCK_FD = resource.fileno()
        with locked(path):
            record = read(path)
            if record.get("version") != 2 or record["status"] != "prepared" or not record.get("release_authorized"):
                raise RuntimeError("Expected an explicitly authorized prepared record")
            if digest(SELF) != record["guard_sha256"] or digest(LOAD) != record["load_sha256"]:
                raise RuntimeError("Source changed after preparation")
            validate_idle(record, require_fresh=True)
            validate_hardware(record, gpu_samples())
            if stopped(record["idle_guard"]["identity"]["pid"]):
                raise RuntimeError("Idle guard already suspended by another owner")
            for gpu, load in record["loads"].items():
                validate_load(inspect(load["worker"]["identity"]["pid"]), int(gpu), load["worker"])
            record["owner"] = identity(os.getpid())
            record["status"] = "starting"
            write(path, record)
            record["rescue"] = spawn("rescue", path)
            write(path, record)
            record["failsafe"] = spawn("failsafe", path)
            write(path, record)
        monitor(path, "owner")


def confirm_release(path):
    with locked(path):
        record = read(path)
        if (not record.get("release_authorized") or not record.get("release_ready")
                or not record.get("release_planned")
                or not monitor_valid(record, "rescue") or not monitor_valid(record, "failsafe")
                or time.time() - record.get("failsafe_heartbeat", 0) >= 30
                or time.time() - record.get("rescue_heartbeat", 0) >= 30):
            raise RuntimeError("Release requires independent live CPU rescue/failsafe monitors")
        if record.get("restore_started"):
            raise RuntimeError("Restoration already begun")
        validate_idle(record)
        validate_hardware(record, gpu_samples())
        if not stopped(record["idle_guard"]["identity"]["pid"]):
            raise RuntimeError("Release refused: exact idle guard is not suspended")
        if digest(LOAD) != record["load_sha256"] or digest(SELF) != record["guard_sha256"]:
            raise RuntimeError("Original/deadline source changed")
        for gpu, load in record["loads"].items():
            if alive(load["worker"]["identity"]):
                validate_load(inspect(load["worker"]["identity"]["pid"]), int(gpu), load["worker"])
        record["parent_release_confirmed"] = time.time()
        write(path, record)


def main(argv=None):
    global RESOURCE_LOCK_FD
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "execute", "confirm-release", "rescue", "failsafe"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--load-pids", type=int, nargs=8)
    parser.add_argument("--idle-guard-pid", type=int)
    parser.add_argument("--release-authorized", action="store_true")
    parser.add_argument("--deadline")
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--quiet-seconds", type=float, default=120)
    parser.add_argument("--restore-timeout", type=float, default=300)
    parser.add_argument("--launch-owner-pid", type=int)
    parser.add_argument("--launch-owner-start-ticks", type=int)
    parser.add_argument("--resource-lock-fd", type=int)
    args = parser.parse_args(argv)
    if min(args.poll_seconds, args.quiet_seconds, args.restore_timeout) <= 0:
        parser.error("Timing bounds must be positive")
    if args.poll_seconds > 10 or args.quiet_seconds > 600 or args.restore_timeout > 3600:
        parser.error("Maximum poll/quiet/restore bounds are 10/600/3600 seconds")
    if args.prepare:
        if any(getattr(args, name) is None for name in
               ("output", "results", "load_pids", "idle_guard_pid", "deadline")):
            parser.error("Preparation needs output/results/eight load PIDs/idle guard PID/deadline")
        prepare(args)
    else:
        if args.record is None:
            parser.error("--record is required")
        path = args.record.resolve()
        launch_owner = None
        if args.launch_owner_pid is not None or args.launch_owner_start_ticks is not None:
            if args.launch_owner_pid is None or args.launch_owner_start_ticks is None:
                parser.error("Both launch owner identity fields are required")
            launch_owner = {"pid": args.launch_owner_pid, "start_ticks": args.launch_owner_start_ticks}
        if args.rescue or args.failsafe:
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise RuntimeError("CPU monitors must have an empty CUDA selector")
            if args.resource_lock_fd is not None:
                RESOURCE_LOCK_FD = args.resource_lock_fd
                if os.fstat(RESOURCE_LOCK_FD).st_uid != os.getuid():
                    raise PermissionError("Foreign inherited resource lock")
            monitor(path, "rescue" if args.rescue else "failsafe", launch_owner)
        elif args.confirm_release:
            confirm_release(path)
        else:
            execute(path)


if __name__ == "__main__":
    main()
