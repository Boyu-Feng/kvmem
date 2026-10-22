"""Run one additional Llama accuracy worker alongside known Qwen accuracy jobs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_component_ablation_queue as queue
from scripts.run_component_ablation_rebuttal import patch_attribute
from scripts.run_qa32_rebuttal import file_hash


def existing_accuracy(pid, proc=Path("/proc")):
    directory = proc / str(pid)
    try:
        if directory.stat().st_uid != os.getuid():
            return False
        args = [os.fsdecode(part) for part in
                directory.joinpath("cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
    except FileNotFoundError:
        return False
    return (
        len(args) > 2 and "python" in Path(args[0]).name
        and str(ROOT / "scripts/run_component_ablation_rebuttal.py") in args
        and "--execute" in args and "--model-ready" in args
        and args[args.index("--model-ready") + 1].endswith("/assets/7b_ready.json")
    )


def assigned_blocker(gpu, uuid, proc=Path("/proc")):
    for directory in proc.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            command = directory.joinpath("cmdline").read_bytes().rstrip(b"\0").split(b"\0")
            if not command or "python" not in Path(os.fsdecode(command[0])).name:
                continue
            environment = directory.joinpath("environ").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        except PermissionError:
            return True
        visible = [part.split(b"=", 1)[1].decode() for part in environment
                   if part.startswith(b"CUDA_VISIBLE_DEVICES=")]
        if any(str(gpu) in value.split(",") or uuid in value.split(",") for value in visible):
            pid = int(directory.name)
            if not queue.own_load(pid, gpu, proc) and not existing_accuracy(pid, proc):
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-ready", type=Path, required=True)
    parser.add_argument("--data-ready", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    args = parser.parse_args()
    args.output, args.tmp_root, args.data_ready = (
        args.output.resolve(), args.tmp_root.resolve(), args.data_ready.resolve())
    args.model_readys = {"llama31_8b": args.model_ready.resolve()}
    args.gpus, args.max_workers, args.min_free_gib, args.poll_seconds = [0, 1, 2], 1, 32, 10
    original_snapshot, original_gate = queue.gpu_snapshot, queue.source_gate

    def snapshot():
        devices = original_snapshot()
        for device in devices.values():
            device["pids"] = [pid for pid in device["pids"] if not existing_accuracy(pid)]
        return devices

    def gate():
        return {**original_gate(), "scripts/run_shared_llama_ablation.py": file_hash(Path(__file__))}

    # This separate resource namespace admits intentional accuracy co-residency,
    # but cannot bypass the shared GPU7 maintenance gate or existing output locks.
    if args.output.parent == Path("/mnt/checkpoints/bruli/fengboyu/kvmem_rebuttal_20260929/results"):
        raise RuntimeError("Use a dedicated shared-accuracy namespace, not the exclusive GPU lease directory")
    with patch_attribute(queue, "gpu_snapshot", snapshot), \
            patch_attribute(queue, "assigned_experiment", assigned_blocker), \
            patch_attribute(queue, "source_gate", gate):
        queue.execute(args)


if __name__ == "__main__":
    main()
