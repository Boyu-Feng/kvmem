#!/usr/bin/env python3
"""Run the frozen BrowseComp judge with an explicitly recorded host allocation.

Only resource allocation changes: TP=1, BF16, model, engine seed, benchmark
sampling, scorer source, contracts, existing items and calls remain frozen.
The original contract records its nominal .65 allocation; host_execution.json
records the actual allocation override and CPU offload. A separate synthetic
probe must pass on the destination before any benchmark scoring is attempted.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

GIB = 1 << 30
NOMINAL_UTILIZATION = .65


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(record, sha256=digest({k: v for k, v in record.items() if k != "sha256"}))
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def host_identity():
    return {"hostname": os.uname().nodename,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def load_context(plan_path, host_contract_path):
    plan = read(plan_path)
    host = read(host_contract_path)
    require(digest({k: v for k, v in plan.items() if k != "sha256"}) == plan["sha256"], "Plan digest changed")
    require(host.get("schema") == "webbench_migration_host_resources_v1", "Unsupported host contract")
    require(digest({k: v for k, v in host.items() if k != "sha256"}) == host["sha256"], "Host contract digest changed")
    require(all(host.get("target_host", {}).get(key) == value for key, value in host_identity().items()),
            "Host contract belongs to another host or boot")
    wrapper = str(Path(__file__).resolve())
    require(host.get("source_hashes", {}).get(wrapper) == file_hash(wrapper), "Judge wrapper differs from host contract")
    for name, expected in host["source_hashes"].items():
        require(file_hash(name) == expected, "Host adapter source changed: " + name)
    require(host["route_plan_sha256"]["browsecomp_plus"] == plan["sha256"], "Host route plan differs")
    require(host["tensor_parallel_size"] == 1 and host["dtype"] == "bfloat16", "Judge requires frozen TP=1 and BF16")
    require(host["workers_per_gpu"] == 1, "Judge requires one admitted worker per GPU")
    require(host["gpu_budget_gib"] > 0 and host["judge_cpu_offload_gb"] >= 0, "Invalid host allocation")
    output = Path(plan["output"])
    source = output / "source_snapshot/scripts/webbench_scoring_v1.py"
    require(file_hash(source) == plan["source_hashes"]["scripts/webbench_scoring_v1.py"], "Frozen scorer source changed")
    frozen_queue = read(output / "score_queue_contract.json")
    runtime = frozen_queue["judge_runtime_manifest"]
    require(file_hash(runtime["path"]) == runtime["sha256"], "Frozen judge runtime manifest changed")
    require(file_hash(Path(frozen_queue["assets"]) / "model_ready.json") == frozen_queue["model_ready_sha256"], "Judge model inventory changed")
    return plan, host, frozen_queue, source


def load_scorer(source):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("migration_frozen_answer_scorer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_job(plan, name):
    jobs = [job for job in plan["jobs"] if job["name"] == name]
    require(len(jobs) == 1 and jobs[0]["benchmark"] == "browsecomp_plus" and jobs[0]["purpose"] == "main", "Expected one frozen BrowseComp main job")
    return jobs[0]


def gpu_snapshot(gpu):
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    rows = list(csv.reader(output.splitlines()))
    matches = [r for r in rows if int(r[0].strip()) == gpu]
    require(len(matches) == 1, "Selected physical GPU is not present")
    index, uuid, name, total, free = [v.strip() for v in matches[0]]
    return dict(index=int(index), uuid=uuid, name=name, total_gib=float(total) / 1024,
                free_gib=float(free) / 1024)


def runtime_allocation(host, gpu):
    require(gpu in host["devices"], "GPU is outside the host contract")
    inventory = gpu_snapshot(gpu)
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:")) / GIB
    require(available >= host["minimum_available_host_ram_gib"], "Insufficient host MemAvailable for pinned CPU offload")
    require(inventory["free_gib"] >= host["gpu_budget_gib"] + host["headroom_gib"], "Insufficient free GPU memory for budget plus headroom")
    utilization = min(NOMINAL_UTILIZATION, host["gpu_budget_gib"] / inventory["total_gib"])
    return dict(gpu=inventory, available_host_ram_gib=available,
                gpu_memory_utilization=utilization,
                effective_gpu_budget_gib=utilization * inventory["total_gib"],
                cpu_offload_gb=host["judge_cpu_offload_gb"])


def original_engine_kwargs(model_path):
    return dict(model=model_path, tokenizer=model_path, dtype="bfloat16", tensor_parallel_size=1,
                seed=0, gpu_memory_utilization=NOMINAL_UTILIZATION, max_model_len=16384,
                enforce_eager=True, max_num_seqs=1)


def effective_engine_kwargs(original, allocation):
    return dict(original, gpu_memory_utilization=allocation["gpu_memory_utilization"],
                cpu_offload_gb=allocation["cpu_offload_gb"])


def verify_engine(engine, allocation):
    config = engine.llm_engine.vllm_config
    observed = dict(dtype=str(config.model_config.dtype),
                    tensor_parallel_size=config.parallel_config.tensor_parallel_size,
                    pipeline_parallel_size=config.parallel_config.pipeline_parallel_size,
                    engine_seed=config.model_config.seed,
                    max_model_len=config.model_config.max_model_len,
                    gpu_memory_utilization=config.cache_config.gpu_memory_utilization,
                    cpu_offload_gb=config.cache_config.cpu_offload_gb)
    validate_engine_config(observed, allocation)
    return observed


def validate_engine_config(observed, allocation):
    require(observed["dtype"] == "torch.bfloat16" and observed["tensor_parallel_size"] == 1
            and observed["pipeline_parallel_size"] == 1 and observed["engine_seed"] == 0
            and observed["max_model_len"] == 16384, "Actual judge protocol differs from frozen engine settings")
    require(math.isclose(observed["gpu_memory_utilization"], allocation["gpu_memory_utilization"])
            and observed["cpu_offload_gb"] == allocation["cpu_offload_gb"], "Actual engine allocation differs")


def validate_record_allocation(host, record):
    allocation = record["allocation"]
    gpu = allocation["gpu"]
    require(gpu["index"] in host["devices"] and gpu["total_gib"] > 0, "Recorded GPU differs from host contract")
    expected_utilization = min(NOMINAL_UTILIZATION, host["gpu_budget_gib"] / gpu["total_gib"])
    require(allocation["cpu_offload_gb"] == host["judge_cpu_offload_gb"]
            and math.isclose(allocation["gpu_memory_utilization"], expected_utilization)
            and math.isclose(allocation["effective_gpu_budget_gib"], expected_utilization * gpu["total_gib"]),
            "Recorded judge resource allocation differs from host contract")
    if "engine_config" in record:
        validate_engine_config(record["engine_config"], allocation)


def probe_record_path(host_contract_path, gpu):
    return Path(host_contract_path).parent / f"judge_probe_gpu{gpu}.json"


def base_record(plan, host, source, mode):
    return dict(schema="webbench_migration_judge_host_execution_v1", mode=mode, state="preflight",
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                host=host_identity(), pid=os.getpid(), plan_sha256=plan["sha256"],
                host_contract_sha256=host["sha256"], wrapper_sha256=file_hash(__file__),
                frozen_scorer_sha256=file_hash(source),
                nominal_gpu_memory_utilization=NOMINAL_UTILIZATION,
                resource_override_disclosure="Frozen score contract retains nominal gpu_memory_utilization=.65; this host record binds the actual lower allocation and CPU weight offload. No precision/model/sampling/TP change.",
                destination_runtime_tested=False)


def validate_probe(plan, host, source, host_contract_path, gpu):
    path = probe_record_path(host_contract_path, gpu)
    record = read(path)
    require(record.get("sha256") == digest({k: v for k, v in record.items() if k != "sha256"}), "Judge probe record digest changed")
    expected = dict(mode="synthetic_probe", state="completed", host=host_identity(),
                    plan_sha256=plan["sha256"], host_contract_sha256=host["sha256"],
                    frozen_scorer_sha256=file_hash(source), wrapper_sha256=file_hash(__file__),
                    destination_runtime_tested=True)
    require(all(record.get(k) == v for k, v in expected.items()), "Successful judge probe for this exact host/contract is required")
    require(record["allocation"]["gpu"]["uuid"] == gpu_snapshot(gpu)["uuid"], "Probed physical GPU identity changed")
    validate_record_allocation(host, record)
    require("engine_config" in record, "Synthetic judge probe lacks verified engine configuration")
    return path, record


def validate_probe_record(plan_path, host_contract_path, gpu):
    plan, host, queue, source = load_context(plan_path, host_contract_path)
    return validate_probe(plan, host, source, host_contract_path, gpu)[1]


def validate_host_record(plan_path, job_name, host_contract_path):
    """Additional stage-gate check; frozen validate_scores remains mandatory."""
    plan = read(plan_path)
    path = Path(plan["output"]) / "jobs" / job_name / "scoring/host_execution.json"
    if not path.exists():
        return None  # Original completed results have no migration host record.
    plan, host, queue, source = load_context(plan_path, host_contract_path)
    job = selected_job(plan, job_name)
    record = read(path)
    require(record.get("sha256") == digest({k: v for k, v in record.items() if k != "sha256"}), "Host execution record digest changed")
    expected = dict(mode="benchmark_scoring", state="completed", plan_sha256=plan["sha256"],
                    host_contract_sha256=host["sha256"], job=job,
                    wrapper_sha256=file_hash(__file__), frozen_scorer_sha256=file_hash(source),
                    destination_runtime_tested=True)
    require(all(record.get(k) == v for k, v in expected.items()), "Incomplete or mismatched judge host execution record")
    scoring = path.parent
    require(record["predictions_sha256"] == file_hash(scoring.parent / "predictions.json"), "Judge prediction input changed")
    require(record["scoring_contract_sha256"] == digest(read(scoring / "contract.json"))
            and record["scores_sha256"] == file_hash(scoring / "scores.json"), "Judge terminal scoring binding changed")
    validate_record_allocation(host, record)
    require(file_hash(record["probe_path"]) == record["probe_sha256"], "Bound judge probe changed")
    return record


def prepare_cuda(gpu):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    require(visible in (None, str(gpu)), "CUDA_VISIBLE_DEVICES must select exactly the admitted physical GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def run_probe(args):
    plan, host, queue, source = load_context(args.plan, args.host_contract)
    path = probe_record_path(args.host_contract, args.gpu)
    if getattr(args, "output", None) is not None:
        require(args.output.resolve() == path.resolve(), "Probe output must use the host contract's judge_probe_gpu<N>.json path")
    require(not path.exists(), "Judge probe already recorded; preserve it and inspect before explicitly rerunning")
    record = base_record(plan, host, source, "synthetic_probe")
    write(path, record)
    try:
        prepare_cuda(args.gpu)
        allocation = runtime_allocation(host, args.gpu)
        scorer = load_scorer(source)
        scorer.validate_runtime_manifest(queue["judge_runtime_manifest"]["path"])
        model = read(Path(queue["assets"]) / "model_ready.json")
        for name, info in model["inventory"].items():
            weight = Path(model["path"]) / name
            require(weight.is_file() and weight.stat().st_size == info["bytes"]
                    and file_hash(weight) == info["sha256"], "Judge inventory file differs: " + name)
        original = original_engine_kwargs(model["path"])
        effective = effective_engine_kwargs(original, allocation)
        record.update(state="engine_starting", allocation=allocation, original_engine_kwargs=original,
                      effective_engine_kwargs=effective)
        write(path, record)
        from vllm import LLM, SamplingParams
        engine = LLM(**effective)
        observed = verify_engine(engine, allocation)
        try:
            outputs = engine.chat([[dict(role="user", content="Write the digit 1.")]],
                                  SamplingParams(temperature=0, max_tokens=1, seed=0),
                                  chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
            require(len(outputs) == 1 and len(outputs[0].outputs) == 1, "Synthetic judge probe did not return one completion")
        finally:
            shutdown = getattr(getattr(engine.llm_engine, "engine_core", None), "shutdown", None)
            if shutdown is not None:
                shutdown()
        record.update(state="completed", destination_runtime_tested=True, engine_config=observed,
                      synthetic=True, benchmark_samples_used=0,
                      synthetic_output=outputs[0].outputs[0].text, completed_unix=time.time())
        write(path, record)
        print(json.dumps(dict(state="completed", probe=str(path), synthetic=True)), flush=True)
    except BaseException as exc:
        record.update(state="failed", error_type=type(exc).__name__, error=str(exc), failed_unix=time.time())
        write(path, record)
        raise


def run_score(args):
    plan, host, queue, source = load_context(args.plan, args.host_contract)
    job = selected_job(plan, args.job)
    directory = Path(plan["output"]) / "jobs" / job["name"]
    require((directory / "predictions.json").is_file(), "Actor predictions are not terminal")
    probe_path, probe = validate_probe(plan, host, source, args.host_contract, args.gpu)
    path = directory / "scoring/host_execution.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / "host_execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not path.exists(), "Judge host attempt already exists; no automatic retry is allowed")
        require(not (path.parent / "scores.json").exists(), "Existing terminal scores must not be rerun")
        record = base_record(plan, host, source, "benchmark_scoring")
        record.update(job=job, predictions_sha256=file_hash(directory / "predictions.json"),
                      probe_path=str(probe_path), probe_sha256=file_hash(probe_path),
                      destination_runtime_tested=True)
        write(path, record)
        try:
            prepare_cuda(args.gpu)
            allocation = runtime_allocation(host, args.gpu)
            record.update(state="preparing_scorer", allocation=allocation)
            write(path, record)
            scorer = load_scorer(source)
            import vllm
            original_llm = vllm.LLM
            model = read(Path(queue["assets"]) / "model_ready.json")
            expected = original_engine_kwargs(model["path"])

            def host_llm(*positional, **kwargs):
                require(not positional and kwargs == expected, "Frozen scorer LLM arguments changed")
                effective = effective_engine_kwargs(kwargs, allocation)
                record.update(state="engine_starting", original_engine_kwargs=kwargs,
                              effective_engine_kwargs=effective)
                write(path, record)
                engine = original_llm(**effective)
                record.update(state="scoring", engine_config=verify_engine(engine, allocation))
                write(path, record)
                return engine

            vllm.LLM = host_llm
            manifest = read(plan["assets"]["browsecomp_plus"]["manifest"])
            options = argparse.Namespace(benchmark="browsecomp_plus", predictions=str(directory / "predictions.json"),
                output=str(directory / "scoring"), assets=queue["assets"], endpoint=queue["endpoint"],
                qrels=manifest["qrel_evidence"]["path"], runtime_manifest=queue["judge_runtime_manifest"]["path"],
                gpu_memory_utilization=NOMINAL_UTILIZATION, max_model_len=16384)
            try:
                result = scorer.score(options)
            finally:
                vllm.LLM = original_llm
            record.update(state="completed", scoring_contract_sha256=digest(read(path.parent / "contract.json")),
                          scores_sha256=file_hash(path.parent / "scores.json"), completed_unix=time.time())
            write(path, record)
            validate_host_record(args.plan, args.job, args.host_contract)
            print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
        except BaseException as exc:
            record.update(state="failed", error_type=type(exc).__name__, error=str(exc), failed_unix=time.time())
            write(path, record)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "score", "validate-host-record", "validate-probe-record"):
        command = sub.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--host-contract", type=Path, required=True)
        if name != "validate-host-record":
            command.add_argument("--gpu", type=int, required=True)
        if name in ("score", "validate-host-record"):
            command.add_argument("--job", required=True)
        if name == "probe":
            command.add_argument("--output", type=Path, help="Optional explicit judge_probe_gpu<N>.json path beside the host contract")
    args = parser.parse_args(argv)
    if args.command == "probe":
        run_probe(args)
    elif args.command == "score":
        run_score(args)
    elif args.command == "validate-probe-record":
        validate_probe_record(args.plan, args.host_contract, args.gpu)
        print(json.dumps(dict(valid=True, probe=str(probe_record_path(args.host_contract, args.gpu)))))
    else:
        record = validate_host_record(args.plan, args.job, args.host_contract)
        print(json.dumps(dict(valid=True, host_record_present=record is not None)))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Judge host execution stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
