"""Isolated helpers for the frozen QA latency experiment; never launch a queue."""
from __future__ import annotations

import ast
import contextlib
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.qa32_runtime import legacy_config

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "fullkv": "none", "stepkv": "step_aware_h2o", "h2o": "h2o", "tova": "tova",
    "h2o_step": "h2o", "tova_step": "tova", "flowkv_style": "flowkv_style",
    "lazyeviction": "none",
}
ADAPTERS = frozenset({"h2o_step", "tova_step", "flowkv_style", "lazyeviction"})
MODEL = ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28")
MIB = 1024 ** 2


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frozen_core_hashes():
    manifest = read_json(ROOT / "rebuttal/adapters/manifest.json")
    expected = manifest["methods"]["h2o_step"]["original_core_hashes"]
    for name, checksum in expected.items():
        if file_hash(ROOT / name) != checksum:
            raise RuntimeError(f"Frozen QA source checksum mismatch: {name}")
    return dict(expected)


def experiment_hashes():
    values = frozen_core_hashes()
    for name in (
        "scripts/latency_runtime.py", "scripts/latency_metrics.py",
        "scripts/run_latency_experiment.py", "scripts/summarize_latency_experiment.py",
        "scripts/legacy_efficiency_metrics.py", "scripts/run_rebuttal_efficiency.py",
        "scripts/qa32_runtime.py", "rebuttal/adapters/manifest.json",
        "config/rebuttal_assets_20260928.json",
    ):
        values[name] = file_hash(ROOT / name)
    for method in ADAPTERS:
        spec = read_json(ROOT / "rebuttal/adapters/manifest.json")["methods"][method]
        for name, record in spec["source_files"].items():
            if record["from"] == "adapter":
                relative = f"rebuttal/adapters/{method}/{name}"
                if file_hash(ROOT / relative) != record["sha256"]:
                    raise RuntimeError(f"Frozen adapter checksum mismatch: {relative}")
                values[relative] = record["sha256"]
    return values


@contextlib.contextmanager
def patch_attribute(obj, name, value):
    own = name in vars(obj)
    previous = vars(obj).get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if own:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


@dataclass(frozen=True)
class Method:
    name: str
    mode: str
    llm_class: type
    overrides: dict
    modules: tuple
    scope: str

    def config(self, base, ratio):
        default = {"flowkv_style": "tova", "lazyeviction": "fullkv"}.get(self.name, self.name)
        config = legacy_config(base, default, ratio)
        return {**config, **self.overrides, "cache_ratio": ratio}


@contextlib.contextmanager
def frozen_method(name):
    if name not in METHODS:
        raise ValueError(f"Unsupported latency method: {name}")
    frozen_core_hashes()
    from models.QwenLLMWithKVCache import QwenLLMWithKVCache

    if name not in ADAPTERS:
        yield Method(name, METHODS[name], QwenLLMWithKVCache, {}, (),
                     "Unchanged frozen QA generation, scoring, selection, stop/crop and positions.")
        return
    spec = read_json(ROOT / "rebuttal/adapters/manifest.json")["methods"][name]
    for relative, record in spec["source_files"].items():
        path = ROOT / (f"rebuttal/adapters/{name}/{relative}" if record["from"] == "adapter" else relative)
        if file_hash(path) != record["sha256"]:
            raise RuntimeError(f"Frozen adapter dependency checksum mismatch: {path}")
    installed = []
    modules = []

    def load(module_name, relative):
        path = ROOT / f"rebuttal/adapters/{name}/{relative}"
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != spec["source_files"][relative]["sha256"]:
            raise RuntimeError(f"Adapter changed before import: {path}")
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(module_spec)
        prior = sys.modules.get(module_name)
        installed.append((module_name, prior))
        sys.modules[module_name] = module
        exec(compile(source, str(path), "exec"), module.__dict__)
        modules.append(module)
        return module

    try:
        if name == "lazyeviction":
            if "models.LazyEvictionLLM" in sys.modules:
                raise RuntimeError("LazyEviction namespace is already loaded; use a fresh profiling process")
            load("models.LazyEvictionLLM", "models/LazyEvictionLLM.py")
        module = load(f"scripts._latency_adapter_{name}", "models/LegacyBaseline.py")
        if module.LEGACY_METHOD != name or not issubclass(module.LegacyBaselineLLM, QwenLLMWithKVCache):
            raise RuntimeError(f"Unexpected adapter identity: {name}")
        yield Method(name, METHODS[name], module.LegacyBaselineLLM,
                     dict(module.LEGACY_CONFIG_OVERRIDE), tuple(modules), module.LEGACY_SCOPE)
    finally:
        for module_name, prior in reversed(installed):
            if prior is None:
                del sys.modules[module_name]
            else:
                sys.modules[module_name] = prior


def _query(command):
    return subprocess.check_output(["nvidia-smi", *command], text=True, timeout=20)


def gpu_state():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible.startswith("MIG-"):
        raise RuntimeError("Select exactly one non-MIG GPU externally with CUDA_VISIBLE_DEVICES")
    raw = _query([f"--id={visible}", "--query-gpu=uuid,name,memory.total,driver_version",
                  "--format=csv,noheader,nounits"]).strip().splitlines()
    if len(raw) != 1:
        raise RuntimeError("Exactly one physical GPU is required")
    uuid, name, total, driver = [value.strip() for value in raw[0].split(",")]
    processes = []
    for line in _query(["--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                        "--format=csv,noheader,nounits"]).splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 3:
            raise RuntimeError(f"Invalid nvidia-smi process row: {line}")
        if fields[0] == uuid:
            processes.append({"pid": int(fields[1]), "used_bytes": int(fields[2]) * MIB})
    return {"uuid": uuid, "name": name, "memory_total_bytes": int(total) * MIB,
            "driver": driver, "visible_selector": visible, "processes": processes}


def require_exclusive():
    state = gpu_state()
    others = [entry["pid"] for entry in state["processes"] if entry["pid"] != os.getpid()]
    if others:
        raise RuntimeError(f"GPU is shared; latency measurement rejected, other PIDs: {others}")
    return state


def process_memory_reader():
    state = require_exclusive()
    own = [entry for entry in state["processes"] if entry["pid"] == os.getpid()]
    if len(own) != 1:
        raise RuntimeError("Loaded model must have one observable CUDA process-memory record")
    return {"cuda:0": own[0]["used_bytes"]}


def validate_loaded_model(model):
    import torch

    parameters = list(model.parameters())
    buffers = list(model.buffers())
    if not parameters or any(value.device != torch.device("cuda:0") for value in parameters + buffers):
        raise RuntimeError("Single-GPU model required; CPU/disk offload is forbidden")
    if any(value.dtype != torch.bfloat16 for value in parameters if value.is_floating_point()):
        raise RuntimeError("Latency conditions require BF16 model parameters")
    if any(str(value) not in ("0", "cuda:0") for value in getattr(model, "hf_device_map", {}).values()):
        raise RuntimeError("Model device map includes offload or another GPU")


@contextlib.contextmanager
def single_gpu_loader():
    from transformers import AutoModelForCausalLM

    original = AutoModelForCausalLM.from_pretrained

    def load(*args, **kwargs):
        kwargs = {**kwargs, "device_map": {"": "cuda:0"}, "local_files_only": True,
                  "trust_remote_code": False}
        model = original(*args, **kwargs)
        validate_loaded_model(model)
        return model

    with patch_attribute(AutoModelForCausalLM, "from_pretrained", load):
        yield


def utility_ranges(function):
    tree = ast.parse(Path(function.__code__.co_filename).read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                and item.name == function.__name__)
    spans = {}
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.If) or not candidate.body:
            continue
        first = candidate.body[0]
        if not isinstance(first, ast.Assign):
            continue
        names = {target.id for target in first.targets if isinstance(target, ast.Name)}
        if "prev_action_norm" in names:
            spans["reward_novelty_repeat"] = (candidate.lineno, candidate.end_lineno)
        if "ref_terms" in names:
            spans["citation_history_scan"] = (candidate.lineno, candidate.end_lineno)
    if set(spans) != {"reward_novelty_repeat", "citation_history_scan"}:
        raise RuntimeError("Frozen utility blocks could not be identified; refuse incomplete attribution")
    return spans


class EpisodeTrace:
    """Diagnostic-only tracing of inline utility and actual interaction endpoints."""

    def __init__(self, function, monitor):
        self.function = function
        self.monitor = monitor
        self.ranges = utility_ranges(function)
        self.boundaries = []
        self._utility = None
        self._utility_region = None
        self._seen_interactions = 0
        self._prior = None

    def _close_utility(self):
        if self._utility is not None:
            self._utility.__exit__(None, None, None)
            self._utility = None
            self._utility_region = None

    def _trace(self, frame, event, arg):
        if frame.f_code is not self.function.__code__:
            return None
        if event in ("line", "return"):
            region = (next((name for name, (start, end) in self.ranges.items()
                            if start <= frame.f_lineno <= end), None) if event == "line" else None)
            if region != self._utility_region:
                self._close_utility()
            if region is not None and self._utility is None:
                self._utility = self.monitor.measure("step_utility")
                self._utility.__enter__()
                self._utility_region = region
            trajectory = frame.f_locals.get("trajectory_log", [])
            timings = frame.f_locals.get("step_timings", [])
            if len(trajectory) > self._seen_interactions:
                if len(trajectory) != self._seen_interactions + 1:
                    raise RuntimeError("Interaction endpoint trace skipped a completed step")
                self._seen_interactions = len(trajectory)
                self.boundaries.append({
                    "step": trajectory[-1]["step"], "endpoint": "action_and_tool_return",
                    "memory": self.monitor.sample_memory("interaction_step_end"),
                })
            if event == "return" and timings and len(self.boundaries) < len(timings):
                self.boundaries.append({
                    "step": timings[-1]["step"], "endpoint": "terminal_generation_no_tool_return",
                    "memory": self.monitor.sample_memory("terminal_step_end"),
                })
        return self._trace

    def __enter__(self):
        self._prior = sys.gettrace()
        if self._prior is not None:
            raise RuntimeError("Existing Python trace/debugger is active; use a fresh diagnostic process")
        sys.settrace(self._trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(self._prior)
        self._close_utility()


def instrument_episode(base, method, llm, monitor):
    import importlib

    monitor.wrap(base, "execute_action", "tool")
    monitor.wrap(base, "parse_action_original", "agent_parse")
    monitor.wrap(importlib.import_module("kv_cache.pruning_strategy"),
                 "_build_token_score_snapshot", "debug_snapshot_export")
    if method.name == "lazyeviction":
        import torch
        import torch.nn.functional as functional

        module = method.modules[-1]
        monitor.wrap(torch, "matmul", "token_saliency")
        monitor.wrap(functional, "softmax", "token_saliency")
        monitor.wrap(module, "repeat_kv", "head_expansion")
        monitor.wrap(module, "update_recurrence", "token_saliency")
        monitor.wrap(module, "_select_at_boundary", "selection_and_cache_management")
        monitor.wrap(module, "gather_state", "cache_gather")
    if method.name == "flowkv_style":
        monitor.wrap(llm, "_freeze_completed_block", "selection_and_cache_management")


@contextlib.contextmanager
def observe_kv_gathers(llm, monitor):
    """Time explicit Tensor gathers only when their input aliases live K/V."""
    import torch
    from scripts.legacy_efficiency_metrics import cache_tensor_pairs

    def is_live_kv(tensor):
        identity = (str(tensor.device), tensor.untyped_storage().data_ptr())
        caches = (llm.past_key_values, getattr(llm, "_legacy_active_cache", None))
        return any(identity == (str(value.device), value.untyped_storage().data_ptr())
                   for cache in caches for pair in cache_tensor_pairs(cache) for value in pair)

    def wrap(original):
        def gather(tensor, *args, **kwargs):
            if not is_live_kv(tensor):
                return original(tensor, *args, **kwargs)
            with monitor.measure("cache_gather"):
                return original(tensor, *args, **kwargs)
        return gather

    with contextlib.ExitStack() as stack:
        for name in ("gather", "index_select"):
            stack.enter_context(patch_attribute(torch.Tensor, name, wrap(getattr(torch.Tensor, name))))
        yield


def validate_diagnostics(stats, interaction_boundaries, expected_blocks):
    if stats.get("instrumentation_errors") or stats.get("failed_calls"):
        raise RuntimeError("Failed diagnostic instrumentation/calls cannot be reported as measurements")
    if len(stats["steps"]) != expected_blocks or len(interaction_boundaries) != expected_blocks:
        raise RuntimeError("Diagnostic generation/interaction endpoints do not cover every actual step")
    if not stats["steps"]:
        raise RuntimeError("No generation blocks were measured")
    for memory in [step["memory"] for step in stats["steps"]] + [
        step["memory"] for step in interaction_boundaries
    ] + [stats["final"]]:
        kv = memory["kv"]
        if kv["total_bytes"] != kv["protected_bytes"] + kv["trajectory_bytes"]:
            raise RuntimeError("Invalid prompt/trajectory KV byte partition")
        if not kv["layout"] or kv["total_bytes"] <= 0:
            raise RuntimeError("Missing actual per-layer KV measurements")
    for phase, seconds in stats["timing"]["exclusive_seconds"].items():
        if not isinstance(seconds, (int, float)) or not 0 <= seconds < float("inf"):
            raise RuntimeError(f"Invalid diagnostic timing: {phase}")


def native_equivalent_budget(memory, nominal_ratio, processed_nonprompt):
    """Assess actual bytes against a native-GQA target; never change selection."""
    kv = memory["kv"]
    if not 0 < nominal_ratio <= 1 or not isinstance(processed_nonprompt, int) or processed_nonprompt < 0:
        raise ValueError("Native byte reference requires a valid ratio and nonnegative logical token count")
    native_per_token = 0
    for layer in kv["layout"]:
        key_shape, value_shape = layer["key_shape"], layer["value_shape"]
        stored_heads = key_shape[1]
        if stored_heads != value_shape[1] or stored_heads <= 0:
            raise RuntimeError("Inconsistent stored KV heads")
        native_heads = layer["native_heads"]
        per_token = (layer["key_bytes"] + layer["value_bytes"]) // key_shape[2]
        native_per_token += per_token * native_heads // stored_heads
    protected = kv["protected_tokens"]
    if protected is None:
        raise RuntimeError("Native byte reference requires a uniform protected prefix and logical clock")
    native_prompt = native_per_token * protected
    target = native_prompt + native_per_token * int(nominal_ratio * processed_nonprompt)
    expanded_floor = kv["protected_bytes"]
    return {
        "native_gqa_reference_bytes": target, "actual_kv_bytes": kv["total_bytes"],
        "actual_over_native_reference": kv["total_bytes"] / target if target else None,
        "expanded_protected_prompt_bytes": expanded_floor,
        "protected_prompt_alone_exceeds_reference": expanded_floor > target,
        "matched_byte_budget": False,
        "scope": "Observed payload vs nominal native-GQA reference only; not a byte-matched run. "
                 "LazyEviction additionally reserves its W=175 recent window and initially disables eviction.",
    }
