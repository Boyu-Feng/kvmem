"""Isolated ALFWorld text-environment accuracy evaluation (not a latency benchmark).

Main defaults to every released OOD task. Explicit main slices are supported for
sharding, but are labelled partial-split evidence. Smoke is exactly the first two
seed-233 ID tasks. Every arm sees the same zero-shot prompt and native admissible
commands. There are no demonstrations, expert wrappers, action repairs, reward
inputs, or QA surrogates. Runtime errors stop the run and are never game losses.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_rebuttal_legacy import atomic_json
from scripts.alfworld_rebuttal_env import (
    ALFWORLD_VERSION, SELECTION_PROTOCOL, SPLITS, AlfworldTextEnvironment, Game,
    Observation, discover_games, file_sha256, select_games, validate_release_inventory,
    verify_split_separation,
)


METHODS = ("fullkv", "stepkv", "h2o_step", "tova_step", "stepkv_no_step_score")
SCHEMA_VERSION = 1
TRANSFORMERS_VERSION = "5.0.0"
SYSTEM_PROMPT = (
    "You are controlling an agent in an ALFWorld text-based household environment. "
    "Complete the task described in the initial observation. At each turn, output "
    "exactly one command from the current admissible commands, on one line. "
    "Admissible commands are supplied by the environment, not an expert or a "
    "recommended plan. Output only the command: no thoughts, explanations, quotes, "
    "code fences, or 'Action:' prefix. Use observations from previous turns to "
    "decide what to do next."
)
OBSERVATION_SUFFIX = "\n\nAdmissible commands (environment-provided, not a plan):\n"
ACTION_PROTOCOL = (
    "Strip only leading/trailing whitespace. Require a single line exactly in the "
    "current native admissible_commands. Preserve the raw generation and send the "
    "stripped command to TextWorld even when invalid. No repair, first-line "
    "extraction, oracle substitution, or retry within a step."
)
IMPORTED_SOURCE_HASHES = {
    name: file_sha256(ROOT / name) for name in (
        "scripts/run_alfworld_rebuttal.py", "scripts/alfworld_rebuttal_env.py",
        "run_rebuttal_legacy.py",
    )
}


def json_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def render_observation(observation: Observation) -> str:
    return observation.text + OBSERVATION_SUFFIX + "\n".join(observation.admissible_commands)


def parse_action(raw: str, admissible_commands: tuple[str, ...] | list[str]) -> dict:
    if not isinstance(raw, str):
        raise TypeError("AgentKVPolicy.act() must return a string")
    command = raw.strip()
    reason = (
        "empty" if not command else
        "multiple_lines" if "\n" in command or "\r" in command else
        "not_admissible" if command not in admissible_commands else None
    )
    return {"raw_action": raw, "action": command, "valid": reason is None,
            "invalid_reason": reason}


class EpisodeRuntimeError(RuntimeError):
    def __init__(self, stage: str, partial_episode: dict):
        super().__init__(f"ALFWorld episode runtime failure at {stage}")
        self.partial_episode = partial_episode
        self.stage = stage


def evaluate_episode(policy, game: Game, *, episode_index: int, seed: int,
                     max_steps: int, env_factory=AlfworldTextEnvironment) -> dict:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    started = time.perf_counter()
    row = {
        "episode_index": episode_index, "game": game.manifest(), "episode_seed": seed,
        "trajectory": [], "shared_gpu_latency_not_valid": True,
    }
    stage = "verify_inputs"
    try:
        game.verify_inputs()
        stage = "environment_reset"
        with env_factory(game, seed=seed) as env:
            state = env.reset()
            row.update(initial_observation=state.text,
                       initial_admissible_commands=list(state.admissible_commands),
                       initial_won=state.won, initial_done=state.done,
                       initial_policy_observation=None, policy_used=False)
            if not state.done and not state.won:
                stage = "policy_reset"
                row["initial_policy_observation"] = render_observation(state)
                policy.reset(SYSTEM_PROMPT, row["initial_policy_observation"])
                row["policy_used"] = True
                for step in range(1, max_steps + 1):
                    stage = "policy_act"
                    item = {"step": step, "admissible_commands": list(state.admissible_commands)}
                    row["trajectory"].append(item)
                    item.update(parse_action(policy.act(), state.admissible_commands))
                    stage = "environment_step"
                    state = env.step(item["action"])
                    item.update(observation=state.text,
                                next_admissible_commands=list(state.admissible_commands),
                                environment_won=state.won, environment_done=state.done,
                                policy_observation=None)
                    if state.won or state.done or step == max_steps:
                        break
                    stage = "policy_observe"
                    item["policy_observation"] = render_observation(state)
                    policy.observe(item["action"], item["policy_observation"], valid=item["valid"])
            row.update(
                success=state.won,
                termination="won" if state.won else
                            "environment_terminal" if state.done else "step_limit",
                steps=len(row["trajectory"]),
                invalid_actions=sum(not item["valid"] for item in row["trajectory"]),
            )
            stage = "environment_close"
        stage = "policy_audit"
        audit = policy.audit() if row["policy_used"] else {}
        if not isinstance(audit, dict):
            raise TypeError("AgentKVPolicy.audit() must return a JSON dictionary")
        row["audit"] = json_copy(audit)
        row["diagnostic_seconds"] = time.perf_counter() - started
        row["status"] = "completed"
        json_hash(row)
        return row
    except BaseException as error:
        row["diagnostic_seconds"] = time.perf_counter() - started
        if not isinstance(error, Exception):
            error.partial_episode, error.stage = row, stage
            raise
        raise EpisodeRuntimeError(stage, row) from error


def source_hashes() -> dict:
    paths = [ROOT / name for name in (
        "scripts/run_alfworld_rebuttal.py", "scripts/alfworld_rebuttal_env.py",
        "models/AgentKVPolicy.py", "run_rebuttal_legacy.py", "token_tracker.py",
    )]
    for directory in ("models", "kv_cache"):
        paths.extend((ROOT / directory).rglob("*.py"))
    requirements = ROOT / "config/environments/h200_accuracy_20260929.requirements.txt"
    if requirements.exists():
        paths.append(requirements)
    hashes = {path.relative_to(ROOT).as_posix(): file_sha256(path)
              for path in sorted(set(paths))}
    if any(hashes[name] != expected for name, expected in IMPORTED_SOURCE_HASHES.items()):
        raise RuntimeError("Loaded evaluation source changed; restart with a new output directory")
    return hashes


def runtime_versions() -> dict:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use the isolated Python 3.12 ALFWorld environment")
    required = {name: importlib.metadata.version(name)
                for name in ("alfworld", "textworld", "torch", "transformers", "numpy")}
    if required["alfworld"] != ALFWORLD_VERSION:
        raise RuntimeError(f"This protocol requires alfworld=={ALFWORLD_VERSION}")
    if required["transformers"] != TRANSFORMERS_VERSION:
        raise RuntimeError(f"This protocol requires transformers=={TRANSFORMERS_VERSION}")
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                if dist.metadata.get("Name")}
    return {"python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
            "platform": platform.platform(), "required": required,
            "packages": dict(sorted(packages.items()))}


def model_inputs(model_path: Path) -> dict:
    model_path = model_path.resolve()
    config_path = model_path / "config.json"
    config = read_json(config_path)
    if config.get("model_type") != "qwen2":
        raise RuntimeError("This protocol requires standard-attention Qwen2.5-Instruct model inputs")
    files = sorted({*model_path.glob("*.json"), *model_path.glob("*.jinja"),
                    *model_path.glob("*.model"), *model_path.glob("merges.txt"),
                    *model_path.glob("vocab.txt")})
    weights = sorted({*model_path.glob("*.safetensors"), *model_path.glob("pytorch_model*.bin")})
    if not weights or any(path.stat().st_size == 0 for path in weights):
        raise RuntimeError(f"No complete local model weights found at {model_path}")
    for index in model_path.glob("*.index.json"):
        for shard in set(read_json(index).get("weight_map", {}).values()):
            if Path(shard).name != shard or not (model_path / shard).is_file():
                raise RuntimeError(f"Missing or invalid local model shard: {shard}")
    return {
        "path": str(model_path), "config": config,
        "generation_config": read_json(model_path / "generation_config.json")
        if (model_path / "generation_config.json").exists() else None,
        "input_sha256": {path.name: file_sha256(path) for path in files},
        "weights": [{"name": path.name, "resolved_path": str(path.resolve()),
                     "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                    for path in weights],
        "weight_identity_scope": (
            "Resolved local paths, sizes and nanosecond mtimes; multi-GB weights "
            "are not rehashed. Config/tokenizer/index files are SHA-256 hashed."
        ),
    }


def make_protocol(args, inventory: dict, selected: list[Game]) -> dict:
    return {
        "protocol": "alfworld_native_text_accuracy_v1",
        "purpose": args.purpose, "split": args.split,
        "method": args.method, "ratio": args.ratio, "seed": args.seed,
        "samples": args.samples, "sample_start": args.sample_start,
        "limits": {"max_steps": args.max_steps, "max_new_tokens": args.max_new_tokens,
                   "max_context_tokens": args.max_context_tokens,
                   "memory_limit_gib": args.memory_limit_gib},
        "placement": {"device": args.device,
                      "memory_limit_scope": "uniform max_memory GiB per visible GPU",
                      "model_scope": ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-32B-Instruct"]},
        "data_root": str(args.data_root.resolve()),
        "model": model_inputs(args.model_path),
        "source_hashes": source_hashes(), "runtime": runtime_versions(),
        "inventory": inventory, "selected_games": [game.manifest() for game in selected],
        "selection_protocol": SELECTION_PROTOCOL,
        "split_separation": "All eligible ID/OOD task IDs and game-file hashes checked disjoint",
        "system_prompt": SYSTEM_PROMPT, "observation_suffix": OBSERVATION_SUFFIX,
        "policy_inputs": (
            "Only zero-shot system prompt, native feedback, sorted native admissible "
            "commands, previous submitted command, and its admissibility boolean. "
            "The environment and observe() receive the act() return string with only "
            "outer whitespace stripped. Raw output is retained in the trajectory; "
            "the policy preserves emitted token IDs without replacement. "
            "Never rewards, won/done flags, task metadata, facts, expert plans or "
            "walkthroughs. No terminal/step-limit observation is sent to the policy."
        ),
        "action_protocol": ACTION_PROTOCOL,
        "environment": {
            "project": "https://github.com/alfworld/alfworld",
            "source_reference": (
                "https://github.com/alfworld/alfworld/blob/0.4.2/"
                "alfworld/agents/environment/alfred_tw_env.py"
            ),
            "backend": "textworld.start native PddlEnv; synchronous, one game per episode",
            "wrappers": ["AlfredDemangler(shuffle=False)", "AlfredInfos"],
            "request_infos": {"won": True, "admissible_commands": True, "extras": ["gamefile"]},
            "adaptation": (
                "Official evaluation wrappers without Gym batching/asynchronous workers "
                "or AlfredExpert. Loop enforces max_steps, counting invalid attempts. "
                "A fresh native environment and policy reset for every selected game."
            ),
        },
        "policy_adaptation": (
            "Shared AgentKVPolicy model/greedy decoder for FullKV and pruning arms. "
            "Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct use standard attention "
            "with KV caches in every layer. H2O/TOVA are step-boundary adaptations, "
            "not official replicas. Automatic placement supports one visible GPU "
            "for 7B and two visible GPUs for 32B with uniform per-GPU memory limits. "
            "Exact model/cache/selector settings are in policy_fingerprint."
        ),
        "success_definition": "Native TextWorld won boolean only; never reward or text matching",
        "runtime_failure_protocol": "Stop, preserve partial trajectory, exit nonzero; no lost-game row",
        "shared_gpu_latency_not_valid": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def assert_unchanged(protocol: dict, game: Game | None = None) -> None:
    if source_hashes() != protocol["source_hashes"]:
        raise RuntimeError("Evaluation source changed during the run; refusing to mix results")
    if model_inputs(Path(protocol["model"]["path"])) != protocol["model"]:
        raise RuntimeError("Local model inputs changed during the run")
    if game is not None:
        game.verify_inputs()


@contextmanager
def output_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another runner owns output directory: {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def validate_episode(row: dict, games: list[Game], protocol: dict) -> None:
    index = row["episode_index"]
    if type(index) is not int or not 0 <= index < len(games):
        raise RuntimeError("Checkpoint episode index is outside selected tasks")
    if row["game"] != games[index].manifest() or row["episode_seed"] != protocol["seed"]:
        raise RuntimeError("Checkpoint game/input/seed identity mismatch")
    trajectory = row["trajectory"]
    if (row.get("status") != "completed" or type(row["success"]) is not bool or
            type(row["steps"]) is not int or type(row["invalid_actions"]) is not int or
            row["steps"] < 0 or row["invalid_actions"] < 0 or
            not isinstance(trajectory, list) or
            row["steps"] != len(trajectory) or row["steps"] > protocol["limits"]["max_steps"]):
        raise RuntimeError("Checkpoint is not a completed, bounded episode")
    commands = row["initial_admissible_commands"]
    won, done = row["initial_won"], row["initial_done"]
    if type(won) is not bool or type(done) is not bool:
        raise RuntimeError("Checkpoint terminal flags must be native booleans")
    for step, item in enumerate(trajectory, start=1):
        if won or done or item["step"] != step or item["admissible_commands"] != commands:
            raise RuntimeError("Checkpoint trajectory has inconsistent environment transitions")
        parsed = parse_action(item["raw_action"], commands)
        if any(item[key] != value for key, value in parsed.items()):
            raise RuntimeError("Checkpoint action validity differs from the recorded admissible list")
        won, done = item["environment_won"], item["environment_done"]
        commands = item["next_admissible_commands"]
        if type(won) is not bool or type(done) is not bool:
            raise RuntimeError("Checkpoint terminal flags must be native booleans")
    termination = "won" if won else "environment_terminal" if done else "step_limit"
    if (row["success"] != won or row["termination"] != termination or
            row["invalid_actions"] != sum(not item["valid"] for item in trajectory) or
            (termination == "step_limit" and len(trajectory) != protocol["limits"]["max_steps"])):
        raise RuntimeError("Checkpoint success/termination accounting is inconsistent")
    if not isinstance(row["audit"], dict):
        raise RuntimeError("Checkpoint is missing the policy audit")


class CheckpointStore:
    def __init__(self, directory: Path, protocol: dict):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "manifest.json"
        self.protocol = json_copy(protocol)
        digest = json_hash(protocol)
        if self.path.exists():
            self.manifest = read_json(self.path)
            if (self.manifest.get("schema_version") != SCHEMA_VERSION or
                    self.manifest.get("protocol_sha256") != digest or
                    self.manifest.get("protocol") != protocol):
                raise RuntimeError("Refusing changed source/protocol/runtime/tasks; use a new output directory")
            policy = self.manifest["policy_fingerprint"]
            expected = json_hash({"protocol": protocol, "policy": policy}) if policy is not None else None
            if self.manifest["run_sha256"] != expected:
                raise RuntimeError("Corrupt manifest run fingerprint")
        else:
            if any(path.name != ".run.lock" for path in self.directory.iterdir()):
                raise RuntimeError("Existing output has no matching manifest; use a new output directory")
            self.manifest = {
                "schema_version": SCHEMA_VERSION, "created_utc": utc_now(),
                "protocol": self.protocol, "protocol_sha256": digest,
                "policy_fingerprint": None, "run_sha256": None,
            }
            atomic_json(self.path, self.manifest)

    def bind_policy(self, fingerprint: dict) -> None:
        if not isinstance(fingerprint, dict) or not fingerprint:
            raise TypeError("AgentKVPolicy.fingerprint() must return a nonempty JSON dictionary")
        fingerprint = json_copy(fingerprint)
        existing = self.manifest["policy_fingerprint"]
        if existing is not None and existing != fingerprint:
            raise RuntimeError("Refusing changed policy fingerprint; use a new output directory")
        if existing is None:
            if list((self.directory / "episodes").glob("*.json")):
                raise RuntimeError("Checkpoints exist without a bound policy fingerprint")
            self.manifest["policy_fingerprint"] = fingerprint
            self.manifest["run_sha256"] = json_hash({"protocol": self.protocol, "policy": fingerprint})
            atomic_json(self.path, self.manifest)

    def assert_policy_unchanged(self, fingerprint: dict) -> None:
        if json_copy(fingerprint) != self.manifest["policy_fingerprint"]:
            raise RuntimeError("Policy fingerprint changed during the run")

    def checkpoint_path(self, row: dict) -> Path:
        suffix = hashlib.sha256(row["game"]["game_id"].encode()).hexdigest()[:16]
        return self.directory / "episodes" / f"{row['episode_index']:05d}-{suffix}.json"

    def load(self, games: list[Game]) -> list[dict]:
        rows = {}
        for path in sorted((self.directory / "episodes").glob("*.json")):
            checkpoint = read_json(path)
            row = checkpoint["episode"]
            if (checkpoint.get("schema_version") != SCHEMA_VERSION or
                    self.manifest["run_sha256"] is None or
                    checkpoint["run_sha256"] != self.manifest["run_sha256"] or
                    checkpoint["episode_sha256"] != json_hash(row)):
                raise RuntimeError(f"Checkpoint fingerprint/content mismatch: {path}")
            validate_episode(row, games, self.protocol)
            index = row["episode_index"]
            if index in rows or self.checkpoint_path(row) != path:
                raise RuntimeError(f"Duplicate or misnamed checkpoint: {path}")
            rows[index] = row
        return [rows[index] for index in sorted(rows)]

    def save(self, row: dict, games: list[Game]) -> None:
        if self.manifest["run_sha256"] is None:
            raise RuntimeError("Cannot checkpoint before binding the policy fingerprint")
        validate_episode(row, games, self.protocol)
        checkpoint = {"schema_version": SCHEMA_VERSION,
                      "run_sha256": self.manifest["run_sha256"],
                      "episode_sha256": json_hash(row), "episode": row}
        path = self.checkpoint_path(row)
        if path.exists() and read_json(path) != checkpoint:
            raise RuntimeError(f"Refusing to overwrite a committed episode: {path}")
        atomic_json(path, checkpoint)

    def failures(self) -> list[dict]:
        failures = []
        for path in sorted((self.directory / "failures").glob("*.json")):
            failure = read_json(path)
            if failure["protocol_sha256"] != self.manifest["protocol_sha256"]:
                raise RuntimeError(f"Failure record belongs to a different protocol: {path}")
            failures.append(failure)
        return failures

    def record_failure(self, error: BaseException, *, partial_episode: dict | None = None) -> None:
        failure = {
            "protocol_sha256": self.manifest["protocol_sha256"],
            "run_sha256": self.manifest["run_sha256"], "created_utc": utc_now(),
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
            "partial_episode": getattr(error, "partial_episode", partial_episode),
            "stage": getattr(error, "stage", "runner"), "counted_as_game_loss": False,
        }
        atomic_json(self.directory / "failures" / f"{time.time_ns()}-{uuid.uuid4().hex}.json", failure)


def audit_metrics(rows: list[dict]) -> dict:
    values = {}

    def collect(value, path=""):
        if "token_ids" in path or "global_ids" in path:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                collect(child, f"{path}.{key}" if path else key)
        elif isinstance(value, list):
            for child in value:
                collect(child, path + "[]")
        elif type(value) in (int, float) and ("byte" in path.lower() or "token" in path.lower()):
            if not math.isfinite(value):
                raise ValueError(f"Non-finite policy audit metric: {path}")
            values.setdefault(path, []).append(value)

    for row in rows:
        collect(row["audit"])
    return {path: {"count": len(items), "sum": sum(items), "mean": sum(items) / len(items),
                   "min": min(items), "max": max(items),
                   "unit": "bytes" if "byte" in path.lower() else "tokens"}
            for path, items in sorted(values.items())}


def cache_summary(rows: list[dict]) -> dict:
    """Equal-episode payload statistics, separate from allocator memory and time."""
    required = {
        "resident_tokens", "protected_resident_tokens", "nonprompt_logical_tokens",
        "nonprompt_resident_tokens", "full_attention_kv_bytes", "peak_kv_bytes", "boundaries",
    }
    samples, device_peaks = [], {}
    for row in rows:
        audit = row["audit"]
        if not required.issubset(audit):
            continue
        tokens, payload = audit["resident_tokens"], audit["full_attention_kv_bytes"]
        if tokens <= 0 or payload < 0 or payload % tokens:
            raise ValueError("KV tensor payload must have an exact positive resident-position count")
        per_position = payload // tokens
        protected = audit["protected_resident_tokens"]
        if not 0 <= protected <= tokens:
            raise ValueError("Protected cache exceeds total resident cache")
        values = {
            "final_total_kv_tokens": tokens,
            "final_trajectory_kv_tokens": audit["nonprompt_resident_tokens"],
            "final_total_kv_mib": payload / 2**20,
            "final_prompt_kv_mib": protected * per_position / 2**20,
            "final_trajectory_kv_mib": (tokens - protected) * per_position / 2**20,
            "peak_active_total_kv_mib": audit["peak_kv_bytes"] / 2**20,
        }
        logical = audit["nonprompt_logical_tokens"]
        values["final_trajectory_retention"] = audit["nonprompt_resident_tokens"] / logical if logical else 1.0
        boundaries = audit["boundaries"]
        if boundaries:
            for field, source in (("total_kv_tokens", "after_tokens"), ("total_kv_mib", "after_kv_bytes")):
                scale = 2**20 if field.endswith("mib") else 1
                values[f"episode_mean_boundary_{field}"] = sum(
                    event[source] / scale for event in boundaries
                ) / len(boundaries)
            values["episode_mean_boundary_trajectory_kv_mib"] = sum(
                event["after_kv_bytes"] * (event["after_tokens"] - event["protected_tokens"])
                / event["after_tokens"] / 2**20 for event in boundaries
            ) / len(boundaries)
        samples.append(values)
        for device, size in audit.get("peak_kv_bytes_by_device", {}).items():
            device_peaks.setdefault(device, []).append(size / 2**20)
    names = sorted({name for sample in samples for name in sample})
    result = {
        "episodes_with_cache_audit": len(samples),
        "episodes_missing_cache_audit": len(rows) - len(samples),
        "mean": {
            name: sum(sample[name] for sample in samples if name in sample)
                  / sum(name in sample for sample in samples)
            for name in names
        },
        "max_peak_active_total_kv_mib": max(
            (sample["peak_active_total_kv_mib"] for sample in samples), default=None,
        ),
        "mean_peak_kv_mib_by_device": {
            device: sum(sizes) / len(sizes) for device, sizes in sorted(device_peaks.items())
        },
        "definition": (
            "Actual live K/V tensor payload, including protected prompt; excludes parameters, "
            "allocator workspace and temporary duplicate tensors. Boundary means are averaged "
            "within each episode then equally across episodes with observed boundaries. "
            "Terminal pending actions remain in final/peak cache, without inventing a boundary. "
            "Prompt/trajectory bytes use the measured ordinary-Qwen payload per resident position."
        ),
    }
    return result


def summarize(rows: list[dict], protocol: dict, failures: list[dict], *, failed: bool = False) -> dict:
    expected = len(protocol["selected_games"])
    inventory_count = protocol["inventory"][protocol["split"]]["eligible_count"]
    count = len(rows)
    successes = sum(row["success"] for row in rows)
    steps = sum(row["steps"] for row in rows)
    invalid = sum(row["invalid_actions"] for row in rows)
    full_split = expected == inventory_count and protocol["sample_start"] == 0
    complete = count == expected and not failed
    return {
        "status": "failed" if failed else "complete" if complete else "partial",
        "purpose": protocol["purpose"], "split": protocol["split"],
        "method": protocol["method"], "ratio": protocol["ratio"], "seed": protocol["seed"],
        "selected_count": expected, "inventory_count": inventory_count,
        "completed_count": count, "successes": successes,
        "success_rate": successes / count if count else None,
        "success_rate_denominator": "completed native episodes only; failed attempts excluded",
        "benchmark_complete": complete and full_split and protocol["purpose"] == "main",
        "full_split_selected": full_split,
        "full_split_success_rate": successes / count if complete and full_split else None,
        "failures": {"unsuccessful_completed_episodes": count - successes,
                     "runtime_error_attempts": len(failures)},
        "runtime_failures_count_as_losses": False,
        "invalid_actions": invalid, "action_attempts": steps,
        "invalid_action_rate": invalid / steps if steps else None,
        "total_steps": steps, "mean_steps": steps / count if count else None,
        "audit_byte_token_metrics": audit_metrics(rows),
        "kv_cache_statistics": cache_summary(rows),
        "audit_aggregation": "Exact audit JSON paths; [] pools samples within lists, not episode totals",
        "mean_diagnostic_episode_seconds":
            sum(row["diagnostic_seconds"] for row in rows) / count if count else None,
        "shared_gpu_latency_not_valid": True,
        "accuracy_scope": "smoke validation only" if protocol["purpose"] == "smoke" else
                          "full OOD split" if complete and full_split else "incomplete/sliced OOD evidence",
    }


def persist_reports(store: CheckpointStore, rows: list[dict], *, failed: bool = False) -> dict:
    rows = sorted(rows, key=lambda row: row["episode_index"])
    failures = store.failures()
    summary = summarize(rows, store.protocol, failures, failed=failed)
    atomic_json(store.directory / "results.json", {
        "run_sha256": store.manifest["run_sha256"], "summary": summary,
        "episodes": rows, "runtime_failures": failures,
    })
    atomic_json(store.directory / "SUMMARY.json", summary)
    return summary


def assert_report_matches_checkpoints(store: CheckpointStore, rows: list[dict]) -> None:
    results_path = store.directory / "results.json"
    if not results_path.exists():
        return
    results = read_json(results_path)
    if results.get("run_sha256") != store.manifest["run_sha256"]:
        raise RuntimeError("Existing results belong to a different run fingerprint")
    committed = {row["episode_index"]: row for row in rows}
    reported = results.get("episodes")
    if not isinstance(reported, list):
        raise RuntimeError("Existing results are missing their episode records")
    if len({row["episode_index"] for row in reported}) != len(reported):
        raise RuntimeError("Existing results contain duplicate episodes")
    for row in reported:
        if row != committed.get(row["episode_index"]):
            raise RuntimeError("Existing results reference missing/changed episode checkpoints")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--ratio", type=float, choices=[0.2, 0.5], default=0.2)
    parser.add_argument("--split", choices=SPLITS, default="eval_out_of_distribution")
    parser.add_argument("--samples", type=int, default=0, help="0 = all remaining tasks; smoke requires 2")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--device", default="auto",
                        help="Policy device/placement, default auto across visible GPUs")
    parser.add_argument("--memory-limit-gib", type=float,
                        help="Uniform max_memory budget in GiB per visible GPU")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", choices=["smoke", "main"], default="main")
    return parser


def validate_args(args) -> None:
    if (args.samples < 0 or args.sample_start < 0 or args.seed < 0 or args.seed >= 2**32 or
            min(args.max_steps, args.max_new_tokens, args.max_context_tokens) <= 0):
        raise ValueError("Nonnegative sample offsets, uint32 seed and positive token/step limits required")
    if args.memory_limit_gib is not None and (
        not math.isfinite(args.memory_limit_gib) or args.memory_limit_gib <= 0
    ):
        raise ValueError("memory-limit-gib must be finite and positive")
    if not args.device.strip():
        raise ValueError("device must be a nonempty placement string")
    if args.purpose == "smoke":
        if (args.split != "eval_in_distribution" or args.samples != 2 or
                args.sample_start != 0 or args.seed != 233):
            raise ValueError("Smoke requires eval_in_distribution, samples=2, sample-start=0, seed=233")
    elif args.split != "eval_out_of_distribution":
        raise ValueError("Main accuracy evaluation must use eval_out_of_distribution")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    args.output = args.output.resolve()
    inventory, catalogs = {}, {}
    for split in SPLITS:
        catalogs[split], inventory[split] = discover_games(args.data_root, split)
    validate_release_inventory(inventory)
    verify_split_separation(catalogs["eval_in_distribution"], catalogs["eval_out_of_distribution"])
    selected = select_games(catalogs[args.split], seed=args.seed, samples=args.samples,
                            sample_start=args.sample_start)
    protocol = make_protocol(args, inventory, selected)
    with output_lock(args.output):
        store = CheckpointStore(args.output, protocol)
        rows = store.load(selected)
        assert_report_matches_checkpoints(store, rows)
        store.failures()
        partial_episode = None
        state_path = args.output / "status.json"
        try:
            if len(rows) != len(selected):
                atomic_json(state_path, {"status": "loading_policy", "pid": os.getpid(),
                                        "updated_utc": utc_now(), "completed_count": len(rows)})
                os.environ["ALFWORLD_DATA"] = str(args.data_root.resolve())
                random.seed(args.seed)
                import numpy as np
                import torch
                from models.AgentKVPolicy import AgentKVPolicy

                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                policy = AgentKVPolicy(
                    model_path=str(args.model_path.resolve()), method=args.method,
                    keep_ratio=args.ratio, max_new_tokens=args.max_new_tokens,
                    device=args.device, max_context_tokens=args.max_context_tokens,
                    memory_limit_gib=args.memory_limit_gib,
                )
                store.bind_policy(policy.fingerprint())
                completed = {row["episode_index"] for row in rows}
                for index, game in enumerate(selected):
                    if index in completed:
                        continue
                    partial_episode = {"episode_index": index, "game": game.manifest()}
                    assert_unchanged(protocol, game)
                    atomic_json(state_path, {
                        "status": "running", "pid": os.getpid(), "updated_utc": utc_now(),
                        "completed_count": len(rows), "selected_count": len(selected),
                        "episode_index": index, "game_id": game.game_id,
                    })
                    row = evaluate_episode(policy, game, episode_index=index, seed=args.seed,
                                           max_steps=args.max_steps)
                    partial_episode = row
                    assert_unchanged(protocol, game)
                    store.assert_policy_unchanged(policy.fingerprint())
                    store.save(row, selected)
                    rows.append(row)
                    persist_reports(store, rows)
                    partial_episode = None
            assert_unchanged(protocol)
            for game in selected:
                game.verify_inputs()
            summary = persist_reports(store, rows)
            atomic_json(state_path, {"status": "complete", "pid": os.getpid(),
                                    "updated_utc": utc_now(), "completed_count": len(rows)})
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0
        except BaseException as error:
            store.record_failure(error, partial_episode=partial_episode)
            persist_reports(store, rows, failed=True)
            atomic_json(state_path, {
                "status": "failed", "pid": os.getpid(), "updated_utc": utc_now(),
                "completed_count": len(rows), "error_type": type(error).__name__,
                "error": str(error), "counted_as_game_loss": False,
            })
            raise


if __name__ == "__main__":
    raise SystemExit(main())
