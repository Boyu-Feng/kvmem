"""Native, text-only ALFWorld adapter; importing this module needs only Python.

The environment-only diagnostic at the bottom does not load a policy or use a
GPU. Game selection follows ALFWorld 0.4.2's released supported/solvable inventory,
not a model's performance. Neither expert plans nor walkthroughs are requested.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
from typing import Mapping


ALFWORLD_VERSION = "0.4.2"
SPLITS = {
    "eval_in_distribution": "valid_seen",
    "eval_out_of_distribution": "valid_unseen",
}
# Verified from the official 0.4.2 json_2.1.3_tw-pddl.zip evaluation assets.
RELEASE_GAME_COUNTS = {"eval_in_distribution": 140, "eval_out_of_distribution": 134}
TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)
SELECTION_PROTOCOL = (
    "ALFWorld 0.4.2 six supported task types; exclude movable/Sliced paths and "
    "games marked solvable=false in released assets. Original ALFRED trajectories "
    "without released text games are documented exclusions, as in AlfredTWEnv. "
    "Missing/corrupt metadata for a released game is an error. The benchmark "
    "entrypoints require the verified complete 140 ID / 134 OOD game inventory. "
    "No model score, rollout outcome, or expert action is used for selection. "
    "Sort task/trial IDs, then random.Random(seed).shuffle."
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Game:
    game_id: str
    split: str
    data_root: Path
    game_path: Path
    trajectory_path: Path
    task_type: str
    input_hashes: dict[str, str]

    def manifest(self) -> dict:
        return {
            "game_id": self.game_id,
            "split": self.split,
            "game_path": str(self.game_path),
            "relative_game_path": self.game_path.relative_to(self.data_root).as_posix(),
            "trajectory_path": str(self.trajectory_path),
            "task_type": self.task_type,
            "input_hashes": self.input_hashes,
        }

    def verify_inputs(self) -> None:
        for relative, expected in self.input_hashes.items():
            path = self.data_root / relative
            if not path.is_file() or file_sha256(path) != expected:
                raise RuntimeError(f"ALFWorld input changed: {path}")


def discover_games(data_root: Path, split: str) -> tuple[list[Game], dict]:
    if split not in SPLITS:
        raise ValueError(f"Unknown ALFWorld split: {split}")
    data_root = Path(data_root).resolve()
    split_path = data_root / "json_2.1.1" / SPLITS[split]
    if not split_path.is_dir():
        raise FileNotFoundError(f"Missing official ALFWorld split directory: {split_path}")
    trajectories = sorted(split_path.rglob("traj_data.json"))
    if not trajectories:
        raise RuntimeError(f"Empty ALFWorld task inventory: {split_path}")
    text_games = sorted(split_path.rglob("game.tw-pddl"))
    for path in text_games:
        if not path.with_name("traj_data.json").is_file():
            raise FileNotFoundError(f"Missing trajectory metadata for released text game: {path}")
    games, excluded = [], []
    for trajectory in trajectories:
        game_id = trajectory.parent.relative_to(split_path).as_posix()
        game_path = trajectory.with_name("game.tw-pddl")
        hashes = {trajectory.relative_to(data_root).as_posix(): file_sha256(trajectory)}
        if game_path.is_file():
            hashes[game_path.relative_to(data_root).as_posix()] = file_sha256(game_path)
        reason = None
        if "movable" in game_id or "Sliced" in game_id:
            reason = "official_unsupported_movable_or_sliced"
        else:
            metadata = json.loads(trajectory.read_text(encoding="utf-8"))
            task_type = metadata["task_type"]
            if task_type not in TASK_TYPES:
                reason = "official_unsupported_task_type"
            elif not game_path.is_file():
                reason = "official_no_released_text_game"
            else:
                game_data = json.loads(game_path.read_text(encoding="utf-8"))
                if type(game_data.get("solvable")) is not bool:
                    raise RuntimeError(f"Missing/invalid official solvable flag: {game_path}")
                if not game_data["solvable"]:
                    reason = "official_solvable_false"
        if reason:
            excluded.append({"game_id": game_id, "reason": reason, "input_hashes": hashes})
        else:
            games.append(Game(game_id, split, data_root, game_path, trajectory,
                              task_type, hashes))
    if not games:
        raise RuntimeError(f"No supported, released-solvable ALFWorld games in {split_path}")
    report = {
        "split": split,
        "split_directory": str(split_path),
        "inventory_count": len(trajectories),
        "released_text_game_count": len(text_games),
        "eligible_count": len(games),
        "task_type_counts": dict(sorted(Counter(g.task_type for g in games).items())),
        "eligible": [game.manifest() for game in games],
        "excluded": excluded,
    }
    return games, report


def validate_release_inventory(inventory: dict) -> None:
    for split, expected in RELEASE_GAME_COUNTS.items():
        report = inventory[split]
        if report["released_text_game_count"] != expected or report["eligible_count"] != expected:
            raise RuntimeError(
                f"Incomplete/unexpected ALFWorld 0.4.2 inventory for {split}: "
                f"expected {expected} released eligible text games, got "
                f"{report['released_text_game_count']} released / {report['eligible_count']} eligible"
            )


def select_games(games: list[Game], *, seed: int, samples: int = 0,
                 sample_start: int = 0) -> list[Game]:
    if samples < 0 or sample_start < 0:
        raise ValueError("samples and sample_start must be nonnegative")
    ordered = sorted(games, key=lambda game: game.game_id)
    if len({game.game_id for game in ordered}) != len(ordered):
        raise RuntimeError("Duplicate ALFWorld game IDs")
    random.Random(seed).shuffle(ordered)
    end = len(ordered) if samples == 0 else sample_start + samples
    if sample_start >= len(ordered) or end > len(ordered):
        raise ValueError("Requested ALFWorld task slice is not fully available")
    return ordered[sample_start:end]


def verify_split_separation(seen: list[Game], unseen: list[Game]) -> None:
    overlap = {game.game_id for game in seen} & {game.game_id for game in unseen}
    seen_hashes = {g.input_hashes[g.game_path.relative_to(g.data_root).as_posix()] for g in seen}
    unseen_hashes = {g.input_hashes[g.game_path.relative_to(g.data_root).as_posix()] for g in unseen}
    if overlap or seen_hashes & unseen_hashes:
        raise RuntimeError(f"Main/smoke ALFWorld inventories overlap: {sorted(overlap)}")


@dataclass(frozen=True)
class Observation:
    text: str
    admissible_commands: tuple[str, ...]
    won: bool
    done: bool


def public_observation(state: Mapping, *, done: bool | None = None) -> Observation:
    """Keep evaluator terminal flags separate from policy-visible text/actions."""
    text = state["feedback"]
    commands = state["admissible_commands"]
    won = state["won"]
    done = state.get("done", won) if done is None else done
    if not isinstance(text, str):
        raise TypeError("TextWorld feedback must be text")
    if not isinstance(commands, (tuple, list)) or any(
        not isinstance(command, str) or not command or "\n" in command or "\r" in command
        for command in commands
    ):
        raise TypeError("TextWorld admissible_commands must be a list of single-line strings")
    if type(won) is not bool or type(done) is not bool:
        raise TypeError("TextWorld won/done must be booleans, not rewards or missing values")
    return Observation(text, tuple(sorted(set(commands))), won, done)


class AlfworldTextEnvironment:
    """One synchronous native TextWorld PDDL game, with the official wrappers."""

    def __init__(self, game: Game, *, seed: int):
        self.game = game
        self.seed = seed
        self._env = None

    def __enter__(self):
        return self

    def reset(self) -> Observation:
        if self._env is not None:
            raise RuntimeError("Create a fresh ALFWorld environment for each episode")
        if importlib.metadata.version("alfworld") != ALFWORLD_VERSION:
            raise RuntimeError(f"This protocol requires alfworld=={ALFWORLD_VERSION}")
        os.environ["ALFWORLD_DATA"] = str(self.game.data_root)
        import numpy as np
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        random.seed(self.seed)
        np.random.seed(self.seed)
        textworld.g_rng.set_seed(self.seed)
        infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        self._env = textworld.start(
            str(self.game.game_path),
            request_infos=infos,
            wrappers=[partial(AlfredDemangler, shuffle=False), AlfredInfos],
        )
        self._env.seed(self.seed)
        state = self._env.reset()
        if Path(state["extra.gamefile"]).resolve() != self.game.game_path.resolve():
            raise RuntimeError("TextWorld loaded a different game than the selected task")
        return public_observation(state)

    def step(self, command: str) -> Observation:
        if self._env is None:
            raise RuntimeError("ALFWorld environment has not been reset")
        state, _reward, done = self._env.step(command)
        return public_observation(state, done=done)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def environment_smoke(data_root: Path, seed: int = 233) -> dict:
    """Validate actual native stepping; these scripted probes are not accuracy."""
    seen, seen_report = discover_games(data_root, "eval_in_distribution")
    unseen, unseen_report = discover_games(data_root, "eval_out_of_distribution")
    validate_release_inventory({seen_report["split"]: seen_report, unseen_report["split"]: unseen_report})
    verify_split_separation(seen, unseen)
    games = select_games(seen, seed=seed, samples=2)
    probes = []
    for game in games:
        game.verify_inputs()
        with AlfworldTextEnvironment(game, seed=seed) as env:
            initial = env.reset()
            invalid_command = "__alfworld_invalid_action_probe__"
            if initial.done or initial.won or invalid_command in initial.admissible_commands:
                raise RuntimeError(f"Unexpected initial state for environment probe: {game.game_id}")
            rejected = env.step(invalid_command)
            if (rejected.won != initial.won or rejected.done or
                    rejected.admissible_commands != initial.admissible_commands):
                raise RuntimeError("Invalid action unexpectedly changed the native game state")
            if not rejected.admissible_commands:
                raise RuntimeError("Native nonterminal game provided no admissible actions")
            diagnostic_command = rejected.admissible_commands[0]
            stepped = env.step(diagnostic_command)
            probes.append({
                "game": game.manifest(),
                "initial_observation": initial.text,
                "initial_admissible_commands": list(initial.admissible_commands),
                "raw_invalid_command": invalid_command,
                "invalid_feedback": rejected.text,
                "scripted_diagnostic_command": diagnostic_command,
                "diagnostic_feedback": stepped.text,
                "diagnostic_won": stepped.won,
            })
    return {
        "validation_only": True, "accuracy_evidence": False,
        "shared_gpu_latency_not_valid": True, "seed": seed,
        "alfworld": importlib.metadata.version("alfworld"),
        "textworld": importlib.metadata.version("textworld"),
        "inventory_counts": {r["split"]: r["eligible_count"] for r in (seen_report, unseen_report)},
        "selection_protocol": SELECTION_PROTOCOL,
        "probes": probes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[233], default=233)
    args = parser.parse_args()
    print(json.dumps(environment_smoke(args.data_root, args.seed), indent=2, ensure_ascii=False))
