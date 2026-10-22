import json
from pathlib import Path
import zipfile

import pytest

from scripts.prepare_alfworld_rebuttal_assets import extract_evaluation_files
from scripts.alfworld_rebuttal_env import Game, file_sha256
from scripts.run_alfworld_rebuttal import CheckpointStore, audit_metrics, cache_summary
from scripts import run_alfworld_rebuttal_queue as queue
from scripts.run_alfworld_rebuttal_queue import enough_memory, jobs, summarize, validate_result


def put(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def completed_run(directory, job, *, pruning=True, state="complete", successful_first=True):
    data_root = directory / "data"
    games = []
    for key in ("a", "b"):
        game_path = data_root / key / "game.tw-pddl"
        trajectory_path = game_path.with_name("traj_data.json")
        put(game_path, {"solvable": True})
        put(trajectory_path, {"task_type": "pick_and_place_simple"})
        games.append(Game(key, job["split"], data_root, game_path, trajectory_path,
                          "pick_and_place_simple",
                          {p.relative_to(data_root).as_posix(): file_sha256(p)
                           for p in (game_path, trajectory_path)}))
    # Input assets are external to a worker output in actual runs.
    output = directory / "output"
    protocol = {
        **job, "seed": 233, "data_root": str(data_root),
        "limits": {"max_steps": job["max_steps"]},
        "selected_games": [game.manifest() for game in games],
    }
    store = CheckpointStore(output, protocol)
    store.bind_policy({"implementation": "test"})
    rows = []
    for index, game in enumerate(games):
        success = successful_first and index == 0
        trajectory = [
            {
                "step": step, "raw_action": "look", "action": "look", "valid": True,
                "invalid_reason": None, "admissible_commands": ["look"],
                "environment_won": success and step == 2, "environment_done": step == 2,
                "next_admissible_commands": ["look"],
            }
            for step in (1, 2)
        ]
        row = {
            "episode_index": index, "game": game.manifest(), "episode_seed": 233,
            "status": "completed", "success": success, "steps": 2, "invalid_actions": 0,
            "initial_admissible_commands": ["look"], "initial_won": False,
            "initial_done": False, "trajectory": trajectory,
            "termination": "won" if success else "environment_terminal",
            "audit": {"prune_event_count": int(pruning),
                      "nonprompt_logical_tokens": 10,
                      "nonprompt_resident_tokens": 5 if pruning else 10,
                      "resident_tokens": 15 if pruning else 20,
                      "protected_resident_tokens": 10,
                      "full_attention_kv_bytes": 150 if pruning else 200,
                      "peak_kv_bytes": 200,
                      "boundaries": []},
        }
        store.save(row, games)
        rows.append(row)
    put(output / "status.json", {"status": state})
    put(output / "results.json", {"run_sha256": store.manifest["run_sha256"], "episodes": rows})
    return output, rows


def test_queue_gates_all_five_main_conditions_with_separate_smokes():
    plan = jobs()
    assert len(plan) == 10
    assert all(job["purpose"] == "smoke" for job in plan[:5])
    assert all(job["purpose"] == "main" for job in plan[5:])
    assert {job["split"] for job in plan[:5]}.isdisjoint({job["split"] for job in plan[5:]})
    assert [job["method"] for job in plan[:5]] == [job["method"] for job in plan[5:]]
    assert all(job["samples"] == 0 for job in plan[5:])


def test_free_memory_threshold_applies_to_every_selected_gpu():
    assert enough_memory([{"free_mib": 48 * 1024}, {"free_mib": 50 * 1024}], 48)
    assert not enough_memory([{"free_mib": 48 * 1024}, {"free_mib": 47 * 1024}], 48)
    assert not enough_memory([], 48)


def test_validation_requires_ids_continuation_and_eviction_not_accuracy(tmp_path):
    job = jobs()[1]
    output, _ = completed_run(tmp_path / "valid", job, successful_first=False)
    assert validate_result(output, job) == 2
    result = json.loads((output / "results.json").read_text())
    result["episodes"][1]["game"]["game_id"] = "a"
    put(output / "results.json", result)
    with pytest.raises(RuntimeError, match="IDs"):
        validate_result(output, job)
    output, _ = completed_run(tmp_path / "no_pruning", job, pruning=False)
    with pytest.raises(RuntimeError, match="pruning"):
        validate_result(output, job)


def test_extract_only_official_evaluation_splits_and_reject_traversal(tmp_path):
    archive = tmp_path / "games.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("json_2.1.1/train/a/traj_data.json", "{}")
        output.writestr("json_2.1.1/valid_unseen/b/game.tw-pddl", "game")
    destination = tmp_path / "data"
    assert extract_evaluation_files(archive, destination) == 1
    assert not (destination / "json_2.1.1/train").exists()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../../escape", "invalid")
    with pytest.raises(ValueError, match="escapes"):
        extract_evaluation_files(archive, destination)


def test_summary_excludes_smoke_and_incomplete_conditions(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "RELEASE_GAME_COUNTS", {"eval_out_of_distribution": 2})
    plan = {"protocol": "test", "model": "test", "jobs": jobs(), "latency_policy": "diagnostic"}
    for index, job in enumerate((plan["jobs"][0], plan["jobs"][5], plan["jobs"][6])):
        state = "running" if job == plan["jobs"][6] else "complete"
        output, _ = completed_run(tmp_path / f"fixture{index}", job, state=state)
        target = tmp_path / job["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(output, target_is_directory=True)
    summarize(tmp_path, plan)
    summary = json.loads((tmp_path / "SUMMARY.json").read_text())
    assert summary["completed_main"] == 1
    assert summary["results"][0]["success_rate_percent"] == 50
    assert len(summary["results"]) == 1


def test_main_validation_rejects_a_completed_but_incomplete_task_set(tmp_path):
    job = jobs()[5]
    output, _ = completed_run(tmp_path, job)
    with pytest.raises(RuntimeError, match="full released split"):
        validate_result(output, job)


def test_kv_summary_uses_equal_episode_boundary_means_and_excludes_token_ids():
    def audit(boundaries):
        return {
            "resident_tokens": 20, "protected_resident_tokens": 10,
            "nonprompt_resident_tokens": 10, "nonprompt_logical_tokens": 50,
            "full_attention_kv_bytes": 200, "peak_kv_bytes": 500,
            "pending_generated_token_ids": [123, 456],
            "boundaries": [
                {"after_tokens": count, "after_kv_bytes": count * 10, "protected_tokens": 10}
                for count in boundaries
            ],
        }
    rows = [{"audit": audit([10, 20])}, {"audit": audit([30])}]
    result = cache_summary(rows)
    assert result["mean"]["episode_mean_boundary_total_kv_tokens"] == 22.5
    assert result["mean"]["final_prompt_kv_mib"] == 100 / 2**20
    assert result["mean"]["final_trajectory_kv_mib"] == 100 / 2**20
    assert result["mean"]["final_trajectory_retention"] == .2
    assert all("token_ids" not in key for key in audit_metrics(rows))


def test_policy_fingerprint_compares_json_normalized_gpu_keys(tmp_path):
    store = CheckpointStore(tmp_path, {"protocol": "test"})
    fingerprint = {"loading_plan": {"max_memory": {0: 40, 1: 40}, "allocator_limits_gib": {0: 44, 1: 44}}}
    store.bind_policy(fingerprint)
    store.assert_policy_unchanged(fingerprint)
    reloaded = CheckpointStore(tmp_path, {"protocol": "test"})
    reloaded.assert_policy_unchanged(fingerprint)
    fingerprint["loading_plan"]["max_memory"][0] = 41
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        reloaded.assert_policy_unchanged(fingerprint)
