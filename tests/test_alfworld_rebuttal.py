"""Bounded CPU checks; no downloads, model loads, GPU access, or OS temp files."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from scripts import alfworld_rebuttal_env as native
from scripts import run_alfworld_rebuttal as runner


@contextmanager
def fixture_directory():
    directory = Path("tests") / (".alfworld-check-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    try:
        yield directory.resolve()
    finally:
        shutil.rmtree(directory)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_game(root, split, name, *, solvable=True, task_type="pick_and_place_simple", score=0):
    directory = root / "json_2.1.1" / native.SPLITS[split] / name / "trial"
    write_json(directory / "traj_data.json",
               {"task_type": task_type, "hidden_expert_plan": "NEVER_SEND_EXPERT"})
    write_json(directory / "game.tw-pddl", {
        "solvable": solvable, "fixture_id": name, "score": score,
        "walkthrough": ["NEVER_SEND_WALKTHROUGH"],
    })
    return directory


def populate(root, seen_count=4, unseen_count=3):
    for split, prefix, count in (
        ("eval_in_distribution", "seen", seen_count),
        ("eval_out_of_distribution", "unseen", unseen_count),
    ):
        for index in range(count):
            make_game(root, split, f"{prefix}-{index}", score=index - 100)
    return {split: native.discover_games(root, split)[0] for split in native.SPLITS}


def observation(text="room", *, won=False, done=False, commands=("look",)):
    return native.Observation(text, commands, won, done)


class ScriptedEnvironment:
    def __init__(self, states):
        self.states = list(states)
        self.commands = []
        self.closed = False

    def __enter__(self):
        return self

    def reset(self):
        return self.states.pop(0)

    def step(self, command):
        self.commands.append(command)
        state = self.states.pop(0)
        if isinstance(state, BaseException):
            raise state
        return state

    def __exit__(self, *_):
        self.closed = True


class RecordingPolicy:
    def __init__(self, actions=()):
        self.actions = list(actions)
        self.resets = []
        self.observations = []
        self.action_count = 0

    def reset(self, system_prompt, initial_observation):
        self.resets.append((system_prompt, initial_observation))

    def act(self):
        self.action_count += 1
        return self.actions.pop(0)

    def observe(self, action, text, *, valid):
        self.observations.append((action, text, valid))

    def audit(self):
        return {"peak_cache_bytes": 4096, "generated_tokens": self.action_count,
                "events": [{"before_tokens": 8, "after_tokens": 4}]}

    def fingerprint(self):
        return {"runtime": "cpu-test-double", "decoder": "fixed-test-actions"}


def completed_episode(game, index=0, *, win=True):
    policy = RecordingPolicy(["look"] if win else ["look", "look"])
    states = [observation(), observation(won=win, done=win)]
    if not win:
        states.append(observation())
    env = ScriptedEnvironment(states)
    return runner.evaluate_episode(policy, game, episode_index=index, seed=233,
                                   max_steps=2, env_factory=lambda *a, **k: env)


def test_protocol(selected, inventory=None):
    split = "eval_out_of_distribution"
    return {
        "seed": 233, "purpose": "main", "split": split, "method": "fullkv",
        "ratio": 0.2, "sample_start": 0, "limits": {"max_steps": 2},
        "source_hashes": {"runner": "source-v1"},
        "runtime": {"python": "3.12", "alfworld": "0.4.2"},
        "selected_games": [game.manifest() for game in selected],
        "inventory": inventory or {split: {"eligible_count": len(selected)}},
    }


# This is a fixture helper rather than a pytest test.
test_protocol.__test__ = False


class SelectionTests(unittest.TestCase):
    def test_selection_is_seeded_order_independent_and_split_disjoint(self):
        with fixture_directory() as root:
            catalogs = populate(root)
            seen, unseen = (catalogs[split] for split in native.SPLITS)
            state = random.getstate()
            selected = native.select_games(seen, seed=233, samples=2)
            self.assertEqual(state, random.getstate())
            self.assertEqual(selected, native.select_games(list(reversed(seen)), seed=233, samples=2))
            self.assertEqual(selected, native.select_games(seen, seed=233)[:2])
            self.assertEqual(native.select_games(seen, seed=233, sample_start=2),
                             native.select_games(seen, seed=233)[2:])
            native.verify_split_separation(seen, unseen)
            self.assertFalse({g.game_id for g in selected} & {g.game_id for g in unseen})
            with self.assertRaises(ValueError):
                native.select_games(seen, seed=233, samples=len(seen) + 1)
            with self.assertRaises(ValueError):
                native.select_games(seen, seed=233, sample_start=len(seen))

    def test_official_inventory_gates_are_not_model_score_filtering(self):
        with fixture_directory() as root:
            split = "eval_out_of_distribution"
            make_game(root, split, "high-score", score=999)
            make_game(root, split, "low-score", score=-999)
            make_game(root, split, "unsolvable", solvable=False)
            make_game(root, split, "Sliced-task")
            make_game(root, split, "movable-task")
            make_game(root, split, "unsupported", task_type="not_an_alfworld_task")
            original_only = make_game(root, split, "no-released-text-game")
            (original_only / "game.tw-pddl").unlink()
            games, report = native.discover_games(root, split)
            self.assertEqual({g.game_id for g in games}, {"high-score/trial", "low-score/trial"})
            self.assertEqual(report["eligible_count"], 2)
            self.assertEqual(report["inventory_count"], 7)
            self.assertEqual(report["released_text_game_count"], 6)
            self.assertEqual(len(report["excluded"]), 5)
            self.assertIn("official_no_released_text_game", {x["reason"] for x in report["excluded"]})
            self.assertTrue(all(len(g.input_hashes) == 2 for g in games))

    def test_missing_corrupt_or_modified_inputs_fail_loudly(self):
        with fixture_directory() as root:
            split = "eval_out_of_distribution"
            directory = make_game(root, split, "supported")
            game = native.discover_games(root, split)[0][0]
            write_json(directory / "game.tw-pddl", {"solvable": True, "changed": True})
            with self.assertRaisesRegex(RuntimeError, "input changed"):
                game.verify_inputs()
            (directory / "game.tw-pddl").unlink()
            with self.assertRaisesRegex(RuntimeError, "No supported"):
                native.discover_games(root, split)
            write_json(directory / "game.tw-pddl", {"solvable": "true"})
            with self.assertRaisesRegex(RuntimeError, "solvable flag"):
                native.discover_games(root, split)

    def test_benchmark_rejects_incomplete_release_inventories(self):
        with fixture_directory() as root:
            populate(root)
            inventory = {split: native.discover_games(root, split)[1] for split in native.SPLITS}
            with self.assertRaisesRegex(RuntimeError, "Incomplete/unexpected"):
                native.validate_release_inventory(inventory)
            exact = {split: {"released_text_game_count": n, "eligible_count": n}
                     for split, n in native.RELEASE_GAME_COUNTS.items()}
            native.validate_release_inventory(exact)
            exact["eval_out_of_distribution"]["released_text_game_count"] -= 1
            with self.assertRaisesRegex(RuntimeError, "134"):
                native.validate_release_inventory(exact)

    def test_overlap_is_refused_instead_of_filtered(self):
        with fixture_directory() as root:
            for split in native.SPLITS:
                make_game(root, split, "same-task")
            catalogs = [native.discover_games(root, split)[0] for split in native.SPLITS]
            with self.assertRaisesRegex(RuntimeError, "overlap"):
                native.verify_split_separation(*catalogs)

    def test_cli_enforces_disjoint_bounded_smoke(self):
        parser = runner.build_parser()
        base = ["--model-path", "local-model", "--data-root", "assets",
                "--method", "fullkv", "--output", "output"]
        main = parser.parse_args(base)
        runner.validate_args(main)
        self.assertEqual((main.seed, main.samples, main.max_steps, main.max_new_tokens,
                          main.max_context_tokens), (233, 0, 50, 128, 8192))
        self.assertEqual(main.device, "auto")
        smoke = parser.parse_args(base + ["--purpose", "smoke", "--split",
                                         "eval_in_distribution", "--samples", "2"])
        runner.validate_args(smoke)
        for key, value in (("seed", 1), ("samples", 0), ("sample_start", 1),
                           ("split", "eval_out_of_distribution")):
            changed = copy.copy(smoke)
            setattr(changed, key, value)
            with self.assertRaises(ValueError):
                runner.validate_args(changed)
        main.split = "eval_in_distribution"
        with self.assertRaises(ValueError):
            runner.validate_args(main)

    def test_manifest_records_standard_attention_auto_placement_and_per_gpu_memory(self):
        with fixture_directory() as root:
            catalogs = populate(root / "assets")
            model = root / "local-model"
            write_json(model / "config.json", {"model_type": "qwen2"})
            (model / "model.safetensors").write_bytes(b"fixture-not-a-loaded-model")
            args = runner.build_parser().parse_args([
                "--model-path", str(model), "--data-root", str(root / "assets"),
                "--method", "fullkv", "--output", str(root / "output"),
                "--memory-limit-gib", "30",
            ])
            inventory = {split: native.discover_games(root / "assets", split)[1]
                         for split in native.SPLITS}
            with patch.object(runner, "source_hashes", return_value={"runner": "test-source"}), \
                    patch.object(runner, "runtime_versions", return_value={"transformers": "5.0.0"}):
                protocol = runner.make_protocol(args, inventory, catalogs[args.split])
            self.assertEqual(protocol["placement"]["device"], "auto")
            self.assertEqual(protocol["limits"]["memory_limit_gib"], 30)
            self.assertIn("per visible GPU", protocol["placement"]["memory_limit_scope"])
            self.assertEqual(protocol["runtime"]["transformers"], "5.0.0")
            self.assertIn("Qwen2.5-32B-Instruct", protocol["policy_adaptation"])
            self.assertNotIn("hybrid", protocol["policy_adaptation"])
            self.assertTrue(protocol["shared_gpu_latency_not_valid"])
            write_json(model / "config.json", {"model_type": "unsupported_family"})
            with self.assertRaisesRegex(RuntimeError, "standard-attention Qwen2.5"):
                runner.model_inputs(model)


class EpisodeTests(unittest.TestCase):
    def test_parser_never_repairs_or_extracts_an_oracle_action(self):
        self.assertTrue(runner.parse_action("  look\n", ["look"])["valid"])
        for raw in ("", "Action: look", "```look```", "Look", '"look"',
                    "look\nlook", "think first\nlook", "go to hidden room"):
            parsed = runner.parse_action(raw, ["look"])
            self.assertEqual(parsed["raw_action"], raw)
            self.assertEqual(parsed["action"], raw.strip())
            self.assertFalse(parsed["valid"], raw)
        with self.assertRaises(TypeError):
            runner.parse_action(None, ["look"])

    def test_invalid_action_is_really_stepped_and_terminal_state_never_reaches_policy(self):
        with fixture_directory() as root:
            game = populate(root)["eval_out_of_distribution"][0]
            policy = RecordingPolicy(["Action: look", "look"])
            env = ScriptedEnvironment([
                observation("initial room"), observation("Nothing happens."),
                observation("SECRET_TERMINAL_SUCCESS", won=True, done=True),
            ])
            row = runner.evaluate_episode(policy, game, episode_index=0, seed=233, max_steps=5,
                                          env_factory=lambda *a, **k: env)
            self.assertEqual(env.commands, ["Action: look", "look"])
            self.assertTrue(env.closed)
            self.assertTrue(row["success"])
            self.assertEqual((row["steps"], row["invalid_actions"]), (2, 1))
            self.assertEqual(len(policy.observations), 1)
            self.assertFalse(policy.observations[0][2])
            self.assertEqual(row["audit"]["generated_tokens"], 2)
            policy_inputs = repr(policy.resets + policy.observations)
            for hidden in ("SECRET_TERMINAL_SUCCESS", "NEVER_SEND_EXPERT",
                           "NEVER_SEND_WALKTHROUGH", "'won'", "game.tw-pddl"):
                self.assertNotIn(hidden, policy_inputs)
            self.assertIsNone(row["trajectory"][-1]["policy_observation"])

    def test_observe_gets_submitted_action_while_raw_model_output_is_preserved(self):
        with fixture_directory() as root:
            game = populate(root)["eval_out_of_distribution"][0]
            raw_action = " \nlook \n"
            policy = RecordingPolicy([raw_action, "look"])
            env = ScriptedEnvironment([
                observation(), observation(), observation(won=True, done=True),
            ])
            row = runner.evaluate_episode(policy, game, episode_index=0, seed=233, max_steps=2,
                                          env_factory=lambda *a, **k: env)
            self.assertEqual(env.commands, ["look", "look"])
            self.assertEqual(policy.observations[0][0], "look")
            self.assertTrue(policy.observations[0][2])
            self.assertEqual(row["trajectory"][0]["raw_action"], raw_action)
            self.assertEqual(row["trajectory"][0]["action"], "look")
            self.assertTrue(row["success"])

    def test_observation_allowlist_and_native_boolean_success(self):
        state = {
            "feedback": "You won! (text alone is not a metric)",
            "admissible_commands": ["look"], "won": False,
            "reward": 1, "score": 999, "facts": ["secret"],
            "extra.expert_plan": ["secret"], "extra.walkthrough": ["secret"],
        }
        public = native.public_observation(state)
        self.assertFalse(public.won)
        self.assertNotIn("secret", runner.render_observation(public))
        for bad in (None, 1, [False]):
            with self.assertRaises(TypeError):
                native.public_observation({**state, "won": bad})
        with fixture_directory() as root:
            game = populate(root)["eval_out_of_distribution"][0]
            policy = RecordingPolicy(["look"])
            env = ScriptedEnvironment([observation(), public])
            row = runner.evaluate_episode(policy, game, episode_index=0, seed=233, max_steps=1,
                                          env_factory=lambda *a, **k: env)
            self.assertFalse(row["success"])
            self.assertEqual(row["termination"], "step_limit")
            self.assertEqual(policy.observations, [])

    def test_already_terminal_game_never_queries_or_reuses_policy_audit(self):
        with fixture_directory() as root:
            game = populate(root)["eval_out_of_distribution"][0]
            policy = RecordingPolicy()
            env = ScriptedEnvironment([observation("terminal", won=True, done=True)])
            row = runner.evaluate_episode(policy, game, episode_index=0, seed=233, max_steps=2,
                                          env_factory=lambda *a, **k: env)
            self.assertTrue(row["success"])
            self.assertEqual(row["steps"], 0)
            self.assertEqual(row["audit"], {})
            self.assertEqual(policy.resets, [])

    def test_runtime_failure_preserves_raw_partial_not_a_lost_game(self):
        with fixture_directory() as root:
            game = populate(root)["eval_out_of_distribution"][0]
            policy = RecordingPolicy(["not a valid action"])
            env = ScriptedEnvironment([observation(), RuntimeError("native engine failed")])
            with self.assertRaises(runner.EpisodeRuntimeError) as caught:
                runner.evaluate_episode(policy, game, episode_index=0, seed=233, max_steps=2,
                                        env_factory=lambda *a, **k: env)
            error = caught.exception
            self.assertEqual(error.stage, "environment_step")
            self.assertEqual(error.partial_episode["trajectory"][0]["raw_action"],
                             "not a valid action")
            self.assertNotIn("success", error.partial_episode)
            self.assertTrue(env.closed)
            report = runner.summarize([], test_protocol([game]), [{}], failed=True)
            self.assertEqual(report["completed_count"], 0)
            self.assertIsNone(report["success_rate"])
            self.assertEqual(report["failures"], {
                "unsuccessful_completed_episodes": 0, "runtime_error_attempts": 1})

    def test_audit_is_snapshotted_and_invalid_audit_preserves_serializable_failure(self):
        with fixture_directory() as root:
            game = populate(root)["eval_out_of_distribution"][0]
            shared_audit = {"peak_cache_bytes": 1024, "events": [{"generated_tokens": 1}]}
            policy = RecordingPolicy(["look"])
            env = ScriptedEnvironment([observation(), observation(won=True, done=True)])
            with patch.object(policy, "audit", return_value=shared_audit):
                row = runner.evaluate_episode(policy, game, episode_index=0, seed=233, max_steps=2,
                                              env_factory=lambda *a, **k: env)
            shared_audit["events"].clear()
            self.assertEqual(row["audit"]["events"], [{"generated_tokens": 1}])
            env = ScriptedEnvironment([observation(), observation(won=True, done=True)])
            policy = RecordingPolicy(["look"])
            with patch.object(policy, "audit", return_value={"bytes": object()}):
                with self.assertRaises(runner.EpisodeRuntimeError) as caught:
                    runner.evaluate_episode(policy, game, episode_index=0, seed=233, max_steps=2,
                                            env_factory=lambda *a, **k: env)
            self.assertEqual(caught.exception.stage, "policy_audit")
            runner.json_hash(caught.exception.partial_episode)


class CheckpointTests(unittest.TestCase):
    def test_atomic_checkpoint_roundtrip_and_fingerprint_guards(self):
        with fixture_directory() as root:
            games = populate(root / "assets")["eval_out_of_distribution"]
            protocol = test_protocol(games)
            store = runner.CheckpointStore(root / "output", protocol)
            store.bind_policy(RecordingPolicy().fingerprint())
            row = completed_episode(games[0])
            store.save(row, games)
            self.assertEqual(store.load(games), [row])
            self.assertEqual(list((root / "output").rglob("*.tmp")), [])
            with self.assertRaisesRegex(RuntimeError, "policy fingerprint"):
                store.bind_policy({"runtime": "changed"})
            for key, replacement in (("source_hashes", {"runner": "changed"}),
                                     ("runtime", {"alfworld": "changed"}),
                                     ("seed", 42), ("ratio", 0.5)):
                changed = copy.deepcopy(protocol)
                changed[key] = replacement
                with self.assertRaisesRegex(RuntimeError, "Refusing changed"):
                    runner.CheckpointStore(root / "output", changed)
            changed = copy.deepcopy(protocol)
            changed["selected_games"][0]["input_hashes"]["new-input"] = "changed"
            with self.assertRaises(RuntimeError):
                runner.CheckpointStore(root / "output", changed)

    def test_policy_fingerprint_is_an_immutable_snapshot(self):
        with fixture_directory() as root:
            games = populate(root / "assets")["eval_out_of_distribution"]
            store = runner.CheckpointStore(root / "output", test_protocol(games))
            fingerprint = {"decoder": {"mode": "original"}}
            store.bind_policy(fingerprint)
            fingerprint["decoder"]["mode"] = "changed"
            self.assertEqual(store.manifest["policy_fingerprint"]["decoder"]["mode"], "original")
            with self.assertRaisesRegex(RuntimeError, "policy fingerprint"):
                store.bind_policy(fingerprint)

    def test_corrupted_or_relabelled_episode_is_not_resumed(self):
        with fixture_directory() as root:
            games = populate(root / "assets")["eval_out_of_distribution"]
            store = runner.CheckpointStore(root / "output", test_protocol(games))
            store.bind_policy(RecordingPolicy().fingerprint())
            row = completed_episode(games[0])
            store.save(row, games)
            path = store.checkpoint_path(row)
            checkpoint = runner.read_json(path)
            checkpoint["episode"]["success"] = False
            write_json(path, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "fingerprint/content"):
                store.load(games)
            checkpoint["episode_sha256"] = runner.json_hash(checkpoint["episode"])
            write_json(path, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "success/termination"):
                store.load(games)
            checkpoint["episode"] = row
            checkpoint["episode_sha256"] = runner.json_hash(row)
            checkpoint["run_sha256"] = "another-run"
            write_json(path, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "fingerprint/content"):
                store.load(games)

    def test_orphan_outputs_and_concurrent_writer_are_refused(self):
        with fixture_directory() as root:
            games = populate(root / "assets")["eval_out_of_distribution"]
            output = root / "orphan"
            write_json(output / "results.json", {"episodes": []})
            with self.assertRaisesRegex(RuntimeError, "no matching manifest"):
                runner.CheckpointStore(output, test_protocol(games))
            with runner.output_lock(root / "locked"):
                with self.assertRaisesRegex(RuntimeError, "Another runner"):
                    with runner.output_lock(root / "locked"):
                        self.fail("Second writer acquired the output lock")

    def test_results_cannot_silently_replace_a_missing_checkpoint(self):
        with fixture_directory() as root:
            games = populate(root / "assets")["eval_out_of_distribution"]
            store = runner.CheckpointStore(root / "output", test_protocol(games))
            store.bind_policy(RecordingPolicy().fingerprint())
            row = completed_episode(games[0])
            store.save(row, games)
            runner.persist_reports(store, [row])
            runner.assert_report_matches_checkpoints(store, store.load(games))
            store.checkpoint_path(row).unlink()
            with self.assertRaisesRegex(RuntimeError, "missing/changed episode"):
                runner.assert_report_matches_checkpoints(store, store.load(games))

    def test_runtime_failure_checkpoint_resume_and_completed_fast_path(self):
        with fixture_directory() as root:
            catalogs = populate(root / "assets", unseen_count=2)
            games = native.select_games(catalogs["eval_out_of_distribution"], seed=233)
            first = completed_episode(games[0])
            second = completed_episode(games[1], index=1, win=False)
            protocol = test_protocol(games)
            policy = RecordingPolicy()
            constructor_options = []

            def make_policy(**options):
                constructor_options.append(options)
                return policy

            modules = {
                "torch": SimpleNamespace(manual_seed=lambda _: None),
                "numpy": SimpleNamespace(random=SimpleNamespace(seed=lambda _: None)),
                "models.AgentKVPolicy": SimpleNamespace(AgentKVPolicy=make_policy),
            }
            argv = ["--model-path", str(root / "no-model-loaded"), "--data-root", str(root / "assets"),
                    "--method", "fullkv", "--max-steps", "2", "--memory-limit-gib", "30",
                    "--max-context-tokens", "32768",
                    "--output", str(root / "output")]
            error = runner.EpisodeRuntimeError("policy_act", {
                "episode_index": 1, "game": games[1].manifest(), "trajectory": []})
            with patch.object(runner, "make_protocol", return_value=protocol), \
                    patch.object(runner, "validate_release_inventory"), \
                    patch.object(runner, "assert_unchanged"), \
                    patch.dict(sys.modules, modules), \
                    patch("builtins.print"):
                with patch.object(runner, "evaluate_episode", side_effect=[first, error]):
                    with self.assertRaises(runner.EpisodeRuntimeError):
                        runner.main(argv)
                self.assertEqual(constructor_options[0]["device"], "auto")
                self.assertEqual(constructor_options[0]["memory_limit_gib"], 30)
                self.assertEqual(constructor_options[0]["max_context_tokens"], 32768)
                self.assertEqual(constructor_options[0]["max_new_tokens"], 128)
                summary = runner.read_json(root / "output" / "SUMMARY.json")
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(summary["completed_count"], 1)
                self.assertEqual(summary["failures"]["runtime_error_attempts"], 1)
                self.assertEqual(len(list((root / "output" / "episodes").glob("*.json"))), 1)
                with patch.object(runner, "evaluate_episode", return_value=second) as evaluate:
                    self.assertEqual(runner.main(argv), 0)
                    self.assertEqual(evaluate.call_count, 1)
                    self.assertEqual(evaluate.call_args.kwargs["episode_index"], 1)
                summary = runner.read_json(root / "output" / "SUMMARY.json")
                self.assertTrue(summary["benchmark_complete"])
                self.assertEqual(summary["completed_count"], 2)
                self.assertEqual(summary["success_rate"], 0.5)
                self.assertEqual(summary["failures"]["runtime_error_attempts"], 1)
                modules["models.AgentKVPolicy"].AgentKVPolicy = lambda **_: self.fail("Reloaded finished model")
                with patch.object(runner, "evaluate_episode", side_effect=AssertionError("Replayed episode")):
                    self.assertEqual(runner.main(argv), 0)

    def test_source_change_is_detected_before_another_episode(self):
        with patch.object(runner, "source_hashes", return_value={"runner": "changed"}), \
                patch.object(runner, "model_inputs", side_effect=AssertionError("Too late")):
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                runner.assert_unchanged({"source_hashes": {"runner": "original"}})

    def test_summary_marks_partial_slices_and_reports_byte_token_metrics(self):
        with fixture_directory() as root:
            games = populate(root)["eval_out_of_distribution"]
            protocol = test_protocol(games[:1], {
                "eval_out_of_distribution": {"eligible_count": len(games)}})
            row = completed_episode(games[0])
            summary = runner.summarize([row], protocol, [])
            self.assertFalse(summary["benchmark_complete"])
            self.assertIsNone(summary["full_split_success_rate"])
            self.assertEqual(summary["audit_byte_token_metrics"]["peak_cache_bytes"]["max"], 4096)
            self.assertEqual(summary["audit_byte_token_metrics"]["generated_tokens"]["sum"], 1)
            self.assertTrue(summary["shared_gpu_latency_not_valid"])


class ImportAndNativeTests(unittest.TestCase):
    def test_runtime_refuses_the_superseded_transformers_version(self):
        versions = {"alfworld": "0.4.2", "textworld": "1.7.0", "torch": "2.10.0",
                    "transformers": "5.0.0", "numpy": "1.26.4"}
        with patch.object(runner.importlib.metadata, "version", side_effect=versions.__getitem__), \
                patch.object(runner.importlib.metadata, "distributions", return_value=[]):
            self.assertEqual(runner.runtime_versions()["required"]["transformers"], "5.0.0")
            versions["transformers"] = "5.17.0"
            with self.assertRaisesRegex(RuntimeError, "transformers==5.0.0"):
                runner.runtime_versions()

    def test_import_and_help_do_not_import_ml_or_environment_packages(self):
        code = (
            "import sys; import scripts.run_alfworld_rebuttal as r; "
            "assert not any(n in sys.modules for n in "
            "('torch','transformers','numpy','alfworld','textworld')); "
            "r.build_parser().parse_args(['--help'])"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True,
                                text=True, cwd=runner.ROOT, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--memory-limit-gib", result.stdout)
        self.assertIn("--device", result.stdout)

    @unittest.skipUnless(os.environ.get("ALFWORLD_REBUTTAL_TEST_DATA_ROOT"),
                         "Native CPU integration requires explicit, already-installed ALFWorld assets")
    def test_native_environment_only_smoke(self):
        result = native.environment_smoke(Path(os.environ["ALFWORLD_REBUTTAL_TEST_DATA_ROOT"]))
        self.assertTrue(result["validation_only"])
        self.assertFalse(result["accuracy_evidence"])
        self.assertEqual(len(result["probes"]), 2)
        self.assertEqual(result["alfworld"], "0.4.2")
        self.assertGreater(result["inventory_counts"]["eval_out_of_distribution"], 0)
        for probe in result["probes"]:
            self.assertEqual(probe["invalid_feedback"], "Nothing happens.")


if __name__ == "__main__":
    unittest.main()
