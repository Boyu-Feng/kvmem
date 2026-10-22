"""Host allocation and provenance checks without importing CUDA/vLLM."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/webbench_migrate_judge.py"
SPEC = importlib.util.spec_from_file_location("webbench_migrate_judge", SCRIPT)
judge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(judge)


class HostJudgeTests(unittest.TestCase):
    def setUp(self):
        self.host = {"devices": [6], "gpu_budget_gib": 26, "headroom_gib": 2,
                     "minimum_available_host_ram_gib": 96, "judge_cpu_offload_gb": 48}
        self.gpu = {"index": 6, "uuid": "GPU-fixture", "name": "A100", "total_gib": 80, "free_gib": 30}

    def allocation(self, memory_kib=104857600):
        with mock.patch.object(judge, "gpu_snapshot", return_value=self.gpu), \
             mock.patch.object(Path, "read_text", return_value=f"MemAvailable: {memory_kib} kB\n"):
            return judge.runtime_allocation(self.host, 6)

    def test_80g_card_with_30g_free_uses_26g_budget(self):
        allocation = self.allocation()
        self.assertEqual(allocation["gpu_memory_utilization"], .325)
        self.assertEqual(allocation["cpu_offload_gb"], 48)
        self.assertEqual(allocation["effective_gpu_budget_gib"], 26)

    def test_requires_headroom_and_cpu_memory(self):
        with self.assertRaisesRegex(RuntimeError, "MemAvailable"):
            self.allocation(95 * 1024 * 1024)
        self.gpu["free_gib"] = 27
        with self.assertRaisesRegex(RuntimeError, "headroom"):
            self.allocation()

    def test_effective_args_preserve_model_precision_seed_tp_and_context(self):
        original = judge.original_engine_kwargs("/models/frozen")
        effective = judge.effective_engine_kwargs(original, self.allocation())
        self.assertEqual(original["gpu_memory_utilization"], .65)
        self.assertNotIn("cpu_offload_gb", original)
        for key in original.keys() - {"gpu_memory_utilization"}:
            self.assertEqual(original[key], effective[key])

    def test_engine_configuration_mismatch_is_rejected(self):
        allocation = self.allocation()
        config = SimpleNamespace(
            model_config=SimpleNamespace(dtype="torch.bfloat16", seed=0, max_model_len=16384),
            parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
            cache_config=SimpleNamespace(gpu_memory_utilization=.325, cpu_offload_gb=48))
        engine = SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=config))
        self.assertEqual(judge.verify_engine(engine, allocation)["dtype"], "torch.bfloat16")
        config.parallel_config.tensor_parallel_size = 4
        with self.assertRaisesRegex(RuntimeError, "protocol"):
            judge.verify_engine(engine, allocation)

    def test_original_completed_scores_do_not_require_a_new_host_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = Path(tmp) / "plan.json"
            plan.write_text(json.dumps({"output": tmp}))
            self.assertIsNone(judge.validate_host_record(plan, "old/job", Path(tmp) / "absent-contract.json"))

    def test_context_rejects_changed_host_bound_wrapper_before_runtime_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "plan.json"
            host_path = Path(tmp) / "host.json"
            judge.write(plan_path, {"output": tmp})
            host = dict(schema="webbench_migration_host_resources_v1",
                        target_host=judge.host_identity(), source_hashes={str(SCRIPT): "changed"})
            judge.write(host_path, host)
            with self.assertRaisesRegex(RuntimeError, "wrapper differs"):
                judge.load_context(plan_path, host_path)
            host["source_hashes"][str(SCRIPT)] = judge.file_hash(SCRIPT)
            host["target_host"]["boot_id"] = "different-boot"
            judge.write(host_path, host)
            with self.assertRaisesRegex(RuntimeError, "another host or boot"):
                judge.load_context(plan_path, host_path)

    def test_real_frozen_host_contract_with_extended_identity_loads_judge_context(self):
        host_spec = importlib.util.spec_from_file_location("judge_test_host_adapter", SCRIPT.with_name("webbench_migrate_host.py"))
        host_adapter = importlib.util.module_from_spec(host_spec)
        host_spec.loader.exec_module(host_adapter)
        profile = judge.read(SCRIPT.parents[1] / "config/environments/webbench_migration_a100_30gib.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source_snapshot/scripts/webbench_scoring_v1.py"
            source.parent.mkdir(parents=True)
            source.write_text("# frozen scorer fixture\n")
            runtime = root / "runtime_ready.json"
            runtime.write_text("{}\n")
            assets = root / "assets"
            assets.mkdir()
            (assets / "model_ready.json").write_text("{}\n")
            queue = {"assets": str(assets), "model_ready_sha256": judge.file_hash(assets / "model_ready.json"),
                     "judge_runtime_manifest": {"path": str(runtime), "sha256": judge.file_hash(runtime)}}
            (root / "score_queue_contract.json").write_text(json.dumps(queue))
            plan_path = root / "plan.json"
            judge.write(plan_path, {"output": str(root), "source_hashes": {"scripts/webbench_scoring_v1.py": judge.file_hash(source)}})
            expected_plan = judge.read(plan_path)
            contract_path = root / "host_resources.json"
            with mock.patch.object(host_adapter.m, "EXPECTED_PLANS", {"browsecomp_plus": expected_plan["sha256"]}):
                contract = host_adapter.freeze_contract(profile, contract_path)
            self.assertTrue({"machine", "python", "platform"}.issubset(contract["target_host"]))
            plan, host, loaded_queue, loaded_source = judge.load_context(plan_path, contract_path)
            self.assertEqual(plan, expected_plan)
            self.assertEqual(host, contract)
            self.assertEqual(loaded_queue, queue)
            self.assertEqual(loaded_source, source)

    def test_probe_rejects_changed_host_or_gpu_before_formal_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "scorer.py"
            source.write_text("# frozen\n")
            host_path = Path(tmp) / "host.json"
            host = dict(self.host, sha256="host-hash")
            plan = {"sha256": "plan-hash"}
            record = judge.base_record(plan, host, source, "synthetic_probe")
            record.update(state="completed", destination_runtime_tested=True, allocation=self.allocation(),
                          engine_config=dict(dtype="torch.bfloat16", tensor_parallel_size=1,
                                             pipeline_parallel_size=1, engine_seed=0, max_model_len=16384,
                                             gpu_memory_utilization=.325, cpu_offload_gb=48))
            path = judge.probe_record_path(host_path, 6)
            judge.write(path, record)
            with mock.patch.object(judge, "gpu_snapshot", return_value=self.gpu):
                judge.validate_probe(plan, host, source, host_path, 6)
            with mock.patch.object(judge, "host_identity", return_value={"hostname": "another", "boot_id": "new"}):
                with self.assertRaisesRegex(RuntimeError, "Successful judge probe"):
                    judge.validate_probe(plan, host, source, host_path, 6)
            with mock.patch.object(judge, "gpu_snapshot", return_value={"uuid": "different-gpu"}):
                with self.assertRaisesRegex(RuntimeError, "GPU identity"):
                    judge.validate_probe(plan, host, source, host_path, 6)


if __name__ == "__main__":
    unittest.main()
