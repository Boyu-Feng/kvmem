"""Admission and durable-completion checks for the cross-host resource adapter."""
import json
import contextlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import webbench_migrate_host as host
from scripts import webbench_migrate_resume as migration


class HostResourcesTest(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((Path(__file__).resolve().parents[1] /
            'config/environments/webbench_migration_a100_30gib.json').read_text())
        self.inventory = {0: {'free_gib': 30}, 1: {'free_gib': 28}, 2: {'free_gib': 27.9}}
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_one_actor_or_local_judge_per_gpu_and_shared_free_memory(self):
        self.assertEqual(host.select_gpu(self.profile, self.inventory, {}), 0)
        active = {'actor:a': {'gpu': 0}, 'score:api': {'gpu': None}}
        self.assertEqual(host.select_gpu(self.profile, self.inventory, active), 1)
        active['score:local'] = {'gpu': 1}
        self.assertIsNone(host.select_gpu(self.profile, self.inventory, active))
        self.inventory[2]['free_gib'] = 28
        self.assertEqual(host.select_gpu(self.profile, self.inventory, active), 2)

    def test_profiles_cannot_change_precision_parallelism_or_endpoint(self):
        for key, value in [('dtype', 'float16'), ('tensor_parallel_size', 2), ('workers_per_gpu', 2),
                           ('endpoint', 'http://other/v1'), ('actor_allocator_gib', 48),
                           ('gpu_budget_gib', float('nan')), ('devices', [0, 0])]:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                host.validate_profile({**self.profile, key: value})

    def test_waits_for_live_wrapper_to_commit_final_evidence(self):
        record = self.root / 'host_execution.json'
        launch = {'identity': {'pid': 999, 'start_ticks': '123'}}
        alive = lambda pid: launch['identity']
        self.assertTrue(host.waiting_for_host_record(record, launch, alive))
        for state in ('started', 'preflight', 'engine_starting', 'scoring'):
            record.write_text(json.dumps({'state': state}))
            self.assertTrue(host.waiting_for_host_record(record, launch, alive))
            self.assertFalse(host.waiting_for_host_record(record, launch, lambda pid: None))
        for state in ('completed', 'failed'):
            record.write_text(json.dumps({'state': state}))
            self.assertFalse(host.waiting_for_host_record(record, launch, alive))
        self.assertFalse(host.waiting_for_host_record(record, None, alive))

    def test_completed_actor_cannot_lose_host_execution_evidence(self):
        with self.assertRaisesRegex(RuntimeError, 'execution record missing'):
            host.validate_actor_host_record({}, {}, self.root, {}, required=True)
        host.validate_actor_host_record({}, {}, self.root, {}, required=False)

    def test_gate_never_executes_unadmitted_worker(self):
        record = self.root / 'launch.json'
        record.write_text(json.dumps({'token': 'correct', 'identity': {'pid': -1}, 'command': ['worker']}))
        args = SimpleNamespace(record=record, token='correct', child_command=['--', 'worker'])
        with patch.object(host.os, 'execvpe') as execute:
            with self.assertRaisesRegex(RuntimeError, 'identity differs'):
                host.gate(args)
        execute.assert_not_called()

    def test_hardware_profile_uses_available_memory_without_frozen_gpu_remap(self):
        result = SimpleNamespace(stdout='0, GPU-0, NVIDIA A100, 81920, 30720, 595\n'
                                        '1, GPU-1, NVIDIA A100, 81920, 30720, 595\n'
                                        '2, GPU-2, NVIDIA A100, 81920, 30720, 595\n')
        with patch.object(migration.subprocess, 'run', return_value=result), \
             patch.object(migration, 'host_adapter', return_value=host), \
             patch.object(host, 'available_host_ram_gib', return_value=96):
            verifier = migration.Verifier(host_profile=self.profile)
            verifier.hardware()
            self.assertEqual(len(verifier.gpus), 3)
            self.assertEqual(self.profile['devices'], [0, 1, 2])
        with patch.object(migration.subprocess, 'run', return_value=result), \
             patch.object(migration, 'host_adapter', return_value=host), \
             patch.object(host, 'available_host_ram_gib', return_value=95):
            with self.assertRaisesRegex(RuntimeError, 'available RAM'):
                migration.Verifier(host_profile=self.profile).hardware()

    def start_fixture(self, probe_exit):
        owner, webwalker = self.root / 'owner', self.root / 'webwalker'
        for output in (owner, webwalker):
            (output / 'source_snapshot').mkdir(parents=True)
            (output / 'score_queue_contract.json').write_text(json.dumps({'endpoint': self.profile['endpoint']}))
        handoff = self.root / 'handoff.json'
        handoff.write_text('{}')
        def freeze(profile, path):
            value = {**profile, 'sha256': 'contract'}
            migration.atomic_json(path, value)
            return value
        adapter = SimpleNamespace(freeze_contract=freeze)
        child = SimpleNamespace(pid=999, returncode=None, poll=lambda: None, wait=lambda: probe_exit)
        report = dict(jobs=[], errors=[], skip_model_weights=False, gpus=[], host_profile=self.profile)
        with contextlib.ExitStack() as stack:
            for key, value in dict(BASE=self.root, OWNER=owner, WEBWALKER=webwalker,
                    ROUTES={'browsecomp_plus': owner, 'webwalkerqa': webwalker}).items():
                stack.enter_context(patch.object(migration, key, value))
            stack.enter_context(patch.object(migration, 'host_identity', return_value={'boot_id': 'target'}))
            stack.enter_context(patch.object(migration, 'validate_handoff', return_value={}))
            stack.enter_context(patch.object(migration, 'host_adapter', return_value=adapter))
            stack.enter_context(patch.object(migration, 'require_endpoint'))
            stack.enter_context(patch.object(migration, 'ensure_no_local_workers'))
            stack.enter_context(patch.object(migration, 'process_identity', return_value={'pid': 999, 'start_ticks': '1'}))
            stack.enter_context(patch.object(migration.time, 'sleep'))
            popen = stack.enter_context(patch.object(migration.subprocess, 'Popen', return_value=child))
            if probe_exit:
                with self.assertRaisesRegex(RuntimeError, 'no benchmark worker started'):
                    migration.resume(report, handoff)
            else:
                migration.resume(report, handoff)
        return [call.args[0] for call in popen.call_args_list], migration.read(owner / 'migration/host_target/manifest.json')

    def test_all_gpu_capacity_probes_precede_single_host_coordinator(self):
        commands, manifest = self.start_fixture(0)
        self.assertEqual(len(commands), 4)
        for index, command in enumerate(commands[:3]):
            self.assertIn('probe', command)
            self.assertEqual(command[command.index('--gpu') + 1], str(index))
        self.assertIn('queue', commands[-1])
        self.assertEqual(list(manifest['controllers']), ['host_coordinator'])
        self.assertEqual(manifest['state'], 'controllers_started')

    def test_failed_capacity_probe_starts_no_benchmark_coordinator(self):
        commands, manifest = self.start_fixture(1)
        self.assertEqual(len(commands), 1)
        self.assertEqual(manifest['state'], 'judge_probe_failed')
        self.assertNotIn('controllers', manifest)


if __name__ == '__main__':
    unittest.main()
