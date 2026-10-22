"""State-integrity checks for explicit cross-host WebBench recovery."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import webbench_migrate_resume as migration


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class MigrationStateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.job = dict(name='webwalkerqa/test/fullkv/r100/seed233/main',
                        benchmark='webwalkerqa', seed=233, purpose='main')
        self.tasks = [dict(id=str(i), task_sha256=f'task-{i}', question=f'q-{i}', answer=f'a-{i}')
                      for i in range(2)]
        self.plan = dict(output=str(self.root), sha256='plan',
                         tasks={'webwalkerqa': {'233': {'main': self.tasks}}})
        self.directory = self.root / 'jobs' / self.job['name']
        audit = dict(plan_sha256='plan', job=self.job, task_sha256='task-0')
        write(self.directory / 'audits/0000.json', audit)
        row = dict(id='0', task_sha256='task-0', question='q-0', gold_answer='a-0',
                   audit='audits/0000.json', audit_sha256=migration.file_hash(self.directory / 'audits/0000.json'))
        write(self.directory / 'checkpoint.json', dict(plan_sha256='plan', job=self.job, results=[row]))
        write(self.directory / 'status.json', dict(status='running'))
        for i in range(2):
            write(self.directory / 'attempts' / f'{i:04d}.json',
                  dict(plan_sha256='plan', job=self.job, task_sha256=f'task-{i}'))

    def tearDown(self):
        self.temporary.cleanup()

    def snapshot_hashes(self):
        return {str(p.relative_to(self.root)): migration.file_hash(p)
                for p in self.root.rglob('*') if p.is_file()}

    def test_verification_preserves_interrupted_attempt_and_committed_data(self):
        before = self.snapshot_hashes()
        record = migration.Verifier().job(self.plan, self.job)
        self.assertEqual(record['committed_samples'], 1)
        self.assertEqual(record['pending_attempt'], str(self.directory / 'attempts/0001.json'))
        self.assertEqual(before, self.snapshot_hashes())

    def test_rejects_orphan_audit_in_nonatomic_live_copy(self):
        write(self.directory / 'audits/0001.json', {})
        with self.assertRaisesRegex(RuntimeError, 'Orphan/missing audit'):
            migration.Verifier().job(self.plan, self.job)

    def test_rejects_corrupted_committed_audit(self):
        write(self.directory / 'audits/0000.json', {'changed': True})
        with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
            migration.Verifier().job(self.plan, self.job)

    def test_rejects_pending_attempt_bound_to_other_task(self):
        write(self.directory / 'attempts/0001.json', dict(plan_sha256='plan', job=self.job, task_sha256='wrong'))
        with self.assertRaisesRegex(RuntimeError, 'Attempt binding differs'):
            migration.Verifier().job(self.plan, self.job)

    def test_does_not_silently_clear_recorded_actor_failure(self):
        write(self.directory / 'failure.json', dict(error='native crash'))
        with self.assertRaisesRegex(RuntimeError, 'Recorded actor failure'):
            migration.Verifier().job(self.plan, self.job)
        self.assertTrue((self.directory / 'failure.json').exists())

    def test_frozen_scheduler_permanent_score_failure_cannot_be_bypassed(self):
        failure = self.root / 'scheduler/seed_order_v4/score_validation_failures' / self.job['name'] / 'failure.json'
        write(failure, {'error': 'invalid scoring contract'})
        with patch.object(migration, 'OWNER', self.root):
            with self.assertRaisesRegex(RuntimeError, 'scheduler score validation failure'):
                migration.Verifier().scheduler()
        self.assertTrue(failure.exists())

    def scoring_fixture(self, benchmark, count=1, returned=False):
        rows = [dict(id='task', source_row_sha256='source', predicted_answer='answer')]
        predictions = self.directory / 'predictions.json'
        write(predictions, {'raw': 'unchanged'})
        contract = dict(predictions_path=str(predictions), predictions_sha256=migration.file_hash(predictions),
                        rows_sha256=migration.digest(rows), ids=['task'], benchmark=benchmark,
                        judge_attempt_budget=1 if benchmark == 'browsecomp_plus' else 10)
        scoring = self.directory / 'scoring'
        write(scoring / 'contract.json', contract)
        for i in range(count):
            write(scoring / 'calls' / f'0000_{i}.json',
                  dict(id='task', input_sha256=migration.digest(rows[0]),
                       contract_sha256=migration.digest(contract),
                       status='returned' if returned and i == 0 else 'requested'))
        module = SimpleNamespace(load_predictions=lambda path: (path, rows))
        return module

    def test_webwalker_requested_calls_keep_their_attempt_budget(self):
        module = self.scoring_fixture('webwalkerqa', count=9)
        before = self.snapshot_hashes()
        migration.scoring_prefix(module, self.directory)
        self.assertEqual(before, self.snapshot_hashes())
        self.assertEqual(len(list((self.directory / 'scoring/calls').glob('*.json'))), 9)

    def test_exhausted_local_judge_call_is_not_retried(self):
        module = self.scoring_fixture('browsecomp_plus')
        with self.assertRaisesRegex(RuntimeError, 'exhausted its declared budget'):
            migration.scoring_prefix(module, self.directory)

    def test_returned_judgement_can_be_reused_at_budget_limit(self):
        module = self.scoring_fixture('browsecomp_plus', returned=True)
        before = self.snapshot_hashes()
        migration.scoring_prefix(module, self.directory)
        self.assertEqual(before, self.snapshot_hashes())

    def test_handoff_cannot_resume_on_source_boot(self):
        path = self.root / 'handoff.json'
        value = dict(schema='webbench_migration_handoff_v1', source_writers_stopped=True,
                     route_plan_sha256=migration.EXPECTED_PLANS, captured_at_utc='2026-10-22T00:00:00Z',
                     source_host={'hostname': 'source', 'boot_id': 'same-boot'})
        write(path, value)
        with patch.object(migration, 'host_identity', return_value={'boot_id': 'same-boot'}):
            with self.assertRaisesRegex(RuntimeError, 'source boot'):
                migration.validate_handoff(path)
        value['source_writers_stopped'] = False
        write(path, value)
        with self.assertRaisesRegex(RuntimeError, 'not finalized'):
            migration.validate_handoff(path)

    def test_explicit_start_archives_identities_and_only_uncommitted_actor_attempt(self):
        owner, webwalker = self.root / 'owner', self.root / 'webwalker'
        for output in (owner, webwalker):
            output.mkdir()
            (output / 'source_snapshot').mkdir()
            write(output / 'score_queue_contract.json', {'endpoint': 'http://127.0.0.1:8313/v1'})
            write(output / 'queue_status.json', {'source_identity': 'old-host'})
        write(self.directory / 'launch.json', {'source_identity': 'old-host'})
        write(self.directory / 'scoring/queue_launch.json', {'source_identity': 'old-host'})
        write(self.directory / 'scoring/calls/0000_1.json', {'status': 'requested'})
        write(self.directory / 'scoring/items/0000.json', {'immutable': 'result'})
        handoff = self.root / 'handoff.json'
        write(handoff, {'source': 'original'})
        record = migration.Verifier().job(self.plan, self.job)
        report = {'jobs': [record], 'errors': [], 'skip_model_weights': False, 'gpus': []}
        before = self.snapshot_hashes()
        child = SimpleNamespace(pid=999, returncode=None, poll=lambda: None)
        with patch.object(migration, 'BASE', self.root), \
             patch.object(migration, 'OWNER', owner), \
             patch.object(migration, 'WEBWALKER', webwalker), \
             patch.object(migration, 'ROUTES', {'browsecomp_plus': owner, 'webwalkerqa': webwalker}), \
             patch.object(migration, 'host_identity', return_value={'boot_id': 'target-boot'}), \
             patch.object(migration, 'validate_handoff', return_value={'source': 'original'}), \
             patch.object(migration, 'require_endpoint'), \
             patch.object(migration, 'ensure_no_local_workers'), \
             patch.object(migration, 'process_identity', return_value={'pid': 999, 'start_ticks': '1'}), \
             patch.object(migration.subprocess, 'Popen', return_value=child), \
             patch.object(migration.time, 'sleep'):
            result = migration.resume(report, handoff)
        archive = owner / 'migration/host_target-boot'
        self.assertEqual(migration.read(result['migration_manifest'])['state'], 'controllers_started')
        for relative in ['checkpoint.json', 'audits/0000.json', 'attempts/0000.json',
                         'scoring/items/0000.json', 'scoring/calls/0000_1.json']:
            path = self.directory / relative
            self.assertEqual(migration.file_hash(path), before[str(path.relative_to(self.root))])
        for relative in ['launch.json', 'attempts/0001.json', 'scoring/queue_launch.json']:
            path = self.directory / relative
            self.assertFalse(path.exists())
            self.assertEqual(migration.file_hash(archive / 'moved_originals' / path.relative_to(self.root)),
                             before[str(path.relative_to(self.root))])
        self.assertFalse((owner / 'queue_status.json').exists())


if __name__ == '__main__':
    unittest.main()
