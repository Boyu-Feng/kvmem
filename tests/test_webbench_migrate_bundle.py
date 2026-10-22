"""Real age/zstd round trips and non-destructive migration restore checks."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import webbench_migrate_bundle as bundle


PREFIX = 'mnt/checkpoints/bruli/fengboyu/synthetic_migration_test'
MANIFEST = PREFIX + '/payload/files.json'
DATA = PREFIX + '/checkpoint.json'
LATER = PREFIX + '/worker.py'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class EncryptedMigrationBundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates = [os.environ.get('KVMEM_TEST_AGE'), shutil.which('age'),
                      '/mnt/checkpoints/bruli/fengboyu/kvmem_migration_20261022/tools/age/age']
        cls.age = next((str(Path(p)) for p in candidates if p and Path(p).is_file()), None)
        cls.zstd = shutil.which('zstd')
        keygen = str(Path(cls.age).with_name('age-keygen')) if cls.age else None
        if not (cls.age and cls.zstd and Path(keygen).is_file()):
            raise unittest.SkipTest('Real age, age-keygen and zstd executables are required')
        cls.key_temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.key_temp.cleanup)
        cls.identity = Path(cls.key_temp.name) / 'test-only.agekey'
        subprocess.run([keygen, '-o', str(cls.identity)], capture_output=True, check=True)
        cls.recipient = subprocess.run([keygen, '-y', str(cls.identity)], capture_output=True,
                                       text=True, check=True).stdout.strip()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.release = self.root / 'release'
        self.release.mkdir()
        self.destination = self.root / 'destination'
        self.destination.mkdir()

    def archive(self, *, members=None, inventory_transform=None, part_size=97):
        if members is None:
            members = [(DATA, b'{"committed_samples": 17}\n', 0o600),
                       (LATER, b'#!/usr/bin/python3\nprint("test only")\n', 0o755),
                       (PREFIX + '/current', 'checkpoint.json', None)]
        contents = io.BytesIO()
        inventory = []
        with tarfile.open(fileobj=contents, mode='w') as archive:
            for name, value, mode in members:
                item = tarfile.TarInfo(name)
                if mode is None:
                    item.type = tarfile.SYMTYPE
                    item.linkname = value
                    archive.addfile(item)
                    inventory.append(dict(path=name, symlink=value))
                else:
                    item.size, item.mode = len(value), mode
                    archive.addfile(item, io.BytesIO(value))
                    inventory.append(dict(path=name, bytes=len(value), sha256=sha256(value), mode=mode))
            if inventory_transform:
                inventory = inventory_transform(inventory)
            raw = json.dumps(dict(schema='kvmem_migration_files_v1', group='test',
                                  files=inventory)).encode()
            item = tarfile.TarInfo(MANIFEST)
            item.size, item.mode = len(raw), 0o644
            archive.addfile(item, io.BytesIO(raw))
        compressed = subprocess.run([self.zstd, '-q', '-c'], input=contents.getvalue(),
                                    capture_output=True, check=True).stdout
        encrypted = subprocess.run([self.age, '-r', self.recipient], input=compressed,
                                   capture_output=True, check=True).stdout
        parts = []
        for index, offset in enumerate(range(0, len(encrypted), part_size)):
            value = encrypted[offset:offset + part_size]
            name = f'test.tar.zst.age.part{index:03d}'
            (self.release / name).write_bytes(value)
            parts.append(dict(name=name, bytes=len(value), sha256=sha256(value)))
        return dict(group='test', parts=parts, inner_manifest=MANIFEST,
                    inner_manifest_sha256=sha256(raw)), members, raw

    def consume(self, group, *, inspect=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return bundle.consume_archive(group, self.release, self.identity,
                                          None if inspect else self.destination,
                                          self.age, self.zstd)

    def test_real_multipart_inspect_restore_and_idempotence(self):
        group, members, manifest = self.archive()
        self.assertGreater(len(group['parts']), 1)
        inspected = self.consume(group, inspect=True)
        self.assertEqual(inspected['verified_files'], len(members))
        self.assertEqual(list(self.destination.iterdir()), [])
        restored = self.consume(group)
        self.assertEqual(restored['verified_files'], len(members))
        for name, value, mode in members:
            path = self.destination / name
            if mode is None:
                self.assertTrue(path.is_symlink())
                self.assertEqual(os.readlink(path), value)
            else:
                self.assertEqual(path.read_bytes(), value)
                self.assertEqual(path.stat().st_mode & 0o777, mode)
        self.assertEqual((self.destination / MANIFEST).read_bytes(), manifest)
        before = {p: p.lstat().st_mtime_ns for p in self.destination.rglob('*')}
        self.consume(group)
        self.assertEqual(before, {p: p.lstat().st_mtime_ns for p in self.destination.rglob('*')})

    def test_conflicting_destination_preserved_before_any_files_installed(self):
        group, _, _ = self.archive()
        conflicting = self.destination / LATER
        conflicting.parent.mkdir(parents=True)
        conflicting.write_bytes(b'preexisting experiment evidence')
        with self.assertRaises(RuntimeError):
            self.consume(group)
        self.assertEqual(conflicting.read_bytes(), b'preexisting experiment evidence')
        self.assertFalse((self.destination / DATA).exists())

    def test_corrupt_ciphertext_part_rejected_without_restoring(self):
        group, _, _ = self.archive()
        path = self.release / group['parts'][0]['name']
        path.write_bytes(b'corrupt' + path.read_bytes())
        with self.assertRaisesRegex(RuntimeError, 'corrupt'):
            self.consume(group)
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_age_authentication_failure_never_installs_plaintext(self):
        # More than one age payload chunk: plaintext from earlier authenticated
        # chunks must still stay staged if the final chunk fails authentication.
        group, _, _ = self.archive(members=[(DATA, b'committed evidence', 0o600),
                                           (LATER, os.urandom(200_000), 0o600)],
                                   part_size=8192)
        part = group['parts'][-1]
        path = self.release / part['name']
        corrupt = path.read_bytes()[:-1] + bytes([path.read_bytes()[-1] ^ 1])
        path.write_bytes(corrupt)
        part['sha256'] = sha256(corrupt)
        with self.assertRaises(Exception):
            self.consume(group)
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_traversal_member_rejected(self):
        group, _, _ = self.archive(members=[(PREFIX + '/../../escape', b'bad', 0o600)])
        with self.assertRaisesRegex(RuntimeError, 'Unsafe|unsafe'):
            self.consume(group)
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_manifest_dangling_symlink_cannot_write_outside_destination(self):
        group, _, _ = self.archive()
        external = self.root / 'outside-destination.json'
        target = self.destination / MANIFEST
        target.parent.mkdir(parents=True)
        target.symlink_to(external)
        with self.assertRaises(RuntimeError):
            self.consume(group)
        self.assertFalse(external.exists())
        self.assertTrue(target.is_symlink())
        self.assertFalse((self.destination / DATA).exists())

    def test_parent_symlink_cannot_redirect_restored_file(self):
        group, _, _ = self.archive()
        external = self.root / 'outside-directory'
        external.mkdir()
        target = self.destination / PREFIX
        target.parent.mkdir(parents=True)
        target.symlink_to(external, target_is_directory=True)
        with self.assertRaises(RuntimeError):
            self.consume(group)
        self.assertEqual(list(external.iterdir()), [])

    def test_inventory_mismatch_does_not_leave_unverified_outputs(self):
        def alter(inventory):
            inventory[0]['sha256'] = '0' * 64
            return inventory
        group, _, _ = self.archive(inventory_transform=alter)
        with self.assertRaises(RuntimeError):
            self.consume(group)
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_duplicate_archive_names_are_rejected(self):
        group, _, _ = self.archive(members=[(DATA, b'same', 0o600), (DATA, b'same', 0o600)])
        with self.assertRaises(RuntimeError):
            self.consume(group)
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_duplicate_manifest_entries_are_rejected(self):
        group, _, _ = self.archive(inventory_transform=lambda entries: entries + [dict(entries[0])])
        with self.assertRaises(RuntimeError):
            self.consume(group)
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_part_names_cannot_escape_release_directory(self):
        external = self.root / 'outside-volume'
        external.write_bytes(b'not a release volume')
        part = dict(name='../outside-volume', bytes=external.stat().st_size,
                    sha256=bundle.file_hash(external))
        with self.assertRaises(RuntimeError):
            bundle.verify_part(self.release, part)

    def test_download_does_not_follow_a_partial_file_symlink(self):
        external = self.root / 'unrelated-file'
        external.write_bytes(b'preexisting evidence')
        name = 'test.tar.zst.age.part000'
        (self.release / (name + '.partial')).symlink_to(external)
        response = io.BytesIO(b'new release data')
        response.status = 200
        part = dict(name=name, bytes=len(b'new release data'), sha256=sha256(b'new release data'))
        with patch.object(bundle.urllib.request, 'urlopen', return_value=response):
            with self.assertRaises(RuntimeError):
                bundle.download(self.release, part, 'https://example.invalid/release')
        self.assertEqual(external.read_bytes(), b'preexisting evidence')
        self.assertFalse((self.release / name).exists())

    def test_destination_created_after_preflight_is_still_validated(self):
        stage = self.root / 'stage'
        staged = stage / DATA
        staged.parent.mkdir(parents=True)
        staged.write_bytes(b'verified source')
        entry = dict(path=DATA, bytes=len(b'verified source'),
                     sha256=sha256(b'verified source'), mode=0o600)
        original = bundle.safe_target
        calls = 0

        def create_concurrent_file(destination, name):
            nonlocal calls
            target = original(destination, name)
            if destination == self.destination and name == DATA:
                calls += 1
                if calls == 2:
                    target.parent.mkdir(parents=True)
                    target.write_bytes(b'concurrent evidence')
            return target

        with patch.object(bundle, 'safe_target', side_effect=create_concurrent_file):
            with self.assertRaises(RuntimeError):
                bundle.install_validated(stage, self.destination, [entry], set())
        self.assertEqual((self.destination / DATA).read_bytes(), b'concurrent evidence')


if __name__ == '__main__':
    unittest.main()
