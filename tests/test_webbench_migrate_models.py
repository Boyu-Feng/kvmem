"""Model transfer safeguards without contacting Hugging Face or using GPUs."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.request


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/webbench_migrate_models.py"
SPEC = importlib.util.spec_from_file_location("webbench_migrate_models", SCRIPT)
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(body))}


class ModelTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = b"frozen-shard-content"
        self.name = "model-00001-of-00001.safetensors"
        self.info = {"bytes": len(self.data), "sha256": hashlib.sha256(self.data).hexdigest(), "kind": "weights"}
        self.model = {"id": "fixture", "repo": "owner/model", "revision": "a" * 40,
                      "path": str(self.root), "files": {self.name: self.info}}

    def partial(self):
        return self.root / f".{self.name}.{self.info['sha256'][:16]}.download.part"

    def test_existing_bad_shard_is_never_replaced_or_requested(self):
        target = self.root / self.name
        target.write_bytes(b"bad")
        opener = mock.Mock()
        with self.assertRaises(migration.MigrationError):
            migration.download_shard(self.model, self.name, None, opener=opener)
        self.assertEqual(target.read_bytes(), b"bad")
        opener.open.assert_not_called()

    def test_range_resume_uses_prefix_and_installs_exact_shard(self):
        self.partial().write_bytes(self.data[:7])
        opener = mock.Mock()
        opener.open.return_value = Response(self.data[7:], 206, {
            "Content-Range": f"bytes 7-{len(self.data) - 1}/{len(self.data)}",
            "Content-Length": str(len(self.data) - 7)})
        migration.download_shard(self.model, self.name, "fixture-token", opener=opener)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_header("Range"), "bytes=7-")
        self.assertIn("/resolve/" + "a" * 40 + "/", request.full_url)
        self.assertEqual((self.root / self.name).read_bytes(), self.data)
        self.assertFalse(self.partial().exists())

    def test_server_ignoring_range_restarts_only_partial(self):
        self.partial().write_bytes(b"old-prefix")
        opener = mock.Mock()
        opener.open.return_value = Response(self.data)
        migration.download_shard(self.model, self.name, None, opener=opener)
        self.assertEqual((self.root / self.name).read_bytes(), self.data)

    def test_wrong_range_or_hash_never_creates_final_shard(self):
        self.partial().write_bytes(self.data[:7])
        opener = mock.Mock()
        opener.open.return_value = Response(self.data[7:], 206, {"Content-Range": "bytes 0-2/3"})
        with self.assertRaises(migration.MigrationError):
            migration.download_shard(self.model, self.name, None, opener=opener)
        self.assertEqual(self.partial().read_bytes(), self.data[:7])
        opener.open.return_value = Response(b"x" * len(self.data))
        with self.assertRaises(migration.MigrationError):
            migration.download_shard(self.model, self.name, None, opener=opener)
        self.assertFalse((self.root / self.name).exists())

    def test_atomic_publication_cannot_replace_existing_file(self):
        target = self.root / self.name
        target.write_bytes(b"original")
        self.partial().write_bytes(self.data)
        with self.assertRaises(FileExistsError):
            migration.publish_no_replace(self.partial(), target)
        self.assertEqual(target.read_bytes(), b"original")
        self.assertEqual(self.partial().read_bytes(), self.data)

    def test_redirect_drops_token_across_hosts_and_forbids_http(self):
        req = urllib.request.Request("https://huggingface.co/owner/model", headers={"Authorization": "Bearer secret"})
        handler = migration.SafeRedirectHandler()
        redirected = handler.redirect_request(req, None, 302, "", {}, "https://cdn.example/shard")
        self.assertIsNone(redirected.get_header("Authorization"))
        with self.assertRaises(migration.MigrationError):
            handler.redirect_request(req, None, 302, "", {}, "http://cdn.example/shard")

    def test_verify_only_missing_shards_has_no_writes_or_token_access(self):
        ready = self.root / "ready.json"
        ready.write_text(json.dumps({key: self.model[key] for key in ("repo", "revision", "path")}))
        self.model.update(ready_path=str(ready), ready_sha256=hashlib.sha256(ready.read_bytes()).hexdigest(),
                          weight_bytes=len(self.data))
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"schema": "webbench_migration_models_v1", "models": [self.model]}))
        before = {str(p): p.read_bytes() for p in self.root.iterdir()}
        with mock.patch.object(migration, "target_hf_token", side_effect=AssertionError("token access")):
            self.assertEqual(migration.main(["--manifest", str(manifest)]), 2)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.iterdir()})


if __name__ == "__main__":
    unittest.main()
