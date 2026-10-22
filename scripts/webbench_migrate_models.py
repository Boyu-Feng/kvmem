#!/usr/bin/env python3
"""Verify frozen WebBench models, or download missing weight shards explicitly.

Uses only the Python standard library. The default mode is entirely read-only.
Restore the encrypted support files first; --download never replaces an existing
file or downloads a different revision. Incomplete downloads remain resumable.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / (
    "config/environments/webbench_migration_models_20261022.json"
)


class MigrationError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise MigrationError(f"File changed during verification: {path}")
    return value.hexdigest()


def verify_file(path: Path, info: dict) -> None:
    if not path.is_file():
        raise MigrationError(f"Missing regular file: {path}")
    if path.stat().st_size != info["bytes"] or file_sha256(path) != info["sha256"]:
        raise MigrationError(f"Existing file does not match its frozen hash; refusing replacement: {path}")


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "webbench_migration_models_v1":
        raise MigrationError("Unsupported model manifest schema")
    ids = set()
    for model in manifest["models"]:
        if model["id"] in ids:
            raise MigrationError("Duplicate model ID")
        ids.add(model["id"])
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model["repo"]):
            raise MigrationError("Invalid Hugging Face repository")
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise MigrationError("A fixed 40-character model revision is required")
        if not Path(model["path"]).is_absolute() or not Path(model["ready_path"]).is_absolute():
            raise MigrationError("Model and ready paths must retain their original absolute paths")
        if not re.fullmatch(r"[0-9a-f]{64}", model["ready_sha256"]):
            raise MigrationError("Invalid ready manifest hash")
        total = 0
        for name, info in model["files"].items():
            if not name or name in {".", ".."} or Path(name).name != name or "\\" in name:
                raise MigrationError("Model inventory filenames must be plain basenames")
            if not re.fullmatch(r"[0-9a-f]{64}", info["sha256"]) or not isinstance(info["bytes"], int) or info["bytes"] <= 0:
                raise MigrationError("Invalid model inventory hash or size")
            expected_kind = "weights" if name.endswith(".safetensors") else "support"
            if info["kind"] != expected_kind:
                raise MigrationError("Model inventory file kind does not match its filename")
            if expected_kind == "weights":
                total += info["bytes"]
        if total != model["weight_bytes"]:
            raise MigrationError("Model weight byte total differs from its inventory")
    return manifest


def preflight(model: dict) -> list[str]:
    ready = Path(model["ready_path"])
    if not ready.is_file() or file_sha256(ready) != model["ready_sha256"]:
        raise MigrationError(f"Restore the exact original ready manifest first: {ready}")
    saved = json.loads(ready.read_text())
    if any(saved.get(key) != model[key] for key in ("repo", "revision", "path")):
        raise MigrationError(f"Ready manifest model identity differs: {ready}")
    missing = []
    for name, info in model["files"].items():
        path = Path(model["path"]) / name
        if os.path.lexists(path):
            verify_file(path, info)
        elif info["kind"] == "weights":
            missing.append(name)
        else:
            raise MigrationError(f"Restore encrypted model config/tokenizer support files first: {path}")
    return missing


def target_hf_token() -> str | None:
    """Read target-machine credentials only; never include them in output."""
    def checked(value: str) -> str | None:
        value = value.strip()
        if value and not re.fullmatch(r"[!-~]+", value):
            raise MigrationError("Target Hugging Face token contains invalid header characters")
        return value or None

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(key, "").strip()
        if token:
            return checked(token)
    if os.environ.get("HF_TOKEN_PATH"):
        path = Path(os.environ["HF_TOKEN_PATH"]).expanduser()
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        path = Path(os.environ.get("HF_HOME", str(cache / "huggingface"))) / "token"
    if path.is_file():
        return checked(path.read_text())
    return None


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise MigrationError("Refusing a model download redirect without HTTPS")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            redirected.remove_header("Authorization")
        return redirected


@contextlib.contextmanager
def model_lock(directory: Path):
    path = directory / ".webbench-migration-models.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MigrationError(f"Another model transfer holds the lock: {directory}") from exc
        yield
    finally:
        os.close(fd)


def publish_no_replace(partial: Path, target: Path) -> None:
    """Atomic rename with no replacement on Linux; atomic hard-link fallback."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is not None:
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(partial), -100, os.fsencode(target), 1) == 0:
            return
        code = ctypes.get_errno()
        if code not in (errno.ENOSYS, errno.EINVAL):
            raise OSError(code, os.strerror(code), str(target))
    os.link(partial, target)
    partial.unlink()


def download_shard(model: dict, name: str, token: str | None, *, opener=None) -> None:
    info = model["files"][name]
    directory = Path(model["path"])
    target = directory / name
    if os.path.lexists(target):
        verify_file(target, info)
        return
    partial = directory / f".{name}.{info['sha256'][:16]}.download.part"
    fd = os.open(partial, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+b") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise MigrationError(f"Partial download is not a regular file: {partial}")
        offset = os.fstat(stream.fileno()).st_size
        if offset > info["bytes"]:
            raise MigrationError(f"Partial download exceeds frozen size; inspect it before retrying: {partial}")
        if offset < info["bytes"]:
            url = "https://huggingface.co/{}/resolve/{}/{}".format(
                model["repo"], model["revision"], urllib.parse.quote(name, safe="")
            )
            headers = {"User-Agent": "kvmem-webbench-migration/1", "Accept-Encoding": "identity"}
            if token:
                headers["Authorization"] = "Bearer " + token
            if offset:
                headers["Range"] = f"bytes={offset}-"
            client = opener or urllib.request.build_opener(SafeRedirectHandler())
            try:
                response = client.open(urllib.request.Request(url, headers=headers), timeout=120)
                with response:
                    code = response.status
                    if code == 206:
                        value = response.headers.get("Content-Range", "")
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value)
                        if not match or tuple(map(int, match.groups())) != (offset, info["bytes"] - 1, info["bytes"]):
                            raise MigrationError(f"Unexpected HTTP byte range for {name}")
                    elif code == 200:
                        offset = 0  # The server ignored Range; restart only our partial file.
                        stream.seek(0)
                        stream.truncate()
                    else:
                        raise MigrationError(f"Unexpected HTTP status {code} for {name}")
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) != info["bytes"] - offset:
                        raise MigrationError(f"Unexpected HTTP content length for {name}")
                    stream.seek(offset)
                    received = offset
                    for block in iter(lambda: response.read(8 << 20), b""):
                        received += len(block)
                        if received > info["bytes"]:
                            raise MigrationError(f"Download exceeds frozen shard size: {name}")
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                    if received != info["bytes"]:
                        raise MigrationError(f"Incomplete download retained for resume: {name}")
            except urllib.error.HTTPError as exc:
                hint = "; supply an authorized HF_TOKEN on this server" if exc.code in (401, 403) else ""
                raise MigrationError(f"HTTP {exc.code} downloading {name}{hint}; partial retained") from None
            except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError, OSError) as exc:
                raise MigrationError(f"Transfer interrupted for {name} ({type(exc).__name__}); partial retained") from None
            except ValueError:
                raise MigrationError(f"Invalid HTTP transfer metadata for {name}; partial retained") from None
    if file_sha256(partial) != info["sha256"]:
        raise MigrationError(f"Downloaded shard hash mismatch; inspect/remove only this partial before retrying: {partial}")
    os.chmod(partial, 0o644)
    try:
        publish_no_replace(partial, target)
    except FileExistsError:
        # Another writer cannot be allowed to replace either version silently.
        verify_file(target, info)
        raise MigrationError(f"Target appeared during download; matching partial retained: {target}") from None
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(f"Verified and installed {model['id']}/{name}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", action="append", help="Select a manifest model ID; repeat to select several")
    parser.add_argument("--download", action="store_true", help="Download missing weight shards; never replace existing files")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    known = {model["id"] for model in manifest["models"]}
    if args.model and set(args.model) - known:
        raise MigrationError("Unknown model ID: " + ", ".join(sorted(set(args.model) - known)))
    models = [m for m in manifest["models"] if not args.model or m["id"] in args.model]
    missing = {}
    # Validate every existing selected file before any download or lock-file write.
    for model in models:
        print(f"Verifying {model['id']} at fixed revision {model['revision']}", flush=True)
        missing[model["id"]] = preflight(model)
    if not args.download:
        for model_id, names in missing.items():
            print(f"{model_id}: {len(names)} missing weight shard(s)" if names else f"{model_id}: verified")
        return 2 if any(missing.values()) else 0
    token = target_hf_token() if any(missing.values()) else None
    for model in models:
        if not missing[model["id"]]:
            continue
        if model.get("requires_hf_token") and not token:
            raise MigrationError(f"{model['id']} requires an authorized target-server HF_TOKEN or Hugging Face cache token")
        with model_lock(Path(model["path"])):
            for name in missing[model["id"]]:
                download_shard(model, name, token)
    print("All selected model files verified.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; incomplete shards retained for --download resume.", file=sys.stderr)
        raise SystemExit(130)
    except (MigrationError, OSError, ValueError, KeyError) as exc:
        print(f"Model migration failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
