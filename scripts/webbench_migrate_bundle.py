"""Download, verify and restore the encrypted WebBench migration release.

Requires age and zstd for inspection/restoration. Model weights are downloaded
separately by webbench_migrate_models.py. No experiment process is started here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import threading
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / 'config/environments/webbench_migration_bundle_20261022.json'
BLOCK = 8 * 1024 * 1024
ALLOWED_ROOTS = ('mnt/checkpoints/bruli/fengboyu/', 'home/bruli/fengboyu/kvmem/')


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(BLOCK), b''):
            value.update(block)
    return value.hexdigest()


def verify_part(directory, part):
    if (not isinstance(part['name'], str) or Path(part['name']).name != part['name']
            or part['name'] in {'.', '..'} or '\\' in part['name']):
        raise RuntimeError('Release volume name must be a plain filename')
    path = directory / part['name']
    if path.is_symlink():
        raise RuntimeError(f'Release volume cannot be a symlink: {path}')
    if (not path.is_file() or path.stat().st_size != part['bytes']
            or file_hash(path) != part['sha256']):
        raise RuntimeError(f'Missing or corrupt release volume: {path}')


def download(directory, part, base_url):
    if (Path(part['name']).name != part['name'] or part['name'] in {'.', '..'}
            or '\\' in part['name']):
        raise RuntimeError('Release volume name must be a plain filename')
    path = directory / part['name']
    if path.is_symlink():
        raise RuntimeError(f'Release volume cannot be a symlink: {path}')
    if path.exists():
        verify_part(directory, part)
        print(f'Already verified: {path.name}', flush=True)
        return
    partial = path.with_name(path.name + '.partial')
    if partial.is_symlink():
        raise RuntimeError(f'Partial download cannot be a symlink: {partial}')
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(base_url + '/' + part['name'],
        headers={'Range': f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        append = bool(offset and response.status == 206)
        flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | (os.O_APPEND if append else os.O_TRUNC)
        with os.fdopen(os.open(partial, flags, 0o644), 'wb') as handle:
            shutil.copyfileobj(response, handle, BLOCK)
    if partial.stat().st_size != part['bytes'] or file_hash(partial) != part['sha256']:
        raise RuntimeError(f'Incomplete or corrupt download: {partial}; expected SHA256 {part["sha256"]}')
    os.link(partial, path)
    partial.unlink()
    print(f'Downloaded and verified: {path.name}', flush=True)


def safe_target(destination, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or not name.startswith(ALLOWED_ROOTS):
        raise RuntimeError(f'Unsafe archive path: {name}')
    path = destination.joinpath(*relative.parts)
    current = destination
    for component in relative.parts[:-1]:
        current = current / component
        if current.is_symlink():
            raise RuntimeError(f'Refusing to write through directory symlink: {current}')
    return path


def install_validated(stage, destination, entries, directories):
    """Preflight the entire group before installing any verified file."""
    for name in directories:
        path = safe_target(destination, name)
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise RuntimeError(f'Conflicting destination directory: {path}')
    for entry in entries:
        path = safe_target(destination, entry['path'])
        if path.exists() or path.is_symlink():
            if 'symlink' in entry:
                if not path.is_symlink() or os.readlink(path) != entry['symlink']:
                    raise RuntimeError(f'Conflicting destination: {path}')
            elif path.is_symlink() or not path.is_file() or file_hash(path) != entry['sha256']:
                raise RuntimeError(f'Existing file differs; no overwrite performed: {path}')
    for name in sorted(directories, key=lambda value: (len(PurePosixPath(value).parts), value)):
        safe_target(destination, name).mkdir(parents=True, exist_ok=True)
    for entry in entries:
        path = safe_target(destination, entry['path'])
        if path.exists() or path.is_symlink():
            matches = (path.is_symlink() and os.readlink(path) == entry['symlink']) if 'symlink' in entry else (
                not path.is_symlink() and path.is_file() and file_hash(path) == entry['sha256'])
            if not matches:
                raise RuntimeError(f'Destination changed after preflight: {path}')
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if 'symlink' in entry:
            path.symlink_to(entry['symlink'])
            continue
        # A temporary in the destination filesystem permits atomic, no-clobber
        # installation even when /home and /mnt are different mounts.
        fd, temporary_name = tempfile.mkstemp(prefix='.kvmem-restore-', dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, 'wb') as output, (stage / entry['path']).open('rb') as source:
                shutil.copyfileobj(source, output, BLOCK)
            os.chmod(temporary, entry['mode'])
            os.link(temporary, path)  # Refuses to replace a concurrently created file.
        finally:
            temporary.unlink(missing_ok=True)


def consume_archive(group, directory, identity, destination, age, zstd):
    for part in group['parts']:
        verify_part(directory, part)
    decrypt = subprocess.Popen([age, '-d', '-i', str(identity)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    decompress = subprocess.Popen([zstd, '-d', '-c'], stdin=decrypt.stdout, stdout=subprocess.PIPE)
    decrypt.stdout.close()
    feed_errors = []

    def feed():
        try:
            with decrypt.stdin:
                for part in group['parts']:
                    with (directory / part['name']).open('rb') as handle:
                        shutil.copyfileobj(handle, decrypt.stdin, BLOCK)
        except BaseException as error:
            feed_errors.append(error)

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    staging = tempfile.TemporaryDirectory(prefix='.kvmem-restore-', dir=directory) if destination else None
    stage = Path(staging.name) if staging else None
    observed, inner, inner_entry = {}, None, None
    seen, directories = set(), set()
    try:
        with tarfile.open(fileobj=decompress.stdout, mode='r|') as archive:
            for item in archive:
                if item.name in seen:
                    raise RuntimeError(f'Duplicate archive member: {item.name}')
                seen.add(item.name)
                # Validate every member, including in inspect-only mode.
                path = safe_target(stage or Path('/'), item.name)
                if item.isdir():
                    directories.add(item.name)
                    if stage is not None:
                        if path.is_symlink() or (path.exists() and not path.is_dir()):
                            raise RuntimeError(f'Conflicting destination: {path}')
                        path.mkdir(parents=True, exist_ok=True)
                    continue
                if item.issym():
                    if stage is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        if path.exists() or path.is_symlink():
                            if not path.is_symlink() or os.readlink(path) != item.linkname:
                                raise RuntimeError(f'Conflicting destination: {path}')
                        else:
                            path.symlink_to(item.linkname)
                    observed[item.name] = {'path': item.name, 'symlink': item.linkname}
                    continue
                if not item.isfile():
                    raise RuntimeError(f'Unsupported archive member type: {item.name}')
                stream = archive.extractfile(item)
                if item.name == group['inner_manifest']:
                    raw = stream.read()
                    if hashlib.sha256(raw).hexdigest() != group['inner_manifest_sha256']:
                        raise RuntimeError('Inner file manifest hash mismatch')
                    inner = json.loads(raw)
                    inner_entry = dict(path=item.name, sha256=hashlib.sha256(raw).hexdigest(), mode=item.mode)
                    if stage is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        if path.is_symlink() or path.exists():
                            raise RuntimeError(f'Conflicting destination: {path}')
                        path.write_bytes(raw)
                    continue
                digest = hashlib.sha256()
                temporary = None
                output = None
                exists = path.exists() or path.is_symlink()
                if stage is not None:
                    if path.is_symlink() or (exists and not path.is_file()):
                        raise RuntimeError(f'Conflicting destination: {path}')
                    if not exists:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = path.with_name(path.name + f'.restore-{os.getpid()}')
                        output = temporary.open('xb')
                try:
                    for block in iter(lambda: stream.read(BLOCK), b''):
                        digest.update(block)
                        if output is not None:
                            output.write(block)
                    if output is not None:
                        output.close()
                        os.chmod(temporary, item.mode)
                        os.replace(temporary, path)
                    if stage is not None and exists and file_hash(path) != digest.hexdigest():
                        raise RuntimeError(f'Existing file differs; no overwrite performed: {path}')
                finally:
                    if output is not None and not output.closed:
                        output.close()
                observed[item.name] = dict(path=item.name, bytes=item.size,
                    sha256=digest.hexdigest(), mode=item.mode)
        # Drain the decompressor so authentication/compression trailers are checked.
        while decompress.stdout.read(BLOCK):
            pass
        feeder.join()
        if feed_errors or decompress.wait() != 0 or decrypt.wait() != 0:
            raise RuntimeError(f'Decryption/decompression failed: {feed_errors}')
        if (inner is None or len(inner['files']) != len({entry['path'] for entry in inner['files']})
                or observed != {entry['path']: entry for entry in inner['files']}):
            raise RuntimeError('Restored file inventory or SHA256 does not match the captured source')
        if destination is not None:
            install_validated(stage, destination, list(observed.values()) + [inner_entry], directories)
    except BaseException:
        decrypt.kill()
        decompress.kill()
        raise
    finally:
        decrypt.wait()
        decompress.wait()
        feeder.join(timeout=5)
        if decrypt.stdin and not decrypt.stdin.closed:
            decrypt.stdin.close()
        if decompress.stdout:
            decompress.stdout.close()
        if staging:
            staging.cleanup()
    print(f'{group["group"]}: verified {len(observed)} files/symlinks', flush=True)
    return dict(group=group['group'], verified_files=len(observed), destination=str(destination) if destination else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['download', 'verify', 'inspect', 'restore'])
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--directory', type=Path, required=True, help='Local release volume directory')
    parser.add_argument('--identity', type=Path, help='Private age key, transferred separately from this server')
    parser.add_argument('--destination', type=Path, help='Restore root; / recreates frozen absolute paths')
    parser.add_argument('--group', action='append', help='Optional archive group selection')
    parser.add_argument('--age', default='age')
    parser.add_argument('--zstd', default='zstd')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    groups = [g for g in manifest['archives'] if not args.group or g['group'] in args.group]
    if not groups or (args.group and set(args.group) - {g['group'] for g in groups}):
        parser.error('Unknown or empty archive group selection')
    if args.action in ['inspect', 'restore'] and args.identity is None:
        parser.error('--identity is required for decryption')
    if args.action == 'restore' and args.destination is None:
        parser.error('--destination must explicitly name the restore root')
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.destination:
        args.destination.mkdir(parents=True, exist_ok=True)
        args.destination = args.destination.resolve()
    for group in groups:
        if args.action == 'download':
            for part in group['parts']:
                download(args.directory, part, manifest['release_download_url'])
        elif args.action == 'verify':
            for part in group['parts']:
                verify_part(args.directory, part)
            print(f'{group["group"]}: all encrypted volumes verified', flush=True)
        else:
            consume_archive(group, args.directory, args.identity,
                args.destination if args.action == 'restore' else None, args.age, args.zstd)


if __name__ == '__main__':
    main()
