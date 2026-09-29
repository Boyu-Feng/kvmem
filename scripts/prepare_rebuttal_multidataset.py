"""Fetch pinned development sets, normalize with the repository's original loaders."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_legacy import atomic_json

ASSETS = Path('/data/experiment/fengboyu/stepkv/datasets')
SPECS = {
    '2wiki': ('voidful/2WikiMultihopQA', '16852fde9d85cba158cf7e6517e7a3f9415a28c0',
              'dev.json', 12576, 'run_all_2wiki_experiments_v2', '_load_2wiki_from_local'),
    'musique': ('dgslibisey/MuSiQue', 'c8f4f8c9465fb69d31a8eae894c3fd509c4ca321',
                'musique_ans_v1.0_dev.jsonl', 2417, 'run_all_musique_experiments_v2', '_load_musique_from_local'),
}


def prepare(name):
    repo, revision, filename, count, module, loader = SPECS[name]
    directory = ASSETS / name
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / filename
    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}'
    if not raw.exists():
        temp = raw.with_suffix(raw.suffix + '.download')
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temp.open('wb') as target:
                for chunk in response.iter_content(1024 * 1024):
                    target.write(chunk)
        temp.replace(raw)
    rows = getattr(importlib.import_module(module), loader)(str(raw))
    if len(rows) != count or len({r['id'] for r in rows}) != count:
        raise RuntimeError(f'{name}: unexpected size or duplicate IDs: {len(rows)}')
    if any(not r['question'] or not r['answer'] for r in rows):
        raise RuntimeError(f'{name}: empty questions or answers')
    path = directory / 'normalized_dev.json'
    atomic_json(path, rows)
    manifest = dict(dataset=name, repository=repo, revision=revision, split='dev', url=url,
                    raw_path=str(raw), raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                    path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    rows=count, normalizer=module + '.' + loader,
                    normalizer_sha256=hashlib.sha256((ROOT / (module + '.py')).read_bytes()).hexdigest())
    atomic_json(directory / 'ready.json', manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    for name in SPECS:
        prepare(name)
