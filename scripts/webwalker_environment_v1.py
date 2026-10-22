"""Pinned WebWalkerQA assets and a shared, rendered, root-constrained environment.

The browser follows the public root and links observed on visited pages. Gold
answers, source websites and golden paths never enter the environment. This is
a common single-actor ReAct adaptation, not the upstream multi-agent WebWalker
algorithm. Primary answer scoring remains the upstream GPT-4 cot_qa evaluator.

No static-HTML fallback is allowed. Cache misses use an isolated Crawl4AI /
Chromium subprocess; both successful pages and failures are immutable snapshots.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
from html.parser import HTMLParser
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from typing import Callable
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

BASE = Path('/mnt/checkpoints/bruli/fengboyu/kvmem_webbench_20261021')
DEFAULT_ASSETS = BASE / 'assets/webwalkerqa'
DEFAULT_PYTHON = BASE / 'webwalker_runtime/venv/bin/python'
DEFAULT_BROWSERS = BASE / 'browsers'
DATASET = 'callanwu/WebWalkerQA'
REVISION = 'c7ab3329a2a4a35ca569fdca369b61a3d1787e5b'
DATA_SHA256 = 'cbe95d9de714ef52fe4a9ddf160dfab462407cb1ba526f1c31b4305ca32c34cb'
UPSTREAM_REVISION = 'f72f75d8c3eb842f2bbbab096a12206ff66e270f'
SEEDS = (233, 42, 3407)
RENDERER = {'kind': 'crawl4ai_chromium', 'crawl4ai': '0.7.4',
            'javascript': True, 'wait_until': 'domcontentloaded',
            'page_timeout_ms': 45000, 'delay_before_return_html_seconds': 1.0,
            'screenshot': False, 'cache_mode': 'BYPASS',
            'markdown': 'upstream_clean_markdown',
            'link_extraction': 'observed_rendered_dom_root_path_v1'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_bytes(path, content):
    """Commit without replace; concurrent writers must agree byte for byte."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise RuntimeError(f'Refusing to replace frozen asset: {path}')
    finally:
        temporary.unlink()


def immutable_json(path, value):
    immutable_bytes(path, (canonical(value) + '\n').encode())


def download(url, path, expected=None):
    path = Path(path)
    if not path.exists():
        request = Request(url, headers={'User-Agent': 'KVMem-WebWalkerQA-Asset-Preparation/1.0'})
        with urlopen(request, timeout=60) as response:
            content = response.read()
        if expected and hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError('Public asset checksum does not match pinned source')
        immutable_bytes(path, content)
    actual = file_hash(path)
    if expected and actual != expected:
        raise RuntimeError(f'Frozen public asset changed: {path}')
    return {'url': url, 'path': str(path.resolve()), 'sha256': actual, 'bytes': path.stat().st_size}


def normalize_url(url):
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in ('http', 'https') or not parts.netloc or parts.username or parts.password:
        raise ValueError('Expected a public HTTP(S) URL without embedded credentials')
    # Fragments identify an already visible page, not a new network observation.
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or '/', parts.query, ''))


def allowed_url(root, candidate):
    try:
        a, b = urlsplit(normalize_url(root)), urlsplit(normalize_url(candidate))
    except ValueError:
        return False
    if (a.scheme, a.netloc) != (b.scheme, b.netloc):
        return False
    prefix = a.path if a.path.endswith('/') else a.path + '/'
    return b.path == a.path or b.path.startswith(prefix)


def canonical_root_redirect(original, redirected):
    """Accept an observed HTTPS/www canonical redirect, never infer a target.

    The actual crawler response supplies ``redirected``. A changed root path,
    query, unrelated hostname, nondefault port or HTTPS downgrade is refused.
    """
    try:
        a, b = urlsplit(normalize_url(original)), urlsplit(normalize_url(redirected))
        a_port, b_port = a.port, b.port
    except ValueError:
        return False
    if (a.path, a.query) != (b.path, b.query):
        return False
    if (a.scheme, b.scheme) not in (('http', 'http'), ('http', 'https'), ('https', 'https')):
        return False
    a_host, b_host = a.hostname, b.hostname
    if a_host != b_host and not (a_host == 'www.' + b_host or b_host == 'www.' + a_host):
        return False
    if a_port not in (None, 80 if a.scheme == 'http' else 443):
        return a.scheme == b.scheme and a_port == b_port and a_host == b_host
    return b_port in (None, 80 if b.scheme == 'http' else 443)


def actor_task(task):
    """A strict allowlist is the boundary between scoring and actor inputs."""
    value = {key: str(task[key]) for key in ('id', 'question', 'root_url')}
    if not value['id'] or not value['question'].strip():
        raise ValueError('A nonempty task ID and question are required')
    value['root_url'] = normalize_url(value['root_url'])
    return value


def task_rows(rows):
    tasks = []
    for index, row in enumerate(rows):
        task = {'id': f'webwalkerqa_{index:05d}', 'index': index,
                'question': row['question'], 'answer': row['answer'],
                'root_url': row['root_url'], 'evaluation_info': row['info'],
                'source_row_sha256': digest(row)}
        task['input_sha256'] = digest(actor_task(task))
        task['target_sha256'] = digest({'answer': task['answer'], 'info': task['evaluation_info']})
        tasks.append(task)
    if len({task['question'] for task in tasks}) != len(tasks):
        raise RuntimeError('Duplicate official questions')
    return tasks


def select_tasks(tasks, seed, count=100, smoke_count=2):
    if seed not in SEEDS or count < 1 or smoke_count < 1 or count + smoke_count > len(tasks):
        raise ValueError('Invalid common task selection')
    order = list(range(len(tasks)))
    random.Random(seed).shuffle(order)
    main_union = set()
    for other_seed in SEEDS:
        other_order = list(range(len(tasks)))
        random.Random(other_seed).shuffle(other_order)
        main_union.update(other_order[:count])
    smoke = [index for index in order[count:] if index not in main_union][:smoke_count]
    if len(smoke) != smoke_count:
        raise ValueError('Insufficient globally held-out smoke tasks')
    return {'main': [tasks[i] for i in order[:count]],
            'smoke': [tasks[i] for i in smoke]}


def prepare_assets(root=DEFAULT_ASSETS):
    root = Path(root).resolve()
    hf = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}'
    gh = f'https://raw.githubusercontent.com/Alibaba-NLP/DeepResearch/{UPSTREAM_REVISION}'
    records = {'main': download(f'{hf}/data/main-00000-of-00001.jsonl', root / 'main.jsonl', DATA_SHA256),
               'dataset_readme': download(f'{hf}/README.md', root / 'provenance/dataset_README.md')}
    for name in ('LICENSE', 'WebAgent/LICENSE', 'WebAgent/WebWalker/README.md',
                 'WebAgent/WebWalker/requirements.txt', 'WebAgent/WebWalker/src/evaluate.py',
                 'WebAgent/WebWalker/src/prompts.py', 'WebAgent/WebWalker/src/utils.py',
                 'WebAgent/WebWalker/src/app.py', 'WebAgent/WebWalker/src/agent.py'):
        records[name] = download(f'{gh}/{name}', root / 'provenance/upstream' / name)
    rows = [json.loads(line) for line in (root / 'main.jsonl').read_text().splitlines() if line.strip()]
    if len(rows) != 680:
        raise RuntimeError('Official main split must contain 680 questions')
    tasks = task_rows(rows)
    selections = {str(seed): select_tasks(tasks, seed) for seed in SEEDS}
    immutable_json(root / 'tasks.json', selections)
    manifest = {'schema': 1, 'benchmark': 'webwalkerqa', 'dataset': DATASET,
                'dataset_revision': REVISION, 'split': 'main', 'source_rows': 680,
                'upstream_revision': UPSTREAM_REVISION, 'assets': records,
                'seeds': list(SEEDS), 'main_per_seed': 100, 'smoke_per_seed': 2,
                'selection': 'random.Random(seed).shuffle(source row indices); first 100 main; first 2 tail rows outside all three main sets for smoke',
                'tasks': {'path': str(root / 'tasks.json'), 'sha256': file_hash(root / 'tasks.json')},
                'actor_input_allowlist': ['id', 'question', 'root_url'],
                'renderer': RENDERER,
                'scope': 'Shared single-actor ReAct root-link traversal; not the upstream multi-agent algorithm',
                'primary_scoring': 'Pinned upstream evaluate.py cot_qa GPT-4; no EM/F1 replacement',
                'page_cache': 'First rendered response or failure per URL is immutable and shared across arms'}
    immutable_json(root / 'manifest.json', manifest)
    return manifest


def load_tasks(root=DEFAULT_ASSETS, seed=233, purpose='main'):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if file_hash(root / 'tasks.json') != manifest['tasks']['sha256']:
        raise RuntimeError('Frozen WebWalkerQA tasks changed')
    if file_hash(root / 'main.jsonl') != DATA_SHA256:
        raise RuntimeError('Frozen WebWalkerQA dataset changed')
    if purpose not in ('main', 'smoke') or seed not in SEEDS:
        raise ValueError('Unregistered seed or purpose')
    return json.loads((root / 'tasks.json').read_text())[str(seed)][purpose]


class _Links(HTMLParser):
    """Extract rendered HTML anchor/button targets without evaluating scripts."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.current = [], None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag not in ('a', 'button'):
            return
        url = attrs.get('href') or attrs.get('data-url')
        if not url and attrs.get('onclick'):
            match = re.search(r"window\.location(?:\.href)?\s*=\s*['\"]([^'\"]+)['\"]", attrs['onclick'])
            url = match.group(1) if match else None
        self.current = {'url': url, 'text': [], 'fallback': attrs.get('title') or attrs.get('aria-label') or ''}

    def handle_data(self, data):
        if self.current is not None:
            self.current['text'].append(data)

    def handle_endtag(self, tag):
        if tag in ('a', 'button') and self.current is not None:
            value, self.current = self.current, None
            text = ' '.join(' '.join(value['text']).split()) or value['fallback']
            if value['url'] and text:
                self.links.append({'url': value['url'], 'text': text})


def extract_links(html, page_url):
    parser = _Links()
    parser.feed(html)
    result, seen = [], set()
    for link in parser.links:
        try:
            target = normalize_url(urljoin(page_url, link['url']))
        except ValueError:
            continue
        if urlsplit(target).path.lower().endswith(('.jpg', '.png', '.gif', '.jpeg', '.pdf')):
            continue
        pair = (target, link['text'])
        if pair not in seen:
            result.append({'url': target, 'text': link['text']})
            seen.add(pair)
    return result


def clean_markdown(value):
    # This is the upstream utils.py cleaning rule; links are exposed separately.
    value = re.sub(r'\[.*?\]\(.*?\)', '', value)
    value = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', value)
    return re.sub(r'\n\n+', '\n', value.replace('* \n', ''))


async def crawl_page(url):
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
    if importlib.metadata.version('crawl4ai') != RENDERER['crawl4ai']:
        raise RuntimeError('Crawl4AI runtime version does not match frozen renderer')
    ready = json.loads((DEFAULT_ASSETS / 'provenance/crawler_runtime.json').read_text())
    actual = {name: importlib.metadata.version(name) for name in ready['packages']}
    if actual != ready['packages']:
        raise RuntimeError('Browser package versions changed after preflight')
    browser = BrowserConfig(headless=True, verbose=False)
    config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, screenshot=False, verbose=False,
                              wait_until=RENDERER['wait_until'], page_timeout=RENDERER['page_timeout_ms'],
                              delay_before_return_html=RENDERER['delay_before_return_html_seconds'])
    async with AsyncWebCrawler(config=browser) as crawler:
        result = await crawler.arun(url, config=config)
    html = result.html or ''
    final_url = normalize_url(getattr(result, 'redirected_url', None) or result.url or url)
    markdown = result.markdown
    if hasattr(markdown, 'raw_markdown'):
        markdown = markdown.raw_markdown
    status = getattr(result, 'status_code', None)
    success = bool(result.success and html and (status is None or 200 <= status < 400))
    return {'success': success, 'status_code': status, 'final_url': final_url, 'html': html,
            'text': clean_markdown(str(markdown or '')),
            'links': extract_links(html, final_url), 'error': str(result.error_message or '')}


class Crawl4AIFetcher:
    def __init__(self, python=DEFAULT_PYTHON, browsers=DEFAULT_BROWSERS, temporary=BASE / 'tmp'):
        self.python, self.browsers, self.temporary = Path(python), Path(browsers), Path(temporary)
        self._runtime_verified = False

    def __call__(self, url):
        if not self.python.is_file():
            raise RuntimeError(f'Required isolated browser Python is missing: {self.python}')
        if not self._runtime_verified:
            ready = json.loads((DEFAULT_ASSETS / 'provenance/crawler_runtime.json').read_text())
            for path, checksum in ready['files'].items():
                if file_hash(path) != checksum:
                    raise RuntimeError(f'Frozen browser runtime file changed: {path}')
            self._runtime_verified = True
        self.temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.temporary, prefix='webwalker_') as directory:
            result_path = Path(directory) / 'result.json'
            environment = dict(os.environ, PLAYWRIGHT_BROWSERS_PATH=str(self.browsers),
                               TMPDIR=str(self.temporary), CRAWL4_AI_BASE_DIRECTORY=str(BASE / 'webwalker_runtime'),
                               CUDA_VISIBLE_DEVICES='')
            local_libraries = str(BASE / 'webwalker_runtime/system_packages/usr/lib/x86_64-linux-gnu')
            environment['LD_LIBRARY_PATH'] = local_libraries + (
                ':' + environment['LD_LIBRARY_PATH'] if environment.get('LD_LIBRARY_PATH') else '')
            command = [str(self.python), str(Path(__file__).resolve()), '--fetch-page', url,
                       '--fetch-output', str(result_path)]
            completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=90)
            if completed.returncode or not result_path.is_file():
                # Dependency failures are not persistent website failures.
                raise RuntimeError('Rendered crawler subprocess failed: ' + completed.stderr[-4000:])
            return json.loads(result_path.read_text())


class SnapshotCache:
    def __init__(self, root, fetcher=None, renderer=None):
        self.root = Path(root)
        self.fetcher = fetcher or Crawl4AIFetcher()
        self.renderer = dict(RENDERER if renderer is None else renderer)
        if isinstance(self.fetcher, Crawl4AIFetcher):
            runtime = DEFAULT_ASSETS / 'provenance/crawler_runtime.json'
            self.renderer['runtime_manifest_sha256'] = file_hash(runtime)
        immutable_json(self.root / 'renderer.json', self.renderer)

    def get(self, url):
        url = normalize_url(url)
        key = hashlib.sha256(url.encode()).hexdigest()
        directory = self.root / key[:2]
        directory.mkdir(parents=True, exist_ok=True)
        record_path = directory / (key + '.json')
        with (directory / (key + '.lock')).open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if not record_path.exists():
                fetched = self.fetcher(url)
                if not isinstance(fetched.get('success'), bool):
                    raise RuntimeError('Malformed crawler result')
                body = str(fetched.pop('html', '')).encode()
                # A crash before the record commit can leave an unreferenced
                # object, but can never make a later snapshot overwrite bytes.
                body_sha256 = hashlib.sha256(body).hexdigest()
                body_path = self.root / 'objects' / body_sha256[:2] / (body_sha256 + '.html')
                immutable_bytes(body_path, body)
                payload = {'schema': 1, 'url': url, 'renderer_sha256': digest(self.renderer),
                           'fetched_at_utc': datetime.now(timezone.utc).isoformat(),
                           'html': {'path': str(body_path.relative_to(self.root)), 'sha256': body_sha256},
                           **fetched}
                immutable_json(record_path, {'payload': payload, 'sha256': digest(payload)})
            record = json.loads(record_path.read_text())
            payload = record['payload']
            if (record['sha256'] != digest(payload) or payload['url'] != url
                    or payload['renderer_sha256'] != digest(self.renderer)
                    or file_hash(self.root / payload['html']['path']) != payload['html']['sha256']):
                raise RuntimeError('Shared page snapshot is corrupt or has changed')
        return {**payload, 'snapshot_path': str(record_path), 'snapshot_sha256': record['sha256']}


class EnvironmentFailure(RuntimeError):
    """A root/page or browser failure is not an incorrect model answer."""


class WebWalkerEnvironment:
    def __init__(self, cache_dir=DEFAULT_ASSETS / 'pages', *, cache=None, fetcher=None,
                 crawler_python=DEFAULT_PYTHON, browser_path=DEFAULT_BROWSERS):
        self.cache = cache or SnapshotCache(cache_dir, fetcher or Crawl4AIFetcher(crawler_python, browser_path))
        self.task = None
        self.done, self.final_answer = False, None
        self.events, self.known_links = [], {}
        self.pending_links = []
        self.current_page = None
        self.last_event = None
        self.original_root = self.effective_root = None

    def _event(self, **value):
        self.last_event = value
        self.events.append(value)

    def reset(self, task):
        self.task = actor_task(task)
        self.original_root = self.effective_root = self.task['root_url']
        self.done, self.final_answer = False, None
        self.events, self.known_links, self.current_page = [], {}, None
        self.pending_links = []
        return self._visit(self.task['root_url'], root=True)

    def _visit(self, url, root=False):
        self.pending_links = []
        page = self.cache.get(url)
        event = {'action': 'root' if root else 'visit', 'url': url,
                 'snapshot_sha256': page['snapshot_sha256'], 'snapshot_path': page['snapshot_path'],
                 'status_code': page.get('status_code'), 'success': page['success'],
                 'original_root': self.original_root, 'effective_root': self.effective_root,
                 'final_url': page.get('final_url')}
        if not page['success']:
            self._event(**event, error=page.get('error', ''), environmental_failure=True)
            if root:
                raise EnvironmentFailure(f'Root page unavailable: {url}; snapshot={page["snapshot_sha256"]}')
            return 'Page unavailable. The failed response has been recorded; choose another observed link.'
        if root and page['final_url'] != self.original_root and canonical_root_redirect(self.original_root, page['final_url']):
            self.effective_root = page['final_url']
            event['effective_root'] = self.effective_root
            event['root_canonical_redirect'] = True
        if not allowed_url(self.effective_root, page['final_url']):
            self._event(**event, error='redirect_outside_root', environmental_failure=True)
            if root:
                raise EnvironmentFailure(f'Root redirected outside declared root: {page["final_url"]}')
            return 'Navigation redirected outside the allowed root. No outside content is exposed.'
        self.current_page = page
        links = [link for link in page['links'] if allowed_url(self.effective_root, link['url'])]
        self.pending_links = links
        self._event(**event, rendered_page_links=len(links), environmental_failure=False)
        # The shared worker bounds observations in tokenizer tokens. Put the
        # available actions first so a long body cannot hide all navigation.
        return ('URL: ' + page['final_url'] + '\nClickable links:\n' + '\n'.join(
            f'- {link["text"]}: {link["url"]}' for link in links) + '\nPage:\n' + page['text'])

    def acknowledge_observation(self, rendered_text):
        """Admit only complete link lines actually delivered to the actor.

        The shared worker calls this after applying its tokenizer token cap,
        for the initial observation and every subsequent tool observation.
        Raw snapshot links beyond the cap never become Search/Visit targets.
        """
        lines = set(rendered_text.splitlines())
        newly_visible = []
        for link in self.pending_links:
            if f'- {link["text"]}: {link["url"]}' in lines:
                self.known_links[link['url']] = link['text']
                newly_visible.append(link['url'])
        self.pending_links = []
        self._event(action='observation_acknowledged', observation_sha256=digest(rendered_text),
                    acknowledged_link_urls=newly_visible, cumulative_visible_links=len(self.known_links),
                    original_root=self.original_root, effective_root=self.effective_root,
                    success=True, environmental_failure=False)
        return newly_visible

    def step(self, action):
        if self.task is None or self.done:
            raise RuntimeError('Environment must be reset and unfinished')
        match = re.fullmatch(r'\s*(Search|Visit|Finish)\[(.*)\]\s*', action, re.I | re.S)
        if not match:
            self._event(action='invalid', input=action, success=False, environmental_failure=False)
            return 'Invalid action. Use Search[keywords], Visit[observed URL or exact link text], or Finish[answer].'
        name, argument = match[1].lower(), match[2].strip()
        if name == 'finish':
            self.done, self.final_answer = True, argument
            self._event(action=name, success=True, environmental_failure=False)
            return 'Answer submitted.'
        if name == 'search':
            self.pending_links = []
            terms = argument.casefold().split()
            found = [{'url': url, 'text': text} for url, text in self.known_links.items()
                     if terms and all(word in (text + ' ' + url).casefold() for word in terms)]
            self._event(action=name, query=argument, success=True, environmental_failure=False, matches=len(found))
            return ('Matching previously observed links:\n' + '\n'.join(f'- {r["text"]}: {r["url"]}' for r in found)
                    if found else 'No matching observed links. Search searches links already exposed by visited pages.')
        if argument in (self.original_root, self.effective_root):
            return self._visit(argument)
        targets = [url for url, text in self.known_links.items() if argument in (url, text)]
        if len(targets) != 1 or not allowed_url(self.effective_root, targets[0]):
            self._event(action=name, input=argument, success=False, environmental_failure=False)
            return 'Invalid or ambiguous link. Visit one exact URL already observed within the root website.'
        return self._visit(targets[0])


Environment = WebWalkerEnvironment


def browser_preflight(root=DEFAULT_ASSETS):
    """Real JavaScript/navigation check plus one globally held-out public root.

    No model, judge or formal task is executed. Result/cache records are kept
    separately from the formal experiment cache.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    root = Path(root)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            html = ('<html><body><p>Child content marker.</p></body></html>' if self.path == '/child' else
                    '<html><body><p id="value">Before JavaScript</p><script>'
                    'setTimeout(()=>{document.getElementById("value").textContent="JS_RENDERED_MARKER";'
                    'const a=document.createElement("a");a.href="/child";a.textContent="Rendered child";'
                    'document.body.appendChild(a)},100)</script></body></html>')
            payload = html.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/'
    fetcher = Crawl4AIFetcher()
    env = WebWalkerEnvironment(root / 'preflight/pages', fetcher=fetcher)
    try:
        first = env.reset({'id': 'synthetic_browser_js', 'question': 'Read the public child marker.', 'root_url': url})
        if 'JS_RENDERED_MARKER' not in first or 'Rendered child' not in first:
            raise RuntimeError('Real browser failed to execute JavaScript before snapshot')
        env.acknowledge_observation(first)
        second = env.step('Visit[Rendered child]')
        env.acknowledge_observation(second)
        if 'Child content marker.' not in second:
            raise RuntimeError('Rendered dynamic link was not navigable')
        synthetic = {'javascript_executed': True, 'rendered_link_navigation': True, 'events': env.events}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    public = actor_task(load_tasks(root, 233, 'smoke')[0])
    formal_ids = {task['id'] for seed in SEEDS for task in load_tasks(root, seed, 'main')}
    if public['id'] in formal_ids:
        raise RuntimeError('Browser preflight must not use a formal task')
    try:
        observation = env.reset(public)
        env.acknowledge_observation(observation)
        official = {'status': 'accessible', 'task': public, 'observation_sha256': digest(observation),
                    'observation_characters': len(observation), 'events': env.events}
    except EnvironmentFailure as error:
        official = {'status': 'environment_failure', 'task': public, 'error': str(error), 'events': env.events}
    record = {'schema': 1, 'synthetic': synthetic, 'heldout_root': official,
              'runtime_manifest_sha256': file_hash(root / 'provenance/crawler_runtime.json'),
              'source_sha256': file_hash(__file__), 'actor_calls': 0, 'formal_episodes': 0}
    # Each attempt retains its own record, including inaccessible roots.
    path = root / 'preflight' / ('browser_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '.json')
    immutable_json(path, record)
    return {'path': str(path), 'sha256': file_hash(path), 'synthetic': 'passed',
            'heldout_root_status': official['status'], 'actor_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--browser-preflight', action='store_true')
    parser.add_argument('--assets', type=Path, default=DEFAULT_ASSETS)
    parser.add_argument('--fetch-page')
    parser.add_argument('--fetch-output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        value = prepare_assets(args.assets)
        print(canonical({'manifest': str(args.assets / 'manifest.json'), 'sha256': digest(value), 'rows': value['source_rows']}))
    elif args.browser_preflight:
        print(canonical(browser_preflight(args.assets)))
    elif args.fetch_page and args.fetch_output:
        import asyncio
        value = asyncio.run(crawl_page(normalize_url(args.fetch_page)))
        immutable_json(args.fetch_output, value)
    else:
        parser.error('Choose --prepare or --fetch-page URL --fetch-output FILE')


if __name__ == '__main__':
    main()
