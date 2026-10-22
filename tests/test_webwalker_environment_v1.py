import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts import webwalker_environment_v1 as w


ROOT = 'https://example.org/site/'
PAGE = ROOT + 'details'


def fake_page(url, html=None, **changes):
    html = html or '<a href="details">Details</a><a href="https://elsewhere.org/">Outside</a>'
    return dict(success=True, status_code=200, final_url=url, html=html,
                text='Public page content.', links=w.extract_links(html, url), error='', **changes)


def task():
    return dict(id='x', question='What is public?', root_url=ROOT, answer='SECRET_ANSWER',
                info={'golden_path': 'SECRET_GOLDEN', 'source_website': ['https://secret.org/']})


def test_all_seeds_share_frozen_selection_and_globally_held_out_smoke():
    tasks = [{'id': str(i)} for i in range(680)]
    selected = {seed: w.select_tasks(tasks, seed) for seed in w.SEEDS}
    union = {row['id'] for group in selected.values() for row in group['main']}
    assert len({tuple(r['id'] for r in group['main']) for group in selected.values()}) == 3
    for seed, group in selected.items():
        assert len(group['main']) == 100 and len(group['smoke']) == 2
        assert not union & {row['id'] for row in group['smoke']}
        assert w.select_tasks(tasks, seed) == group


def test_actor_never_receives_or_retains_answer_or_golden_metadata(tmp_path):
    env = w.WebWalkerEnvironment(cache_dir=tmp_path, fetcher=fake_page)
    observation = env.reset(task())
    assert set(env.task) == {'id', 'question', 'root_url'}
    public = observation + json.dumps(env.task) + json.dumps(env.events)
    assert 'SECRET' not in public and 'secret.org' not in public
    assert 'elsewhere.org' not in observation
    assert 'Details' in observation and PAGE in observation


def test_navigation_requires_seen_in_root_link_and_supports_backtracking(tmp_path):
    visited = []

    def fetch(url):
        visited.append(url)
        return fake_page(url)

    env = w.WebWalkerEnvironment(cache_dir=tmp_path, fetcher=fetch)
    env.acknowledge_observation(env.reset(task()))
    assert 'Invalid' in env.step('Visit[https://example.org/site/unseen]')
    assert 'Invalid' in env.step('Visit[https://elsewhere.org/]')
    assert 'Invalid' in env.step('Visit[https://example.org/site-evil/]')
    assert visited == [ROOT]
    assert PAGE in env.step('Search[Details]')
    assert PAGE in env.step('Visit[Details]')
    assert visited == [ROOT, PAGE]
    env.step(f'Visit[{ROOT}]')
    assert visited == [ROOT, PAGE]  # Shared cached snapshot; no live refetch.
    assert env.step('Finish[Answer]') == 'Answer submitted.'
    assert env.done and env.final_answer == 'Answer'


def test_relative_links_use_current_page_and_exclude_non_http_assets():
    html = '''<a href="child">Nested child</a><button onclick="window.location.href='../parent'">Parent</button>
    <a href="javascript:alert(1)">JS</a><a href="/paper.pdf">PDF</a><a href="child">Nested child</a>'''
    links = w.extract_links(html, ROOT + 'nested/')
    assert links == [{'url': ROOT + 'nested/child', 'text': 'Nested child'},
                     {'url': ROOT + 'parent', 'text': 'Parent'}]


def test_shared_snapshot_one_fetch_under_concurrency_and_detects_corruption(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        return fake_page(url)

    cache = w.SnapshotCache(tmp_path, fetch)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(cache.get, [ROOT] * 6))
    assert calls == [ROOT]
    assert len({row['snapshot_sha256'] for row in results}) == 1
    body = tmp_path / results[0]['html']['path']
    body.write_text('tampered')
    with pytest.raises(RuntimeError, match='corrupt'):
        cache.get(ROOT)
    assert calls == [ROOT]


def test_root_http_failure_is_frozen_and_never_a_model_zero(tmp_path):
    calls = []

    def fail(url):
        calls.append(url)
        row = fake_page(url)
        row.update(success=False, status_code=503, error='Service unavailable')
        return row

    for _ in range(2):
        env = w.WebWalkerEnvironment(cache_dir=tmp_path, fetcher=fail)
        with pytest.raises(w.EnvironmentFailure, match='Root page unavailable'):
            env.reset(task())
        assert env.events[-1]['environmental_failure']
        assert env.final_answer is None
    assert calls == [ROOT]


def test_redirect_outside_root_never_exposes_outside_body(tmp_path):
    def redirect(url):
        row = fake_page(url)
        row.update(final_url='https://elsewhere.org/', text='OUTSIDE_SECRET')
        return row

    env = w.WebWalkerEnvironment(cache_dir=tmp_path, fetcher=redirect)
    with pytest.raises(w.EnvironmentFailure, match='redirected'):
        env.reset(task())
    assert env.current_page is None
    assert 'OUTSIDE_SECRET' not in json.dumps(env.events)


def test_public_root_https_www_redirect_uses_observed_evidence_and_preserves_path(tmp_path):
    original = 'http://example.org/site/'
    effective = 'https://www.example.org/site/'

    def fetch(url):
        row = fake_page(url)
        if url == original:
            row.update(final_url=effective, links=w.extract_links('<a href="details">Details</a>', effective))
        return row

    env = w.WebWalkerEnvironment(cache_dir=tmp_path, fetcher=fetch)
    public = {**task(), 'root_url': original}
    observation = env.reset(public)
    env.acknowledge_observation(observation)
    assert effective in observation
    assert env.task['root_url'] == original  # Input identity never rewritten.
    assert env.original_root == original and env.effective_root == effective
    event = env.events[0]
    assert event['original_root'] == original and event['effective_root'] == effective
    assert event['root_canonical_redirect'] is True and event['final_url'] == effective
    assert effective + 'details' in env.step('Visit[Details]')
    assert effective in env.step(f'Visit[{original}]')
    assert not w.canonical_root_redirect(original, 'https://www.example.org/other/')
    assert not w.canonical_root_redirect(original, 'https://www.unrelated.org/site/')
    assert not w.canonical_root_redirect(effective, 'http://example.org/site/')
    assert not w.canonical_root_redirect(original, effective + '?redirect=elsewhere')
    assert not w.canonical_root_redirect(original, 'https://www.example.org:8443/site/')


@pytest.mark.parametrize('redirected', ['https://www.example.org/elsewhere/',
                                      'https://www.unrelated.org/site/'])
def test_unrelated_root_redirect_still_preserves_environment_failure(tmp_path, redirected):
    def fetch(url):
        row = fake_page(url)
        row.update(final_url=redirected)
        return row

    env = w.WebWalkerEnvironment(cache_dir=tmp_path, fetcher=fetch)
    with pytest.raises(w.EnvironmentFailure, match='redirected outside'):
        env.reset({**task(), 'root_url': 'http://example.org/site/'})
    assert env.effective_root == env.original_root
    assert env.events[-1]['environmental_failure']


def test_cache_refuses_changed_renderer_and_frozen_assets(tmp_path):
    w.SnapshotCache(tmp_path, fake_page)
    with pytest.raises(RuntimeError, match='replace frozen'):
        w.SnapshotCache(tmp_path, fake_page, renderer={'kind': 'requests'})
    with pytest.raises(RuntimeError, match='replace frozen'):
        w.immutable_json(tmp_path / 'renderer.json', {})


def test_truncated_link_targets_are_not_searchable_or_visitable(tmp_path):
    hidden = ROOT + 'private_tail'

    def fetch(url):
        return fake_page(url, '<a href="details">Visible</a><a href="private_tail">Hidden</a>')

    env = w.WebWalkerEnvironment(cache_dir=tmp_path, fetcher=fetch)
    raw = env.reset(task())
    # The public URL header remains visible. The second emitted link is cut
    # part-way through its URL by the worker's tokenizer observation cap.
    delivered = raw[:raw.index(hidden) + len(hidden) - 2]
    admitted = env.acknowledge_observation(delivered)
    assert admitted == [PAGE]
    assert hidden not in env.known_links
    assert 'No matching' in env.step('Search[Hidden]')
    assert 'Invalid' in env.step(f'Visit[{hidden}]')
    assert 'Invalid' in env.step('Visit[Hidden]')
    assert PAGE in env.step('Visit[Visible]')


def test_links_require_acknowledgement_and_reset_clears_old_visibility(tmp_path):
    env = w.WebWalkerEnvironment(cache_dir=tmp_path, fetcher=fake_page)
    first = env.reset(task())
    assert 'Invalid' in env.step(f'Visit[{PAGE}]')
    assert env.acknowledge_observation(first) == [PAGE]
    assert PAGE in env.step('Search[Details]')
    env.reset(task())
    assert not env.known_links
    assert 'Invalid' in env.step(f'Visit[{PAGE}]')
