"""Shared execution/audit routing for the explicitly versioned WW repair."""
from types import SimpleNamespace

import pytest

from scripts import webbench_v1 as runner
from scripts.webbench_action_format_v2 import VERSION


POLICY = {'action_format': {'webwalkerqa': {'version': VERSION}}}
BARE = 'Action 1: Visit https://example.org/news'


def test_legacy_and_browsecomp_keep_their_original_parser():
    for plan, benchmark in [({}, 'webwalkerqa'), (POLICY, 'browsecomp_plus')]:
        parser, metadata = runner.action_interface(plan, {'benchmark': benchmark})
        assert parser is runner.parse_action and metadata is None
        assert parser(BARE, 1)[1] is None


@pytest.mark.parametrize('version,text,expected', [
    ('webwalker_action_format_v2', BARE, ('visit', 'https://example.org/news')),
    ('webwalker_action_format_v3', BARE, ('visit', 'https://example.org/news')),
    ('webwalker_action_format_v3', 'Action 1: Search new feature', ('search', 'new feature')),
    ('webwalker_action_format_v3', 'Action 1: Finish literal answer.', ('finish', 'literal answer.')),
    ('webwalker_action_format_v3', 'Action 1: Visit News Archive', ('visit', 'News Archive')),
])
def test_frozen_version_dispatch_preserves_literal_arguments(version, text, expected):
    policy = {'action_format': {'webwalkerqa': {'version': version}}}
    parser, metadata = runner.action_interface(policy, {'benchmark': 'webwalkerqa'})
    assert parser(text, 1)[1:] == parser('Thought 1:' + text, 1)[1:] == expected
    assert metadata(text, 1)['parser_version'] == version
    assert metadata(text, 1)['normalized'] is True


def test_boundary_and_legacy_episode_execute_the_same_raw_action():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[-1]['content']

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=list(text.encode()))

    class Actor:
        tokenizer = Tokenizer()

        def generate_first(self, *args, **kwargs):
            return BARE, None, None

        def generate_incremental(self, *args, **kwargs):
            return 'Action 2: Finish[answer]'

    actor = Actor()
    parser, metadata = runner.action_interface(POLICY, {'benchmark': 'webwalkerqa'})
    boundary = runner.EpisodeBoundary(actor, {'logical_context_tokens': 2000}, parser, metadata)
    with boundary:
        result = actor.generate_first('q', 256)
        assert result[0] == BARE
        executed = parser('Thought 1:' + result[0], 1)
    assert executed[1:] == ('visit', 'https://example.org/news')
    assert boundary.actions == [{'type': executed[1], 'argument': executed[2]}]
    assert boundary.calls[0]['response'] == BARE
    assert boundary.calls[0]['action_parse']['normalized'] is True
    assert boundary.logical_tokens == 257


def make_results(directory, successful_visit=True):
    job = {'benchmark': 'webwalkerqa', 'seed': 233, 'purpose': 'smoke'}
    task = {'id': 'synthetic', 'question': 'public question', 'answer': 'private target',
            'task_sha256': 'bound-task'}
    plan = {**POLICY, 'sha256': 'new-plan',
            'tasks': {'webwalkerqa': {'233': {'smoke': [task]}}}}
    parser, metadata = runner.action_interface(plan, job)
    audit = {'plan_sha256': plan['sha256'], 'job': job, 'task_sha256': task['task_sha256'],
             'actor_calls': [{'response': BARE, 'action_parse': metadata(BARE, 1)}],
             'actions': [{'type': 'visit', 'argument': 'https://example.org/news'}],
             'environment_events': [{'action': 'visit', 'success': successful_visit}]}
    runner.write(directory / 'audit.json', audit)
    row = {'id': task['id'], 'question': task['question'], 'gold_answer': task['answer'],
           'task_sha256': task['task_sha256'], 'audit': 'audit.json',
           'audit_sha256': runner.file_hash(directory / 'audit.json')}
    predictions = {'plan_sha256': plan['sha256'], 'job': job, 'results': [row]}
    runner.write(directory / 'predictions.json', predictions)
    return plan, job, audit, predictions


def test_repaired_smoke_requires_real_successful_navigation(tmp_path):
    plan, job, _, _ = make_results(tmp_path, successful_visit=False)
    with pytest.raises(RuntimeError, match='successful website visit'):
        runner.validate_completed(plan, job, tmp_path)
    plan, job, _, _ = make_results(tmp_path, successful_visit=True)
    runner.validate_completed(plan, job, tmp_path)


def test_out_of_scope_redirect_cannot_qualify_repaired_smoke(tmp_path):
    plan, job, audit, predictions = make_results(tmp_path)
    audit['environment_events'][0]['environmental_failure'] = True
    audit['environment_events'][0]['reason'] = 'redirect_outside_root'
    runner.write(tmp_path / 'audit.json', audit)
    predictions['results'][0]['audit_sha256'] = runner.file_hash(tmp_path / 'audit.json')
    runner.write(tmp_path / 'predictions.json', predictions)
    with pytest.raises(RuntimeError, match='successful website visit'):
        runner.validate_completed(plan, job, tmp_path)


@pytest.mark.parametrize('field', ['action_parse', 'action'])
def test_valid_file_hash_does_not_hide_a_fabricated_normalization(tmp_path, field):
    plan, job, audit, predictions = make_results(tmp_path)
    if field == 'action_parse':
        audit['actor_calls'][0]['action_parse']['normalized'] = False
    else:
        audit['actions'][0]['argument'] = 'https://invented.example/'
    runner.write(tmp_path / 'audit.json', audit)
    predictions['results'][0]['audit_sha256'] = runner.file_hash(tmp_path / 'audit.json')
    runner.write(tmp_path / 'predictions.json', predictions)
    with pytest.raises(RuntimeError, match='differs from raw model output'):
        runner.validate_completed(plan, job, tmp_path)
