"""CPU-only checks for literal bare WebWalker actions, with no inference."""
import pytest

from scripts import webbench_v1 as legacy
from scripts import webbench_action_format_v2 as v2
from scripts import webbench_action_format_v3 as v3


@pytest.mark.parametrize(('kind', 'argument', 'syntax'), [
    ('Visit', 'https://www.ironhidegames.com/Games/junkworld', 'bare_visit_url'),
    ('Visit', 'http://ac.dragonest.com/en/news', 'bare_visit_url'),
    ('Visit', 'https://example.test/a%20b?x=1&y=%5Bok%5D#part', 'bare_visit_url'),
    ('Visit', 'Game Introduction', 'bare_visit_label'),
    ('Visit', '新闻  和更新', 'bare_visit_label'),
    ('Visit', 'Release 1.2', 'bare_visit_label'),
    ('Visit', 'FAQ:', 'bare_visit_label'),
    ('Visit', 'News: New Features', 'bare_visit_label'),
    ('Search', 'new feature', 'bare_search'),
    ('Search', '"launch  date"', 'bare_search'),
    ('Finish', 'September 22, 2023.', 'bare_finish'),
    ('Finish', 'Smith [1999]', 'bare_finish'),
    ('Finish', '答案：两项，保持原文。', 'bare_finish'),
])
def test_literal_arguments_and_metadata_are_preserved(kind, argument, syntax):
    raw = f'Action 12: {kind} {argument}'
    parsed, metadata = v3.parse_with_metadata(raw, 12, legacy.parse_action)
    assert parsed[1:] == (kind.lower(), argument)
    assert v3.parse_action(raw, 12, legacy.parse_action) == parsed
    assert metadata == {
        'parser_version': v3.VERSION, 'syntax': syntax, 'normalized': True,
        'original_action_line': raw, 'original_action_text': raw,
        'normalized_action': f'{kind}[{argument}]',
    }


@pytest.mark.parametrize(('raw', 'kind', 'argument'), [
    ('Action 12: Search "release date" "new features" ', 'search', '"release date" "new features"'),
    ('Action 1: Search   new  feature  \n\n', 'search', 'new  feature'),
    ('Action 1: Visit   Game Introduction  ', 'visit', 'Game Introduction'),
    ('Action 1: Visit https://example.test/news  ', 'visit', 'https://example.test/news'),
    ('Action 1: Finish answer\u00a0', 'finish', 'answer'),
    ('Action 4: Finish \nA literal complete answer.\n\n', 'finish', 'A literal complete answer.'),
    ('Action 4: Finish\nA literal complete answer.', 'finish', 'A literal complete answer.'),
    ('Action 4: Finish First line.\nSecond line.\n\n', 'finish', 'First line.\nSecond line.'),
    ('Action 4: Finish \n  First line.\n\n  Second [literal] line.\n  ',
     'finish', 'First line.\n\n  Second [literal] line.'),
    ('Action 4: Finish\nI thought about the Action movie.',
     'finish', 'I thought about the Action movie.'),
])
def test_only_outer_formatting_whitespace_is_removed_and_multiline_finish_is_literal(raw, kind, argument):
    parsed, metadata = v3.parse_with_metadata(raw, 4, legacy.parse_action)
    executed, executed_metadata = v3.parse_with_metadata('Thought 4:' + raw, 4, legacy.parse_action)
    assert parsed[1:] == executed[1:] == (kind, argument)
    assert metadata == executed_metadata
    assert metadata['normalized_action'] == f'{kind.title()}[{argument}]'
    assert metadata['original_action_line'] == raw.split('\n', 1)[0]
    assert metadata['original_action_text'] == (raw if kind == 'finish' else raw.split('\n', 1)[0])


@pytest.mark.parametrize('action', [
    'Visit https://example.test/path', 'Visit Game Introduction',
    'Search new feature', 'Finish September 22, 2023.',
])
@pytest.mark.parametrize('prefix', [
    '', 'Thought 12:', 'Thought 12: ', 'Thought 12:Thought 12:',
    'Thought 12: I will use the available evidence.\n', '\n',
])
def test_actor_and_legacy_thought_prefixes_have_identical_actions_and_metadata(action, prefix):
    raw = prefix + 'Action 12: ' + action
    original, metadata = v3.parse_with_metadata(raw, 12, legacy.parse_action)
    executed, executed_metadata = v3.parse_with_metadata('Thought 12:' + raw, 12, legacy.parse_action)
    assert original[1] is not None
    assert original[1:] == executed[1:]
    assert metadata == executed_metadata


@pytest.mark.parametrize('text', [
    'Thought 1: choose a link\nAction 1: Visit[Game Introduction]',
    'Action 2: Search[launch date]', 'Action 3: Finish[September 22, 2023]',
    'Action 4: Lookup[original-document-id]', 'Finish[Original answer text.]',
    'Action 1: Visit[https://example.test/a]\nAction 2: Search[legacy precedence]',
    'Action 1: Search the phrase Finish[original strict precedence]',
    '```\nAction 1: Finish[original strict code behavior]\n```',
    'Action 1: Search[legacy\tcontrol behavior]',
])
def test_every_strict_result_and_its_original_precedence_are_preserved(text):
    expected = legacy.parse_action(text, 1)
    assert expected[1] is not None
    parsed, metadata = v3.parse_with_metadata(text, 1, legacy.parse_action)
    assert parsed == expected
    assert metadata['syntax'] == 'strict' and metadata['normalized'] is False


@pytest.mark.parametrize('argument', [
    '/relative/path', '//example.test/path', './path', '../path',
    'www.example.test/path', 'https://', 'https:///path', 'https:/example.test/path',
    'ftp://example.test/path', 'https://example.test:bad/path',
    'https://example.test:99999/path', 'https://user:password@example.test/path',
    'https://example.test/path and then read it', 'https://example.test/path.',
    'https://example.test/path,', 'https://example.test/a\\b',
])
def test_url_shaped_visit_uses_the_unchanged_absolute_url_check(argument):
    assert not v2._absolute_url(argument)
    assert v3._absolute_url(argument) == v2._absolute_url(argument)
    text = 'Action 1: Visit ' + argument
    original = legacy.parse_action(text, 1)
    assert original[1] is None
    parsed, metadata = v3.parse_with_metadata(text, 1, legacy.parse_action)
    assert parsed == original
    assert metadata['rejection_reason'] == 'invalid_absolute_url'


@pytest.mark.parametrize('text', [
    'Action 1: Search', 'Action 1: Search ', 'Action 1: Finish ',
    'Action 1: Finish\n \n  ',
    'Action 1: Search\nnew feature',
    'Action 1: Search new feature\nThis is my next action.',
    'Action 1: Finish answer\nObservation: answered',
    'Action 1: Finish\nanswer\nObservation 2: answered',
    'Action 1: Finish\nanswer\nQuestion: another question',
    'Action 1: Finish\nanswer\nAction Input: another argument',
    'Action 1: Finish\nanswer\nThought 2: another thought',
    'Action 1: Finish\nanswer\nAction 2: Search new feature',
    'Action 1: Visit News\nAction 2: Search new feature',
    'Action 1: Search Action 2: Finish answer',
    'I would use Action 1: Visit News',
    '> Action 1: Search new feature',
    '"Action 1: Finish answer"',
    '`Action 1: Finish answer`',
    '```\nAction 1: Search new feature\n```',
    '~~~\nAction 1: Finish answer\n~~~',
    'Observation: next page\nAction 1: Visit News',
    'Observation 1: next page\nAction 1: Visit News',
    'Question: which feature?\nAction 1: Search new feature',
    'Action Input: new feature\nAction 1: Search new feature',
    'Action 1:\tSearch new feature', 'Action 1: Search new\tfeature',
    'Action 1: Visit News\r', 'Action 1: Search new\x00feature',
    'Action 1: Finish ans\x1bwer', 'Action 1: Search new\u200bfeature',
    'Action 1: Finish ans\u202ewer',
    'Action 1: Search new\u2028feature', 'Action 1: Finish answer\u2029tail',
    'Action 1: Lookup document-id',
])
def test_ambiguous_or_incomplete_bare_forms_remain_invalid_with_legacy_prefix(text):
    original = legacy.parse_action(text, 1)
    assert original[1] is None
    parsed, metadata = v3.parse_with_metadata(text, 1, legacy.parse_action)
    executed, executed_metadata = v3.parse_with_metadata('Thought 1:' + text, 1, legacy.parse_action)
    assert parsed == original
    assert executed[1:] == parsed[1:] == (None, None)
    assert metadata == executed_metadata
    assert metadata['syntax'] == 'invalid' and metadata['normalized'] is False


def test_strict_parser_is_called_once_with_unmodified_text_and_result():
    calls = []
    marker = ('unchanged', 'finish', 'literal argument')

    def strict(text, step):
        calls.append((text, step))
        return marker

    parsed, metadata = v3.parse_with_metadata('literal\tinput', 9, strict)
    assert parsed is marker
    assert calls == [('literal\tinput', 9)]
    assert metadata['syntax'] == 'strict'


def test_v2_and_original_parser_remain_unchanged():
    text = 'Action 12: Search new feature'
    assert legacy.parse_action(text, 12)[1] is None
    assert v2.parse_action(text, 12, legacy.parse_action)[1] is None
    assert v3.parse_action(text, 12, legacy.parse_action)[1:] == ('search', 'new feature')
