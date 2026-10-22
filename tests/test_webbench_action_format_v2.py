"""CPU-only tests for the explicitly scoped WebWalker syntax alias."""
import pytest

from scripts import webbench_v1 as legacy
from scripts.webbench_action_format_v2 import VERSION, parse_action, parse_with_metadata


@pytest.mark.parametrize('url', [
    'https://www.ironhidegames.com/Games/junkworld',
    'http://ac.dragonest.com/en/news',
    'https://example.test/a%20b?x=1&y=%5Bok%5D#part',
])
@pytest.mark.parametrize('prefix', ['', 'Thought 1:', 'Thought 1: I will inspect the page.\n'])
def test_real_failures_and_synthetic_thought_prefix_have_identical_actions(url, prefix):
    text = prefix + f'Action 1: Visit {url}'
    result, metadata = parse_with_metadata(text, 1, legacy.parse_action)
    assert result[1:] == ('visit', url)
    assert parse_action(text, 1, legacy.parse_action) == result
    assert metadata == {
        'parser_version': VERSION, 'syntax': 'bare_visit_url', 'normalized': True,
        'original_action_line': f'Action 1: Visit {url}',
        'normalized_action': f'Visit[{url}]',
    }


@pytest.mark.parametrize('prefix', ['Thought 1:', 'Thought 1: '])
def test_actor_thought_prefix_survives_the_additional_legacy_prefix(prefix):
    raw = prefix + 'Action 1: Visit https://example.test/path'
    original, original_metadata = parse_with_metadata(raw, 1, legacy.parse_action)
    executed, executed_metadata = parse_with_metadata('Thought 1:' + raw, 1, legacy.parse_action)
    assert original[1:] == executed[1:] == ('visit', 'https://example.test/path')
    assert original_metadata == executed_metadata


@pytest.mark.parametrize('text', [
    'Thought 1: choose a link\nAction 1: Visit[Game Introduction]',
    'Action 2: Search[launch date]',
    'Action 3: Finish[September 22, 2023]',
    'Action 4: Lookup[original-document-id]',
    'Finish[Original answer text.]',
    'Action 1: Visit[https://example.test/a]\nAction 2: Search[legacy precedence]',
])
def test_every_existing_strict_result_is_preserved(text):
    expected = legacy.parse_action(text, 1)
    actual, metadata = parse_with_metadata(text, 1, legacy.parse_action)
    assert actual == expected
    assert metadata['syntax'] == 'strict'
    assert metadata['normalized'] is False


@pytest.mark.parametrize('text', [
    'Action 1: Visit /relative/path',
    'Action 1: Visit www.example.test/path',
    'Action 1: Visit https://',
    'Action 1: Visit https:///path',
    'Action 1: Visit https://example.test:bad/path',
    'Action 1: Visit https://example.test:99999/path',
    'Action 1: Visit https://user:password@example.test/path',
    'Action 1: Visit https://example.test/path and then read it',
    'Action 1: Visit https://example.test/path\nThis is my next action.',
    'Action 1: Visit https://example.test/path.',
    'Action 1: Visit "https://example.test/path"',
    'Action 1: Visit `https://example.test/path`',
    'Action 1: Visit https://example.test/a\nAction 2: Visit https://example.test/b',
    'I would use Action 1: Visit https://example.test/path',
    '> Action 1: Visit https://example.test/path',
    '```\nAction 1: Visit https://example.test/path\n```',
    'Observation: the page recommends this\nAction 1: Visit https://example.test/path',
    'Action 1: Visit https://example.test/a\nVisit https://example.test/b',
    'Action 1: Visit\nhttps://example.test/path',
    'Action 1:\tVisit https://example.test/path',
    'Action 1: Visit https://example.test/path\r',
    'Action 1: Visit https://example.test/\x00path',
    'Action 1: Visit https://example.test/\x1bpath',
    'Action 1: Visit https://example.test/\u200bpath',
    'Action 1: Visit https://example.test/\u202epath',
    'Action 1: Search launch date',
    'Action 1: Finish September 22, 2023',
])
def test_fallback_does_not_guess_or_accept_ambiguous_actions(text):
    expected = legacy.parse_action(text, 1)
    assert expected[1] is None
    result, metadata = parse_with_metadata(text, 1, legacy.parse_action)
    assert result == expected
    assert metadata['syntax'] == 'invalid'
    assert metadata['normalized'] is False


def test_strict_parser_receives_original_text_once_and_is_not_modified():
    calls = []
    marker = ('unchanged', 'finish', 'original argument')

    def strict(text, step):
        calls.append((text, step))
        return marker

    result, metadata = parse_with_metadata('literal\tinput', 9, strict)
    assert result is marker
    assert calls == [('literal\tinput', 9)]
    assert metadata['syntax'] == 'strict'

    bare = 'Action 1: Visit https://example.test/path'
    assert legacy.parse_action(bare, 1)[1] is None
    assert parse_action(bare, 1, legacy.parse_action)[1] == 'visit'
    # BrowseComp and any caller retaining the strict parser are unaffected.
    assert legacy.parse_action(bare, 1)[1] is None
