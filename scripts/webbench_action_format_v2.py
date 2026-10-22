"""Pure, deterministic WebWalker action-format compatibility.

The caller supplies its unchanged strict parser. Only a strict parse failure
can use the new alias ``Action N: Visit https://...``. This module neither
changes actor text nor checks/fetches links: the existing environment must
still enforce previously observed links and the root boundary.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable
from urllib.parse import urlsplit

VERSION = 'webwalker_action_format_v2'
ParsedAction = tuple[str, str | None, str | None]
StrictParser = Callable[[str, int], ParsedAction]

_ACTION_MARKER = re.compile(r'\bAction[ ]*\d*[ ]*:', re.I)
_BARE_VISIT = re.compile(
    r'[ ]*(?:Thought[ ]+\d+:[ ]*)*'
    r'(?P<action>Action[ ]*\d*[ ]*:[ ]*Visit[ ]+(?P<url>https?://[^ ]+))[ ]*',
    re.I,
)
_OTHER_SECTION = re.compile(r'^[ ]*(?:Observation|Question|Action Input)[ ]*:', re.I | re.M)


def _absolute_url(value: str) -> bool:
    """Accept a literal URL; never remove punctuation or complete a target."""
    if (any(char.isspace() for char in value)
            or any(char in value for char in '<>"\'`{}\\|')
            or value.endswith(('.', ',', ';', ':', '!', '?', ')', '}'))):
        return False
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in ('http', 'https') or not parsed.netloc or not parsed.hostname:
            return False
        # Invalid ports and unmatched IPv6 brackets must not pass as URLs.
        parsed.port
        if parsed.username is not None or parsed.password is not None:
            return False
        return True
    except ValueError:
        return False


def parse_with_metadata(text: str, step: int, strict_parser: StrictParser
                        ) -> tuple[ParsedAction, dict]:
    """Delegate strict forms unchanged; otherwise admit one bare Visit URL.

    Blank surrounding lines and preceding thought text are allowed. The
    legacy episode prefixes responses with ``Thought N:`` without a newline;
    consecutive prefixes are accepted so an actor-emitted prefix has the same
    action before and after that synthetic prefix is added. Bare actions with
    multiple action markers, following prose, quoted/code-block examples, control or
    formatting characters, non-URL arguments, or inferred punctuation are not.
    Existing strict behavior takes precedence, including its failure tuple.
    """
    strict = strict_parser(text, step)
    metadata = {
        'parser_version': VERSION,
        'syntax': 'strict' if strict[1] is not None else 'invalid',
        'normalized': False,
        'original_action_line': None,
        'normalized_action': None,
    }
    if strict[1] is not None:
        return strict, metadata

    def reject(reason: str):
        return strict, {**metadata, 'rejection_reason': reason}

    if any(char != '\n' and unicodedata.category(char).startswith('C') for char in text):
        return reject('control_or_format_character')
    if len(_ACTION_MARKER.findall(text)) != 1:
        return reject('ambiguous_action_count')
    if '```' in text or '~~~' in text or _OTHER_SECTION.search(text):
        return reject('quoted_code_or_other_section')
    lines = text.split('\n')
    matches = [(index, found) for index, line in enumerate(lines)
               if (found := _BARE_VISIT.fullmatch(line))]
    if len(matches) != 1:
        return reject('not_one_bare_visit_line')
    index, found = matches[0]
    if any(line.strip() for line in lines[index + 1:]):
        return reject('trailing_content')
    argument = found['url']
    if not _absolute_url(argument):
        return reject('invalid_absolute_url')
    return (strict[0], 'visit', argument), {
        **metadata,
        'syntax': 'bare_visit_url',
        'normalized': True,
        'original_action_line': found['action'],
        'normalized_action': f'Visit[{argument}]',
    }


def parse_action(text: str, step: int, strict_parser: StrictParser) -> ParsedAction:
    """Return the legacy tuple interface for the shared episode parser."""
    return parse_with_metadata(text, step, strict_parser)[0]
