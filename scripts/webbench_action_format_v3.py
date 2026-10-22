"""Deterministic literal-argument aliases for the shared WebWalker actor.

Strict parsing retains its existing precedence. A strict failure may use one
bare Visit, Search or Finish action; only surrounding argument whitespace is
removed, as in the strict parser, and no argument content is inferred.
The environment must still verify exact Visit labels/URLs against links the
actor actually saw and enforce the root boundary.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable
from urllib.parse import urlsplit

VERSION = 'webwalker_action_format_v3'
ParsedAction = tuple[str, str | None, str | None]
StrictParser = Callable[[str, int], ParsedAction]

_THOUGHT_PREFIXES = r'(?:Thought[ ]+\d+:[ ]*)*'
_ACTION_MARKER = re.compile(r'\bAction[ ]*\d*[ ]*:', re.I)
_BARE_ACTION = re.compile(
    r'[ ]*' + _THOUGHT_PREFIXES
    + r'(?P<action>Action[ ]*\d*[ ]*:[ ]*(?P<kind>Visit|Search|Finish)'
      r'(?:[ ]+(?P<argument>.*))?)',
    re.I,
)
_OTHER_SECTION = re.compile(
    r'^[ ]*' + _THOUGHT_PREFIXES
    + r'(?:Observation|Question|Action Input)[ ]*\d*[ ]*:', re.I | re.M,
)
_LATER_THOUGHT = re.compile(r'^[ ]*Thought[ ]*\d*[ ]*:', re.I | re.M)
_URL_SCHEME = re.compile(r'^(?:https?:|[A-Za-z][A-Za-z0-9+.-]*:/)', re.I)


def _absolute_url(value: str) -> bool:
    """Use the v2 literal URL check without removing or completing anything."""
    if (any(char.isspace() for char in value)
            or any(char in value for char in '<>"\'`{}\\|')
            or value.endswith(('.', ',', ';', ':', '!', '?', ')', '}'))):
        return False
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in ('http', 'https') or not parsed.netloc or not parsed.hostname:
            return False
        parsed.port
        if parsed.username is not None or parsed.password is not None:
            return False
        return True
    except ValueError:
        return False


def _looks_like_url(value: str) -> bool:
    """Recognize explicit URL/path syntax without guessing from label words."""
    return bool(_URL_SCHEME.match(value)) or value.lower().startswith(
        ('www.', '//', '/', './', '../', '\\'))


def parse_with_metadata(text: str, step: int, strict_parser: StrictParser
                        ) -> tuple[ParsedAction, dict]:
    """Preserve strict results; otherwise accept one unambiguous bare action.

    Visit and Search take one line. Finish takes the literal remainder of the
    response, including subsequent answer lines. Like the strict parser, only
    outer argument whitespace is stripped; case, internal spaces, quotes,
    punctuation and paragraph breaks remain unchanged. Empty arguments and
    additional action/observation/question sections are rejected. Blank
    surrounding lines and preceding thought text are allowed, including
    repeated Thought N: prefixes from the legacy loop.
    """
    strict = strict_parser(text, step)
    metadata = {
        'parser_version': VERSION,
        'syntax': 'strict' if strict[1] is not None else 'invalid',
        'normalized': False,
        'original_action_line': None,
        'original_action_text': None,
        'normalized_action': None,
    }
    if strict[1] is not None:
        return strict, metadata

    def reject(reason: str):
        return strict, {**metadata, 'rejection_reason': reason}

    if any((char != '\n' and unicodedata.category(char).startswith('C'))
           or unicodedata.category(char) in ('Zl', 'Zp') for char in text):
        return reject('control_format_or_line_separator')
    if len(_ACTION_MARKER.findall(text)) != 1:
        return reject('ambiguous_action_count')
    if '```' in text or '~~~' in text or _OTHER_SECTION.search(text):
        return reject('quoted_code_or_other_section')
    lines = text.split('\n')
    matches = [(index, found) for index, line in enumerate(lines)
               if (found := _BARE_ACTION.fullmatch(line))]
    if len(matches) != 1:
        return reject('not_one_bare_action_line')
    index, found = matches[0]
    kind = found['kind'].lower()
    tail = '\n'.join(lines[index + 1:])
    argument_text = found['argument'] or ''
    original_action_text = found['action']
    if kind == 'finish':
        if _LATER_THOUGHT.search(tail):
            return reject('additional_thought_section')
        if index + 1 < len(lines):
            argument_text += '\n' + tail
            original_action_text += '\n' + tail
    elif tail.strip():
        return reject('trailing_content')
    argument = argument_text.strip()
    if not argument:
        return reject('empty_argument')
    if kind == 'visit':
        url_shaped = _looks_like_url(argument)
        if url_shaped and not _absolute_url(argument):
            return reject('invalid_absolute_url')
        syntax = 'bare_visit_url' if url_shaped else 'bare_visit_label'
    else:
        syntax = 'bare_' + kind
    return (strict[0], kind, argument), {
        **metadata,
        'syntax': syntax,
        'normalized': True,
        'original_action_line': found['action'],
        'original_action_text': original_action_text,
        'normalized_action': f'{kind.title()}[{argument}]',
    }


def parse_action(text: str, step: int, strict_parser: StrictParser) -> ParsedAction:
    """Return the same tuple interface used by the shared episode loop."""
    return parse_with_metadata(text, step, strict_parser)[0]
