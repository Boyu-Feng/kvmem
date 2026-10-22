"""Shared selections and content-binding failures, without model inference."""
import base64
import hashlib
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from scripts import webbench_assets_v1 as assets


def rows():
    output = []
    for i in range(assets.QUERY_COUNT):
        row = {'id': str(i), 'question': f'public question {i}', 'answer': f'answer {i}',
               'source_index': i, 'gold_docids': [str(i)],
               'evidence_docids': [str(i)], 'negative_docids': []}
        row['input_content_sha256'] = assets.digest(assets.input_content(row))
        row['scoring_content_sha256'] = assets.digest(assets.scoring_content(row))
        output.append(row)
    return output


def test_selections_share_exact_shuffle_and_global_smoke_holdout():
    data = rows()
    selections = assets.make_selections(data)['selections']
    main_union = {row['id'] for values in selections.values() for row in values['main']}
    for seed in assets.SEEDS:
        expected = list(range(len(data)))
        random.Random(seed).shuffle(expected)
        value = selections[str(seed)]
        assert [row['index'] for row in value['main']] == expected[:100]
        assert len(value['smoke']) == 2
        assert not {row['id'] for row in value['smoke']} & main_union
        assert [row['index'] for row in value['smoke']] == [
            index for index in expected[100:] if str(index) not in main_union][:2]


def test_official_decryption_is_strict_and_unicode_safe():
    text = 'A small nonbenchmark string 中文.'
    key = hashlib.sha256(assets.CANARY.encode()).digest()
    encrypted = bytes(value ^ key[i % len(key)] for i, value in enumerate(text.encode()))
    assert assets.decrypt_string(base64.b64encode(encrypted).decode()) == text
    with pytest.raises(Exception):
        assets.decrypt_string('not valid base64!!!')


def manifest(tmp_path):
    data = rows()
    query = tmp_path / 'query.jsonl'
    assets.write_jsonl(query, data)
    selections = assets.make_selections(data)
    selection_path = tmp_path / 'selections.json'
    assets.atomic_json(selection_path, selections)
    return {'query_path': str(query), 'selection_path': str(selection_path),
            'selection_sha256': assets.digest(selections),
            'files': {str(query): {'sha256': assets.file_hash(query)}}}, data


def test_input_only_helper_excludes_answers_and_relevance():
    assert set(assets.input_content(rows()[0])) == {'id', 'question'}


def test_selection_order_tampering_is_rejected(tmp_path):
    value, data = manifest(tmp_path)
    assert len(assets.select_rows(value, 233)) == 100
    selection = assets.read_json(value['selection_path'])
    selection['selections']['233']['main'].reverse()
    assets.atomic_json(value['selection_path'], selection)
    value['selection_sha256'] = assets.digest(selection)
    with pytest.raises(RuntimeError, match='selection changed'):
        assets.select_rows(value, 233)


@pytest.mark.parametrize('field', ['question', 'answer'])
def test_same_id_changed_content_rejected_even_with_new_file_hash(tmp_path, field):
    value, data = manifest(tmp_path)
    data[0][field] = 'replacement content'
    assets.write_jsonl(Path(value['query_path']), data)
    value['files'][value['query_path']]['sha256'] = assets.file_hash(value['query_path'])
    with pytest.raises(RuntimeError, match='content binding mismatch'):
        assets.load_rows(value)


def test_invalid_seed_and_purpose_rejected(tmp_path):
    value, _ = manifest(tmp_path)
    with pytest.raises(ValueError):
        assets.select_rows(value, 1)
    with pytest.raises(ValueError):
        assets.select_rows(value, 233, 'validation')


def test_browse_environment_restricts_lookup_and_shares_document_limit(monkeypatch):
    from scripts.webbench_v1 import BrowseEnvironment

    class Retriever:
        def __init__(self, manifest):
            self.looked_up = []

        def search(self, query, k):
            assert k == 5
            return [{'docid': str(i), 'text': '0123456789', 'score': 1.0} for i in range(k)]

        def get_document(self, docid):
            self.looked_up.append(docid)
            return {'docid': docid, 'text': '0123456789'}

        def close(self):
            pass

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=list(text))

        def decode(self, tokens, **kwargs):
            return ''.join(tokens)

    monkeypatch.setattr(assets, 'BrowseCompRetriever', Retriever)
    environment = BrowseEnvironment({}, Tokenizer(), {'search_top_k': 5, 'document_tokens': 4})
    assert 'Invalid action' in environment.step('lookup', '0')
    assert 'Invalid action' in environment.step('search', '')
    assert 'Invalid action' in environment.step('visit', '0')
    assert environment.retriever.looked_up == []
    text = environment.step('search', 'public query')
    assert len(environment.retrieved) == 5
    assert all(line.endswith('0123') for line in text.split('\n\n'))
    assert environment.step('lookup', '0') == '0123'
    assert environment.retriever.looked_up == ['0']
    environment.reset({'question': 'next question'})
    assert 'Invalid action' in environment.step('lookup', '0')
