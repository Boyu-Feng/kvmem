import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import webbench_scoring_v1 as s


@pytest.fixture
def bcp():
    return s.extract_namespace(s.BCP_SOURCE, {'GRADER_TEMPLATE', 'create_judge_prompt',
        'parse_judge_response', 'extract_citations_from_response', 'compute_citation_metrics'})


def prediction(**changes):
    return dict(id='test-1', question='What is 2 + 2?', gold_answer='4', predicted_answer='4',
        termination='finish', task_sha256='public-task-binding', retrieved_docids=['123'],
        tool_call_counts={'search': 1}, **changes)


def normalized(tmp_path, changes=None):
    row = prediction()
    row.update(changes or {})
    path = tmp_path / 'predictions.json'
    s.atomic(path, {'results': [row]})
    return path, s.load_predictions(path)[1][0]


@pytest.mark.parametrize('termination', ['step_budget', 'context_budget', 'invalid_action'])
def test_browsecomp_normal_noncompletion_is_official_failure_without_judge(tmp_path, bcp, termination):
    _, row = normalized(tmp_path, {'termination': termination})
    result = s.score_row(row, 'browsecomp_plus', bcp, {},
        judge=lambda _: pytest.fail('Incomplete official runs must not invoke judge'),
        qrels={'test-1': ['123', '456']})
    assert result['score'] == 0
    assert result['scientific_failure'] is True
    assert result['judge_result']['parse_error'] is True
    assert result['retrieval_recall'] == 0.5


def test_browsecomp_empty_finish_and_unparseable_judge_are_distinct(tmp_path, bcp):
    _, row = normalized(tmp_path, {'predicted_answer': ''})
    assert s.score_row(row, 'browsecomp_plus', bcp, {}, judge=None)['score'] == 0
    _, row = normalized(tmp_path)
    with pytest.raises(RuntimeError, match='score remains missing'):
        s.score_row(row, 'browsecomp_plus', bcp, {}, judge=lambda _: ('undecidable', {}))


def test_browsecomp_uses_original_template_and_parser(tmp_path, bcp):
    _, row = normalized(tmp_path, {'predicted_answer': '4 [123]'})
    prompts = []
    def judge(prompt):
        prompts.append(prompt)
        return ('extracted_final_answer: 4\nreasoning: equivalent\ncorrect: yes\nconfidence: 75', {})
    result = s.score_row(row, 'browsecomp_plus', bcp, {}, judge=judge, qrels={'test-1': ['123']})
    assert prompts == [bcp['GRADER_TEMPLATE'].format(question=row['question'],
                       response='4 [123]', correct_answer='4')]
    assert result['score'] == 1 and result['judge_result']['confidence'] == 75
    assert result['citations']['metrics']['precision'] == 1


def test_content_binding_and_invalid_prediction_rejected(tmp_path):
    path, row = normalized(tmp_path)
    changed = json.loads(path.read_text())
    changed['results'][0]['question'] = 'Different public content with the same ID'
    s.atomic(path, changed)
    assert s.load_predictions(path)[1][0]['source_row_sha256'] != row['source_row_sha256']
    changed['results'].append(changed['results'][0])
    s.atomic(path, changed)
    with pytest.raises(ValueError, match='Duplicate'):
        s.load_predictions(path)
    changed['results'] = [dict(changed['results'][0], termination='cuda_oom')]
    s.atomic(path, changed)
    with pytest.raises(ValueError, match='cannot be scored'):
        s.load_predictions(path)


@pytest.mark.parametrize('model', ['gpt-4.1-2025-04-14', 'gpt-4o-2024-11-20', None])
def test_proxy_alias_cannot_substitute_another_judge(monkeypatch, model):
    response = {'model': model, 'choices': [{'finish_reason': 'stop', 'message': {'content': 'GRADE: CORRECT'}}]}
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return json.dumps(response).encode()
    monkeypatch.setattr(s.urllib.request, 'urlopen', lambda *a, **kw: Response())
    with pytest.raises(s.JudgeFailure) as exc:
        s.webwalker_request('synthetic-only')
    assert exc.value.evidence == response


@pytest.fixture
def qualified(tmp_path, monkeypatch):
    if not (s.DEFAULT_ASSETS / 'sources/manifest.json').exists():
        pytest.skip('Run CPU-only scorer prepare once to fetch pinned public sources')
    bcp, lc, source = s.official(s.DEFAULT_ASSETS)
    monkeypatch.setattr(s, 'official', lambda _: (bcp, lc, source))
    assets = tmp_path / 'assets'
    policy = s.webwalker_policy(assets)
    s.atomic(assets / 'webwalker_probe.json', dict(benchmark_score=False, synthetic=True,
        parsed={'score': 1}, sampling=policy['sampling'], policy_sha256=s.digest(policy),
        source_manifest_sha256=s.digest(source),
        returned_model='gpt-4-0613', endpoint='http://localhost:8313/v1'))
    path, _ = normalized(tmp_path)
    args = SimpleNamespace(assets=assets, output=tmp_path / 'scoring', predictions=path,
        benchmark='webwalkerqa', endpoint='http://localhost:8313/v1', qrels=None,
        gpu_memory_utilization=.6, max_model_len=16384)
    return args, lc


def test_cotqa_template_and_official_parsing(qualified):
    _, lc = qualified
    assert 'Write out in a step by step manner' in lc['cot_template']
    assert lc['_parse_string_eval_output']('Reasoning.\nGRADE: INCORRECT')['score'] == 0
    assert lc['_parse_string_eval_output']('Reasoning. CORRECT.')['score'] == 1
    assert lc['_parse_string_eval_output']('ambiguous')['score'] is None


def test_failure_durable_then_resume_without_rescoring_completed(qualified, monkeypatch):
    args, _ = qualified
    def unavailable(*args):
        raise OSError('Synthetic transport failure')
    monkeypatch.setattr(s, 'webwalker_request', unavailable)
    with pytest.raises(OSError, match='transport failure'):
        s.score(args)
    output = Path(args.output)
    assert s.read(output / 'status.json')['scored'] == 0
    assert not (output / 'scores.json').exists()
    assert list((output / 'failures').glob('*.json'))
    assert s.read(next((output / 'calls').glob('*.json')))['status'] == 'failed'
    monkeypatch.setattr(s, 'webwalker_request', lambda *a: ('GRADE: CORRECT', {'model': 'gpt-4-0613'}))
    assert s.score(args)['accuracy'] == 1
    monkeypatch.setattr(s, 'webwalker_request', lambda *a: pytest.fail('Completed item was rescored'))
    assert s.score(args)['correct'] == 1
    data = s.read(args.predictions)
    data['results'][0]['predicted_answer'] = '5'
    s.atomic(args.predictions, data)
    with pytest.raises(RuntimeError, match='resume contract differs'):
        s.score(args)


def test_returned_parse_failure_is_reparsed_without_sampling_another_judgement(qualified, monkeypatch):
    args, _ = qualified
    monkeypatch.setattr(s, 'webwalker_request', lambda *a: ('ambiguous', {'model': 'gpt-4-0613'}))
    with pytest.raises(RuntimeError, match='score remains missing'):
        s.score(args)
    monkeypatch.setattr(s, 'webwalker_request', lambda *a: pytest.fail('Returned judgement was sampled again'))
    with pytest.raises(RuntimeError, match='score remains missing'):
        s.score(args)
    output = Path(args.output)
    assert len(list((output / 'calls').glob('*.json'))) == 1
    assert s.read(next((output / 'calls').glob('*.json')))['text'] == 'ambiguous'


def test_qualified_api_identity_cannot_drift(qualified, monkeypatch):
    args, _ = qualified
    monkeypatch.setattr(s, 'webwalker_request', lambda *a: ('GRADE: CORRECT', {'model': 'gpt-4-0314'}))
    with pytest.raises(s.JudgeFailure, match='changed'):
        s.score(args)
    assert s.read(Path(args.output) / 'status.json')['scored'] == 0


def test_environment_failure_stays_missing_and_other_rows_are_scored(qualified, monkeypatch):
    args, _ = qualified
    value = s.read(args.predictions)
    failed = dict(value['results'][0], id='blocked-root', status='environment_failure',
                  predicted_answer=None, inference_started=False, termination='environment_failure',
                  error='Public root URL unreachable')
    value['results'].insert(0, failed)
    s.atomic(args.predictions, value)
    calls = []
    def judge(prompt, endpoint, *args):
        calls.append(prompt)
        return 'GRADE: CORRECT', {'model': 'gpt-4-0613'}
    monkeypatch.setattr(s, 'webwalker_request', judge)
    result = s.score(args)
    assert len(calls) == 1
    assert result['requested'] == 2 and result['inferred'] == 1 and result['scored'] == 1
    assert result['environment_failed'] == 1 and result['accuracy'] is None
    assert result['status'] == 'incomplete_environment_failures'
    assert result['results'][0]['score'] is None


def test_changed_judge_requires_explicit_bound_user_authorization(tmp_path):
    policy = s.webwalker_policy(tmp_path)
    assert 'gpt-4.1-2025-04-14' not in policy['accepted_models']
    path = tmp_path / 'webwalker_authorization.json'
    s.atomic(path, {'authorized': True})
    with pytest.raises(RuntimeError, match='authorization manifest'):
        s.webwalker_policy(tmp_path)
    authorization = dict(schema='webwalker_judge_authorization_v1', benchmark='webwalkerqa',
        authorized=True, authorized_by='user', user_instruction='Synthetic test approval only',
        approved_at_utc='2026-10-21T00:00:00Z', requested_model='gpt-4.1',
        expected_api_model='gpt-4.1-2025-04-14')
    s.atomic(path, authorization)
    policy = s.webwalker_policy(tmp_path)
    assert policy['label'] == 'user_authorized_gpt41_changed_judge'
    assert policy['authorization'] == authorization
    assert policy['sampling']['model'] == 'gpt-4.1'
    assert policy['accepted_models'] == ['gpt-4.1-2025-04-14']
