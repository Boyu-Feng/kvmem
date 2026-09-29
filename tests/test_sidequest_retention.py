import pytest

from scripts.summarize_sidequest_retention import aggregate, measure_row


def example(deleted=2):
    return dict(id='q1', method='sidequest_untrained', audit=dict(
        logical_tokens=6, final_tokens=6-deleted,
        aux_events=[dict(applied=True, collection_turn=1, before_tokens=4, after_tokens=4-deleted)],
        boundaries=[dict(turn=1, tokens=4), dict(turn=2, tokens=6-deleted)]))


def test_eviction_applies_to_next_boundary_not_previous():
    question, steps = measure_row(example(), 2)
    assert [s['trajectory_retention'] for s in steps] == [1., .5]
    assert question['final_trajectory_retention'] == .5
    assert question['step_mean_trajectory_retention'] == .75


def test_no_eviction_questions_included_and_pooling_differs_from_mean():
    pruned, _ = measure_row(example(), 2)
    unpruned = example(0)
    unpruned['id'] = 'q2'
    unpruned['audit'].update(logical_tokens=4, final_tokens=4,
                             aux_events=[], boundaries=[dict(turn=1, tokens=4)])
    full, _ = measure_row(unpruned, 2)
    result = aggregate([pruned, full])
    assert result['mean_final_trajectory_retention'] == .75
    assert result['pooled_final_trajectory_retention'] == pytest.approx(4/6)
    assert result['mean_step_trajectory_retention'] == .875
    assert result['questions_with_eviction'] == 1


def test_unused_proposal_does_not_count_as_deletion():
    row = example(0)
    row['audit']['aux_events'] = [dict(applied=False, collection_turn=2, del_cursors=[0])]
    question, _ = measure_row(row, 2)
    assert question['deleted_tokens'] == 0
    assert question['final_trajectory_retention'] == 1


def test_invalid_eviction_ledger_rejected():
    row = example()
    row['audit']['logical_tokens'] = 7
    with pytest.raises(ValueError, match='ledger mismatch'):
        measure_row(row, 2)


def test_prompt_deletion_rejected():
    row = example(3)
    with pytest.raises(ValueError, match='prompt protection'):
        measure_row(row, 2)


def test_empty_trajectory_kept_as_no_eviction():
    row = example(0)
    row['audit'].update(logical_tokens=2, final_tokens=2, aux_events=[],
                        boundaries=[dict(turn=1, tokens=2)])
    measured, _ = measure_row(row, 2)
    assert aggregate([measured])['mean_final_trajectory_retention'] == 1
