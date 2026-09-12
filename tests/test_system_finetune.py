"""Frozen CAL checkpoint selection must preserve previously correct rows."""
import copy
import json

import numpy as np
import pytest

from training.system_finetune import (MAX_SIZE_SPREAD, PARENT_GATES, PARENT_TEMPERATURE,
                                     SYSTEM_FAMILIES, accepted_mask, assess_checkpoint,
                                     selection_stats)


FAMILIES = ['PingFang', 'SF Pro', 'Helvetica', 'MiSans', 'Noto Sans CJK SC']


def calibration():
    targets = np.repeat(np.arange(len(FAMILIES)), 4)
    rows = [{'index': 101 + 3 * i} for i in range(len(targets))]
    data = {'split': 'calibration', 'rows': rows, 'targets': targets}
    obs = {'predicted': targets.copy(), 'score': np.full(len(rows), .49),
           'margin': np.full(len(rows), .1), 'agreement': np.ones(len(rows)),
           'valid_output': np.ones(len(rows), dtype=bool),
           'size_relative_error': np.full(len(rows), .1)}
    # Five accepted correct rows and four accepted errors. Other rows abstain.
    obs['score'][[0, 4, 8, 12, 16, 3, 7, 15, 19]] = .9
    obs['predicted'][[3, 7, 15, 19]] = [3, 4, 4, 3]
    return data, obs


def stats(data, obs):
    return selection_stats(obs, data, FAMILIES, PARENT_GATES)


def improved(obs, *indices):
    candidate = copy.deepcopy(obs)
    for index in indices:
        candidate['score'][index] = .9
        candidate['predicted'][index] = index // 4
    return candidate


def test_fixed_parent_policy_is_explicit():
    assert SYSTEM_FAMILIES == ('PingFang', 'SF Pro', 'Helvetica')
    assert PARENT_TEMPERATURE == .5 and MAX_SIZE_SPREAD == .2
    assert PARENT_GATES == {'min_score': .5, 'min_margin': .01, 'min_patch_agreement': .7}


def test_acceptance_includes_threshold_equality_but_masks_invalid_outputs():
    obs = {'score': np.array([.5, np.nextafter(.5, 0), .5, .5, .5, np.nan]),
           'margin': np.array([.01, .01, np.nextafter(.01, 0), .01, .01, .01]),
           'agreement': np.array([.7, .7, .7, np.nextafter(.7, 0), .7, .7]),
           'valid_output': np.array([True, True, True, True, False, True]),
           'size_spread': np.full(6, 99.)}
    np.testing.assert_array_equal(accepted_mask(obs, PARENT_GATES), [True, False, False, False, False, False])


def test_zero_gate_still_rejects_ties_and_exact_epsilon_margin():
    obs = {'score': np.zeros(5), 'margin': np.array([0., 1e-8, np.nextafter(1e-8, np.inf), -.1, .1]),
           'agreement': np.zeros(5), 'valid_output': np.array([True, True, True, True, False])}
    zero_gates = {'min_score': 0., 'min_margin': 0., 'min_patch_agreement': 0.}
    np.testing.assert_array_equal(accepted_mask(obs, zero_gates), [False, False, True, False, False])


def test_stats_use_true_and_predicted_families_and_actual_row_indices():
    data, obs = calibration()
    obs['predicted'][19] = 0  # A non-system true font wrongly enters PingFang.
    obs['correct'] = np.ones(20, dtype=bool)  # Stale cached correctness must not override labels.
    result = stats(data, obs)
    assert result['system_correct'] == 3 and result['overall_correct'] == 5
    assert result['overall_wrong'] == 4
    assert result['nonsystem_correct'] == result['nonsystem_wrong'] == 2
    assert result['systems']['PingFang'] == {'rows': 4, 'correct': 1, 'wrong_true': 1,
                                           'wrong_pred': 1, 'correct_coverage': .25}
    assert result['systems']['SF Pro']['wrong_true'] == 1
    assert result['systems']['Helvetica']['wrong_true'] == 0
    assert result['correct_rows'] == [101, 113, 125, 137, 149]
    assert result['size_mean_relative_error'] == pytest.approx(.1)
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_size_rank_uses_all_calibration_rows_including_abstentions():
    data, obs = calibration()
    obs['size_relative_error'][1] = 1.1  # Row 1 is not accepted.
    assert stats(data, obs)['size_mean_relative_error'] == pytest.approx(.15)


def test_candidate_is_eligible_only_when_existing_correct_rows_are_retained():
    data, parent_obs = calibration()
    parent = stats(data, parent_obs)
    candidate = stats(data, improved(parent_obs, 1))
    verdict = assess_checkpoint(candidate, parent)
    assert verdict['eligible'] is True and verdict['reasons'] == []
    assert verdict['rank'][:3] == [4, .25, 6]
    assert verdict['rank'][3] == pytest.approx(-.1)
    assert set(parent['correct_rows']) < set(candidate['correct_rows'])
    json.dumps(verdict, allow_nan=False)


@pytest.mark.parametrize('failure,reason', [
    ('overall', 'overall_wrong_increased'),
    ('wrong_true', 'system_wrong_true_increased:Helvetica'),
    ('wrong_pred', 'system_wrong_pred_increased:PingFang'),
    ('nonsystem_correct', 'nonsystem_correct_decreased'),
    ('nonsystem_wrong', 'nonsystem_wrong_increased'),
    ('same_family_swap', 'parent_correct_rows_lost')
])
def test_more_system_correct_results_cannot_hide_regressions(failure, reason):
    data, parent_obs = calibration()
    candidate_obs = improved(parent_obs, 1)
    if failure == 'overall':
        candidate_obs['score'][2] = .9
        candidate_obs['predicted'][2] = 3
    elif failure == 'wrong_true':
        candidate_obs = improved(candidate_obs, 3)  # Fix an old PF error.
        candidate_obs['score'][11] = .9
        candidate_obs['predicted'][11] = 3  # Replace it with a Helvetica error.
    elif failure == 'wrong_pred':
        candidate_obs['predicted'][19] = 0  # Same total errors, newly false PF.
    elif failure == 'nonsystem_correct':
        candidate_obs['score'][12] = .49
    elif failure == 'nonsystem_wrong':
        candidate_obs = improved(candidate_obs, 3)  # Hold total errors constant.
        candidate_obs['score'][14] = .9
        candidate_obs['predicted'][14] = 4
    elif failure == 'same_family_swap':
        candidate_obs['score'][0] = .49  # Lose an existing correct PF row.
        candidate_obs = improved(candidate_obs, 2)  # Gain enough PF rows to hide it in totals.
    parent, candidate = stats(data, parent_obs), stats(data, candidate_obs)
    assert candidate['system_correct'] > parent['system_correct']
    if failure in ('wrong_true', 'wrong_pred', 'nonsystem_wrong'):
        assert candidate['overall_wrong'] == parent['overall_wrong']
    if failure == 'same_family_swap':
        assert candidate['systems']['PingFang']['correct'] > parent['systems']['PingFang']['correct']
    verdict = assess_checkpoint(candidate, parent)
    assert verdict['eligible'] is False and reason in verdict['reasons']


def test_no_system_improvement_is_rejected_even_if_overall_and_size_improve():
    data, parent_obs = calibration()
    candidate_obs = improved(parent_obs, 19)
    candidate_obs['size_relative_error'][:] = .01
    parent, candidate = stats(data, parent_obs), stats(data, candidate_obs)
    assert candidate['overall_correct'] > parent['overall_correct']
    verdict = assess_checkpoint(candidate, parent)
    assert verdict == {'eligible': False, 'reasons': ['system_correct_not_improved'],
                       'rank': [3, .25, 6, -candidate['size_mean_relative_error']]}


def test_lexicographic_rank_prioritizes_system_count_then_weakest_coverage():
    data, parent_obs = calibration()
    parent = stats(data, parent_obs)
    concentrated = assess_checkpoint(stats(data, improved(parent_obs, 1, 2, 3)), parent)
    balanced = assess_checkpoint(stats(data, improved(parent_obs, 1, 5, 9)), parent)
    higher_count = assess_checkpoint(stats(data, improved(parent_obs, 1, 2, 3, 5, 6)), parent)
    assert all(item['eligible'] for item in (concentrated, balanced, higher_count))
    assert concentrated['rank'][0] == balanced['rank'][0] == 6
    assert concentrated['rank'][1] == .25 and balanced['rank'][1] == .5
    assert balanced['rank'] > concentrated['rank']
    assert higher_count['rank'][1] < balanced['rank'][1]
    assert higher_count['rank'] > balanced['rank']


def test_overall_correct_precedes_size_error_and_exact_ranks_preserve_ties():
    data, parent_obs = calibration()
    parent = stats(data, parent_obs)
    base = improved(parent_obs, 1)
    smaller_error = copy.deepcopy(base)
    smaller_error['size_relative_error'][:] = .01
    more_correct = improved(base, 13)
    more_correct['size_relative_error'][:] = .8
    rank = lambda obs: assess_checkpoint(stats(data, obs), parent)['rank']
    assert rank(more_correct) > rank(smaller_error) > rank(base)
    assert rank(copy.deepcopy(base)) == rank(base)
    assert not rank(copy.deepcopy(base)) > rank(base)


def test_empty_system_family_has_zero_coverage_without_nan():
    data = {'rows': [{'index': 7}], 'targets': np.array([0])}
    obs = {'predicted': np.array([0]), 'score': np.array([.9]), 'margin': np.array([.1]),
           'agreement': np.array([1.]), 'valid_output': np.array([True]), 'size_relative_error': np.array([.1])}
    result = selection_stats(obs, data, ['MiSans'], PARENT_GATES)
    assert result['system_correct'] == 0 and result['nonsystem_correct'] == 1
    assert all(value['rows'] == 0 and value['correct_coverage'] == 0 for value in result['systems'].values())
    json.dumps(result, allow_nan=False)


def test_selection_rejects_test_split_and_nonfinite_diagnostics():
    data, obs = calibration()
    with pytest.raises(ValueError, match='calibration split'):
        stats({**data, 'split': 'test'}, obs)
    obs['size_relative_error'][0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        stats(data, obs)
