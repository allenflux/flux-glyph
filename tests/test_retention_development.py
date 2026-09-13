import copy
import pytest

from evaluate_unified_retention import comparison


def sample():
    value = {'named_precision': .97, 'known_correct_coverage': .65, 'unknown_not_named_rate': .95,
             'size': {'median_ape': .03, 'p90_ape': .09, 'coverage_of_correct_names': 1.}}
    return {**copy.deepcopy(value), 'per_domain': {'ios': copy.deepcopy(value), 'android': copy.deepcopy(value)}}


def test_development_comparison_does_not_mask_android_regression_with_global_improvement():
    before = sample(); after = sample()
    after['named_precision'] = .99
    after['known_correct_coverage'] = .80
    after['per_domain']['android']['known_correct_coverage'] = .60
    result = comparison(after, before)
    assert not result['passed']
    failed = [row for row in result['checks'] if not row['passed']]
    assert [(row['population'], row['metric']) for row in failed] == [('android', 'known_correct_coverage')]


@pytest.mark.parametrize('missing', ['named_precision', 'known_correct_coverage', 'unknown_not_named_rate'])
def test_missing_metrics_fail_instead_of_counting_abstention_as_success(missing):
    after = sample(); after['per_domain']['ios'][missing] = None
    assert not comparison(after, sample())['passed']


def test_size_withholding_and_large_errors_remain_release_failures():
    after = sample()
    after['per_domain']['android']['size']['coverage_of_correct_names'] = .2
    after['size']['p90_ape'] = .3
    assert not comparison(after, sample())['passed']


def test_same_baseline_passes_but_does_not_claim_blind_evaluation():
    result = comparison(sample(), sample())
    assert result['passed']
    assert result['policy']['is_blind_test'] is False
    assert result['policy']['used_to_select_training_checkpoint'] is False
