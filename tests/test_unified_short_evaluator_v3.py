"""Dual-baseline DEV policy and final preview gate for short-region continuation."""
from copy import deepcopy

import pytest

from training import evaluate_unified_short_regions_v3 as evaluator


def population(precision=.95, coverage=.70, withholding=.85):
    return {'named_precision': precision, 'known_correct_coverage': coverage,
            'unknown_not_named_rate': withholding,
            'size': {'median_ape': .05, 'p90_ape': .15, 'coverage_of_correct_names': .8}}


def metrics():
    result = population()
    result['per_domain'] = {'ios': population(.96, .75, .86),
                            'android': population(.94, .65, .84)}
    result['per_family'] = {}
    return result


def test_comparison_has_exact_18_checks_and_explicit_baseline_version():
    current = metrics(); baseline = deepcopy(current)
    for version in ('r21-unified-font-v1-preview', 'r22-unified-font-retention-v1-preview'):
        result = evaluator.comparison(current, baseline, version)
        assert result['passed'] and len(result['checks']) == 18
        assert result['policy']['baseline_version'] == version
        assert result['policy']['is_blind_test'] is False
    with pytest.raises(ValueError):
        evaluator.comparison(current, baseline, 'misleading-r21-name')


@pytest.mark.parametrize('population_name,metric', [
    ('all', 'named_precision'), ('ios', 'known_correct_coverage'),
    ('android', 'unknown_not_named_rate'), ('all', 'median_ape'),
    ('ios', 'p90_ape'), ('android', 'coverage_of_correct_names')])
def test_each_metric_family_can_fail_development_comparison(population_name, metric):
    current = metrics(); baseline = metrics()
    target = current if population_name == 'all' else current['per_domain'][population_name]
    if metric == 'named_precision': target[metric] -= .006
    elif metric == 'known_correct_coverage': target[metric] -= .011
    elif metric == 'unknown_not_named_rate': target[metric] -= .021
    elif metric == 'median_ape': target['size'][metric] = .101
    elif metric == 'p90_ape': target['size'][metric] = .251
    else: target['size'][metric] = .699
    result = evaluator.comparison(current, baseline, 'r22-unified-font-retention-v1-preview')
    assert not result['passed']
    assert any(row['population'] == population_name and row['metric'].endswith(metric)
               and not row['passed'] for row in result['checks'])


def report_fixture(r21_passed=True, r22_passed=True):
    measured = metrics()
    def compared(version, passed):
        rows = [{'population': 'all', 'metric': f'metric-{index}', 'passed': passed or index > 0}
                for index in range(18)]
        return {'passed': passed, 'checks': rows,
                'policy': {**evaluator.POLICY, 'baseline_version': version}}
    selection = {'calibration_promotion_allowed': True, 'promotion_allowed': False,
                 'development_evaluated': False}
    parity = {'passed': True, 'calibration_promotion_allowed': True,
              'promotion_allowed': False, 'development_evaluated': False,
              'calibration_checks_passed': 53, 'model_sha256': 'a' * 64,
              'selection_sha256': 'b' * 64, 'fixed_runtime': {'temperature': 1., 'gates': {},
                                                               'max_size_relative_spread': .2}}
    bindings = {str(evaluator.R21_BASELINE.resolve()): 'c' * 64,
                str(evaluator.R22_BASELINE.resolve()): 'd' * 64}
    report = evaluator.build_development_report(selection, parity, measured, {}, {}, {}, [],
        compared('r21-unified-font-v1-preview', r21_passed),
        compared('r22-unified-font-retention-v1-preview', r22_passed),
        bindings, 'e' * 64, 'f' * 64)
    return report


def test_final_preview_requires_both_18_check_comparisons_and_cal53_parity():
    report = report_fixture()
    assert report['promotion_allowed'] is True
    assert report['development_checks_required'] == report['development_checks_passed'] == 36
    assert report['proof_counts'] == {'calibration_regions': 18672, 'calibration_checks': 53,
        'development_views': 3504, 'development_tiles': 5967,
        'r21_development_checks': 18, 'r22_development_checks': 18,
        'development_checks': 36, 'deployed_cnns': 1}
    assert report['source_version_summaries']['r22_baseline']['version'] == 'r22-unified-font-retention-v1-preview'
    assert report['source_version_summaries']['candidate']['version'] == 'r22-short-region-continuation-v3'
    assert report['schema'] == 'flux-glyph-unified-short-development-regression-v3'
    assert report['release_tier'] == 'preview'
    assert report['stable_validation_passed'] is report['test_passed'] is False
    assert report['test_read'] is False and report['blind_test_performed'] is False
    assert report_fixture(r21_passed=False)['promotion_allowed'] is False
    assert report_fixture(r22_passed=False)['promotion_allowed'] is False


@pytest.mark.parametrize('side,key,value', [
    ('selection', 'calibration_promotion_allowed', False),
    ('selection', 'promotion_allowed', True),
    ('selection', 'development_evaluated', True),
    ('parity', 'passed', False),
    ('parity', 'calibration_promotion_allowed', False),
    ('parity', 'promotion_allowed', True),
    ('parity', 'development_evaluated', True),
    ('parity', 'calibration_checks_passed', 52),
])
def test_report_builder_rejects_boundary_drift(side, key, value):
    measured = metrics()
    selection = {'calibration_promotion_allowed': True, 'promotion_allowed': False,
                 'development_evaluated': False}
    parity = {'passed': True, 'calibration_promotion_allowed': True,
              'promotion_allowed': False, 'development_evaluated': False,
              'calibration_checks_passed': 53, 'model_sha256': 'a',
              'selection_sha256': 'b', 'fixed_runtime': {}}
    locals()[side][key] = value
    def compared(version):
        return {'passed': True, 'checks': [{'passed': True}] * 18,
                'policy': {**evaluator.POLICY, 'baseline_version': version}}
    bindings = {str(evaluator.R21_BASELINE.resolve()): 'c',
                str(evaluator.R22_BASELINE.resolve()): 'd'}
    with pytest.raises(ValueError):
        evaluator.build_development_report(selection, parity, measured, {}, {}, {}, [],
            compared('r21-unified-font-v1-preview'),
            compared('r22-unified-font-retention-v1-preview'), bindings, 'e', 'f')


def test_malformed_or_misnamed_secondary_policy_is_rejected():
    measured = metrics()
    selection = {'calibration_promotion_allowed': True, 'promotion_allowed': False,
                 'development_evaluated': False}
    parity = {'passed': True, 'calibration_promotion_allowed': True,
              'promotion_allowed': False, 'development_evaluated': False,
              'calibration_checks_passed': 53, 'model_sha256': 'a',
              'selection_sha256': 'b', 'fixed_runtime': {}}
    good = {'passed': True, 'checks': [{'passed': True}] * 18,
            'policy': {**evaluator.POLICY, 'baseline_version': 'r21-unified-font-v1-preview'}}
    bad = deepcopy(good); bad['policy']['baseline_version'] = 'r21-unified-font-v1-preview'
    bindings = {str(evaluator.R21_BASELINE.resolve()): 'c', str(evaluator.R22_BASELINE.resolve()): 'd'}
    with pytest.raises(ValueError):
        evaluator.build_development_report(selection, parity, measured, {}, {}, {}, [], good, bad,
                                           bindings, 'e', 'f')
