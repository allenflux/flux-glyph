"""CAL selection counts must reproduce the serving consensus rules."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

import flux_glyph.region_font as runtime
from training.train_sans_verifier import POLICY, observations, summarize

PRIMARY = ['HarmonyOS Sans SC', 'MiSans', 'Noto Sans CJK SC', 'OPPO Sans',
           'PingFang', 'SF Pro', 'Helvetica', 'Alipay Number']
VERIFIER = ['Roboto', 'Helvetica', 'PingFang', 'Alipay Number', 'SF Pro',
            'OPPO Sans', 'MiSans', 'Noto Sans CJK SC', 'HarmonyOS Sans SC']


def observation(family, families, passed=True, known=True):
    return {'predicted': families.index(family), 'passed': passed, 'known': known}


def row(index, family, domain='anchor', view='native'):
    return {'family': family, 'domain': domain, 'view': view, 'source_id': 'source-'+str(index),
            'region_id': 'R001', 'tile_start': index, 'tile_count': 1}


def test_observations_average_tile_probabilities_using_own_temperature():
    logits = np.array([[99., -99.], [0., 8.], [2., 0.], [-99., 99.]], dtype=np.float32)
    rows = [{'tile_start': 1, 'tile_count': 2}]
    original = logits.copy()
    scores = observations(logits, rows, 2., {'min_score': .5, 'min_margin': .01, 'min_patch_agreement': .7})
    values = np.exp(logits[1:3].astype(np.float64)/2)
    expected = (values/values.sum(axis=1, keepdims=True)).mean(axis=0)
    np.testing.assert_allclose(scores[0]['probabilities'], expected)
    assert scores[0]['predicted'] == 1 and scores[0]['agreement'] == .5
    assert not scores[0]['passed']
    np.testing.assert_array_equal(logits, original)


def test_exact_score_margin_agreement_boundaries_match_runtime_and_ties_abstain():
    logits = np.array([[3., 0.]], dtype=np.float32)
    rows = [{'tile_start': 0, 'tile_count': 1}]
    initial = observations(logits, rows, 1., {'min_score': 0., 'min_margin': 0., 'min_patch_agreement': 0.})[0]
    gates = {'min_score': initial['score'], 'min_margin': initial['margin'], 'min_patch_agreement': 1.}
    assert observations(logits, rows, 1., gates)[0]['passed']
    for key in ('min_score', 'min_margin'):
        changed = {**gates, key: np.nextafter(gates[key], np.inf)}
        assert not observations(logits, rows, 1., changed)[0]['passed']
    tied = observations(np.zeros((1, 2)), rows, 1., {'min_score': 0., 'min_margin': 0., 'min_patch_agreement': 0.})[0]
    assert tied['predicted'] == 0 and tied['margin'] == 0 and not tied['passed']


def test_summarize_projects_nine_classes_by_name_and_cannot_upgrade_unaccepted_primary():
    truths = ['SF Pro', 'SF Pro', 'SF Pro', 'SF Pro', 'Roboto', 'Helvetica', 'PingFang']
    rows = [row(i, family, 'anchor' if i != 4 else 'new_native') for i, family in enumerate(truths)]
    baseline = [observation('SF Pro', PRIMARY) for _ in rows]
    baseline[1]['passed'] = False
    baseline[2]['known'] = False
    checks = [observation(family, VERIFIER) for family in ['SF Pro', 'SF Pro', 'SF Pro', 'SF Pro', 'Roboto', 'SF Pro', 'Helvetica']]
    checks[3]['passed'] = False
    _, details = summarize(rows, baseline, checks, VERIFIER, PRIMARY)
    assert [item['after_correct'] for item in details] == [True, False, False, False, False, False, False]
    assert [item['after_wrong'] for item in details] == [False, False, False, False, False, True, False]
    assert details[4]['roboto_veto'] and details[4]['verifier_correct']
    assert all(not item['after_wrong'] or item['before_wrong'] for item in details)
    assert all(not item['after_correct'] or item['before_correct'] for item in details)


def test_cal_accepted_correct_wrong_counts_match_actual_runtime_for_all_gate_combinations(monkeypatch):
    cases = [('PingFang', 'PingFang', 'PingFang', True, True, True),
             ('SF Pro', 'SF Pro', 'Helvetica', True, True, True),
             ('SF Pro', 'SF Pro', 'SF Pro', False, True, True),
             ('SF Pro', 'SF Pro', 'SF Pro', True, False, True),
             ('SF Pro', 'SF Pro', 'SF Pro', True, True, False),
             ('Helvetica', 'SF Pro', 'SF Pro', True, True, True),
             ('Roboto', 'SF Pro', 'Roboto', True, True, True),
             ('Roboto', 'SF Pro', 'Roboto', True, True, False)]
    rows, before, checks, expected = [], [], [], []
    gates = {'min_score': .5, 'min_margin': .01, 'min_patch_agreement': .7}
    monkeypatch.setattr(runtime, 'preprocess_region', lambda _:
        {'status': 'ok', 'reason': 'region_pixels', 'tiles': np.zeros((1, 1, 64, 256), dtype=np.float32),
         'tile_count': 1, 'whole_width_covered': True, 'ink_height_px': 20})
    for i, (truth, main, verified, known, main_pass, verifier_pass) in enumerate(cases):
        first = np.zeros((1, len(PRIMARY)), dtype=np.float32)
        first[0, PRIMARY.index(main)] = 8. if main_pass else .1
        second = np.zeros((1, len(VERIFIER)), dtype=np.float32)
        second[0, VERIFIER.index(verified)] = 8. if verifier_pass else .1
        model = runtime.RegionFontClassifier.__new__(runtime.RegionFontClassifier)
        model.families, model.font_label_groups = PRIMARY, {}
        model.meta = {'temperature': 1., 'gates': gates, 'max_size_relative_spread': .2}
        model.rejection_meta = {'temperature': 1., 'min_known_score': .8}
        model.verifier_meta = {'families': VERIFIER, 'temperature': 1., 'gates': gates}
        model.session = SimpleNamespace(run=lambda *args: [first, np.zeros(1, dtype=np.float32)])
        model.verifier_session = SimpleNamespace(run=lambda *args: [second, np.zeros(1, dtype=np.float32)])
        model.rejection_session = SimpleNamespace(run=lambda *args:
            [np.array([[0., 8.]] if known else [[8., 0.]], dtype=np.float32)])
        outcome = model.predict(None)
        expected.append(outcome)
        entry = row(i, truth)
        rows.append(entry)
        value = observations(first, [{'tile_start': 0, 'tile_count': 1}], 1., gates)[0]
        value['known'] = known
        before.append(value)
        checks.extend(observations(second, [{'tile_start': 0, 'tile_count': 1}], 1., gates))
    _, details = summarize(rows, before, checks, VERIFIER, PRIMARY)
    for truth, detail, outcome in zip(rows, details, expected):
        assert bool(detail['after_correct']) == (outcome['status'] == 'candidate' and outcome['family'] == truth['family'])
        assert bool(detail['after_wrong']) == (outcome['status'] == 'candidate' and outcome['family'] != truth['family'])
    assert expected[6]['reason_code'] == 'verifier_font_out_of_scope'
    assert expected[7]['reason_code'] == 'neural_model_disagreement'


def eligible_case():
    rows = [row(i, 'PingFang') for i in range(100)]
    rows += [row(i+100, 'SF Pro', view='half') for i in range(100)]
    rows += [row(i+200, 'Roboto', domain='new_native') for i in range(10)]
    baseline = [observation(item['family'] if item['family'] in PRIMARY else 'SF Pro', PRIMARY) for item in rows]
    verifier = [observation(item['family'], VERIFIER) for item in rows]
    # Exact boundary losses: 2% native, 15% degraded, 80% Roboto veto recall.
    for i in [0, 1, *range(100, 115)]: verifier[i]['passed'] = False
    for i in (208, 209): verifier[i] = observation('SF Pro', VERIFIER)
    return rows, baseline, verifier


def test_fixed_calibration_retention_and_roboto_boundaries_are_inclusive():
    rows, baseline, verifier = eligible_case()
    metrics, _ = summarize(rows, baseline, verifier, VERIFIER, PRIMARY)
    assert metrics['passed'] and all(metrics['checks'].values())
    assert metrics['native_anchor']['correct_retention'] == .98
    assert metrics['degraded']['correct_retention'] == .85
    assert metrics['roboto_veto_recall'] == .8
    assert metrics['wrong_name_reduction'] == .8


@pytest.mark.parametrize('fault,check', [('native', 'native_anchor_retention'),
    ('degraded', 'degraded_retention'), ('roboto', 'roboto_veto'), ('system', 'native_system_retention')])
def test_eligibility_rejects_each_retention_regression(fault, check):
    rows, baseline, verifier = eligible_case()
    if fault == 'native': verifier[2]['passed'] = False
    elif fault == 'degraded': verifier[115]['passed'] = False
    elif fault == 'roboto': verifier[207] = observation('SF Pro', VERIFIER)
    else:
        # Losing a small system group must not hide behind overall retention.
        for i in range(2, 10):
            rows[i]['family'] = 'Helvetica'
            baseline[i] = observation('Helvetica', PRIMARY)
            verifier[i] = observation('Helvetica', VERIFIER)
        verifier[2]['passed'] = False
        verifier[0]['passed'] = True
    metrics, _ = summarize(rows, baseline, verifier, VERIFIER, PRIMARY)
    assert not metrics['passed'] and metrics['checks'][check] is False


def test_missing_anchor_or_misaligned_observations_cannot_be_scored():
    rows, baseline, verifier = eligible_case()
    with pytest.raises(ValueError, match='unaligned'): summarize(rows, baseline[:-1], verifier, VERIFIER, PRIMARY)
    for entry in rows: entry['domain'] = 'new_native'
    with pytest.raises(ValueError, match='anchor'): summarize(rows, baseline, verifier, VERIFIER, PRIMARY)


def test_policy_selects_from_calibration_only():
    assert POLICY['test_used_for_selection'] is False
    assert POLICY['rank'][-1] == 'earliest_step'
