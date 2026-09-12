"""Parity acceptance rules, independent of expensive model training."""
import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location('neural_parity', Path(__file__).parents[1] / 'scripts/check_neural_parity.py')
parity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parity)

META = {'families': ['PingFang SC', 'Noto Sans CJK SC', 'SF Pro', 'Helvetica'],
        'scripts': {'han': ['PingFang SC', 'Noto Sans CJK SC'], 'latin': ['SF Pro', 'Helvetica']},
        'temperature': {'han': 1., 'latin': 1.},
        'gates': {'han': {'min_score': .6, 'min_margin': .1}, 'latin': {'min_score': .6, 'min_margin': .1}}}
ROWS = [{'array_row': 10, 'character': '中', 'script': 'han', 'family': 'PingFang SC'},
        {'array_row': 25, 'character': 'A', 'script': 'latin', 'family': 'SF Pro'}]
LOGITS = np.asarray([[2., 0., -1., -2.], [-1., -2., 2., 0.]], dtype=np.float32)
DECISIONS = [{'top1': family, 'reason_code': 'neural_family_candidate', 'status': 'candidate',
              'family': family, 'score': .88, 'margin': .76} for family in ['PingFang SC', 'SF Pro']]


def test_fixed_sample_is_balanced_and_has_no_replacement():
    rows = [{'script': script, 'family': family} for script, names in META['scripts'].items()
            for family in names for _ in range(100)]
    first = parity.sample_indices(rows)
    assert first == parity.sample_indices(rows)
    assert len(first) == len(set(first)) == 128
    assert sum(rows[i]['script'] == 'han' for i in first) == 64
    assert all(sum(rows[i]['family'] == family for i in first) == 32 for family in META['families'])


def test_common_logit_offset_is_not_a_false_failure():
    result = parity.compare_outputs(LOGITS, LOGITS + 5, META, ROWS, DECISIONS, DECISIONS)
    assert result['passed']
    assert result['raw_logits']['max_abs'] == 5
    assert all(item['probabilities']['max_abs'] == 0 for item in result['scripts'].values())


def test_gate_decision_mismatch_fails_even_with_identical_logits():
    changed = [dict(row) for row in DECISIONS]
    changed[0].update(status='uncertain', reason_code='below_score_gate', family=None)
    result = parity.compare_outputs(LOGITS, LOGITS, META, ROWS, DECISIONS, changed)
    assert not result['passed']
    assert result['scripts']['han']['frozen_gate_decision_mismatches'] == 1


def test_top1_mismatch_fails_even_when_both_abstain():
    before = [dict(row, status='uncertain', reason_code='below_score_gate', family=None) for row in DECISIONS]
    after = [dict(row) for row in before]
    after[1]['top1'] = 'Helvetica'
    result = parity.compare_outputs(LOGITS, LOGITS, META, ROWS, before, after)
    assert not result['passed']
    assert result['scripts']['latin']['top1_mismatches'] == 1


def test_probability_drift_fails_with_unchanged_ranking_and_gate():
    changed = LOGITS.copy()
    changed[0, 0] += .01
    result = parity.compare_outputs(LOGITS, changed, META, ROWS, DECISIONS, DECISIONS)
    assert not result['passed']
    assert result['scripts']['han']['probabilities']['max_abs'] > 1e-4
