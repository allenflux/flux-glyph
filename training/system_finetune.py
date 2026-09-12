"""Select checkpoints on fixed calibration rows without changing acceptance gates.

This module consumes NumPy observations only. It does not load data, run a
network, inspect the test split, or fit temperatures and thresholds.
"""
from __future__ import annotations

import numpy as np


SYSTEM_FAMILIES = ('PingFang', 'SF Pro', 'Helvetica')
PARENT_TEMPERATURE = .5
PARENT_GATES = {'min_score': .5, 'min_margin': .01, 'min_patch_agreement': .7}
MAX_SIZE_SPREAD = .2


def accepted_mask(obs, gates):
    """Match runtime acceptance, including strict rejection of tied families."""
    valid = np.asarray(obs['valid_output'], dtype=bool)
    score = np.asarray(obs['score'])
    margin = np.asarray(obs['margin'])
    agreement = np.asarray(obs['agreement'])
    if valid.ndim != 1 or any(value.shape != valid.shape for value in (score, margin, agreement)):
        raise ValueError('Calibration observations must be aligned one-dimensional arrays')
    return (valid & (score >= gates['min_score']) & (margin >= gates['min_margin'])
            & (margin > 1e-8) & (agreement >= gates['min_patch_agreement']))


def selection_stats(obs, data, families, gates):
    """Count accepted results by true family and predicted system family.

    Coverage divides accepted correct rows by all rows of that true family.
    Row identities retain metadata indices, even for non-contiguous subsets.
    Size error uses every calibration row, as in the existing training metrics;
    rejecting font predictions must not improve this size ranking artificially.
    """
    if data.get('split', 'calibration') != 'calibration':
        raise ValueError('Checkpoint selection requires the calibration split')
    accepted = accepted_mask(obs, gates)
    rows = data['rows']
    targets = np.asarray(data['targets'])
    predicted = np.asarray(obs['predicted'])
    size_error = np.asarray(obs['size_relative_error'], dtype=np.float64)
    if not rows or any(value.shape != (len(rows),) for value in (accepted, targets, predicted, size_error)):
        raise ValueError('Calibration rows and observations must be nonempty and aligned')
    if not np.isfinite(size_error).all() or (size_error < 0).any():
        raise ValueError('Calibration size errors must be finite and nonnegative')
    indices = [int(row.get('index', i)) for i, row in enumerate(rows)]
    if len(set(indices)) != len(indices):
        raise ValueError('Calibration row indices must be unique')
    correct = accepted & (predicted == targets)
    wrong = accepted & (predicted != targets)
    system_rows = np.zeros(len(rows), dtype=bool)
    systems = {}
    for family in SYSTEM_FAMILIES:
        family_index = families.index(family) if family in families else None
        truth = targets == family_index if family_index is not None else np.zeros(len(rows), dtype=bool)
        prediction = predicted == family_index if family_index is not None else np.zeros(len(rows), dtype=bool)
        system_rows |= truth
        count = int(truth.sum())
        family_correct = int((correct & truth).sum())
        systems[family] = {'rows': count, 'correct': family_correct,
                           'wrong_true': int((wrong & truth).sum()),
                           'wrong_pred': int((wrong & prediction).sum()),
                           'correct_coverage': family_correct / count if count else 0.}
    return {'system_correct': int((correct & system_rows).sum()),
            'overall_correct': int(correct.sum()), 'overall_wrong': int(wrong.sum()),
            'nonsystem_correct': int((correct & ~system_rows).sum()),
            'nonsystem_wrong': int((wrong & ~system_rows).sum()), 'systems': systems,
            'size_mean_relative_error': float(size_error.mean()),
            'correct_rows': [indices[i] for i in np.flatnonzero(correct)]}


def assess_checkpoint(candidate, parent):
    """Apply frozen safety conditions, then return a lexicographic rank.

    Callers choose only eligible candidates and replace a current best checkpoint
    only when the new rank is strictly greater, preserving the earlier tie.
    """
    reasons = []
    if candidate['overall_wrong'] > parent['overall_wrong']:
        reasons.append('overall_wrong_increased')
    for family in SYSTEM_FAMILIES:
        for key in ('wrong_true', 'wrong_pred'):
            if candidate['systems'][family][key] > parent['systems'][family][key]:
                reasons.append(f'system_{key}_increased:{family}')
    if candidate['nonsystem_correct'] < parent['nonsystem_correct']:
        reasons.append('nonsystem_correct_decreased')
    if candidate['nonsystem_wrong'] > parent['nonsystem_wrong']:
        reasons.append('nonsystem_wrong_increased')
    if candidate['system_correct'] <= parent['system_correct']:
        reasons.append('system_correct_not_improved')
    if not set(parent['correct_rows']).issubset(candidate['correct_rows']):
        reasons.append('parent_correct_rows_lost')
    rank = [candidate['system_correct'],
            min(candidate['systems'][family]['correct_coverage'] for family in SYSTEM_FAMILIES),
            candidate['overall_correct'], -candidate['size_mean_relative_error']]
    return {'eligible': not reasons, 'reasons': reasons, 'rank': rank}
