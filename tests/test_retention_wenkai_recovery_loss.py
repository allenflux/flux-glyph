"""Tests for the single-family TRAIN cross-entropy increment."""
from copy import deepcopy

import pytest

torch = pytest.importorskip('torch')
from torch.nn import functional as F

from training.prepare_unified_regions import FAMILIES
from training.retention_wenkai_recovery_loss import (
    WENKAI_FAMILY, WENKAI_RECOVERY_WEIGHTING, wenkai_recovery_extra_ce)


def rows():
    target = FAMILIES.index('PingFang')
    return [{'family': FAMILIES[target], 'target': target, 'split': 'train',
             'native_font_verified': True} for _ in range(96)]


def test_exact_smoothed_sum_over_fixed_denominator_and_gradient_scope():
    batch = rows(); wenkai = FAMILIES.index(WENKAI_FAMILY)
    for index in (2, 19, 77):
        batch[index].update(family=WENKAI_FAMILY, target=wenkai)
    targets = torch.tensor([row['target'] for row in batch])
    logits = torch.randn(96, 25, dtype=torch.float64, requires_grad=True)
    actual = wenkai_recovery_extra_ce(logits, targets, batch, FAMILIES)
    expected = F.cross_entropy(logits[[2, 19, 77]], targets[[2, 19, 77]],
                               label_smoothing=.03, reduction='sum') / 96
    ga, = torch.autograd.grad(actual, logits, retain_graph=True)
    ge, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ga, ge, rtol=1e-12, atol=1e-12)
    assert bool((ga[[index for index in range(96) if index not in (2, 19, 77)]] == 0).all())


def test_no_wenkai_is_differentiable_zero():
    batch = rows(); targets = torch.tensor([row['target'] for row in batch])
    logits = torch.randn(96, 25, requires_grad=True)
    loss = wenkai_recovery_extra_ce(logits, targets, batch, FAMILIES)
    loss.backward()
    assert loss.item() == 0 and bool((logits.grad == 0).all())


def test_canonical_weighting_is_train_only_and_changes_no_other_term():
    spec = WENKAI_RECOVERY_WEIGHTING
    assert spec['family'] == WENKAI_FAMILY
    assert (spec['base_known_ce_weight'], spec['extra_known_ce_weight'],
            spec['resulting_known_ce_weight']) == (1., 1., 2.)
    assert spec['label_smoothing'] == .03 and spec['denominator'] == 96
    assert spec['teacher_eligibility_gated'] is False
    assert all(spec[key] is False for key in ('native_core_weighting_changed',
        'angular_margin_weighting_changed', 'unknown_loss_changed', 'size_loss_changed',
        'teacher_preservation_changed', 'adds_parameters', 'inference_rule', 'deployed_operator'))


@pytest.mark.parametrize('fault', ('family','target','split','verification','shape','dtype','nan'))
def test_rejects_relabeling_nontrain_or_invalid_tensors(fault):
    batch = rows(); wenkai = FAMILIES.index(WENKAI_FAMILY)
    batch[0].update(family=WENKAI_FAMILY, target=wenkai)
    targets = torch.tensor([row['target'] for row in batch])
    logits = torch.randn(96, 25)
    if fault == 'family': batch[0]['family'] = 'PingFang'
    elif fault == 'target': batch[0]['target'] = 0
    elif fault == 'split': batch[0]['split'] = 'calibration'
    elif fault == 'verification': batch[0]['native_font_verified'] = False
    elif fault == 'shape': logits = logits[:95]
    elif fault == 'dtype': targets = targets.float()
    else: logits[0, 0] = float('nan')
    with pytest.raises(ValueError):
        wenkai_recovery_extra_ce(logits, targets, batch, FAMILIES)
