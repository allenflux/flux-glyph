"""Check actual auxiliary gradients and TRAIN provenance, not just metadata."""
from copy import deepcopy
import math

import pytest

torch = pytest.importorskip('torch')
from training.native_unknown_oe import COEFFICIENT, POLICY, unknown_named_uniformity
from training.prepare_unified_regions import FAMILIES


def rows_and_targets():
    targets = torch.tensor([24 if i % 6 == 0 else i % 24 for i in range(96)])
    rows = [{'split': 'train', 'native_font_verified': True, 'target': int(t),
             'family': FAMILIES[int(t)], 'domain': 'android', 'view': 'native',
             'source_font_family': 'Smiley Sans' if int(t) == 24 else FAMILIES[int(t)]}
            for t in targets]
    return rows, targets


def test_gradient_uniformizes_only_named_alternatives_of_true_unknown_rows():
    rows, targets = rows_and_targets()
    torch.manual_seed(94)
    logits = torch.randn(96, 25, dtype=torch.float64, requires_grad=True)
    value = unknown_named_uniformity(logits, targets, rows, FAMILIES)
    value.backward()
    mask = targets == 24
    expected = torch.zeros_like(logits)
    expected[mask, :24] = COEFFICIENT / 96 * (logits.detach()[mask, :24].softmax(1) - 1 / 24)
    torch.testing.assert_close(logits.grad, expected, atol=1e-14, rtol=1e-12)
    assert torch.count_nonzero(logits.grad[:, 24]) == 0
    assert torch.count_nonzero(logits.grad[~mask]) == 0
    assert value.item() > 0


def test_named_common_offset_and_unknown_logit_do_not_change_auxiliary():
    rows, targets = rows_and_targets()
    torch.manual_seed(95)
    logits = torch.randn(96, 25, dtype=torch.float64)
    expected = unknown_named_uniformity(logits, targets, rows, FAMILIES)
    for unknown in [-100., 0., 100.]:
        changed = logits.clone(); changed[:, :24] += 1000; changed[:, 24] = unknown
        torch.testing.assert_close(unknown_named_uniformity(changed, targets, rows, FAMILIES),
                                  expected, atol=1e-12, rtol=1e-12)


def test_uniform_named_distribution_has_zero_loss_and_gradient_at_any_unknown_mass():
    rows, targets = rows_and_targets()
    for unknown in [-100., 100.]:
        logits = torch.zeros(96, 25, dtype=torch.float64)
        logits[:, 24] = unknown; logits.requires_grad_()
        value = unknown_named_uniformity(logits, targets, rows, FAMILIES)
        value.backward()
        assert abs(value.item()) < 1e-14
        assert logits.grad.abs().max().item() < 1e-14


def test_no_unknown_rows_is_differentiable_zero():
    rows, targets = rows_and_targets()
    for row in rows: row.update(target=0, family=FAMILIES[0])
    targets.zero_()
    logits = torch.randn(96, 25, requires_grad=True)
    value = unknown_named_uniformity(logits, targets, rows, FAMILIES)
    value.backward()
    assert value.item() == 0 and torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize('change', ['target', 'family', 'split', 'proof', 'shape', 'nan'])
def test_rejects_invalid_training_provenance_or_tensors(change):
    rows, targets = rows_and_targets(); logits = torch.zeros(96, 25)
    if change == 'target': rows[0]['target'] = 0
    elif change == 'family': rows[0]['family'] = FAMILIES[0]
    elif change == 'split': rows[0]['split'] = 'calibration'
    elif change == 'proof': rows[0]['native_font_verified'] = False
    elif change == 'shape': logits = logits[:32]
    else: logits[0, 0] = math.nan
    with pytest.raises(ValueError):
        unknown_named_uniformity(logits, targets, rows, FAMILIES)


def test_auxiliary_has_no_runtime_or_unknown_probability_target():
    assert POLICY['inference_rule'] is False and POLICY['new_parameters'] == 0
    assert POLICY['focus_rows_regularized'] is False
    assert POLICY['unknown_probability_target'] is None
    assert POLICY['original_unknown_floor_preserved'] and POLICY['original_full_teacher_kl_preserved']
