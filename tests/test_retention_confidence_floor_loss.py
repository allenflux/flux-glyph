"""The confidence-floor objective is one-sided and preserves full-CNN training."""
import pytest

torch = pytest.importorskip('torch')

from training.retention_confidence_floor_loss import (
    teacher_confidence_floor,
    unknown_floor_loss,
)
from training import train_unified_retention_confidence_floor as m
from training import train_unified_retention_teacher_preserve as old
from training.prepare_unified_regions import FAMILIES
from test_unified_retention_teacher_preserve import sampler


def test_unknown_above_half_has_exactly_zero_loss_and_gradient():
    # exp(2) / (exp(2) + 3) > .5.
    logits = torch.tensor([[0., 0., 0., 2.]], dtype=torch.float64, requires_grad=True)
    loss = unknown_floor_loss(logits, 3)
    assert loss.shape == (1,) and loss.item() == 0.
    torch.testing.assert_close(torch.autograd.grad(loss.sum(), logits)[0], torch.zeros_like(logits))


def test_unknown_below_half_is_pushed_up_and_matches_negative_log_probability_floor():
    logits = torch.tensor([[1.2, -.4, .3, -.2]], dtype=torch.float64, requires_grad=True)
    probability = logits.softmax(1)[0, 3]
    loss = unknown_floor_loss(logits, 3)
    torch.testing.assert_close(loss[0], -probability.log() - torch.log(torch.tensor(2., dtype=logits.dtype)))
    gradient = torch.autograd.grad(loss.sum(), logits)[0]
    assert gradient[0, 3] < 0. and bool((gradient[0, :3] > 0.).all())


def test_unknown_floor_is_invariant_to_named_distribution_at_fixed_named_logsumexp():
    first = torch.tensor([[0., 0., 0., .2]], dtype=torch.float64)
    # Preserve the aggregate named evidence while changing its distribution.
    second_known = torch.tensor([[1., -1., -2.]], dtype=torch.float64)
    second_known += torch.logsumexp(first[:, :3], 1) - torch.logsumexp(second_known, 1)
    second = torch.cat((second_known, first[:, 3:]), 1)
    torch.testing.assert_close(unknown_floor_loss(first, 3), unknown_floor_loss(second, 3))


def test_teacher_mask_is_same_tile_truth_correctness_and_false_teachers_are_ignored():
    targets = torch.tensor([0, 1, 2])
    teacher = torch.tensor([[3., 0., 0.], [3., 0., 0.], [0., 0., 3.]], dtype=torch.float64)
    student = torch.tensor([[0., 3., 0.], [0., -8., 0.], [0., 3., 0.]],
                           dtype=torch.float64, requires_grad=True)
    loss, mask = teacher_confidence_floor(student, teacher, targets)
    assert mask.tolist() == [True, False, True]
    gradient = torch.autograd.grad(loss, student)[0]
    assert not gradient[1].any() and gradient[0, 0] < 0. and gradient[2, 2] < 0.


def test_higher_true_probability_has_no_penalty_despite_changed_wrong_class_distribution():
    targets = torch.tensor([0])
    teacher = torch.log(torch.tensor([[.6, .2, .2]], dtype=torch.float64))
    student = torch.log(torch.tensor([[.7, .299, .001]], dtype=torch.float64)).requires_grad_()
    loss, mask = teacher_confidence_floor(student, teacher, targets)
    assert mask.tolist() == [True] and loss.item() == 0.
    torch.testing.assert_close(torch.autograd.grad(loss, student)[0], torch.zeros_like(student))


def test_teacher_floor_is_mean_one_sided_log_probability_gap_over_eligible_rows():
    targets = torch.tensor([0, 1, 2])
    teacher = torch.tensor([[2., 0., 0.], [0., 2., 0.], [2., 0., 0.]], dtype=torch.float64)
    student = torch.tensor([[0., 2., 0.], [0., 3., 0.], [0., 0., 2.]], dtype=torch.float64)
    loss, mask = teacher_confidence_floor(student, teacher, targets)
    teacher_true = teacher.log_softmax(1)[torch.arange(3), targets]
    student_true = student.log_softmax(1)[torch.arange(3), targets]
    expected = torch.relu(teacher_true[mask] - student_true[mask]).mean()
    assert mask.tolist() == [True, True, False]
    torch.testing.assert_close(loss, expected)


def test_all_false_teacher_mask_returns_differentiable_zero():
    student = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0, 1, 2])
    teacher = torch.full((3, 4), -2., dtype=torch.float64)
    teacher[torch.arange(3), torch.tensor([1, 2, 3])] = 2.
    loss, mask = teacher_confidence_floor(student, teacher, targets)
    assert not mask.any() and loss.item() == 0. and loss.requires_grad
    torch.testing.assert_close(torch.autograd.grad(loss, student)[0], torch.zeros_like(student))


def _batch_tensors():
    torch.set_num_threads(4)
    rows = sampler().batch()
    targets = torch.tensor([row['target'] for row in rows])
    logits = torch.linspace(-1., 1., 96 * 25, dtype=torch.float64).reshape(96, 25).requires_grad_()
    ratios = torch.linspace(-.2, .4, 96, dtype=torch.float64).requires_grad_()
    sizes = torch.full((96,), .2, dtype=torch.float64)
    teacher = torch.full((96, 25), -2., dtype=torch.float64)
    teacher[torch.arange(96), targets] = 2.
    return rows, targets, logits, ratios, sizes, teacher


def test_full_loss_changes_only_unknown_and_teacher_terms_from_prior_objective():
    rows, targets, logits, ratios, sizes, teacher = _batch_tensors()
    total, ce, size, preservation, mask = m.full_losses(
        logits, ratios, targets, sizes, teacher, rows, FAMILIES)
    _, old_ce, old_size, _, old_mask = old.full_losses(
        logits, ratios, targets, sizes, teacher, rows, FAMILIES)
    known = targets != FAMILIES.index('__unknown__')
    weights = torch.tensor(m.core.native_core_weights(rows, targets.tolist(), FAMILIES), dtype=logits.dtype)
    expected_known = torch.nn.functional.cross_entropy(
        logits[known], targets[known], label_smoothing=.03, reduction='none')
    expected_unknown = unknown_floor_loss(logits[~known], FAMILIES.index('__unknown__'))
    expected_ce = ((expected_known * weights[known]).sum()
                   + (expected_unknown * weights[~known]).sum()) / 96
    torch.testing.assert_close(ce, expected_ce)
    torch.testing.assert_close(size, old_size, rtol=0, atol=0)
    assert mask.all() and old_mask.all()
    torch.testing.assert_close(total, ce + .2 * size + 2 * preservation)
    # The old and new objectives deliberately differ only on unknown supervision and preservation.
    assert old_ce != ce


def test_one_full_cnn_step_updates_all_four_parameter_groups():
    from training.region_network import RegionFontClassifier

    torch.manual_seed(1413)
    model = RegionFontClassifier(25)
    before = m.parameter_groups(model.state_dict())
    rows = sampler().batch()
    targets = torch.tensor([row['target'] for row in rows])
    teacher = torch.full((96, 25), -2.)
    teacher[torch.arange(96), targets] = 2.
    logits, ratios = model(torch.rand(96, 1, 64, 256))
    loss, *_ = m.full_losses(logits, ratios, targets, torch.full((96,), .2), teacher, rows, FAMILIES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters())
    optimizer.step()
    after = m.parameter_groups(model.state_dict())
    assert all(after[name] != digest for name, digest in before.items())


@pytest.mark.parametrize('fault', ['teacher_grad', 'teacher_nan', 'teacher_shape', 'teacher_dtype',
                                    'target_shape', 'target_dtype', 'target_range'])
def test_teacher_floor_rejects_invalid_pairing(fault):
    student = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(3, 4, dtype=torch.float64)
    targets = torch.tensor([0, 1, 2])
    if fault == 'teacher_grad': teacher.requires_grad_()
    elif fault == 'teacher_nan': teacher[0, 0] = float('nan')
    elif fault == 'teacher_shape': teacher = teacher[:2]
    elif fault == 'teacher_dtype': teacher = teacher.float()
    elif fault == 'target_shape': targets = targets[:2]
    elif fault == 'target_dtype': targets = targets.float()
    else: targets[0] = 4
    with pytest.raises(ValueError):
        teacher_confidence_floor(student, teacher, targets)
