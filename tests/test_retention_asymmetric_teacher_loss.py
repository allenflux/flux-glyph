"""Focused CPU contracts for contingency asymmetric teacher preservation."""
import pytest

torch = pytest.importorskip('torch')
from torch.nn import functional as F

from training.retention_asymmetric_teacher_loss import (
    ASYMMETRIC_PRESERVATION,
    asymmetric_teacher_loss,
)
from training.retention_dual_teacher_loss import correct_teacher_kl


def test_canonical_spec_distinguishes_global_and_unknown_only_distribution_kl():
    assert ASYMMETRIC_PRESERVATION == {
        'outer_weight': 2.0,
        'returned_loss_includes_outer_weight': False,
        'training_scope': 'TRAIN only',
        'teacher_frozen': True,
        'eligibility': 'Selected same-tile TRAIN teacher argmax equals true target; known=b08, unknown=R21',
        'named_loss': 'relu(log p_teacher(true) - log p_student(true))',
        'named_behavior': 'One-sided teacher true-class confidence floor; no upper confidence cap',
        'unknown_loss': 'KL(teacher || student) at temperature 1',
        'temperature': 1.0,
        'distribution_kl_scope': 'Eligible true-unknown TRAIN rows only; selected R21 teacher distribution',
        'shared_reduction': 'Sum eligible named-floor and unknown-KL terms divided by total eligible row count',
        'empty_eligibility': 'Differentiable zero',
        'teacher_distribution_kl': False,
        'teacher_distribution_kl_all_eligible_rows': False,
        'named_distribution_kl': False,
        'unknown_distribution_kl': True,
        'confidence_threshold': None,
        'adds_parameters': False,
        'inference_rule': False,
        'deployed_operator': False,
        'labels_unchanged': True,
    }


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_named_student_above_floor_has_zero_loss_and_gradient_despite_other_changes(dtype):
    targets = torch.tensor([0])
    teacher = torch.log(torch.tensor([[.60, .25, .15]], dtype=dtype))
    student = torch.log(torch.tensor([[.70, .001, .299]], dtype=dtype)).requires_grad_()
    loss, mask = asymmetric_teacher_loss(student, teacher, targets, unknown_index=2)
    assert mask.tolist() == [True] and loss.item() == 0.
    torch.testing.assert_close(torch.autograd.grad(loss, student)[0], torch.zeros_like(student))


def test_named_student_below_floor_has_one_sided_log_probability_loss_and_gradient():
    targets = torch.tensor([1])
    teacher = torch.log(torch.tensor([[.15, .70, .15]], dtype=torch.float64))
    student = torch.log(torch.tensor([[.30, .40, .30]], dtype=torch.float64)).requires_grad_()
    loss, mask = asymmetric_teacher_loss(student, teacher, targets, unknown_index=2)
    expected = teacher.log_softmax(1)[0, 1] - student.log_softmax(1)[0, 1]
    assert mask.tolist() == [True]
    torch.testing.assert_close(loss, expected)
    gradient = torch.autograd.grad(loss, student)[0]
    assert gradient[0, 1] < 0 and bool((gradient[0, [0, 2]] > 0).all())


def test_all_unknown_rows_are_exact_current_correct_teacher_kl():
    targets = torch.tensor([3, 3, 3])
    teacher = torch.tensor([[0., 0., 0., 3.], [0., 0., 0., 2.],
                            [0., 0., 4., 0.]], dtype=torch.float64)
    student = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
                            [0., 0., 0., 1.]], dtype=torch.float64, requires_grad=True)
    actual, mask = asymmetric_teacher_loss(student, teacher, targets, unknown_index=3)
    expected, old_mask = correct_teacher_kl(student, teacher, targets)
    assert mask.tolist() == [True, True, False]
    assert torch.equal(mask, old_mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, student, retain_graph=True)[0],
        torch.autograd.grad(expected, student)[0], rtol=0, atol=0)


def test_combined_terms_sum_over_one_total_eligible_denominator():
    targets = torch.tensor([0, 1, 3, 2])
    teacher = torch.tensor([[3., 0., 0., 0.], [0., 3., 0., 0.],
                            [0., 0., 0., 3.], [3., 0., 0., 0.]], dtype=torch.float64)
    student = torch.tensor([[0., 1., 0., 0.], [0., 4., 0., -2.],
                            [1., 0., 0., 0.], [0., 0., -8., 0.]], dtype=torch.float64,
                           requires_grad=True)
    loss, mask = asymmetric_teacher_loss(student, teacher, targets, unknown_index=3)
    named = mask & (targets != 3)
    unknown = mask & (targets == 3)
    named_term = torch.relu(
        teacher[named].log_softmax(1).gather(1, targets[named, None])
        - student[named].log_softmax(1).gather(1, targets[named, None])).sum()
    unknown_term = F.kl_div(student[unknown].log_softmax(1), teacher[unknown].softmax(1),
                            reduction='none').sum()
    assert mask.tolist() == [True, True, True, False]
    torch.testing.assert_close(loss, (named_term + unknown_term) / 3)
    gradient = torch.autograd.grad(loss, student)[0]
    torch.testing.assert_close(gradient[1], torch.zeros_like(gradient[1]))
    torch.testing.assert_close(gradient[3], torch.zeros_like(gradient[3]))
    assert gradient[0].abs().sum() > 0 and gradient[2].abs().sum() > 0


def test_empty_eligibility_returns_differentiable_zero():
    targets = torch.tensor([0, 1, 3])
    teacher = torch.tensor([[0., 3., 0., 0.], [0., 0., 3., 0.],
                            [3., 0., 0., 0.]], dtype=torch.float64)
    student = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    loss, mask = asymmetric_teacher_loss(student, teacher, targets, unknown_index=3)
    assert not mask.any() and loss.item() == 0. and loss.requires_grad
    torch.testing.assert_close(torch.autograd.grad(loss, student)[0], torch.zeros_like(student))


def test_frozen_teacher_never_receives_gradient():
    targets = torch.tensor([0, 2])
    teacher = torch.tensor([[3., 0., 0.], [0., 0., 3.]], dtype=torch.float64)
    student = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    asymmetric_teacher_loss(student, teacher, targets, unknown_index=2)[0].backward()
    assert teacher.grad is None and student.grad is not None


@pytest.mark.parametrize('fault', [
    'student_rank', 'student_nan', 'teacher_shape', 'teacher_dtype', 'teacher_nan',
    'teacher_grad', 'target_shape', 'target_dtype', 'target_range',
    'unknown_index_type', 'unknown_index_range',
])
def test_rejects_invalid_inputs(fault):
    student = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(3, 4, dtype=torch.float64)
    targets = torch.tensor([0, 1, 3])
    unknown_index = 3
    if fault == 'student_rank': student = student[0]
    elif fault == 'student_nan':
        student = student.detach(); student[0, 0] = float('nan'); student.requires_grad_()
    elif fault == 'teacher_shape': teacher = teacher[:2]
    elif fault == 'teacher_dtype': teacher = teacher.float()
    elif fault == 'teacher_nan': teacher[0, 0] = float('nan')
    elif fault == 'teacher_grad': teacher.requires_grad_()
    elif fault == 'target_shape': targets = targets[:2]
    elif fault == 'target_dtype': targets = targets.float()
    elif fault == 'target_range': targets[0] = 4
    elif fault == 'unknown_index_type': unknown_index = True
    else: unknown_index = 4
    with pytest.raises(ValueError):
        asymmetric_teacher_loss(student, teacher, targets, unknown_index)
