"""True-label teacher routing and correct-teacher KL contracts."""
import pytest

torch = pytest.importorskip('torch')
from torch.nn import functional as F

from training.retention_dual_teacher_loss import correct_teacher_kl, select_teacher


def teachers():
    named = torch.arange(20, dtype=torch.float64).reshape(4, 5)
    unknown = -torch.arange(20, dtype=torch.float64).reshape(4, 5) - 100.
    targets = torch.tensor([0, 4, 2, 4])
    return named, unknown, targets


def test_true_known_uses_b08_and_true_unknown_uses_r21_exactly():
    named, unknown, targets = teachers()
    selected, mask = select_teacher(named, unknown, targets, unknown_index=4)
    assert mask.tolist() == [False, True, False, True]
    torch.testing.assert_close(selected[~mask], named[~mask], rtol=0, atol=0)
    torch.testing.assert_close(selected[mask], unknown[mask], rtol=0, atol=0)
    assert not selected.requires_grad


def test_selection_ignores_teacher_predictions_and_confidence():
    targets = torch.tensor([0, 3])
    # On each row, the non-selected teacher is more confident and has a
    # different argmax. Native truth still determines the source exactly.
    named = torch.tensor([[0., 9., 0., 0.], [99., 0., 0., 0.]], dtype=torch.float64)
    unknown = torch.tensor([[100., 0., 0., 0.], [0., 8., 0., 0.]], dtype=torch.float64)
    selected, mask = select_teacher(named, unknown, targets, unknown_index=3)
    assert mask.tolist() == [False, True]
    torch.testing.assert_close(selected, torch.stack((named[0], unknown[1])), rtol=0, atol=0)


def test_default_unknown_index_is_the_fixed_25_class_unknown_row():
    named = torch.zeros(2, 25); unknown = torch.ones(2, 25)
    selected, mask = select_teacher(named, unknown, torch.tensor([23, 24]))
    assert mask.tolist() == [False, True]
    torch.testing.assert_close(selected[0], named[0], rtol=0, atol=0)
    torch.testing.assert_close(selected[1], unknown[1], rtol=0, atol=0)


def test_correct_teacher_mask_includes_known_and_unknown_truth():
    targets = torch.tensor([0, 1, 3, 3])
    reference = torch.tensor([[4., 0., 0., 0.], [3., 0., 0., 0.],
                              [0., 0., 0., 4.], [0., 0., 4., 0.]], dtype=torch.float64)
    student = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    _, mask = correct_teacher_kl(student, reference, targets)
    assert mask.tolist() == [True, False, True, False]


def test_eligible_kl_matches_torch_definition_at_temperature_one():
    targets = torch.tensor([0, 1, 2, 0])
    reference = torch.tensor([[3., 0., 0.], [0., 2., 0.], [2., 0., 0.],
                              [1., 0., 0.]], dtype=torch.float64)
    student = torch.tensor([[0., 2., 0.], [0., 1., 2.], [0., 0., 3.],
                            [-1., 0., 1.]], dtype=torch.float64, requires_grad=True)
    loss, mask = correct_teacher_kl(student, reference, targets)
    expected = F.kl_div(student[mask].log_softmax(1), reference[mask].softmax(1),
                        reduction='none').sum(1).mean()
    assert mask.tolist() == [True, True, False, True]
    torch.testing.assert_close(loss, expected)


def test_incorrect_teacher_rows_have_zero_student_gradient():
    targets = torch.tensor([0, 1, 2])
    reference = torch.tensor([[3., 0., 0.], [3., 0., 0.], [0., 0., 3.]], dtype=torch.float64)
    student = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    loss, mask = correct_teacher_kl(student, reference, targets)
    gradient = torch.autograd.grad(loss, student)[0]
    assert mask.tolist() == [True, False, True]
    torch.testing.assert_close(gradient[1], torch.zeros_like(gradient[1]), rtol=0, atol=0)
    assert bool((gradient[[0, 2]].abs().sum(1) > 0).all())


def test_empty_correct_mask_returns_differentiable_zero():
    targets = torch.tensor([0, 1, 2])
    reference = torch.tensor([[0., 4., 0.], [0., 0., 4.], [4., 0., 0.]], dtype=torch.float64)
    student = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    loss, mask = correct_teacher_kl(student, reference, targets)
    assert not mask.any() and loss.item() == 0. and loss.requires_grad
    torch.testing.assert_close(torch.autograd.grad(loss, student)[0], torch.zeros_like(student), rtol=0, atol=0)


def test_neither_teacher_receives_gradients():
    named, unknown, targets = teachers()
    selected, _ = select_teacher(named, unknown, targets, unknown_index=4)
    student = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    loss, _ = correct_teacher_kl(student, selected, targets)
    loss.backward()
    assert named.grad is None and unknown.grad is None and selected.grad is None
    assert student.grad is not None


@pytest.mark.parametrize('fault', ['named_grad', 'unknown_grad', 'nan', 'shape', 'dtype',
                                    'device', 'target_shape', 'target_dtype', 'target_range',
                                    'unknown_index_type', 'unknown_index_range'])
def test_select_teacher_rejects_invalid_inputs(fault):
    named, unknown, targets = teachers()
    if fault == 'named_grad': named.requires_grad_()
    elif fault == 'unknown_grad': unknown.requires_grad_()
    elif fault == 'nan': unknown[0, 0] = float('nan')
    elif fault == 'shape': unknown = unknown[:3]
    elif fault == 'dtype': unknown = unknown.float()
    elif fault == 'device': unknown = unknown.to('meta')
    elif fault == 'target_shape': targets = targets[:3]
    elif fault == 'target_dtype': targets = targets.float()
    elif fault == 'target_range': targets[0] = 5
    elif fault == 'unknown_index_type':
        with pytest.raises(ValueError): select_teacher(named, unknown, targets, unknown_index=True)
        return
    elif fault == 'unknown_index_range':
        with pytest.raises(ValueError): select_teacher(named, unknown, targets, unknown_index=5)
        return
    with pytest.raises(ValueError): select_teacher(named, unknown, targets, unknown_index=4)


@pytest.mark.parametrize('fault', ['reference_grad', 'nan', 'shape', 'dtype', 'device',
                                    'student_nan', 'student_rank',
                                    'target_shape', 'target_dtype', 'target_range'])
def test_correct_teacher_kl_rejects_invalid_inputs(fault):
    student = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    reference = torch.randn(3, 4, dtype=torch.float64)
    targets = torch.tensor([0, 1, 2])
    if fault == 'reference_grad': reference.requires_grad_()
    elif fault == 'nan': reference[0, 0] = float('nan')
    elif fault == 'shape': reference = reference[:2]
    elif fault == 'dtype': reference = reference.float()
    elif fault == 'device': reference = reference.to('meta')
    elif fault == 'student_nan':
        student = student.detach(); student[0, 0] = float('nan'); student.requires_grad_()
    elif fault == 'student_rank': student = student[0]
    elif fault == 'target_shape': targets = targets[:2]
    elif fault == 'target_dtype': targets = targets.float()
    else: targets[0] = 4
    with pytest.raises(ValueError): correct_teacher_kl(student, reference, targets)
