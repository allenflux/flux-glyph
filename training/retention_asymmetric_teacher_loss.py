"""Asymmetric TRAIN-only preservation for truth-routed frozen teachers.

Known targets retain only the selected teacher's true-class confidence floor.
Unknown targets retain the selected teacher's complete probability distribution.
The loss adds no inference rule, threshold, model parameter, or runtime operator.
"""
from __future__ import annotations

ASYMMETRIC_PRESERVATION = {
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


def _validate(student_logits, teacher_logits, targets, unknown_index):
    import torch

    valid_student = (isinstance(student_logits, torch.Tensor)
                     and student_logits.ndim == 2
                     and student_logits.shape[0] >= 1
                     and student_logits.shape[1] >= 2
                     and student_logits.dtype in (torch.float32, torch.float64)
                     and student_logits.device.type != 'meta'
                     and bool(torch.isfinite(student_logits).all()))
    if not valid_student:
        raise ValueError('Student, frozen teacher and targets must describe the same finite TRAIN tiles')
    valid_teacher = (isinstance(teacher_logits, torch.Tensor)
                     and teacher_logits.ndim == 2
                     and teacher_logits.shape == student_logits.shape
                     and teacher_logits.dtype == student_logits.dtype
                     and teacher_logits.device == student_logits.device
                     and not teacher_logits.requires_grad
                     and bool(torch.isfinite(teacher_logits).all()))
    valid_targets = (isinstance(targets, torch.Tensor)
                     and targets.dtype == torch.int64
                     and targets.device == student_logits.device
                     and targets.shape == (len(student_logits),)
                     and bool(((targets >= 0) & (targets < student_logits.shape[1])).all()))
    if not (valid_teacher and valid_targets):
        raise ValueError('Student, frozen teacher and targets must describe the same finite TRAIN tiles')
    if type(unknown_index) is not int or not 0 <= unknown_index < student_logits.shape[1]:
        raise ValueError('Invalid unknown class index')


def asymmetric_teacher_loss(student_logits, teacher_logits, targets, unknown_index=24):
    """Return asymmetric preservation and the teacher-correct eligibility mask.

    For eligible named rows, the per-row term is
    ``relu(log p_teacher(true) - log p_student(true))``. For eligible unknown
    rows, it is the temperature-one full-distribution
    ``KL(teacher || student)``. The sum of both kinds is divided by the total
    eligible count, so neither subset receives an independent mean. A batch
    without an eligible row returns a differentiable zero.
    """
    import torch
    from torch.nn import functional as F

    _validate(student_logits, teacher_logits, targets, unknown_index)
    mask = teacher_logits.argmax(1) == targets
    if not bool(mask.any()):
        return student_logits.sum() * 0.0, mask

    named = mask & (targets != unknown_index)
    unknown = mask & (targets == unknown_index)
    total = student_logits.sum() * 0.0
    if bool(named.any()):
        indices = targets[named, None]
        reference = F.log_softmax(teacher_logits[named], dim=1).gather(1, indices).squeeze(1)
        actual = F.log_softmax(student_logits[named], dim=1).gather(1, indices).squeeze(1)
        total = total + F.relu(reference - actual).sum()
    if bool(unknown.any()):
        total = total + F.kl_div(
            F.log_softmax(student_logits[unknown], dim=1),
            F.softmax(teacher_logits[unknown], dim=1),
            reduction='none').sum(1).sum()
    loss = total / mask.sum()
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite asymmetric teacher preservation')
    return loss, mask
