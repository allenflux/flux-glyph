"""Truth-routed dual-teacher selection and correct-teacher KL for TRAIN only."""
from __future__ import annotations


def _logits(value, name):
    import torch
    if (not isinstance(value, torch.Tensor) or value.ndim != 2
            or value.shape[0] < 1 or value.shape[1] < 2
            or value.dtype not in (torch.float32, torch.float64) or value.device.type == 'meta'
            or not bool(torch.isfinite(value).all())):
        raise ValueError(name + ' must be finite floating-point class logits')


def _targets(targets, logits):
    import torch
    if (not isinstance(targets, torch.Tensor) or targets.dtype != torch.int64
            or targets.device != logits.device or targets.shape != (len(logits),)
            or not bool(((targets >= 0) & (targets < logits.shape[1])).all())):
        raise ValueError('Targets must identify the true class of every TRAIN tile')


def select_teacher(named_logits, unknown_logits, targets, unknown_index=24):
    """Select b08 for known truth and R21 for unknown truth on the same TRAIN tile.

    Selection depends only on the native true target. It never uses either
    teacher's prediction, confidence, platform, source, or runtime score.
    """
    import torch
    _logits(named_logits, 'Named teacher')
    _logits(unknown_logits, 'Unknown teacher')
    if (named_logits.shape != unknown_logits.shape
            or named_logits.dtype != unknown_logits.dtype
            or named_logits.device != unknown_logits.device
            or named_logits.requires_grad or unknown_logits.requires_grad):
        raise ValueError('Both frozen teachers must describe the same TRAIN tiles')
    _targets(targets, named_logits)
    if type(unknown_index) is not int or not 0 <= unknown_index < named_logits.shape[1]:
        raise ValueError('Invalid unknown class index')
    unknown_source_mask = targets == unknown_index
    selected = torch.where(unknown_source_mask[:, None], unknown_logits, named_logits)
    return selected, unknown_source_mask


def correct_teacher_kl(student_logits, reference_logits, targets):
    """Mean T=1 KL(teacher || student) where teacher argmax matches truth."""
    import torch
    from torch.nn import functional as F
    _logits(student_logits, 'Student')
    _logits(reference_logits, 'Reference teacher')
    if (student_logits.shape != reference_logits.shape
            or student_logits.dtype != reference_logits.dtype
            or student_logits.device != reference_logits.device
            or reference_logits.requires_grad):
        raise ValueError('Frozen reference and student must describe the same TRAIN tiles')
    _targets(targets, student_logits)
    mask = reference_logits.argmax(1) == targets
    if not bool(mask.any()):
        return student_logits.sum() * 0.0, mask
    loss = F.kl_div(student_logits[mask].log_softmax(1),
                    reference_logits[mask].softmax(1), reduction='none').sum(1).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite correct-teacher KL')
    return loss, mask
