"""One-sided TRAIN losses: protect correct confidence without capping improvement.

These functions only affect gradient training. They add no inference rule, model
input, font lookup, platform selector, or probability rescaling.
"""
from __future__ import annotations

import math


def _logits(value):
    import torch
    if (not isinstance(value, torch.Tensor) or value.ndim != 2
            or value.shape[0] < 1 or value.shape[1] < 2
            or value.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(value).all())):
        raise ValueError('Expected finite floating-point class logits')


def unknown_floor_loss(logits, unknown_index):
    """Per-row -log(p_unknown) above log(2), zero once p_unknown >= 0.5."""
    import torch
    from torch.nn import functional as F
    _logits(logits)
    if type(unknown_index) is not int or not 0 <= unknown_index < logits.shape[1]:
        raise ValueError('Invalid unknown class index')
    known = torch.cat((logits[:, :unknown_index], logits[:, unknown_index + 1:]), dim=1)
    odds = logits[:, unknown_index] - torch.logsumexp(known, dim=1)
    return F.relu(F.softplus(-odds) - math.log(2.0))


def teacher_confidence_floor(logits, teacher_logits, targets):
    """Preserve teacher's true-class probability only where its argmax is correct.

Unlike distribution KL, this does not penalize reducing competing classes or
increasing the probability of the true font above the teacher's value.
"""
    import torch
    from torch.nn import functional as F
    _logits(logits)
    _logits(teacher_logits)
    if (logits.shape != teacher_logits.shape or logits.dtype != teacher_logits.dtype
            or logits.device != teacher_logits.device or teacher_logits.requires_grad
            or not isinstance(targets, torch.Tensor) or targets.dtype != torch.int64
            or targets.device != logits.device or targets.shape != (len(logits),)
            or not bool(((targets >= 0) & (targets < logits.shape[1])).all())):
        raise ValueError('Teacher, targets and student must describe the same TRAIN tiles')
    mask = teacher_logits.argmax(1) == targets
    if not bool(mask.any()):
        return logits.sum() * 0.0, mask
    indices = targets[mask, None]
    reference = F.log_softmax(teacher_logits[mask], dim=1).gather(1, indices).squeeze(1)
    actual = F.log_softmax(logits[mask], dim=1).gather(1, indices).squeeze(1)
    return F.relu(reference - actual).mean(), mask
