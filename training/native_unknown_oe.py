"""TRAIN-only uniformity of named alternatives on verified unknown fonts.

An adaptation of Outlier Exposure (https://arxiv.org/abs/1812.04606):
regularize the conditional distribution over the 24 named outputs, keeping
the separate unknown logit, its binary floor and original teacher loss intact.
"""
from __future__ import annotations

import math

COEFFICIENT = .5
BATCH_SIZE = 96
POLICY = {
    'coefficient': COEFFICIENT,
    'denominator': BATCH_SIZE,
    'mask': 'verified true-unknown rows in the original 96-row replay batch only',
    'loss': 'KL(uniform24 || softmax(logits_named24))',
    'known_row_gradient': 'exactly zero',
    'unknown_logit_gradient': 'exactly zero from this auxiliary',
    'focus_rows_regularized': False,
    'conditional_named_distribution_only': True,
    'unknown_probability_target': None,
    'original_unknown_floor_preserved': True,
    'original_full_teacher_kl_preserved': True,
    'teacher_competition': 'Original KL2 can favor nonuniform named tails on eligible unknown rows; both losses remain active.',
    'unknown_mass_effect': 'No explicit probability target; changes to named logits can indirectly change p_unknown.',
    'teacher_logits_used_by_auxiliary': False,
    'labels_unchanged': True,
    'new_parameters': 0,
    'inference_rule': False,
    'reference': 'https://arxiv.org/abs/1812.04606',
    'scope': 'Hypothesis for unseen-font confidence; no generalization claim before evaluation.',
}


def unknown_named_uniformity(logits, targets, rows, families):
    """Coefficient-inclusive scalar, normalized by all 96 replay rows."""
    import torch
    from train_regions import require
    from train_unified_retention_core import native_core_weights, registry

    registry(families)
    require(len(families) == 25 and families[-1] == '__unknown__'
            and isinstance(logits, torch.Tensor) and logits.shape == (BATCH_SIZE, 25)
            and logits.dtype in (torch.float32, torch.float64)
            and isinstance(targets, torch.Tensor) and targets.shape == (BATCH_SIZE,)
            and targets.dtype == torch.int64 and targets.device == logits.device
            and bool(torch.isfinite(logits).all()), 'Invalid unknown OE tensors')
    # Validate every original TRAIN row, including rows whose auxiliary is zero.
    native_core_weights(rows, targets.detach().cpu().tolist(), families)
    mask = targets == 24
    if not bool(mask.any()):
        return logits.sum() * 0.0
    named = logits[mask, :24]
    # Subtract a shared offset before both terms for numerical stability. This
    # does not change the conditional distribution or introduce a new parameter.
    centered = named - named.max(dim=1, keepdim=True).values
    divergence = torch.logsumexp(centered, dim=1) - centered.mean(dim=1) - math.log(24)
    value = COEFFICIENT * divergence.sum() / BATCH_SIZE
    require(value.ndim == 0 and bool(torch.isfinite(value)), 'Nonfinite unknown OE loss')
    return value
