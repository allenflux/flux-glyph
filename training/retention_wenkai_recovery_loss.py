"""One extra smoothed CE unit for true LXGW WenKai TRAIN rows."""
from __future__ import annotations

WENKAI_FAMILY = 'LXGW WenKai'
WENKAI_RECOVERY_WEIGHTING = {
    'family': WENKAI_FAMILY,
    'training_scope': 'True-label TRAIN rows only',
    'base_known_ce_weight': 1.0,
    'extra_known_ce_weight': 1.0,
    'resulting_known_ce_weight': 2.0,
    'label_smoothing': 0.03,
    'denominator': 96,
    'teacher_eligibility_gated': False,
    'native_core_weighting_changed': False,
    'angular_margin_weighting_changed': False,
    'unknown_loss_changed': False,
    'size_loss_changed': False,
    'teacher_preservation_changed': False,
    'labels_unchanged': True,
    'adds_parameters': False,
    'inference_rule': False,
    'deployed_operator': False,
}


def wenkai_recovery_extra_ce(logits, targets, rows, families):
    """Return the additional coefficient-inclusive WenKai CE divided by 96."""
    import torch
    from torch.nn import functional as F

    valid = (isinstance(logits, torch.Tensor) and logits.ndim == 2
             and logits.shape == (96, 25)
             and logits.dtype in (torch.float32, torch.float64)
             and logits.device.type != 'meta' and bool(torch.isfinite(logits).all())
             and isinstance(targets, torch.Tensor) and targets.shape == (96,)
             and targets.dtype == torch.int64 and targets.device == logits.device
             and isinstance(rows, list) and len(rows) == 96
             and isinstance(families, list) and len(families) == 25
             and len(set(families)) == 25 and WENKAI_FAMILY in families
             and bool(((targets >= 0) & (targets < 25)).all()))
    if not valid:
        raise ValueError('WenKai recovery requires one finite canonical 96-row TRAIN batch')
    target_values = targets.detach().cpu().tolist()
    for row, target in zip(rows, target_values):
        if not (isinstance(row, dict) and type(row.get('target')) is int
                and row['target'] == target and row.get('family') == families[target]
                and row.get('split') == 'train' and row.get('native_font_verified') is True):
            raise ValueError('WenKai recovery cannot relabel or leave verified TRAIN rows')
    mask = targets == families.index(WENKAI_FAMILY)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], targets[mask], label_smoothing=.03,
                           reduction='sum') / 96
