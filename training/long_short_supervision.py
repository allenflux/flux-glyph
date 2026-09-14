"""Small, TRAIN-only supervision for verified three/four-glyph known crops."""
from __future__ import annotations

from prepare_unified_regions import FAMILIES
from train_regions import require

CONTRACT = {
    'coefficient': .125, 'denominator': 32, 'label_smoothing': .03,
    'glyph_counts': [3, 4], 'known_targets_only': True,
    'size_coefficient': .2, 'size_beta': .05,
    'one_two_glyph_gradient': 0., 'short_unknown_gradient': 0.,
    'labels': 'verified native TRAIN truth; no predicted labels',
    'runtime_changes': False,
}


def long_short_loss(logits, ratios, targets, sizes, rows):
    import torch
    from torch.nn import functional as F
    require(logits.shape == (32, 25) and ratios.shape == targets.shape == sizes.shape == (32,)
            and logits.dtype in (torch.float32, torch.float64)
            and logits.dtype == ratios.dtype == sizes.dtype and targets.dtype == torch.int64
            and logits.device == ratios.device == targets.device == sizes.device
            and all(bool(torch.isfinite(v).all()) for v in (logits, ratios, sizes))
            and len(rows) == 32, 'Invalid long-short supervision tensors')
    eligible = []
    for target, row in zip(targets.detach().cpu().tolist(), rows):
        count = row.get('glyph_count')
        require(0 <= target < 25 and row.get('split') == 'train'
                and row.get('native_font_verified') is True and type(row.get('target')) is int
                and row.get('target') == target
                and row.get('family') == FAMILIES[target]
                and type(count) is int and 1 <= count <= 4, 'Short crop truth changed')
        eligible.append(target != 24 and count in (3, 4))
    mask = torch.tensor(eligible, dtype=torch.bool, device=logits.device)
    if not any(eligible):
        zero = logits.sum() * 0. + ratios.sum() * 0.
        return zero, zero, zero, mask
    font = F.cross_entropy(logits[mask], targets[mask], label_smoothing=.03, reduction='sum') / 32
    size = F.smooth_l1_loss(ratios[mask], sizes[mask], beta=.05, reduction='sum') / 32
    return .125 * (font + .2 * size), font, size, mask
