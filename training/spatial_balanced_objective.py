"""Additional verified-TRAIN supervision; no inference score or platform rule."""
from __future__ import annotations

import train_unified_spatial_detail_v2 as parent

POLICY = {
    'schema': 'flux-glyph-spatial-rejection-supervision-v2',
    'rows': 'existing replay, focus and mobile rows only; pair rows excluded',
    'targets': 'unaltered verified native TRAIN font labels',
    'unknown_cross_entropy_weight': .25,
    'kaiti_extra_cross_entropy_weight': .05,
    'non_system_penalty': '-log(1 - sum p(PingFang, SF Pro, Helvetica)) for true non-system targets',
    'non_system_penalty_applies_to': 'all TRAIN domains equally; no platform features or routes',
    'original_objective_preserved': True,
    'second_forward': False, 'runtime_changes': False,
}


def repair_losses(logits, targets, confusion_weight):
    import torch
    from torch.nn import functional as F
    parent.require(logits.ndim == 2 and logits.shape[1] == 25 and targets.shape == (len(logits),)
                   and targets.dtype == torch.int64 and targets.device == logits.device
                   and bool(torch.isfinite(logits).all())
                   and bool(((targets >= 0) & (targets < 25)).all())
                   and confusion_weight >= 0, 'Invalid verified-TRAIN repair batch')
    zero = logits.sum() * 0.
    unknown = targets == 24
    kaiti = targets == parent.FAMILIES.index('Kaiti SC')
    system_indices = [parent.FAMILIES.index(name) for name in ('PingFang', 'SF Pro', 'Helvetica')]
    other_indices = [i for i in range(25) if i not in system_indices]
    non_system = ~torch.isin(targets, targets.new_tensor(system_indices))
    unknown_ce = F.cross_entropy(logits[unknown], targets[unknown]) if bool(unknown.any()) else zero
    kaiti_ce = F.cross_entropy(logits[kaiti], targets[kaiti]) if bool(kaiti.any()) else zero
    # Stable grouped CE, finite even when the network is extremely confident in a wrong system family.
    confusion = F.softplus(torch.logsumexp(logits[non_system][:, system_indices], 1)
                          - torch.logsumexp(logits[non_system][:, other_indices], 1)).mean() \
        if bool(non_system.any()) else zero
    total = POLICY['unknown_cross_entropy_weight'] * unknown_ce + \
        POLICY['kaiti_extra_cross_entropy_weight'] * kaiti_ce + confusion_weight * confusion
    return total, {'repair_unknown_ce': unknown_ce, 'repair_kaiti_ce': kaiti_ce,
                   'repair_system_confusion': confusion, 'repair_total': total}


def objective(model, batch, confusion_weight):
    """Original loss uses the same outputs once; only the declared TRAIN terms are added."""
    import torch
    from torch.nn import functional as F
    font, size, features = parent.forward_with_features(model, batch['images'])
    parent.require(font.shape == (180, 25) and size.shape == (180,) and features.shape[0] == 180,
                   'Joint optimizer requires one aligned 180-row CNN forward')
    replay = parent.full_losses(font[:96], size[:96], features[:96], model.family_head.weight,
        batch['targets'], batch['sizes'], batch['teacher'], batch['rows'], parent.FAMILIES)
    focus = parent.focus_full_loss(font[96:128], size[96:128], features[96:128], model.family_head.weight,
        batch['focus_targets'], batch['focus_sizes'], batch['focus_teacher'], batch['focus_rows'])
    oe = parent.unknown_named_uniformity(font[:96], batch['targets'], batch['rows'], parent.FAMILIES)
    mobile_ce = F.cross_entropy(font[128:148], batch['mobile_targets'])
    mobile_size = F.smooth_l1_loss(size[128:148], batch['mobile_sizes'], beta=.05)
    mobile_loss = parent.MOBILE_WEIGHT * (mobile_ce + .2 * mobile_size)
    pair_loss = parent.pair_margin_loss(font[148:180], batch['pair_targets'])
    pair_weighted = parent.PAIR_WEIGHT * pair_loss
    base_loss = replay[0] + parent.FOCUS_WEIGHT * focus[0] + parent.OE_MULTIPLIER * oe + mobile_loss + pair_weighted
    extra, repair_parts = repair_losses(font[:148], torch.cat((batch['targets'], batch['focus_targets'],
                                                             batch['mobile_targets'])), confusion_weight)
    return base_loss + extra, {
        'unknown_oe': oe, 'unknown_oe_weighted': parent.OE_MULTIPLIER * oe,
        'unknown_oe_rows': (batch['targets'] == 24).sum(),
        'replay': replay[0], 'replay_ce': replay[1], 'teacher_kl': replay[3],
        'focus_weighted': parent.FOCUS_WEIGHT * focus[0], 'focus_ce': focus[1],
        'focus_teacher_kl': focus[3], 'focus_size': focus[2],
        'focus_teacher_eligible': focus[5][:32].sum(), 'mobile_loss': mobile_loss,
        'mobile_ce': mobile_ce, 'mobile_size': mobile_size,
        'pair_loss': pair_loss, 'pair_weighted': pair_weighted,
        'pair_count': pair_loss.new_tensor(parent.PAIR_COUNT),
        'pair_rows': pair_loss.new_tensor(parent.PAIR_ROWS), 'original_total': base_loss, **repair_parts}
