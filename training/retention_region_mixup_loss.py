"""TRAIN-only region probability supervision and manifold mixup for one CNN."""
from __future__ import annotations

import math
from numbers import Integral, Real

import train_unified_retention_core as core
from train_regions import require


def _integer_vector(value, name, torch):
    if isinstance(value, torch.Tensor):
        require(value.device.type == 'cpu' and value.dtype == torch.int64 and value.shape == (96,)
                and not value.requires_grad, 'Invalid ' + name)
        return value.tolist()
    require(isinstance(value, (list, tuple)) and len(value) == 96
            and all(isinstance(v, Integral) and not isinstance(v, bool) for v in value), 'Invalid ' + name)
    return [int(v) for v in value]


def region_mixup_loss(model, features, base_logits, tile_counts, targets, rows, families,
                      mix_indices, permutation, lam):
    """Return unscaled loss components and the fixed TRAIN identity evidence.

    Region probabilities are the mean of per-tile softmax probabilities. Their
    logarithms use logsumexp for stability, without averaging the raw logits.
    ce_weights is a Python float list; all other returned values are tensors.
    """
    import torch
    core.registry(families)
    counts = _integer_vector(tile_counts, 'region tile counts', torch)
    indices = _integer_vector(mix_indices, 'mixup tile indices', torch)
    order = _integer_vector(permutation, 'mixup permutation', torch)
    require(all(1 <= count <= 8 for count in counts) and sorted(order) == list(range(96)),
            'Invalid tile count or mixup permutation')
    offset = 0
    for count, index in zip(counts, indices):
        require(offset <= index < offset + count, 'Mixup tile index does not belong to its region')
        offset += count
    require(isinstance(lam, Real) and not isinstance(lam, bool) and math.isfinite(lam) and 0 <= lam <= 1,
            'Invalid mixup lambda')
    require(len(families) == 25 and all(f in families for f in core.CORE_FAMILIES)
            and isinstance(features, torch.Tensor) and isinstance(base_logits, torch.Tensor)
            and isinstance(targets, torch.Tensor) and features.shape == (offset, 128)
            and base_logits.shape == (offset, 25) and targets.shape == (96,)
            and features.dtype in (torch.float32, torch.float64) and base_logits.dtype == features.dtype
            and targets.dtype == torch.int64
            and all(v.device.type == 'cpu' and not v.requires_grad for v in (features, base_logits, targets))
            and all(bool(torch.isfinite(v).all()) for v in (features, base_logits)),
            'Invalid frozen region loss tensors')
    require(hasattr(model, 'family_head') and hasattr(model, 'residual_family_head')
            and not model.family_head.training
            and all(not p.requires_grad and p.grad is None for n, p in model.named_parameters()
                    if not n.startswith('residual_family_head.'))
            and all(p.device.type == 'cpu' and p.dtype == features.dtype for p in model.parameters()),
            'Region mixup requires a frozen CPU base and matching parameter dtype')
    weights = core.native_core_weights(rows, targets.tolist(), families)
    weights = [2. if row['family'] == core.UNKNOWN else weight for row, weight in zip(rows, weights)]
    weight_tensor = features.new_tensor(weights)
    logits = base_logits + model.residual_family_head(features)
    require(logits.shape == base_logits.shape and bool(torch.isfinite(logits).all()), 'Invalid adapted tile logits')
    tile_logp = logits.log_softmax(dim=1)
    region_logp = torch.stack([torch.logsumexp(part, dim=0) - math.log(count)
                              for part, count in zip(tile_logp.split(counts), counts)])
    region_probabilities = region_logp.exp()
    base_probabilities = torch.stack([part.mean(dim=0) for part in base_logits.softmax(dim=1).split(counts)])
    one_hot = torch.nn.functional.one_hot(targets, 25).to(features.dtype)
    smoothed = one_hot * .97 + .03 / 25
    region_ce = (-(smoothed * region_logp).sum(dim=1) * weight_tensor).sum() / 96
    eligible = targets != families.index(core.UNKNOWN)
    for family in core.CORE_FAMILIES:
        eligible &= targets != families.index(family)
    mask = eligible & (base_probabilities.argmax(dim=1) == targets)
    base_kl = (torch.nn.functional.kl_div(region_logp[mask], base_probabilities[mask], reduction='none').sum(dim=1).mean()
               if bool(mask.any()) else logits.sum() * 0.)
    selected = features[indices]
    mixed_features = float(lam) * selected + (1. - float(lam)) * selected[order]
    mixed_targets = float(lam) * one_hot + (1. - float(lam)) * one_hot[order]
    mixed_logits = model.family_head(mixed_features) + model.residual_family_head(mixed_features)
    mixup_ce = -(mixed_targets * mixed_logits.log_softmax(dim=1)).sum(dim=1).mean()
    loss = region_ce + base_kl + .5 * mixup_ce
    require(bool(torch.isfinite(loss)), 'Nonfinite region mixup loss')
    return {'loss': loss, 'region_ce': region_ce, 'base_kl': base_kl, 'mixup_ce': mixup_ce,
            'kl_mask': mask, 'ce_weights': weights, 'region_probabilities': region_probabilities}
