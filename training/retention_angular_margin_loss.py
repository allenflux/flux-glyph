"""TRAIN-only normalized angular-margin auxiliary for known font rows.

The target correction uses the monotone ArcFace continuation below
``cos(pi-margin)``. Raw ``cos(theta+margin)`` reverses direction near pi;
the linear continuation keeps harder angles monotonically more costly.
"""
from __future__ import annotations

import math

SCALE = 16.0
MARGIN_RADIANS = 0.10
COEFFICIENT = 0.10
BATCH_SIZE = 96
NORM_EPSILON = 1e-12
ANGULAR_MARGIN = {'weight': COEFFICIENT, 'scale': SCALE,
    'angle_radians': MARGIN_RADIANS, 'denominator': BATCH_SIZE,
    'mask': 'TRAIN rows whose true class is one of the 24 named families',
    'row_weighting': 'unchanged native-core row weights',
    'features': 'existing 256-dimensional style features',
    'prototypes': 'all 25 normalized family_head.weight rows; bias excluded',
    'target_transform': 'monotone cos(theta_y + angle_radians)',
    'non_target_transform': 'cos(theta_j)',
    'monotone_threshold': 'cos(theta_y) > -cos(angle_radians)',
    'monotone_continuation': 'cos(theta_y) - sin(angle_radians) * angle_radians',
    'norm_epsilon': NORM_EPSILON, 'label_smoothing': 0.,
    'inference_rule': False, 'deployed_operator': False,
    'unknown_rows_are_anchors': False, 'labels_unchanged': True,
    'monotone_guard': True}


def _require(condition, message):
    if not condition:
        raise ValueError('Angular margin loss: ' + message)


def monotone_angular_target(cosine, margin=MARGIN_RADIANS):
    """Return the monotone target cosine used by the angular-margin loss."""
    import torch
    _require(isinstance(cosine, torch.Tensor)
             and cosine.dtype in (torch.float32, torch.float64)
             and cosine.device.type != 'meta' and bool(torch.isfinite(cosine).all()),
             'cosine must be a finite float32/float64 tensor')
    _require(type(margin) is float and 0.0 < margin < math.pi / 2,
             'margin must be a float strictly between zero and pi/2')
    value = cosine.clamp(-1.0, 1.0)
    cos_m, sin_m = math.cos(margin), math.sin(margin)
    sine = torch.sqrt((1.0 - value.square()).clamp_min(0.0) + NORM_EPSILON)
    rotated = value * cos_m - sine * sin_m
    threshold = -cos_m  # cos(pi - margin)
    continuation = value - sin_m * margin
    return torch.where(value > threshold, rotated, continuation)


def angular_margin_loss(features, family_head_weight, targets, rows, families,
                        *, scale=SCALE, margin=MARGIN_RADIANS,
                        coefficient=COEFFICIENT):
    """Return the coefficient-inclusive known-row angular auxiliary scalar.

    Unknown TRAIN rows contribute exactly zero. Known per-row cross entropy is
    weighted by the existing native-core weights, summed, divided by 96, and
    multiplied by ``coefficient``.
    """
    import torch
    from train_unified_retention_core import UNKNOWN, native_core_weights, registry

    registry(families)
    _require(isinstance(features, torch.Tensor) and isinstance(family_head_weight, torch.Tensor)
             and isinstance(targets, torch.Tensor), 'inputs must be Torch tensors')
    _require(features.ndim == 2 and features.shape[0] == BATCH_SIZE
             and family_head_weight.ndim == 2
             and family_head_weight.shape == (len(families), features.shape[1])
             and targets.shape == (BATCH_SIZE,) and targets.dtype == torch.int64,
             'feature, prototype, or target shape differs from the 96-row classifier')
    _require(features.dtype in (torch.float32, torch.float64)
             and family_head_weight.dtype == features.dtype
             and features.device == family_head_weight.device == targets.device
             and features.device.type != 'meta', 'inputs must share a real float32/float64 device')
    _require(bool(torch.isfinite(features).all()) and bool(torch.isfinite(family_head_weight).all())
             and bool(((targets >= 0) & (targets < len(families))).all()),
             'features, prototypes, and targets must be finite and valid')
    _require(type(scale) is float and scale > 0.0 and math.isfinite(scale)
             and type(coefficient) is float and coefficient >= 0.0 and math.isfinite(coefficient),
             'scale and coefficient must be finite floats with valid signs')
    _require(len(rows) == BATCH_SIZE, 'rows must describe the complete 96-row TRAIN batch')
    values = native_core_weights(rows, targets.detach().cpu().tolist(), families)
    known = targets != families.index(UNKNOWN)
    if not bool(known.any()):
        return (features.sum() + family_head_weight.sum()) * 0.0

    known_features = features[known]
    feature_norm = torch.linalg.vector_norm(known_features, dim=1, keepdim=True)
    weight_norm = torch.linalg.vector_norm(family_head_weight, dim=1, keepdim=True)
    _require(bool(torch.isfinite(feature_norm).all()) and bool(torch.isfinite(weight_norm).all())
             and bool((feature_norm > NORM_EPSILON).all())
             and bool((weight_norm > NORM_EPSILON).all()),
             'feature and prototype vectors must have stable nonzero norms')
    normalized_features = known_features / feature_norm.clamp_min(NORM_EPSILON)
    normalized_weight = family_head_weight / weight_norm.clamp_min(NORM_EPSILON)
    cosine = normalized_features @ normalized_weight.transpose(0, 1)
    known_cosine = cosine
    known_targets = targets[known]
    row_index = torch.arange(len(known_targets), device=features.device)
    target_cosine = known_cosine[row_index, known_targets]
    corrected = monotone_angular_target(target_cosine, margin)
    logits = known_cosine.clone()
    logits[row_index, known_targets] = corrected
    per_row = torch.nn.functional.cross_entropy(logits * scale, known_targets, reduction='none')
    weight = torch.tensor(values, dtype=features.dtype, device=features.device)[known]
    result = coefficient * (per_row * weight).sum() / BATCH_SIZE
    _require(result.ndim == 0 and bool(torch.isfinite(result)), 'loss must be a finite scalar')
    return result
