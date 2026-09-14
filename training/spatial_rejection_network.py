"""Train fine spatial features and the existing unknown class, then fold one CNN."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from spatial_residual_network import (
    add_residual as add_spatial_residual, residual_check as spatial_check,
    CachedSpatialHead,
)
from spatial_region_network import SpatialRegionFontClassifier

POLICY = {
    'schema': 'flux-glyph-spatial-rejection-residual-v1',
    'trainable': ['style.0.residual_raw', 'family_head.unknown_weight_delta',
                  'family_head.unknown_bias_delta'],
    'constraint': 'zero-sum horizontal spatial residual; only existing unknown classifier row may adapt',
    'frozen': 'all inherited tensors; the 24 known classifier rows remain bitwise unchanged',
    'folded_architecture': 'region-cnn64x256-unified-spatial4x16-v1',
    'model_count': 1, 'platform_routing': False, 'extra_output_heads': False,
    'runtime_score_changes': False,
}


class UnknownDeltaLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        if not isinstance(linear, nn.Linear) or linear.weight.shape != (25, 256):
            raise ValueError('Expected the canonical 25 x 256 font classifier')
        if linear.bias is None or linear.bias.shape != (25,):
            raise ValueError('Expected the canonical 25-value font bias')
        self.register_buffer('base_weight', linear.weight.detach().clone())
        self.register_buffer('base_bias', linear.bias.detach().clone())
        self.unknown_weight_delta = nn.Parameter(torch.zeros_like(self.base_weight[24:]))
        self.unknown_bias_delta = nn.Parameter(torch.zeros_like(self.base_bias[24:]))

    @property
    def weight(self):
        return torch.cat((self.base_weight[:24], self.base_weight[24:] + self.unknown_weight_delta))

    @property
    def bias(self):
        return torch.cat((self.base_bias[:24], self.base_bias[24:] + self.unknown_bias_delta))

    def forward(self, features):
        return F.linear(features, self.weight, self.bias)


def add_residual(spatial):
    model = add_spatial_residual(spatial)
    model.family_head = UnknownDeltaLinear(model.family_head)
    assert [n for n, p in model.named_parameters() if p.requires_grad] == POLICY['trainable']
    return model


def frozen_state(model):
    return {key: value for key, value in model.state_dict().items() if key not in POLICY['trainable']}


def residual_check(model):
    report = spatial_check(model)
    if not isinstance(model.family_head, UnknownDeltaLinear):
        raise ValueError('Missing unknown-row adaptation')
    for key in ('unknown_weight_delta', 'unknown_bias_delta'):
        value = getattr(model.family_head, key).detach()
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite unknown-row residual')
        report[key + '_max_absolute'] = float(value.abs().max())
    return report


def fold_residual(model):
    residual_check(model)
    layer = model.style[0]
    with torch.random.fork_rng(devices=[]):
        result = SpatialRegionFontClassifier(25)
    result.to(device=layer.base_weight.device, dtype=layer.base_weight.dtype)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()
             if key not in POLICY['trainable'] and key not in (
                 'style.0.base_weight', 'family_head.base_weight', 'family_head.base_bias')}
    state.update({'style.0.weight': layer.weight.detach().clone(),
                  'family_head.weight': model.family_head.weight.detach().clone(),
                  'family_head.bias': model.family_head.bias.detach().clone()})
    result.load_state_dict(state, strict=True)
    result.train(model.training)
    return result
