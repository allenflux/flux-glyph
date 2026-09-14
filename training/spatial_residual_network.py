"""TRAIN-only zero-sum spatial adaptation, folded into one ordinary font CNN."""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F

from spatial_region_network import SpatialRegionFontClassifier

POLICY = {
    'schema': 'flux-glyph-zero-sum-spatial-residual-v1',
    'trainable': ['style.0.residual_raw'],
    'constraint': 'subtract the mean across each four horizontal child cells',
    'frozen': 'all inherited R22 parameters and buffers',
    'folded_architecture': 'region-cnn64x256-unified-spatial4x16-v1',
    'model_count': 1, 'platform_routing': False, 'extra_output_heads': False,
    'runtime_score_changes': False,
}


class ZeroSumSpatialLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        if not isinstance(linear, nn.Linear) or linear.weight.shape != (384, 8192):
            raise ValueError('Expected the inherited 384 x 8192 spatial linear layer')
        if linear.bias is None or linear.bias.shape != (384,):
            raise ValueError('Expected the inherited 384-value bias')
        self.register_buffer('base_weight', linear.weight.detach().clone())
        self.register_buffer('bias', linear.bias.detach().clone())
        self.residual_raw = nn.Parameter(torch.zeros_like(self.base_weight))

    def residual(self):
        children = self.residual_raw.reshape(384, 128, 4, 4, 4)
        return (children - children.mean(dim=-1, keepdim=True)).reshape(384, 8192)

    @property
    def weight(self):
        return self.base_weight + self.residual()

    def forward(self, pooled):
        return F.linear(pooled, self.weight, self.bias)


def add_residual(spatial):
    if not isinstance(spatial, SpatialRegionFontClassifier):
        raise ValueError('Expected the complete inherited spatial font model')
    result = copy.deepcopy(spatial)
    result.requires_grad_(False)
    result.style[0] = ZeroSumSpatialLinear(result.style[0])
    assert [n for n, p in result.named_parameters() if p.requires_grad] == POLICY['trainable']
    return result


def frozen_state(model):
    return {key: value for key, value in model.state_dict().items()
            if key != 'style.0.residual_raw'}


def residual_check(model):
    layer = model.style[0]
    if not isinstance(layer, ZeroSumSpatialLinear):
        raise ValueError('Missing zero-sum training layer')
    residual = layer.residual().detach().reshape(384, 128, 4, 4, 4)
    if not bool(torch.isfinite(residual).all()):
        raise ValueError('Nonfinite spatial residual')
    error = float(residual.sum(-1).abs().max())
    limit = 1e-10 if residual.dtype == torch.float64 else 1e-7
    if error > limit:
        raise ValueError('Spatial residual no longer sums to zero')
    return {'horizontal_sum_max_error': error, 'horizontal_sum_limit': limit,
            'max_absolute_residual': float(residual.abs().max()),
            'residual_l2': float(torch.linalg.vector_norm(residual)),
            'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad)}


def fold_residual(model):
    residual_check(model)
    layer = model.style[0]
    with torch.random.fork_rng(devices=[]):
        result = SpatialRegionFontClassifier(25)
    result.to(device=layer.base_weight.device, dtype=layer.base_weight.dtype)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()
             if key not in ('style.0.base_weight', 'style.0.residual_raw')}
    state['style.0.weight'] = layer.weight.detach().clone()
    result.load_state_dict(state, strict=True)
    result.train(model.training)
    return result


class CachedSpatialHead(nn.Module):
    """Reuse the same head parameters after a verified frozen-TRAIN trunk cache."""
    def __init__(self, model):
        super().__init__()
        self.trunk = nn.Identity()
        self.pool = nn.Identity()
        self.style = model.style
        self.family_head = model.family_head
        self.size_head = model.size_head

