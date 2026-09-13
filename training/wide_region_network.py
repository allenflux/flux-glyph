"""A doubled-width region CNN and deterministic function-preserving transfer."""
from __future__ import annotations

import torch
from torch import nn


class WideRegionFontClassifier(nn.Module):
    """One 25-class region CNN with twice the hidden width of the original."""

    def __init__(self, family_count=25):
        super().__init__()
        if type(family_count) is not int or family_count != 25:
            raise ValueError('Wide region model requires exactly 25 font families')
        self.trunk = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.GroupNorm(16, 64), nn.SiLU(),
            nn.Conv2d(64, 96, 4, 2, 1), nn.GroupNorm(16, 96), nn.SiLU(),
            nn.Conv2d(96, 128, 4, 2, 1), nn.GroupNorm(16, 128), nn.SiLU(),
            nn.Conv2d(128, 128, 4, 2, 1), nn.GroupNorm(16, 128), nn.SiLU())
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
        self.style = nn.Sequential(nn.Linear(2048, 384), nn.SiLU(), nn.Linear(384, 256))
        self.family_head = nn.Linear(256, family_count)
        self.size_head = nn.Linear(256, 1)
        nn.init.zeros_(self.size_head.weight)
        nn.init.zeros_(self.size_head.bias)

    def forward(self, tiles):
        features = self.style(self.pool(self.trunk(tiles)))
        return self.family_head(features), self.size_head(features).squeeze(-1)


_SOURCE_SHAPES = {
    'trunk.0.weight': (32, 1, 3, 3), 'trunk.0.bias': (32,),
    'trunk.1.weight': (32,), 'trunk.1.bias': (32,),
    'trunk.3.weight': (48, 32, 4, 4), 'trunk.3.bias': (48,),
    'trunk.4.weight': (48,), 'trunk.4.bias': (48,),
    'trunk.6.weight': (64, 48, 4, 4), 'trunk.6.bias': (64,),
    'trunk.7.weight': (64,), 'trunk.7.bias': (64,),
    'trunk.9.weight': (64, 64, 4, 4), 'trunk.9.bias': (64,),
    'trunk.10.weight': (64,), 'trunk.10.bias': (64,),
    'style.0.weight': (192, 1024), 'style.0.bias': (192,),
    'style.2.weight': (128, 192), 'style.2.bias': (128,),
    'family_head.weight': (25, 128), 'family_head.bias': (25,),
    'size_head.weight': (1, 128), 'size_head.bias': (1,),
}


def _source_state(source):
    if not isinstance(source, nn.Module):
        raise ValueError('Source must be one original RegionFontClassifier module')
    norms = [source.trunk[index] for index in (1, 4, 7, 10)] if (
        isinstance(getattr(source, 'trunk', None), nn.Sequential)
        and len(source.trunk) == 12) else []
    pool = getattr(source, 'pool', None)
    if (len(norms) != 4 or any(not isinstance(norm, nn.GroupNorm) for norm in norms)
            or [(norm.num_groups, norm.num_channels) for norm in norms]
                != [(8, 32), (8, 48), (8, 64), (8, 64)]
            or any(norm.eps != 1e-5 or norm.affine is not True for norm in norms)
            or not isinstance(pool, nn.Sequential) or len(pool) != 2
            or not isinstance(pool[0], nn.AdaptiveAvgPool2d) or pool[0].output_size != (4, 4)
            or not isinstance(pool[1], nn.Flatten)
            or pool[1].start_dim != 1 or pool[1].end_dim != -1):
        raise ValueError('Source normalization or 4x4 pooling layout differs')
    state = source.state_dict()
    if set(state) != set(_SOURCE_SHAPES):
        raise ValueError('Source region model has incomplete or unexpected parameters')
    values = list(state.values())
    if (not values or any(not isinstance(value, torch.Tensor) for value in values)
            or any(tuple(value.shape) != _SOURCE_SHAPES[name] for name, value in state.items())
            or any(value.device.type != 'cpu' for value in values)
            or len({value.dtype for value in values}) != 1
            or values[0].dtype not in (torch.float32, torch.float64)
            or any(not bool(torch.isfinite(value).all()) for value in values)):
        raise ValueError('Source must contain finite CPU float32 or float64 region parameters')
    return state, values[0].dtype


def _split_weight(weight, generator, *, outputs):
    """Duplicate output rows and split every duplicated input connection."""
    base = torch.cat((weight, weight), dim=0) if outputs else weight
    alpha = .25 + .5 * torch.rand(base.shape, generator=generator, dtype=torch.float64)
    alpha = alpha.to(dtype=weight.dtype)
    widened = torch.cat((base * alpha, base * (1. - alpha)), dim=1)
    return widened, alpha


def _duplicate(value):
    return torch.cat((value, value), dim=0)


def widen_region_model(source, seed=2026091414):
    """Return a deterministic Net2Wider initialization without mutating ``source``.

    Hidden vectors are laid out as ``[original, original]``. Each following
    weight tensor splits every original incoming connection into alpha and
    1-alpha parts, so the two equal inputs sum to the original contribution.
    Whole channel vectors are duplicated, preserving the original GroupNorm
    per-group channel counts of 4, 6, 8 and 8 as groups double from 8 to 16.
    """
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Transfer seed must be a nonnegative 63-bit integer')
    state, dtype = _source_state(source)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        widened = WideRegionFontClassifier(25).to(dtype=dtype)
    target = widened.state_dict()
    alpha_ranges = {}

    def split(name, *, outputs=True):
        value, alpha = _split_weight(state[name], generator, outputs=outputs)
        target[name].copy_(value)
        alpha_ranges[name] = {'minimum': float(alpha.min()), 'maximum': float(alpha.max())}

    with torch.no_grad():
        target['trunk.0.weight'].copy_(_duplicate(state['trunk.0.weight']))
        target['trunk.0.bias'].copy_(_duplicate(state['trunk.0.bias']))
        for norm in (1, 4, 7, 10):
            target[f'trunk.{norm}.weight'].copy_(_duplicate(state[f'trunk.{norm}.weight']))
            target[f'trunk.{norm}.bias'].copy_(_duplicate(state[f'trunk.{norm}.bias']))
        for convolution in (3, 6, 9):
            split(f'trunk.{convolution}.weight')
            target[f'trunk.{convolution}.bias'].copy_(_duplicate(state[f'trunk.{convolution}.bias']))
        split('style.0.weight')
        target['style.0.bias'].copy_(_duplicate(state['style.0.bias']))
        split('style.2.weight')
        target['style.2.bias'].copy_(_duplicate(state['style.2.bias']))
        split('family_head.weight', outputs=False)
        target['family_head.bias'].copy_(state['family_head.bias'])
        split('size_head.weight', outputs=False)
        target['size_head.bias'].copy_(state['size_head.bias'])
    widened.load_state_dict(target, strict=True)
    widened.train(source.training)
    source_parameters = sum(value.numel() for value in state.values())
    widened_parameters = sum(value.numel() for value in target.values())
    report = {
        'schema': 'flux-glyph-wide-region-transfer-v1',
        'method': 'Net2Wider with whole-vector GroupNorm duplication',
        'seed': seed, 'family_count': 25, 'dtype': str(dtype).removeprefix('torch.'),
        'source_parameters': source_parameters, 'widened_parameters': widened_parameters,
        'source_widths': [32, 48, 64, 64, 192, 128],
        'widened_widths': [64, 96, 128, 128, 384, 256],
        'source_groupnorm_groups': 8, 'widened_groupnorm_groups': 16,
        'pool_shape': [4, 4], 'single_encoder': True, 'score_merging': False,
        'function_preserving_initialization': True,
        'incoming_split_bounds': [.25, .75], 'incoming_split_ranges': alpha_ranges,
    }
    if not all(.25 <= bounds['minimum'] <= bounds['maximum'] <= .75
               for bounds in alpha_ranges.values()):
        raise ValueError('Deterministic widening split escaped its declared bounds')
    return widened, report
