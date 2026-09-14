"""One font CNN with finer horizontal pooling and R22-preserving transfer."""
from __future__ import annotations

import torch
from torch import nn

from wide_region_network import WideRegionFontClassifier

ARCHITECTURE = 'region-cnn64x256-unified-spatial4x16-v1'
TRANSFER = {'schema': 'flux-glyph-spatial-font-transfer-v1',
    'source_pool': [4, 4], 'target_pool': [4, 16], 'horizontal_subcells': 4,
    'incoming_weight_rule': 'repeat each horizontal coefficient four times, divide by four',
    'fixed_input': [1, 64, 256], 'single_encoder': True, 'score_merging': False,
    'new_output_rows': 0, 'random_new_parameters': False,
    'early_downsampling_changed': False, 'loss_and_sampling_changed': False}


class SpatialRegionFontClassifier(WideRegionFontClassifier):
    """Preserve four times as many horizontal trunk cells before the style MLP."""

    def __init__(self, family_count=25):
        super().__init__(family_count)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((4, 16)), nn.Flatten())
        self.style[0] = nn.Linear(8192, 384)

    def forward(self, tiles):
        if not torch.jit.is_tracing() and (tiles.ndim != 4 or tuple(tiles.shape[1:]) != (1, 64, 256)):
            raise ValueError('Spatial font model requires N x 1 x 64 x 256 inputs')
        return super().forward(tiles)


def expand_spatial_pool(source):
    """Transfer the complete CPU R22 model without changing source or RNG state.

    A source 8-column trunk average is the mean of four adjacent 2-column
    averages. Repeating its following linear weight with coefficient 1/4
    preserves that linear map at initialization, up to floating-point order.
    The four coefficients then train independently on distinct spatial inputs.
    """
    if not isinstance(source, nn.Module):
        raise ValueError('Source must be the complete wide font model')
    with torch.random.fork_rng(devices=[]):
        expected = WideRegionFontClassifier(25)
        target = SpatialRegionFontClassifier(25)
    state = source.state_dict(); shapes = expected.state_dict()
    if (set(state) != set(shapes) or any(not isinstance(value, torch.Tensor) for value in state.values())
            or any(value.shape != shapes[name].shape for name, value in state.items())
            or any(value.device.type != 'cpu' or value.dtype not in (torch.float32, torch.float64)
                   or not bool(torch.isfinite(value).all()) for value in state.values())
            or len({value.dtype for value in state.values()}) != 1):
        raise ValueError('Source requires complete finite CPU float32/float64 wide model parameters')
    # Check operations as well as tensor shapes: altered stride or normalization
    # could preserve parameter shapes while breaking the transfer identity.
    source_modules = dict(source.named_modules()); expected_modules = dict(expected.named_modules())
    if set(source_modules) != set(expected_modules) or any(
            type(source_modules[name]) is not type(module)
            or source_modules[name].extra_repr() != module.extra_repr()
            for name, module in expected_modules.items() if name):
        raise ValueError('Source convolution, normalization or 4x4 pooling layout differs')
    target.to(dtype=state['style.0.weight'].dtype)
    transferred = {name: value.detach().clone() for name, value in state.items()}
    weight = state['style.0.weight'].reshape(384, 128, 4, 4)
    transferred['style.0.weight'] = (weight.repeat_interleave(4, dim=-1) / 4).reshape(384, 8192)
    target.load_state_dict(transferred, strict=True)
    target.train(source.training)
    report = {**TRANSFER, 'source_parameters': sum(v.numel() for v in state.values()),
              'target_parameters': sum(v.numel() for v in target.state_dict().values()),
              'all_source_values_inherited': True, 'all_parameters_trainable': True,
              'dtype': str(state['style.0.weight'].dtype).removeprefix('torch.')}
    return target, report
