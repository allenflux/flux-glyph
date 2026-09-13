"""Independent Android font evidence heads on an immutable image encoder.

The tenth output is a fixed zero baseline, not a learned common unknown style.
No reference images, OCR text, or platform labels are model inputs.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from region_network import RegionFontClassifier


ARCHITECTURE = 'android_ovr_frozen_encoder_9x_mlp128_32_1_v1'


class AndroidOVRClassifier(nn.Module):
    def __init__(self, family_count=10, unknown_index=9):
        super().__init__()
        if (type(family_count) is not int or family_count != 10
                or type(unknown_index) is not int or not 0 <= unknown_index < family_count):
            raise ValueError('OVR requires nine known families and one valid unknown index')
        parent = RegionFontClassifier(family_count)
        self.family_count = family_count
        self.unknown_index = unknown_index
        self.trunk = parent.trunk
        self.pool = parent.pool
        self.style = parent.style
        self.size_head = parent.size_head
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 32), nn.SiLU(), nn.Linear(32, 1))
            for _ in range(family_count - 1)
        ])
        self._freeze_parent()

    def _freeze_parent(self):
        for module in (self.trunk, self.pool, self.style, self.size_head):
            module.requires_grad_(False)
            module.eval()

    def train(self, mode=True):
        super().train(mode)
        # Preserve inference behavior for the inherited encoder even when the
        # independent heads train. This also protects future stateful layers.
        self._freeze_parent()
        return self

    def from_parent(self, state_dict):
        """Strictly copy the complete ten-class parent without retaining its head.

        Every parameter is checked before any value is copied. Tensors are
        detached and cloned so subsequent changes to a caller's state cannot
        modify this model. Source file/selection hashes belong to the trainer.
        """
        if not isinstance(state_dict, Mapping):
            raise ValueError('Parent state must be a complete parameter mapping')
        state = self.state_dict()
        inherited = {name: value for name, value in state.items() if not name.startswith('heads.')}
        shapes = {name: tuple(value.shape) for name, value in inherited.items()}
        shapes.update({'family_head.weight': (self.family_count, 128),
                       'family_head.bias': (self.family_count,)})
        if set(state_dict) != set(shapes):
            raise ValueError('Parent state has missing or unexpected parameters')
        dtype = self.size_head.weight.dtype
        for name, shape in shapes.items():
            value = state_dict[name]
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or value.dtype != dtype or not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid parent tensor: ' + name)
        for name in inherited:
            state[name] = state_dict[name].detach().clone()
        self.load_state_dict(state, strict=True)
        self._freeze_parent()
        return self

    def features(self, tiles):
        return self.style(self.pool(self.trunk(tiles)))

    def classify(self, features):
        known = torch.cat([head(features) for head in self.heads], dim=1)
        unknown = torch.zeros_like(known[:, :1])
        return torch.cat((known[:, :self.unknown_index], unknown,
                          known[:, self.unknown_index:]), dim=1)

    def forward(self, tiles):
        features = self.features(tiles)
        return self.classify(features), self.size_head(features).squeeze(-1)
