"""Image-only known/unknown network; the font classifier remains frozen."""
from __future__ import annotations

import torch
from torch import nn

from region_network import RegionFontClassifier


class RegionRejector(nn.Module):
    def __init__(self, family_count=8):
        super().__init__()
        source = RegionFontClassifier(family_count)
        self.trunk, self.pool, self.style = source.trunk, source.pool, source.style
        for module in (self.trunk, self.pool, self.style):
            module.requires_grad_(False)
        self.register_buffer('feature_mean', torch.zeros(128))
        self.register_buffer('feature_scale', torch.ones(128))
        self.head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Dropout(.1), nn.Linear(64, 2))

    def features(self, tiles):
        return self.style(self.pool(self.trunk(tiles)))

    def classify_features(self, features):
        return self.head((features - self.feature_mean) / self.feature_scale)

    def forward(self, tiles):
        return self.classify_features(self.features(tiles))

    def initialize_encoder(self, state):
        for name in ('trunk', 'pool', 'style'):
            prefix = name + '.'
            getattr(self, name).load_state_dict({key[len(prefix):]: value for key, value in state.items()
                                               if key.startswith(prefix)}, strict=True)
