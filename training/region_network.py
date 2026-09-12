"""OCR-free region font and size model, initialized from the glyph CNN."""
from __future__ import annotations

import torch
from torch import nn

from network import FontClassifier


class RegionFontClassifier(nn.Module):
    def __init__(self, family_count):
        super().__init__()
        if not 2 <= family_count <= 64:
            raise ValueError('Expected between 2 and 64 trained font families')
        glyph = FontClassifier()
        self.trunk = glyph.trunk
        self.pool = glyph.pool
        self.style = glyph.style
        self.family_head = nn.Linear(128, family_count)
        self.size_head = nn.Linear(128, 1)
        nn.init.zeros_(self.size_head.weight)
        nn.init.zeros_(self.size_head.bias)

    def forward(self, tiles):
        features = self.style(self.pool(self.trunk(tiles)))
        return self.family_head(features), self.size_head(features).squeeze(-1)

    def warm_start(self, checkpoint_path, families):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        names = checkpoint.get('families')
        if not isinstance(names, list) or len(set(names)) != len(names) or not set(families) <= set(names):
            raise ValueError('Glyph checkpoint cannot cover the region font family registry')
        indices = [names.index(family) for family in families]
        state = self.state_dict()
        expected = {key for key in state if not key.startswith('size_head.')}
        received = {key for key in checkpoint['state_dict'] if not key.startswith('size_head.')}
        if received != expected:
            raise ValueError('Glyph checkpoint has incomplete or unexpected network parameters')
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('size_head.'):
                continue
            state[key] = value[indices].clone() if key.startswith('family_head.') else value.clone()
        self.load_state_dict(state, strict=True)
        return names
