"""Trainable 64-pixel font CNN; compatible with the proven R8 trunk."""
from __future__ import annotations

import torch
from torch import nn

FAMILIES = ['Baoli SC', 'HarmonyOS Sans SC', 'Heiti SC', 'Hiragino Sans GB',
            'Kaiti SC', 'MiSans', 'Noto Sans CJK SC', 'OPPO Sans', 'PingFang SC',
            'Songti SC', 'Yuanti SC', 'SF Pro', 'Helvetica', 'Alipay Number', 'Roboto']
SCRIPTS = {'han': FAMILIES[:11], 'latin': [FAMILIES[i] for i in (1, 5, 7, 11, 12, 13, 14)]}


class FontClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 48, 4, 2, 1), nn.GroupNorm(8, 48), nn.SiLU(),
            nn.Conv2d(48, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU())
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
        self.style = nn.Sequential(nn.Linear(1024, 192), nn.SiLU(), nn.Linear(192, 128))
        self.family_head = nn.Linear(128, len(FAMILIES))

    def forward(self, glyphs):
        return self.family_head(self.style(self.pool(self.trunk(glyphs))))

    def warm_start(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['config']['mode'] != 'expanded_control':
            raise ValueError('Expected the expanded-control R8 parent')
        if checkpoint['classes']['families'] != FAMILIES[:11]:
            raise ValueError('Parent class ordering differs')
        state = self.state_dict()
        with torch.no_grad():
            for name, value in checkpoint['state_dict'].items():
                if name.startswith('family_head.'):
                    state[name][:11].copy_(value)
                else:
                    state[name].copy_(value)
        self.load_state_dict(state, strict=True)
