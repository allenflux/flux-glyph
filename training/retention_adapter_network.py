"""A zero-initialized residual font head over one immutable region CNN.

The original parameter names and size computation are preserved. Only the
residual head is trainable; cached features and complete-image inference use
the same family-head computation.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib

import torch
from torch import nn

from region_network import RegionFontClassifier


ARCHITECTURE = 'region-cnn64x256-residual-head-v1'
RESIDUAL_PREFIX = 'residual_family_head.'


def frozen_state_sha(model_or_state):
    """Hash inherited tensors with the original training state_sha algorithm."""
    state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
    if not isinstance(state, Mapping):
        raise ValueError('Expected a model or parameter mapping')
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not name.startswith(RESIDUAL_PREFIX):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class RetentionAdapterClassifier(RegionFontClassifier):
    def __init__(self, family_count=25):
        if type(family_count) is not int or family_count != 25:
            raise ValueError('Retention adapter requires exactly 25 font classes')
        super().__init__(family_count)
        self.family_count = family_count
        self.residual_family_head = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(), nn.Linear(128, family_count))
        nn.init.zeros_(self.residual_family_head[2].weight)
        nn.init.zeros_(self.residual_family_head[2].bias)
        self._freeze_parent()

    def _freeze_parent(self):
        for module in (self.trunk, self.pool, self.style, self.family_head, self.size_head):
            module.requires_grad_(False)
            module.eval()

    def train(self, mode=True):
        super().train(mode)
        self._freeze_parent()
        return self

    def from_parent(self, parent):
        """Copy a complete 25-class parent, preserving the new residual head.

        Accept a RegionFontClassifier or its state_dict. Validate every tensor
        before copying any value, and do not share caller-owned storage. The
        trainer separately binds the checkpoint identity and class ordering.
        Full adapter checkpoints use ordinary load_state_dict(strict=True).
        """
        state = parent.state_dict() if isinstance(parent, RegionFontClassifier) else parent
        if not isinstance(state, Mapping):
            raise ValueError('Parent state must be a complete parameter mapping')
        current = self.state_dict()
        inherited = {name: value for name, value in current.items()
                     if not name.startswith(RESIDUAL_PREFIX)}
        if set(state) != set(inherited):
            raise ValueError('Parent state has missing or unexpected parameters')
        for name, expected in inherited.items():
            value = state[name]
            if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                    or value.dtype != expected.dtype or not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid parent tensor: ' + name)
        for name, value in state.items():
            current[name] = value.detach().clone()
        self.load_state_dict(current, strict=True)
        self._freeze_parent()
        return self

    def features(self, tiles):
        return self.style(self.pool(self.trunk(tiles)))

    def classify(self, features):
        return self.family_head(features) + self.residual_family_head(features)

    def forward(self, tiles):
        features = self.features(tiles)
        return self.classify(features), self.size_head(features).squeeze(-1)
