"""CPU contracts for a single frozen encoder with a trainable residual head."""
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from region_network import RegionFontClassifier
from retention_adapter_network import (
    ARCHITECTURE, RESIDUAL_PREFIX, RetentionAdapterClassifier, frozen_state_sha)
from train_regions import state_sha


@pytest.fixture
def parent():
    torch.set_num_threads(2)
    torch.manual_seed(1405)
    model = RegionFontClassifier(25).eval()
    with torch.no_grad():
        model.size_head.weight.normal_(0, .1)
        model.size_head.bias.fill_(.25)
    return model


@pytest.mark.parametrize('batch_size', [1, 3])
def test_initial_features_logits_and_nonzero_sizes_match_parent_exactly(parent, batch_size):
    model = RetentionAdapterClassifier().from_parent(parent).eval()
    tiles = torch.rand(batch_size, 1, 64, 256)
    with torch.no_grad():
        original_features = parent.style(parent.pool(parent.trunk(tiles)))
        features = model.features(tiles)
        original_logits, original_size = parent(tiles)
        logits, sizes = model(tiles)
    assert features.shape == (batch_size, 128)
    assert torch.equal(features, original_features)
    assert torch.equal(logits, original_logits)
    assert torch.equal(sizes, original_size) and sizes.abs().sum() > 0
    assert logits.shape == (batch_size, 25) and sizes.shape == (batch_size,)
    assert ARCHITECTURE == 'region-cnn64x256-residual-head-v1'
    assert isinstance(model.residual_family_head[1], torch.nn.GELU)
    assert model.residual_family_head[0].weight.shape == (128, 128)
    assert model.residual_family_head[2].weight.shape == (25, 128)
    assert not model.residual_family_head[2].weight.any()
    assert not model.residual_family_head[2].bias.any()
    assert sum(isinstance(m, torch.nn.Conv2d) for m in model.modules()) == 4


def test_complete_parent_keys_are_preserved_and_hash_matches_original(parent):
    model = RetentionAdapterClassifier().from_parent(parent.state_dict())
    inherited = {k: v for k, v in model.state_dict().items() if not k.startswith(RESIDUAL_PREFIX)}
    assert inherited.keys() == parent.state_dict().keys()
    assert all(torch.equal(v, parent.state_dict()[k]) for k, v in inherited.items())
    assert frozen_state_sha(model) == frozen_state_sha(model.state_dict()) == state_sha(parent.state_dict())
    before = frozen_state_sha(model)
    with torch.no_grad():
        model.residual_family_head[2].bias.add_(1)
    assert frozen_state_sha(model) == before


def test_optimizer_changes_only_residual_and_size_stays_exact(parent):
    model = RetentionAdapterClassifier().from_parent(parent).train()
    tiles = torch.rand(4, 1, 64, 256)
    labels = torch.tensor([0, 4, 7, 24])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        _, original_size = parent(tiles)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits, sizes = model(tiles)
        assert torch.equal(sizes, original_size)
        torch.nn.functional.cross_entropy(logits, labels).backward()
        assert model.residual_family_head[2].weight.grad.abs().sum() > 0
        # A zero final layer blocks the first-layer gradient on step zero only.
        first_gradient = model.residual_family_head[0].weight.grad.abs().sum()
        assert first_gradient == 0 if step == 0 else first_gradient > 0
        assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith(RESIDUAL_PREFIX))
        optimizer.step()
    after = model.state_dict()
    assert all(torch.equal(before[k], v) for k, v in after.items() if not k.startswith(RESIDUAL_PREFIX))
    assert all(not torch.equal(before[k], v) for k, v in after.items() if k.startswith(RESIDUAL_PREFIX))
    assert frozen_state_sha(model) == state_sha(parent.state_dict())
    with torch.no_grad():
        assert torch.equal(model(tiles)[1], original_size)


def test_cached_features_and_full_forward_remain_identical_after_update(parent):
    model = RetentionAdapterClassifier().from_parent(parent)
    tiles = torch.rand(3, 1, 64, 256)
    with torch.no_grad():
        cached = model.features(tiles).detach().clone()
    optimizer = torch.optim.AdamW(model.residual_family_head.parameters(), lr=.01)
    torch.nn.functional.cross_entropy(model.classify(cached), torch.tensor([1, 4, 24])).backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        logits, sizes = model(tiles)
        assert torch.equal(model.features(tiles), cached)
        assert torch.equal(logits, model.classify(cached))
        assert torch.equal(sizes, model.size_head(cached).squeeze(-1))
        assert not torch.equal(logits, parent(tiles)[0])


def test_train_eval_keeps_every_inherited_module_frozen(parent):
    model = RetentionAdapterClassifier().from_parent(parent)
    for mode in [True, False, True]:
        assert model.train(mode) is model
        assert model.training == mode and model.residual_family_head.training == mode
        for module in (model.trunk, model.pool, model.style, model.family_head, model.size_head):
            assert all(not child.training for child in module.modules())
        assert all(p.requires_grad == n.startswith(RESIDUAL_PREFIX) for n, p in model.named_parameters())


@pytest.mark.parametrize('fault', ['missing', 'extra', 'shape', 'dtype', 'nan', 'family_shape', 'size_shape', 'not_tensor'])
def test_invalid_parent_rejected_before_any_copy(parent, fault):
    model = RetentionAdapterClassifier()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state = {k: v.clone() for k, v in parent.state_dict().items()}
    if fault == 'missing':
        del state['size_head.bias']
    elif fault == 'extra':
        state['unexpected'] = torch.zeros(1)
    elif fault == 'shape':
        state['trunk.0.weight'] = state['trunk.0.weight'][:1]
    elif fault == 'dtype':
        state['trunk.0.weight'] = state['trunk.0.weight'].double()
    elif fault == 'nan':
        state['family_head.bias'][0] = float('nan')
    elif fault == 'family_shape':
        state['family_head.weight'] = state['family_head.weight'][:24]
    elif fault == 'size_shape':
        state['size_head.weight'] = torch.zeros(2, 128)
    else:
        state['trunk.0.weight'] = state['trunk.0.weight'].tolist()
    with pytest.raises(ValueError, match='Parent state|Invalid parent tensor'):
        model.from_parent(state)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize('count', [24, 26, 0, True, 25.0])
def test_invalid_class_count_is_rejected(count):
    with pytest.raises(ValueError, match='exactly 25'):
        RetentionAdapterClassifier(count)


def test_parent_copy_does_not_share_storage_or_reset_residual(parent):
    model = RetentionAdapterClassifier()
    with torch.no_grad():
        model.residual_family_head[2].bias.fill_(.25)
    residual = {k: v.clone() for k, v in model.residual_family_head.state_dict().items()}
    model.from_parent(parent)
    before = frozen_state_sha(model)
    with torch.no_grad():
        parent.trunk[0].weight.add_(1)
        parent.family_head.bias.add_(1)
        parent.size_head.bias.add_(1)
    assert frozen_state_sha(model) == before
    assert all(torch.equal(v, residual[k]) for k, v in model.residual_family_head.state_dict().items())


def test_full_adapter_checkpoint_roundtrip_preserves_outputs(parent):
    model = RetentionAdapterClassifier().from_parent(parent).eval()
    with torch.no_grad():
        model.residual_family_head[2].weight.normal_(0, .01)
    restored = RetentionAdapterClassifier().eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    tiles = torch.rand(2, 1, 64, 256)
    with torch.no_grad():
        assert all(torch.equal(a, b) for a, b in zip(model(tiles), restored(tiles)))
    with pytest.raises(RuntimeError):
        restored.load_state_dict(parent.state_dict(), strict=True)
    with pytest.raises(ValueError, match='unexpected'):
        restored.from_parent(model)
