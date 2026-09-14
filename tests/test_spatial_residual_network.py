"""Exercise conservation, real gradients and the deployed folded network."""
import pytest

torch = pytest.importorskip('torch')
from spatial_region_network import SpatialRegionFontClassifier
from spatial_residual_network import (add_residual, fold_residual, frozen_state,
                                     residual_check, CachedSpatialHead)
from train_regions import state_sha
from train_unified_retention_micro_recovery import forward_with_features


def test_zero_sum_projection_freezes_inherited_state_and_folds_real_updates():
    torch.set_num_threads(4)
    torch.manual_seed(681)
    base = SpatialRegionFontClassifier(25).double().eval()
    model = add_residual(base)
    assert [n for n, p in model.named_parameters() if p.requires_grad] == ['style.0.residual_raw']
    frozen = state_sha(frozen_state(model))
    images = torch.rand(3, 1, 64, 256, dtype=torch.float64)
    with torch.no_grad():
        pooled = model.pool(model.trunk(images))
        old = base(images)
        initial = model(images)
    for expected, observed in zip(old, initial):
        torch.testing.assert_close(expected, observed, atol=0, rtol=0)
    head = CachedSpatialHead(model)
    optimizer = torch.optim.AdamW([model.style[0].residual_raw], lr=1e-3)
    font, size, _ = forward_with_features(head, pooled)
    loss = torch.nn.functional.cross_entropy(font, torch.tensor([1, 4, 8])) + size.square().mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().max() > 0
        else:
            assert parameter.grad is None
    optimizer.step()
    assert state_sha(frozen_state(model)) == frozen
    proof = residual_check(model)
    assert proof['max_absolute_residual'] > 0 and proof['horizontal_sum_max_error'] < 1e-10
    # A uniform child-cell component must be ignored, even with nonzero Adam state.
    with torch.no_grad():
        projected = model.style[0].residual().clone()
        model.style[0].residual_raw.add_(.03125)
    torch.testing.assert_close(projected, model.style[0].residual(), atol=1e-16, rtol=1e-12)
    folded = fold_residual(model)
    assert set(folded.state_dict()) == set(base.state_dict())
    assert sum(p.numel() for p in folded.parameters()) == 3810234
    assert all('residual' not in name for name in folded.state_dict())
    with torch.no_grad():
        trained = model(images)
        exported = folded(images)
        cached = forward_with_features(head, pooled)[:2]
    for actual, expected, via_cache in zip(trained, exported, cached):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual, via_cache, atol=1e-12, rtol=1e-12)


def test_projection_rejects_nonfinite_residual():
    model = add_residual(SpatialRegionFontClassifier(25))
    with torch.no_grad():
        model.style[0].residual_raw[0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        fold_residual(model)
