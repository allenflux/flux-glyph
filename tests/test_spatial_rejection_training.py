"""Actual gradients, exact inherited values, and deployable single-network parity."""
import pytest

torch = pytest.importorskip('torch')
from spatial_region_network import SpatialRegionFontClassifier
from spatial_rejection_network import (add_residual, fold_residual, frozen_state,
                                      residual_check, CachedSpatialHead)
from spatial_rejection_objective import repair_losses
from train_unified_retention_micro_recovery import forward_with_features
from train_regions import state_sha
from native_font_pairs import pair_margin_loss


def test_actual_updates_only_change_spatial_and_unknown_row_and_fold_exactly():
    torch.set_num_threads(4)
    torch.manual_seed(9021)
    base = SpatialRegionFontClassifier(25).double().eval()
    model = add_residual(base)
    before = state_sha(frozen_state(model))
    images = torch.rand(4, 1, 64, 256, dtype=torch.float64)
    with torch.no_grad():
        pooled = model.pool(model.trunk(images))
        for old, new in zip(base(images), model(images)):
            torch.testing.assert_close(old, new, atol=0, rtol=0)
    head = CachedSpatialHead(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    font, size, _ = forward_with_features(head, pooled)
    repair, _ = repair_losses(font, torch.tensor([21, 24, 8, 4]), .5)
    (repair + size.square().mean()).backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().max() > 0
        else:
            assert parameter.grad is None
    optimizer.step()
    assert state_sha(frozen_state(model)) == before
    proof = residual_check(model)
    assert proof['max_absolute_residual'] > 0 and proof['unknown_weight_delta_max_absolute'] > 0
    folded = fold_residual(model)
    assert set(folded.state_dict()) == set(base.state_dict())
    assert sum(p.numel() for p in folded.parameters()) == 3810234
    for name, original in base.state_dict().items():
        if name in ('family_head.weight', 'family_head.bias'):
            assert torch.equal(folded.state_dict()[name][:24], original[:24])
        elif name != 'style.0.weight':
            assert torch.equal(folded.state_dict()[name], original)
    with torch.no_grad():
        for actual, expected, cached in zip(model(images), folded(images), forward_with_features(head, pooled)[:2]):
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(actual, cached, atol=1e-12, rtol=1e-12)


def test_repair_gradient_uses_only_true_non_system_labels_and_is_stable():
    # First three rows have true system fonts: no added supervision is allowed.
    logits = torch.zeros(6, 25, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        logits[3:, 4] = 1000.
    targets = torch.tensor([4, 5, 6, 8, 21, 24])
    loss, parts = repair_losses(logits, targets, .5)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.equal(logits.grad[:3], torch.zeros_like(logits.grad[:3]))
    assert (logits.grad[3:, 4] > 0).all()  # descent suppresses the false PingFang logit
    assert logits.grad[4, 21] < 0 and logits.grad[5, 24] < 0
    only_system = torch.randn(3, 25, dtype=torch.float64, requires_grad=True)
    empty, _ = repair_losses(only_system, targets[:3], .5)
    empty.backward()
    assert empty.item() == 0 and torch.equal(only_system.grad, torch.zeros_like(only_system))


def test_known_unknown_pair_trains_unknown_weight_but_class_bias_cancels():
    model = add_residual(SpatialRegionFontClassifier(25).double())
    features = torch.randn(32, 256, dtype=torch.float64)
    # Actual pair loss validates a 16-pair batch; known/unknown pairs are valid.
    targets = torch.tensor([8, 24] * 16)
    loss = pair_margin_loss(model.family_head(features), targets)
    loss.backward()
    assert model.family_head.unknown_weight_delta.grad.abs().max() > 0
    torch.testing.assert_close(model.family_head.unknown_bias_delta.grad,
                               torch.zeros(1, dtype=torch.float64), atol=1e-12, rtol=0)
    with torch.no_grad():
        model.family_head.unknown_weight_delta[0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        fold_residual(model)
