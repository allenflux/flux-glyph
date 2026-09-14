"""Keep the unknown/false-system supervision exact while reducing rare-font pressure."""
import pytest

torch = pytest.importorskip('torch')
from spatial_rejection_objective import repair_losses as previous
from spatial_balanced_objective import repair_losses, POLICY


@pytest.mark.parametrize('targets', [[21, 24, 8, 4, 5, 6], [24, 8, 4, 5, 6, 2]])
def test_only_extra_kaiti_weight_changes_and_gradients_stay_finite(targets):
    torch.manual_seed(102)
    logits = torch.randn(len(targets), 25, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor(targets)
    old, old_parts = previous(logits, labels, .5)
    new, new_parts = repair_losses(logits, labels, .5)
    assert POLICY['kaiti_extra_cross_entropy_weight'] == .05
    for key in ('repair_unknown_ce', 'repair_kaiti_ce', 'repair_system_confusion'):
        torch.testing.assert_close(old_parts[key], new_parts[key], atol=0, rtol=0)
    torch.testing.assert_close(new, old - .95 * old_parts['repair_kaiti_ce'], atol=1e-14, rtol=1e-14)
    new.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[labels == 24, 24].item() < 0
    assert torch.equal(logits.grad[labels == 4], torch.zeros_like(logits.grad[labels == 4]))
