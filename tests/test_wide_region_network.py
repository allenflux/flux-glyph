"""CPU checks for deterministic function-preserving region-model widening."""
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))

from region_network import RegionFontClassifier
from wide_region_network import WideRegionFontClassifier, widen_region_model


def source(dtype=torch.float32):
    torch.manual_seed(1414)
    model = RegionFontClassifier(25).to(dtype=dtype).eval()
    with torch.no_grad():
        model.size_head.weight.normal_(0., .03)
        model.size_head.bias.fill_(.17)
    return model


@pytest.mark.parametrize('dtype,rtol,atol', [
    (torch.float32, 4e-5, 4e-5),
    (torch.float64, 1e-11, 1e-11),
])
def test_initial_font_and_nonzero_learned_size_outputs_match(dtype, rtol, atol):
    original = source(dtype)
    widened, report = widen_region_model(original)
    tiles = torch.rand(2, 1, 64, 256, dtype=dtype)
    with torch.no_grad():
        expected_font, expected_size = original(tiles)
        actual_font, actual_size = widened(tiles)
    assert expected_size.abs().sum() > 0
    torch.testing.assert_close(actual_font, expected_font, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_size, expected_size, rtol=rtol, atol=atol)
    assert report['function_preserving_initialization'] is True
    assert report['single_encoder'] is True and report['score_merging'] is False


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_outputs_are_batch_invariant(dtype):
    widened, _ = widen_region_model(source(dtype))
    widened.eval()
    tiles = torch.rand(3, 1, 64, 256, dtype=dtype)
    with torch.no_grad():
        together = widened(tiles)
        separate = [torch.cat([widened(tile[None])[head] for tile in tiles]) for head in (0, 1)]
    for actual, expected in zip(together, separate):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_transfer_is_deterministic_does_not_consume_global_rng_and_seed_changes_splits():
    original = source(torch.float64)
    state = torch.random.get_rng_state().clone()
    first, first_report = widen_region_model(original, seed=2026091414)
    assert torch.equal(torch.random.get_rng_state(), state)
    second, second_report = widen_region_model(original, seed=2026091414)
    different, different_report = widen_region_model(original, seed=2026091415)
    assert first_report == second_report and first_report != different_report
    assert all(torch.equal(value, second.state_dict()[name]) for name, value in first.state_dict().items())
    assert any(not torch.equal(value, different.state_dict()[name])
               for name, value in first.state_dict().items())


def test_source_is_unchanged_and_widened_parameters_do_not_share_storage():
    original = source()
    before = {name: value.clone() for name, value in original.state_dict().items()}
    widened, _ = widen_region_model(original)
    assert all(torch.equal(value, original.state_dict()[name]) for name, value in before.items())
    with torch.no_grad():
        widened.trunk[0].weight.add_(1.)
        widened.family_head.bias.add_(1.)
        widened.size_head.weight.add_(1.)
    assert all(torch.equal(value, original.state_dict()[name]) for name, value in before.items())


def test_expanded_layout_groupnorm_and_parameter_report_are_exact():
    original = source()
    widened, report = widen_region_model(original)
    assert [module.weight.shape for module in widened.trunk if isinstance(module, torch.nn.Conv2d)] == [
        (64, 1, 3, 3), (96, 64, 4, 4), (128, 96, 4, 4), (128, 128, 4, 4)]
    norms = [module for module in widened.trunk if isinstance(module, torch.nn.GroupNorm)]
    assert [(module.num_groups, module.num_channels) for module in norms] == [
        (16, 64), (16, 96), (16, 128), (16, 128)]
    assert widened.style[0].weight.shape == (384, 2048)
    assert widened.style[2].weight.shape == (256, 384)
    assert widened.family_head.weight.shape == (25, 256)
    assert widened.size_head.weight.shape == (1, 256)
    assert report['source_parameters'] == sum(p.numel() for p in original.parameters())
    assert report['widened_parameters'] == sum(p.numel() for p in widened.parameters())
    assert report['widened_parameters'] > 3 * report['source_parameters']
    assert all(.25 <= bounds['minimum'] < bounds['maximum'] <= .75
               for bounds in report['incoming_split_ranges'].values())


def test_transfer_duplicates_whole_vectors_and_each_incoming_pair_sums_to_source():
    original = source(torch.float64)
    widened, _ = widen_region_model(original)
    torch.testing.assert_close(widened.trunk[0].weight[:32], original.trunk[0].weight)
    torch.testing.assert_close(widened.trunk[0].weight[32:], original.trunk[0].weight)
    for old_norm, new_norm in zip(
            [module for module in original.trunk if isinstance(module, torch.nn.GroupNorm)],
            [module for module in widened.trunk if isinstance(module, torch.nn.GroupNorm)]):
        torch.testing.assert_close(new_norm.weight[:len(old_norm.weight)], old_norm.weight)
        torch.testing.assert_close(new_norm.weight[len(old_norm.weight):], old_norm.weight)
    pairs = [
        (original.trunk[3].weight, widened.trunk[3].weight),
        (original.trunk[6].weight, widened.trunk[6].weight),
        (original.trunk[9].weight, widened.trunk[9].weight),
        (original.style[0].weight, widened.style[0].weight),
        (original.style[2].weight, widened.style[2].weight),
    ]
    for old, new in pairs:
        expected = torch.cat((old, old), 0)
        torch.testing.assert_close(new[:, :old.shape[1]] + new[:, old.shape[1]:], expected)
    torch.testing.assert_close(
        widened.family_head.weight[:, :128] + widened.family_head.weight[:, 128:],
        original.family_head.weight)
    torch.testing.assert_close(
        widened.size_head.weight[:, :128] + widened.size_head.weight[:, 128:],
        original.size_head.weight)


def test_whole_vector_duplication_preserves_flatten_order():
    original = source(torch.float64)
    widened, _ = widen_region_model(original)
    tiles = torch.rand(2, 1, 64, 256, dtype=torch.float64)
    with torch.no_grad():
        old_flat = original.pool(original.trunk(tiles))
        wide_flat = widened.pool(widened.trunk(tiles))
    assert old_flat.shape == (2, 1024) and wide_flat.shape == (2, 2048)
    torch.testing.assert_close(wide_flat[:, :1024], old_flat, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(wide_flat[:, 1024:], old_flat, rtol=1e-11, atol=1e-11)


def test_nonuniform_splits_break_replica_gradient_symmetry_and_all_groups_update():
    widened, _ = widen_region_model(source())
    widened.train()
    before = {name: value.clone() for name, value in widened.state_dict().items()}
    tiles = torch.rand(3, 1, 64, 256)
    targets = torch.tensor([1, 7, 24])
    logits, sizes = widened(tiles)
    loss = torch.nn.functional.cross_entropy(logits, targets) + .2 * (sizes - torch.tensor([.1, -.2, .3])).square().mean()
    optimizer = torch.optim.SGD(widened.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    first = widened.trunk[0].weight.grad
    final_style = widened.style[2].weight.grad
    assert not torch.allclose(first[:32], first[32:])
    assert not torch.allclose(final_style[:128], final_style[128:])
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
               for parameter in widened.parameters())
    optimizer.step()
    after = widened.state_dict()
    for group in ('trunk', 'style', 'family_head', 'size_head'):
        assert any(not torch.equal(before[name], value)
                   for name, value in after.items() if name.startswith(group + '.'))


@pytest.mark.parametrize('fault', ['family_count', 'source_type', 'dtype', 'nonfinite', 'seed'])
def test_invalid_sources_and_seed_are_rejected(fault):
    original = source()
    if fault == 'family_count':
        with pytest.raises(ValueError):
            WideRegionFontClassifier(24)
        return
    if fault == 'source_type': original = original.state_dict()
    elif fault == 'dtype': original = original.half()
    elif fault == 'nonfinite':
        with torch.no_grad(): original.trunk[0].weight[0, 0, 0, 0] = float('nan')
    seed = True if fault == 'seed' else 2026091414
    with pytest.raises(ValueError):
        widen_region_model(original, seed)


@pytest.mark.parametrize('fault', ['groups', 'epsilon', 'pool'])
def test_mutated_source_normalization_or_pool_layout_is_rejected(fault):
    original = source()
    if fault == 'groups': original.trunk[4].num_groups = 4
    elif fault == 'epsilon': original.trunk[7].eps = 1e-4
    else: original.pool[0].output_size = (2, 4)
    with pytest.raises(ValueError, match='normalization or 4x4 pooling'):
        widen_region_model(original)
