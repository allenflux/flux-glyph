"""Tests for the 4x16 spatial pooling transfer from the frozen wide CNN."""

import hashlib

import pytest

torch = pytest.importorskip('torch')

from training.spatial_region_network import (
    ARCHITECTURE,
    TRANSFER,
    SpatialRegionFontClassifier,
    expand_spatial_pool,
)
from training.wide_region_network import WideRegionFontClassifier


def _source():
    torch.manual_seed(20260916)
    model = WideRegionFontClassifier(25).cpu().eval()
    # The stock fixture initializes the size head to zero.  Give the transfer
    # parity check a real nonzero affine map as well as the family logits.
    with torch.no_grad():
        model.size_head.weight.copy_(torch.linspace(-.02, .02, model.size_head.weight.numel()).reshape_as(model.size_head.weight))
        model.size_head.bias.fill_(.13)
    return model


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_transfer_repeats_horizontal_coefficients_and_preserves_random_and_structured_outputs():
    source = _source()
    source_before = _state_digest(source)
    rng_before = torch.get_rng_state().clone()
    target, report = expand_spatial_pool(source)

    assert ARCHITECTURE == 'region-cnn64x256-unified-spatial4x16-v1'
    assert target.pool[0].output_size == (4, 16)
    assert tuple(target.style[0].weight.shape) == (384, 8192)
    assert report['all_source_values_inherited'] is True
    assert report['random_new_parameters'] is False
    assert report['source_parameters'] < report['target_parameters']
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert _state_digest(source) == source_before

    source_weight = source.style[0].weight.detach().view(384, 128, 4, 4)
    target_weight = target.style[0].weight.detach().view(384, 128, 4, 16)
    expected = source_weight.repeat_interleave(4, dim=-1) / 4
    torch.testing.assert_close(target_weight, expected, rtol=0, atol=0)

    random_tiles = torch.rand(2, 1, 64, 256)
    structured_tiles = torch.zeros(2, 1, 64, 256)
    structured_tiles[0, :, :, :128] = 0.2
    structured_tiles[0, :, :, 128:] = 0.8
    structured_tiles[1, :, ::4, ::8] = 1.0
    with torch.no_grad():
        for tiles in (random_tiles, structured_tiles):
            source_font, source_size = source(tiles)
            target_font, target_size = target(tiles)
            torch.testing.assert_close(target_font, source_font, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(target_size, source_size, rtol=2e-5, atol=2e-6)


def test_spatial_pool_has_exact_four_subcells_per_source_cell():
    source = _source()
    target, _ = expand_spatial_pool(source)
    feature_map = torch.arange(2 * 128 * 8 * 32, dtype=torch.float32).reshape(2, 128, 8, 32)
    source_pool = source.pool[0](feature_map)
    target_pool = target.pool[0](feature_map)
    subcells = target_pool.reshape(2, 128, 4, 4, 4)
    torch.testing.assert_close(subcells.mean(dim=-1), source_pool, rtol=0, atol=0)


def test_repeated_weights_receive_distinct_gradients_after_spatial_expansion():
    source = _source()
    target, _ = expand_spatial_pool(source)
    target.train()
    tiles = torch.rand(2, 1, 64, 256)
    font, _ = target(tiles)
    (font.square().mean()).backward()
    gradient = target.style[0].weight.grad.detach().view(384, 128, 4, 4, 4)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[..., 1:] - gradient[..., :1]) > 0
    assert target.style[0].weight.requires_grad is True


def test_spatial_model_exports_to_onnx_and_rejects_noncompliant_inputs_or_sources(tmp_path):
    source = _source()
    target, _ = expand_spatial_pool(source)
    example = torch.rand(1, 1, 64, 256)
    output = tmp_path / 'spatial.onnx'
    torch.onnx.export(target, example, output, input_names=['tiles'], output_names=['font', 'size'],
                      opset_version=17, dynamo=False)
    onnx = pytest.importorskip('onnx')
    onnx.checker.check_model(onnx.load(output))

    with pytest.raises(ValueError, match='Spatial font model requires'):
        target(torch.zeros(1, 1, 32, 256))
    with pytest.raises(ValueError):
        expand_spatial_pool(torch.nn.Linear(4, 4))

    altered_pool = _source()
    altered_pool.pool = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((4, 8)), torch.nn.Flatten())
    with pytest.raises(ValueError, match='layout differs'):
        expand_spatial_pool(altered_pool)

    nonfinite = _source()
    with torch.no_grad():
        nonfinite.style[0].weight[0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite CPU'):
        expand_spatial_pool(nonfinite)
