"""The region model must run using source pixels without any text recognizer."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import flux_glyph.pipeline as module
from flux_glyph.models import file_sha, verify_bundle


def write_bundle(root, omitted=None):
    files = {
        'pp/onnx/paddle_ocr_det.onnx': b'detector',
        'pp/paddle_ocr_delivery.contract.json': b'{}',
        'region_neural/metadata.json': json.dumps({'model': {'path': 'model.onnx'}}).encode(),
        'region_neural/model.onnx': b'region weights',
    }
    rows = []
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        if name != omitted:
            rows.append({'path': name, 'bytes': len(value), 'sha256': file_sha(path)})
    (root / 'MANIFEST.json').write_text(json.dumps({'version': 'region-test', 'files': rows}))


def test_detector_and_region_weights_are_a_complete_bundle(tmp_path):
    write_bundle(tmp_path)
    assert len(verify_bundle(tmp_path)['files']) == 4


def test_region_package_manifest_does_not_require_legacy_font_directory(tmp_path):
    from scripts.package_models import write_models_manifest
    write_bundle(tmp_path)
    manifest = write_models_manifest(tmp_path)
    assert len(manifest['files']) == 4
    assert all(row['path'].startswith(('pp/', 'region_neural/')) for row in manifest['files'])
    assert verify_bundle(tmp_path) == manifest


@pytest.mark.parametrize('omitted', [
    'region_neural/model.onnx', 'region_neural/metadata.json',
    'pp/onnx/paddle_ocr_det.onnx', 'pp/paddle_ocr_delivery.contract.json',
])
def test_even_on_disk_region_dependencies_must_be_hash_declared(tmp_path, omitted):
    write_bundle(tmp_path, omitted)
    with pytest.raises(ValueError, match='lacks'):
        verify_bundle(tmp_path)


def test_region_constructor_and_run_never_use_ocr_or_reference_banks(monkeypatch, tmp_path):
    write_bundle(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('OCR, segmentation and reference matching must never be used')
    for name in ('PPReader', 'CompactFontBank', 'CompactLatinBank',
                 'NeuralFontClassifier', 'segment_characters', 'score_font'):
        monkeypatch.setattr(module, name, forbidden)
    boxes = [dict(source_bbox=[12, 10, 80, 40],
                  quad=[[12, 10], [80, 10], [80, 40], [12, 40]], score=.99)] * 2
    monkeypatch.setattr(module, 'PPRegionDetector', lambda _: SimpleNamespace(detect=lambda _: boxes))
    crops = []
    def predict(crop):
        crops.append(crop.copy())
        return dict(status='candidate', family='SF Pro', candidates=[dict(family='SF Pro', score=.99)],
                    score=.99, margin=.9, patch_agreement=1., reason_code='region_neural_family_candidate',
                    font_size_px_estimate=23.5, size_relative_spread=.01, ocr_performed=False)
    monkeypatch.setattr(module, 'RegionFontClassifier', lambda _: SimpleNamespace(predict=predict))
    engine = module.FontPipeline(tmp_path, max_regions=1)
    assert engine.reader is engine.bank is engine.latin_bank is engine.neural is engine.size_metrics is None
    yy, xx = np.indices((60, 100))
    source = Image.fromarray(np.stack((xx, yy, xx + yy), axis=2).astype(np.uint8))
    source_path = tmp_path / 'source.png'
    source.save(source_path)
    result = engine.run(source_path, tmp_path / 'output', 'test')
    assert result['ocr_performed'] is False
    assert result['font_method'] == 'region_neural_network'
    assert result['summary']['processing_limited_regions'] == 1
    assert 'ocr' not in result['timing_seconds']
    assert len(crops) == 1
    first, capped = result['regions']
    np.testing.assert_array_equal(np.asarray(crops[0]), np.asarray(source.crop(first['source_bbox'])))
    assert first['font']['family'] == 'SF Pro'
    assert first['text_style']['font_size_px_estimate'] == 23.5
    assert first['text_style']['font_size_px_interval'] is None
    assert capped['font']['reason_code'] == 'too_many_regions'
    for region in result['regions']:
        assert region['text'] is None and region['ocr_performed'] is False and region['glyphs'] == []
        with Image.open(tmp_path / 'output' / region['crop_file']) as crop:
            np.testing.assert_array_equal(np.asarray(crop), np.asarray(source.crop(region['source_bbox'])))
