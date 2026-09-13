"""Preview status must survive inference, saved results and the public API."""
import json
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest

from flux_glyph.api import public_result
from flux_glyph.pipeline import FontPipeline


@pytest.mark.parametrize('experimental', [False, True])
def test_preview_disclosure_is_preserved_without_changing_candidates(tmp_path, experimental):
    engine = object.__new__(FontPipeline)
    engine.version = 'android-preview-fixture'
    engine.font_mode = 'android'
    engine.max_regions = 10
    engine.detector = SimpleNamespace(detect=lambda image: [
        dict(source_bbox=[10, 10, 90, 40], quad=[[10, 10], [90, 10], [90, 40], [10, 40]], score=.99)])
    prediction = dict(status='candidate', family='Roboto', candidates=[dict(family='Roboto', score=.95)],
                      score=.95, margin=.9, patch_agreement=1., reason_code='region_neural_family_candidate',
                      font_size_px_estimate=24., size_relative_spread=.01, ocr_performed=False)
    engine.region_neural = SimpleNamespace(
        meta={'release_tier': 'experimental'} if experimental else {}, predict=lambda crop: prediction)
    image = Image.new('RGB', (100, 50), 'white')
    ImageDraw.Draw(image).rectangle((15, 15, 80, 35), fill='#333333')
    source = tmp_path/'source.png'
    image.save(source)
    result = engine.run_regions(source, tmp_path/'output', 'fixture', lambda update: None)
    saved = json.loads((tmp_path/'output/result.json').read_text())
    public = public_result('fixture', result)
    for value in (result, saved, public):
        assert value['font_mode'] == 'android'
        assert value['regions'][0]['font']['family'] == 'Roboto'
        assert value['regions'][0]['font']['status'] == 'candidate'
        if experimental:
            assert value['model_release_tier'] == 'experimental'
            assert value['stable_validation_passed'] is False
        else:
            assert 'model_release_tier' not in value
            assert 'stable_validation_passed' not in value
