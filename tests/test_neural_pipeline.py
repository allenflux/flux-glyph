"""Neural pipeline routing uses the classifier and preserves source evidence."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import flux_glyph.pipeline as pipeline_module
from flux_glyph.neural_font import ALGORITHM, SCHEMA, NeuralFontClassifier
from flux_glyph.pipeline import FontPipeline


def network():
    # Exercise the real neural softmax/mask/aggregation/gates while replacing
    # only the ONNX engine. Trained-model accuracy has its own evaluation set.
    classifier = NeuralFontClassifier.__new__(NeuralFontClassifier)
    classifier.families = ["PingFang SC", "MiSans", "SF Pro", "Alipay Number"]
    classifier.meta = {"schema": SCHEMA, "algorithm": ALGORITHM,
                       "families": classifier.families,
                       "scripts": {"han": ["PingFang SC", "MiSans"],
                                   "latin": ["MiSans", "SF Pro", "Alipay Number"]},
                       "gates": {script: {"min_score": .7, "min_margin": .2} for script in ("han", "latin")},
                       "temperature": {"han": 1., "latin": 1.}}
    classifier.indices = {"han": [0, 1], "latin": [1, 2, 3]}
    batches = []

    def run(names, supplied):
        assert names == ["logits"]
        batch = supplied["glyphs"]
        assert batch.dtype == np.float32 and batch.shape[1:] == (1, 64, 64)
        batches.append(batch.copy())
        return [np.repeat(np.array([[6., 0., 9., 0.]], dtype=np.float32), len(batch), axis=0)]

    classifier.session = SimpleNamespace(run=run)
    classifier.batches = batches
    return classifier


def run_region(monkeypatch, tmp_path, text, *, confidence=.99, rotation=0, missing=(), duplicate_index=False):
    # Non-symmetric pixels make incorrect 180-degree source boxes observable.
    yy, xx = np.indices((96, 280))
    pixels = np.stack(((xx + 2 * yy) % 256, (3 * xx + yy) % 256,
                       (xx + 7 * yy) % 256), axis=2).astype(np.uint8)
    source_image = Image.fromarray(pixels)
    source = tmp_path / "source.png"
    source_image.save(source)
    reading = {"text": text, "confidence": confidence, "tokens": [],
               "metadata": {"orientation_degrees": rotation}}
    seen_images = []

    def segmentation(image, actual_text, tokens, **kwargs):
        assert kwargs["segment_latin"] is True
        seen_images.append(image.copy())
        rows = [{"index": i, "character": character, "status": "uncertain" if i in missing else "ok",
                 "reason": "missing_ink" if i in missing else "source_ink",
                 "bbox": None if i in missing else [12 + i * 25, 12, 33 + i * 25, 54]}
                for i, character in enumerate(actual_text)]
        if duplicate_index and len(rows) > 1:
            rows[1] = deepcopy(rows[0])
        return {"characters": rows}

    def reference_forbidden(*args, **kwargs):
        pytest.fail("A neural prediction must not use reference matching")

    monkeypatch.setattr(pipeline_module, "segment_characters", segmentation)
    monkeypatch.setattr(pipeline_module, "score_font", reference_forbidden)
    pipeline = FontPipeline.__new__(FontPipeline)
    pipeline.detector = SimpleNamespace(detect=lambda image: [{
        "source_bbox": [15, 10, 245, 80], "quad": [[15, 10], [245, 10], [245, 80], [15, 80]], "score": .99}])
    pipeline.reader = SimpleNamespace(read=lambda images: [deepcopy(reading) for _ in images])
    # Deliberately empty: the model can handle characters absent from the old
    # archive and must not route through its coverage check or match method.
    pipeline.bank = SimpleNamespace(entries={}, match=reference_forbidden, cache_bytes=0)
    pipeline.latin_bank = SimpleNamespace(score=reference_forbidden, cache_bytes=0)
    pipeline.neural = network()
    pipeline.version = "test-neural-model"
    pipeline.max_regions = 200
    output = tmp_path / "result"
    result = pipeline.run(source, output, "neural-routing")
    return result, output, source_image, pipeline.neural, seen_images


@pytest.mark.parametrize("text", ["青", "青青", "龘"])
def test_single_distinct_and_reference_absent_characters_use_neural_classifier(monkeypatch, tmp_path, text):
    result, _, _, classifier, _ = run_region(monkeypatch, tmp_path, text)
    region = result["regions"][0]
    assert region["font"]["status"] == "candidate"
    assert region["font"]["family"] == "PingFang SC"
    assert region["font"]["method"] == result["font_method"] == "neural_network"
    assert region["font_evidence"]["han"]["evidence"]["distinct_character_count"] == 1
    assert region["font_evidence"]["han"]["evidence"]["sampled_characters"] == len(text)
    assert len(classifier.batches) == 1
    assert region["font"]["font_identity_verified"] is False
    assert result["summary"]["pingfang_supported"] == 0


def test_mixed_scripts_make_independent_neural_predictions(monkeypatch, tmp_path):
    result, _, _, classifier, _ = run_region(monkeypatch, tmp_path, "青12字")
    region = result["regions"][0]
    han, latin = region["font"]["components"]
    assert han["family"] == "PingFang SC" and han["character_indices"] == [0, 3]
    assert latin["family"] == "SF Pro" and latin["character_indices"] == [1, 2]
    assert han["scope"] == "Chinese glyphs only"
    assert latin["scope"] == "Latin letters and digits only"
    assert all(component["status"] == "candidate" for component in (han, latin))
    assert [len(batch) for batch in classifier.batches] == [2, 2]
    assert result["summary"]["latin_candidates"] == 1
    assert [glyph["family_candidate"] for glyph in region["glyphs"]] == ["PingFang SC", "SF Pro", "SF Pro", "PingFang SC"]


@pytest.mark.parametrize("text", ["青青", "22:43", "青12字"])
def test_low_ocr_confidence_blocks_font_and_glyph_candidates(monkeypatch, tmp_path, text):
    result, _, _, _, _ = run_region(monkeypatch, tmp_path, text, confidence=.79)
    region = result["regions"][0]
    assert region["font"]["status"] == "uncertain" and region["font"]["family"] is None
    assert region["font"]["reason_code"] == "text_unreliable"
    for component in region["font"].get("components", []):
        assert component["status"] == "uncertain" and component["family"] is None
    assert all(glyph["family_candidate"] is None for glyph in region["glyphs"])
    assert result["summary"]["latin_candidates"] == 0


def test_incomplete_han_does_not_block_independent_latin_evidence(monkeypatch, tmp_path):
    result, _, _, _, _ = run_region(monkeypatch, tmp_path, "青12字", missing=(3,))
    han, latin = result["regions"][0]["font"]["components"]
    assert han["status"] == "uncertain" and han["family"] is None
    assert han["reason_code"] == "incomplete_segmentation"
    assert latin["status"] == "candidate" and latin["family"] == "SF Pro"


@pytest.mark.parametrize("rotation", [0, 180])
def test_saved_glyphs_and_reported_source_boxes_preserve_pixels(monkeypatch, tmp_path, rotation):
    result, output, source, _, seen = run_region(monkeypatch, tmp_path, "青12字", rotation=rotation)
    region = result["regions"][0]
    expected_roi = source.crop(region["source_bbox"])
    if rotation:
        expected_roi = expected_roi.transpose(Image.Transpose.ROTATE_180)
    np.testing.assert_array_equal(np.asarray(seen[0]), np.asarray(expected_roi))
    for glyph in region["glyphs"]:
        expected = source.crop(glyph["source_bbox"])
        if rotation:
            expected = expected.transpose(Image.Transpose.ROTATE_180)
        with Image.open(output / glyph["crop_file"]) as actual:
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        assert glyph["source_rotation_degrees"] == rotation


def test_duplicate_segment_index_cannot_claim_complete_evidence(monkeypatch, tmp_path):
    result, _, _, _, _ = run_region(monkeypatch, tmp_path, "青字", duplicate_index=True)
    region = result["regions"][0]
    assert region["font"]["status"] == "uncertain" and region["font"]["family"] is None
    assert region["font"]["reason_code"] == "incomplete_segmentation"


@pytest.mark.parametrize("declared", [True, False])
def test_only_manifest_declared_neural_model_is_loaded(monkeypatch, tmp_path, declared):
    (tmp_path / "neural").mkdir()
    files = [{"path": "font/metadata.json"}]
    if declared:
        files.append({"path": "neural/metadata.json"})
    monkeypatch.setattr(pipeline_module, "load_active", lambda root: (tmp_path, "test", {"files": files}))
    monkeypatch.setattr(pipeline_module, "PPRegionDetector", lambda path: object())
    monkeypatch.setattr(pipeline_module, "PPReader", lambda path: object())
    monkeypatch.setattr(pipeline_module, "CompactFontBank", lambda path, size: object())
    calls = []
    expected = network()

    def create(path):
        calls.append(path)
        return expected

    monkeypatch.setattr(pipeline_module, "NeuralFontClassifier", create)
    pipeline = FontPipeline(tmp_path)
    assert pipeline.neural is (expected if declared else None)
    assert calls == ([tmp_path / "neural"] if declared else [])
