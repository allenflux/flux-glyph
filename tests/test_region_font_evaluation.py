import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_region_fonts as evaluation


def truth(identifier="t1", box=(0, 0, 100, 20), family="PingFang SC"):
    return {"id": identifier, "text": "文字仅作真值审计", "script": "han", "font_family": family,
            "font_truth_verified": True, "ink_bbox_pixels": list(box), "font_size_screen_px": 40,
            "actual_text_color_hex": "#173967FF", "background_hex": "#FFFFFFFF"}


def prediction(identifier="R1", box=(0, 0, 100, 20), family="PingFang SC", status="candidate"):
    return {"id": identifier, "detector_bbox": list(box), "text": None, "glyphs": [], "ocr_performed": False,
            "font": {"method": evaluation.METHOD, "status": status, "family": family}}


def result(regions):
    return {"font_method": evaluation.METHOD, "ocr_performed": False, "regions": regions}


def source(regions):
    return {"source_id": "native:test:1", "page_id": "test-1", "regions": regions}


def test_font_scoring_has_no_ocr_metrics_and_geometry_ignores_truth_text():
    first = evaluation.score_page(result([prediction()]), source([truth()]))
    changed = truth(); changed["text"] = "abcdef123"; changed["script"] = "latin"
    second = evaluation.score_page(result([prediction()]), source([changed]))
    assert first["rows"][0]["font_name_correct"] == second["rows"][0]["font_name_correct"] is True
    metrics = evaluation.aggregate([first])["overall"]
    assert metrics["font_name_correct"] == 1
    assert not any("ocr" in key for key in metrics)
    assert not any("text" in key for key in first["rows"][0])


def test_misses_abstentions_wrong_names_and_duplicate_boxes_remain_separate():
    truths = [truth(), truth("t2", (0, 40, 100, 60)), truth("t3", (0, 80, 100, 100))]
    predictions = [prediction(family="MiSans"), prediction("duplicate", family="MiSans"),
                   prediction("R2", (0, 40, 100, 60), family=None, status="uncertain")]
    page = evaluation.score_page(result(predictions), source(truths))
    metrics = evaluation.aggregate([page])["overall"]
    assert metrics["detected"] == 2
    assert metrics["missed"] == metrics["wrong_font_names"] == metrics["font_abstained_after_detection"] == 1
    assert metrics["false_positives"] == metrics["false_positive_named_fonts"] == 1
    assert metrics["font_precision_among_accepted"] == 0


def test_one_prediction_cannot_claim_two_native_truths():
    page = evaluation.score_page(result([prediction()]), source([truth(), truth("t2")]))
    assert evaluation.aggregate([page])["overall"]["missed"] == 1


@pytest.mark.parametrize("alter", [
    lambda value: value.update(ocr_performed=True),
    lambda value: value.update(font_method="neural_network"),
    lambda value: value["regions"][0].update(text="read by OCR"),
    lambda value: value["regions"][0].update(glyphs=[{"character": "a"}]),
    lambda value: value["regions"][0].update(script="han"),
    lambda value: value["regions"][0]["font"].update(method="template"),
])
def test_result_must_prove_no_text_or_character_path(alter):
    value = result([prediction()])
    alter(value)
    with pytest.raises(ValueError):
        evaluation.validate_prediction(value)


@pytest.mark.parametrize("kind", ["reader", "ctc", "segmentation", "old_classifier", "recognizer_session"])
def test_guard_prevents_real_ocr_or_glyph_calls_even_if_exception_is_caught(kind):
    with pytest.raises(ValueError, match="forbidden inference"):
        with evaluation.prohibit_ocr() as state:
            with pytest.raises(RuntimeError, match="forbids"):
                if kind == "reader":
                    evaluation.ppocr.PPReader(ROOT / "models/pp")
                elif kind == "ctc":
                    evaluation.ppocr.decode_ctc(None, None)
                elif kind == "segmentation":
                    evaluation.segmentation.segment_characters(None, None, None)
                elif kind == "old_classifier":
                    evaluation.pipeline_module.CompactFontBank(ROOT / "models/font")
                else:
                    evaluation.ppocr.session(None, "rec", None)
            assert any(state[key] > 0 for key in ("ocr_calls", "segmentation_calls", "glyph_classifier_calls"))


def test_guard_allows_pp_detection_session_without_recognition(monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setattr(evaluation.ppocr, "session", lambda directory, kind, contract: calls.append(kind) or sentinel)
    with evaluation.prohibit_ocr() as state:
        assert evaluation.ppocr.session(None, "det", None) is sentinel
    assert calls == ["det"] and state["pp_detector_sessions"] == 1
    assert state["ocr_calls"] == state["segmentation_calls"] == state["glyph_classifier_calls"] == 0


def test_style_scores_independent_of_font_and_uses_source_pixels():
    pred = prediction(family="wrong")
    pred["text_style"] = {"text_color_hex": "#173967", "color": {"status": "estimated"},
                          "font_size_px_estimate": 42, "font_size_px_interval": [39, 44],
                          "size": {"status": "estimated"}}
    page = evaluation.score_page(result([pred]), source([truth()]))
    metrics = evaluation.aggregate([page])["overall"]
    assert metrics["wrong_font_names"] == 1
    assert metrics["style"]["size_mae_px"] == 2
    assert metrics["style"]["size_within_tolerance"] == metrics["style"]["color_within_tolerance"] == 1
    json.dumps(page, allow_nan=False)


def test_mixed_font_component_is_not_promoted_to_one_family():
    pred = prediction()
    pred["font"]["components"] = [{"family": "PingFang SC"}, {"family": "SF Pro"}]
    page = evaluation.score_page(result([pred]), source([truth()]))
    assert page["rows"][0]["font_abstained"]
    assert evaluation.RULES["mixed_font_region_handling_validated"] is False
    assert evaluation.RULES["test_partition_reused_from_previous_pipeline_evaluations"] is True


def test_complete_fixed_test_partition_is_required(monkeypatch):
    monkeypatch.setattr(evaluation.shared, "checked_captures", lambda _: ([], {"test_pages": 99, "source_split_counts": {"test": 99, "train": 800, "calibration": 100}}))
    with pytest.raises(ValueError, match="preassigned"):
        evaluation.checked_sources("unused")


def test_orchestrator_only_passes_png_to_inference_and_checks_cached_hashes(tmp_path, monkeypatch):
    from PIL import Image
    image_path = tmp_path / "input.png"
    Image.new("RGB", (120, 40), "white").save(image_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "MANIFEST.json").write_text("{}")
    records = [{**source([truth()]), "image": str(image_path), "source_sha256": evaluation.sha(image_path)}]
    pinned = tmp_path / "provenance.json"
    pinned.write_text("{}")
    protocol = {key: {"path": str(pinned), "sha256": evaluation.sha(pinned)} for key in ("captures", "scenes", "capture_protocol")}
    calls = []

    class PixelOnlyPipeline:
        def __init__(self, directory):
            assert directory == models
            self.directory, self.version, self.region_neural = models, "test", object()
            self.reader = self.neural = self.bank = self.latin_bank = None

        def run(self, supplied_image, output, identifier):
            assert isinstance(supplied_image, Path) and supplied_image == image_path
            assert isinstance(output, Path) and isinstance(identifier, str)
            calls.append(supplied_image)
            value = {**result([prediction()]), "source_sha256": evaluation.sha(image_path)}
            evaluation.write_json(output / "result.json", value)
            return value

    monkeypatch.setattr(evaluation, "checked_sources", lambda _: (records, protocol))
    monkeypatch.setattr(evaluation, "source_hashes", lambda: {"fixed-test-runtime": "0" * 64})
    monkeypatch.setattr(evaluation, "verify_bundle", lambda _: None)
    monkeypatch.setattr(evaluation.pipeline_module, "FontPipeline", PixelOnlyPipeline)
    output = tmp_path / "evaluation"
    first = evaluation.evaluate(models, "not_passed_to_pipeline", output)
    second = evaluation.evaluate(models, "not_passed_to_pipeline", output)
    assert first["metrics"] == second["metrics"]
    assert len(calls) == 1  # second pass verifies and reuses cached original output
    assert first["guard"]["ocr_calls"] == first["guard"]["segmentation_calls"] == 0
    cache = json.loads(next((output / "cache").glob("*.json")).read_text())
    result_path = output / cache["result_path"]
    result_path.write_text(result_path.read_text() + " ")
    with pytest.raises(ValueError, match="pinned file changed"):
        evaluation.evaluate(models, "unused", output)
