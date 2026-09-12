"""Evaluation scoring cannot hide wrong fonts, missed rows or duplicate boxes."""
from copy import deepcopy
import importlib.util
from itertools import permutations
import json
from pathlib import Path
import random

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_ios_screenshots", ROOT / "scripts/evaluate_ios_screenshots.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)
FIXTURES = importlib.util.spec_from_file_location("captured_contract_fixtures", ROOT / "tests/test_prepare_captured.py")
fixtures = importlib.util.module_from_spec(FIXTURES)
FIXTURES.loader.exec_module(fixtures)


def truth(identifier="r1", box=(10, 10, 110, 30), text="青甲", family="PingFang SC"):
    return {"id": identifier, "ink_bbox_pixels": list(box), "text": text, "font_family": family,
            "script": "han", "font_truth_verified": True, "font_size_screen_px": 30,
            "actual_text_color_hex": "#112233FF", "background_hex": "#FFFFFFFF"}


def prediction(box=(10, 10, 110, 30), text="青甲", family="PingFang SC", status="candidate"):
    return {"id": "R001", "detector_bbox": list(box), "text": text,
            "font": {"status": status, "family": family}}


def page(truths=None):
    return {"source_id": "ios:page1", "page_id": "page1", "regions": [truth()] if truths is None else truths}


def test_pairing_maximizes_count_before_iou_and_beats_greedy():
    truths = [truth(box=(10, 0, 20, 10)), truth("r2", box=(15, 0, 25, 10))]
    predictions = [prediction(box=(12, 0, 22, 10)), prediction(box=(7, 0, 17, 10))]
    paired = evaluation.pair_regions(truths, predictions)
    assert [(left, right) for left, right, _ in paired] == [(0, 1), (1, 0)]
    assert evaluation.iou(truths[0]["ink_bbox_pixels"], predictions[0]["detector_bbox"]) > paired[0][2]


def test_pairing_never_uses_ocr_text_or_font_name():
    truths = [truth(box=(0, 0, 100, 20)), truth("r2", box=(0, 40, 100, 60), text="青乙", family="MiSans")]
    predictions = [prediction(box=(0, 0, 100, 20), text="青乙", family="MiSans"),
                   prediction(box=(0, 40, 100, 60), text="青甲", family="PingFang SC")]
    first = evaluation.pair_regions(truths, predictions)
    for pred in predictions:
        pred.update(text="completely different", detector_score=.00001)
        pred["font"].update(family="unknown", status="uncertain")
    assert first == evaluation.pair_regions(truths, predictions)
    assert [(left, right) for left, right, _ in first] == [(0, 0), (1, 1)]


def test_hungarian_matches_brute_force_global_optimum():
    randomizer = random.Random(20260912)
    for n in range(1, 5):
        for _ in range(10):
            weights = [[randomizer.randrange(9) for _ in range(n + 2)] for _ in range(n)]
            assignment = evaluation.maximum_assignment(weights)
            actual = sum(weights[i][column] for i, column in enumerate(assignment))
            expected = max(sum(weights[i][column] for i, column in enumerate(columns))
                           for columns in permutations(range(n + 2), n))
            assert actual == expected
            assert len(set(assignment)) == n


@pytest.mark.parametrize("truths,predictions,expected", [([], [], []), ([], [prediction()], []), ([truth()], [], [])])
def test_empty_matching(truths, predictions, expected):
    assert evaluation.pair_regions(truths, predictions) == expected


def test_iou_half_is_included_and_expanded_crop_is_not_used():
    source = [truth(box=(0, 0, 20, 10))]
    pred = prediction(box=(0, 0, 10, 10))
    pred["source_bbox"] = [0, 0, 20, 10]
    assert evaluation.pair_regions(source, [pred]) == [(0, 0, .5)]
    pred["detector_bbox"] = [0, 0, 9, 10]
    assert evaluation.pair_regions(source, [pred]) == []


@pytest.mark.parametrize("box", [None, [0, 0, 0, 5], [0, 0, float("nan"), 5], [0, 0, float("inf"), 5]])
def test_invalid_detector_geometry_remains_false_positive(box):
    pred = prediction()
    pred["detector_bbox"] = box
    result = evaluation.score_page({"regions": [pred]}, page())
    assert result["rows"][0]["detected"] is False
    assert len(result["false_positives"]) == 1


def test_duplicate_predictions_count_once_plus_false_positive():
    score = evaluation.score_page({"regions": [prediction(), prediction(family="MiSans")]}, page())
    metrics = evaluation.aggregate([score])["overall"]
    assert metrics["detected"] == metrics["font_name_correct"] == 1
    assert metrics["false_positives"] == metrics["false_positive_named_fonts"] == 1
    assert metrics["detection_precision"] == .5 and metrics["detection_recall"] == 1
    assert metrics["wrong_font_names"] == 0  # Wrong extra font is explicitly an FP claim.


def test_one_prediction_cannot_score_two_truths():
    score = evaluation.score_page({"regions": [prediction()]}, page([truth(), truth("r2")]))
    assert sum(row["detected"] for row in score["rows"]) == 1
    assert evaluation.aggregate([score])["overall"]["missed"] == 1


def test_ocr_error_does_not_turn_correct_font_name_into_wrong_font():
    score = evaluation.score_page({"regions": [prediction(text="育甲")]}, page())
    row = score["rows"][0]
    assert row["font_name_correct"] and not row["wrong_font_name"]
    assert not row["ocr_normalized_exact"] and not row["strict_e2e_correct"]
    assert row["ocr_edit_distance"] == 1
    assert evaluation.aggregate([score])["overall"]["ocr_cer_detected"] == .5


def test_miss_font_abstention_wrong_name_and_ocr_error_have_distinct_denominators():
    truths = [truth(f"r{i}", box=(10, i * 40 + 10, 110, i * 40 + 30)) for i in range(4)]
    preds = [prediction(box=truths[0]["ink_bbox_pixels"]),
             prediction(box=truths[1]["ink_bbox_pixels"], family="MiSans"),
             prediction(box=truths[2]["ink_bbox_pixels"], family=None, status="uncertain")]
    metrics = evaluation.aggregate([evaluation.score_page({"regions": preds}, page(truths))])["overall"]
    assert metrics["truth_regions"] == 4 and metrics["detected"] == 3 and metrics["missed"] == 1
    assert metrics["font_accepted"] == 2 and metrics["font_name_correct"] == metrics["wrong_font_names"] == 1
    assert metrics["font_abstained_after_detection"] == 1
    assert metrics["font_name_precision_among_accepted"] == .5 and metrics["font_name_accuracy_all_truth"] == .25
    assert metrics["ocr_cer_detected"] == 0 and metrics["ocr_cer_all_truth_including_misses"] == .25
    assert metrics["ocr_normalized_accuracy_detected"] == 1
    assert metrics["ocr_normalized_accuracy_all_truth"] == .75
    assert metrics["style"]["size_unavailable"] == 4 and metrics["style"]["size_mae_px"] is None


def test_ocr_normalization_removes_whitespace_only():
    item = page([truth(text="Ab 12.")])
    good = evaluation.score_page({"regions": [prediction(text="A b12.")]}, item)["rows"][0]
    assert good["ocr_normalized_exact"] and not good["ocr_exact"]
    bad = evaluation.score_page({"regions": [prediction(text="ab12")]}, item)["rows"][0]
    assert not bad["ocr_normalized_exact"] and bad["ocr_edit_distance"] == 2


def test_mixed_components_cannot_promote_one_family_using_native_truth():
    pred = prediction()
    pred["font"]["components"] = [{"family": "PingFang SC", "status": "candidate"}, {"family": "MiSans", "status": "candidate"}]
    score = evaluation.score_page({"regions": [pred]}, page())["rows"][0]
    assert score["font_abstained"] and not score["accepted"]


def test_style_compares_visible_rgba_and_source_pixel_size():
    native = truth()
    native.update(actual_text_color_hex="#00000080", font_size_screen_px=60)
    style = {"text_color_hex": "#7F7F7F", "font_size_px_estimate": 63., "font_size_px_interval": [59., 66.],
             "color": {"status": "estimated"}, "size": {"status": "estimated"}}
    scored = evaluation.score_style(style, native)
    assert scored["color_available"] and scored["color_max_channel_error"] == pytest.approx(0)
    assert scored["size_absolute_error_px"] == 3 and scored["size_relative_error"] == .05
    assert scored["size_interval_covers_native"] and scored["size_within_tolerance"]
    style.update(font_size_px_estimate=67, text_color_hex="#999999")
    scored = evaluation.score_style(style, native)
    assert not scored["color_within_tolerance"] and not scored["size_within_tolerance"]


@pytest.mark.parametrize("size", [float("nan"), float("inf"), 0, -1, True, "30"])
def test_invalid_size_predictions_are_unavailable(size):
    score = evaluation.score_style({"font_size_px_estimate": size, "size": {"status": "estimated"}}, truth())
    assert not score["size_available"] and score["size_predicted_px"] is None


def test_paired_comparison_uses_identical_region_keys_and_reports_new_errors():
    baseline = evaluation.score_page({"regions": [prediction()]}, page())
    neural = evaluation.score_page({"regions": [prediction(family="MiSans")]}, page())
    comparison = evaluation.paired_comparison([baseline], [neural])
    assert comparison["font_name_correct"] == {"baseline_only_correct": 1}
    assert comparison["new_wrong_font_names"] == 1 and comparison["resolved_wrong_font_names"] == 0
    neural["rows"][0]["region_id"] = "different"
    with pytest.raises(ValueError, match="identical held-out truths"):
        evaluation.paired_comparison([baseline], [neural])


def native_test_dataset(tmp_path):
    labels, rows, scenes = fixtures.dataset(tmp_path, texts=("青甲", "青乙", "青丙"), splits=("train", "calibration", "test"))
    for row in rows:
        row["regions"][0]["ink_bbox_pixels"] = [15, 20, 68, 50]
        row["regions"][0]["background_hex"] = "#FFFFFFFF"
    fixtures.write_labels(labels, rows)
    return labels, rows, scenes


def test_capture_audit_returns_only_test_and_fixed_source_counts(tmp_path):
    labels, rows, _ = native_test_dataset(tmp_path)
    sources, protocol = evaluation.checked_captures(labels)
    assert len(sources) == 1 and sources[0]["source_id"] == rows[2]["source_id"]
    assert protocol["source_split_counts"] == {"train": 1, "calibration": 1, "test": 1}
    assert protocol["ui_content_is_generated"] and not protocol["independent_third_party_app_accuracy"]
    assert all(value == 0 for value in protocol["split_isolation"].values())
    assert "glyphs" not in sources[0]["regions"][0]


def test_capture_audit_rejects_missing_test_pages_and_font_fallback(tmp_path):
    labels, rows, _ = native_test_dataset(tmp_path)
    fixtures.write_labels(labels, rows[:2])
    with pytest.raises(ValueError, match="all predetermined test captures"):
        evaluation.checked_captures(labels)
    rows[2]["regions"][0]["fallback_detected"] = True
    fixtures.write_labels(labels, rows)
    with pytest.raises(ValueError, match="native truth not verified"):
        evaluation.checked_captures(labels)


def test_capture_audit_rejects_cross_split_source_id(tmp_path):
    labels, rows, _ = native_test_dataset(tmp_path)
    rows[2]["source_id"] = rows[0]["source_id"]
    fixtures.write_labels(labels, rows)
    with pytest.raises(ValueError, match="cross-split leakage: source_id"):
        evaluation.checked_captures(labels)


def test_full_pipeline_receives_png_only_and_resume_does_not_rerun(tmp_path, monkeypatch):
    calls = []
    models = tmp_path / "models"
    models.mkdir()
    evaluation.write_json(models / "MANIFEST.json", {"version": "test"})
    image = tmp_path / "original.png"
    image.write_bytes(b"mocked pipeline input; not a real capture dataset")
    source = page()
    source.update(image=str(image), source_sha256=evaluation.sha(image))

    class Pipeline:
        def __init__(self, directory):
            assert directory == models
            self.directory, self.version, self.neural = models, "test", None

        def run(self, source_path, output, identifier):
            assert isinstance(source_path, Path) and source_path == image
            assert isinstance(identifier, str) and identifier != source["source_id"]
            calls.append(source_path)
            result = {"source_sha256": evaluation.sha(source_path), "regions": [prediction()]}
            evaluation.write_json(output / "result.json", result)
            return result

    monkeypatch.setattr(evaluation, "FontPipeline", Pipeline)
    monkeypatch.setattr(evaluation, "verify_bundle", lambda directory: None)
    output = tmp_path / "evaluation"
    first = evaluation.evaluate_method("baseline", models, [source], {"rules": evaluation.RULES}, output)
    second = evaluation.evaluate_method("baseline", models, [source], {"rules": evaluation.RULES}, output)
    assert len(calls) == 1 and first["metrics"] == second["metrics"]
    assert evaluation.checked_method_report("baseline", [source], {"rules": evaluation.RULES}, output) == second
    report_path = output / "baseline/report.json"
    tampered_report = deepcopy(second)
    tampered_report["metrics"]["overall"]["wrong_font_names"] = 999
    evaluation.write_json(report_path, tampered_report)
    with pytest.raises(ValueError, match="report differs"):
        evaluation.checked_method_report("baseline", [source], {"rules": evaluation.RULES}, output)
    evaluation.write_json(report_path, second)
    cached_result = next((output / "baseline/runs").glob("*/result.json"))
    value = evaluation.read_json(cached_result)
    value["regions"][0]["font"]["family"] = "tampered"
    evaluation.write_json(cached_result, value)
    with pytest.raises(ValueError, match="pinned file changed"):
        evaluation.evaluate_method("baseline", models, [source], {"rules": evaluation.RULES}, output)
    assert len(calls) == 1
