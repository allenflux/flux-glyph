"""The fixed evaluation must distinguish abstention from successful rejection."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/evaluate_neural.py"
SPEC = importlib.util.spec_from_file_location("evaluate_neural", PATH)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def result(text, family, status="candidate"):
    return {"regions": [{"id": "R001", "text": text, "source_bbox": [0, 0, 80, 30],
                         "detector_score": .99, "glyphs": [],
                         "font": {"status": status, "family": family, "candidates": [], "reason": "test"}}],
            "timing_seconds": {"total": .01}}


@pytest.mark.parametrize("expected,status,family,outcome", [
    ("SF Pro", "candidate", "SF Pro", "correct_candidate"),
    ("SF Pro", "uncertain", None, "unconfirmed"),
    ("SF Pro", "candidate", "MiSans", "wrong_candidate"),
    ("SF Pro", "supported", "SF Pro", "wrong_candidate"),
    (None, "uncertain", None, "correct_rejection"),
    (None, "candidate", "SF Pro", "wrong_candidate"),
])
def test_latin_preserves_positive_and_unknown_negative_rules(expected, status, family, outcome):
    case = {"id": "test", "text": "22:43", "expected_family": expected,
            "image_sha256": "fixture", "source_sha256": "font"}
    row = evaluation.score_latin(result(case["text"], family, status), case)
    assert row["outcome"] == outcome
    assert row["correct"] is (outcome in {"correct_candidate", "correct_rejection"})
    assert row["wrong_font_name"] is (outcome == "wrong_candidate")


def test_latin_wrong_partial_region_cannot_be_hidden_by_a_correct_region():
    case = {"id": "test", "text": "22:43", "expected_family": "SF Pro",
            "image_sha256": "fixture", "source_sha256": "font"}
    prediction = result("22", "SF Pro")
    second = result(":43", "MiSans")["regions"][0]
    second["source_bbox"] = [90, 0, 150, 30]
    prediction["regions"].append(second)
    assert evaluation.score_latin(prediction, case)["outcome"] == "wrong_candidate"


def han_case():
    return {"case_id": "songti-negative", "text": "青", "expected_family": "Songti SC",
            "negative_family": True, "font_id": "songti", "input_sha256": "fixture", "source_font_sha256": "font",
            "expected_glyph_bboxes": [[0, 0, 30, 30]], "expected_ink_bboxes": [[4, 4, 26, 26]]}


def test_han_known_negative_songti_still_requires_correct_family():
    case = han_case()
    prediction = result(case["text"], "Songti SC")
    prediction["regions"][0]["glyphs"] = [{"character": "青", "status": "ok", "source_bbox": [3, 3, 27, 27]}]
    row = evaluation.score_han(prediction, case)
    assert row["correct"] and row["outcome"] == "correct_candidate"
    assert row["positive_negative"] == "negative"
    abstained = deepcopy(prediction)
    abstained["regions"][0]["font"].update(status="uncertain", family=None)
    assert evaluation.score_han(abstained, case)["outcome"] == "unconfirmed"


def test_han_correct_font_name_with_bad_geometry_is_not_strict_correct():
    case = han_case()
    prediction = result(case["text"], "Songti SC")
    prediction["regions"][0]["glyphs"] = [{"character": "青", "status": "ok", "source_bbox": [15, 4, 26, 26]}]
    row = evaluation.score_han(prediction, case)
    assert row["font_name_correct"] and not row["correct"]
    assert row["failure_stage"] == "segmentation" and row["outcome"] == "wrong_candidate"
    assert not row["wrong_font_name"]
    counts = evaluation.counts([row])
    assert counts["font_name_correct"] == counts["accepted"] == 1
    assert counts["wrong_font_names"] == counts["correct"] == 0
