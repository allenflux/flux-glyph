#!/usr/bin/env python3
"""Run the frozen Latin fixtures through detection, OCR, segmentation and matching.

Each existing PNG is pasted without resizing onto a 750 x 300 white canvas,
horizontally centered with its top edge at y=125. There are no oracle boxes or
OCR tokens supplied to FontPipeline. This is a controlled rendered-image test,
not evidence of real iOS/Android device accuracy.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flux_glyph.pipeline import FontPipeline

FIXTURES = ROOT / "tests/fixtures/latin_accuracy"
OUTPUT = ROOT / "docs/mobile-latin-e2e-validation.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return "".join(text.split())


def evaluate():
    cases_path = FIXTURES / "cases.json"
    fixtures = json.loads(cases_path.read_text())["cases"]
    if len(fixtures) != 31:
        raise ValueError("Expected the frozen 31-case Latin fixture set")
    artifacts = ROOT / "artifacts/mobile-coverage/end-to-end"
    artifacts.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="latin-", dir=artifacts))
    pipeline = FontPipeline(ROOT / "models")
    if pipeline.latin_bank is None:
        raise ValueError("Active model has no Latin reference bank")
    rows = []
    try:
        for fixture in fixtures:
            source = FIXTURES / fixture["image"]
            if sha(source) != fixture["image_sha256"]:
                raise ValueError("Frozen fixture image checksum mismatch: " + source.name)
            with Image.open(source) as raw:
                image = raw.convert("RGB")
            x, y = (750 - image.width) // 2, 125
            if x < 0 or y + image.height > 300:
                raise ValueError("Native fixture does not fit the declared canvas")
            canvas = Image.new("RGB", (750, 300), "white")
            canvas.paste(image, (x, y))
            pasted = run_dir / (fixture["id"] + ".png")
            canvas.save(pasted)
            result_dir = run_dir / fixture["id"]
            result = pipeline.run(pasted, result_dir, fixture["id"])
            regions = sorted(result["regions"], key=lambda region: (region["source_bbox"][1], region["source_bbox"][0]))
            ocr_text = " ".join(region["text"] for region in regions)
            ocr_exact = normalized(ocr_text) == normalized(fixture["text"])
            expected = fixture["expected_family"]
            named = [region for region in regions if region["font"]["status"] in {"candidate", "supported"}]
            wrong = [region for region in named if not expected or region["font"]["family"] != expected or region["font"]["status"] == "supported"]
            complete = bool(regions) and len(named) == len(regions) and ocr_exact
            if wrong:
                outcome = "wrong_candidate"
            elif expected and complete:
                outcome = "correct_candidate"
            elif not expected and not named:
                outcome = "correct_rejection"
            else:
                outcome = "unconfirmed"
            row = {"id": fixture["id"], "expected_text": fixture["text"], "expected_family": expected,
                   "fixture_sha256": fixture["image_sha256"], "source_font_sha256": fixture["source_sha256"],
                   "canvas_sha256": sha(pasted), "placement": [x, y, image.width, image.height],
                   "ocr_text": ocr_text, "ocr_exact_ignoring_whitespace": ocr_exact,
                   "detected_regions": len(regions), "outcome": outcome,
                   "candidate_families": [region["font"]["family"] for region in named],
                   "regions": [{"text": region["text"], "ocr_confidence": region.get("ocr_confidence"),
                                "status": region["font"]["status"], "family": region["font"]["family"],
                                "reason": region["font"]["reason"], "candidates": region["font"]["candidates"],
                                "source_bbox": region["source_bbox"],
                                "glyph_statuses": Counter(glyph["status"] for glyph in region["glyphs"])} for region in regions],
                   "result_file": (result_dir / "result.json").relative_to(ROOT).as_posix(),
                   "elapsed_seconds": result["timing_seconds"]["total"]}
            rows.append(row)
            print(f"{fixture['id']}: {outcome}; OCR={ocr_text!r}; families={row['candidate_families']}", flush=True)
    finally:
        pipeline.bank.archive.close()
        if pipeline.latin_bank:
            pipeline.latin_bank.archive.close()
    positive = [row for row in rows if row["expected_family"]]
    negative = [row for row in rows if not row["expected_family"]]
    by_family = defaultdict(Counter)
    for row in rows:
        by_family[row["expected_family"] or "unknown_negative"][row["outcome"]] += 1
    report = {"schema": "flux-glyph-mobile-latin-end-to-end-v1",
              "scope": "Frozen Pillow-rendered native-pixel fixtures on a white canvas; full detector/OCR/segmentation/matcher; not real-device accuracy.",
              "model_version": pipeline.version, "manifest_sha256": sha(pipeline.directory / "MANIFEST.json"),
              "latin_archive_sha256": pipeline.latin_bank.meta["archive_sha256"],
              "fixture_manifest_sha256": sha(cases_path), "canvas_size": [750, 300], "paste_top": 125,
              "fixture_resizing_used": False, "oracle_boxes_or_tokens_used": False, "thresholds_modified": False,
              "decision": "Correct positive requires complete OCR ignoring whitespace, at least one detection, and all detected regions returning the expected candidate family. Any wrong named family or supported claim is wrong; remaining positives are unconfirmed. Negative cases are correct rejections only when no candidate is named.",
              "positive_cases": len(positive), "positive_correct_candidates": sum(row["outcome"] == "correct_candidate" for row in positive),
              "positive_wrong_candidates": sum(row["outcome"] == "wrong_candidate" for row in positive),
              "positive_unconfirmed": sum(row["outcome"] == "unconfirmed" for row in positive),
              "negative_cases": len(negative), "negative_false_candidates": sum(row["outcome"] == "wrong_candidate" for row in negative),
              "negative_correct_rejections": sum(row["outcome"] == "correct_rejection" for row in negative),
              "exact_ocr_cases_ignoring_whitespace": sum(row["ocr_exact_ignoring_whitespace"] for row in rows),
              "by_family": dict(by_family), "cases": rows}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"cases", "by_family"}}, indent=2))
    return report


if __name__ == "__main__":
    evaluate()
