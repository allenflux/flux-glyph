#!/usr/bin/env python3
"""Evaluate the trained CNN on the existing fixed 36 Han + 31 Latin fixtures.

This is full detector/OCR/segmentation/classifier inference, not inference from
oracle boxes. Inputs, labels and thresholds are never changed by this script.
The two suites retain their original scoring definitions. These rendered
fixtures do not establish accuracy on real, independently labelled screenshots.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flux_glyph.pipeline import FontPipeline  # noqa: E402

HAN = ROOT / "tests/fixtures/font_accuracy"
LATIN = ROOT / "tests/fixtures/latin_accuracy"
DEFAULT_OUTPUT = ROOT / "artifacts/neural-font-v1/evaluation"
IOS_FAMILIES = {"PingFang SC", "SF Pro", "Helvetica"}
ANDROID_FAMILIES = {"Noto Sans CJK SC", "HarmonyOS Sans SC", "MiSans", "OPPO Sans", "Roboto"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def checked_fixtures():
    protocol = read_json(HAN / "cases.json")
    generated = read_json(HAN / "generated_manifest.json")
    han_cases, latin_cases = generated["cases"], read_json(LATIN / "cases.json")["cases"]
    if (not protocol.get("protocol_frozen_before_first_run") or
            generated["protocol_sha256"] != sha(HAN / "cases.json") or
            generated["case_count"] != len(han_cases) or len(han_cases) != 36 or
            len(han_cases) != len(protocol["fonts"]) * len(protocol["render_profiles"]) or len(latin_cases) != 31):
        raise ValueError("Fixed evaluation fixture protocol/count mismatch")
    for root, cases, file_key, hash_key in ((HAN, han_cases, "input_file", "input_sha256"),
                                           (LATIN, latin_cases, "image", "image_sha256")):
        for case in cases:
            path = (root / case[file_key]).resolve()
            if not path.is_relative_to(root.resolve()) or sha(path) != case[hash_key]:
                raise ValueError("Fixed evaluation image checksum mismatch: " + case[file_key])
    return han_cases, latin_cases


def cohort(family):
    if family in IOS_FAMILIES:
        return "ios_font_references"
    if family in ANDROID_FAMILIES:
        return "android_font_references"
    if family == "Alipay Number":
        return "application_font_reference"
    return "other_or_negative_font_references"


def selected_han_region(result, text):
    # Exact selection rule from tests/accuracy_eval.py; it is deliberately not
    # selected based on the font label or neural scores.
    return max(result.get("regions", []), key=lambda row: (
        row.get("text") == text, SequenceMatcher(a=text, b=row.get("text", "")).ratio(),
        row.get("detector_score", 0.)), default=None)


def geometry_rows(glyphs, case):
    # Same >=90% ink coverage and +/-2px horizontal cell tolerance as the
    # original Han benchmark. Truth boxes are used only after inference.
    rows = []
    for index, character in enumerate(case["text"]):
        glyph = glyphs[index] if index < len(glyphs) and glyphs[index].get("character") == character else {}
        predicted = glyph.get("source_bbox")
        cell, ink = case["expected_glyph_bboxes"][index], case["expected_ink_bboxes"][index]
        area = max(0, ink[2] - ink[0]) * max(0, ink[3] - ink[1])
        intersection = (max(0, min(predicted[2], ink[2]) - max(predicted[0], ink[0])) *
                        max(0, min(predicted[3], ink[3]) - max(predicted[1], ink[1]))) if predicted else 0
        coverage = intersection / area if area else 0.
        within_cell = bool(predicted and predicted[0] >= cell[0] - 2 and predicted[2] <= cell[2] + 2)
        rows.append({"character": character, "predicted_bbox": predicted, "truth_cell_bbox": cell,
                     "truth_ink_bbox": ink, "truth_ink_bbox_coverage": round(coverage, 6),
                     "within_character_cell_tolerance": within_cell, "geometry_ok": coverage >= .9 and within_cell})
    return rows


def score_han(result, case):
    region = selected_han_region(result, case["text"])
    text = region.get("text", "") if region else ""
    glyphs = region.get("glyphs", []) if region else []
    font = region.get("font", {}) if region else {}
    status, family = font.get("status"), font.get("family")
    accepted = status in {"supported", "candidate"} and family is not None
    ocr_exact = text == case["text"]
    status_ok = bool(ocr_exact and [g.get("character") for g in glyphs] == list(case["text"]) and
                     all(g.get("status") == "ok" for g in glyphs))
    geometry = geometry_rows(glyphs, case)
    complete = status_ok and all(row["geometry_ok"] for row in geometry)
    if region is None:
        stage = "detection"
    elif not ocr_exact:
        stage = "ocr"
    elif not complete:
        stage = "segmentation"
    elif not accepted:
        stage = "font_abstention"
    elif family != case["expected_family"]:
        stage = "font_mismatch"
    else:
        stage = None
    correct = stage is None
    return {"case_id": case["case_id"], "script": "han", "expected_text": case["text"],
            "expected_family": case["expected_family"], "cohort": cohort(case["expected_family"]),
            "positive_negative": "negative" if case["negative_family"] else "positive",
            "negative_kind": "known_Songti_counterexample_to_PingFang" if case["negative_family"] else None,
            "font_id": case["font_id"], "fixture_sha256": case["input_sha256"],
            "source_font_sha256": case["source_font_sha256"], "actual_text": text, "ocr_exact": ocr_exact,
            "selected_region_id": region.get("id") if region else None,
            "status": status, "predicted_family": family, "accepted": accepted,
            "font_name_correct": bool(accepted and family == case["expected_family"]),
            "wrong_font_name": bool(accepted and family != case["expected_family"]),
            "correct": correct, "failure_stage": stage,
            "outcome": "correct_candidate" if correct else "wrong_candidate" if accepted else "unconfirmed",
            "segmentation_status_ok": status_ok, "segmentation_complete": complete,
            "segmentation_geometry": geometry, "candidates": font.get("candidates", []),
            "reason_code": font.get("reason_code", font.get("reason")),
            "detected_regions": len(result.get("regions", [])), "elapsed_seconds": result["timing_seconds"]["total"]}


def score_latin(result, case):
    # Original tests/mobile_latin_eval.py scoring, including whitespace-only
    # OCR normalization and the stricter prohibition on a supported claim.
    regions = sorted(result["regions"], key=lambda row: (row["source_bbox"][1], row["source_bbox"][0]))
    text = " ".join(row["text"] for row in regions)
    ocr_exact = "".join(text.split()) == "".join(case["text"].split())
    expected = case["expected_family"]
    named = [row for row in regions if row["font"]["status"] in {"candidate", "supported"}]
    wrong = [row for row in named if not expected or row["font"]["family"] != expected or row["font"]["status"] == "supported"]
    complete = bool(regions) and len(named) == len(regions) and ocr_exact
    if wrong:
        outcome = "wrong_candidate"
    elif expected and complete:
        outcome = "correct_candidate"
    elif not expected and not named:
        outcome = "correct_rejection"
    else:
        outcome = "unconfirmed"
    return {"case_id": case["id"], "script": "latin", "expected_text": case["text"],
            "expected_family": expected, "cohort": cohort(expected),
            "positive_negative": "positive" if expected else "negative",
            "negative_kind": None if expected else "unknown_Times_or_Courier_rejection",
            "fixture_sha256": case["image_sha256"], "source_font_sha256": case["source_sha256"],
            "actual_text": text, "ocr_exact": ocr_exact, "accepted": bool(named),
            "font_name_correct": bool(expected and named and not wrong),
            "wrong_font_name": bool(wrong),
            "correct": outcome in {"correct_candidate", "correct_rejection"}, "outcome": outcome,
            "candidate_families": [row["font"]["family"] for row in named],
            "detected_regions": len(regions), "elapsed_seconds": result["timing_seconds"]["total"],
            "regions": [{"text": row["text"], "status": row["font"]["status"],
                         "family": row["font"]["family"], "candidates": row["font"]["candidates"],
                         "reason_code": row["font"].get("reason_code", row["font"].get("reason")),
                         "source_bbox": row["source_bbox"]} for row in regions]}


def counts(rows):
    outcomes = Counter(row["outcome"] for row in rows)
    return {"total": len(rows), "correct": sum(row["correct"] for row in rows),
            "wrong": outcomes["wrong_candidate"], "abstention": outcomes["unconfirmed"],
            "correct_candidates": outcomes["correct_candidate"], "correct_rejections": outcomes["correct_rejection"],
            "font_name_correct": sum(row["font_name_correct"] for row in rows),
            "wrong_font_names": sum(row["wrong_font_name"] for row in rows),
            "accepted": sum(row["accepted"] for row in rows), "ocr_exact": sum(row["ocr_exact"] for row in rows)}


def grouped(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key] or "unknown_negative"].append(row)
    return {name: counts(items) for name, items in sorted(groups.items())}


def markdown(report):
    lines = ["# CNN 固定样本端到端评测", "",
             "这是固定字体渲染样本的检测、OCR、分字、神经网络分类评测；没有独立标注的真实手机截图真值。",
             "iOS／Android 分组仅表示参考字体家族，不是图片拍摄设备或系统标签。", "",
             f"模型：`{report['model_version']}`；权重 SHA-256：`{report['neural_model_sha256']}`。", "",
             "| 分组 | 总数 | 正确字体名 | 错误字体名 | 命名数 | 严格正确 | 弃权 | 正确拒判 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for section in ("by_script", "by_cohort", "by_positive_negative", "by_family"):
        for name, item in report["metrics"][section].items():
            lines.append(f"| {section}/{name} | {item['total']} | {item['font_name_correct']} | {item['wrong_font_names']} | {item['accepted']} | {item['correct']} | {item['abstention']} | {item['correct_rejections']} |")
    lines += ["", "中文沿用原严格口径：OCR 完全一致、全部分字状态正常、字框覆盖真实墨迹至少 90%、水平字框容差 ±2px，且字体候选正确。",
              "拉丁沿用原口径：OCR 忽略空白后一致，所有检测区域给出正确候选；未知字体没有候选才是正确拒判。",
              "中文宋体 negative 是针对苹方的已知字体反例，仍应识别为 Songti SC；它与 Latin 未知字体拒判样本含义不同。",
              "字体名正确但裁字几何不完整的中文样例，仅未通过严格正确口径，不记为错误字体名。",
              "正例待确认计为弃权，不计正确；阈值在评测前后均校验未修改。", ""]
    return "\n".join(lines)


def evaluate(models, output):
    han_cases, latin_cases = checked_fixtures()
    pipeline = FontPipeline(models)
    if pipeline.neural is None:
        raise ValueError("Evaluation requires a manifest-declared trained neural model")
    now = datetime.now(timezone.utc)
    output = Path(output).resolve()
    run_dir = output / "runs" / now.strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True)
    model_meta = pipeline.directory / "neural/metadata.json"
    model_weights = pipeline.directory / "neural" / pipeline.neural.meta["model"]["path"]
    pinned = {"model_manifest": sha(pipeline.directory / "MANIFEST.json"), "neural_metadata": sha(model_meta),
              "neural_weights": sha(model_weights)}
    rows = []
    try:
        for index, case in enumerate(han_cases + latin_cases, 1):
            script = "han" if index <= len(han_cases) else "latin"
            case_id = case["case_id"] if script == "han" else case["id"]
            folder = run_dir / script / case_id
            extra = {}
            if script == "han":
                source = HAN / case["input_file"]
            else:
                source = run_dir / "latin_inputs" / (case_id + ".png")
                source.parent.mkdir(exist_ok=True)
                with Image.open(LATIN / case["image"]) as raw:
                    image = raw.convert("RGB")
                x, y = (750 - image.width) // 2, 125
                if x < 0 or y + image.height > 300:
                    raise ValueError("Fixed Latin fixture does not fit the original 750x300 canvas")
                canvas = Image.new("RGB", (750, 300), "white")
                canvas.paste(image, (x, y))
                canvas.save(source)
                extra = {"canvas_sha256": sha(source), "placement": [x, y, image.width, image.height]}
            result = pipeline.run(source, folder, case_id)
            if result.get("font_method") != "neural_network":
                raise ValueError("Evaluation unexpectedly used reference matching")
            row = score_han(result, case) if script == "han" else score_latin(result, case)
            row.update(extra, result_file=relative(folder / "result.json"))
            rows.append(row)
            print(f"[{index:02d}/67] {script}/{case_id}: {row['outcome']}; OCR={row['actual_text']!r}", flush=True)
            write_json(run_dir / "progress.json", {"completed": index, "total": 67, "cases": rows})
        after = {"model_manifest": sha(pipeline.directory / "MANIFEST.json"), "neural_metadata": sha(model_meta),
                 "neural_weights": sha(model_weights)}
        if after != pinned:
            raise ValueError("Model or thresholds changed during evaluation; reject this run")
        report = {"schema": "flux-glyph-neural-fixed-end-to-end-v1", "evaluated_at_utc": now.isoformat(),
                  "claim_scope": "controlled frozen rendered inputs only; no real-device font truth or device inference",
                  "model_version": pipeline.version, "model_directory": relative(pipeline.directory),
                  "model_manifest_sha256": pinned["model_manifest"], "neural_metadata_sha256": pinned["neural_metadata"],
                  "neural_model_sha256": pinned["neural_weights"], "families": pipeline.neural.families,
                  "gates": pipeline.neural.meta["gates"], "temperature": pipeline.neural.meta["temperature"],
                  "thresholds_modified": False, "oracle_boxes_or_tokens_used_for_prediction": False,
                  "fixture_resizing_used": False, "uncertain_positive_counted_as_correct": False,
                  "cohort_definition": {"ios_font_references": sorted(IOS_FAMILIES),
                                        "android_font_references": sorted(ANDROID_FAMILIES),
                                        "application_font_reference": ["Alipay Number"]},
                  "negative_definition": "Han Songti is a known-family negative against PingFang, not an unknown class; Latin Times/Courier require rejection.",
                  "scoring_sources": ["tests/accuracy_eval.py", "tests/mobile_latin_eval.py"],
                  "fixture_manifest_sha256": {"han_protocol": sha(HAN / "cases.json"),
                                               "han_generated": sha(HAN / "generated_manifest.json"),
                                               "latin_cases": sha(LATIN / "cases.json")},
                  "runtime_code_sha256": {name: sha(ROOT / "src/flux_glyph" / name) for name in
                                          ("pipeline.py", "neural_font.py", "ppocr.py", "segmentation.py", "glyph_preprocess.py", "models.py")},
                  "evaluation_code_sha256": sha(Path(__file__)),
                  "metrics": {"overall": counts(rows), "by_script": grouped(rows, "script"),
                              "by_cohort": grouped(rows, "cohort"), "by_family": grouped(rows, "expected_family"),
                              "by_positive_negative": grouped(rows, "positive_negative"),
                              "han": {"ios": counts([r for r in rows if r["script"] == "han" and r["cohort"] == "ios_font_references"]),
                                      "android": counts([r for r in rows if r["script"] == "han" and r["cohort"] == "android_font_references"])},
                              "latin": {key: counts([r for r in rows if r["script"] == "latin" and r["positive_negative"] == key])
                                        for key in ("positive", "negative")}}, "cases": rows}
        write_json(run_dir / "report.json", report)
        write_json(output / "report.json", report)
        (run_dir / "report.md").write_text(markdown(report), encoding="utf-8")
        (output / "report.md").write_text(markdown(report), encoding="utf-8")
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2), flush=True)
        return report
    finally:
        pipeline.bank.archive.close()
        if pipeline.latin_bank:
            pipeline.latin_bank.archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models", help="Manifest-verified model bundle/root with neural metadata")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-fixtures-only", action="store_true", help="Verify the 67 fixed input checksums without inference")
    args = parser.parse_args()
    if args.check_fixtures_only:
        han, latin = checked_fixtures()
        print(json.dumps({"han_cases": len(han), "latin_cases": len(latin), "fixture_checksums_verified": True}))
    else:
        evaluate(args.models, args.output)


if __name__ == "__main__":
    main()
