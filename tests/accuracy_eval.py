#!/usr/bin/env python3
"""Frozen controlled end-to-end accuracy evaluation for Flux Glyph.

The input images are generated once from source fonts whose SHA-256 and exact
PostScript face are declared in ``fixtures/font_accuracy/cases.json``.  A run
evaluates the full detector -> OCR -> source-ink segmentation -> font matcher
pipeline.  Abstentions/uncertain results are never counted as correct.

This is a controlled synthetic benchmark.  It cannot establish accuracy on
real Alipay screenshots, for which independently labelled real-font truth is
still required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from fontTools.ttLib import TTCollection, TTFont
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_glyph.font_matcher import CompactFontBank, han, score_font  # noqa: E402
from flux_glyph.pipeline import FontPipeline  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "font_accuracy"
CASES = FIXTURE / "cases.json"
INPUTS = FIXTURE / "inputs"
GENERATED = FIXTURE / "generated_manifest.json"
REPORT_JSON = ROOT / "docs" / "accuracy-report.json"
REPORT_MD = ROOT / "docs" / "accuracy-report.md"
FAILURES_PNG = ROOT / "docs" / "accuracy-failures.png"
PRE_FIX_JSON = ROOT / "docs" / "accuracy-report.pre-segmentation-fix.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def postscript_names(font: TTFont) -> set[str]:
    names: set[str] = set()
    for record in font["name"].names:
        if record.nameID == 6:
            try:
                names.add(record.toUnicode())
            except Exception:
                continue
    return names


def resolve_face(path: Path, postscript_name: str) -> int:
    """Resolve a TTC face by exact PostScript name; never guess or fall back."""
    if path.suffix.lower() in {".ttc", ".otc"}:
        collection = TTCollection(str(path), lazy=True)
        fonts = collection.fonts
    else:
        collection = None
        fonts = [TTFont(str(path), lazy=True)]
    try:
        matches = [index for index, font in enumerate(fonts) if postscript_name in postscript_names(font)]
    finally:
        for font in fonts:
            font.close()
        if collection is not None:
            collection.close()
    if len(matches) != 1:
        raise ValueError(f"expected one exact PostScript face {postscript_name!r} in {path}, found {matches}")
    return matches[0]


def font_sources(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_path = Path(protocol["font_manifest_source"])
    if not source_path.is_file():
        raise FileNotFoundError(f"source font manifest unavailable: {source_path}")
    source_rows = {row["font_id"]: row for row in load_json(source_path)}
    resolved: dict[str, dict[str, Any]] = {}
    for declared in protocol["fonts"]:
        font_id = declared["font_id"]
        if font_id not in source_rows:
            raise ValueError(f"font absent from source manifest: {font_id}")
        row = source_rows[font_id]
        path = Path(row["path"])
        if not path.is_file():
            raise FileNotFoundError(f"font file unavailable: {path}")
        actual_hash = digest(path)
        for key in ("postscript_name", "source_sha256"):
            if row[key] != declared[key]:
                raise ValueError(f"source manifest disagrees for {font_id} {key}")
        if actual_hash != declared["source_sha256"]:
            raise ValueError(f"font checksum mismatch: {font_id}")
        face_index = resolve_face(path, declared["postscript_name"])
        resolved[font_id] = {**declared, "path": str(path), "face_index": face_index}
    return resolved


def palette(name: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    values = {
        "white": ((250, 251, 253), (30, 34, 40)),
        "alipay_blue": ((22, 119, 255), (255, 255, 255)),
        "pale_blue": ((235, 246, 255), (20, 55, 92)),
    }
    return values[name]


def render_case(font_row: dict[str, Any], profile: dict[str, Any], destination: Path) -> dict[str, Any]:
    width, height = 750, 420
    background, foreground = palette(profile["background"])
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(
        font_row["path"], profile["font_size"], index=font_row["face_index"], layout_engine=ImageFont.Layout.BASIC
    )
    # Make failure loud if Pillow did not load a useful Chinese face.
    if font.getmask(profile["text"]).getbbox() is None:
        raise ValueError(f"font rendered an empty Chinese probe: {font_row['font_id']}")

    text = profile["text"]
    advances = [float(draw.textlength(char, font=font)) for char in text]
    total_width = sum(advances)
    x = int(round((width - total_width) / 2))
    y = 176
    glyph_boxes: list[list[int]] = []
    ink_boxes: list[list[int]] = []
    cursor = float(x)
    for char, advance in zip(text, advances):
        position = (round(cursor), y)
        draw.text(position, char, font=font, fill=foreground)
        # Keep oracle cells disjoint.  Padding each ink bbox independently can
        # leak pixels from its neighbour and makes the supposed oracle invalid.
        glyph_boxes.append([max(0, math.floor(cursor)), max(0, y - 4),
                            min(width, math.ceil(cursor + advance)), min(height, y + profile["font_size"] + 10)])
        cursor += advance
    for cell in glyph_boxes:
        crop = image.crop(tuple(cell))
        difference = ImageChops.difference(crop, Image.new("RGB", crop.size, background))
        local_ink = difference.getbbox()
        if local_ink is None:
            raise ValueError("rendered glyph has no source pixels")
        ink_boxes.append([cell[0] + local_ink[0], cell[1] + local_ink[1],
                          cell[0] + local_ink[2], cell[1] + local_ink[3]])
    union = [min(b[0] for b in glyph_boxes), min(b[1] for b in glyph_boxes),
             max(b[2] for b in glyph_boxes), max(b[3] for b in glyph_boxes)]

    # Sparse non-text UI geometry keeps the fixture screenshot-like without
    # contaminating ground truth with labels or legacy model output.
    accent = (215, 224, 235) if profile["background"] != "alipay_blue" else (90, 158, 248)
    draw.rounded_rectangle((28, 28, 104, 40), radius=6, fill=accent)
    draw.rounded_rectangle((650, 28, 722, 40), radius=6, fill=accent)
    draw.line((28, 360, 722, 360), fill=accent, width=2)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if profile["format"] == "JPEG":
        image.save(destination, "JPEG", quality=profile["jpeg_quality"], subsampling=0, optimize=False, progressive=False)
    else:
        image.save(destination, "PNG", optimize=False)
    return {"input_sha256": digest(destination), "image_size": [width, height],
            "expected_text_bbox": union, "expected_glyph_bboxes": glyph_boxes,
            "expected_ink_bboxes": ink_boxes}


def generate() -> dict[str, Any]:
    protocol = load_json(CASES)
    if not protocol.get("protocol_frozen_before_first_run"):
        raise ValueError("accuracy protocol must be frozen before generation")
    sources = font_sources(protocol)
    INPUTS.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()
    generated_cases = []
    for declared in protocol["fonts"]:
        source = sources[declared["font_id"]]
        for profile in protocol["render_profiles"]:
            suffix = ".jpg" if profile["format"] == "JPEG" else ".png"
            case_id = f"{declared['font_id']}--{profile['id']}"
            name = case_id + suffix
            expected_names.add(name)
            details = render_case(source, profile, INPUTS / name)
            generated_cases.append({
                "case_id": case_id,
                "input_file": f"inputs/{name}",
                "font_id": declared["font_id"],
                "expected_family": declared["expected_family"],
                "negative_family": bool(declared.get("negative_family", False)),
                "source_font_sha256": declared["source_sha256"],
                "postscript_name": declared["postscript_name"],
                "face_index": source["face_index"],
                **profile,
                **details,
            })
    extras = sorted(path.name for path in INPUTS.iterdir() if path.is_file() and path.name not in expected_names)
    if extras:
        raise ValueError(f"undeclared input fixtures present: {extras}")
    manifest = {
        "schema": "flux-glyph-controlled-font-e2e-generated-v1",
        "protocol_sha256": digest(CASES),
        "case_count": len(generated_cases),
        "font_count": len(protocol["fonts"]),
        "render_profile_count": len(protocol["render_profiles"]),
        "source_fonts_verified": True,
        "exact_postscript_faces_verified": True,
        "cases": generated_cases,
    }
    save_json(GENERATED, manifest)
    return manifest


def ratio(expected: str, actual: str) -> float:
    return SequenceMatcher(a=expected, b=actual).ratio()


def target_region(result: dict[str, Any], expected: str) -> dict[str, Any] | None:
    regions = result.get("regions", [])
    if not regions:
        return None
    return max(regions, key=lambda row: (row.get("text") == expected, ratio(expected, row.get("text", "")),
                                         row.get("detector_score", 0.0)))


def oracle_match(bank: CompactFontBank, image_path: Path, case: dict[str, Any]) -> dict[str, Any]:
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    samples = []
    for char, box in zip(case["text"], case["expected_glyph_bboxes"]):
        samples.append({"character": char, "image": image.crop(tuple(box))})
    scored = score_font(bank, samples, complete=True)
    family = "PingFang SC" if scored["accepted_pingfang"] else scored["family"]
    return {"family": family, "correct": family == case["expected_family"],
            "accepted_pingfang": scored["accepted_pingfang"], "candidates": scored["candidates"]}


def safe_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def box_intersection_fraction(candidate: list[int] | None, truth: list[int]) -> float:
    if not candidate:
        return 0.0
    left, top = max(candidate[0], truth[0]), max(candidate[1], truth[1])
    right, bottom = min(candidate[2], truth[2]), min(candidate[3], truth[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    truth_area = max(0, truth[2] - truth[0]) * max(0, truth[3] - truth[1])
    return intersection / truth_area if truth_area else 0.0


def segmentation_geometry(glyphs: list[dict[str, Any]], case: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, character in enumerate(case["text"]):
        glyph = glyphs[index] if index < len(glyphs) and glyphs[index].get("character") == character else {}
        predicted = glyph.get("source_bbox")
        cell = case["expected_glyph_bboxes"][index]
        ink = case["expected_ink_bboxes"][index]
        coverage = box_intersection_fraction(predicted, ink)
        within_cell = bool(predicted and predicted[0] >= cell[0] - 2 and predicted[2] <= cell[2] + 2)
        rows.append({"character": character, "predicted_bbox": predicted, "truth_cell_bbox": cell,
                     "truth_ink_bbox": ink, "truth_ink_bbox_coverage": round(coverage, 6),
                     "within_character_cell_tolerance": within_cell,
                     "geometry_ok": coverage >= 0.9 and within_cell})
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def one(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        accepted = sum(row["accepted"] for row in items)
        name_correct = sum(row["font_name_correct"] for row in items)
        strict_correct = sum(row["correct"] for row in items)
        return {
            "total": total,
            "accepted": accepted,
            "font_name_correct": name_correct,
            "strict_correct": strict_correct,
            "accepted_precision": safe_rate(name_correct, accepted),
            "strict_evidence_complete_precision": safe_rate(strict_correct, accepted),
            "coverage": safe_rate(accepted, total),
            "total_name_correct_rate": safe_rate(name_correct, total),
            "total_correct_rate": safe_rate(strict_correct, total),
            "ocr_exact_rate": safe_rate(sum(row["ocr_exact"] for row in items), total),
            "segmentation_status_ok_rate": safe_rate(sum(row["segmentation_status_ok"] for row in items), total),
            "segmentation_complete_rate": safe_rate(sum(row["segmentation_complete"] for row in items), total),
            "oracle_match_correct_rate": safe_rate(sum(row["oracle_match"]["correct"] for row in items), total),
        }
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["expected_family"]].append(row)
    non_pf = [row for row in rows if row["expected_family"] != "PingFang SC"]
    false_pf = sum(row["accepted"] and row["predicted_family"] == "PingFang SC" for row in non_pf)
    return {
        "overall": one(rows),
        "by_family": {family: one(items) for family, items in sorted(by_family.items())},
        "failure_counts": {
            "detection_miss": sum(row["failure_stage"] == "detection" for row in rows),
            "ocr_or_region_selection": sum(row["failure_stage"] == "ocr" for row in rows),
            "segmentation": sum(row["failure_stage"] == "segmentation" for row in rows),
            "font_abstention": sum(row["failure_stage"] == "font_abstention" for row in rows),
            "font_mismatch": sum(row["failure_stage"] == "font_mismatch" for row in rows),
        },
        "non_pingfang_false_pingfang_accepts": false_pf,
        "non_pingfang_cases": len(non_pf),
        "non_pingfang_false_pingfang_rate": safe_rate(false_pf, len(non_pf)),
    }


def failure_stage(region: dict[str, Any] | None, expected: str, accepted: bool,
                  family: str | None, expected_family: str, segmentation_complete: bool) -> str | None:
    if region is None:
        return "detection"
    if region.get("text") != expected:
        return "ocr"
    if not segmentation_complete:
        return "segmentation"
    if not accepted:
        return "font_abstention"
    if family != expected_family:
        return "font_mismatch"
    return None


def create_failure_montage(rows: list[dict[str, Any]]) -> list[str]:
    failures = [row for row in rows if not row["correct"]]
    # Deterministic representation: first failure from up to six distinct stages/families.
    selected, seen = [], set()
    for row in failures:
        key = (row["failure_stage"], row["expected_family"])
        if key not in seen:
            selected.append(row)
            seen.add(key)
        if len(selected) == 6:
            break
    if not selected:
        if FAILURES_PNG.exists():
            FAILURES_PNG.unlink()
        return []
    cell_w, cell_h = 480, 285
    sheet = Image.new("RGB", (cell_w * 2, cell_h * math.ceil(len(selected) / 2)), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = ImageFont.truetype(str(ROOT / "assets" / "annotation.otf"), 14)
    for index, row in enumerate(selected):
        x, y = (index % 2) * cell_w, (index // 2) * cell_h
        with Image.open(FIXTURE / row["input_file"]) as opened:
            image = ImageOps.contain(opened.convert("RGB"), (cell_w - 20, 195))
        sheet.paste(image, (x + 10, y + 74))
        profile = row["case_id"].split("--", 1)[-1]
        label = (f"font={row['font_id']}  profile={profile}\nstage={row['failure_stage']}\n"
                 f"expected={row['expected_family']}  predicted={row['predicted_family'] or 'NONE'}")
        draw.multiline_text((x + 10, y + 6), label, font=label_font, fill="#111827", spacing=2)
        draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline="#9ca3af", width=1)
    sheet.save(FAILURES_PNG, "PNG", optimize=False)
    return [row["case_id"] for row in selected]


def markdown(report: dict[str, Any]) -> str:
    summary = report["metrics"]["overall"]
    lines = [
        "# Flux Glyph 字体精度验证",
        "",
        f"评测时间：{report['evaluated_at_utc']}",
        "",
        "> 结论边界：这是由已校验原始字体文件生成的受控合成端到端评测，不是真实支付宝截图评测，不能据此声称真实截图字体识别率达到 90%。当前没有独立标注的真实字体真值集。",
        "",
        "## 严格口径结果",
        "",
        f"固定样本共 {summary['total']} 张。字体名 accepted precision = {summary['accepted_precision']}，strict evidence-complete precision = {summary['strict_evidence_complete_precision']}，coverage = {summary['coverage']}，total name-correct rate = {summary['total_name_correct_rate']}，strict total correct rate = {summary['total_correct_rate']}。uncertain/待确认一律计为未正确。",
        "",
        f"OCR exact rate = {summary['ocr_exact_rate']}；分字完整率 = {summary['segmentation_complete_rate']}；真值字框直送 matcher 的 oracle 正确率 = {summary['oracle_match_correct_rate']}。",
        "",
        "| 字体 | name precision | strict precision | coverage | total name correct | strict total correct | OCR exact | 分字状态ok | 分字几何完整 | oracle matcher | name/strict/accepted/total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, item in report["metrics"]["by_family"].items():
        lines.append(f"| {family} | {item['accepted_precision']} | {item['strict_evidence_complete_precision']} | {item['coverage']} | {item['total_name_correct_rate']} | {item['total_correct_rate']} | {item['ocr_exact_rate']} | {item['segmentation_status_ok_rate']} | {item['segmentation_complete_rate']} | {item['oracle_match_correct_rate']} | {item['font_name_correct']}/{item['strict_correct']}/{item['accepted']}/{item['total']} |")
    if "comparison_to_pre_segmentation_fix" in report:
        comparison = report["comparison_to_pre_segmentation_fix"]
        lines += ["", "## 与最初分字修复前基线比较", "",
                  "| 指标 | 修复前 | 修复后 | delta |", "|---|---:|---:|---:|"]
        for key, item in comparison.items():
            lines.append(f"| {key} | {item['before']} | {item['after']} | {item['delta']:+.6f} |")
    failures = report["metrics"]["failure_counts"]
    lines += [
        "",
        "## 失败归因",
        "",
        f"检测遗漏 {failures['detection_miss']}；识字/区域选择失败 {failures['ocr_or_region_selection']}；分字失败 {failures['segmentation']}；字体拒识 {failures['font_abstention']}；字体误判 {failures['font_mismatch']}。",
        "",
        f"非苹方样本误接受为苹方：{report['metrics']['non_pingfang_false_pingfang_accepts']}/{report['metrics']['non_pingfang_cases']} ({report['metrics']['non_pingfang_false_pingfang_rate']})。",
        "",
        "oracle matcher 与端到端之间的差值用于定位 detector/OCR/CTC+墨迹分字带来的退化；它不代表生产上可获得的准确率。每条样本的文字、分字状态、预测和前三候选见 JSON 报告。",
        "",
        "## 固定协议",
        "",
        f"协议哈希 `{report['protocol_sha256']}`；输入清单哈希 `{report['generated_manifest_sha256']}`。共 6 个字体家族 × 6 个预先冻结的字号/背景/压缩组合。TTC/OTC 用 fontTools 按 exact PostScript name 解出 face index，字体文件 SHA-256 与旧仓字体清单逐项一致；没有字体 fallback。",
        "",
        "## 真实截图缺口",
        "",
        "状态栏、页面标签或旧模型输出都没有被当成字体真值。本报告只能证明当前受控条件表现。要验证用户要求的真实支付宝截图 90% 目标，需要另建独立人工标注集，并按同一 accepted precision / coverage / total correct 口径盲测。",
    ]
    if report["failure_montage_cases"]:
        lines += ["", "代表失败样本可视化：`docs/accuracy-failures.png`。"]
    return "\n".join(lines) + "\n"


def evaluate() -> dict[str, Any]:
    protocol = load_json(CASES)
    manifest = load_json(GENERATED)
    if manifest["protocol_sha256"] != digest(CASES):
        raise ValueError("generated fixtures do not match the frozen protocol")
    if manifest["case_count"] != len(protocol["fonts"]) * len(protocol["render_profiles"]):
        raise ValueError("generated fixture count disagrees with frozen cross-product")
    pipeline = FontPipeline(ROOT / "models")
    bank = CompactFontBank(ROOT / "models" / "font", max_characters=64)
    rows = []
    with tempfile.TemporaryDirectory(prefix="flux-glyph-accuracy-") as temporary:
        output_root = Path(temporary)
        for index, case in enumerate(manifest["cases"], 1):
            input_path = FIXTURE / case["input_file"]
            if digest(input_path) != case["input_sha256"]:
                raise ValueError(f"input fixture checksum mismatch: {case['case_id']}")
            result = pipeline.run(input_path, output_root / case["case_id"], case["case_id"])
            region = target_region(result, case["text"])
            actual_text = region.get("text", "") if region else ""
            glyphs = region.get("glyphs", []) if region else []
            segmentation_status_ok = bool(
                actual_text == case["text"]
                and [glyph.get("character") for glyph in glyphs] == list(case["text"])
                and all(glyph.get("status") == "ok" for glyph in glyphs)
            )
            geometry = segmentation_geometry(glyphs, case)
            segmentation_complete = segmentation_status_ok and all(item["geometry_ok"] for item in geometry)
            status = region.get("font", {}).get("status") if region else None
            predicted_family = region.get("font", {}).get("family") if region else None
            accepted = status in {"supported", "candidate"} and predicted_family is not None
            stage = failure_stage(region, case["text"], accepted, predicted_family,
                                  case["expected_family"], segmentation_complete)
            correct = stage is None
            font_name_correct = bool(accepted and predicted_family == case["expected_family"])
            rows.append({
                "case_id": case["case_id"], "input_file": case["input_file"],
                "font_id": case["font_id"], "expected_family": case["expected_family"],
                "negative_family": case["negative_family"], "expected_text": case["text"],
                "actual_text": actual_text, "ocr_exact": actual_text == case["text"],
                "detected_regions": len(result.get("regions", [])),
                "selected_region_id": region.get("id") if region else None,
                "detector_score": region.get("detector_score") if region else None,
                "ocr_confidence": region.get("ocr_confidence") if region else None,
                "segmentation_status_ok": segmentation_status_ok,
                "segmentation_complete": segmentation_complete,
                "segmentation_geometry": geometry,
                "glyph_statuses": [{"character": glyph.get("character"), "status": glyph.get("status"),
                                    "reason": glyph.get("reason")} for glyph in glyphs],
                "status": status, "accepted": accepted, "predicted_family": predicted_family,
                "font_name_correct": font_name_correct, "correct": correct, "failure_stage": stage,
                "candidates": region.get("font", {}).get("candidates", []) if region else [],
                "oracle_match": oracle_match(bank, input_path, case),
                "timing_seconds": result.get("timing_seconds", {}),
            })
            print(f"[{index:02d}/{manifest['case_count']}] {case['case_id']}: "
                  f"{('OK' if correct else stage)} text={actual_text!r} family={predicted_family!r}", flush=True)
    report = {
        "schema": "flux-glyph-controlled-font-e2e-report-v1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "controlled synthetic end-to-end only; no real Alipay screenshot font truth",
        "uncertain_counted_as_correct": False,
        "strict_correct_definition": "detected selected region + exact OCR + every expected Han glyph segmented ok + each source bbox covers >=90% of the known rendered ink bbox and stays within its character cell (+/-2 px) + accepted family + exact expected family",
        "protocol_sha256": digest(CASES),
        "generated_manifest_sha256": digest(GENERATED),
        "model_version": pipeline.version,
        "model_bundle_manifest_sha256": digest(ROOT / "models" / "MANIFEST.json"),
        "runtime_code_sha256": {name: digest(ROOT / "src" / "flux_glyph" / name)
                                for name in ("pipeline.py", "ppocr.py", "segmentation.py", "font_matcher.py",
                                             "detection.py", "foreground.py", "glyph_preprocess.py", "models.py", "latin_matcher.py")},
        "metrics": summarize(rows),
        "cases": rows,
    }
    if PRE_FIX_JSON.is_file():
        baseline = load_json(PRE_FIX_JSON)
        for row in baseline["cases"]:
            row.setdefault("font_name_correct", bool(row.get("accepted") and
                                                      row.get("predicted_family") == row.get("expected_family")))
        current = report["metrics"]["overall"]
        previous = summarize(baseline["cases"])["overall"]
        report["comparison_to_pre_segmentation_fix"] = {
            key: {"before": previous[key], "after": current[key],
                  "delta": round(current[key] - previous[key], 6)}
            for key in ("accepted_precision", "strict_evidence_complete_precision", "coverage",
                        "total_name_correct_rate", "total_correct_rate",
                        "ocr_exact_rate", "segmentation_status_ok_rate", "segmentation_complete_rate")
            if current.get(key) is not None and previous.get(key) is not None
        }
    report["failure_montage_cases"] = create_failure_montage(rows)
    save_json(REPORT_JSON, report)
    REPORT_MD.write_text(markdown(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="regenerate fixed input fixtures from verified source fonts")
    parser.add_argument("--evaluate", action="store_true", help="run the full pipeline and write docs/accuracy-report.*")
    parser.add_argument("--save-pre-fix-baseline", action="store_true",
                        help="also freeze this run as the pre-segmentation-fix comparison")
    args = parser.parse_args()
    if not args.generate and not args.evaluate:
        args.evaluate = True
    if args.generate:
        manifest = generate()
        print(f"generated {manifest['case_count']} verified fixtures")
    if args.evaluate:
        report = evaluate()
        if args.save_pre_fix_baseline:
            report.pop("comparison_to_pre_segmentation_fix", None)
            save_json(PRE_FIX_JSON, report)
            (ROOT / "docs" / "accuracy-report.pre-segmentation-fix.md").write_text(markdown(report), encoding="utf-8")
            save_json(REPORT_JSON, report)
            REPORT_MD.write_text(markdown(report), encoding="utf-8")
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
