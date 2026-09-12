#!/usr/bin/env python3
"""Frozen, resumable full-pipeline evaluation on held-out iOS screenshots.

Native boxes/text/fonts are used for auditing and scoring only. FontPipeline
receives the original PNG path and model bundle, never the native annotation.
These are actual Simulator screenshots of controlled, generated native scenes;
they do not measure accuracy on independently collected third-party app UIs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from flux_glyph.pipeline import FontPipeline  # noqa: E402
from flux_glyph.models import verify_bundle  # noqa: E402
from training.capture import prepare_captured as native  # noqa: E402

SCHEMA = "flux-glyph-ios-screenshot-evaluation-v1"
RULES = {
    "split": "test", "matching": "maximum_cardinality_then_total_iou_one_to_one",
    "truth_bbox": "native.ink_bbox_pixels", "prediction_bbox": "detector_bbox",
    "minimum_iou": .5, "pairing_uses_text_or_font": False,
    "ocr_normalization": "remove_unicode_whitespace_only_case_and_punctuation_preserved",
    "font_accepted_statuses": ["candidate", "supported"],
    "size_relative_error_tolerance": .10, "color_max_channel_error_tolerance": 8,
    "font_name_scoring_independent_of_ocr": True,
    "strict_e2e_correct": "matched_and_ocr_normalized_exact_and_accepted_correct_font",
    "no_detection_is_not_font_abstention": True,
    "negative_font_samples": "none_unless_verified_native_unknown_font_labels_are_supplied",
}


def require(condition, message):
    if not condition:
        raise ValueError("iOS evaluation: " + message)


def sha(path):
    return native.sha(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def valid_box(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4 and
            all(type(x) in (int, float) and math.isfinite(x) for x in box) and
            box[0] < box[2] and box[1] < box[3])


def iou(first, second):
    if not valid_box(first) or not valid_box(second):
        return 0.
    intersection = max(0., min(first[2], second[2]) - max(first[0], second[0])) * max(
        0., min(first[3], second[3]) - max(first[1], second[1]))
    union = ((first[2] - first[0]) * (first[3] - first[1]) +
             (second[2] - second[0]) * (second[3] - second[1]) - intersection)
    return intersection / union


def maximum_assignment(weights):
    """Rectangular Hungarian assignment, rows <= columns; deterministic ties."""
    n, m = len(weights), len(weights[0]) if weights else 0
    require(n <= m and all(len(row) == m for row in weights), "invalid assignment matrix")
    u, v, p, way = [0.] * (n + 1), [0.] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for row in range(1, n + 1):
        p[0], column = row, 0
        distances, used = [math.inf] * (m + 1), [False] * (m + 1)
        while True:
            used[column] = True
            current_row, delta, next_column = p[column], math.inf, 0
            for candidate in range(1, m + 1):
                if not used[candidate]:
                    cost = -weights[current_row - 1][candidate - 1] - u[current_row] - v[candidate]
                    if cost < distances[candidate]:
                        distances[candidate], way[candidate] = cost, column
                    if distances[candidate] < delta:
                        delta, next_column = distances[candidate], candidate
            for candidate in range(m + 1):
                if used[candidate]:
                    u[p[candidate]] += delta
                    v[candidate] -= delta
                else:
                    distances[candidate] -= delta
            column = next_column
            if p[column] == 0:
                break
        while column:
            previous = way[column]
            p[column] = p[previous]
            column = previous
    result = [None] * n
    for column in range(1, m + 1):
        if p[column]:
            result[p[column] - 1] = column - 1
    return result


def pair_regions(truths, predictions):
    """Geometry alone; maximize count before summed IoU, using dummy columns."""
    require(len(truths) <= 128 and len(predictions) <= 5000, "region count exceeds evaluation limit")
    overlaps = [[iou(truth.get("ink_bbox_pixels"), pred.get("detector_bbox"))
                 for pred in predictions] for truth in truths]
    bonus = min(len(truths), len(predictions)) + 1
    weights = [[bonus + overlap if overlap >= RULES["minimum_iou"] else 0. for overlap in row]
               + [0.] * len(truths) for row in overlaps]
    assignments = maximum_assignment(weights)
    return [(truth, pred, overlaps[truth][pred]) for truth, pred in enumerate(assignments)
            if pred is not None and pred < len(predictions) and overlaps[truth][pred] >= RULES["minimum_iou"]]


def normalize_text(value):
    return "".join(value.split())


def edit_distance(expected, actual):
    previous = list(range(len(actual) + 1))
    for i, first in enumerate(expected, 1):
        current = [i]
        for j, second in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (first != second)))
        previous = current
    return previous[-1]


def parse_color(value):
    if not isinstance(value, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?", value):
        return None
    components = [int(value[i:i + 2], 16) for i in range(1, len(value), 2)]
    return components if len(components) == 4 else components + [255]


def score_style(style, truth):
    style = style if isinstance(style, dict) else {}
    actual = parse_color(truth.get("actual_text_color_hex"))
    background = parse_color(truth.get("background_hex"))
    predicted = parse_color(style.get("text_color_hex"))
    # The runtime estimates visible pixels, not source alpha. Compare visible
    # composited RGB; never claim to recover original alpha from a screenshot.
    expected_rgb = ([actual[i] * actual[3] / 255 + background[i] * (1 - actual[3] / 255)
                     for i in range(3)] if actual and background and background[3] == 255 else None)
    color_available = bool(predicted and predicted[3] == 255 and expected_rgb and
                           style.get("color", {}).get("status") == "estimated")
    delta = [abs(predicted[i] - expected_rgb[i]) for i in range(3)] if color_available else None
    size = style.get("font_size_px_estimate")
    expected_size = truth.get("font_size_screen_px")
    size_available = (type(size) in (int, float) and math.isfinite(size) and size > 0 and
                      type(expected_size) in (int, float) and math.isfinite(expected_size) and expected_size > 0 and
                      style.get("size", {}).get("status") == "estimated")
    size_error = abs(size - expected_size) if size_available else None
    interval = style.get("font_size_px_interval")
    valid_interval = (size_available and isinstance(interval, list) and len(interval) == 2 and
                      all(type(x) in (int, float) and math.isfinite(x) for x in interval) and
                      0 < interval[0] <= interval[1])
    return {"color_available": color_available, "color_expected_visible_rgb": expected_rgb,
            "color_predicted_hex": style.get("text_color_hex"),
            "color_max_channel_error": max(delta) if delta else None,
            "color_rgb_euclidean_error": math.sqrt(sum(x * x for x in delta)) if delta else None,
            "color_within_tolerance": bool(delta is not None and max(delta) <= RULES["color_max_channel_error_tolerance"]),
            "size_available": bool(size_available), "size_expected_px": expected_size,
            "size_predicted_px": size if size_available else None,
            "size_absolute_error_px": size_error,
            "size_relative_error": size_error / expected_size if size_available else None,
            "size_within_tolerance": bool(size_available and size_error / expected_size <= RULES["size_relative_error_tolerance"]),
            "size_interval_available": bool(valid_interval),
            "size_interval_covers_native": bool(valid_interval and interval[0] <= expected_size <= interval[1]),
            "color_reason": style.get("color", {}).get("reason"), "size_reason": style.get("size", {}).get("reason")}


def accepted_font(prediction):
    font = prediction.get("font", {})
    # A mixed-scope prediction has no single family. Components are retained in
    # raw outputs; native scenes are pure-script and do not justify choosing one.
    accepted = (font.get("status") in RULES["font_accepted_statuses"] and
                isinstance(font.get("family"), str) and bool(font["family"]) and not font.get("components"))
    return accepted, font


def score_page(result, source):
    truths, predictions = source["regions"], result.get("regions", [])
    require(isinstance(predictions, list) and all(isinstance(p, dict) for p in predictions), "invalid pipeline regions")
    pairs = {truth: (pred, overlap) for truth, pred, overlap in pair_regions(truths, predictions)}
    used = {value[0] for value in pairs.values()}
    rows = []
    for index, truth in enumerate(truths):
        found = index in pairs
        prediction_index, overlap = pairs.get(index, (None, None))
        prediction = predictions[prediction_index] if found else {}
        text = prediction.get("text", "")
        require(isinstance(text, str), "invalid OCR output")
        expected, actual = normalize_text(truth["text"]), normalize_text(text)
        accepted, font = accepted_font(prediction)
        verified = truth.get("font_truth_verified") is True
        name_correct = bool(found and verified and accepted and font["family"] == truth["font_family"])
        wrong_name = bool(found and verified and accepted and font["family"] != truth["font_family"])
        row = {"source_id": source["source_id"], "page_id": source["page_id"], "region_id": truth["id"],
               "script": truth["script"], "expected_family": truth["font_family"],
               "expected_text": truth["text"], "truth_bbox": truth["ink_bbox_pixels"],
               "detected": found, "prediction_index": prediction_index,
               "prediction_id": prediction.get("id"), "iou": overlap, "actual_text": text,
               "ocr_exact": bool(found and text == truth["text"]),
               "ocr_normalized_exact": bool(found and actual == expected),
               "ocr_edit_distance": edit_distance(expected, actual), "ocr_truth_characters": len(expected),
               "font_truth_verified": verified, "accepted": bool(found and accepted),
               "predicted_family": font.get("family"), "font_status": font.get("status"),
               "font_name_correct": name_correct, "wrong_font_name": wrong_name,
               "font_abstained": bool(found and verified and not accepted),
               "font_unscorable": bool(found and not verified),
               "strict_e2e_correct": bool(name_correct and actual == expected),
               "font_reason": font.get("reason_code", font.get("reason")),
               "style": score_style(prediction.get("text_style"), truth)}
        rows.append(row)
    extras = []
    for index, prediction in enumerate(predictions):
        if index in used:
            continue
        accepted, font = accepted_font(prediction)
        extras.append({"prediction_index": index, "prediction_id": prediction.get("id"),
                       "detector_bbox": prediction.get("detector_bbox"), "text": prediction.get("text", ""),
                       "accepted_font": accepted, "family": font.get("family"), "font_status": font.get("status")})
    return {"source_id": source["source_id"], "page_id": source["page_id"], "truth_count": len(truths),
            "prediction_count": len(predictions), "rows": rows, "false_positives": extras,
            "timing_seconds": result.get("timing_seconds", {})}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(rows, extras=(), *, prediction_count=None):
    n = len(rows)
    detected = sum(row["detected"] for row in rows)
    chars = sum(row["ocr_truth_characters"] for row in rows)
    matched_chars = sum(row["ocr_truth_characters"] for row in rows if row["detected"])
    distances = sum(row["ocr_edit_distance"] for row in rows)
    matched_distances = sum(row["ocr_edit_distance"] for row in rows if row["detected"])
    accepted, correct, wrong, abstained = (sum(row[key] for row in rows) for key in
                                         ("accepted", "font_name_correct", "wrong_font_name", "font_abstained"))
    result = {"truth_regions": n, "detected": detected, "missed": n - detected,
              "detection_recall": ratio(detected, n),
              "ocr_exact": sum(row["ocr_exact"] for row in rows),
              "ocr_normalized_exact": sum(row["ocr_normalized_exact"] for row in rows),
              "ocr_normalized_accuracy_all_truth": ratio(sum(row["ocr_normalized_exact"] for row in rows), n),
              "ocr_normalized_accuracy_detected": ratio(sum(row["ocr_normalized_exact"] for row in rows), detected),
              "ocr_edit_distance_all_truth": distances, "ocr_truth_characters": chars,
              "ocr_cer_all_truth_including_misses": ratio(distances, chars),
              "ocr_cer_detected": ratio(matched_distances, matched_chars),
              "font_accepted": accepted, "font_name_correct": correct, "wrong_font_names": wrong,
              "font_abstained_after_detection": abstained,
              "font_unscorable_after_detection": sum(row["font_unscorable"] for row in rows),
              "font_name_accuracy_all_truth": ratio(correct, n),
              "font_name_precision_among_accepted": ratio(correct, correct + wrong),
              "strict_e2e_correct": sum(row["strict_e2e_correct"] for row in rows),
              "strict_e2e_accuracy": ratio(sum(row["strict_e2e_correct"] for row in rows), n)}
    if prediction_count is not None:
        result.update(predicted_regions=prediction_count, false_positives=len(extras),
                      false_positive_named_fonts=sum(row["accepted_font"] for row in extras),
                      false_positive_ocr_characters=sum(len(normalize_text(row["text"])) for row in extras),
                      detection_precision=ratio(detected, prediction_count))
    styles = [row["style"] for row in rows]
    color = [row for row in styles if row["color_available"]]
    size = [row for row in styles if row["size_available"]]
    intervals = [row for row in styles if row["size_interval_available"]]
    result["style"] = {"color_available": len(color), "color_unavailable": n - len(color),
                       "color_within_tolerance": sum(row["color_within_tolerance"] for row in color),
                       "color_max_channel_mae": ratio(sum(row["color_max_channel_error"] for row in color), len(color)),
                       "size_available": len(size), "size_unavailable": n - len(size),
                       "size_within_tolerance": sum(row["size_within_tolerance"] for row in size),
                       "size_mae_px": ratio(sum(row["size_absolute_error_px"] for row in size), len(size)),
                       "size_mean_relative_error": ratio(sum(row["size_relative_error"] for row in size), len(size)),
                       "size_empirical_intervals": len(intervals),
                       "size_empirical_interval_coverage": ratio(sum(row["size_interval_covers_native"] for row in intervals), len(intervals))}
    return result


def aggregate(pages):
    rows = [row for page in pages for row in page["rows"]]
    extras = [row for page in pages for row in page["false_positives"]]
    result = {"pages": len(pages), "overall": summarize(rows, extras, prediction_count=sum(p["prediction_count"] for p in pages))}
    for key in ("script", "expected_family"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        result["by_" + key] = {name: summarize(items) for name, items in sorted(groups.items())}
    return result


def paired_comparison(baseline, neural):
    left = {(row["source_id"], row["region_id"]): row for page in baseline for row in page["rows"]}
    right = {(row["source_id"], row["region_id"]): row for page in neural for row in page["rows"]}
    require(left.keys() == right.keys(), "paired methods must contain identical held-out truths")
    return {"truth_regions": len(left), **{metric: dict(Counter(
        "both_correct" if left[key][metric] and right[key][metric] else
        "neural_only_correct" if right[key][metric] else
        "baseline_only_correct" if left[key][metric] else "neither_correct" for key in left))
        for metric in ("detected", "ocr_normalized_exact", "font_name_correct", "strict_e2e_correct")},
        "new_wrong_font_names": sum(right[key]["wrong_font_name"] and not left[key]["wrong_font_name"] for key in left),
        "resolved_wrong_font_names": sum(left[key]["wrong_font_name"] and not right[key]["wrong_font_name"] for key in left)}


def checked_captures(path):
    """Audit every split, then expose only test PNGs to the evaluator."""
    path = Path(path).resolve()
    scenes_path = path.parent / "Scenes.json"
    labels_sha, scene_sha = sha(path), sha(scenes_path)
    scenes, pages, isolation = native.load_scenes(scenes_path)
    protocol_path, protocol = native.capture_protocol(path.parent, scene_sha)
    protocol_sha = sha(protocol_path)
    registered = {font["id"]: font for font in scenes.get("fonts", [])}
    records, seen, source_counts, pixel_labels = [], set(), Counter(), {}
    with path.open(encoding="utf-8") as incoming:
        for line in incoming:
            if not line.strip():
                continue
            row = json.loads(line)
            page, image_path, image, requests = native.native_source(row, scene_sha, pages, path.parent)
            require(row["page_id"] not in seen, "duplicate captured page")
            require(row.get("simulator_id") == protocol.get("simulator_id") and row.get("bundle_id") == protocol.get("bundle_id"), "capture device/app differs")
            seen.add(row["page_id"])
            source_counts[row["split"]] += 1
            pixel_sha = native.pixels_sha(image)
            for kind, value in {"source_id": row["source_id"], "source_file_sha256": row["source_sha256"],
                                "decoded_pixel_sha256": pixel_sha, "page_id": row["page_id"],
                                "content_group_id": page.get("content_group_id", page["id"])}.items():
                isolation.bind(kind, value, row["split"])
            truths = []
            for region in row["regions"]:
                reason = native.font_evidence(region, region["font_family"], region["script"])
                reason = reason or native.asset_evidence(region, requests[region["id"]], registered)
                require(reason is None, f"native truth not verified: {row['page_id']}/{region['id']}: {reason}")
                require(native.bounds(region.get("bbox"), image.size), "invalid native region box")
                ink = region.get("ink_bbox_pixels")
                require(valid_box(ink) and native.contained(ink, region["bbox"]), "missing/outside native line ink bbox")
                style = native.native_style(region, row["native"])
                region_sha = native.pixels_sha(image.crop(region["bbox"]))
                isolation.bind("region_rgb_sha256", region_sha, row["split"])
                isolation.bind("normalized_region_text", native.normalized_text(region["text"]), row["split"])
                labels = (region["font_family"], native.normalized_text(region["text"]))
                require(pixel_labels.setdefault((region_sha, region["script"]), labels) == labels,
                        "identical region pixels have conflicting labels")
                truths.append({key: region[key] for key in ("id", "text", "script", "font_family", "font_postscript", "ink_bbox_pixels")} | {
                    "font_truth_verified": True, "region_rgb_sha256": region_sha, **style})
            if row["split"] == "test":
                records.append({"source_id": row["source_id"], "page_id": row["page_id"], "image": str(image_path),
                                "source_sha256": row["source_sha256"], "decoded_pixel_sha256": pixel_sha,
                                "pixel_size": row["pixel_size"], "content_group_id": page.get("content_group_id", page["id"]),
                                "native_frame_sha256": digest(row["native"]), "regions": truths})
            image.close()
            if len(seen) % 100 == 0:
                print(json.dumps({"audited_captures": len(seen), "test_pages": len(records)}), flush=True)
    expected_test = {page["id"] for page in pages.values() if page["split"] == "test"}
    require(expected_test and {record["page_id"] for record in records} == expected_test, "all predetermined test captures are required")
    require(set(pages) == seen, "all planned captures are required for train/calibration/test source isolation")
    records.sort(key=lambda row: row["page_id"])
    for filename, expected in ((path, labels_sha), (scenes_path, scene_sha), (protocol_path, protocol_sha)):
        pinned(filename, expected)
    protocol = {"schema": SCHEMA, "rules": RULES, "captures": {"path": str(path), "sha256": labels_sha},
                "scenes": {"path": str(scenes_path), "sha256": scene_sha},
                "capture_protocol": {"path": str(protocol_path), "sha256": protocol_sha},
                "source_kind": "ios_simulator_screenshot", "ui_content_is_generated": True,
                "independent_third_party_app_accuracy": False, "source_split_counts": dict(source_counts),
                "test_pages": len(records), "test_regions": sum(len(row["regions"]) for row in records),
                "split_isolation": {key: 0 for key in ("source_id", "source_file_sha256", "decoded_pixel_sha256",
                                      "page_id", "content_group_id", "region_rgb_sha256", "normalized_region_text")},
                "test_source_catalogue_sha256": digest(records)}
    return records, protocol


def pinned(path, expected):
    require(sha(path) == expected, "pinned file changed: " + str(path))


def runtime_hashes():
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT / "src/flux_glyph").glob("*.py"))}


def verify_cache(cache, run_identity, source, output):
    require(cache.get("schema") == SCHEMA and cache.get("run_identity") == run_identity and
            cache.get("source_sha256") == source["source_sha256"], "cached inference provenance differs")
    path = (output / cache["result_path"]).resolve()
    require(path.is_relative_to(output.resolve()), "cached result escaped output")
    pinned(path, cache["result_sha256"])
    result = read_json(path)
    require(result.get("source_sha256") == source["source_sha256"], "cached result names a different source")
    return result


def evaluate_method(method, models, sources, protocol, output):
    pipeline = FontPipeline(models)
    require((pipeline.neural is not None) == (method == "neural"), "requested method does not match model bundle")
    provenance = {"method": method, "model_directory": str(pipeline.directory), "model_version": pipeline.version,
                  "model_manifest_sha256": sha(pipeline.directory / "MANIFEST.json"),
                  "runtime_source_sha256": runtime_hashes(), "scoring_source_sha256": sha(Path(__file__)),
                  "native_audit_source_sha256": sha(Path(native.__file__)), "evaluation_protocol_sha256": digest(protocol)}
    identity = digest(provenance)
    method_dir = output / method
    provenance_path = method_dir / "RUN.json"
    if provenance_path.exists():
        require(read_json(provenance_path) == provenance, "existing method run differs; use a new output directory")
    else:
        write_json(provenance_path, provenance)
    scored = []
    for index, source in enumerate(sources):
        case = hashlib.sha256(source["source_id"].encode()).hexdigest()[:20]
        cache_path = method_dir / "cache" / (case + ".json")
        pinned(Path(source["image"]), source["source_sha256"])
        if cache_path.exists():
            result = verify_cache(read_json(cache_path), identity, source, output)
            cached = True
        else:
            run_dir = method_dir / "runs" / case
            # Only the PNG, output location and opaque ID reach inference.
            result = pipeline.run(Path(source["image"]), run_dir, case)
            require(result.get("source_sha256") == source["source_sha256"], "inference image changed")
            result_path = run_dir / "result.json"
            write_json(cache_path, {"schema": SCHEMA, "run_identity": identity,
                                   "source_sha256": source["source_sha256"],
                                   "result_path": str(result_path.relative_to(output)), "result_sha256": sha(result_path)})
            cached = False
        scored.append(score_page(result, source))
        print(json.dumps({"method": method, "completed_pages": index + 1, "total_pages": len(sources),
                          "cached": cached, "source_id": source["source_id"]}, ensure_ascii=False), flush=True)
    verify_bundle(pipeline.directory)
    pinned(pipeline.directory / "MANIFEST.json", provenance["model_manifest_sha256"])
    require(runtime_hashes() == provenance["runtime_source_sha256"], "runtime changed during inference")
    pinned(Path(__file__), provenance["scoring_source_sha256"])
    pinned(Path(native.__file__), provenance["native_audit_source_sha256"])
    report = {"schema": SCHEMA, "method": method, "provenance": provenance,
              "protocol": protocol, "metrics": aggregate(scored), "pages": scored,
              "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(method_dir / "report.json", report)
    return report


def checked_method_report(method, sources, protocol, output):
    """Re-score verified cached inference when merging two separate invocations."""
    method_dir = output / method
    report = read_json(method_dir / "report.json")
    provenance = read_json(method_dir / "RUN.json")
    require(report.get("protocol") == protocol and report.get("provenance") == provenance,
            "completed method has different evaluation provenance")
    require(provenance["scoring_source_sha256"] == sha(Path(__file__)) and
            provenance["native_audit_source_sha256"] == sha(Path(native.__file__)),
            "completed method used different scoring or source auditing code")
    identity = digest(provenance)
    pages = []
    for source in sources:
        case = hashlib.sha256(source["source_id"].encode()).hexdigest()[:20]
        cache = read_json(method_dir / "cache" / (case + ".json"))
        result = verify_cache(cache, identity, source, output)
        pages.append(score_page(result, source))
    require(report["pages"] == pages and report["metrics"] == aggregate(pages),
            "completed report differs from verified cached inference")
    return report


def markdown(report):
    lines = ["# iOS 原生截图端到端评测", "",
             "实际 iOS Simulator 截图，内容由受控原生采集 App 生成；不代表第三方 App 截图准确率。",
             "仅 test 分区；原生标签只用于来源审计和推理后评分。检测框与原生墨迹框按 IoU ≥ 0.5 全局一对一配对，先最大化配对数，再最大化总 IoU，配对不使用文字或字体名。", "",
             "| 方法 | 真值行 | 配对 | 漏检 | 多余框 | OCR 正确 | 字体名正确 | 字体名错误 | 检测后弃权 | OCR+字体正确 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, method in report["methods"].items():
        row = method["metrics"]["overall"]
        lines.append(f"| {name} | {row['truth_regions']} | {row['detected']} | {row['missed']} | {row['false_positives']} | {row['ocr_normalized_exact']} | {row['font_name_correct']} | {row['wrong_font_names']} | {row['font_abstained_after_detection']} | {row['strict_e2e_correct']} |")
    lines += ["", "OCR 正确忽略空白，保留大小写和标点；JSON 同时保存完全一致与字符编辑距离。漏检单列，不混为字体弃权；多余框（包括重复检测）全部计假阳性。字体名评分与 OCR 是否正确独立，联合正确同时要求两者。", "",
              "字号只比较源图像像素 em size；颜色比较实际可见 RGB（源 RGBA 在背景上合成），不声称恢复透明度。固定容差为字号相对误差 ≤10%，颜色最大通道误差 ≤8；缺失估计单列。", "",
              "| 方法 | 可估字号 | 字号在容差内 | 字号 MAE px | 可估颜色 | 颜色在容差内 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, method in report["methods"].items():
        style = method["metrics"]["overall"]["style"]
        mae = f"{style['size_mae_px']:.3f}" if style["size_mae_px"] is not None else "—"
        lines.append(f"| {name} | {style['size_available']} | {style['size_within_tolerance']} | {mae} | {style['color_available']} | {style['color_within_tolerance']} |")
    lines += ["", "这些受控截图的字体均有原生来源验证；本数据没有未知字体负例，不能据此推断未知字体拒识能力。未进行测试集门槛调整。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "models")
    parser.add_argument("--neural", type=Path)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", choices=("baseline", "neural", "both"), default="both")
    parser.add_argument("--validate-only", action="store_true", help="Audit all source splits without running inference")
    args = parser.parse_args()
    sources, protocol = checked_captures(args.captures)
    output = args.output.resolve()
    protocol_path = output / "PROTOCOL.json"
    if protocol_path.exists():
        require(read_json(protocol_path) == protocol, "capture set or scoring protocol changed; use a new output directory")
    else:
        write_json(protocol_path, protocol)
    if args.validate_only:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return
    if args.method in ("both", "neural"):
        require(args.neural is not None, "--neural is required for neural inference")
    for method in ("baseline", "neural") if args.method == "both" else (args.method,):
        evaluate_method(method, getattr(args, method), sources, protocol, output)
    for name in ("captures", "scenes", "capture_protocol"):
        pinned(Path(protocol[name]["path"]), protocol[name]["sha256"])
    methods = {}
    for method in ("baseline", "neural"):
        path = output / method / "report.json"
        if path.exists():
            methods[method] = checked_method_report(method, sources, protocol, output)
    comparison = paired_comparison(methods["baseline"]["pages"], methods["neural"]["pages"]) if len(methods) == 2 else None
    if comparison is not None:
        comparison["identical_runtime_sources"] = (methods["baseline"]["provenance"]["runtime_source_sha256"] ==
                                                    methods["neural"]["provenance"]["runtime_source_sha256"])
    report = {"schema": SCHEMA, "protocol": protocol, "methods": methods, "paired_comparison": comparison}
    write_json(output / "report.json", report)
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(output), "methods": {name: item["metrics"]["overall"] for name, item in methods.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
