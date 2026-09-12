#!/usr/bin/env python3
"""Evaluate detector -> line-image CNN -> style, with OCR actively prohibited.

The fixed 100-page native test partition is declared fresh or reused explicitly.
Its labels are used only for source auditing and geometry-paired scoring;
inference receives only original PNGs, never text, scripts or native boxes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import evaluate_ios_screenshots as shared
from flux_glyph import pipeline as pipeline_module, ppocr, segmentation
from flux_glyph.models import verify_bundle

SCHEMA = "flux-glyph-no-ocr-region-evaluation-v1"
METHOD = "region_neural_network"
RULES = {"split": "test", "test_pages": 100, "source_pages": 1000,
         "matching": "maximum_cardinality_then_total_iou_one_to_one", "minimum_iou": .5,
         "truth_bbox": "native.ink_bbox_pixels", "prediction_bbox": "detector_bbox",
         "pairing_uses_text_or_font": False, "inference_inputs": ["source_png"],
         "ocr_performed": False, "glyph_segmentation_performed": False,
         "font_accepted_statuses": ["candidate", "supported"],
         "size_relative_error_tolerance": .1, "color_max_channel_error_tolerance": 8,
         "missing_detection_is_font_abstention": False,
         "unknown_font_rejection_validated": False, "mixed_font_region_handling_validated": False,
         "native_truth_scope": "single-font regions in generated native iOS Simulator scenes"}
PINGFANG_GROUP = {"PingFang": ["PingFang SC", "PingFang TC", "PingFang HK"]}

require, sha, digest = shared.require, shared.sha, shared.digest
read_json, write_json, pinned = shared.read_json, shared.write_json, shared.pinned
pair_regions, score_style = shared.pair_regions, shared.score_style


def validated_label_groups(metadata):
    """Only a model's explicit, unambiguous PingFang family group is legal.

    Grouping does not assert native SC/TC/HK identity. No template/old SC-only
    model obtains TC/HK credit just because these names share a prefix.
    """
    families = metadata.get("families")
    require(isinstance(families, list) and families and all(isinstance(f, str) and f for f in families)
            and len(families) == len(set(families)), "invalid model family registry for scoring")
    groups = metadata.get("font_label_groups", {})
    require(isinstance(groups, dict) and (groups == {} or groups == PINGFANG_GROUP),
            "only the exact verified PingFang SC/TC/HK label group is allowed")
    if groups:
        require("PingFang" in families and not any(name in families for name in PINGFANG_GROUP["PingFang"]),
                "grouped PingFang output registry is missing or overlaps native subfamilies")
    else:
        require("PingFang" not in families, "PingFang output requires an explicit native label group")
    return {key: list(values) for key, values in groups.items()}


def reporting_orthography(truth):
    """Use native source truth only; these categories never enter inference."""
    if truth.get("script") == "han":
        declared = truth.get("han_orthography")
        return declared if declared in ("simplified", "traditional") else "han_unspecified"
    text = truth.get("text", "")
    if truth.get("script") == "latin":
        if any(character.isascii() and character.isalpha() for character in text):
            return "english"
        if any(character.isascii() and character.isdigit() for character in text):
            return "numeric"
    return "unspecified"


def scoring_label(native_family, groups):
    return next((label for label, members in groups.items() if native_family in members), native_family)


def bind_orthography(sources, scenes):
    """Bind classification labels to already audited, SHA-pinned requests."""
    pages = {page["id"]: page for page in scenes["pages"]}
    for source in sources:
        require(source["page_id"] in pages, "evaluated page is absent from pinned scenes")
        requests = {region["id"]: region for region in pages[source["page_id"]]["regions"]}
        for truth in source["regions"]:
            request = requests.get(truth["id"])
            require(request is not None and request.get("text") == truth["text"]
                    and request.get("script") == truth["script"] and request.get("font_family") == truth["font_family"],
                    "scoring orthography request differs from verified native truth")
            orthography, language = request.get("han_orthography"), request.get("language")
            require(orthography in (None, "simplified", "traditional"), "invalid native orthography declaration")
            if truth["script"] == "han":
                if orthography == "simplified":
                    require(language == "zh-Hans", "simplified source language differs")
                if orthography == "traditional":
                    require(language in ("zh-Hant", "zh-Hant-TW", "zh-Hant-HK"), "traditional source language differs")
            else:
                require(orthography is None and language in (None, "en"), "Latin source incorrectly declares Han orthography")
            truth["han_orthography"], truth["source_language"] = orthography, language
    return sources


@contextmanager
def prohibit_ocr():
    """Fail closed even if a pipeline accidentally instantiates a legacy path."""
    state = {"ocr_calls": 0, "segmentation_calls": 0, "glyph_classifier_calls": 0,
             "pp_detector_sessions": 0, "guard_installed": True}

    def denied(kind):
        def fail(*args, **kwargs):
            state[kind] += 1
            raise RuntimeError("No-OCR evaluation forbids " + kind)
        return fail

    original_session = ppocr.session

    def detection_only(directory, kind, contract):
        if kind != "det":
            return denied("ocr_calls")()
        state["pp_detector_sessions"] += 1
        return original_session(directory, kind, contract)

    with ExitStack() as stack:
        stack.enter_context(patch.object(ppocr.PPReader, "__init__", denied("ocr_calls")))
        stack.enter_context(patch.object(ppocr.PPReader, "read", denied("ocr_calls")))
        stack.enter_context(patch.object(ppocr, "decode_ctc", denied("ocr_calls")))
        stack.enter_context(patch.object(ppocr, "session", detection_only))
        stack.enter_context(patch.object(segmentation, "segment_characters", denied("segmentation_calls")))
        if hasattr(pipeline_module, "segment_characters"):
            stack.enter_context(patch.object(pipeline_module, "segment_characters", denied("segmentation_calls")))
        # Class-object patches also affect aliases imported before this guard.
        for name in ("NeuralFontClassifier", "CompactFontBank", "CompactLatinBank"):
            cls = getattr(pipeline_module, name, None)
            if cls is not None:
                stack.enter_context(patch.object(cls, "__init__", denied("glyph_classifier_calls")))
        yield state
        require(all(state[key] == 0 for key in ("ocr_calls", "segmentation_calls", "glyph_classifier_calls")),
                "a forbidden inference path was attempted")


def validate_prediction(result):
    require(result.get("font_method") == METHOD and result.get("ocr_performed") is False,
            "result is not an explicit no-OCR region-CNN run")
    require(isinstance(result.get("regions"), list), "invalid pipeline regions")
    for region in result["regions"]:
        require(isinstance(region, dict) and "text" in region and region["text"] is None
                and region.get("glyphs") == [] and region.get("ocr_performed") is False,
                "region contains OCR text or segmented glyphs")
        require(not any(region.get(key) for key in ("ocr", "segmentation", "tokens", "script")),
                "region exposes text/script inference evidence")
        require(isinstance(region.get("font"), dict), "missing region font result")
        require(region["font"].get("method") == METHOD, "non-region font classifier result")


def score_page(result, source, *, model_metadata=None):
    validate_prediction(result)
    groups = validated_label_groups(model_metadata) if model_metadata is not None else {}
    truths, predictions = source["regions"], result["regions"]
    pairs = {truth: (pred, overlap) for truth, pred, overlap in pair_regions(truths, predictions)}
    used = {item[0] for item in pairs.values()}
    rows = []
    for index, truth in enumerate(truths):
        found = index in pairs
        pred_index, overlap = pairs.get(index, (None, None))
        prediction = predictions[pred_index] if found else {}
        accepted, font = shared.accepted_font(prediction)
        verified = truth.get("font_truth_verified") is True
        expected_family = scoring_label(truth["font_family"], groups) if verified else truth["font_family"]
        correct = bool(found and verified and accepted and font["family"] == expected_family)
        wrong = bool(found and verified and accepted and font["family"] != expected_family)
        rows.append({"source_id": source["source_id"], "page_id": source["page_id"], "region_id": truth["id"],
                     "truth_script_for_reporting_only": truth["script"], "expected_family": expected_family,
                     "native_family": truth["font_family"], "native_font_postscript": truth.get("font_postscript"),
                     "orthography": reporting_orthography(truth), "source_language": truth.get("source_language"),
                     "font_label_group_applied": expected_family != truth["font_family"],
                     "truth_bbox": truth["ink_bbox_pixels"], "detected": found, "prediction_index": pred_index,
                     "prediction_id": prediction.get("id"), "iou": overlap, "font_truth_verified": verified,
                     "accepted": bool(found and accepted), "predicted_family": font.get("family"),
                     "font_status": font.get("status"), "font_name_correct": correct, "wrong_font_name": wrong,
                     "font_abstained": bool(found and verified and not accepted), "font_unscorable": bool(found and not verified),
                     "font_reason": font.get("reason_code", font.get("reason")),
                     "style": score_style(prediction.get("text_style"), truth)})
    extras = []
    for index, pred in enumerate(predictions):
        if index not in used:
            accepted, font = shared.accepted_font(pred)
            extras.append({"prediction_index": index, "prediction_id": pred.get("id"),
                           "detector_bbox": pred.get("detector_bbox"), "accepted_font": accepted,
                           "family": font.get("family"), "font_status": font.get("status")})
    return {"source_id": source["source_id"], "page_id": source["page_id"], "truth_count": len(truths),
            "prediction_count": len(predictions), "rows": rows, "false_positives": extras,
            "timing_seconds": result.get("timing_seconds", {})}


def summarize(rows, extras=(), prediction_count=None):
    ratio = shared.ratio
    n = len(rows)
    detected, accepted, correct, wrong, abstained = (sum(row[key] for row in rows) for key in
        ("detected", "accepted", "font_name_correct", "wrong_font_name", "font_abstained"))
    styles = [row["style"] for row in rows]
    sizes = [style for style in styles if style["size_available"]]
    colors = [style for style in styles if style["color_available"]]
    mean = lambda values: sum(values) / len(values) if values else None
    result = {"truth_regions": n, "detected": detected, "missed": n - detected,
              "detection_recall": ratio(detected, n), "font_accepted": accepted, "font_name_correct": correct,
              "wrong_font_names": wrong, "font_abstained_after_detection": abstained,
              "font_accuracy_all_truth": ratio(correct, n), "font_accuracy_detected": ratio(correct, detected),
              "font_precision_among_accepted": ratio(correct, accepted), "font_coverage_detected": ratio(accepted, detected),
              "font_coverage_all_truth": ratio(accepted, n), "wrong_font_rate_all_truth": ratio(wrong, n),
              "pending_after_detection": abstained, "pending_rate_detected": ratio(abstained, detected),
              "style": {"size_available": len(sizes), "size_coverage_all_truth": ratio(len(sizes), n),
                        "size_within_tolerance": sum(s["size_within_tolerance"] for s in sizes),
                        "size_mae_px": mean([s["size_absolute_error_px"] for s in sizes]),
                        "size_mean_relative_error": mean([s["size_relative_error"] for s in sizes]),
                        "size_intervals_available": sum(s["size_interval_available"] for s in sizes),
                        "size_intervals_cover_native": sum(s["size_interval_covers_native"] for s in sizes),
                        "color_available": len(colors), "color_coverage_all_truth": ratio(len(colors), n),
                        "color_within_tolerance": sum(s["color_within_tolerance"] for s in colors),
                        "color_mean_max_channel_error": mean([s["color_max_channel_error"] for s in colors])}}
    if prediction_count is not None:
        result.update(predictions=prediction_count, false_positives=len(extras),
                      false_positive_named_fonts=sum(row["accepted_font"] for row in extras),
                      detection_precision=ratio(detected, prediction_count))
    return result


def aggregate(pages):
    rows = [row for page in pages for row in page["rows"]]
    extras = [row for page in pages for row in page["false_positives"]]
    result = {"pages": len(pages), "overall": summarize(rows, extras, sum(p["prediction_count"] for p in pages))}
    for key in ("expected_family", "native_family", "orthography", "truth_script_for_reporting_only"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        result["by_" + key] = {name: summarize(items) for name, items in sorted(groups.items())}
    for name in ("simplified", "traditional", "english", "numeric"):
        result["by_orthography"].setdefault(name, summarize([]))
    return result


def checked_sources(captures, *, test_history):
    require(test_history in ("fresh", "reused"), "test history must be explicitly fresh or reused")
    sources, audited = shared.checked_captures(captures)
    require(audited["test_pages"] == RULES["test_pages"] and sum(audited["source_split_counts"].values()) == RULES["source_pages"],
            "evaluation requires the complete preassigned 1000-page / 100-test collection")
    scenes_path = audited["scenes"]["path"]
    pinned(scenes_path, audited["scenes"]["sha256"])
    sources = bind_orthography(sources, read_json(scenes_path))
    pinned(scenes_path, audited["scenes"]["sha256"])
    rules = {**RULES, "test_partition_reused_from_previous_pipeline_evaluations": test_history == "reused"}
    protocol = {**audited, "schema": SCHEMA, "rules": rules, "test_history": test_history,
                "test_history_declared_by_caller": True, "parent_training_domain_previously_seen": True,
                "orthography_source": "SHA-pinned Scenes request; English/numeric from native truth only",
                "test_source_catalogue_sha256": digest(sources),
                "scope": ("Fresh fixed native capture partition declared by caller; parent model knows this native rendering domain"
                          if test_history == "fresh" else "Reused fixed native capture test partition"),
                "physical_device_accuracy_established": False}
    return sources, protocol


def source_hashes():
    return {**shared.runtime_hashes(), str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
            str(Path(shared.__file__).relative_to(ROOT)): sha(Path(shared.__file__)),
            str(Path(shared.native.__file__).relative_to(ROOT)): sha(Path(shared.native.__file__))}


def evaluate(models, captures, output, *, test_history):
    sources, protocol = checked_sources(captures, test_history=test_history)
    output = Path(output).resolve()
    protocol_path = output / "PROTOCOL.json"
    if protocol_path.exists():
        require(read_json(protocol_path) == protocol, "fixed capture/scoring protocol changed")
    else:
        write_json(protocol_path, protocol)
    with prohibit_ocr() as guard:
        pipeline = pipeline_module.FontPipeline(models)
        require(getattr(pipeline, "region_neural", None) is not None
                and all(getattr(pipeline, key, None) is None for key in ("reader", "neural", "bank", "latin_bank")),
                "pipeline did not select the isolated region-CNN path")
        model_metadata = pipeline.region_neural.meta
        label_groups = validated_label_groups(model_metadata)
        # No training truth or native region reaches the constructor or run.
        provenance = {"method": METHOD, "model_directory": str(pipeline.directory), "model_version": pipeline.version,
                      "model_manifest_sha256": sha(pipeline.directory / "MANIFEST.json"), "source_sha256": source_hashes(),
                      "font_label_groups": label_groups, "model_families": model_metadata["families"],
                      "evaluation_protocol_sha256": digest(protocol), "ocr_guard_policy": "deny_reader_ctc_segmentation_glyph_classifiers_v1"}
        identity = digest(provenance)
        path = output / "RUN.json"
        if path.exists():
            require(read_json(path) == provenance, "existing evaluation differs; use new output")
        else:
            write_json(path, provenance)
        pages = []
        for index, source in enumerate(sources):
            case = hashlib.sha256(source["source_id"].encode()).hexdigest()[:20]
            cache_path = output / "cache" / (case + ".json")
            pinned(source["image"], source["source_sha256"])
            if cache_path.exists():
                cache = read_json(cache_path)
                require(cache.get("schema") == SCHEMA and cache.get("run_identity") == identity
                        and cache.get("source_sha256") == source["source_sha256"]
                        and cache.get("ocr_guard_passed") is True, "cached no-OCR provenance differs")
                result_path = (output / cache["result_path"]).resolve()
                require(result_path.is_relative_to(output), "cached inference escaped output")
                pinned(result_path, cache["result_sha256"])
                result = read_json(result_path)
            else:
                run_dir = output / "runs" / case
                result = pipeline.run(Path(source["image"]), run_dir, case)
                validate_prediction(result)
                require(all(guard[key] == 0 for key in ("ocr_calls", "segmentation_calls", "glyph_classifier_calls")), "forbidden path attempted")
                result_path = run_dir / "result.json"
                write_json(cache_path, {"schema": SCHEMA, "run_identity": identity, "source_sha256": source["source_sha256"],
                                       "result_path": str(result_path.relative_to(output)), "result_sha256": sha(result_path),
                                       "ocr_guard_passed": True})
            require(result.get("source_sha256") == source["source_sha256"], "inference source changed")
            pages.append(score_page(result, source, model_metadata=model_metadata))
            print(shared.json.dumps({"completed_pages": index + 1, "total_pages": len(sources), "ocr_performed": False}), flush=True)
        verify_bundle(pipeline.directory)
        pinned(pipeline.directory / "MANIFEST.json", provenance["model_manifest_sha256"])
        require(source_hashes() == provenance["source_sha256"], "runtime or evaluator changed during inference")
        for name in ("captures", "scenes", "capture_protocol"):
            pinned(protocol[name]["path"], protocol[name]["sha256"])
        report = {"schema": SCHEMA, "method": METHOD, "protocol": protocol, "provenance": provenance,
                  "ocr_performed": False, "guard": dict(guard), "metrics": aggregate(pages), "pages": pages,
                  "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "report.json", report)
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    return report


def markdown(report):
    row = report["metrics"]["overall"]
    style = row["style"]
    history = report["protocol"].get("test_history")
    history_text = ("本次明确声明为新采集的固定 test 分区；父模型已接触同类原生渲染域，不能称为未知域盲测。"
                    if history == "fresh" else "这 100 张 test 为之前实验复用的固定分区，不能称为全新盲测。")
    lines = ["# 无 OCR 字体与样式评测", "",
        "完整原始截图经文字区域检测、整行图像字体网络和样式估计；未调用文字识读、字符分割或旧单字分类器。真值只用于推理后的几何配对与评分。", "",
        history_text + "内容是受控 iOS Simulator 原生页面，不代表第三方 App、物理设备、未知字体或混合字体行的准确率。", "",
        "| 真值行 | 检出 | 漏检 | 多余框 | 字体正确 | 字体错误 | 检出后弃权 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {row['truth_regions']} | {row['detected']} | {row['missed']} | {row['false_positives']} | {row['font_name_correct']} | {row['wrong_font_names']} | {row['font_abstained_after_detection']} |", "",
        f"字号可估 {style['size_available']} 行，误差 ≤10% 有 {style['size_within_tolerance']} 行，MAE {style['size_mae_px']} px。",
        f"颜色可估 {style['color_available']} 行，RGB 最大通道误差 ≤8 有 {style['color_within_tolerance']} 行。", "",
        "字号单位为源截图像素 em，不是 iOS pt；颜色评分比较合成后的可见 RGB，不恢复透明度。字体、字号、颜色分别评分，漏检与字体弃权分开。未用本测试结果调整门槛。", ""]
    if report["provenance"].get("font_label_groups"):
        lines += ["模型明确声明 PingFang 字体族包含原生 PingFang SC／TC／HK，故按字体族评分；原生字体名另行保留。这不等于已识别出 SC、TC 或 HK 子类。没有该声明的 SC-only 模型不会把 TC／HK 计作正确。", ""]
    for key, title in (("orthography", "文字类型"), ("native_family", "原生字体")):
        lines += [f"| {title} | 真值行 | 字体正确 | 字体错误 | 检出后待确认 | 接受覆盖率 |", "|---|---:|---:|---:|---:|---:|"]
        for name, group in report["metrics"].get("by_" + key, {}).items():
            coverage = group["font_coverage_all_truth"]
            display = f"{coverage:.2%}" if coverage is not None else "—"
            lines.append(f"| {name} | {group['truth_regions']} | {group['font_name_correct']} | {group['wrong_font_names']} | {group['pending_after_detection']} | {display} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-history", choices=("fresh", "reused"), required=True,
                        help="Explicitly declare whether this fixed capture test partition was evaluated before")
    args = parser.parse_args()
    report = evaluate(args.models, args.captures, args.output, test_history=args.test_history)
    print(shared.json.dumps(report["metrics"]["overall"], ensure_ascii=False, indent=2))
