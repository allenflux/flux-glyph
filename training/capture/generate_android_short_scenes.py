#!/usr/bin/env python3
"""Generate paired Android-native TRAIN requests for one-to-four glyph text."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_android import dump, read, require, sha, validate_scenes


NAMESPACE = "android-native-short-train-v2"
SEED = 2026091403
SAMPLES_PER_SCRIPT_LENGTH = 8
MAX_FACES_PER_SOURCE_FAMILY = 8
KNOWN_FAMILIES = (
    "Noto Sans CJK SC", "Noto Serif CJK SC", "LXGW WenKai",
    "WenQuanYi Micro Hei", "ZCOOL KuaiLe", "ZCOOL XiaoWei",
    "ZCOOL QingKe HuangYou", "Ma Shan Zheng", "Roboto",
)
# These are the only TRAIN source families in sources-v2 which remain unknown
# under prepare_unified_regions.family_label.
UNKNOWN_SOURCE_FAMILIES = (
    "Liu Jian Mao Cao", "Smiley Sans", "Lato", "Open Sans",
)
PROMOTED_UNIFIED_FAMILIES = (
    "FandolHei", "FandolKai", "FandolSong", "Long Cang", "Zhi Mang Xing",
)
EXPLICITLY_EXCLUDED_FAMILIES = frozenset({
    "M PLUS 1p", "Yusei Magic", "Ubuntu", "IBM Plex Sans SC",
    "Yuji Syuku", "Liberation Sans",
})
HAN_EXTENSION = (
    "山水风云日月星光春夏秋冬花草树木书画诗词城市乡村桥路湖海"
    "家门衣食学友明暗高低长短新旧左右东西南北"
)


def normalized_text(value):
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def text_sha(value):
    return hashlib.sha256(normalized_text(value).encode()).hexdigest()


def heldout_hashes(root):
    """Read only normalized full-text hashes from reused CAL/DEV row metadata."""
    root = Path(root)
    forbidden, bindings, counts = set(), {}, {}
    for split in ("calibration", "development_holdout"):
        path = root / split / "rows.json"
        rows = read(path)
        require(isinstance(rows, list) and rows, "expected existing CAL/DEV row metadata")
        for row in rows:
            digest = row.get("normalized_text_sha256")
            require(isinstance(digest, str) and len(digest) == 64,
                    "missing normalized full-text hash")
            forbidden.add(digest)
        bindings[str(path.resolve())] = sha(path)
        counts[split] = len(rows)
    return forbidden, bindings, counts


def select_fonts(registry):
    require(registry.get("schema") == "flux-glyph-android-font-sources-v1",
            "source schema differs")
    require(tuple(registry.get("families", ())) == KNOWN_FAMILIES,
            "original nine-family order differs")
    eligible = []
    for font in registry.get("fonts", ()):
        family, target = font.get("family"), font.get("training_family")
        if family in KNOWN_FAMILIES:
            require(target == family and "train" in font.get("allowed_splits", ()),
                    "known source is not an actual TRAIN source")
            eligible.append(font)
        elif family in (*PROMOTED_UNIFIED_FAMILIES, *UNKNOWN_SOURCE_FAMILIES):
            require(target == "__unknown__" and font.get("split") == "train"
                    and font.get("allowed_splits") == ["train"],
                    "promoted/unknown source is not TRAIN-only")
            eligible.append(font)
    grouped = defaultdict(list)
    for font in eligible:
        grouped[font["family"]].append(font)
    selected = []
    for family in (*KNOWN_FAMILIES, *PROMOTED_UNIFIED_FAMILIES, *UNKNOWN_SOURCE_FAMILIES):
        selected.extend(sorted(grouped[family], key=lambda font: font["id"])
                        [:MAX_FACES_PER_SOURCE_FAMILY])
    actual = {font["family"] for font in selected}
    require(actual == set(KNOWN_FAMILIES) | set(PROMOTED_UNIFIED_FAMILIES)
            | set(UNKNOWN_SOURCE_FAMILIES),
            "required Android TRAIN source family is absent")
    require(not actual.intersection(EXPLICITLY_EXCLUDED_FAMILIES),
            "excluded or promoted source selected")
    return selected


def load_sources(path):
    path = Path(path)
    registry = read(path)
    fonts = select_fonts(registry)
    bindings = {str(path.resolve()): sha(path), str(Path(__file__).resolve()): sha(__file__)}
    maps = {}
    for font in fonts:
        require(Path(font["path"]).is_file() and sha(font["path"]) == font["sha256"],
                "source font changed")
        bindings[str(Path(font["path"]).resolve())] = font["sha256"]
        cmap = font.get("cmap", {})
        require(Path(cmap.get("path", "")).is_file() and sha(cmap["path"]) == cmap.get("sha256"),
                "source cmap changed")
        maps[font["id"]] = set(read(cmap["path"]))
        bindings[str(Path(cmap["path"]).resolve())] = cmap["sha256"]
        licenses = font.get("license_files", ())
        require(bool(licenses), "font license binding missing")
        for license_file in licenses:
            require(Path(license_file["path"]).is_file()
                    and sha(license_file["path"]) == license_file["sha256"],
                    "font license changed")
            bindings[str(Path(license_file["path"]).resolve())] = license_file["sha256"]
    return registry, fonts, maps, bindings


def _supports(text, fonts, maps):
    return all(all(ord(character) in maps[font["id"]] for character in text) for font in fonts)


def _rgb(value):
    require(isinstance(value, str) and len(value) in (7, 9) and value.startswith("#"),
            "invalid native color proof")
    return value[:7].upper()


def _box(value, size):
    return (isinstance(value, list) and len(value) == 4
            and all(type(coordinate) is int for coordinate in value)
            and 0 <= value[0] < value[2] <= size[0]
            and 0 <= value[1] < value[3] <= size[1])


def load_ios_capture(root, ios_scenes_path, ios_scenes):
    """Verify TRAIN-only native frame/font evidence and return request styles in pixels."""
    root, ios_scenes_path = Path(root), Path(ios_scenes_path)
    labels = root / "labels.jsonl"
    protocol_path = root / "CAPTURE_PROTOCOL.json"
    summary_path = root / "CAPTURE_SUMMARY.json"
    captured_scenes = root / "Scenes.json"
    require(all(path.is_file() for path in (labels, protocol_path, summary_path, captured_scenes)),
            "iOS native capture metadata is incomplete")
    scenes_digest = sha(ios_scenes_path)
    require(sha(captured_scenes) == scenes_digest, "captured iOS scenes differ from request source")
    protocol, summary = read(protocol_path), read(summary_path)
    require(protocol.get("schema") == "ios-native-screen-capture-v1"
            and protocol.get("source_kind") == "ios_simulator_screenshot"
            and protocol.get("scenes_sha256") == scenes_digest
            and protocol.get("planned_pages") == len(ios_scenes["pages"]),
            "iOS capture protocol differs")
    require(summary.get("captured_pages") == len(ios_scenes["pages"])
            and summary.get("split_counts") == {"train": len(ios_scenes["pages"]),
                                                 "calibration": 0, "test": 0}
            and summary.get("labels_sha256") == sha(labels),
            "iOS capture summary differs")
    bindings = {str(path.resolve()): sha(path) for path in
                (labels, protocol_path, summary_path, captured_scenes)}
    requests = {region["id"]: (page, region) for page in ios_scenes["pages"]
                for region in page["regions"]}
    require(len(requests) == sum(len(page["regions"]) for page in ios_scenes["pages"]),
            "duplicate iOS request region")
    actual, pages = {}, set()
    records = [json.loads(line) for line in labels.read_text().splitlines() if line.strip()]
    require(len(records) == len(ios_scenes["pages"]), "iOS capture page count differs")
    for record in records:
        page_id = record.get("page_id")
        require(record.get("schema") == "ios-native-captured-frame-v1"
                and record.get("split") == "train" and page_id not in pages
                and record.get("scenes_sha256") == scenes_digest
                and record.get("source_kind") == "ios_simulator_screenshot",
                "iOS captured frame identity differs")
        pages.add(page_id)
        native = record.get("native", {})
        scale = native.get("source_screen_scale")
        points = ios_scenes["canvas_points"]
        pixels = [coordinate * scale for coordinate in points] if type(scale) is int else []
        require(native.get("schema") == "flux-glyph-ios-frame-v1"
                and native.get("status") == "ready" and native.get("split") == "train"
                and native.get("page_id") == page_id and native.get("source_screen_scale") == 3
                and native.get("screen_scale") == scale and native.get("view_points") == points
                and native.get("window_bounds_points") == [0, 0, *points]
                and native.get("pixel_size") == pixels == record.get("pixel_size") == [1206, 2622]
                and native.get("rejected_region_count") == 0
                and native.get("verified_region_count") == len(record.get("regions", ()))
                and native.get("scene_manifest_sha256") == scenes_digest,
                "iOS native canvas/scale proof differs")
        frame_path = root / "frames" / f"{page_id}.json"
        require(frame_path.is_file() and read(frame_path) == record,
                "iOS native frame proof differs from labels")
        bindings[str(frame_path.resolve())] = sha(frame_path)
        image_path = Path(record.get("image", ""))
        require(image_path.is_file() and sha(image_path) == record.get("source_sha256"),
                "iOS native TRAIN screenshot binding differs")
        bindings[str(image_path.resolve())] = record["source_sha256"]
        for region in record.get("regions", ()):
            request_entry = requests.get(region.get("id"))
            require(request_entry is not None and region["id"] not in actual,
                    "iOS native region is absent or duplicated")
            page, request = request_entry
            size_points, size_px = region.get("font_size_points"), region.get("font_size_screen_px")
            glyphs = region.get("glyphs", ())
            require(page["id"] == page_id and region.get("text") == request["text"]
                    and region.get("script") == request["script"]
                    and region.get("requested_font_family") == request["font_family"]
                    and region.get("requested_font_postscript") == request["font_postscript"]
                    and region.get("requested_font_size_points") == request["font_size"] == size_points
                    and type(size_points) in (int, float) and size_px == size_points * scale
                    and region.get("source_screen_scale") == scale
                    and region.get("bbox_points") == request["bbox_points"]
                    and region.get("bbox_pixels") == [coordinate * scale for coordinate in request["bbox_points"]]
                    and region.get("bbox") == region.get("bbox_pixels")
                    and _box(region.get("bbox"), pixels) and _box(region.get("ink_bbox_pixels"), pixels)
                    and region.get("status") == "ok"
                    and region.get("reason") == "native_coretext_font_and_glyph_coverage_verified"
                    and region.get("font_match_verified") is True
                    and region.get("fallback_detected") is False and region.get("clipped") is False
                    and _rgb(region.get("text_color_hex")) == request["color"].upper()
                    and _rgb(region.get("background_hex")) == page["background"].upper(),
                    "iOS native request/style proof differs")
            coverage = region.get("glyph_coverage", {})
            require(coverage.get("font_get_glyphs_succeeded") is True
                    and coverage.get("zero_run_glyph_count") == 0
                    and coverage.get("utf16_count") == len(region["text"])
                    and len(glyphs) == len(region["text"])
                    and {glyph.get("text_index") for glyph in glyphs} == set(range(len(region["text"]))),
                    "iOS native glyph coverage differs")
            for glyph in glyphs:
                require(glyph.get("character") == region["text"][glyph["text_index"]]
                        and glyph.get("visible") is True
                        and glyph.get("font_match_verified") is True
                        and glyph.get("font_classification_character") is True
                        and type(glyph.get("glyph_id")) is int and glyph["glyph_id"] > 0
                        and glyph.get("font_size_screen_px") == size_px
                        and glyph.get("source_screen_scale") == scale
                        and glyph.get("bbox_screen_px") == glyph.get("bbox")
                        and _box(glyph.get("bbox"), pixels),
                        "iOS native glyph/font proof differs")
            actual[region["id"]] = {
                "font_size_points": size_points, "font_size_screen_px": size_px,
                "source_screen_scale": scale, "actual_font_postscript": region["actual_font_postscript"],
                "bbox_pixels": region["bbox_pixels"], "ink_bbox_pixels": region["ink_bbox_pixels"],
                "source_id": record["source_id"], "source_sha256": record["source_sha256"],
                "frame_path": str(frame_path.resolve()), "frame_sha256": bindings[str(frame_path.resolve())],
            }
    require(set(actual) == set(requests) and len(actual) == 704,
            "iOS native proof does not cover every TRAIN request")
    return actual, bindings, {"pages": len(records), "regions": len(actual),
                              "pixel_size": [1206, 2622], "screen_scale": 3}


def paired_corpus(ios_scenes, ios_capture, fonts, maps, forbidden=()):
    require(ios_scenes.get("schema") == "flux-glyph-capture-scenes-v1"
            and ios_scenes.get("platform") == "ios" and ios_scenes.get("test_read") is False,
            "iOS short TRAIN request contract differs")
    require(ios_scenes.get("short_region_characters") == [1, 2, 3, 4]
            and all(page.get("split") == "train" for page in ios_scenes.get("pages", ())),
            "iOS request source must be TRAIN-only short text")
    han_fonts = [font for font in fonts if "han" in font.get("scripts", ())]
    latin_fonts = [font for font in fonts if "latin" in font.get("scripts", ())]
    require({font["family"] for font in han_fonts} == set(KNOWN_FAMILIES[:-1])
            | set(PROMOTED_UNIFIED_FAMILIES) | {"Liu Jian Mao Cao", "Smiley Sans"},
            "Han source set differs")
    require(len(latin_fonts) == len(fonts), "selected source lacks Latin coverage")

    pools = defaultdict(list)
    seen = set()
    audit = Counter()
    for page in ios_scenes["pages"]:
        for region in page["regions"]:
            text, kind = region.get("text", ""), region.get("text_kind")
            if kind not in ("han", "english", "numeric") or not 1 <= len(text) <= 4:
                continue
            identity = (kind, normalized_text(text))
            if identity in seen:
                continue
            seen.add(identity)
            audit["unique_ios_candidates"] += 1
            if text_sha(text) in forbidden:
                audit["omitted_heldout_hash"] += 1
                continue
            eligible_fonts = han_fonts if kind == "han" else latin_fonts
            if not _supports(text, eligible_fonts, maps):
                audit["omitted_unsupported_cmap"] += 1
                continue
            native = ios_capture.get(region["id"])
            require(native is not None, "iOS request lacks verified native style proof")
            pools[kind, len(text)].append({
                "text": text, "text_kind": kind,
                "font_size_points": native["font_size_points"],
                "font_size_px": native["font_size_screen_px"], "color": region["color"],
                "background": page["background"], "request_origin": "ios_train_corpus",
                "ios_source_page_id": page["id"], "ios_source_region_id": region["id"],
                "ios_capture_source_id": native["source_id"],
                "ios_capture_source_sha256": native["source_sha256"],
                "ios_capture_frame_path": native["frame_path"],
                "ios_capture_frame_sha256": native["frame_sha256"],
                "ios_actual_font_postscript": native["actual_font_postscript"],
                "ios_source_screen_scale": native["source_screen_scale"],
                "ios_actual_bbox_pixels": native["bbox_pixels"],
                "ios_actual_ink_bbox_pixels": native["ink_bbox_pixels"],
            })
            audit["eligible_ios_examples"] += 1

    result = []
    rng = random.Random(SEED)
    all_digests = set(forbidden)
    all_digests.update(text_sha(item["text"]) for values in pools.values() for item in values)
    common_han = "".join(character for character in HAN_EXTENSION
                         if _supports(character, han_fonts, maps))
    require(len(set(common_han)) >= 32, "insufficient shared Han cmap")
    for length in range(1, 5):
        han = list(pools["han", length][:SAMPLES_PER_SCRIPT_LENGTH])
        require(bool(han), f"iOS Han style pool is empty for length {length}")
        style_index = 0
        attempts = 0
        while len(han) < SAMPLES_PER_SCRIPT_LENGTH and attempts < 10000:
            attempts += 1
            text = "".join(rng.choice(common_han) for _ in range(length))
            digest = text_sha(text)
            if digest in all_digests:
                continue
            all_digests.add(digest)
            style = pools["han", length][style_index % len(pools["han", length])]
            style_index += 1
            han.append({**style, "text": text,
                        "request_origin": "independently_authored_android_extension"})
            audit["authored_android_han_examples"] += 1
        require(len(han) == SAMPLES_PER_SCRIPT_LENGTH,
                f"insufficient fresh supported Han text for length {length}")
        result.extend(han)
        for kind in ("english", "numeric"):
            values = pools[kind, length]
            require(len(values) >= SAMPLES_PER_SCRIPT_LENGTH // 2,
                    f"insufficient reusable iOS {kind} text for length {length}")
            result.extend(values[:SAMPLES_PER_SCRIPT_LENGTH // 2])
    require(len(result) == 8 * SAMPLES_PER_SCRIPT_LENGTH, "short corpus size differs")
    audit["selected_reused_ios_examples"] = sum(
        item["request_origin"] == "ios_train_corpus" for item in result)
    audit["eligible_ios_examples_not_selected"] = (
        audit["eligible_ios_examples"] - audit["selected_reused_ios_examples"])
    return result, dict(audit)


def build_scenes(registry, fonts, maps, ios_scenes, ios_capture, forbidden=(), bindings=None,
                 capture_audit=None):
    require(fonts == select_fonts(registry), "selected font registry differs")
    examples, corpus_audit = paired_corpus(ios_scenes, ios_capture, fonts, maps, forbidden)
    bindings = {} if bindings is None else dict(bindings)
    groups = defaultdict(list)
    for font in fonts:
        groups[font["family"]].append(font)
    for values in groups.values():
        values.sort(key=lambda font: font["id"])
    han_families = (list(KNOWN_FAMILIES[:-1]) + list(PROMOTED_UNIFIED_FAMILIES)
                    + ["Liu Jian Mao Cao", "Smiley Sans"])
    latin_families = (list(KNOWN_FAMILIES) + list(PROMOTED_UNIFIED_FAMILIES)
                      + list(UNKNOWN_SOURCE_FAMILIES))
    face_counts = Counter()
    page_counts = Counter()
    pages = []
    for example in examples:
        script = "han" if example["text_kind"] == "han" else "latin"
        families = han_families if script == "han" else latin_families
        length = len(example["text"])
        group_number = page_counts["groups"]
        page_counts["groups"] += 1
        group_id = f"{NAMESPACE}-g{group_number:05d}"
        style_id = hashlib.sha256((example["text_kind"] + "|" + example["text"] + "|"
            + str(example["font_size_points"]) + "|" + str(example["font_size_px"])
            + "|" + example["color"]).encode()).hexdigest()[:16]
        rotation = page_counts[script] % len(families)
        page_counts[script] += 1
        ordered = families[rotation:] + families[:rotation]
        midpoint = (len(ordered) + 1) // 2
        for page_families in (ordered[:midpoint], ordered[midpoint:]):
            page_number = len(pages)
            page_id = f"{NAMESPACE}-{page_number:05d}"
            regions = []
            for slot, family in enumerate(page_families):
                faces = groups[family]
                key = (family, script, length)
                # Offset successive lengths so eight-face Latin families do
                # not tie one half of their faces to English and the other to
                # numeric text at every length.
                font = faces[(face_counts[key] + 2 * (length - 1)) % len(faces)]
                face_counts[key] += 1
                require(_supports(example["text"], [font], maps), "scene text requests fallback")
                top = 120 + slot * 238
                regions.append({
                    "id": f"{page_id}-r{slot:02d}", "font_id": font["id"],
                    "text": example["text"], "script": script,
                    "text_kind": example["text_kind"], "font_size_px": example["font_size_px"],
                    "ios_font_size_points": example["font_size_points"],
                    "color": example["color"], "bbox": [40, top, 1040, top + 190],
                    "paired_style_id": style_id, "request_origin": example["request_origin"],
                    "ios_source_page_id": example["ios_source_page_id"],
                    "ios_source_region_id": example["ios_source_region_id"],
                    "ios_capture_source_id": example["ios_capture_source_id"],
                    "ios_capture_source_sha256": example["ios_capture_source_sha256"],
                    "ios_capture_frame_path": example["ios_capture_frame_path"],
                    "ios_capture_frame_sha256": example["ios_capture_frame_sha256"],
                    "ios_actual_font_postscript": example["ios_actual_font_postscript"],
                    "ios_source_screen_scale": example["ios_source_screen_scale"],
                    "ios_actual_bbox_pixels": example["ios_actual_bbox_pixels"],
                    "ios_actual_ink_bbox_pixels": example["ios_actual_ink_bbox_pixels"],
                })
            pages.append({
                "id": page_id, "content_group_id": group_id, "split": "train",
                "background": example["background"], "paired_style_id": style_id,
                "text_kind": example["text_kind"], "regions": regions,
            })
    scenes = {
        "schema": "flux-glyph-android-scenes-v1", "seed": SEED,
        "canvas_px": [1080, 2400], "families": list(KNOWN_FAMILIES) + ["__unknown__"],
        "fonts": fonts, "pages": pages, "bindings": bindings,
        "design": {
            "namespace": NAMESPACE, "protocol": NAMESPACE, "only_split": "train",
            "all_pages_training_only": True, "test_read": False,
            "planned_pages": 128, "planned_regions": 1056,
            "planned_content_groups": 64, "pages_per_content_group": 2,
            "page_rows_by_script": {"han": [8, 7], "latin": [9, 9]},
            "short_region_characters": [1, 2, 3, 4],
            "samples_per_source_family_script_length": SAMPLES_PER_SCRIPT_LENGTH,
            "matched_text_size_color_across_eligible_source_families": True,
            "matched_size_unit": "actual iOS native screen pixels",
            "ios_point_to_screen_scale": 3,
            "face_sampling": "balanced within actual source family, script and length",
            "han_source_families": han_families, "latin_source_families": latin_families,
            "unknown_source_families": list(UNKNOWN_SOURCE_FAMILIES),
            "source_family_to_unified_family": {
                **{family: family for family in (*KNOWN_FAMILIES, *PROMOTED_UNIFIED_FAMILIES)},
                **{family: "__unknown__" for family in UNKNOWN_SOURCE_FAMILIES},
            },
            "excluded_source_families": sorted(EXPLICITLY_EXCLUDED_FAMILIES),
            "corpus_audit": corpus_audit, "ios_capture_audit": capture_audit,
            "source_rendering": "Android native only",
            "font_labels_rendered": False, "native_font_proof_required": True,
        },
    }
    validate_short_scenes(scenes, maps, forbidden)
    return scenes


def validate_short_scenes(scenes, maps, forbidden=()):
    fonts_by_id = validate_scenes(scenes)
    require(scenes.get("seed") == SEED and scenes.get("design", {}).get("only_split") == "train"
            and scenes["design"].get("test_read") is False, "short TRAIN protocol differs")
    require(len(scenes["pages"]) == scenes["design"].get("planned_pages") == 128,
            "short request page count differs")
    selected_families = {font["family"] for font in scenes["fonts"]}
    require(selected_families == set(KNOWN_FAMILIES) | set(PROMOTED_UNIFIED_FAMILIES)
            | set(UNKNOWN_SOURCE_FAMILIES)
            and not selected_families.intersection(EXPLICITLY_EXCLUDED_FAMILIES),
            "short source family set differs")
    forbidden = set(forbidden)
    coverage = Counter()
    kinds = Counter()
    face_coverage = defaultdict(Counter)
    total = 0
    page_groups = defaultdict(list)
    for page in scenes["pages"]:
        require(page["split"] == "train", "short page is not TRAIN")
        page_groups[page["content_group_id"]].append(page)
    require(len(page_groups) == scenes["design"].get("planned_content_groups") == 64
            and all(len(group) == 2 for group in page_groups.values()),
            "paired content-group topology differs")
    for group in page_groups.values():
        rows = [row for page in group for row in page["regions"]]
        require(bool(rows), "short content group is empty")
        script = rows[0]["script"]
        expected_families = (set(KNOWN_FAMILIES[:-1]) | set(PROMOTED_UNIFIED_FAMILIES)
                             | {"Liu Jian Mao Cao", "Smiley Sans"} if script == "han"
                             else set(KNOWN_FAMILIES) | set(PROMOTED_UNIFIED_FAMILIES)
                             | set(UNKNOWN_SOURCE_FAMILIES))
        require(sorted(len(page["regions"]) for page in group)
                == sorted(scenes["design"]["page_rows_by_script"][script]),
                "paired page row count differs")
        require(len({fonts_by_id[row["font_id"]]["family"] for row in rows}) == len(rows)
                and {fonts_by_id[row["font_id"]]["family"] for row in rows} == expected_families,
                "paired content-group source families differ")
        require(len({(row["text"], row["font_size_px"], row["ios_font_size_points"],
                      row["color"], row["text_kind"], row["paired_style_id"])
                     for row in rows}) == 1
                and len({page["background"] for page in group}) == 1,
                "text and style must be paired across source families")
        require(all(row["script"] == script and 1 <= len(row["text"]) <= 4 for row in rows),
                "short script or length differs")
        for row in rows:
            font = fonts_by_id[row["font_id"]]
            require(text_sha(row["text"]) not in forbidden, "CAL/DEV full-text hash reused")
            require(_supports(row["text"], [font], maps), "unsupported source glyph")
            require(row["ios_source_screen_scale"] == 3
                    and row["font_size_px"] == row["ios_font_size_points"] * 3
                    and 36 <= row["font_size_px"] <= 120
                    and row["ios_capture_source_id"] == "ios:" + row["ios_source_page_id"]
                    and len(row["ios_capture_source_sha256"]) == 64
                    and len(row["ios_capture_frame_sha256"]) == 64,
                    "actual iOS screen-pixel style proof differs")
            require(script != "han" or "han" in font.get("scripts", ()),
                    "Latin-only source received Han text")
            length = len(row["text"])
            family = font["family"]
            coverage[family, script, length] += 1
            kinds[family, script, length, row["text_kind"]] += 1
            face_coverage[family, script, length][font["id"]] += 1
            total += 1
    require(total == scenes["design"].get("planned_regions") == 1056,
            "short request region count differs")
    for family in set(KNOWN_FAMILIES) | set(PROMOTED_UNIFIED_FAMILIES) | set(UNKNOWN_SOURCE_FAMILIES):
        scripts = ("latin",) if family in {"Roboto", "Lato", "Open Sans"} else ("han", "latin")
        for script in scripts:
            for length in range(1, 5):
                require(coverage[family, script, length] == SAMPLES_PER_SCRIPT_LENGTH,
                        "source family/script/length coverage differs")
                face_counts = face_coverage[family, script, length]
                family_faces = [font for font in scenes["fonts"] if font["family"] == family]
                require(set(face_counts) == {font["id"] for font in family_faces}
                        and max(face_counts.values()) - min(face_counts.values()) <= 1,
                        "source face balance differs")
                if script == "latin":
                    require(kinds[family, script, length, "english"] == 4
                            and kinds[family, script, length, "numeric"] == 4,
                            "Latin kind balance differs")


def request_report(scenes, metadata_bindings, scene_path):
    rows = [row for page in scenes["pages"] for row in page["regions"]]
    fonts = {font["id"]: font for font in scenes["fonts"]}
    unified_mapping = scenes["design"]["source_family_to_unified_family"]
    return {
        "schema": "flux-glyph-android-short-train-requests-v1",
        "pages": len(scenes["pages"]), "regions": len(rows),
        "source_family_counts": dict(sorted(Counter(fonts[row["font_id"]]["family"] for row in rows).items())),
        "registry_training_family_counts": dict(sorted(Counter(
            fonts[row["font_id"]]["training_family"] for row in rows).items())),
        "unified_family_counts": dict(sorted(Counter(
            unified_mapping[fonts[row["font_id"]]["family"]] for row in rows).items())),
        "face_counts": dict(sorted(Counter(row["font_id"] for row in rows).items())),
        "script_counts": dict(Counter(row["script"] for row in rows)),
        "length_counts": dict(Counter(len(row["text"]) for row in rows)),
        "kind_counts": dict(Counter(row["text_kind"] for row in rows)),
        "ios_point_size_counts": dict(Counter(row["ios_font_size_points"] for row in rows)),
        "matched_screen_pixel_size_counts": dict(Counter(row["font_size_px"] for row in rows)),
        "page_row_counts": dict(Counter(len(page["regions"]) for page in scenes["pages"])),
        "corpus_audit": scenes["design"]["corpus_audit"],
        "full_text_cal_dev_hash_overlap": 0, "metadata_bindings": metadata_bindings,
        "generator_sha256": sha(__file__), "scenes_sha256": sha(scene_path),
        "test_read": False, "native_font_labels_verified": False,
        "samples_used_in_training": False, "source_kind": "authored_native_capture_requests",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fonts", type=Path, required=True)
    parser.add_argument("--ios-scenes", type=Path, required=True)
    parser.add_argument("--ios-capture", type=Path, required=True)
    parser.add_argument("--reused-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "refusing to overwrite an existing request bundle")
    registry, fonts, maps, bindings = load_sources(args.fonts)
    ios_scenes = read(args.ios_scenes)
    bindings[str(args.ios_scenes.resolve())] = sha(args.ios_scenes)
    ios_capture, capture_bindings, capture_audit = load_ios_capture(
        args.ios_capture, args.ios_scenes, ios_scenes)
    bindings.update(capture_bindings)
    forbidden, heldout_bindings, heldout_counts = heldout_hashes(args.reused_data)
    bindings.update(heldout_bindings)
    scenes = build_scenes(registry, fonts, maps, ios_scenes, ios_capture, forbidden,
                          bindings, capture_audit)
    args.output.mkdir(parents=True)
    scene_path = args.output / "Scenes.json"
    dump(scene_path, scenes)
    report = request_report(scenes, {
        **heldout_bindings, str(args.fonts.resolve()): sha(args.fonts),
        str(args.ios_scenes.resolve()): sha(args.ios_scenes),
        str((args.ios_capture / "labels.jsonl").resolve()): sha(args.ios_capture / "labels.jsonl"),
        str((args.ios_capture / "CAPTURE_PROTOCOL.json").resolve()): sha(args.ios_capture / "CAPTURE_PROTOCOL.json"),
        str((args.ios_capture / "CAPTURE_SUMMARY.json").resolve()): sha(args.ios_capture / "CAPTURE_SUMMARY.json"),
    }, scene_path)
    report["heldout_rows_read_by_split"] = heldout_counts
    dump(args.output / "REQUESTS.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
