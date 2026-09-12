#!/usr/bin/env python3
"""Prepare CNN glyphs from real iOS Simulator PNGs and native CTRun evidence.

No OCR, desktop font renderer or reference matcher is involved. Every crop is
bound to the checked screenshot, scene request, actual CTFont/CTRun glyph and
its instrumented screen-pixel box. Corrupt source contracts and split leakage
abort the import; explicitly rejected native regions/glyphs are counted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unicodedata

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))
from flux_glyph.neural_font import preprocess_glyph  # noqa: E402
from prepare_screenshots import family_names, script_families  # noqa: E402

SCHEMA = "flux-glyph-native-captured-glyphs-v1"
METADATA_SCHEMA = "flux-glyph-native-captured-metadata-v1"
SPLITS = ("train", "calibration", "test")
SOURCE_KIND = "ios_simulator_screenshot"
SHA = re.compile(r"[0-9a-f]{64}\Z")
MAX_PAGES = 20000
MAX_PIXELS = 32_000_000
MAX_GLYPHS = 2_000_000


def require(condition, message):
    if not condition:
        raise ValueError("Captured screenshots: " + message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def pixels_sha(image):
    prefix = f"RGB:{image.width}x{image.height}".encode() + b"\0"
    return hashlib.sha256(prefix + image.tobytes()).hexdigest()


def bounds(box, size):
    return (isinstance(box, list) and len(box) == 4 and all(type(x) is int for x in box) and
            0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1])


def contained(inner, outer):
    return outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3]


def script_of(character):
    if not isinstance(character, str) or len(character) != 1:
        return None
    if "\u4e00" <= character <= "\u9fff":
        return "han"
    return "latin" if character.isascii() and character.isalnum() else None


def normalized_text(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


class Isolation:
    def __init__(self):
        self.values = {}

    def bind(self, kind, value, split):
        require(isinstance(value, str) and bool(value), "missing split identity: " + kind)
        key = (kind, value)
        prior = self.values.setdefault(key, split)
        require(prior == split, f"cross-split leakage: {kind} belongs to {prior} and {split}")


def load_scenes(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    require(document.get("schema") == "flux-glyph-capture-scenes-v1" and document.get("platform") == "ios",
            "only the native iOS scene protocol is accepted")
    pages = document.get("pages")
    require(isinstance(pages, list) and 0 < len(pages) <= MAX_PAGES, "invalid scene page registry")
    registry, isolation = {}, Isolation()
    for page in pages:
        require(isinstance(page, dict) and isinstance(page.get("id"), str) and page["id"] not in registry,
                "duplicate or missing scene page ID")
        split = page.get("split")
        require(split in SPLITS, "unknown scene split")
        group = page.get("content_group_id", page["id"])
        isolation.bind("content_group_id", group, split)
        require(isinstance(page.get("regions"), list) and 0 < len(page["regions"]) <= 128, "invalid scene regions")
        ids = set()
        for region in page["regions"]:
            require(isinstance(region, dict) and isinstance(region.get("id"), str) and region["id"] not in ids,
                    "duplicate or missing scene region ID")
            ids.add(region["id"])
            require(isinstance(region.get("text"), str) and 0 < len(region["text"]) <= 256, "invalid scene text")
            isolation.bind("normalized_region_text", normalized_text(region["text"]), split)
        registry[page["id"]] = page
    return document, registry, isolation


def capture_protocol(label_root, scenes_sha):
    path = label_root / "CAPTURE_PROTOCOL.json"
    require(path.is_file(), "CAPTURE_PROTOCOL.json from the simctl capture driver is required")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("schema") == "ios-native-screen-capture-v1" and value.get("source_kind") == SOURCE_KIND,
            "only the native simulator screen capture protocol is accepted")
    require(value.get("scenes_sha256") == scenes_sha, "capture protocol scene SHA differs")
    require(value.get("capture_command") == "xcrun simctl io <simulator> screenshot --type=png <file>",
            "capture protocol did not use actual simctl PNG screenshots")
    require(valid_sha(value.get("capture_driver_sha256")), "missing native capture driver provenance")
    return path, value


def native_source(row, scenes_sha, pages, label_root):
    require(isinstance(row, dict) and row.get("schema") == "ios-native-captured-frame-v1", "wrong capture record schema")
    require(row.get("source_kind") == SOURCE_KIND, "desktop renders and unverified source kinds are not accepted")
    require(row.get("split") in SPLITS, "invalid capture split")
    require(isinstance(row.get("source_id"), str) and row["source_id"].strip(), "missing source_id")
    require(row.get("page_id") in pages, "capture page absent from checked scene protocol")
    page = pages[row["page_id"]]
    require(row["split"] == page["split"], "captured split differs from predetermined scene split")
    require(valid_sha(row.get("source_sha256")) and row.get("scenes_sha256") == scenes_sha, "missing or mismatched source/scenes SHA")
    require(isinstance(row.get("image"), str) and row["image"], "missing screenshot path")
    path = Path(row["image"])
    path = (label_root / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_file() and path.stat().st_size <= 128 * 1024 * 1024, "screenshot missing or exceeds file limit")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == row["source_sha256"], "screenshot SHA-256 mismatch: " + path.name)
    with Image.open(io.BytesIO(raw)) as opened:
        require(opened.format == "PNG", "simulator screen source must be PNG")
        require(opened.width * opened.height <= MAX_PIXELS and max(opened.size) <= 8192, "screenshot exceeds pixel limit")
        require(opened.getexif().get(274, 1) == 1, "screenshot orientation must already match instrumented pixel boxes")
        require(row.get("pixel_size") == list(opened.size), "PNG dimensions differ from capture record")
        image = opened.convert("RGB")
    native = row.get("native")
    require(isinstance(native, dict) and native.get("schema") == "flux-glyph-ios-frame-v1", "missing native instrumentation")
    require(native.get("platform") == "ios" and native.get("renderer") == "UIView.draw + CTLineDraw", "unsupported native renderer")
    for key in ("page_id", "split", "pixel_size"):
        require(native.get(key) == row.get(key), "native/captured " + key + " mismatch")
    require(native.get("scene_manifest_sha256") == scenes_sha, "native scene source SHA mismatch")
    require(isinstance(native.get("regions"), list) and native["regions"] == row.get("regions"),
            "native region evidence differs from capture annotation")
    require(len(native["regions"]) == len(page["regions"]), "native scene region count differs")
    requested = {region["id"]: region for region in page["regions"]}
    ids = [region.get("id") for region in native["regions"]]
    require(len(ids) == len(set(ids)) and set(ids) == set(requested), "native scene region identity mismatch")
    for region in native["regions"]:
        request = requested[region["id"]]
        for key in ("text", "script", "font_family"):
            require(region.get(key) == request.get(key), "native/requested region " + key + " mismatch")
        require(region.get("requested_font_postscript") == request.get("font_postscript"),
                "native/requested PostScript name mismatch")
        if request.get("language") is not None:
            require(region.get("requested_language") == request["language"], "native/requested shaping language mismatch")
    return page, path, image, requested


def font_evidence(region, family, script):
    """Return an explicit rejection; do not reinterpret a native fallback."""
    if region.get("font_match_verified") is not True or region.get("status") != "ok":
        return "native_region_unverified:" + str(region.get("reason", "unknown"))
    if region.get("fallback_detected") is not False:
        return "native_font_fallback_or_missing_proof"
    if region.get("clipped") is not False:
        return "native_region_clipped_or_missing_proof"
    actual_ps = region.get("actual_font_postscript")
    if not isinstance(actual_ps, str) or not actual_ps or region.get("font_postscript") != actual_ps:
        return "actual_font_postscript_mismatch"
    if region.get("requested_font_family") != family:
        return "native_family_request_mismatch"
    if family in {"PingFang SC", "PingFang TC", "PingFang HK"}:
        prefix = "PingFang" + family.rsplit(" ", 1)[1] + "-"
        if (not actual_ps.startswith(prefix) or region.get("actual_font_family") != family
                or region.get("requested_font_postscript") != actual_ps):
            return "native_pingfang_regional_family_mismatch"
    coverage = region.get("glyph_coverage", {})
    if (not isinstance(coverage, dict) or coverage.get("font_get_glyphs_succeeded") is not True or
            coverage.get("zero_run_glyph_count") != 0 or
            coverage.get("utf16_count") != len(region["text"].encode("utf-16-le")) // 2 or
            not isinstance(coverage.get("requested_font_glyph_ids"), list) or
            len(coverage["requested_font_glyph_ids"]) != coverage["utf16_count"] or
            any(type(value) is not int or value <= 0 for value in coverage["requested_font_glyph_ids"])):
        return "native_glyph_coverage_unverified"
    if (not isinstance(region.get("glyphs"), list) or not isinstance(region.get("font_runs"), list) or
            not region["font_runs"] or not region["glyphs"]):
        return "missing_native_glyph_or_run_evidence"
    runs = []
    for run in region["font_runs"]:
        if (run.get("postscript_name") != actual_ps or not isinstance(run.get("glyph_ids"), list) or
                not isinstance(run.get("string_indices_utf16"), list) or
                run.get("glyph_count") != len(run["glyph_ids"]) or len(run["glyph_ids"]) != len(run["string_indices_utf16"])):
            return "native_font_run_mismatch"
        if family in {"PingFang SC", "PingFang TC", "PingFang HK"} and run.get("family") != family:
            return "native_pingfang_regional_run_mismatch"
        if region.get("requested_language") is not None and run.get("language") != region["requested_language"]:
            return "native_font_run_language_mismatch"
        runs.extend(zip(run["string_indices_utf16"], run["glyph_ids"]))
    glyphs = region["glyphs"]
    if len(runs) != len(glyphs) or Counter(runs) != Counter((g.get("text_index"), g.get("glyph_id")) for g in glyphs):
        return "native_glyph_run_binding_mismatch"
    indices = [glyph.get("text_index") for glyph in glyphs]
    if any(type(index) is not int for index in indices) or len(indices) != len(set(indices)):
        return "non_unique_character_glyph_mapping"
    expected = {i for i, char in enumerate(region["text"]) if script_of(char) == script}
    present = {g.get("text_index") for g in glyphs if g.get("font_classification_character") is True}
    if present != expected:
        return "incomplete_native_classification_characters"
    return None


def asset_evidence(region, request, registered):
    """For bundled fonts, bind the rendered native font to the checked file."""
    entry = registered.get(request.get("font_id"))
    if not isinstance(entry, dict):
        return "requested_font_absent_from_scene_registry"
    if entry.get("family") != region["font_family"]:
        return "font_registry_family_mismatch"
    if entry.get("kind") == "system":
        return None
    proof = region.get("font_source")
    if (entry.get("kind") != "asset" or not isinstance(proof, dict) or proof.get("kind") != "asset" or
            proof.get("sha256_verified") is not True or proof.get("registration_succeeded") is not True or
            proof.get("native_font_url_verified") is not True or not valid_sha(entry.get("sha256")) or
            proof.get("sha256") != entry["sha256"] or proof.get("postscript") != entry.get("postscript") or
            proof.get("postscript") != region.get("actual_font_postscript") or
            not isinstance(proof.get("file_url"), str) or proof.get("native_font_url") != proof["file_url"]):
        return "asset_font_source_unverified"
    if any(g.get("font_source_kind") != "asset" or g.get("font_source_sha256") != entry["sha256"]
           for g in region["glyphs"] if g.get("font_classification_character") is True):
        return "asset_glyph_source_unverified"
    return None


def native_style(region, native):
    """Keep resolved screen geometry/color separate from 64px CNN features."""
    keys = ("actual_font_size_points", "font_size_screen_px", "source_screen_scale")
    for key in keys:
        value = region.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and 0 < value <= 4096,
                "missing or invalid resolved native style: " + key)
    require(abs(region["font_size_screen_px"] - region["actual_font_size_points"] * region["source_screen_scale"]) < .001,
            "resolved screen font size does not equal points times screen scale")
    require(native.get("screen_scale") == region["source_screen_scale"], "native source screen scales differ")
    color = region.get("text_color_hex")
    require(isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{8}", color) is not None,
            "missing actual resolved RGBA text color")
    require(region.get("actual_text_color_hex") == color, "actual resolved text colors differ")
    return {**{key: region[key] for key in keys}, "text_color_hex": color,
            "actual_text_color_hex": color, "background_hex": region.get("background_hex"),
            "requested_font_size_points": region.get("requested_font_size_points")}


def finalized_array(raw, destination, count):
    with destination.open("wb") as output:
        np.lib.format.write_array_header_2_0(output, {"descr": "<f4", "fortran_order": False, "shape": (count, 64, 64)})
        with raw.open("rb") as source:
            shutil.copyfileobj(source, output, length=8 << 20)
    require(raw.stat().st_size == count * 64 * 64 * 4, "raw glyph byte count differs")
    raw.unlink()


def finalized_metadata(rows_file, destination, split, count):
    with destination.open("w", encoding="utf-8") as output, rows_file.open(encoding="utf-8") as source:
        header = {"schema": METADATA_SCHEMA, "split": split, "row_count": count}
        output.write(json.dumps(header)[:-1] + ', "rows": [\n')
        for index, line in enumerate(source):
            if index:
                output.write(",\n")
            output.write(line.rstrip("\n"))
        output.write("\n]}\n")
    rows_file.unlink()


def prepare(input_file, output_dir, *, scenes_file=None):
    input_file, output_dir = Path(input_file).resolve(), Path(output_dir).resolve()
    scenes_file = Path(scenes_file).resolve() if scenes_file else input_file.parent / "Scenes.json"
    require(input_file.is_file() and input_file.stat().st_size <= 1024 * 1024 * 1024, "missing or oversized labels JSONL")
    require(scenes_file.is_file(), "checked Scenes.json is required beside labels or via --scenes")
    require(not output_dir.exists() or not any(output_dir.iterdir()), "output directory must be new or empty")
    scenes, pages, isolation = load_scenes(scenes_file)
    scenes_sha, labels_sha = sha(scenes_file), sha(input_file)
    protocol_path, protocol = capture_protocol(input_file.parent, scenes_sha)
    protocol_sha = sha(protocol_path)
    registered = {font["id"]: font for font in scenes.get("fonts", [])}
    families = family_names()
    scripts = script_families(families)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native-capture-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir()
        streams, records = {}, {}
        totals = {split: 0 for split in SPLITS}
        family_counts = {split: Counter() for split in SPLITS}
        script_counts = {split: Counter() for split in SPLITS}
        rejection_counts, rejection_examples, sources = Counter(), [], []
        seen_sources, seen_source_keys, pixel_labels = {}, set(), {}
        for split in SPLITS:
            (stage / split).mkdir()
            streams[split] = (stage / split / "glyphs.raw").open("wb")
            records[split] = (stage / split / "rows.jsonl").open("w", encoding="utf-8")

        def reject(reason, row, region=None, glyph=None):
            rejection_counts[reason] += 1
            if len(rejection_examples) < 200:
                rejection_examples.append({"reason": reason, "source_id": row.get("source_id"),
                                           "region_id": region.get("id") if region else None,
                                           "text_index": glyph.get("text_index") if glyph else None})

        try:
            with input_file.open(encoding="utf-8") as incoming:
                for line_number, line in enumerate(incoming, 1):
                    if not line.strip():
                        continue
                    require(len(line) <= 8 * 1024 * 1024 and len(sources) < MAX_PAGES, "capture annotation exceeds bounds")
                    row = json.loads(line)
                    page, path, image, requests = native_source(row, scenes_sha, pages, input_file.parent)
                    require(row.get("simulator_id") == protocol.get("simulator_id") and
                            row.get("bundle_id") == protocol.get("bundle_id"), "capture device/app differs from protocol")
                    split, decoded_sha = row["split"], pixels_sha(image)
                    identities = {"source_id": row["source_id"], "source_file_sha256": row["source_sha256"],
                                  "decoded_pixel_sha256": decoded_sha, "page_id": row["page_id"],
                                  "content_group_id": page.get("content_group_id", page["id"])}
                    for kind, value in identities.items():
                        isolation.bind(kind, value, split)
                    previous = seen_sources.setdefault(row["source_id"], row["source_sha256"])
                    require(previous == row["source_sha256"], "one source_id names multiple screenshot files")
                    source_key = (row["source_id"], row["source_sha256"])
                    if source_key in seen_source_keys:
                        reject("duplicate_source_within_split", row)
                        continue
                    seen_source_keys.add(source_key)
                    source = {"source_id": row["source_id"], "page_id": row["page_id"], "split": split,
                              "source_kind": row["source_kind"], "image": str(path), "image_path": str(path), "source_sha256": row["source_sha256"],
                              "decoded_pixel_sha256": decoded_sha, "pixel_size": list(image.size),
                              "scenes_sha256": scenes_sha, "content_group_id": identities["content_group_id"],
                              "label_line": line_number, "native_frame_sha256": hashlib.sha256(json.dumps(row["native"], sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                              "native_os_version": row["native"].get("os_version"), "regions": []}
                    sources.append(source)
                    for region in row["regions"]:
                        family, script, text = region["font_family"], region["script"], region["text"]
                        require(script in scripts and family in scripts[script], "family/script outside training registry")
                        require(all(ord(c) <= 0xffff for c in text), "only unambiguous BMP/UTF16 instrumentation is supported")
                        require(not any(script_of(c) not in (None, script) for c in text), "mixed-script native region")
                        # Rejected native regions need not have a drawable box.
                        reason = font_evidence(region, family, script)
                        if reason is None:
                            reason = asset_evidence(region, requests[region["id"]], registered)
                        if reason:
                            reject(reason, row, region)
                            continue
                        style = native_style(region, row["native"])
                        require(bounds(region.get("bbox"), image.size), "invalid instrumented region bbox")
                        region_image = image.crop(region["bbox"])
                        region_sha = pixels_sha(region_image)
                        isolation.bind("region_rgb_sha256", region_sha, split)
                        isolation.bind("normalized_region_text", normalized_text(text), split)
                        label_key = (region_sha, script)
                        label_value = (family, normalized_text(text))
                        require(pixel_labels.setdefault(label_key, label_value) == label_value,
                                "same region pixels have conflicting font/text labels")
                        region_source = {"id": region["id"], "text": text, "script": script, "font_family": family,
                                         "bbox": region["bbox"], "region_rgb_sha256": region_sha,
                                         "font_postscript": region["font_postscript"], "font_match_verified": True,
                                         "font_runs": region["font_runs"], "glyph_coverage": region["glyph_coverage"],
                                         "font_source": region.get("font_source"),
                                         **style,
                                         "fallback_detected": False, "native_rejection_reason": region.get("reason")}
                        source["regions"].append(region_source)
                        data = np.asarray(region_image)
                        border = np.concatenate((data[0], data[-1], data[:, 0], data[:, -1]))
                        background = tuple(int(value) for value in np.median(border, axis=0))
                        for glyph in region["glyphs"]:
                            if glyph.get("font_classification_character") is not True:
                                rejection_counts["non_classification_character"] += 1
                                continue
                            index, character = glyph.get("text_index"), glyph.get("character")
                            require(type(index) is int and 0 <= index < len(text) and character == text[index], "glyph character/index differs from source text")
                            require(script_of(character) == script, "glyph script differs from source annotation")
                            if (glyph.get("font_match_verified") is not True or glyph.get("visible") is not True or
                                    glyph.get("font_family") != family or glyph.get("font_postscript") != region["actual_font_postscript"] or
                                    type(glyph.get("glyph_id")) is not int or glyph["glyph_id"] <= 0):
                                reject("native_glyph_font_unverified", row, region, glyph)
                                continue
                            box = glyph.get("bbox")
                            require(bounds(box, image.size) and contained(box, region["bbox"]), "invalid or out-of-region instrumented glyph bbox")
                            if "bbox_screen_px" in glyph:
                                require(glyph["bbox_screen_px"] == box, "glyph bbox aliases differ")
                            for key in ("actual_font_size_points", "font_size_screen_px", "source_screen_scale", "text_color_hex"):
                                if key in glyph:
                                    require(glyph[key] == style[key], "glyph and region resolved style differ: " + key)
                            crop = image.crop(box)
                            padded = Image.new("RGB", (crop.width + 8, crop.height + 8), background)
                            padded.paste(crop, (4, 4))
                            raster = preprocess_glyph(padded)
                            if raster is None:
                                reject("runtime_preprocess_rejected", row, region, glyph)
                                continue
                            require(sum(totals.values()) < MAX_GLYPHS, "prepared glyph count exceeds bounds")
                            raster = np.asarray(raster, dtype="<f4", order="C")
                            raw = raster.tobytes()
                            entry = {"array_row": totals[split], "target": families.index(family), "family": family,
                                     "character": character, "script": script, "source_id": row["source_id"],
                                     "source_kind": SOURCE_KIND, "page_id": row["page_id"], "region_id": region["id"],
                                     "content_group_id": identities["content_group_id"], "text": text, "text_index": index,
                                     "font_face": glyph["font_postscript"], "font_size_points": region.get("font_size_points"),
                                     **style,
                                     "ct_font_size": region.get("ct_font_size"), "bbox": box, "region_bbox": region["bbox"],
                                     "source_sha256": row["source_sha256"], "decoded_pixel_sha256": decoded_sha,
                                     "region_rgb_sha256": region_sha, "crop_sha256": pixels_sha(crop),
                                     "glyph_sha256": hashlib.sha256(raw).hexdigest(), "scenes_sha256": scenes_sha,
                                     "background_rgb": background, "padding_pixels": 4,
                                     "requested_font_postscript": region.get("requested_font_postscript"),
                                     "native_font_source": region.get("font_source"),
                                     "native_glyph_provenance": glyph}
                            streams[split].write(raw)
                            records[split].write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                            totals[split] += 1
                            family_counts[split][family] += 1
                            script_counts[split][script] += 1
                    if len(sources) % 20 == 0:
                        print(json.dumps({"screenshots": len(sources), "glyphs": totals}), flush=True)
        finally:
            for stream in list(streams.values()) + list(records.values()):
                stream.close()
        require(sources, "no captured screenshots")
        require(sum(totals.values()) > 0, "no verified usable native glyphs")
        require(sha(input_file) == labels_sha and sha(scenes_file) == scenes_sha, "capture labels/scenes changed during preparation")
        require(sha(protocol_path) == protocol_sha, "capture protocol changed during preparation")
        partitions = {}
        for split in SPLITS:
            folder = stage / split
            finalized_array(folder / "glyphs.raw", folder / "glyphs.npy", totals[split])
            finalized_metadata(folder / "rows.jsonl", folder / "metadata.json", split, totals[split])
            partitions[split] = {"rows": totals[split],
                                 "glyphs": {"path": f"{split}/glyphs.npy", "sha256": sha(folder / "glyphs.npy"),
                                            "shape": [totals[split], 64, 64], "dtype": "float32"},
                                 "metadata": {"path": f"{split}/metadata.json", "sha256": sha(folder / "metadata.json")},
                                 "family_counts": dict(family_counts[split]), "script_counts": dict(script_counts[split])}
        manifest = {"schema": SCHEMA, "families": families, "scripts": scripts, "splits": partitions,
                    "input_labels": {"path": str(input_file), "sha256": labels_sha},
                    "scenes": {"path": str(scenes_file), "sha256": scenes_sha},
                    "capture_protocol": {"path": str(protocol_path), "sha256": protocol_sha},
                    "capture_protocol_sha256": protocol_sha, "ui_content_is_generated": True,
                    "image_source": "simctl_png", "accepted_native_verified_only": True,
                    "source_kind_counts": dict(Counter(source["source_kind"] for source in sources)),
                    "screenshot_count": len(sources), "glyph_count": sum(totals.values()), "sources": sources,
                    "split_isolation": {kind: 0 for kind in ("source_id", "source_file_sha256", "decoded_pixel_sha256",
                                                             "page_id", "region_rgb_sha256", "content_group_id", "normalized_region_text")},
                    "split_policy": "Predetermined scene/source/content groups and full decoded screenshot/region pixels cannot cross splits; individual characters may repeat.",
                    "decoded_pixel_hash_algorithm": "sha256(UTF8('RGB:<width>x<height>') + NUL + RGB bytes)",
                    "preprocessing": {"function": "flux_glyph.neural_font.preprocess_glyph", "padding_pixels": 4,
                                      "padding_background": "median RGB of instrumented region border, same as runtime pipeline",
                                      "shape": [64, 64], "dtype": "float32", "range": [0, 1],
                                      "runtime_source_sha256": sha(ROOT / "src/flux_glyph/neural_font.py"),
                                      "glyph_extraction_source_sha256": sha(ROOT / "src/flux_glyph/glyph_preprocess.py")},
                    "font_label_source": "Actual iOS UIView CTLineDraw CTFont/CTRun glyph identities verified against native request and scene manifest",
                    "ocr_performed": False, "ocr_oracle_boxes_claimed": False, "desktop_rendered_pixels_accepted": False,
                    "instrumented_boxes_used": True, "source_font_file_hash_required_for_builtins": False,
                    "rejected": {"counts": dict(rejection_counts), "examples": rejection_examples},
                    "preparation_code_sha256": sha(Path(__file__))}
        (stage / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        if output_dir.exists():
            require(not any(output_dir.iterdir()), "output changed during preparation")
            output_dir.rmdir()
        stage.rename(output_dir)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="capture_ios.py labels.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenes", type=Path, help="Defaults to Scenes.json beside labels")
    args = parser.parse_args()
    manifest = prepare(args.input, args.output, scenes_file=args.scenes)
    print(json.dumps({"screenshots": manifest["screenshot_count"], "glyphs": manifest["glyph_count"],
                      "splits": {key: value["rows"] for key, value in manifest["splits"].items()},
                      "rejected": manifest["rejected"]["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
