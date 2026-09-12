"""Contract-only mock capture records; never used as a real screenshot corpus."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin
import pytest

PATH = Path(__file__).resolve().parents[1] / "training/capture/prepare_captured.py"
SPEC = importlib.util.spec_from_file_location("prepare_captured", PATH)
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


def write_labels(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def dataset(tmp_path, texts=("青甲",), splits=("train",)):
    fonts = [{"id": "pingfang", "family": "PingFang SC", "postscript": "PingFangSC-Regular", "kind": "system"}]
    pages = [{"id": f"page-{i}", "content_group_id": f"content-{i}", "split": split,
              "regions": [{"id": "r1", "text": text, "script": "han", "font_family": "PingFang SC",
                           "font_id": "pingfang", "font_postscript": "PingFangSC-Regular"}]}
             for i, (text, split) in enumerate(zip(texts, splits))]
    scenes = {"schema": "flux-glyph-capture-scenes-v1", "platform": "ios", "fonts": fonts, "pages": pages}
    scenes_path = tmp_path / "Scenes.json"
    scenes_path.write_text(json.dumps(scenes, ensure_ascii=False))
    scene_sha = capture.sha(scenes_path)
    protocol = {"schema": "ios-native-screen-capture-v1", "source_kind": "ios_simulator_screenshot",
                "scenes_sha256": scene_sha, "capture_command": "xcrun simctl io <simulator> screenshot --type=png <file>",
                "capture_driver_sha256": "a" * 64, "simulator_id": "unit-test-simulator", "bundle_id": "test.capture"}
    (tmp_path / "CAPTURE_PROTOCOL.json").write_text(json.dumps(protocol))
    rows = []
    for i, page in enumerate(pages):
        image = Image.new("RGB", (100, 80), "white")
        drawing = ImageDraw.Draw(image)
        drawing.rectangle((17, 21, 20, 48), fill="black")
        drawing.rectangle((17, 21, 32, 25), fill="black")
        drawing.rectangle((17, 36, 29, 40), fill="black")
        drawing.rectangle((50, 21, 54, 48), fill=(i * 20, i * 20, i * 20))
        drawing.rectangle((50, 21 + i * 3, 65, 26 + i * 3), fill="black")
        drawing.rectangle((50, 42, 65, 48), fill="black")
        source = tmp_path / f"screen-{i}.png"
        image.save(source)
        boxes = [[15, 20, 35, 50], [48, 20, 68, 50]]
        glyphs = [{"character": character, "text_index": j, "glyph_id": 100 + j,
                   "bbox": boxes[j], "bbox_screen_px": boxes[j], "visible": True,
                   "font_classification_character": True, "font_postscript": "PingFangSC-Regular",
                   "font_family": "PingFang SC", "actual_font_family": "PingFang SC", "font_match_verified": True}
                  for j, character in enumerate(page["regions"][0]["text"])]
        region = {"id": "r1", "text": page["regions"][0]["text"], "script": "han", "font_family": "PingFang SC",
                  "requested_font_family": "PingFang SC", "requested_font_postscript": "PingFangSC-Regular",
                  "font_postscript": "PingFangSC-Regular", "actual_font_postscript": "PingFangSC-Regular",
                  "actual_font_family": "PingFang SC", "font_match_verified": True, "status": "ok",
                  "fallback_detected": False, "clipped": False, "bbox": [10, 10, 90, 60],
                  "font_size_points": 20, "ct_font_size": 20, "glyphs": glyphs,
                  "actual_font_size_points": 20, "font_size_screen_px": 20, "source_screen_scale": 1,
                  "requested_font_size_points": 20, "text_color_hex": "#000000FF", "actual_text_color_hex": "#000000FF",
                  "font_runs": [{"postscript_name": "PingFangSC-Regular", "family": "PingFang SC",
                                 "glyph_count": 2, "glyph_ids": [100, 101], "string_indices_utf16": [0, 1]}],
                  "glyph_coverage": {"utf16_count": 2, "font_get_glyphs_succeeded": True,
                                     "requested_font_glyph_ids": [100, 101], "zero_run_glyph_count": 0}}
        native = {"schema": "flux-glyph-ios-frame-v1", "platform": "ios", "renderer": "UIView.draw + CTLineDraw",
                  "page_id": page["id"], "split": page["split"], "pixel_size": [100, 80],
                  "scene_manifest_sha256": scene_sha, "os_version": "unit-test", "screen_scale": 1, "regions": [region]}
        rows.append({"schema": "ios-native-captured-frame-v1", "source_id": "ios:" + page["id"],
                     "page_id": page["id"], "split": page["split"], "source_kind": "ios_simulator_screenshot",
                     "image": str(source), "source_sha256": capture.sha(source), "scenes_sha256": scene_sha,
                     "pixel_size": [100, 80], "native": native, "regions": native["regions"],
                     "simulator_id": protocol["simulator_id"], "bundle_id": protocol["bundle_id"]})
    labels = tmp_path / "labels.jsonl"
    write_labels(labels, rows)
    return labels, rows, scenes


def test_native_boxes_produce_runtime_identical_mmap_arrays(tmp_path):
    labels, rows, _ = dataset(tmp_path)
    output = tmp_path / "prepared"
    manifest = capture.prepare(labels, output)
    assert manifest["schema"] == capture.SCHEMA and manifest["glyph_count"] == 2
    assert manifest["ocr_performed"] is False and manifest["desktop_rendered_pixels_accepted"] is False
    assert manifest["source_kind_counts"] == {"ios_simulator_screenshot": 1}
    actual = np.load(output / "train/glyphs.npy", mmap_mode="r", allow_pickle=False)
    assert isinstance(actual, np.memmap) and actual.shape == (2, 64, 64) and actual.dtype == np.float32
    metadata = json.loads((output / "train/metadata.json").read_text())
    assert metadata["schema"] == capture.METADATA_SCHEMA and metadata["row_count"] == 2
    with Image.open(rows[0]["image"]) as opened:
        original = opened.convert("RGB")
    for index, row in enumerate(metadata["rows"]):
        crop = original.crop(row["bbox"])
        padded = Image.new("RGB", (crop.width + 8, crop.height + 8), "white")
        padded.paste(crop, (4, 4))
        np.testing.assert_array_equal(actual[index], capture.preprocess_glyph(padded))
        assert row["native_glyph_provenance"]["font_match_verified"] is True
        assert hashlib.sha256(actual[index].tobytes()).hexdigest() == row["glyph_sha256"]
        assert row["family"] == manifest["families"][row["target"]]
        assert row["actual_font_size_points"] == row["font_size_screen_px"] == 20
        assert row["text_color_hex"] == "#000000FF" and row["source_screen_scale"] == 1
    assert np.load(output / "test/glyphs.npy", allow_pickle=False).shape == (0, 64, 64)
    assert capture.sha(output / "train/glyphs.npy") == manifest["splits"]["train"]["glyphs"]["sha256"]
    assert all(value == 0 for value in manifest["split_isolation"].values())


def test_same_character_in_different_content_groups_can_cross_splits(tmp_path):
    labels, _, _ = dataset(tmp_path, texts=("青甲", "青乙", "青丙"), splits=("train", "calibration", "test"))
    manifest = capture.prepare(labels, tmp_path / "prepared")
    assert {split: row["rows"] for split, row in manifest["splits"].items()} == {"train": 2, "calibration": 2, "test": 2}


@pytest.mark.parametrize('variant', ['TC', 'HK'])
def test_regional_pingfang_identity_is_strict_and_not_sc_alias(tmp_path, variant):
    _, records, _ = dataset(tmp_path, texts=('帳單',))
    region = records[0]['regions'][0]
    family, ps = 'PingFang ' + variant, 'PingFang' + variant + '-Regular'
    region.update(font_family=family, requested_font_family=family, actual_font_family=family,
                  font_postscript=ps, requested_font_postscript=ps, actual_font_postscript=ps,
                  requested_language='zh-Hant')
    region['font_runs'][0].update(postscript_name=ps, family=family, language='zh-Hant')
    for glyph in region['glyphs']:
        glyph.update(font_postscript=ps, font_family=family, actual_font_family=family)
    assert capture.font_evidence(region, family, 'han') is None
    region['actual_font_family'] = 'PingFang SC'
    assert capture.font_evidence(region, family, 'han') == 'native_pingfang_regional_family_mismatch'
    region['actual_font_family'] = family
    region['font_runs'][0]['family'] = 'PingFang SC'
    assert capture.font_evidence(region, family, 'han') == 'native_pingfang_regional_run_mismatch'
    region['font_runs'][0]['family'] = family
    region['font_runs'][0]['language'] = 'zh-Hans'
    assert capture.font_evidence(region, family, 'han') == 'native_font_run_language_mismatch'


def test_screenshot_hash_tampering_is_fatal_and_no_partial_bundle_published(tmp_path):
    labels, rows, _ = dataset(tmp_path)
    with Path(rows[0]["image"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="screenshot SHA"):
        capture.prepare(labels, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("change", [
    lambda r: r.update(source_kind="pillow_render"),
    lambda r: r.update(pixel_size=[80, 100]),
    lambda r: r["native"].update(scene_manifest_sha256="b" * 64),
    lambda r: r["native"].update(renderer="desktop CoreText image"),
    lambda r: r.update(regions=[]),
])
def test_desktop_or_unbound_capture_provenance_is_rejected(tmp_path, change):
    labels, rows, _ = dataset(tmp_path)
    change(rows[0])
    write_labels(labels, rows)
    with pytest.raises(ValueError):
        capture.prepare(labels, tmp_path / "prepared")


@pytest.mark.parametrize("box", [[-1, 20, 35, 50], [15, 20, 105, 50], [15, 20, 15, 50], [15., 20, 35, 50], [5, 20, 35, 50]])
def test_invalid_or_neighboring_region_glyph_boxes_are_fatal(tmp_path, box):
    labels, rows, _ = dataset(tmp_path)
    rows[0]["regions"][0]["glyphs"][0]["bbox"] = box
    write_labels(labels, rows)
    with pytest.raises(ValueError, match="glyph bbox"):
        capture.prepare(labels, tmp_path / "prepared")


@pytest.mark.parametrize("change,reason", [
    (lambda r: r.update(fallback_detected=True), "native_font_fallback_or_missing_proof"),
    (lambda r: r["font_runs"][0].update(postscript_name="FallbackFont"), "native_font_run_mismatch"),
    (lambda r: r["font_runs"][0].update(glyph_ids=[999, 101]), "native_glyph_run_binding_mismatch"),
    (lambda r: r["glyph_coverage"].update(font_get_glyphs_succeeded=False), "native_glyph_coverage_unverified"),
    (lambda r: r.update(clipped=True), "native_region_clipped_or_missing_proof"),
])
def test_font_fallback_and_invalid_native_evidence_are_counted_not_trained(tmp_path, change, reason):
    labels, rows, _ = dataset(tmp_path, texts=("青甲", "青乙"), splits=("train", "train"))
    change(rows[1]["regions"][0])
    write_labels(labels, rows)
    manifest = capture.prepare(labels, tmp_path / "prepared")
    assert manifest["glyph_count"] == 2
    assert manifest["rejected"]["counts"][reason] == 1


def test_cross_split_source_id_is_rejected_even_with_different_screenshot_bytes(tmp_path):
    labels, rows, _ = dataset(tmp_path, texts=("青甲", "青乙"), splits=("train", "test"))
    rows[1]["source_id"] = rows[0]["source_id"]
    write_labels(labels, rows)
    with pytest.raises(ValueError, match="cross-split leakage: source_id"):
        capture.prepare(labels, tmp_path / "prepared")


def test_differently_encoded_identical_png_pixels_cannot_cross_splits(tmp_path):
    labels, rows, _ = dataset(tmp_path, texts=("青甲", "青乙"), splits=("train", "test"))
    with Image.open(rows[0]["image"]) as opened:
        image = opened.convert("RGB")
    info = PngImagePlugin.PngInfo()
    info.add_text("encoding", "different bytes, same RGB")
    image.save(rows[1]["image"], pnginfo=info, compress_level=1)
    rows[1]["source_sha256"] = capture.sha(Path(rows[1]["image"]))
    assert rows[0]["source_sha256"] != rows[1]["source_sha256"]
    write_labels(labels, rows)
    with pytest.raises(ValueError, match="decoded_pixel_sha256"):
        capture.prepare(labels, tmp_path / "prepared")


def test_identical_region_pixels_cannot_hide_inside_different_pages(tmp_path):
    labels, rows, _ = dataset(tmp_path, texts=("青甲", "青乙"), splits=("train", "test"))
    with Image.open(rows[0]["image"]) as opened:
        image = opened.convert("RGB")
    image.putpixel((0, 0), (255, 0, 0))
    image.save(rows[1]["image"])
    rows[1]["source_sha256"] = capture.sha(Path(rows[1]["image"]))
    write_labels(labels, rows)
    with pytest.raises(ValueError, match="region_rgb_sha256"):
        capture.prepare(labels, tmp_path / "prepared")


def test_stale_same_split_frame_cannot_receive_different_text_labels(tmp_path):
    labels, rows, _ = dataset(tmp_path, texts=("青甲", "青乙"), splits=("train", "train"))
    with Image.open(rows[0]["image"]) as opened:
        image = opened.convert("RGB")
    image.putpixel((0, 0), (255, 0, 0))
    image.save(rows[1]["image"])
    rows[1]["source_sha256"] = capture.sha(Path(rows[1]["image"]))
    write_labels(labels, rows)
    with pytest.raises(ValueError, match="conflicting font/text labels"):
        capture.prepare(labels, tmp_path / "prepared")


def test_normalized_region_content_is_partitioned_before_import(tmp_path):
    labels, _, _ = dataset(tmp_path, texts=("青甲", "青甲"), splits=("train", "test"))
    with pytest.raises(ValueError, match="normalized_region_text"):
        capture.prepare(labels, tmp_path / "prepared")


def test_missing_capture_protocol_cannot_be_called_native_capture(tmp_path):
    labels, _, _ = dataset(tmp_path)
    (tmp_path / "CAPTURE_PROTOCOL.json").unlink()
    with pytest.raises(ValueError, match="CAPTURE_PROTOCOL"):
        capture.prepare(labels, tmp_path / "prepared")


@pytest.mark.parametrize("changes", [{"font_size_screen_px": 64}, {"actual_font_size_points": None},
                                      {"source_screen_scale": 3}, {"text_color_hex": "black"}])
def test_resolved_native_size_and_color_must_be_bound_to_screen_geometry(tmp_path, changes):
    labels, rows, _ = dataset(tmp_path)
    rows[0]["regions"][0].update(changes)
    write_labels(labels, rows)
    with pytest.raises(ValueError):
        capture.prepare(labels, tmp_path / "prepared")


def asset_dataset(tmp_path):
    labels, rows, scenes = dataset(tmp_path)
    digest = "b" * 64
    scenes["fonts"][0].update(kind="asset", sha256=digest)
    scenes_path = tmp_path / "Scenes.json"
    scenes_path.write_text(json.dumps(scenes, ensure_ascii=False))
    scene_sha = capture.sha(scenes_path)
    protocol = json.loads((tmp_path / "CAPTURE_PROTOCOL.json").read_text())
    protocol["scenes_sha256"] = scene_sha
    (tmp_path / "CAPTURE_PROTOCOL.json").write_text(json.dumps(protocol))
    rows[0]["scenes_sha256"] = rows[0]["native"]["scene_manifest_sha256"] = scene_sha
    region = rows[0]["regions"][0]
    region["font_source"] = {"kind": "asset", "sha256": digest, "sha256_verified": True,
                             "registration_succeeded": True, "native_font_url_verified": True,
                             "postscript": "PingFangSC-Regular", "family": "PingFang SC",
                             "file_url": "file:///test/registered-font.ttf", "native_font_url": "file:///test/registered-font.ttf"}
    for glyph in region["glyphs"]:
        glyph.update(font_source_kind="asset", font_source_sha256=digest)
    write_labels(labels, rows)
    return labels, rows


def test_asset_font_registration_hash_and_native_url_provenance_are_retained(tmp_path):
    labels, _ = asset_dataset(tmp_path)
    output = tmp_path / "prepared"
    manifest = capture.prepare(labels, output)
    metadata = json.loads((output / "train/metadata.json").read_text())
    assert manifest["glyph_count"] == 2
    assert metadata["rows"][0]["native_font_source"]["sha256_verified"] is True
    assert metadata["rows"][0]["native_glyph_provenance"]["font_source_sha256"] == "b" * 64


@pytest.mark.parametrize("field,value", [("sha256", "c" * 64), ("sha256_verified", False),
                                         ("registration_succeeded", False), ("native_font_url_verified", False),
                                         ("native_font_url", "file:///test/fallback.ttf")])
def test_asset_font_fallback_or_unverified_registration_cannot_enter_training(tmp_path, field, value):
    labels, rows = asset_dataset(tmp_path)
    rows[0]["regions"][0]["font_source"][field] = value
    write_labels(labels, rows)
    with pytest.raises(ValueError, match="no verified usable native glyphs"):
        capture.prepare(labels, tmp_path / "prepared")
