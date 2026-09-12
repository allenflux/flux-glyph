"""Portable Latin matching regressions; font files are never runtime inputs."""
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from flux_glyph.latin_matcher import ALPHABET, CompactLatinBank, latin_raster
from flux_glyph.ppocr import PPReader

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/latin_accuracy"
CASES = json.loads((FIXTURES / "cases.json").read_text())["cases"]


@pytest.fixture(scope="module")
def bank():
    value = CompactLatinBank(ROOT / "models/latin")
    yield value
    value.archive.close()


def case(name):
    value = next(row for row in CASES if row["id"] == name)
    path = FIXTURES / value["image"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == value["image_sha256"]
    return value, Image.open(path).convert("RGB")


@pytest.mark.parametrize("name,family", [
    ("sfpro_17_semibold_38_0", "SF Pro"), ("sfpro_17_semibold_38_1", "SF Pro"),
    ("sfpro_17_semibold_38_2", "SF Pro"), ("sfpro_17_semibold_38_3", "SF Pro"),
    ("alipay_number_regular_38_0", "Alipay Number"), ("misans_regular_38_1", "MiSans"),
    ("harmonyos_regular_38_1", "HarmonyOS Sans SC"), ("oppo_regular_38_1", "OPPO Sans"),
])
def test_known_fonts_produce_candidate_without_platform_certification(bank, name, family):
    row, image = case(name)
    result = bank.score(image, row["text"], [], segmentation=row["segmentation"])
    assert result["status"] == "candidate"
    assert result["family"] == family
    assert result["evidence"]["candidate_only"]
    assert result["evidence"]["complete"]
    assert result["glyphs"]
    json.dumps(result)  # public API must not leak PIL/numpy values


@pytest.mark.parametrize("name", [row["id"] for row in CASES if row["expected_family"] is None])
def test_unknown_serif_and_monospace_are_not_promoted(bank, name):
    row, image = case(name)
    result = bank.score(image, row["text"], [], segmentation=row["segmentation"])
    assert result["status"] == "uncertain"
    assert result["family"] is None


@pytest.mark.parametrize("name", ["sfpro_17_semibold_38_0", "sfpro_17_semibold_38_1", "sfpro_17_semibold_38_2",
                                  "sfpro_17_semibold_38_3", "alipay_number_regular_38_0", "oppo_regular_38_1"])
def test_real_ocr_ctc_and_pixel_segmentation_feed_latin_candidates(bank, name):
    row, image = case(name)
    reader = PPReader(ROOT / "models/pp")
    reading = reader.read([image])[0]
    assert reading["text"].strip() == row["text"]
    result = bank.score(image, reading["text"], reading["tokens"], reading["metadata"])
    assert result["status"] == "candidate", result["reason"]
    assert result["family"] == row["expected_family"]
    assert all(glyph["status"] == "ok" for glyph in result["glyphs"])
    assert not result["segmentation"]["diagnostics"]["equal_width_fallback_used"]


def test_single_distinct_digit_and_missing_boxes_stay_uncertain(bank):
    row, image = case("alipay_number_regular_38_0")
    glyph = next(glyph for glyph in row["segmentation"]["characters"] if glyph["character"] == "0")
    only = {**glyph, "index": 0}
    result = bank.score(image, "0", [], segmentation={"characters": [only]})
    assert result["status"] == "uncertain" and result["family"] is None
    assert result["reason"] == "too_few_distinct_latin_characters"
    incomplete = {"characters": [{**glyph, "status": "uncertain", "bbox": None} for glyph in row["segmentation"]["characters"]]}
    result = bank.score(image, row["text"], [], segmentation=incomplete)
    assert result["status"] == "uncertain" and result["family"] is None
    assert result["reason"] == "latin_segmentation_uncertain"


def test_mixed_line_keeps_latin_indices_without_claiming_han(bank):
    row, image = case("sfpro_17_semibold_38_3")
    text = row["text"] + "浏览器"
    segmented = {"characters": row["segmentation"]["characters"] + [
        {"index": 6 + index, "character": character, "status": "uncertain", "bbox": None}
        for index, character in enumerate("浏览器")]}
    result = bank.score(image, text, [], segmentation=segmented)
    assert result["status"] == "candidate" and result["family"] == "SF Pro"
    assert [glyph["character"] for glyph in result["glyphs"]] == list("Safari")
    assert bank.score(image, "浏览器", [])["status"] == "unsupported_script"


def test_low_contrast_blank_and_tiny_source_ink_are_rejected():
    assert latin_raster(Image.new("RGB", (32, 32), "white")) is None
    assert latin_raster(Image.new("RGB", (2, 9), "black")) is None
    data = np.full((32, 32), 240, dtype=np.uint8)
    data[8:24, 12:20] = 230
    assert latin_raster(Image.fromarray(data)) is None


def test_cache_is_bounded_and_alipay_missing_letters_are_not_font_fallback():
    bank = CompactLatinBank(ROOT / "models/latin", max_characters=2)
    try:
        for character in ALPHABET:
            bank.references(character)
        assert len(bank.cache) == 2
        assert bank.cache_bytes == 2 * len(bank.font_ids) * 3 * 1024 * 4
        index = bank.font_ids.index("alipay_number_regular")
        assert np.all(bank.references("A")[index] == 0)
        assert np.any(bank.references("0")[index])
    finally:
        bank.archive.close()


def copy_bank(tmp_path):
    directory = tmp_path / "latin"
    shutil.copytree(ROOT / "models/latin", directory)
    return directory


def change_metadata(directory, change):
    path = directory / "metadata.json"
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("field,value", [("archive", "../references.zip"), ("shape", [17, 3, 32, 8192]),
                                        ("characters", {"A": "../A.npy"}), ("face_characters", {})])
def test_invalid_model_contract_is_rejected(tmp_path, field, value):
    directory = copy_bank(tmp_path)
    change_metadata(directory, lambda metadata: metadata.__setitem__(field, value))
    with pytest.raises(ValueError):
        CompactLatinBank(directory)


@pytest.mark.parametrize("name", ["references.zip", "GATES.json"])
def test_asset_checksum_is_verified_before_use(tmp_path, name):
    directory = copy_bank(tmp_path)
    with (directory / name).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        CompactLatinBank(directory)


@pytest.mark.parametrize("mutation", ["unexpected_member", "duplicate_member", "wrong_shape", "empty_vector", "trailing_bytes"])
def test_archive_members_and_numpy_headers_are_bounded(tmp_path, mutation):
    directory = copy_bank(tmp_path)
    archive = directory / "references.zip"
    with zipfile.ZipFile(archive) as handle:
        members = {info.filename: handle.read(info) for info in handle.infolist()}
    first = next(iter(members))
    if mutation in {"wrong_shape", "empty_vector"}:
        old = np.load(io.BytesIO(members[first]), allow_pickle=False)
        data = np.zeros((1, 1, 1, 1), dtype=np.uint8) if mutation == "wrong_shape" else old.copy()
        if mutation == "empty_vector":
            data[0, 0] = 0
        encoded = io.BytesIO()
        np.save(encoded, data, allow_pickle=False)
        members[first] = encoded.getvalue()
    elif mutation == "trailing_bytes":
        members[first] += b"extra"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for name, data in members.items():
            handle.writestr(name, data)
        if mutation == "unexpected_member":
            handle.writestr("../unexpected.npy", b"bad")
        elif mutation == "duplicate_member":
            with pytest.warns(UserWarning):
                handle.writestr(first, members[first])
    change_metadata(directory, lambda metadata: metadata.__setitem__("archive_sha256", hashlib.sha256(archive.read_bytes()).hexdigest()))
    with pytest.raises(ValueError):
        CompactLatinBank(directory)
