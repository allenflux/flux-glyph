import copy
from collections import Counter, defaultdict
import json
from pathlib import Path

import pytest

from training.capture import generate_android_short_scenes as scenes


ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "artifacts/android-font-v1/sources-v2/FONTS.json"
IOS_SCENES = ROOT / "artifacts/ios-native-short-train-v1/requests/Scenes.json"
IOS_CAPTURE = ROOT / "artifacts/ios-native-short-train-v1/capture"
REUSED = ROOT / "artifacts/unified-font-v1/data-v1"


@pytest.fixture(scope="module")
def generated():
    registry, fonts, maps, bindings = scenes.load_sources(FONTS)
    ios = scenes.read(IOS_SCENES)
    capture, capture_bindings, capture_audit = scenes.load_ios_capture(
        IOS_CAPTURE, IOS_SCENES, ios)
    forbidden, heldout_bindings, _ = scenes.heldout_hashes(REUSED)
    bindings.update(heldout_bindings)
    bindings.update(capture_bindings)
    bindings[str(IOS_SCENES.resolve())] = scenes.sha(IOS_SCENES)
    result = scenes.build_scenes(registry, fonts, maps, ios, capture, forbidden,
                                 bindings, capture_audit)
    return result, maps, forbidden


def test_train_only_plan_has_exact_bounded_coverage(generated):
    result, _, _ = generated
    rows = [row for page in result["pages"] for row in page["regions"]]
    design = result["design"]
    assert result["families"] == list(scenes.KNOWN_FAMILIES) + ["__unknown__"]
    assert len(result["pages"]) == design["planned_pages"] == 128
    assert len(rows) == design["planned_regions"] == 1056
    assert design["planned_content_groups"] == 64
    assert design["pages_per_content_group"] == 2
    assert design["all_pages_training_only"] is True
    assert design["test_read"] is False
    assert {page["split"] for page in result["pages"]} == {"train"}
    assert Counter(len(page["regions"]) for page in result["pages"]) == {7: 32, 8: 32, 9: 64}
    assert Counter(len(row["text"]) for row in rows) == {1: 264, 2: 264, 3: 264, 4: 264}
    assert Counter(row["script"] for row in rows) == {"han": 480, "latin": 576}
    assert Counter(row["text_kind"] for row in rows) == {"han": 480, "english": 288, "numeric": 288}


def test_two_page_groups_pair_text_and_style_across_every_eligible_family(generated):
    result, _, _ = generated
    fonts = {font["id"]: font for font in result["fonts"]}
    groups = defaultdict(list)
    for page in result["pages"]:
        groups[page["content_group_id"]].append(page)
    assert len(groups) == 64
    for pages in groups.values():
        assert len(pages) == 2
        rows = [row for page in pages for row in page["regions"]]
        script = rows[0]["script"]
        expected = (set(scenes.KNOWN_FAMILIES[:-1]) | set(scenes.PROMOTED_UNIFIED_FAMILIES)
                    | {"Liu Jian Mao Cao", "Smiley Sans"} if script == "han"
                    else set(scenes.KNOWN_FAMILIES) | set(scenes.PROMOTED_UNIFIED_FAMILIES)
                    | set(scenes.UNKNOWN_SOURCE_FAMILIES))
        assert {fonts[row["font_id"]]["family"] for row in rows} == expected
        assert len({fonts[row["font_id"]]["family"] for row in rows}) == len(rows)
        assert len({(row["text"], row["font_size_px"], row["ios_font_size_points"],
                     row["color"], row["text_kind"],
                     row["paired_style_id"]) for row in rows}) == 1
        assert len({page["background"] for page in pages}) == 1


def test_only_referenced_train_sources_are_bound_and_unified_mapping_is_explicit(generated):
    result, _, _ = generated
    fonts = result["fonts"]
    referenced = {row["font_id"] for page in result["pages"] for row in page["regions"]}
    assert referenced == {font["id"] for font in fonts}
    actual = {font["family"] for font in fonts}
    assert actual == (set(scenes.KNOWN_FAMILIES) | set(scenes.PROMOTED_UNIFIED_FAMILIES)
                      | set(scenes.UNKNOWN_SOURCE_FAMILIES))
    assert not actual.intersection(scenes.EXPLICITLY_EXCLUDED_FAMILIES)
    mapping = result["design"]["source_family_to_unified_family"]
    assert all(mapping[family] == family for family in scenes.PROMOTED_UNIFIED_FAMILIES)
    assert all(mapping[family] == "__unknown__" for family in scenes.UNKNOWN_SOURCE_FAMILIES)
    promoted = [font for font in fonts if font["family"] in scenes.PROMOTED_UNIFIED_FAMILIES]
    true_unknown = [font for font in fonts if font["family"] in scenes.UNKNOWN_SOURCE_FAMILIES]
    assert promoted and true_unknown
    assert all(font["training_family"] == "__unknown__" and font["split"] == "train"
               and font["allowed_splits"] == ["train"] for font in promoted + true_unknown)
    bindings = result["bindings"]
    for font in fonts:
        assert bindings[str(Path(font["path"]).resolve())] == font["sha256"]
        assert bindings[str(Path(font["cmap"]["path"]).resolve())] == font["cmap"]["sha256"]
        for license_file in font["license_files"]:
            assert bindings[str(Path(license_file["path"]).resolve())] == license_file["sha256"]


def test_source_family_length_kind_and_face_balance_has_no_font_text_confound(generated):
    result, maps, _ = generated
    fonts = {font["id"]: font for font in result["fonts"]}
    coverage = Counter()
    kinds = Counter()
    faces = defaultdict(Counter)
    for page in result["pages"]:
        for row in page["regions"]:
            font = fonts[row["font_id"]]
            key = font["family"], row["script"], len(row["text"])
            coverage[key] += 1
            kinds[key + (row["text_kind"],)] += 1
            faces[key][font["id"]] += 1
            assert all(ord(character) in maps[font["id"]] for character in row["text"])
            assert row["script"] != "han" or "han" in font["scripts"]
    for family in set(scenes.KNOWN_FAMILIES) | set(scenes.PROMOTED_UNIFIED_FAMILIES) | set(scenes.UNKNOWN_SOURCE_FAMILIES):
        scripts = ("latin",) if family in {"Roboto", "Lato", "Open Sans"} else ("han", "latin")
        for script in scripts:
            for length in range(1, 5):
                key = family, script, length
                assert coverage[key] == 8
                assert max(faces[key].values()) - min(faces[key].values()) <= 1
                if script == "latin":
                    assert kinds[key + ("english",)] == 4
                    assert kinds[key + ("numeric",)] == 4


def test_ios_point_sizes_use_verified_native_screen_pixels_and_faces_cross_kinds(generated):
    result, _, _ = generated
    rows = [row for page in result["pages"] for row in page["regions"]]
    assert result["design"]["matched_size_unit"] == "actual iOS native screen pixels"
    assert result["design"]["ios_point_to_screen_scale"] == 3
    assert {row["font_size_px"] for row in rows} == {36, 45, 54, 66, 81, 96, 108, 120}
    assert all(row["font_size_px"] == row["ios_font_size_points"] * 3 for row in rows)
    assert all(row["font_size_px"] == 36 for row in rows if row["ios_font_size_points"] == 12)
    assert all(row["ios_capture_source_id"] == "ios:" + row["ios_source_page_id"] for row in rows)
    assert all(Path(row["ios_capture_frame_path"]).is_file()
               and scenes.sha(row["ios_capture_frame_path"]) == row["ios_capture_frame_sha256"]
               for row in rows)

    fonts = {font["id"]: font for font in result["fonts"]}
    kinds_by_face = defaultdict(set)
    for row in rows:
        family = fonts[row["font_id"]]["family"]
        if family in {"Roboto", "Open Sans"}:
            kinds_by_face[row["font_id"]].add(row["text_kind"])
    expected_faces = {font["id"] for font in result["fonts"]
                      if font["family"] in {"Roboto", "Open Sans"}}
    assert set(kinds_by_face) == expected_faces
    assert all(kinds == {"english", "numeric"} for kinds in kinds_by_face.values())


def test_generation_is_deterministic_and_validation_fails_closed(generated):
    result, maps, forbidden = generated
    registry, fonts, fresh_maps, _ = scenes.load_sources(FONTS)
    ios = scenes.read(IOS_SCENES)
    capture, _, capture_audit = scenes.load_ios_capture(IOS_CAPTURE, IOS_SCENES, ios)
    rebuilt = scenes.build_scenes(registry, fonts, fresh_maps, ios, capture, forbidden,
                                  {}, capture_audit)
    expected = copy.deepcopy(result)
    expected["bindings"] = {}
    assert rebuilt == expected

    contaminated = copy.deepcopy(result)
    group_id = contaminated["pages"][0]["content_group_id"]
    for page in contaminated["pages"]:
        if page["content_group_id"] == group_id:
            for row in page["regions"]:
                row["text"] = "\U0010ffff"
    with pytest.raises(ValueError, match="unsupported source glyph"):
        scenes.validate_short_scenes(contaminated, maps, forbidden)
    first_text = result["pages"][0]["regions"][0]["text"]
    with pytest.raises(ValueError, match="CAL/DEV full-text hash reused"):
        scenes.validate_short_scenes(result, maps, {*forbidden, scenes.text_sha(first_text)})


def test_cli_writes_new_scene_and_request_manifests(tmp_path):
    output = tmp_path / "requests"
    scenes.main(["--fonts", str(FONTS), "--ios-scenes", str(IOS_SCENES),
                 "--ios-capture", str(IOS_CAPTURE),
                 "--reused-data", str(REUSED), "--output", str(output)])
    scene_path, report_path = output / "Scenes.json", output / "REQUESTS.json"
    result, report = json.loads(scene_path.read_text()), json.loads(report_path.read_text())
    assert report["pages"] == result["design"]["planned_pages"] == 128
    assert report["regions"] == result["design"]["planned_regions"] == 1056
    assert report["scenes_sha256"] == scenes.sha(scene_path)
    assert report["test_read"] is False
    assert report["native_font_labels_verified"] is False
    assert report["full_text_cal_dev_hash_overlap"] == 0
    assert report["unified_family_counts"]["__unknown__"] == 192
    assert report["corpus_audit"]["selected_reused_ios_examples"] == 64
    assert report["corpus_audit"]["omitted_unsupported_cmap"] == 17
    assert report["matched_screen_pixel_size_counts"]["36"] > 0
    assert result["design"]["ios_capture_audit"] == {
        "pages": 59, "regions": 704, "pixel_size": [1206, 2622], "screen_scale": 3}
    with pytest.raises(ValueError, match="overwrite"):
        scenes.main(["--fonts", str(FONTS), "--ios-scenes", str(IOS_SCENES),
                     "--ios-capture", str(IOS_CAPTURE),
                     "--reused-data", str(REUSED), "--output", str(output)])
