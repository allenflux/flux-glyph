import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from flux_glyph.text_style import SizeMetrics, build_size_metrics, estimate_text_style, measure_glyph_ink

ROOT = Path(__file__).resolve().parents[1]


def render(text="苹方未", size=40, color="#173967", background="#F3F6F8", jpeg=False):
    # These are deterministic unit-test images; they are never training data.
    image = Image.new("RGB", (size * (len(text) + 2), size * 2), background)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(ROOT / "assets/annotation.otf"), size)
    glyphs = []
    for index, char in enumerate(text):
        pos = (size // 2 + index * size, size // 3)
        box = list(draw.textbbox(pos, char, font=font))
        draw.text(pos, char, font=font, fill=color)
        glyphs.append({"character": char, "source_bbox": box, "status": "ok"})
    if jpeg:
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=88, subsampling=0)
        image = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    return image, glyphs


@pytest.mark.parametrize("color,background", [("#173967", "#F3F6F8"), ("#F1D739", "#102132"),
                                               ("#737373", "#FFFFFF"), ("#FF0000", "#009000")])
def test_rgb_ink_colour_with_antialiasing_and_both_polarities(color, background):
    image, glyphs = render(color=color, background=background)
    value = estimate_text_style(image, glyphs, region_bbox=[0, 0, *image.size])
    assert value["text_color_hex"] == color
    assert value["font_size_px_estimate"] is None
    assert value["color"]["alpha_recovered"] is False
    json.dumps(value, allow_nan=False)


def test_jpeg_colour_tolerance_and_no_alpha_claim():
    image, glyphs = render(jpeg=True)
    value = estimate_text_style(image, glyphs, region_bbox=[0, 0, *image.size])
    assert value["text_color_hex"] is not None
    rgb = [int(value["text_color_hex"][i:i + 2], 16) for i in (1, 3, 5)]
    assert max(abs(a - b) for a, b in zip(rgb, (23, 57, 103))) <= 8


def trained_metrics():
    rows = []
    for i, size in enumerate((28, 36, 44, 52)):
        image, glyphs = render(size=size)
        for glyph in glyphs:
            measured = measure_glyph_ink(image, glyph["source_bbox"], background_bbox=[0, 0, *image.size])
            rows.append({"split": "train", "family": "Test Sans", "character": glyph["character"],
                         "source_id": f"train-{i}", "observed_ink_height_px": measured["ink_height_px"],
                         "font_size_screen_px": size})
    return SizeMetrics(build_size_metrics(rows))


def test_pixel_size_on_heldout_size_and_resized_original():
    metrics = trained_metrics()
    image, glyphs = render(size=61)
    value = estimate_text_style(image, glyphs, family="Test Sans", metrics=metrics, region_bbox=[0, 0, *image.size])
    assert value["font_size_px_estimate"] == pytest.approx(61, abs=2)
    assert value["font_size_px_interval"][0] <= 61 <= value["font_size_px_interval"][1]
    doubled = image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST)
    boxes = [{**g, "source_bbox": [v * 2 for v in g["source_bbox"]]} for g in glyphs]
    value2 = estimate_text_style(doubled, boxes, family="Test Sans", metrics=metrics, region_bbox=[0, 0, *doubled.size])
    assert value2["font_size_px_estimate"] == pytest.approx(value["font_size_px_estimate"] * 2, abs=.02)
    assert "pt" not in value2
    assert value2["size"]["unit"] == "source_image_px"


def test_unknown_family_or_unseen_han_does_not_invent_size():
    image, glyphs = render(text="一")
    for family in (None, "Unseen Sans", "Test Sans"):
        value = estimate_text_style(image, glyphs, family=family, metrics=trained_metrics())
        assert value["font_size_px_estimate"] is None


def test_low_contrast_blank_and_invalid_boxes_are_unavailable():
    image, glyphs = render(color="#F8F8F8", background="#FFFFFF")
    assert estimate_text_style(image, glyphs)["text_color_hex"] is None
    assert measure_glyph_ink(Image.new("RGB", (20, 20), "white"), [0, 0, 20, 20])["status"] == "unavailable"
    for box in ([0, 0, 100000, 100000], [-1, 0, 2, 2], [0, 0, 0, 4], [False, 0, 3, 4]):
        assert measure_glyph_ink(image, box)["reason"] == "invalid_source_bbox"


def test_multiple_colours_are_not_averaged_into_an_invented_hex():
    image, glyphs = render(text="苹方", color="#112233")
    other, _ = render(text="苹方", color="#F01836")
    box = glyphs[1]["source_bbox"]
    image.paste(other.crop(box), box)
    assert estimate_text_style(image, glyphs, region_bbox=[0, 0, *image.size])["text_color_hex"] is None


def test_mixed_sizes_abstain_and_duplicate_boxes_do_not_add_evidence():
    image, glyphs = render(size=40)
    big, bigger = render(size=72)
    canvas = Image.new("RGB", (image.width + big.width, big.height), "#F3F6F8")
    canvas.paste(image, (0, 0)); canvas.paste(big, (image.width, 0))
    shifted = [{**g, "source_bbox": [g["source_bbox"][0] + image.width, g["source_bbox"][1],
                                     g["source_bbox"][2] + image.width, g["source_bbox"][3]]} for g in bigger]
    value = estimate_text_style(canvas, glyphs + shifted, family="Test Sans", metrics=trained_metrics(), region_bbox=[0, 0, *canvas.size])
    assert value["font_size_px_estimate"] is None
    assert value["size"]["reason"] == "inconsistent_or_imprecise_glyph_sizes"
    duplicated = estimate_text_style(image, glyphs + glyphs, family="Test Sans", metrics=trained_metrics(), region_bbox=[0, 0, *image.size])
    assert duplicated["size"]["glyph_count"] == len(glyphs)


def test_metric_split_and_contract_validation(tmp_path):
    with pytest.raises(ValueError, match="train"):
        build_size_metrics([{"split": "test"}])
    metrics = trained_metrics()
    path = tmp_path / "metrics.json"
    metrics.save(path)
    assert SizeMetrics.load(path).lookup("Test Sans", "苹") == metrics.lookup("Test Sans", "苹")
    bad = json.loads(path.read_text())
    bad["families"]["Test Sans"]["characters"]["苹"]["median_ratio"] = float("nan")
    with pytest.raises(ValueError, match="ratio"):
        SizeMetrics(bad)


def test_only_explicit_accepted_style_family_can_estimate_mixed_script_size():
    image, glyphs = render()
    candidates = [{**glyph, "family_candidate": "Test Sans"} for glyph in glyphs]
    assert estimate_text_style(image, candidates, metrics=trained_metrics())["font_size_px_estimate"] is None
    accepted = [{**glyph, "style_family": "Test Sans"} for glyph in candidates]
    value = estimate_text_style(image, accepted, metrics=trained_metrics(), region_bbox=[0, 0, *image.size])
    assert value["font_size_px_estimate"] == pytest.approx(40, abs=2)
    assert value["size"]["families_assumed"] == ["Test Sans"]


def test_gradient_background_and_multicolour_within_one_glyph_are_rejected():
    image, glyphs = render(text="苹")
    gradient = np.tile(np.arange(image.width, dtype=np.uint8)[None, :, None], (image.height, 1, 3))
    varied = Image.fromarray(gradient)
    box = glyphs[0]["source_bbox"]
    varied.paste(image.crop(box), box)
    assert estimate_text_style(varied, glyphs, region_bbox=[0, 0, *image.size])["color"]["reason"] == "background_not_uniform"
    source = np.asarray(image).copy()
    foreground = np.max(np.abs(source.astype(float) - [243, 246, 248]), axis=2) > 50
    foreground[:, :(box[0] + box[2]) // 2] = False
    source[foreground] = [230, 20, 50]
    result = measure_glyph_ink(Image.fromarray(source), box, background_bbox=[0, 0, *image.size])
    assert result["status"] == "unavailable"


def test_region_colour_survives_unknown_font_and_failed_character_segmentation():
    image, glyphs = render()
    for characters in ([], [{**g, "status": "uncertain"} for g in glyphs]):
        value = estimate_text_style(image, characters, family=None, region_bbox=[0, 0, *image.size])
        assert value["text_color_hex"] == "#173967"
        assert value["color"]["glyph_count"] == 0
        assert value["font_size_px_estimate"] is None
