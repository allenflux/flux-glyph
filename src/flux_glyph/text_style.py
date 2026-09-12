"""Measure visible text colour and estimate source-image pixel em size.

No source pixels are resized here. Size ratios are learned from the *train*
partition's original screenshot ink and known native em sizes. An interval is
an empirical rendering range, not a statistical confidence interval or iOS pt.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import math

import numpy as np
from PIL import Image

SCHEMA = "flux-glyph-text-size-metrics-v1"
METHOD = "source-rgb-ink-height-v1"
MAX_GLYPHS = 128
MAX_CROP_PIXELS = 4_000_000


def _box(value, size):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(type(x) is not int for x in value)):
        return None
    left, top, right, bottom = value
    if (0 <= left < right <= size[0] and 0 <= top < bottom <= size[1]
            and (right - left) * (bottom - top) <= MAX_CROP_PIXELS):
        return left, top, right, bottom
    return None


def _hex(rgb):
    return "#" + "".join(f"{int(x):02X}" for x in np.clip(np.rint(rgb), 0, 255))


def _background(image, box):
    values = np.asarray(image.crop(box).convert("RGB"), dtype=np.float32)
    border = np.concatenate((values[0], values[-1], values[:, 0], values[:, -1]))
    colour = np.median(border, axis=0)
    deviation = np.max(np.abs(border - colour), axis=1)
    # Most of the border must describe one local background. A few ink pixels
    # at a tight OCR boundary are tolerated; a gradient/photo is not.
    return colour, float(np.quantile(deviation, .8))


def measure_glyph_ink(image, bbox, *, background_rgb=None, background_bbox=None):
    """Measure a source glyph, retaining native pixel units; never normalise it.

    ``bbox`` is absolute in ``image``. The caller may supply a measured local
    background; otherwise a 3px surrounding ring is sampled. Public results are
    ordinary JSON-compatible values. Invalid/weak/multicolour ink is rejected.
    """
    box = _box(bbox, image.size)
    if box is None:
        return {"status": "unavailable", "reason": "invalid_source_bbox"}
    l, t, r, b = box
    if background_rgb is None:
        context = _box(background_bbox, image.size) or (max(0, l - 3), max(0, t - 3), min(image.width, r + 3), min(image.height, b + 3))
        bg, bg_spread = _background(image, context)
    else:
        bg = np.asarray(background_rgb, dtype=np.float32)
        if bg.shape != (3,) or not np.isfinite(bg).all() or np.any((bg < 0) | (bg > 255)):
            return {"status": "unavailable", "reason": "invalid_background"}
        bg_spread = 0.
    report = {"status": "unavailable", "reason": "insufficient_ink", "background_rgb": bg.round().astype(int).tolist(),
              "background_hex": _hex(bg), "background_spread_rgb": round(bg_spread, 2)}
    if bg_spread > 14:
        report["reason"] = "background_not_uniform"
        return report
    values = np.asarray(image.crop(box).convert("RGB"), dtype=np.float32)
    # RGB distance, not luminance: chromatic text can have the same luminance
    # as its background. Quantiles limit isolated JPEG/sensor outliers.
    delta = values - bg
    distance = np.linalg.norm(delta, axis=2)
    contrast = float(np.quantile(distance, .995))
    report["contrast_rgb"] = round(contrast, 2)
    if contrast < 24:
        report["reason"] = "low_contrast"
        return report
    mask = distance >= max(12., .22 * contrast)
    ys, xs = np.where(mask)
    if len(xs) < 10 or len(ys) == 0 or ys.max() - ys.min() + 1 < 5:
        return report
    core = values[(distance >= .93 * contrast) & (distance <= contrast + max(10., .06 * contrast))]
    if len(core) < 4:
        report["reason"] = "no_stable_core_pixels"
        return report
    colour = np.median(core, axis=0)
    spread = float(np.quantile(np.max(np.abs(core - colour), axis=1), .9))
    direction = colour - bg
    norm = float(np.dot(direction, direction))
    if norm < 24 ** 2:
        report["reason"] = "low_contrast"
        return report
    # Antialiased pixels should lie on the background-to-ink colour segment.
    # More than one foreground colour must not become an invented average HEX.
    foreground = values[mask]
    alpha = np.clip(np.sum((foreground - bg) * direction, axis=1) / norm, 0, 1)
    residual = np.max(np.abs(foreground - (bg + alpha[:, None] * direction)), axis=1)
    if spread > 18 or float((residual > 24).mean()) > .15:
        report["reason"] = "multiple_or_unstable_ink_colours"
        return report
    return {**report, "status": "estimated", "reason": "visible_core_pixels",
            "text_color_hex": _hex(colour), "colour_rgb": np.rint(colour).astype(int).tolist(),
            "spread_rgb": round(spread, 2), "core_pixels": len(core), "ink_pixels": len(xs),
            "ink_height_px": int(ys.max() - ys.min() + 1), "ink_width_px": int(xs.max() - xs.min() + 1),
            "ink_bbox": [l + int(xs.min()), t + int(ys.min()), l + int(xs.max()) + 1, t + int(ys.max()) + 1]}


def _character_group(character):
    if not isinstance(character, str) or len(character) != 1:
        return None
    if character.isascii() and character.isdigit():
        return "digits"
    if "A" <= character <= "Z":
        return "uppercase"
    if "a" <= character <= "z":
        if character in "bdfhklt":
            return "ascenders"
        if character in "gjpqy":
            return "descenders"
        return "xheight"
    return None  # Han heights vary too much (一 versus 国) for this fallback.


def build_size_metrics(records, *, provenance=None):
    """Fit compact ratios from measured, labelled *train* glyphs only.

    Each record needs split, family, character, source_id,
    observed_ink_height_px, and font_size_screen_px. Native bounding-box height
    is deliberately not accepted as a substitute for observed raster ink.
    """
    groups = defaultdict(list)
    for row in records:
        if row.get("split") != "train":
            raise ValueError("Size metrics may only use the train partition")
        family, char = row.get("family"), row.get("character")
        source = row.get("source_id")
        height, size = row.get("observed_ink_height_px"), row.get("font_size_screen_px")
        if (not isinstance(family, str) or not family or not isinstance(char, str) or len(char) != 1
                or not isinstance(source, str) or not source or isinstance(height, bool) or isinstance(size, bool)
                or not isinstance(height, (int, float)) or not isinstance(size, (int, float))
                or not math.isfinite(height) or not math.isfinite(size) or height < 8 or size <= 0):
            continue
        ratio = height / size
        if not .08 <= ratio <= 1.6:
            continue
        groups[(family, "characters", char)].append((ratio, source))
        group = _character_group(char)
        if group:
            groups[(family, "groups", group)].append((ratio, source))
    result = {"schema": SCHEMA, "measurement": METHOD, "split": "train", "families": {}, "provenance": provenance or {}}
    for (family, kind, name), values in sorted(groups.items()):
        sources = len({source for _, source in values})
        if len(values) < 3 or sources < 2:
            continue
        ratios = np.asarray([ratio for ratio, _ in values])
        low, median, high = np.quantile(ratios, [.05, .5, .95])
        result["families"].setdefault(family, {"characters": {}, "groups": {}})[kind][name] = {
            "q05_ratio": float(low), "median_ratio": float(median), "q95_ratio": float(high),
            "samples": len(values), "sources": sources}
    return result


class SizeMetrics:
    """Validated compact screenshot measurements, without runtime font files."""
    def __init__(self, value):
        if (not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("measurement") != METHOD
                or value.get("split") != "train" or not isinstance(value.get("families"), dict)
                or len(value["families"]) > 64):
            raise ValueError("Invalid text size metric contract")
        for family, info in value["families"].items():
            if not isinstance(family, str) or not family or not isinstance(info, dict):
                raise ValueError("Invalid size family")
            for kind in ("characters", "groups"):
                if not isinstance(info.get(kind), dict) or len(info[kind]) > 20000:
                    raise ValueError("Invalid size metric registry")
                for name, item in info[kind].items():
                    if not isinstance(name, str) or not name or not isinstance(item, dict):
                        raise ValueError("Invalid size metric record")
                    ratios = [item.get(key) for key in ("q05_ratio", "median_ratio", "q95_ratio")]
                    if (any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in ratios)
                            or not .08 <= ratios[0] <= ratios[1] <= ratios[2] <= 1.6
                            or type(item.get("samples")) is not int or item["samples"] < 3
                            or type(item.get("sources")) is not int or not 2 <= item["sources"] <= item["samples"]):
                        raise ValueError("Invalid size ratio/count")
        # Isolate the validated registry from callers mutating their JSON dict.
        self.value = json.loads(json.dumps(value, allow_nan=False))

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Size metrics exceed bounded registry")
        return cls(json.loads(path.read_text()))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")

    def lookup(self, family, character):
        info = self.value["families"].get(family, {})
        item = info.get("characters", {}).get(character)
        kind = "character"
        if item is None:
            item = info.get("groups", {}).get(_character_group(character))
            kind = "character_group"
        if item is None or (item["q95_ratio"] - item["q05_ratio"]) / item["median_ratio"] > .30:
            return None
        return {**item, "kind": kind}


def estimate_text_style(image, glyphs, *, family=None, metrics=None, region_bbox=None):
    """Estimate visible RGB and pixel em size using original absolute boxes.

    Font identity remains an assumption supplied by the font classifier. A
    known family is required for size; no system default or platform guess is
    substituted. For mixed scripts the caller may supply ``style_family`` on
    each glyph after accepting that script's font. Ordinary ``family_candidate``
    on glyphs is intentionally not promoted.
    """
    result = {"text_color_hex": None, "font_size_px_estimate": None, "font_size_px_interval": None,
              "color": {"status": "unavailable", "reason": "insufficient_source_glyphs"},
              "size": {"status": "unavailable", "reason": "font_family_unconfirmed" if not family else "size_metrics_unavailable",
                       "method": METHOD, "unit": "source_image_px", "family_assumed": family,
                       "glyph_count": 0, "interval_kind": "empirical_rendering_range"}}
    if not isinstance(image, Image.Image) or not isinstance(glyphs, (list, tuple)):
        return result
    box = _box(region_bbox, image.size)
    bg = None
    if box:
        bg, spread = _background(image, box)
        if spread > 14:
            result["color"] = {"status": "unavailable", "reason": "background_not_uniform", "background_spread_rgb": round(spread, 2)}
            return result
    measurements, seen = [], set()
    for glyph in glyphs[:MAX_GLYPHS]:
        if not isinstance(glyph, dict) or glyph.get("status", "ok") != "ok":
            continue
        glyph_box = _box(glyph.get("source_bbox"), image.size)
        if glyph_box is None or glyph_box in seen:
            continue
        seen.add(glyph_box)
        measured = measure_glyph_ink(image, glyph_box, background_rgb=bg)
        if measured["status"] == "estimated":
            measurements.append((glyph, measured))
    if not measurements:
        if box:
            # Colour does not depend on a covered font or successful character
            # segmentation. Use the OCR region itself only when its pixels pass
            # the same uniform-background/single-ink checks; no size is inferred.
            measured = measure_glyph_ink(image, box, background_rgb=bg)
            if measured["status"] == "estimated":
                result["text_color_hex"] = measured["text_color_hex"]
                result["color"] = {"status": "estimated", "reason": "visible_region_core_pixels",
                                   "method": "source_rgb_core_pixels", "background_hex": measured["background_hex"],
                                   "glyph_count": 0, "sample_pixels": measured["core_pixels"],
                                   "spread_rgb": measured["spread_rgb"], "alpha_recovered": False}
            else:
                result["color"] = {"status": "unavailable", "reason": measured["reason"]}
        return result
    colours = np.asarray([item["colour_rgb"] for _, item in measurements], dtype=np.float64)
    colour = np.median(colours, axis=0)
    dispersion = float(np.max(np.max(np.abs(colours - colour), axis=1)))
    if dispersion > 20:
        result["color"] = {"status": "unavailable", "reason": "multiple_text_colours", "spread_rgb": round(dispersion, 2)}
    else:
        result["text_color_hex"] = _hex(colour)
        result["color"] = {"status": "estimated", "reason": "visible_core_pixels", "method": "source_rgb_core_pixels",
                           "background_hex": measurements[0][1]["background_hex"], "glyph_count": len(measurements),
                           "sample_pixels": sum(item["core_pixels"] for _, item in measurements),
                           "spread_rgb": round(max(dispersion, max(item["spread_rgb"] for _, item in measurements)), 2),
                           "alpha_recovered": False}
    if not isinstance(metrics, SizeMetrics):
        return result
    estimates = defaultdict(list)
    used_families = set()
    for glyph, item in measurements:
        char = glyph.get("character")
        glyph_family = glyph.get("style_family", family)
        metric = metrics.lookup(glyph_family, char) if isinstance(char, str) and isinstance(glyph_family, str) else None
        height = item["ink_height_px"]
        if metric is None or height < 8:
            continue
        used_families.add(glyph_family)
        estimates[(glyph_family, char)].append((height / metric["median_ratio"], max(0, height - 1) / metric["q95_ratio"],
                                               (height + 1) / metric["q05_ratio"]))
    if not estimates:
        result["size"]["reason"] = "matching_size_metrics_absent_or_ambiguous"
        return result
    # Repeated digits do not outvote distinct glyphs. A mixed-size region must
    # not turn into a plausible but nonexistent average size.
    all_values = np.asarray([item for items in estimates.values() for item in items])
    values = np.asarray([np.median(items, axis=0) for items in estimates.values()])
    center, low, high = np.median(values, axis=0)
    spread = float(np.max(np.abs(all_values[:, 0] - center)) / center)
    result["size"].update(glyph_count=len(all_values), distinct_characters=len(values), relative_spread=round(spread, 4))
    result["size"]["families_assumed"] = sorted(used_families)
    if spread > .18 or (high - low) / center > .4:
        result["size"]["reason"] = "inconsistent_or_imprecise_glyph_sizes"
        return result
    result["font_size_px_estimate"] = round(float(center), 2)
    result["font_size_px_interval"] = [round(float(low), 2), round(float(high), 2)]
    result["size"].update(status="estimated", reason="calibrated_source_ink_height")
    return result
