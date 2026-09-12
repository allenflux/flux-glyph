"""Conservative character localisation from CTC anchors and source-pixel ink.

CTC steps provide an approximate ordering and horizontal search window only. They
are never exposed as glyph boxes: every ``ok`` box has two image-derived
vertical cuts and an image-derived vertical ink extent. This module deliberately
returns ``uncertain`` rather than manufacturing an equal-width segmentation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import cv2
from PIL import Image


def _han(value: str) -> bool:
    return len(value) == 1 and "\u3400" <= value <= "\u9fff"


def _latin(value: str) -> bool:
    return len(value) == 1 and value.isascii() and value.isalnum()


def _token_by_text_index(text: str, tokens: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Return only unambiguous, index-matched CTC alignments."""
    # Some callers preserve spaces which the recogniser did not emit. Only
    # reconcile this exact omission; never shift anchors past a missing glyph,
    # reorder claims, or hide duplicate token indices.
    visible_indices = [i for i, character in enumerate(text) if not character.isspace()]
    if len(visible_indices) < len(text) and len(tokens) == len(visible_indices):
        if all(isinstance(token, dict) and token.get("index") == i and
               token.get("character") == text[visible_indices[i]]
               for i, token in enumerate(tokens)):
            tokens = [{**token, "index": visible_indices[i], "ctc_original_index": i,
                       "text_index_reconciled": "omitted_whitespace"}
                      for i, token in enumerate(tokens)]
    result: dict[int, dict[str, Any]] = {}
    for token in tokens:
        if not isinstance(token, dict):
            continue
        try:
            index = int(token["index"])
            start, end, peak = int(token["start_step"]), int(token["end_step"]), int(token["peak_step"])
            confidence = float(token["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= index < len(text) and token.get("character") == text[index] and 0 <= start <= peak < end and confidence >= 0):
            continue
        if index not in result:
            result[index] = {**token, "index": index, "start_step": start, "end_step": end,
                             "peak_step": peak, "confidence": confidence}
        else:
            result[index] = {}  # duplicate CTC claims are ambiguous
    return result


def _ink(image: Image.Image) -> tuple[np.ndarray, float, bool, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    # Three channels do not need a BLAS matrix multiply. Elementwise luma also
    # avoids platform BLAS floating-point flags leaking into this small raster.
    gray = .299 * rgb[:, :, 0] + .587 * rgb[:, :, 1] + .114 * rgb[:, :, 2]
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    background = float(np.median(border))
    dark_range = background - float(np.percentile(gray, 1))
    light_range = float(np.percentile(gray, 99)) - background
    dark = dark_range >= light_range
    departure = np.maximum(background - gray, 0.0) if dark else np.maximum(gray - background, 0.0)
    contrast = float(np.percentile(departure, 99.5))
    if contrast <= 0:
        return np.zeros_like(gray), background, dark, contrast
    return np.where(departure >= max(8.0, .18 * contrast), departure / contrast, 0.0), background, dark, contrast


def _metadata_scale(metadata: dict[str, Any] | None, width: int) -> tuple[float | None, float, list[str]]:
    """Map network CTC steps to original ROI pixels.

    PP recognition gives ``input_width`` and ``content_width`` in its resized
    48px-high network raster, not in the original ROI.  A CTC step spans
    ``input_width / timesteps`` network pixels.  The original-pixel scale is
    therefore ``input_width/timesteps * original_width/content_width``.
    """
    metadata = metadata or {}
    notes: list[str] = []
    try:
        steps = int(metadata["timesteps"])
        input_width = float(metadata["input_width"])
        content_width = float(metadata["content_width"])
        left = float(metadata.get("padding_left", 0.0))
    except (KeyError, TypeError, ValueError):
        return None, 0.0, ["invalid_rec_metadata"]
    if steps <= 0 or input_width <= 0 or content_width <= 0 or left < 0 or left + content_width > input_width + 1e-6:
        return None, 0.0, ["missing_or_invalid_timestep_mapping"]
    return (input_width / steps) * (width / content_width), left, notes


def _foreground_components(ink: np.ndarray) -> np.ndarray:
    """Discard isolated pixel noise while retaining every plausible stroke."""
    binary = (ink > 0).astype(np.uint8)
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    minimum_area = max(2, int(round(binary.shape[0] * .06)))
    keep = np.zeros(count, dtype=bool)
    for label in range(1, count):
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        keep[label] = area >= minimum_area and component_width >= 1 and component_height >= 2
    return keep[labels]


def _low_runs(mass: np.ndarray, limit: float) -> list[tuple[int, int]]:
    """Return half-open runs whose source-pixel ink stays below ``limit``."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for x, low in enumerate(mass <= limit):
        if low and start is None:
            start = x
        elif not low and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, len(mass)))
    return runs


def _gap_boundary(
    mass: np.ndarray,
    runs: list[tuple[int, int]],
    left_anchor: tuple[float, float, float],
    right_anchor: tuple[float, float, float],
    limit: float,
    minimum_width: int,
) -> tuple[int | None, bool, dict[str, Any]]:
    """Choose an image gap ordered between two coarse CTC token intervals."""
    left_start, left_end, _ = left_anchor
    right_start, right_end, _ = right_anchor
    target = (left_end + right_start) / 2.0
    primary = (min(left_end, right_start), max(left_end, right_start))
    expanded = (left_start, right_end)

    def candidates(window: tuple[float, float]) -> list[tuple[int, int]]:
        lo, hi = window
        return [run for run in runs if run[1] - run[0] >= minimum_width and run[1] > lo and run[0] < hi]

    choices = candidates(primary)
    search_kind = "between_ctc_intervals"
    search = primary
    if not choices:
        choices = candidates(expanded)
        search_kind = "between_ctc_tokens_expanded"
        search = expanded
    if not choices:
        return None, False, {
            "target": target,
            "search": [max(0.0, search[0]), min(float(len(mass)), search[1])],
            "search_kind": search_kind,
            "limit": limit,
            "minimum_gap_width": minimum_width,
            "low_ink_gap": False,
        }
    best = min(
        choices,
        key=lambda run: (
            abs(((run[0] + run[1]) / 2.0) - target),
            -(run[1] - run[0]),
            float(mass[run[0]:run[1]].max(initial=0.0)),
            run[0],
        ),
    )
    cut = int(round((best[0] + best[1]) / 2.0))
    return cut, True, {
        "target": target,
        "search": [max(0.0, search[0]), min(float(len(mass)), search[1])],
        "search_kind": search_kind,
        "gap": [best[0], best[1]],
        "mass_max": float(mass[best[0]:best[1]].max(initial=0.0)),
        "limit": limit,
        "minimum_gap_width": minimum_width,
        "low_ink_gap": True,
    }


def segment_characters(image: Image.Image, text: str, tokens: list[dict[str, Any]], *,
                       metadata: dict[str, Any] | None = None, segment_latin: bool = False) -> dict[str, Any]:
    """Locate each recognised character in an upright original-pixel ROI.

    ``tokens`` use ``character,index,start_step,end_step,peak_step,confidence``.
    ``metadata`` must supply network-raster ``timesteps``, ``input_width`` and
    ``content_width`` for any ``ok`` output; optional padding is in that same
    network-pixel space. CTC intervals are coarse ordering anchors, never output
    boxes. One record is returned for every character in ``text``. By default
    only Han characters receive boxes; ``segment_latin=True`` also enables
    ASCII letters and digits. Punctuation remains an ordering neighbour, while
    whitespace carries no ink and is skipped when locating neighbouring cuts.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if not isinstance(text, str) or not isinstance(tokens, list):
        raise TypeError("text must be str and tokens must be a list")
    width, height = image.size
    scale, padding_left, notes = _metadata_scale(metadata, width)
    indexed = _token_by_text_index(text, tokens)
    ink, background, dark, contrast = _ink(image)
    foreground = _foreground_components(ink)
    mass = np.where(foreground, ink, 0.0).sum(axis=0)
    peak = float(mass.max()) if mass.size else 0.0
    threshold = max(.15, .012 * peak)
    minimum_gap_width = max(2, int(round(height * .04)))
    gaps = _low_runs(mass, threshold)
    # Latin glyphs often have only one source-pixel column of inter-letter
    # whitespace. Such a gap is safe only when it contains no retained stroke.
    latin_gaps = _low_runs(mass, 0.0) if segment_latin else []
    anchors: dict[int, tuple[float, float, float]] = {}
    if scale is not None:
        for index, token in indexed.items():
            if not token or text[index].isspace():
                continue
            # Convert steps into resized network x then back to original ROI x.
            # A token touching padding is not a source-pixel anchor.
            start_network = token["start_step"] * (float(metadata["input_width"]) / int(metadata["timesteps"]))
            end_network = token["end_step"] * (float(metadata["input_width"]) / int(metadata["timesteps"]))
            content_width = float(metadata["content_width"])
            if padding_left <= start_network < end_network <= padding_left + content_width + 1e-6:
                start = (start_network - padding_left) * width / content_width
                end = (end_network - padding_left) * width / content_width
                # Retain a token-interval midpoint for diagnostics.  The CTC
                # peak is deliberately not interpreted as a glyph centre.
                center = (start + end) / 2.0
                anchors[index] = (start, end, center)

    cuts: dict[int, tuple[int | None, bool, dict[str, Any]]] = {}
    left_cuts: dict[int, tuple[int | None, bool, dict[str, Any]]] = {}
    valid_anchor_indices = sorted(anchors)
    for left_index, right_index in zip(valid_anchor_indices, valid_anchor_indices[1:]):
        if any(not character.isspace() for character in text[left_index + 1:right_index]):
            continue
        left_anchor, right_anchor = anchors[left_index], anchors[right_index]
        if right_anchor[0] <= left_anchor[0]:
            continue
        latin_boundary = segment_latin and (_latin(text[left_index]) or _latin(text[right_index]))
        cut, ok, evidence = _gap_boundary(
            mass, latin_gaps if latin_boundary else gaps, left_anchor, right_anchor,
            0.0 if latin_boundary else threshold, 1 if latin_boundary else minimum_gap_width)
        evidence["between"] = [left_index, right_index]
        if right_index > left_index + 1:
            evidence["skipped_whitespace_indices"] = list(range(left_index + 1, right_index))
        cuts[left_index] = (cut, ok, evidence)
        left_cuts[right_index] = (cut, ok, evidence)

    edges: dict[tuple[int, str], tuple[int | None, bool, dict[str, Any]]] = {}
    visible_indices = [i for i, character in enumerate(text) if not character.isspace()]
    first_index = visible_indices[0] if visible_indices else None
    last_index = visible_indices[-1] if visible_indices else None
    ink_columns = np.flatnonzero(foreground.any(axis=0))
    if len(ink_columns):
        ink_left, ink_right = int(ink_columns[0]), int(ink_columns[-1]) + 1
        if first_index in anchors:
            ok = ink_left > 0
            edges[(first_index, "left")] = (ink_left if ok else None, ok, {
                "edge": "left", "foreground_extent": [ink_left, ink_right],
                "background_margin": ink_left, "touches_roi_edge": not ok,
            })
        if last_index in anchors:
            ok = ink_right < width
            edges[(last_index, "right")] = (ink_right if ok else None, ok, {
                "edge": "right", "foreground_extent": [ink_left, ink_right],
                "background_margin": width - ink_right, "touches_roi_edge": not ok,
            })

    output: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        item: dict[str, Any] = {"character": character, "index": index, "bbox": None, "status": "uncertain"}
        token = indexed.get(index)
        if character.isspace():
            item["reason"] = "whitespace_preserved_not_font_segmented"
        elif token == {}:
            item["reason"] = "ambiguous_duplicate_ctc_token"
        elif token is None:
            item["reason"] = "ctc_token_missing_or_mismatched"
        elif scale is None:
            item["reason"] = notes[0]
        elif token["confidence"] < .8:
            item.update({"reason": "low_ctc_token_confidence", "ctc_anchor": list(anchors.get(index, ()))})
        elif index not in anchors:
            item["reason"] = "ctc_anchor_outside_content_width"
        elif not (_han(character) or (segment_latin and _latin(character))):
            item.update({"reason": "non_han_token_preserved_not_font_segmented", "ctc_anchor": list(anchors[index])})
        else:
            left_evidence = edges.get((index, "left")) if index == first_index else left_cuts.get(index)
            right_evidence = edges.get((index, "right")) if index == last_index else cuts.get(index)
            if left_evidence is None or right_evidence is None:
                item.update({"reason": "missing_neighbour_ctc_anchor", "ctc_anchor": list(anchors[index])})
            elif not left_evidence[1] or not right_evidence[1]:
                edge_touch = any(e[2].get("touches_roi_edge") for e in (left_evidence, right_evidence))
                item.update({"reason": "foreground_touches_roi_edge" if edge_touch else "no_low_ink_boundary", "ctc_anchor": list(anchors[index]),
                             "boundary_evidence": [left_evidence[2], right_evidence[2]]})
            else:
                left, right = left_evidence[0], right_evidence[0]
                assert left is not None and right is not None
                glyph_ink = foreground[:, max(0, left):min(width, right)]
                ys = np.flatnonzero(glyph_ink.any(axis=1))
                if right - left < 3 or len(ys) < 2:
                    item.update({"reason": "insufficient_character_ink", "ctc_anchor": list(anchors[index])})
                elif _latin(character) and len(_low_runs(~glyph_ink.any(axis=0), 0.0)) > 1:
                    # A skipped OCR symbol or a nearby status-bar icon must not
                    # be absorbed into a digit. Require a contiguous horizontal
                    # ink projection; disconnected decorative styles abstain.
                    item.update({"reason": "multiple_ink_groups_for_latin_token", "ctc_anchor": list(anchors[index]),
                                 "boundary_evidence": [left_evidence[2], right_evidence[2]]})
                else:
                    top, bottom = int(ys[0]), int(ys[-1]) + 1
                    item.update({"bbox": [int(left), top, int(right), bottom], "status": "ok",
                                 "reason": "ctc_order_refined_by_connected_foreground_gaps", "ctc_anchor": list(anchors[index]),
                                 "boundary_evidence": [left_evidence[2], right_evidence[2]]})
        output.append(item)
    return {"characters": output,
            "diagnostics": {"image_size": [width, height], "token_count": len(tokens), "valid_ctc_tokens": sum(bool(v) for v in indexed.values()),
                            "latin_segmentation_enabled": bool(segment_latin),
                            "whitespace_reconciled_tokens": sum(v.get("text_index_reconciled") == "omitted_whitespace" for v in indexed.values()),
                            "metadata_notes": notes, "background_luma": background, "dark_ink": dark, "contrast": contrast,
                            "projection_peak": peak, "low_ink_limit": threshold, "ctc_is_coarse_anchor_only": True,
                            "equal_width_fallback_used": False, "boundaries": [v[2] for _, v in sorted(cuts.items())],
                            "edges": [v[2] for _, v in sorted(edges.items())]}}
