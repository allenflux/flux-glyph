#!/usr/bin/env python3
"""Evaluate frozen style metrics on native calibration or test screenshots.

Uses verified native character boxes and true font labels, so this measures
style estimation conditional on correct segmentation/font identity, not OCR or
end-to-end font recognition. Nothing is fitted or adjusted by this script.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flux_glyph.text_style import SizeMetrics, estimate_text_style


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(data, metrics_path, split):
    if split not in ("calibration", "test"):
        raise ValueError("Evaluation requires calibration or test")
    data = Path(data).resolve()
    manifest_path = data / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    metrics_sha = sha(metrics_path)
    metrics = SizeMetrics.load(metrics_path)
    if (manifest.get("schema") != "flux-glyph-native-captured-glyphs-v1"
            or metrics.value.get("provenance", {}).get("data_manifest_sha256") != sha(manifest_path)):
        raise ValueError("Frozen size metrics are not bound to this screenshot collection")
    descriptor = manifest["splits"][split]["metadata"]
    path = (data / descriptor["path"]).resolve()
    if not path.is_relative_to(data) or sha(path) != descriptor["sha256"]:
        raise ValueError("Evaluation metadata SHA differs")
    metadata = json.loads(path.read_text())
    if metadata.get("split") != split:
        raise ValueError("Evaluation partition differs")
    sources = {source["source_id"]: source for source in manifest["sources"] if source["split"] == split}
    groups = defaultdict(lambda: defaultdict(list))
    for row in metadata["rows"]:
        source = sources.get(row["source_id"])
        if source is None or source["source_sha256"] != row["source_sha256"]:
            raise ValueError("Glyph is not bound to this evaluation source")
        groups[row["source_id"]][row["region_id"]].append(row)
    results = []
    for source_id, regions in groups.items():
        source = sources[source_id]
        path = Path(source.get("image_path", source.get("image", "")))
        if sha(path) != source["source_sha256"]:
            raise ValueError("Original evaluation screenshot SHA differs")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        for region_id, rows in regions.items():
            row = rows[0]
            glyphs = [{"source_bbox": item["bbox"], "character": item["character"], "status": "ok"} for item in rows]
            style = estimate_text_style(image, glyphs, family=row["family"], metrics=metrics, region_bbox=row["region_bbox"])
            true_size = float(row["font_size_screen_px"])
            expected = row["text_color_hex"][:7]
            color_error = None
            if style["text_color_hex"]:
                color_error = max(abs(int(style["text_color_hex"][i:i+2], 16) - int(expected[i:i+2], 16)) for i in (1, 3, 5))
            actual_size = style["font_size_px_estimate"]
            results.append({"source_id": source_id, "region_id": region_id, "family": row["family"],
                            "text_color_expected": expected, "color_max_channel_error": color_error,
                            "font_size_px_expected": true_size, "style": style,
                            "size_absolute_error_px": abs(actual_size - true_size) if actual_size is not None else None,
                            "size_relative_error": abs(actual_size - true_size) / true_size if actual_size is not None else None})

    def summarize(rows):
        colors = [r["color_max_channel_error"] for r in rows if r["color_max_channel_error"] is not None]
        absolute = [r["size_absolute_error_px"] for r in rows if r["size_absolute_error_px"] is not None]
        relative = [r["size_relative_error"] for r in rows if r["size_relative_error"] is not None]
        covered = [r for r in rows if r["style"]["font_size_px_interval"] is not None]
        return {"regions": len(rows), "color_estimated": len(colors), "color_coverage": len(colors) / len(rows) if rows else 0,
                "color_exact_rate_among_estimates": float(np.mean(np.asarray(colors) == 0)) if colors else None,
                "color_max_channel_error_p95": float(np.quantile(colors, .95)) if colors else None,
                "size_estimated": len(absolute), "size_coverage": len(absolute) / len(rows) if rows else 0,
                "size_mae_px": float(np.mean(absolute)) if absolute else None,
                "size_median_relative_error": float(np.median(relative)) if relative else None,
                "size_p95_relative_error": float(np.quantile(relative, .95)) if relative else None,
                "empirical_interval_coverage_among_estimates": float(np.mean([
                    r["style"]["font_size_px_interval"][0] <= r["font_size_px_expected"] <= r["style"]["font_size_px_interval"][1]
                    for r in covered])) if covered else None}
    if sha(metrics_path) != metrics_sha:
        raise ValueError("Frozen metrics changed during evaluation")
    return {"schema": "flux-glyph-text-style-evaluation-v1", "split": split, "metrics_sha256": metrics_sha,
            "data_manifest_sha256": sha(manifest_path), "fitted_on_evaluation": False,
            "conditions": "verified native glyph boxes and true font family; not end-to-end OCR/font-classification accuracy",
            "summary": summarize(results), "families": {family: summarize([row for row in results if row["family"] == family])
                for family in sorted({row["family"] for row in results})}, "regions": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.data, args.metrics, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
