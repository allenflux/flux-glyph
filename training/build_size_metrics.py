#!/usr/bin/env python3
"""Build text-size ratios from verified native *train* screenshot pixels only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flux_glyph.text_style import SizeMetrics, build_size_metrics, measure_glyph_ink, _background


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(data, output):
    started = time.monotonic()
    data, output = Path(data).resolve(), Path(output).resolve()
    manifest_path = data / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "flux-glyph-native-captured-glyphs-v1"
            or manifest.get("image_source") != "simctl_png"
            or manifest.get("accepted_native_verified_only") is not True):
        raise ValueError("Expected verified native screenshot preparation")
    descriptor = manifest["splits"]["train"]["metadata"]
    metadata_path = (data / descriptor["path"]).resolve()
    if not metadata_path.is_relative_to(data) or sha(metadata_path) != descriptor["sha256"]:
        raise ValueError("Train metadata SHA differs or path escapes data")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("split") != "train":
        raise ValueError("Only train metadata can fit size metrics")
    sources = {row["source_id"]: row for row in manifest["sources"] if row["split"] == "train"}
    groups = defaultdict(list)
    for row in metadata["rows"]:
        source = sources.get(row["source_id"])
        if source is None or row["source_sha256"] != source["source_sha256"]:
            raise ValueError("Glyph is not bound to this train screenshot")
        native = row.get("native_glyph_provenance", {})
        if native.get("font_match_verified") is not True or native.get("font_family") != row["family"]:
            raise ValueError("Native glyph font identity differs")
        groups[row["source_id"]].append(row)
    rejected = Counter()
    counts = Counter()

    def observations():
        for sid, rows in groups.items():
            source = sources[sid]
            image_path = Path(source.get("image_path", source.get("image", "")))
            if sha(image_path) != source["source_sha256"]:
                raise ValueError("Original train screenshot SHA differs")
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            backgrounds = {}
            for row in rows:
                region_box = tuple(row["region_bbox"])
                if region_box not in backgrounds:
                    backgrounds[region_box] = _background(image, region_box)
                background, spread = backgrounds[region_box]
                if spread > 14:
                    rejected["background_not_uniform"] += 1
                    continue
                measured = measure_glyph_ink(image, row["bbox"], background_rgb=background)
                if measured["status"] != "estimated":
                    rejected[measured["reason"]] += 1
                    continue
                counts[row["family"]] += 1
                yield {"split": "train", "family": row["family"], "character": row["character"],
                       "source_id": sid, "observed_ink_height_px": measured["ink_height_px"],
                       "font_size_screen_px": row["font_size_screen_px"]}

    provenance = {"data_manifest_sha256": sha(manifest_path), "train_metadata_sha256": sha(metadata_path),
                  "runtime_source_sha256": sha(ROOT / "src/flux_glyph/text_style.py"),
                  "train_source_count": len(groups), "calibration_pixels_used": False, "test_pixels_used": False,
                  "source_kind": "ios_simulator_controlled_scene", "observed_height": "original screenshot RGB threshold mask"}
    metrics = SizeMetrics(build_size_metrics(observations(), provenance=provenance))
    if not metrics.value["families"]:
        raise ValueError("Insufficient measured train glyphs for any size metric")
    metrics.save(output)
    report = {"output": str(output), "sha256": sha(output), "provenance": provenance,
              "measured_glyphs_by_family": dict(counts), "rejected_glyphs": dict(rejected),
              "character_metrics": {family: len(item["characters"]) for family, item in metrics.value["families"].items()},
              "seconds": round(time.monotonic() - started, 2)}
    output.with_suffix(".build-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.data, args.output), ensure_ascii=False, indent=2))
