"""Build compact Latin candidate rasters from explicitly supplied local fonts.

No fonts are downloaded or copied to the output. The legacy manifest is used
only for Android source discovery and its recorded source hashes are verified.
Pillow/FreeType are build-time renderers; Linux inference needs only the rasters.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flux_glyph.latin_matcher import ALGORITHM, ALPHABET, SCHEMA, SIZES, CompactLatinBank, latin_raster

SOURCE_HASHES = {
    "SFNS.ttf": "2bfd40dc72e6759e248f82a52a40d551338979fffc9b5c070e685b4b7ad19e66",
    "Helvetica.ttc": "25eceb458d4baf628ee0b6a135a9f8ac5ec7b2826646720834bd2bf00dfc825a",
    "AlipayNumber-Regular.ttf": "6074082d8cb92e175184b177e28335e0171f3ddfb2b5818d7853295f7fb0fada",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(legacy_manifest, system_fonts):
    output = []
    sf = system_fonts / "SFNS.ttf"
    # Two optical sizes avoid treating the macOS default optical-size instance
    # as a complete representation of the SF family on iOS.
    for optical in (17, 48):
        for weight, label in ((400, "regular"), (600, "semibold"), (700, "bold")):
            output.append({"id": f"sfpro_{optical}_{label}", "family": "SF Pro", "path": sf,
                           "postscript_name": ".SFNS-Regular", "index": 0, "variations": [100, optical, 400, weight]})
    for index, label in ((0, "regular"), (1, "bold")):
        output.append({"id": "helvetica_" + label, "family": "Helvetica", "path": system_fonts / "Helvetica.ttc",
                       "postscript_name": "Helvetica" + ("-Bold" if index else ""), "index": index, "variations": []})
    for row in json.loads(legacy_manifest.read_text()):
        if row["family_label"] in {"MiSans", "HarmonyOS Sans SC", "OPPO Sans"}:
            path = Path(row["path"])
            if sha(path) != row["source_sha256"]:
                raise ValueError("Source font checksum mismatch: " + path.name)
            output.append({"id": row["font_id"], "family": row["family_label"], "path": path,
                           "postscript_name": row["postscript_name"], "index": 0, "variations": []})
    alipay = legacy_manifest.parents[2] / "alipay-font-real-fields-v9-20260911/discovery/AlipayNumber-Regular.ttf"
    if alipay.is_file():
        output.append({"id": "alipay_number_regular", "family": "Alipay Number", "path": alipay,
                       "postscript_name": "AlipayNumber-Regular", "index": 0, "variations": []})
    if len({row["family"] for row in output}) < 5:
        raise ValueError("Expected iOS and Android reference source families")
    for row in output:
        if row["path"].name in SOURCE_HASHES and sha(row["path"]) != SOURCE_HASHES[row["path"].name]:
            raise ValueError("Pinned source font checksum mismatch: " + row["path"].name)
        with TTFont(row["path"], fontNumber=row["index"]) as source:
            if source["name"].getDebugName(6) != row["postscript_name"]:
                raise ValueError("Source PostScript name mismatch: " + row["id"])
            if row["variations"]:
                if "fvar" not in source or [axis.axisTag for axis in source["fvar"].axes] != ["wdth", "opsz", "GRAD", "wght"]:
                    raise ValueError("Source variation axis order mismatch: " + row["id"])
                if any(not axis.minValue <= value <= axis.maxValue for axis, value in zip(source["fvar"].axes, row["variations"])):
                    raise ValueError("Source variation value outside axis limits: " + row["id"])
            row["characters"] = "".join(c for c in ALPHABET if ord(c) in source.getBestCmap())
    return output


def font_for(source, size):
    font = ImageFont.truetype(str(source["path"]), size=size, index=source["index"])
    if source["variations"]:
        font.set_variation_by_axes(source["variations"])
    return font


def render(character, font):
    box = font.getbbox(character)
    image = Image.new("RGB", (box[2] - box[0] + 12, box[3] - box[1] + 12), "white")
    ImageDraw.Draw(image).text((6 - box[0], 6 - box[1]), character, font=font, fill="black")
    return image


def render_case(text, font):
    """Render individual advances and record oracle boxes, independent of OCR.

    This isolates font matching from OCR/segmentation. It does not simulate iOS
    or Android native rendering and is not a device-accuracy benchmark.
    """
    bounds = font.getbbox(text)
    width = int(np.ceil(sum(font.getlength(c) for c in text))) + 16
    image = Image.new("RGB", (width, bounds[3] - bounds[1] + 16), "white")
    draw, x, y, glyphs = ImageDraw.Draw(image), 8.0, 8 - bounds[1], []
    for index, character in enumerate(text):
        draw.text((x, y), character, font=font, fill="black")
        left, top, right, bottom = font.getbbox(character)
        glyphs.append({"index": index, "character": character, "status": "ok",
                       "bbox": [int(x + left), max(0, y + top - 1), int(x + right) + 1,
                                min(image.height, y + bottom + 1)]})
        x += font.getlength(character)
    return image, {"characters": glyphs, "diagnostics": {"oracle_boxes": True, "source": "Pillow glyph advances"}}


def evaluate(legacy_manifest, system_fonts, model_dir, fixture_dir=None):
    """Run held-out sizes with known-family positives and serif/mono negatives."""
    bank = CompactLatinBank(model_dir)
    source_rows = sources(legacy_manifest, system_fonts)
    for basename in ("Times.ttc", "Courier.ttc"):
        path = system_fonts / basename
        source_rows.append({"id": "negative_" + path.stem.lower(), "family": None, "path": path,
                            "index": 0, "variations": [], "characters": ALPHABET})
    rows, fixture_rows = [], []
    texts = ["-100.00", "22:43", "2026-08-01 22:43:11", "Safari"]
    if fixture_dir:
        fixture_dir.mkdir(parents=True, exist_ok=True)
    for source in source_rows:
        for size in (26, 38):
            font = font_for(source, size)
            for index, text in enumerate(texts):
                if any(c in ALPHABET and c not in source["characters"] for c in text):
                    continue
                image, segmentation = render_case(text, font)
                result = bank.score(image, text, [], segmentation=segmentation)
                row = {"id": f"{source['id']}_{size}_{index}", "expected_family": source["family"], "text": text,
                       "size": size, "status": result["status"], "family": result["family"], "reason": result["reason"],
                       "top_ranked_family": result["candidates"][0]["family"] if result["candidates"] else None,
                       "source_sha256": sha(source["path"])}
                rows.append(row)
                # A compact, portable selection provides regression fixtures;
                # no local fonts are required to exercise these tests on Linux.
                if fixture_dir and size == 38 and (source["id"] in {"sfpro_17_semibold", "helvetica_bold", "misans_regular",
                        "harmonyos_regular", "oppo_regular", "alipay_number_regular", "negative_times", "negative_courier"}):
                    filename = row["id"] + ".png"
                    image.save(fixture_dir / filename)
                    fixture_rows.append({**row, "image": filename, "image_sha256": sha(fixture_dir / filename), "segmentation": segmentation})
    positive = [row for row in rows if row["expected_family"]]
    negative = [row for row in rows if not row["expected_family"]]
    accepted = [row for row in positive if row["status"] == "candidate"]
    report = {"schema": "flux-glyph-latin-controlled-evaluation-v1",
              "scope": "Held-out Pillow render sizes with oracle character boxes. Not real-device or OCR end-to-end accuracy.",
              "reference_sizes": SIZES, "evaluation_sizes": [26, 38], "reference_archive_sha256": bank.meta["archive_sha256"],
              "positive_cases": len(positive), "candidate_cases": len(accepted),
              "correct_candidates": sum(row["family"] == row["expected_family"] for row in accepted),
              "wrong_candidates": sum(row["family"] != row["expected_family"] for row in accepted),
              "correct_top_ranked": sum(row["top_ranked_family"] == row["expected_family"] for row in positive),
              "negative_cases": len(negative), "negative_candidates": sum(row["status"] == "candidate" for row in negative),
              "cases": rows}
    if fixture_dir:
        (fixture_dir / "cases.json").write_text(json.dumps({"scope": report["scope"], "cases": fixture_rows}, indent=2) + "\n")
    bank.archive.close()
    return report


def build(legacy_manifest, system_fonts, output):
    source_rows = sources(legacy_manifest, system_fonts)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "references.zip"
    entries = {}
    fonts = [[font_for(row, size) for size in SIZES] for row in source_rows]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        for character in ALPHABET:
            rasters = np.zeros((len(fonts), len(SIZES), 32, 32), dtype=np.uint8)
            for face, sizes in enumerate(fonts):
                if character not in source_rows[face]["characters"]:
                    continue
                for size_index, font in enumerate(sizes):
                    raster = latin_raster(render(character, font))
                    if raster is None:
                        raise ValueError(f"Cannot render {character!r} for {source_rows[face]['id']}")
                    rasters[face, size_index] = raster
            encoded = io.BytesIO()
            np.save(encoded, rasters, allow_pickle=False)
            member = f"latin_{ord(character):04X}.npy"
            info = zipfile.ZipInfo(member, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zipped.writestr(info, encoded.getvalue(), compresslevel=9)
            entries[character] = member
    gate_path = output / "GATES.json"
    gate_path.write_text(json.dumps({"method": ALGORITHM, "output": "candidate_only", "gates": {
        "2": {"max_distance": .030, "min_margin": .0030},
        "3": {"max_distance": .040, "min_margin": .0015},
    }, "scope": "Conservative candidate gates; no platform certification or real-device accuracy claim."}, indent=2) + "\n")
    meta = {"schema": SCHEMA, "algorithm_version": ALGORITHM, "dtype": "uint8", "sizes": SIZES,
            "font_ids": [row["id"] for row in source_rows],
            "face_family": {row["id"]: row["family"] for row in source_rows}, "characters": entries,
            "face_characters": {row["id"]: row["characters"] for row in source_rows},
            "shape": [len(source_rows), len(SIZES), 32, 32], "archive": archive.name, "archive_sha256": sha(archive),
            "source_paths_embedded": False, "sources": {row["id"]: {
                "basename": row["path"].name, "sha256": sha(row["path"]), "postscript_name": row["postscript_name"],
                "face_index": row["index"], "variations": row["variations"]} for row in source_rows},
            "gates": {"path": gate_path.name, "sha256": sha(gate_path)},
            "builder": {"script": "build_latin_references.py", "pillow_version": pillow_version,
                        "source_manifest_sha256": sha(legacy_manifest)}}
    (output / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    bank = CompactLatinBank(output)
    bank.archive.close()
    return {"characters": len(ALPHABET), "faces": len(source_rows), "families": sorted(set(meta["face_family"].values())),
            "archive_bytes": archive.stat().st_size, "archive_sha256": sha(archive)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--system-fonts", type=Path, default=Path("/System/Library/Fonts"))
    parser.add_argument("--output", type=Path, default=ROOT / "models/latin")
    parser.add_argument("--evaluate", action="store_true", help="Write held-out-size diagnostic report")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/latin-accuracy-validation.json")
    parser.add_argument("--fixture-dir", type=Path, help="Optional portable regression fixture output")
    args = parser.parse_args()
    print(json.dumps(build(args.legacy_manifest, args.system_fonts, args.output), indent=2))
    if args.evaluate:
        report = evaluate(args.legacy_manifest, args.system_fonts, args.output, args.fixture_dir)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
