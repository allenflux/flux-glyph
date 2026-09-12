"""Offline, reproducible importer for the self-contained Flux Glyph model assets.

The font archive retains the frozen R12 blur32 matching numerator losslessly as
uint8 [31 faces, 3 reference sizes, 32, 32] rasters per Han character.  It does
not train or download a model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from flux_glyph.foreground import normalize_foreground
from flux_glyph.glyph_preprocess import extract_glyphs

FONT_SCHEMA = "flux-glyph-r13-compact-v1"
ALGORITHM = "r12-normalized-glyph_blur1_resize32_uint8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(legacy_root: Path) -> tuple[Path, dict]:
    source = legacy_root / "runs/alipay-font-alipay-scope-v13-20260911/compact/references.json"
    return source, json.loads(source.read_text())


def compact_raster(image: Image.Image) -> np.ndarray:
    """Return exactly the uint8 numerator used by R12 blur32 before L2 norm."""
    foreground, _ = normalize_foreground(image.convert("RGB"))
    if foreground is None:
        raise ValueError("normalize_foreground returned no image")
    extracted = extract_glyphs(foreground, 1)
    if extracted.glyphs is None or extracted.glyphs.shape != (1, 64, 64):
        raise ValueError("extract_glyphs failed")
    glyph = extracted.glyphs[0].astype(np.float64)
    numerator = np.clip(np.rint(glyph * 255.0), 0, 255).astype(np.uint8)
    return np.asarray(
        Image.fromarray(numerator)
        .filter(ImageFilter.GaussianBlur(1))
        .resize((32, 32), Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )


def compact_vector(raster: np.ndarray) -> np.ndarray:
    value = raster.astype(np.float64, copy=False).reshape(-1) / 255.0
    norm = np.sqrt(np.einsum("i,i->", value, value, optimize=False))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("blank/nonfinite compact raster")
    return value / norm


def package(legacy_root: Path, output: Path) -> dict:
    source, reference = load_source(legacy_root)
    expected_faces = reference["font_ids"]
    if len(expected_faces) != 31 or len(reference["face_family"]) != 31:
        raise ValueError("expected frozen 31-face source inventory")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "references.npz.zip"
    members: dict[str, str] = {}
    base = legacy_root / "runs"
    checked = 0
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        for character, faces in reference["characters"].items():
            if len(faces) != 31:
                raise ValueError(f"{character}: expected 31 faces")
            packed = np.empty((31, 3, 32, 32), dtype=np.uint8)
            for face_index, sizes in enumerate(faces):
                if len(sizes) != 3:
                    raise ValueError(f"{character}: face {face_index} lacks 3 sizes")
                for size_index, (relative, expected_sha) in enumerate(sizes):
                    source_png = base / relative
                    if sha256(source_png) != expected_sha:
                        raise ValueError(f"source hash mismatch: {source_png}")
                    with Image.open(source_png) as image:
                        packed[face_index, size_index] = compact_raster(image)
                    checked += 1
            member = f"cjk_{ord(character):04X}.npy"
            encoded = io.BytesIO()
            np.save(encoded, packed, allow_pickle=False)
            zipped.writestr(member, encoded.getvalue())
            members[character] = member
    metadata = {
        "schema": FONT_SCHEMA,
        "algorithm_version": ALGORITHM,
        "font_ids": expected_faces,
        "face_family": reference["face_family"],
        "characters": members,
        "shape": [31, 3, 32, 32],
        "dtype": "uint8",
        "sizes": [20, 28, 44],
        "archive": archive.name,
        "archive_sha256": sha256(archive),
        "source_manifest_sha256": sha256(source),
        "source_paths_embedded": False,
        "source_code_sha256": {
            "foreground.py": sha256(PROJECT / "src/flux_glyph/foreground.py"),
            "glyph_preprocess.py": sha256(PROJECT / "src/flux_glyph/glyph_preprocess.py"),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return {"archive": archive, "source": source, "reference": reference, "checked": checked}


def load_r12_matcher(legacy_root: Path):
    matcher_path = legacy_root / "runs/alipay-font-reference-v12-20260911/matcher.py"
    spec = importlib.util.spec_from_file_location("_flux_r12_matcher", matcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen R12 matcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, matcher_path


def verify_equivalence(legacy_root: Path, output: Path) -> dict:
    """Compare every archive raster L2 vector with frozen R12 matcher.vector(..., blur32)."""
    archive = output / "references.npz.zip"
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    source, reference = load_source(legacy_root)
    if metadata.get("archive_sha256") != sha256(archive):
        raise ValueError("metadata does not bind archive")
    if metadata.get("source_manifest_sha256") != sha256(source):
        raise ValueError("metadata source manifest binding mismatch")
    r12, matcher_path = load_r12_matcher(legacy_root)
    base = legacy_root / "runs"
    compared = 0
    max_difference = 0.0
    exact_equal = 0
    with zipfile.ZipFile(archive, "r") as zipped:
        for character, faces in reference["characters"].items():
            member = metadata["characters"].get(character)
            if member != f"cjk_{ord(character):04X}.npy":
                raise ValueError(f"incorrect member mapping for {character}")
            packed = np.load(io.BytesIO(zipped.read(member)), allow_pickle=False)
            if packed.shape != (31, 3, 32, 32) or packed.dtype != np.uint8:
                raise ValueError(f"invalid packed shape/dtype for {character}")
            for face_index, sizes in enumerate(faces):
                for size_index, (relative, expected_sha) in enumerate(sizes):
                    source_png = base / relative
                    if sha256(source_png) != expected_sha:
                        raise ValueError(f"source changed during verification: {source_png}")
                    with Image.open(source_png) as image:
                        expected, diagnostic = r12.vector(image.convert("RGB"), "blur32")
                    if expected is None or not diagnostic.get("ok"):
                        raise ValueError(f"R12 rejected source reference: {source_png}: {diagnostic}")
                    observed = compact_vector(packed[face_index, size_index])
                    if expected.shape != observed.shape or not np.isfinite(observed).all():
                        raise ValueError(f"invalid vector at {source_png}")
                    difference = float(np.max(np.abs(expected - observed)))
                    max_difference = max(max_difference, difference)
                    if np.array_equal(expected, observed):
                        exact_equal += 1
                    if difference > 1e-12:
                        raise ValueError(f"R12 equivalence failed ({difference}) at {source_png}")
                    compared += 1
    verification = {
        "schema": "flux-glyph-r13-compact-verification-v2",
        "characters": len(reference["characters"]),
        "expected_per_character_sources": 93,
        "checked_every_source_png_sha256": True,
        "source_pngs_checked": compared,
        "r12_blur32_vector_comparisons": compared,
        "r12_vector_array_equal": exact_equal,
        "r12_vector_max_abs_difference": max_difference,
        "r12_vector_tolerance": 1e-12,
        "archive": archive.name,
        "archive_sha256": sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "metadata_sha256": sha256(metadata_path),
        "source_manifest_sha256": sha256(source),
        "source_code_sha256": {
            "foreground.py": sha256(PROJECT / "src/flux_glyph/foreground.py"),
            "glyph_preprocess.py": sha256(PROJECT / "src/flux_glyph/glyph_preprocess.py"),
            "frozen_r12_matcher.py": sha256(matcher_path),
        },
        "algorithm_version": ALGORITHM,
    }
    (output / "VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n")
    return verification


def write_models_manifest(models_root: Path) -> dict:
    files = []
    region_model = (models_root / "region_neural" / "metadata.json").is_file()
    if not (models_root / "pp").is_dir() or not (region_model or (models_root / "font").is_dir()):
        raise ValueError("A complete bundle needs pp/ and either font/ or region_neural/ assets")
    # Never sweep ACTIVE.json, old releases, or unrelated local files into a
    # bundle. Selection and historical versions belong to the deployment host.
    assets = [p for folder in ("pp", "font", "latin", "neural", "style", "region_neural") for p in (models_root / folder).rglob("*")
              if p.is_file() and not p.name.startswith(".")]
    for path in sorted(assets):
        files.append({"path": str(path.relative_to(models_root)), "sha256": sha256(path), "bytes": path.stat().st_size})
    manifest = {
        "schema": "flux-glyph-models-v2",
        "files": files,
        "source_note": ("PP text detector plus trained region font/size CNN" if region_model else
                        "user-supplied PP ONNX plus frozen R13 reference rasters compacted without new neural training"),
    }
    (models_root / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT / "models/font")
    parser.add_argument("--package", action="store_true", help="rebuild compact archive")
    parser.add_argument("--verify", action="store_true", help="verify existing archive against frozen R12 vectors")
    parser.add_argument("--write-manifest", action="store_true", help="refresh models/MANIFEST.json from self-contained files")
    args = parser.parse_args()
    if not args.package and not args.verify and not args.write_manifest:
        parser.error("choose --package, --verify, and/or --write-manifest")
    if args.package:
        package(args.legacy_root, args.output)
    if args.verify:
        print(json.dumps(verify_equivalence(args.legacy_root, args.output), indent=2))
    if args.write_manifest:
        print(json.dumps(write_models_manifest(args.output.resolve().parent), indent=2))

if __name__ == "__main__":
    main()
