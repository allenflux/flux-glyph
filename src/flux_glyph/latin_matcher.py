"""Offline Latin/digit family candidates from bounded, aspect-preserving rasters.

This bank is deliberately separate from the frozen Han certification gate.  A
match is evidence for a candidate family; it never certifies Apple support.
"""
from __future__ import annotations

import io
import json
import re
import string
import zipfile
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .font_matcher import views
from .models import file_sha
from .segmentation import segment_characters

ALPHABET = string.digits + string.ascii_uppercase + string.ascii_lowercase
SCHEMA = "flux-glyph-latin-candidates-v1"
ALGORITHM = "latin-aspect28-blur060-v1"
SIZES = [20, 32, 48]
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def latin_character(character):
    return isinstance(character, str) and len(character) == 1 and character in ALPHABET


def latin_raster(image):
    """Preserve width/height and ink polarity; reject tiny/low-contrast samples."""
    a = np.asarray(image.convert("L"), dtype=np.float32)
    if a.ndim != 2 or min(a.shape) < 3 or max(a.shape) > 4096:
        return None
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    background = float(np.median(border))
    dark = background - float(a.min()) >= float(a.max()) - background
    departure = np.maximum(background - a if dark else a - background, 0)
    contrast = float(departure.max())
    if contrast < 20:
        return None
    mask = departure >= .20 * contrast
    ys, xs = np.where(mask)
    if len(xs) < 12 or not len(ys):
        return None
    left, right, top, bottom = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    if bottom - top < 8:
        return None
    tight = np.clip(departure[top:bottom, left:right] * (255 / contrast), 0, 255).astype(np.uint8)
    scale = 28 / max(tight.shape)
    resized = Image.fromarray(tight).resize((max(1, round(tight.shape[1] * scale)),
                                            max(1, round(tight.shape[0] * scale))), Image.Resampling.BICUBIC)
    canvas = Image.new("L", (32, 32))
    canvas.paste(resized, ((32 - resized.width) // 2, (32 - resized.height) // 2))
    return np.asarray(canvas.filter(ImageFilter.GaussianBlur(.60)), dtype=np.uint8)


def _vector(raster):
    a = raster.astype(np.float32).reshape(-1) / 255
    norm = float(np.sqrt(np.einsum("i,i->", a, a, optimize=False)))
    return a / norm if np.isfinite(norm) and norm > 1e-12 else None


def _json(path, limit=256 * 1024):
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Latin metadata exceeds bounded size")
    try:
        value = json.loads(path.read_text())
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid Latin metadata JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Invalid Latin metadata object")
    return value


def _local_file(directory, name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", "..") or "\\" in name:
        raise ValueError("Invalid Latin asset path")
    path = directory / name
    if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Latin asset escapes model directory or is missing")
    return path


def _checked_file(directory, name, digest):
    path = _local_file(directory, name)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Latin asset exceeds size limit")
    if not isinstance(digest, str) or not _HASH.fullmatch(digest) or file_sha(path) != digest:
        raise ValueError("Latin asset checksum mismatch: " + name)
    return path


def _read_raster(archive, name, shape, covered):
    info = archive.getinfo(name)
    expected = int(np.prod(shape))
    if info.is_dir() or info.file_size > expected + 4096 or info.file_size < expected or info.flag_bits & 1:
        raise ValueError("Latin reference exceeds bounded shape")
    stream = io.BytesIO(archive.read(info))
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            actual, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=2048)
        elif version == (2, 0):
            actual, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=2048)
        else:
            raise ValueError("Unsupported NPY version")
        if tuple(actual) != tuple(shape) or dtype != np.dtype("uint8") or fortran:
            raise ValueError("Invalid Latin reference shape or dtype")
        raw = stream.read(expected + 1)
        if len(raw) != expected:
            raise ValueError("Invalid Latin reference byte count")
        value = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    except Exception as error:
        raise ValueError("Invalid Latin reference raster: " + name) from error
    present = np.any(value, axis=(2, 3))
    if not np.array_equal(present, np.repeat(np.asarray(covered, dtype=bool)[:, None], shape[1], axis=1)):
        raise ValueError("Latin reference vector coverage mismatch")
    return value


class CompactLatinBank:
    def __init__(self, directory, max_characters=32):
        self.directory = Path(directory)
        self.meta = meta = _json(_local_file(self.directory, "metadata.json"))
        ids, families, entries = meta.get("font_ids"), meta.get("face_family"), meta.get("characters")
        if (meta.get("schema") != SCHEMA or meta.get("algorithm_version") != ALGORITHM or
                meta.get("dtype") != "uint8" or meta.get("sizes") != SIZES or
                not isinstance(ids, list) or not 4 <= len(ids) <= 48 or
                not all(isinstance(i, str) and re.fullmatch(r"[a-z0-9_]{1,80}", i) for i in ids) or len(set(ids)) != len(ids)):
            raise ValueError("Unsupported Latin bank contract")
        if (not isinstance(families, dict) or set(families) != set(ids) or
                not all(isinstance(f, str) and 1 <= len(f) <= 80 for f in families.values()) or
                len(set(families.values())) < 3):
            raise ValueError("Invalid Latin family registry")
        if (not isinstance(entries, dict) or set(entries) != set(ALPHABET) or
                any(value != f"latin_{ord(c):04X}.npy" for c, value in entries.items()) or
                meta.get("shape") != [len(ids), len(SIZES), 32, 32]):
            raise ValueError("Invalid Latin character registry")
        coverage = meta.get("face_characters")
        if (not isinstance(coverage, dict) or set(coverage) != set(ids) or
                any(not isinstance(chars, str) or not chars or len(set(chars)) != len(chars) or
                    not set(chars) <= set(ALPHABET) for chars in coverage.values()) or
                any(sum(c in chars for chars in coverage.values()) < 3 for c in ALPHABET)):
            raise ValueError("Invalid Latin face character coverage")
        sources = meta.get("sources")
        if (not isinstance(sources, dict) or set(sources) != set(ids) or meta.get("source_paths_embedded") is not False):
            raise ValueError("Invalid Latin source registry")
        for source in sources.values():
            if (not isinstance(source, dict) or set(source) != {"basename", "sha256", "postscript_name", "face_index", "variations"} or
                    not isinstance(source["basename"], str) or Path(source["basename"]).name != source["basename"] or
                    "\\" in source["basename"] or not source["basename"] or
                    not isinstance(source["sha256"], str) or not _HASH.fullmatch(source["sha256"]) or
                    not isinstance(source["postscript_name"], str) or not source["postscript_name"] or
                    type(source["face_index"]) is not int or not 0 <= source["face_index"] <= 128 or
                    not isinstance(source["variations"], list) or len(source["variations"]) > 8 or
                    any(type(v) not in (float, int) or not np.isfinite(v) or abs(v) > 10000 for v in source["variations"])):
                raise ValueError("Invalid Latin source provenance")
        gate_info = meta.get("gates")
        if not isinstance(gate_info, dict) or set(gate_info) != {"path", "sha256"}:
            raise ValueError("Invalid Latin gate reference")
        gates = _json(_checked_file(self.directory, gate_info["path"], gate_info["sha256"]))
        if gates.get("method") != ALGORITHM or gates.get("output") != "candidate_only" or set(gates.get("gates", {})) != {"2", "3"}:
            raise ValueError("Invalid Latin gate contract")
        for gate in gates["gates"].values():
            if (not isinstance(gate, dict) or set(gate) != {"max_distance", "min_margin"} or
                    any(type(v) not in (int, float) or not np.isfinite(v) or not 0 < v < 1 for v in gate.values())):
                raise ValueError("Invalid Latin gate values")
        path = _checked_file(self.directory, meta.get("archive"), meta.get("archive_sha256"))
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Latin archive exceeds size limit")
        opened = zipfile.ZipFile(path)
        try:
            infos = opened.infolist()
            names = [item.filename for item in infos]
            if (len(names) != len(set(names)) or set(names) != set(entries.values()) or
                    sum(item.file_size for item in infos) > 16 * 1024 * 1024 or
                    any((item.external_attr >> 16) & 0o170000 == 0o120000 for item in infos)):
                raise ValueError("Latin archive member registry mismatch")
            for character, name in entries.items():
                _read_raster(opened, name, meta["shape"], [character in coverage[face] for face in ids])
        except Exception:
            opened.close()
            raise
        self.archive, self.font_ids, self.face_family, self.entries = opened, ids, families, entries
        self.coverage = coverage
        self.families, self.gates = sorted(set(families.values())), gates["gates"]
        self.cache = OrderedDict()
        self.max_characters = max(1, min(62, int(max_characters)))

    @property
    def cache_bytes(self):
        return sum(value.nbytes for value in self.cache.values())

    def references(self, character):
        if character not in self.entries:
            return None
        if character not in self.cache:
            while len(self.cache) >= self.max_characters:
                self.cache.popitem(last=False)
            covered = [character in self.coverage[face] for face in self.font_ids]
            values = _read_raster(self.archive, self.entries[character], self.meta["shape"], covered)
            self.cache[character] = np.stack([_vector(a) if a.any() else np.zeros(1024, dtype=np.float32)
                                            for a in values.reshape(-1, 32, 32)]).reshape(len(self.font_ids), 3, 1024)
        self.cache.move_to_end(character)
        return self.cache[character]

    def _match(self, samples):
        by_character = defaultdict(list)
        for sample in samples:
            raster = latin_raster(sample["image"])
            if raster is None:
                return None
            z = _vector(raster)
            refs = self.references(sample["character"])
            if z is None or refs is None:
                return None
            by_character[sample["character"]].append(np.maximum(0, 1 - np.einsum("fsk,k->fs", refs, z, optimize=False)).min(axis=1))
        if not by_character:
            return None
        score = np.mean([np.mean(values, axis=0) for values in by_character.values()], axis=0)
        scores = {family: float(min(score[i] for i, face in enumerate(self.font_ids) if self.face_family[face] == family))
                  for family in self.families}
        ordered = sorted(scores, key=lambda family: (scores[family], family))
        return {"family": ordered[0], "distance": scores[ordered[0]],
                "margin": scores[ordered[1]] - scores[ordered[0]], "family_scores": scores}

    def score(self, image, text, tokens, metadata=None, *, segmentation=None):
        """Score ASCII evidence in a complete upright OCR ROI, including mixed text.

        ``segmentation`` optionally reuses ``segment_characters(...,
        segment_latin=True)`` output. ``glyphs`` contain source-ROI boxes only;
        callers can crop the original image without returning Pillow objects.
        """
        wanted = [i for i, character in enumerate(text) if latin_character(character)]
        result = {"status": "uncertain", "family": None, "candidates": [], "reason": "insufficient_latin_evidence",
                  "glyphs": [], "segmentation": None, "evidence": {"method": ALGORITHM, "candidate_only": True,
                  "requested_characters": len(wanted), "distinct_characters": [], "complete": False}}
        if not wanted:
            result.update(status="unsupported_script", reason="no_latin_or_digit_characters")
            return result
        if len(wanted) > 128:
            result["reason"] = "latin_evidence_limit_exceeded"
            return result
        segmented = segmentation if segmentation is not None else segment_characters(image, text, tokens, metadata=metadata, segment_latin=True)
        result["segmentation"] = segmented
        glyphs = [item for item in segmented["characters"] if item.get("index") in wanted and item.get("character") == text[item["index"]]]
        result["glyphs"] = glyphs
        valid = [item for item in glyphs if item.get("status") == "ok" and item.get("bbox")]
        complete = len(valid) == len(wanted) and len({item["index"] for item in valid}) == len(wanted)
        samples = []
        for item in valid:
            left, top, right, bottom = item["bbox"]
            # Vertical segmentation is ink-tight. Recover a 2px source margin so
            # border-based foreground polarity is not inferred from a glyph cap.
            box = [max(0, left), max(0, top - 2), min(image.width, right), min(image.height, bottom + 2)]
            samples.append({"character": item["character"], "image": image.crop(box)})
        distinct = sorted({item["character"] for item in samples})
        result["evidence"].update(complete=complete, distinct_characters=distinct, sampled_characters=len(samples))
        if not samples:
            result["reason"] = "latin_segmentation_uncertain"
            return result
        variants = {key: [] for key in ("identity", "jpeg75", "resize80_restore")}
        for sample in samples:
            # Keep degradation views from collapsing a narrow "i" or "1" to a
            # two-pixel-wide image. Padding contains only the observed border
            # background and never neighbouring glyph ink.
            rgb = np.asarray(sample["image"].convert("RGB"))
            border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
            background = tuple(int(value) for value in np.median(border, axis=0))
            padded = ImageOps.expand(sample["image"], border=4, fill=background)
            for key, variant in views(padded).items():
                variants[key].append({"character": sample["character"], "image": variant})
        matches = {key: self._match(value) for key, value in variants.items()}
        identity = matches["identity"]
        if identity:
            result["candidates"] = [{"family": family, "distance": distance} for family, distance in
                                    sorted(identity["family_scores"].items(), key=lambda value: (value[1], value[0]))[:3]]
        gate = self.gates["2" if len(distinct) == 2 else "3"]
        result["evidence"].update(views=matches, gate=gate)
        if not complete:
            result["reason"] = "latin_segmentation_incomplete"
        elif len(distinct) < 2:
            result["reason"] = "too_few_distinct_latin_characters"
        elif any(match is None for match in matches.values()):
            result["reason"] = "low_quality_latin_glyph"
        elif len({match["family"] for match in matches.values()}) != 1:
            result["reason"] = "unstable_latin_family"
        elif any(match["distance"] > gate["max_distance"] for match in matches.values()):
            result["reason"] = "latin_font_outside_reference_gate"
        elif any(match["margin"] < gate["min_margin"] for match in matches.values()):
            result["reason"] = "ambiguous_latin_font_families"
        else:
            result.update(status="candidate", family=identity["family"], reason="stable_latin_family_candidate")
        return result
