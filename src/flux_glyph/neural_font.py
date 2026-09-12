"""Bounded ONNX font-family inference, independent of reference glyph banks.

Training and inference share ``preprocess_glyph``. The caller supplies an
individual source glyph with a background margin: this function deliberately
does not add padding or blur, which would change the training distribution.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import string

import numpy as np
import onnxruntime as ort
from PIL import Image

from .glyph_preprocess import extract_glyphs

SCHEMA = "flux-glyph-neural-font-v1"
ALGORITHM = "glyph-cnn64-v1"
MAX_SAMPLES = 128
MAX_MODEL_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 4 * 1024 * 1024
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_LATIN = frozenset(string.ascii_letters + string.digits)


def preprocess_glyph(image: Image.Image) -> np.ndarray | None:
    """Return float32 [64, 64] ink in [0, 1], or reject an unusable crop.

    Callers must retain the observed background margin when cropping. This
    uses the original 64px extraction, without the reference bank's blur32
    transform. Checking true contrast first prevents the extractor's contrast
    floor from turning nearly blank patches into useful evidence.
    """
    if (not isinstance(image, Image.Image) or min(image.size) < 4 or
            max(image.size) > 4096 or image.width * image.height > MAX_IMAGE_PIXELS):
        return None
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    background = float(np.median(border))
    contrast = max(background - float(gray.min()), float(gray.max()) - background)
    if contrast < 16:
        return None
    extracted = extract_glyphs(image, 1)
    values = extracted.glyphs
    if (values is None or not extracted.diagnostics.get("ok") or
            values.shape != (1, 64, 64) or not np.isfinite(values).all()):
        return None
    return np.ascontiguousarray(values[0], dtype=np.float32)


def _local_file(directory: Path, name: str) -> Path:
    if (not isinstance(name, str) or not name or name in (".", "..") or
            Path(name).name != name or "\\" in name):
        raise ValueError("Invalid neural model asset path")
    path = directory / name
    if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Neural model asset escapes directory or is missing")
    return path


def _bounded_bytes(path: Path, limit: int) -> bytes:
    if path.stat().st_size > limit:
        raise ValueError("Neural model asset exceeds size limit")
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError("Neural model asset exceeds size limit")
    return value


def _number(value, *, lower: float, upper: float, inclusive_lower=False) -> bool:
    return (type(value) in (int, float) and np.isfinite(value) and
            (value >= lower if inclusive_lower else value > lower) and value <= upper)


def _metadata(directory: Path) -> dict:
    raw = _bounded_bytes(_local_file(directory, "metadata.json"), 256 * 1024)
    try:
        meta = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid neural model metadata JSON") from error
    if (not isinstance(meta, dict) or meta.get("schema") != SCHEMA or
            meta.get("algorithm") != ALGORITHM):
        raise ValueError("Unsupported neural model contract")
    families = meta.get("families")
    if (not isinstance(families, list) or not 2 <= len(families) <= 64 or
            not all(isinstance(f, str) and 1 <= len(f) <= 80 and f.strip() == f for f in families) or
            len(set(families)) != len(families)):
        raise ValueError("Invalid neural font family registry")
    scripts, gates, temperatures = (meta.get(key) for key in ("scripts", "gates", "temperature"))
    for registry in (scripts, gates, temperatures):
        if not isinstance(registry, dict) or set(registry) != {"han", "latin"}:
            raise ValueError("Invalid neural script registry")
    for script in ("han", "latin"):
        names = scripts[script]
        if (not isinstance(names, list) or not 2 <= len(names) <= len(families) or
                not all(isinstance(name, str) and name in families for name in names) or
                len(set(names)) != len(names)):
            raise ValueError("Invalid neural script family mask")
        gate = gates[script]
        if (not isinstance(gate, dict) or set(gate) != {"min_score", "min_margin"} or
                not _number(gate["min_score"], lower=0, upper=1) or
                not _number(gate["min_margin"], lower=0, upper=1, inclusive_lower=True)):
            raise ValueError("Invalid neural score gates")
        if not _number(temperatures[script], lower=0, upper=100):
            raise ValueError("Invalid neural temperature")
    if set(scripts["han"]) | set(scripts["latin"]) != set(families):
        raise ValueError("Neural family lacks a script")
    model = meta.get("model")
    if (not isinstance(model, dict) or set(model) != {"path", "sha256"} or
            not isinstance(model["path"], str) or not model["path"].endswith(".onnx") or
            not isinstance(model["sha256"], str) or not _HASH.fullmatch(model["sha256"])):
        raise ValueError("Invalid neural model reference")
    return meta


def _dynamic_dimension(value) -> bool:
    return value is None or isinstance(value, str) and 0 < len(value) <= 80


def _validate_session(session, family_count: int) -> None:
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Neural ONNX input/output count mismatch")
    source, target = inputs[0], outputs[0]
    if (source.name != "glyphs" or source.type != "tensor(float)" or
            not isinstance(source.shape, (list, tuple)) or len(source.shape) != 4 or
            not _dynamic_dimension(source.shape[0]) or list(source.shape[1:]) != [1, 64, 64]):
        raise ValueError("Neural ONNX input must be float32 glyphs [N,1,64,64]")
    if (target.name != "logits" or target.type != "tensor(float)" or
            not isinstance(target.shape, (list, tuple)) or len(target.shape) != 2 or
            not _dynamic_dimension(target.shape[0]) or target.shape[1] != family_count):
        raise ValueError("Neural ONNX output must be float32 logits [N,C]")


def _character_in_script(character, script: str) -> bool:
    if not isinstance(character, str) or len(character) != 1:
        return False
    return "\u4e00" <= character <= "\u9fff" if script == "han" else character in _LATIN


class NeuralFontClassifier:
    """An ONNX neural classifier with script masks and explicit abstention."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = _metadata(self.directory)
        self.families = list(self.meta["families"])
        model = self.meta["model"]
        data = _bounded_bytes(_local_file(self.directory, model["path"]), MAX_MODEL_BYTES)
        if hashlib.sha256(data).hexdigest() != model["sha256"]:
            raise ValueError("Neural model checksum mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Loading verified bytes keeps external ONNX initializer paths from
        # resolving relative to the model directory; no pickle is loaded.
        try:
            self.session = ort.InferenceSession(data, sess_options=options,
                                                providers=["CPUExecutionProvider"])
        except Exception as error:
            raise ValueError("Invalid neural ONNX model") from error
        _validate_session(self.session, len(self.families))
        self.indices = {script: [i for i, family in enumerate(self.families)
                                  if family in self.meta["scripts"][script]]
                        for script in ("han", "latin")}

    def _ranking(self, probabilities, script):
        indices = self.indices[script]
        order = sorted(range(len(indices)), key=lambda i: (-float(probabilities[i]), indices[i]))
        candidates = [{"family": self.families[indices[i]], "score": float(probabilities[i])}
                      for i in order[:3]]
        return candidates, candidates[0]["score"], float(probabilities[order[0]] - probabilities[order[1]])

    def _decision(self, score, margin, script):
        gate = self.meta["gates"][script]
        if score < gate["min_score"]:
            return "below_score_gate"
        if margin <= 1e-8 or margin < gate["min_margin"]:
            return "ambiguous_neural_families"
        return "neural_family_candidate"

    def predict(self, samples, script="han", complete=True):
        """Predict from real neural logits; repeated characters receive one vote.

        Softmax scores describe this model's known classes, not certainty that
        an unknown font is present in the vocabulary. Low-quality or incomplete
        segmentation always abstains, retaining available candidates for review.
        """
        if script not in ("han", "latin"):
            raise ValueError("Unsupported neural font script")
        if type(complete) is not bool:
            raise ValueError("Neural segmentation completeness must be boolean")
        evidence = {"script": script, "complete": complete, "sampled_characters": 0,
                    "valid_characters": 0, "distinct_characters": [], "distinct_character_count": 0,
                    "gate": dict(self.meta["gates"][script]), "temperature": self.meta["temperature"][script],
                    "aggregation": "mean_probability_per_distinct_character",
                    "score_type": "temperature_scaled_softmax"}
        result = {"status": "uncertain", "family": None, "candidates": [], "score": None, "margin": None,
                  "reason_code": "no_samples", "reason": "no_samples", "method": "neural_network",
                  "glyph_predictions": [], "evidence": evidence}

        def finish(reason):
            result["reason_code"] = result["reason"] = reason
            return result

        if not isinstance(samples, (list, tuple)):
            return finish("invalid_samples")
        evidence["sampled_characters"] = len(samples)
        if len(samples) > MAX_SAMPLES:
            return finish("sample_limit_exceeded")
        if not samples:
            return result
        arrays, valid_rows, invalid = [], [], False
        for index, sample in enumerate(samples):
            character = sample.get("character") if isinstance(sample, dict) else None
            row = {"index": index, "character": character if isinstance(character, str) else None,
                   "status": "uncertain", "family": None, "candidates": [], "score": None,
                   "margin": None, "reason_code": "invalid_character"}
            result["glyph_predictions"].append(row)
            if not _character_in_script(character, script):
                invalid = True
                continue
            raster = preprocess_glyph(sample.get("image"))
            if raster is None:
                row["reason_code"] = "low_quality_glyph"
                invalid = True
                continue
            arrays.append(raster)
            valid_rows.append(row)
        evidence["valid_characters"] = len(arrays)
        evidence["distinct_characters"] = sorted({row["character"] for row in valid_rows})
        evidence["distinct_character_count"] = len(evidence["distinct_characters"])
        if not arrays:
            return finish("low_quality_or_invalid_glyphs")
        batch = np.ascontiguousarray(np.stack(arrays)[:, None], dtype=np.float32)
        try:
            outputs = self.session.run(["logits"], {"glyphs": batch})
        except Exception:
            return finish("neural_inference_failed")
        if (not isinstance(outputs, (list, tuple)) or len(outputs) != 1 or
                not isinstance(outputs[0], np.ndarray) or outputs[0].dtype != np.float32 or
                outputs[0].shape != (len(arrays), len(self.families)) or not np.isfinite(outputs[0]).all()):
            return finish("invalid_neural_output")
        # Restrict before softmax: Han-only families must not compete with
        # Latin-only ones, and metadata order is the ONNX class order.
        logits = outputs[0][:, self.indices[script]].astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        logits /= self.meta["temperature"][script]
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        by_character = defaultdict(list)
        for row, probability in zip(valid_rows, probabilities):
            candidates, score, margin = self._ranking(probability, script)
            reason = self._decision(score, margin, script)
            row.update(candidates=candidates, score=score, margin=margin, reason_code=reason)
            if reason == "neural_family_candidate":
                row.update(status="candidate", family=candidates[0]["family"])
            by_character[row["character"]].append(probability)
        combined = np.mean([np.mean(values, axis=0) for values in by_character.values()], axis=0)
        candidates, score, margin = self._ranking(combined, script)
        result.update(candidates=candidates, score=score, margin=margin)
        if not complete:
            return finish("incomplete_segmentation")
        if invalid:
            return finish("low_quality_or_invalid_glyphs")
        reason = self._decision(score, margin, script)
        if reason == "neural_family_candidate":
            result.update(status="candidate", family=candidates[0]["family"])
        return finish(reason)
