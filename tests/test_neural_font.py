"""Neural scoring, preprocessing parity and untrusted asset contracts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from flux_glyph.glyph_preprocess import extract_glyphs
import flux_glyph.neural_font as neural


def glyph(dark=True):
    image = Image.new("L", (36, 40), 255 if dark else 0)
    drawing = ImageDraw.Draw(image)
    ink = 0 if dark else 255
    drawing.rectangle((8, 6, 12, 32), fill=ink)
    drawing.rectangle((8, 9, 27, 13), fill=ink)
    drawing.rectangle((8, 23, 27, 27), fill=ink)
    drawing.rectangle((23, 13, 27, 27), fill=ink)
    return image


class FakeSession:
    def __init__(self):
        self.inputs = [SimpleNamespace(name="glyphs", type="tensor(float)", shape=["N", 1, 64, 64])]
        self.outputs = [SimpleNamespace(name="logits", type="tensor(float)", shape=["N", 4])]
        self.logits = np.array([[8., 0., 10., 20.]], dtype=np.float32)
        self.calls = []

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, requested, supplied):
        self.calls.append((requested, supplied))
        if self.logits is None:
            raise RuntimeError("model failed")
        count = len(supplied["glyphs"])
        return [np.repeat(self.logits, count, axis=0) if self.logits.shape[0] == 1 else self.logits]


def asset(tmp_path):
    directory = tmp_path / "neural"
    directory.mkdir()
    model = b"test ONNX bytes; the session is replaced only in contract tests"
    (directory / "model.onnx").write_bytes(model)
    metadata = {"schema": neural.SCHEMA, "algorithm": neural.ALGORITHM,
                "model": {"path": "model.onnx", "sha256": hashlib.sha256(model).hexdigest()},
                "families": ["PingFang SC", "MiSans", "SF Pro", "Alipay Number"],
                "scripts": {"han": ["PingFang SC", "MiSans"],
                            "latin": ["MiSans", "SF Pro", "Alipay Number"]},
                "gates": {script: {"min_score": .7, "min_margin": .2} for script in ("han", "latin")},
                "temperature": {"han": 1., "latin": 1.}}
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return directory, metadata


@pytest.fixture
def classifier(tmp_path, monkeypatch):
    directory, _ = asset(tmp_path)
    session = FakeSession()
    captured = {}

    def create(data, *, sess_options, providers):
        captured.update(data=data, options=sess_options, providers=providers)
        return session

    monkeypatch.setattr(neural.ort, "InferenceSession", create)
    engine = neural.NeuralFontClassifier(directory)
    engine.session_settings = captured
    return engine


@pytest.mark.parametrize("dark", [True, False])
def test_preprocessing_matches_training_extraction_exactly(dark):
    image = glyph(dark)
    actual = neural.preprocess_glyph(image)
    np.testing.assert_array_equal(actual, extract_glyphs(image, 1).glyphs[0])
    assert actual.shape == (64, 64) and actual.dtype == np.float32
    assert actual.flags.c_contiguous and 0 <= actual.min() <= actual.max() <= 1


def test_preprocessing_rejects_blank_tiny_low_contrast_and_oversize():
    low_contrast = np.full((32, 32), 240, dtype=np.uint8)
    low_contrast[5:27, 8:18] = 230
    assert neural.preprocess_glyph(Image.fromarray(low_contrast)) is None
    assert neural.preprocess_glyph(Image.new("L", (32, 32), 255)) is None
    assert neural.preprocess_glyph(Image.new("L", (3, 40))) is None
    assert neural.preprocess_glyph(Image.new("L", (4097, 4))) is None
    assert neural.preprocess_glyph(Image.new("L", (2049, 2049))) is None
    assert neural.preprocess_glyph(None) is None


def test_single_character_can_pass_neural_gate_and_script_mask(classifier):
    result = classifier.predict([{"character": "青", "image": glyph()}])
    assert result["status"] == "candidate" and result["family"] == "PingFang SC"
    assert result["method"] == "neural_network"
    assert result["score"] > .99 and result["margin"] > .99
    assert result["evidence"]["distinct_characters"] == ["青"]
    assert result["evidence"]["distinct_character_count"] == 1
    assert {candidate["family"] for candidate in result["candidates"]} == {"PingFang SC", "MiSans"}
    requested, supplied = classifier.session.calls[0]
    assert requested == ["logits"]
    assert supplied["glyphs"].dtype == np.float32 and supplied["glyphs"].shape == (1, 1, 64, 64)
    assert result["glyph_predictions"][0]["family"] == "PingFang SC"
    json.dumps(result, allow_nan=False)


def test_cpu_execution_uses_checked_bytes_and_one_thread(classifier):
    settings = classifier.session_settings
    assert isinstance(settings["data"], bytes)
    assert settings["providers"] == ["CPUExecutionProvider"]
    assert settings["options"].intra_op_num_threads == 1
    assert settings["options"].inter_op_num_threads == 1
    assert settings["options"].execution_mode == neural.ort.ExecutionMode.ORT_SEQUENTIAL


def test_latin_mask_uses_model_registry_order(classifier):
    classifier.session.logits = np.array([[1000., 0., 8., 0.]], dtype=np.float32)
    result = classifier.predict([{"character": "a", "image": glyph()}], script="latin")
    assert result["status"] == "candidate" and result["family"] == "SF Pro"
    assert "PingFang SC" not in {candidate["family"] for candidate in result["candidates"]}


def test_duplicates_do_not_outvote_other_distinct_characters(classifier):
    high = [np.log(9.), 0., 0., 0.]
    low = [0., np.log(9.), 0., 0.]
    classifier.session.logits = np.asarray([high, low], dtype=np.float32)
    samples = [{"character": character, "image": glyph()} for character in "青字"]
    baseline = classifier.predict(samples)
    classifier.session.logits = np.asarray([high] * 10 + [low], dtype=np.float32)
    repeated = classifier.predict([samples[0]] * 10 + [samples[1]])
    assert repeated["score"] == pytest.approx(baseline["score"], abs=1e-7)
    assert repeated["margin"] == pytest.approx(baseline["margin"], abs=1e-7)
    assert repeated["status"] == "uncertain" and repeated["family"] is None
    assert repeated["evidence"]["distinct_character_count"] == 2
    assert repeated["evidence"]["sampled_characters"] == 11


def test_temperature_and_gates_control_abstention(classifier):
    classifier.session.logits = np.array([[2., 0., 0., 0.]], dtype=np.float32)
    samples = [{"character": "青", "image": glyph()}]
    assert classifier.predict(samples)["status"] == "candidate"
    classifier.meta["temperature"]["han"] = 10.
    result = classifier.predict(samples)
    assert result["status"] == "uncertain" and result["reason_code"] == "below_score_gate"
    assert result["candidates"][0]["family"] == "PingFang SC"
    classifier.meta["temperature"]["han"] = 1.
    classifier.meta["gates"]["han"] = {"min_score": .5, "min_margin": .9}
    assert classifier.predict(samples)["reason_code"] == "ambiguous_neural_families"


def test_incomplete_and_bad_glyphs_never_promote_partial_result(classifier):
    sample = {"character": "青", "image": glyph()}
    result = classifier.predict([sample], complete=False)
    assert result["status"] == "uncertain" and result["family"] is None
    assert result["reason_code"] == "incomplete_segmentation" and result["candidates"]
    result = classifier.predict([sample, {"character": "字", "image": Image.new("L", (32, 32), 255)}])
    assert result["status"] == "uncertain" and result["family"] is None
    assert result["reason_code"] == "low_quality_or_invalid_glyphs"
    assert result["evidence"]["valid_characters"] == 1
    assert result["glyph_predictions"][1]["reason_code"] == "low_quality_glyph"


@pytest.mark.parametrize("samples,reason", [([], "no_samples"), (None, "invalid_samples"),
                                         ([None] * 129, "sample_limit_exceeded"),
                                         ([{"character": "A"}], "low_quality_or_invalid_glyphs")])
def test_invalid_inputs_do_not_run_model(classifier, samples, reason):
    assert classifier.predict(samples)["reason_code"] == reason
    assert classifier.session.calls == []


@pytest.mark.parametrize("invalid", [np.array([[float("nan"), 1., 1., 1.]], dtype=np.float32),
                                     np.ones((1, 4), dtype=np.float64),
                                     np.ones((1, 3), dtype=np.float32),
                                     np.ones((2, 4), dtype=np.float32)])
def test_invalid_runtime_logits_fail_closed(classifier, invalid):
    classifier.session.logits = invalid
    result = classifier.predict([{"character": "青", "image": glyph()}])
    assert result["status"] == "uncertain" and result["reason_code"] == "invalid_neural_output"
    assert result["family"] is None and result["candidates"] == []


def test_inference_failure_and_unsupported_script(classifier):
    classifier.session.logits = None
    result = classifier.predict([{"character": "青", "image": glyph()}])
    assert result["reason_code"] == "neural_inference_failed" and result["family"] is None
    with pytest.raises(ValueError, match="script"):
        classifier.predict([], script="arabic")
    with pytest.raises(ValueError, match="boolean"):
        classifier.predict([], complete="yes")


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(schema="reference-bank"),
    lambda m: m.update(algorithm="blur32"),
    lambda m: m.update(families=["a", "a"]),
    lambda m: m["scripts"].update(han=["PingFang SC", "unknown"]),
    lambda m: m["temperature"].update(han=0),
    lambda m: m["temperature"].update(han=True),
    lambda m: m["temperature"].update(han=float("nan")),
    lambda m: m["gates"]["han"].update(min_score=1.1),
    lambda m: m["gates"]["han"].update(min_margin=-.1),
    lambda m: m["model"].update(path="../model.onnx"),
    lambda m: m["model"].update(path="model.pkl"),
    lambda m: m["model"].update(sha256="invalid"),
])
def test_invalid_metadata_rejected_before_onnx_load(tmp_path, monkeypatch, mutation):
    directory, metadata = asset(tmp_path)
    mutation(metadata)
    (directory / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(neural.ort, "InferenceSession", lambda *a, **k: pytest.fail("must validate first"))
    with pytest.raises(ValueError):
        neural.NeuralFontClassifier(directory)


def test_model_checksum_and_asset_bounds_before_load(tmp_path, monkeypatch):
    directory, _ = asset(tmp_path)
    monkeypatch.setattr(neural.ort, "InferenceSession", lambda *a, **k: pytest.fail("must validate first"))
    with (directory / "model.onnx").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        neural.NeuralFontClassifier(directory)
    monkeypatch.setattr(neural, "MAX_MODEL_BYTES", 1)
    with pytest.raises(ValueError, match="size limit"):
        neural.NeuralFontClassifier(directory)


def test_model_symlink_outside_directory_rejected(tmp_path, monkeypatch):
    directory, _ = asset(tmp_path)
    original = directory / "model.onnx"
    target = tmp_path / "other.onnx"
    original.rename(target)
    original.symlink_to(target)
    monkeypatch.setattr(neural.ort, "InferenceSession", lambda *a, **k: pytest.fail("must validate first"))
    with pytest.raises(ValueError, match="escapes"):
        neural.NeuralFontClassifier(directory)


@pytest.mark.parametrize("kind,changes", [
    ("inputs", {"name": "wrong"}),
    ("inputs", {"type": "tensor(double)"}),
    ("inputs", {"shape": ["N", 3, 64, 64]}),
    ("inputs", {"shape": [128, 1, 64, 64]}),
    ("outputs", {"name": "probabilities"}),
    ("outputs", {"shape": ["N", 3]}),
    ("outputs", {"shape": ["N", "C"]}),
    ("outputs", {"type": "tensor(int64)"}),
])
def test_onnx_tensor_contract_is_validated(tmp_path, monkeypatch, kind, changes):
    directory, _ = asset(tmp_path)
    session = FakeSession()
    for key, value in changes.items():
        setattr(getattr(session, kind)[0], key, value)
    monkeypatch.setattr(neural.ort, "InferenceSession", lambda *a, **k: session)
    with pytest.raises(ValueError, match="ONNX"):
        neural.NeuralFontClassifier(directory)


def test_duplicate_onnx_inputs_rejected(tmp_path, monkeypatch):
    directory, _ = asset(tmp_path)
    session = FakeSession()
    session.inputs.append(deepcopy(session.inputs[0]))
    monkeypatch.setattr(neural.ort, "InferenceSession", lambda *a, **k: session)
    with pytest.raises(ValueError, match="count mismatch"):
        neural.NeuralFontClassifier(directory)


def real_asset(tmp_path, name):
    directory, metadata = asset(tmp_path)
    fixtures = Path(__file__).parent / "fixtures" / "neural_contract"
    # These tiny ONNX fixtures contain ReduceMean -> MatMul with four fixed
    # scalar weights [1,2,3,4], opset 13 / IR 8. They validate runtime behavior,
    # not font accuracy, and require no ONNX authoring dependency in production.
    data = (fixtures / name).read_bytes()
    (directory / "model.onnx").write_bytes(data)
    metadata["model"]["sha256"] = hashlib.sha256(data).hexdigest()
    metadata["temperature"] = {"han": .01, "latin": .01}
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return directory


def test_real_onnx_cpu_session_runs_embedded_weights(tmp_path):
    directory = real_asset(tmp_path, "embedded.onnx")
    classifier = neural.NeuralFontClassifier(directory)
    actual = classifier.session.run(["logits"], {"glyphs": np.ones((2, 1, 64, 64), dtype=np.float32)})[0]
    np.testing.assert_array_equal(actual, np.array([[1., 2., 3., 4.]] * 2, dtype=np.float32))
    result = classifier.predict([{"character": "青", "image": glyph()}])
    assert result["status"] == "candidate" and result["family"] == "MiSans"
    assert result["method"] == "neural_network" and result["score"] > .99


def test_real_onnx_external_initializers_cannot_resolve_in_bytes_mode(tmp_path):
    directory = real_asset(tmp_path, "external.onnx")
    fixtures = Path(__file__).parent / "fixtures" / "neural_contract"
    shutil.copyfile(fixtures / "weights.bin", directory / "weights.bin")
    options = neural.ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    # First establish that the external file exists and the model is otherwise
    # valid: path loading succeeds, but the classifier's byte loading rejects it.
    session = neural.ort.InferenceSession(str(directory / "model.onnx"), sess_options=options,
                                         providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(session.run(["logits"], {"glyphs": np.ones((1, 1, 64, 64), dtype=np.float32)})[0],
                                  np.array([[1., 2., 3., 4.]], dtype=np.float32))
    with pytest.raises(ValueError, match="Invalid neural ONNX model"):
        neural.NeuralFontClassifier(directory)
