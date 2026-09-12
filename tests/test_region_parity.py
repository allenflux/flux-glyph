import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("region_parity", ROOT / "training/region_parity.py")
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


def metadata():
    return {"temperature": 1., "gates": {"min_score": .7, "min_margin": .1, "min_patch_agreement": .7},
            "max_size_relative_spread": .2, "families": ["A", "B"]}


def test_size_uses_exponential_of_median_logs_not_median_exponentials():
    value = parity.aggregate(np.asarray([[4., 0.], [4., 0.]]), np.log(np.asarray([1., 9.])),
                             {"ink_height_px": 20, "whole_width_covered": True}, metadata())
    assert value["size_px"] == pytest.approx(60.)
    assert value["size_px"] != pytest.approx(100.)
    assert value["accepted"] is True and value["size_accepted"] is False  # inconsistent patch sizes


def test_top_class_is_mean_probability_not_mean_logits():
    logits = np.asarray([[100., 0.], [-2., 0.], [-2., 0.]])
    value = parity.aggregate(logits, np.zeros(3), {"ink_height_px": 20, "whole_width_covered": True}, metadata())
    assert value["top"] == 1
    assert value["top"] != int(logits.mean(axis=0).argmax())


def test_uncovered_region_and_invalid_ratio_fail_gates():
    for row, logs in [({"ink_height_px": 20, "whole_width_covered": False}, [0.]),
                      ({"ink_height_px": 20, "whole_width_covered": True}, [3.01])]:
        value = parity.aggregate(np.asarray([[4., 0.]]), np.asarray(logs), row, metadata())
        assert not value["accepted"] and value["size_px"] is None


def test_comparison_reports_shape_nonfinite_and_finite_tolerance_failures():
    tol = {"atol": 1e-5, "rtol": 1e-4}
    assert parity.comparison([1.], [1.00001], tol)["passed"]
    assert not parity.comparison([1.], [1.01], tol)["passed"]
    assert parity.comparison([1.], [[1.]], tol)["reason"] == "shape_mismatch"
    assert parity.comparison([1.], [np.nan], tol)["reason"] == "empty_or_nonfinite_values"


def test_reference_sampler_covers_family_and_tile_count_without_targets():
    rows = [{"family": family, "tile_count": count} for family in ("A", "B") for count in (1, 2, 3) for _ in range(3)]
    indices = parity.choose_regions(rows, 6)
    assert {(rows[i]["family"], rows[i]["tile_count"]) for i in indices} == {(f, c) for f in ("A", "B") for c in (1, 2, 3)}
    assert indices == parity.choose_regions(rows, 6)


def test_alternate_export_binding_preserves_oracle_and_rejects_changed_gate(tmp_path):
    from stage_region_parity_export import stage
    original, alternate = tmp_path / "original", tmp_path / "alternate"
    original.mkdir(); alternate.mkdir()
    (original / "torch-reference.npz").write_bytes(b"unchanged-original-PyTorch-outputs")
    base = {**metadata(), "model": {"path": "model.onnx", "sha256": "a" * 64}, "training": {"selected_step": 3000}}
    parity.dump(original / "metadata.json", base)
    (alternate / "model.onnx").write_bytes(b"alternative-export")
    new = {**base, "model": {"path": "model.onnx", "sha256": parity.sha(alternate / "model.onnx")},
           "export": {"frozen_parameters_unchanged": True, "checkpoint_sha256": "b" * 64}}
    parity.dump(alternate / "metadata.json", new)
    source = {"reference_sha256": parity.sha(original / "torch-reference.npz"),
              "metadata": {"path": str(original / "metadata.json"), "sha256": parity.sha(original / "metadata.json")},
              "checkpoint": {"sha256": "b" * 64}, "onnx": {"sha256": "a" * 64}, "tolerances": parity.TOLERANCES}
    parity.dump(original / "reference-source.json", source)
    stage(original, alternate, tmp_path / "bound")
    assert (tmp_path / "bound/torch-reference.npz").read_bytes() == (original / "torch-reference.npz").read_bytes()
    new["gates"] = {**new["gates"], "min_score": .1}
    parity.dump(alternate / "metadata.json", new)
    with pytest.raises(ValueError, match="metadata or gates"):
        stage(original, alternate, tmp_path / "refused")


@pytest.mark.parametrize("fault", [None, "logits", "log_em_ratio", "nonfinite"])
def test_checker_validates_both_outputs_all_batches_and_persists_failure(tmp_path, monkeypatch, fault):
    # A minimal session double exercises the real orchestration and persisted
    # pass/fail report without installing torch/onnx or testing random weights.
    tiles = np.zeros((3, 1, 64, 256), dtype=np.float32)
    tiles[:, 0, 0, 0] = np.arange(3)
    logits = np.asarray([[4., 0.], [0., 4.], [4., 0.]], dtype=np.float32)
    logs = np.asarray([.1, .2, .3], dtype=np.float32)
    reference = tmp_path / "reference"
    reference.mkdir()
    np.savez_compressed(reference / "torch-reference.npz", tiles=tiles, logits=logits, log_em_ratio=logs)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"session-test-double")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint-test-double")
    meta_path = tmp_path / "metadata.json"
    parity.dump(meta_path, metadata())
    source = {"schema": parity.SCHEMA, "evaluator_sha256": parity.sha(parity.__file__),
              "tolerances": parity.TOLERANCES, "batches": list(parity.BATCHES),
              "runtime_aggregation_sha256": parity.sha(ROOT / "src/flux_glyph/region_font.py"),
              "reference_sha256": parity.sha(reference / "torch-reference.npz"), "torch_version": "test-double",
              "calibration_sources": {}, "regions": [{"source_id": "cal-1", "region_id": "row-1", "tile_start": 0,
                  "tile_count": 3, "ink_height_px": 50, "whole_width_covered": True}]}
    for name, path in (("metadata", meta_path), ("checkpoint", checkpoint), ("onnx", model)):
        source[name] = {"path": str(path), "sha256": parity.sha(path)}
    parity.dump(reference / "reference-source.json", source)
    batches = []

    class Session:
        def __init__(self, *args, **kwargs): pass
        def get_inputs(self): return [SimpleNamespace(name="tiles")]
        def get_outputs(self): return [SimpleNamespace(name="logits"), SimpleNamespace(name="log_em_ratio")]
        def run(self, outputs, inputs):
            assert outputs == ["logits", "log_em_ratio"] and set(inputs) == {"tiles"}
            x = inputs["tiles"]
            assert x.shape[1:] == (1, 64, 256)
            batches.append(len(x))
            indices = x[:, 0, 0, 0].astype(int)
            first, second = logits[indices].copy(), logs[indices].copy()
            if fault == "logits": first += .1
            if fault == "log_em_ratio": second += .01
            if fault == "nonfinite": second[:] = np.nan
            return first, second

    import onnxruntime
    monkeypatch.setattr(onnxruntime, "InferenceSession", Session)
    report_path = tmp_path / "parity.json"
    report = parity.check_reference(reference, report_path)
    assert report["passed"] is (fault is None)
    assert [item["batch_size"] for item in report["batches"]] == [1, 7, 32, 128]
    assert len(batches) == 6 and batches[:3] == [1, 1, 1]
    assert json.loads(report_path.read_text())["passed"] is (fault is None)
    assert report["test_opened"] is False
