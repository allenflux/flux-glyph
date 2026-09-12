#!/usr/bin/env python3
"""Dual-output region CNN export QA, across separate torch and ORT envs.

reference: deterministic calibration tiles -> PyTorch outputs and source hashes.
check: the same tiles -> ORT at batch 1/7/32/128, both raw outputs and full
region aggregation/gate/size decisions. Failures are saved and exit nonzero.
No model weights, thresholds, labels or runtime modules are modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "src")]
SCHEMA = "flux-glyph-region-dual-output-parity-v1"
TOLERANCES = {"logits": {"atol": 2e-4, "rtol": 2e-4},
              "log_em_ratio": {"atol": 2e-5, "rtol": 1e-4},
              "probabilities": {"atol": 2e-5, "rtol": 1e-4},
              "size_px": {"atol": .02, "rtol": 2e-4}}
BATCHES = (1, 7, 32, 128)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError("Region parity: " + message)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def comparison(expected, actual, tolerance):
    expected, actual = np.asarray(expected), np.asarray(actual)
    result = {"passed": False, "expected_shape": list(expected.shape), "actual_shape": list(actual.shape), **tolerance}
    if expected.shape != actual.shape:
        return {**result, "reason": "shape_mismatch"}
    if not expected.size or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        return {**result, "reason": "empty_or_nonfinite_values"}
    delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    failed = delta > tolerance["atol"] + tolerance["rtol"] * np.abs(expected)
    return {**result, "passed": bool(not failed.any()), "max_absolute_error": float(delta.max()),
            "mean_absolute_error": float(delta.mean()), "elements_outside_tolerance": int(failed.sum()),
            "total_elements": int(delta.size)}


def aggregate(logits, logs, row, meta):
    """Independent numerical oracle matching the documented deployed formula."""
    scores = np.asarray(logits, dtype=np.float64)
    log_ratio = np.asarray(logs)
    require(scores.ndim == 2 and scores.shape[0] == len(log_ratio) and scores.shape[1] >= 2
            and np.isfinite(scores).all() and np.isfinite(log_ratio).all(), "invalid aggregate inputs")
    score = (scores - scores.max(axis=1, keepdims=True)) / meta["temperature"]
    probabilities = np.exp(score)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    average = probabilities.mean(axis=0)
    ranking = np.argsort(-average, kind="stable")
    top = int(ranking[0])
    confidence = float(average[top])
    margin = float(average[top] - average[ranking[1]])
    agreement = float((probabilities.argmax(axis=1) == top).mean())
    valid = bool(np.all(np.abs(log_ratio) <= 3) and row.get("whole_width_covered", False))
    # Reject out-of-contract values before exponentiation, exactly as runtime.
    if not valid:
        return {"probabilities": average, "top": top, "accepted": False, "size_accepted": False,
                "size_px": None, "reason": "invalid_output_or_uncovered_region"}
    ratio = float(np.exp(np.median(log_ratio)))
    ratios = np.exp(log_ratio.astype(np.float64))
    spread = float((np.quantile(ratios, .9) - np.quantile(ratios, .1)) / ratio)
    gates = meta["gates"]
    accepted = bool(confidence >= gates["min_score"] and margin > 1e-8 and margin >= gates["min_margin"]
                    and agreement >= gates["min_patch_agreement"])
    size_ok = bool(accepted and spread <= meta["max_size_relative_spread"])
    return {"probabilities": average, "top": top, "score": confidence, "margin": margin,
            "patch_agreement": agreement, "em_ratio": ratio, "size_relative_spread": spread,
            "accepted": accepted, "size_accepted": size_ok, "size_px": row["ink_height_px"] * ratio}


def choose_regions(rows, maximum=64):
    """Deterministic cal-only coverage of families and region tile counts."""
    pools = defaultdict(list)
    for index, row in enumerate(rows):
        pools[(row["family"], row["tile_count"])].append(index)
    selected, offset = [], 0
    while len(selected) < min(maximum, len(rows)):
        before = len(selected)
        for key in sorted(pools):
            if offset < len(pools[key]):
                selected.append(pools[key][offset])
                if len(selected) >= maximum:
                    break
        if len(selected) == before:
            break
        offset += 1
    return selected


def make_reference(run, data, output, maximum=64):
    import torch
    from region_network import RegionFontClassifier

    run, data, output = Path(run).resolve(), Path(data).resolve(), Path(output).resolve()
    require(not output.exists(), "reference output must be new")
    model_dir = run / "region"
    meta = json.loads((model_dir / "metadata.json").read_text())
    manifest_path = data / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    require(meta["schema"] == "flux-glyph-region-font-v1" and manifest["schema"] == "flux-glyph-native-region-training-data-v1",
            "wrong model/data schema")
    require(meta["families"] == manifest["families"] and meta["training"]["data_manifest_sha256"] == sha(manifest_path),
            "model was not trained from this calibration dataset")
    require(meta["training"]["selection_sha256"] == sha(run / "SELECTION_FREEZE.json"), "selection freeze differs")
    require(meta["model"]["sha256"] == sha(model_dir / meta["model"]["path"]), "ONNX SHA differs")
    paths = {}
    for kind, descriptor in manifest["splits"]["calibration"].items():
        if kind not in ("array", "metadata"):
            continue
        path = (data / descriptor["path"]).resolve()
        require(path.is_relative_to(data) and sha(path) == descriptor["sha256"], "calibration asset SHA differs")
        paths[kind] = path
    cal = json.loads(paths["metadata"].read_text())
    require(cal["split"] == "calibration", "reference may only sample calibration")
    tiles = np.load(paths["array"], mmap_mode="r", allow_pickle=False)
    require(tiles.dtype == np.float32 and tiles.ndim == 4 and tiles.shape[1:] == (1, 64, 256), "wrong region calibration shape")
    indices = choose_regions(cal["rows"], maximum)
    require(indices, "empty calibration reference")
    xs, regions, offset = [], [], 0
    for index in indices:
        row = cal["rows"][index]
        start, count = row["tile_start"], row["tile_count"]
        block = np.array(tiles[start:start + count], copy=True)
        require(len(block) == count and hashlib.sha256(block.tobytes()).hexdigest() == row["tiles_sha256"], "calibration region tiles SHA differs")
        xs.append(block)
        regions.append({key: row[key] for key in ("source_id", "region_id", "family", "ink_height_px", "whole_width_covered")} |
                       {"calibration_region_index": index, "tile_start": offset, "tile_count": count})
        offset += count
    x = np.concatenate(xs)
    checkpoint_path = model_dir / "model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint["families"] == meta["families"] and checkpoint["optimizer_steps"] == meta["training"]["selected_step"],
            "PyTorch checkpoint class order/selected step differs")
    state_digest = hashlib.sha256()
    for key, tensor in sorted(checkpoint["state_dict"].items()):
        state_digest.update(key.encode())
        state_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    require(state_digest.hexdigest() == meta["training"]["state_after_sha256"], "checkpoint parameter SHA differs from selected metadata")
    freeze = json.loads((run / "TRAINING_FREEZE.json").read_text())
    for name in ("training/region_network.py", "training/network.py"):
        require(sha(ROOT / name) == freeze["code_sha256"][name], "PyTorch architecture changed after training")
    net = RegionFontClassifier(len(meta["families"]))
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    torch.set_num_threads(4)
    logits, logs = [], []
    with torch.inference_mode():
        for start in range(0, len(x), 32):
            family, ratio = net(torch.from_numpy(x[start:start + 32]))
            logits.append(family.numpy()); logs.append(ratio.numpy())
    output.mkdir(parents=True)
    np.savez_compressed(output / "torch-reference.npz", tiles=x, logits=np.concatenate(logits), log_em_ratio=np.concatenate(logs))
    source = {"schema": SCHEMA, "scope": "deterministic calibration regions; no test pixels opened or tuning",
              "torch_version": torch.__version__, "torch_batch_size": 32, "families": meta["families"],
              "selected_step": checkpoint["optimizer_steps"], "regions": regions, "tolerances": TOLERANCES,
              "batches": list(BATCHES), "reference_sha256": sha(output / "torch-reference.npz"),
              "metadata": {"path": str(model_dir / "metadata.json"), "sha256": sha(model_dir / "metadata.json")},
              "checkpoint": {"path": str(checkpoint_path), "sha256": sha(checkpoint_path)},
              "onnx": {"path": str(model_dir / meta["model"]["path"]), "sha256": meta["model"]["sha256"]},
              "calibration_sources": {kind: {"path": str(path), "sha256": sha(path)} for kind, path in paths.items()},
              "evaluator_sha256": sha(__file__), "network_sha256": sha(ROOT / "training/region_network.py"),
              "runtime_aggregation_sha256": sha(ROOT / "src/flux_glyph/region_font.py")}
    dump(output / "reference-source.json", source)
    return {"reference": str(output), "regions": len(regions), "tiles": len(x), "test_opened": False}


def check_reference(reference, output):
    import onnxruntime as ort
    from flux_glyph.region_font import aggregate_predictions

    reference = Path(reference).resolve()
    source = json.loads((reference / "reference-source.json").read_text())
    require(source.get("schema") == SCHEMA and source.get("evaluator_sha256") == sha(__file__), "parity evaluator/source schema changed")
    require(source["tolerances"] == TOLERANCES and source["batches"] == list(BATCHES), "predeclared tolerances changed")
    require(source["runtime_aggregation_sha256"] == sha(ROOT / "src/flux_glyph/region_font.py"), "runtime aggregation changed since reference")
    for name in ("metadata", "checkpoint", "onnx"):
        require(sha(source[name]["path"]) == source[name]["sha256"], "pinned model changed: " + name)
    for entry in source["calibration_sources"].values():
        require(sha(entry["path"]) == entry["sha256"], "calibration reference source changed")
    require(sha(reference / "torch-reference.npz") == source["reference_sha256"], "reference outputs changed")
    meta = json.loads(Path(source["metadata"]["path"]).read_text())
    with np.load(reference / "torch-reference.npz", allow_pickle=False) as archive:
        x, expected_logits, expected_logs = (np.array(archive[key], copy=True) for key in ("tiles", "logits", "log_em_ratio"))
    require(x.dtype == np.float32 and x.ndim == 4 and x.shape[1:] == (1, 64, 256)
            and expected_logits.shape == (len(x), len(meta["families"])) and expected_logs.shape == (len(x),), "wrong dual-output reference shape")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(Path(source["onnx"]["path"]).read_bytes(), sess_options=options, providers=["CPUExecutionProvider"])
    require([i.name for i in session.get_inputs()] == ["tiles"] and [o.name for o in session.get_outputs()] == ["logits", "log_em_ratio"],
            "wrong region dual-output ONNX names")
    results = []
    for batch in BATCHES:
        outputs = [session.run(["logits", "log_em_ratio"], {"tiles": x[start:start + batch]}) for start in range(0, len(x), batch)]
        actual_logits, actual_logs = (np.concatenate([item[i] for item in outputs]) for i in (0, 1))
        raw = {"logits": comparison(expected_logits, actual_logits, TOLERANCES["logits"]),
               "log_em_ratio": comparison(expected_logs, actual_logs, TOLERANCES["log_em_ratio"])}
        regions = []
        aggregatable = (actual_logits.shape == expected_logits.shape and actual_logs.shape == expected_logs.shape
                        and np.isfinite(actual_logits).all() and np.isfinite(actual_logs).all())
        for row in source["regions"] if aggregatable else []:
            start, end = row["tile_start"], row["tile_start"] + row["tile_count"]
            expected = aggregate(expected_logits[start:end], expected_logs[start:end], row, meta)
            actual = aggregate(actual_logits[start:end], actual_logs[start:end], row, meta)
            runtime = aggregate_predictions(actual_logits[start:end], actual_logs[start:end], temperature=meta["temperature"])
            checks = {"probabilities": comparison(expected["probabilities"], actual["probabilities"], TOLERANCES["probabilities"]),
                      "runtime_aggregation_probability": comparison(actual["probabilities"], runtime["probabilities"], {"atol": 1e-12, "rtol": 1e-12})}
            if expected["size_px"] is not None and actual["size_px"] is not None:
                checks["size_px"] = comparison([expected["size_px"]], [actual["size_px"]], TOLERANCES["size_px"])
                checks["runtime_aggregation_size"] = comparison([actual["size_px"]], [row["ink_height_px"] * runtime["em_ratio"]], {"atol": 1e-9, "rtol": 1e-9})
            decisions = {key: expected[key] == actual[key] for key in ("top", "accepted", "size_accepted")}
            regions.append({"source_id": row["source_id"], "region_id": row["region_id"], "tile_count": row["tile_count"],
                            "passed": all(check["passed"] for check in checks.values()) and all(decisions.values()),
                            "checks": checks, "decisions_identical": decisions,
                            "torch_font_size_px": expected["size_px"], "onnx_font_size_px": actual["size_px"]})
        results.append({"batch_size": batch, "passed": all(item["passed"] for item in raw.values()) and all(row["passed"] for row in regions),
                        "outputs": raw, "regions": regions})
    report = {"schema": SCHEMA, "passed": all(item["passed"] for item in results), "onnxruntime_version": ort.__version__,
              "torch_version": source["torch_version"], "source": source, "test_opened": False, "batches": results,
              "failures_are_not_relabelled_as_pass": True}
    dump(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("reference")
    reference.add_argument("--run", type=Path, required=True)
    reference.add_argument("--data", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--regions", type=int, default=64)
    check = commands.add_parser("check")
    check.add_argument("--reference", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "reference":
        require(1 <= args.regions <= 256, "invalid sample region limit")
        print(json.dumps(make_reference(args.run, args.data, args.output, args.regions), indent=2))
    else:
        report = check_reference(args.reference, args.output)
        print(json.dumps({"passed": report["passed"], "report": str(args.output), "batches": [
            {"batch_size": row["batch_size"], "passed": row["passed"], "outputs": row["outputs"]} for row in report["batches"]]}, indent=2))
        sys.exit(0 if report["passed"] else 1)
