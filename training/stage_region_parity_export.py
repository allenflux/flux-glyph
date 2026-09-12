#!/usr/bin/env python3
"""Bind an alternate ONNX export to the unchanged original torch reference."""
import argparse
import json
from pathlib import Path
import shutil

from region_parity import sha, require, dump


def stage(reference, export, output):
    reference, export, output = Path(reference).resolve(), Path(export).resolve(), Path(output).resolve()
    require(not output.exists(), "alternate reference output must be new")
    source_path = reference / "reference-source.json"
    source = json.loads(source_path.read_text())
    require(sha(reference / "torch-reference.npz") == source["reference_sha256"], "original PyTorch fixture changed")
    original_meta = json.loads(Path(source["metadata"]["path"]).read_text())
    require(sha(source["metadata"]["path"]) == source["metadata"]["sha256"], "original metadata changed")
    meta_path = export / "metadata.json"
    new_meta = json.loads(meta_path.read_text())
    require({k: v for k, v in original_meta.items() if k not in ("model", "export")} ==
            {k: v for k, v in new_meta.items() if k not in ("model", "export")}, "alternate export changed trained metadata or gates")
    evidence = new_meta.get("export", {})
    require(evidence.get("frozen_parameters_unchanged") is True and
            evidence.get("checkpoint_sha256") == source["checkpoint"]["sha256"], "export is not bound to frozen original parameters")
    model_path = (export / new_meta["model"]["path"]).resolve()
    require(model_path.is_relative_to(export) and sha(model_path) == new_meta["model"]["sha256"], "alternate ONNX SHA differs")
    source["export_override"] = {"reason": "Alternate GroupNorm lowering; original PyTorch outputs and tolerances unchanged",
                                 "original_source_sha256": sha(source_path), "original_onnx_sha256": source["onnx"]["sha256"]}
    source["metadata"] = {"path": str(meta_path), "sha256": sha(meta_path)}
    source["onnx"] = {"path": str(model_path), "sha256": sha(model_path)}
    output.mkdir(parents=True)
    shutil.copyfile(reference / "torch-reference.npz", output / "torch-reference.npz")
    require(sha(output / "torch-reference.npz") == source["reference_sha256"], "copied torch oracle differs")
    dump(output / "reference-source.json", source)
    return {"output": str(output), "original_torch_reference_sha256": source["reference_sha256"], "onnx_sha256": source["onnx"]["sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.reference, args.export, args.output), indent=2))
