#!/usr/bin/env python3
"""Export frozen region weights with explicit, centred GroupNorm variance.

This changes only ONNX arithmetic lowering. Original PyTorch GroupNorm remains
the parity oracle; no weights, training choices or tolerances are changed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from region_network import RegionFontClassifier


class CenteredGroupNorm(nn.Module):
    def __init__(self, original, high_precision=False):
        super().__init__()
        self.num_groups, self.eps = original.num_groups, original.eps
        self.high_precision = high_precision
        self.weight = nn.Parameter(original.weight.detach().clone())
        self.bias = nn.Parameter(original.bias.detach().clone())

    def forward(self, value):
        grouped = value.reshape(value.shape[0], self.num_groups, -1)
        if self.high_precision:
            grouped = grouped.to(torch.float64)
        centered = grouped - grouped.mean(dim=2, keepdim=True)
        variance = (centered * centered).mean(dim=2, keepdim=True)
        normalized = (centered / torch.sqrt(variance + self.eps)).to(value.dtype).reshape_as(value)
        return normalized * self.weight.reshape(1, -1, 1, 1) + self.bias.reshape(1, -1, 1, 1)


def replace_groupnorm(module, high_precision=False):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, CenteredGroupNorm(child, high_precision)); count += 1
        else:
            count += replace_groupnorm(child, high_precision)
    return count


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(run, output, high_precision=False):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Stable export output must be new; preserve original model and failure report")
    original = run / "region"
    checkpoint = torch.load(original / "model.pth", map_location="cpu", weights_only=True)
    metadata = json.loads((original / "metadata.json").read_text())
    if checkpoint["families"] != metadata["families"]:
        raise ValueError("Frozen model class order differs")
    model = RegionFontClassifier(len(checkpoint["families"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    count = replace_groupnorm(model, high_precision)
    if count != 4 or any(not torch.equal(value, checkpoint["state_dict"][key]) for key, value in model.state_dict().items()):
        raise ValueError("Lowering altered frozen parameters or unexpected normalization count")
    model.eval(); torch.set_num_threads(4)
    output.mkdir(parents=True)
    torch.onnx.export(model, torch.zeros(2, 1, 64, 256), output / "model.onnx", input_names=["tiles"],
                      output_names=["logits", "log_em_ratio"], dynamic_axes={"tiles": {0: "batch"}, "logits": {0: "batch"}, "log_em_ratio": {0: "batch"}},
                      opset_version=17, dynamo=False)
    metadata = copy.deepcopy(metadata)
    metadata["model"] = {"path": "model.onnx", "sha256": sha(output / "model.onnx")}
    metadata["export"] = {"groupnorm_lowering": "reshape_centered_mean_square_variance_v1", "frozen_parameters_unchanged": True,
                          "reduction_dtype": "float64" if high_precision else "float32",
                          "original_onnx_sha256": sha(original / "model.onnx"), "checkpoint_sha256": sha(original / "model.pth"),
                          "export_source_sha256": sha(__file__), "parity_reference": "original torch.nn.GroupNorm, unchanged tolerances"}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    return {"output": str(output), "sha256": metadata["model"]["sha256"], "groupnorm_count": count,
            "parameters_changed": False, "requires_independent_parity": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--float64-reduction", action="store_true", help="Accumulate only GroupNorm reductions in double precision")
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.output, args.float64_reduction), indent=2))
