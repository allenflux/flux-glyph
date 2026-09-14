"""Train-only one-sided confidence retention for verified native TRAIN rows."""
from __future__ import annotations

import json
import math
from prepare_unified_regions import FAMILIES

BATCH_SIZE = 96
NUM_CLASSES = 25
UNKNOWN_INDEX = FAMILIES.index("__unknown__")
CONTRACT = {
    "name": "short-confidence-retention-v2", "batch_size": 96, "classes": 25,
    "min_probability": 0.8, "outer_weight": 2.0, "full_KL_preserved": True,
    "runtime_changes": False, "single_model": True, "train_only": True,
    "requires_native_verified_train": True,
    "eligibility": "same-tile teacher has unique correct argmax against native TRAIN truth",
    "floor": "teacher true probability, raised to .8 for one-tile known or true unknown rows",
    "normalization": "sum over all 96 replay rows; zero for ineligible rows",
    "teacher_cache_deployed": False,
}


def _fail(message):
    raise ValueError(message)


def confidence_retention_loss(logits, teacher_logits, targets, rows, families=FAMILIES):
    """Return ``(weighted_loss, eligible, boosted)`` for one fixed TRAIN batch.

    ``eligible`` requires a correct frozen-teacher top-1. Known one-tile rows and
    all true unknown rows receive a floor of max(teacher probability, .8); known
    multi-tile rows retain the teacher probability. Rows with wrong teachers
    receive no extra gradient (the caller's base CE remains in force).
    """
    import torch
    from torch.nn import functional as F
    if not isinstance(logits, torch.Tensor) or logits.shape != (BATCH_SIZE, NUM_CLASSES):
        _fail("logits must have shape [96,25]")
    if logits.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(logits).all()):
        _fail("logits must be finite float32/float64")
    if not isinstance(teacher_logits, torch.Tensor) or teacher_logits.shape != logits.shape:
        _fail("teacher_logits must have shape [96,25]")
    if teacher_logits.dtype != logits.dtype or teacher_logits.device != logits.device:
        _fail("teacher_logits must match logits dtype/device")
    if teacher_logits.requires_grad or not bool(torch.isfinite(teacher_logits).all()):
        _fail("teacher_logits must be finite and detached")
    if not isinstance(targets, torch.Tensor) or targets.shape != (BATCH_SIZE,) or targets.dtype != torch.int64:
        _fail("targets must be int64 shape [96]")
    if targets.device != logits.device or targets.requires_grad or not bool(((targets >= 0) & (targets < NUM_CLASSES)).all()):
        _fail("targets must be detached class indices on logits device")
    if list(families) != list(FAMILIES):
        _fail("invalid 25-class family vocabulary")
    if not isinstance(rows, (list, tuple)) or len(rows) != BATCH_SIZE:
        _fail("rows must contain exactly 96 TRAIN rows")
    row_targets = []
    eligible_meta = []
    for row in rows:
        if not isinstance(row, dict) or row.get("split") != "train" or row.get("native_font_verified") is not True:
            _fail("confidence retention requires native verified TRAIN rows")
        target = row.get("target")
        if type(target) is not int or not 0 <= target < NUM_CLASSES:
            _fail("invalid TRAIN row target")
        if row.get("family") != families[target]:
            _fail("TRAIN row target/family mismatch")
        if type(row.get("tile_count")) is not int or not 1 <= row["tile_count"] <= 8:
            _fail("invalid TRAIN tile_count")
        row_targets.append(target)
        eligible_meta.append(target == UNKNOWN_INDEX or row["tile_count"] == 1)
    native_targets = torch.tensor(row_targets, dtype=torch.int64, device=targets.device)
    if not torch.equal(native_targets, targets):
        _fail("targets do not match native TRAIN rows")
    with torch.no_grad():
        teacher_top1 = teacher_logits.argmax(dim=1)
        top_two = teacher_logits.topk(2, dim=1).values
        teacher_logp = F.log_softmax(teacher_logits, dim=1).gather(1, targets[:, None]).squeeze(1)
        eligible = (teacher_top1 == targets) & (top_two[:, 0] > top_two[:, 1])
        boosted = (eligible & torch.tensor(eligible_meta, dtype=torch.bool, device=targets.device)
                   & (teacher_logp < math.log(CONTRACT['min_probability'])))
        target_logp = torch.where(boosted, teacher_logp.clamp_min(math.log(CONTRACT['min_probability'])), teacher_logp)
    student_logp = F.log_softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
    gap = F.relu(target_logp - student_logp) * eligible
    return CONTRACT['outer_weight'] * gap.sum() / BATCH_SIZE, eligible, boosted


def contract_json():
    return json.loads(json.dumps(CONTRACT, sort_keys=True))
