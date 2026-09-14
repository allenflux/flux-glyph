"""Native CPU checks for bounded three/four-glyph R22 continuation v3."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training import short_region_objective as short
from training import train_unified_retention_micro_recovery as r22
from training import train_unified_short_regions_v3 as trainer
from training.prepare_unified_regions import FAMILIES


def _replay_rows():
    rows = []
    for index in range(96):
        target = index % 25
        rows.append({
            "split": "train",
            "native_font_verified": True,
            "target": target,
            "family": FAMILIES[target],
            "source_font_family": FAMILIES[target],
            "domain": "android",
            "view": "native",
            "tile_start": index,
            "tile_count": 1,
            "log_em_ratio": 0.0,
            "source_id": f"source-{index}",
            "region_id": f"region-{index}",
        })
    return rows


def _short_rows(active):
    rows = []
    for target, family in enumerate(FAMILIES[:-1]):
        glyph_count = 3 + target % 2 if active else 1 + target % 2
        rows.append({"split": "train", "native_font_verified": True,
                     "target": target, "family": family,
                     "source_font_family": family, "glyph_count": glyph_count})
    for index in range(8):
        rows.append({"split": "train", "native_font_verified": True,
                     "target": 24, "family": "__unknown__",
                     "source_font_family": short.SHORT_UNKNOWN_SOURCES[index],
                     "glyph_count": 4 if active else 2})
    return rows


def _batch(active):
    rows = _replay_rows()
    small = _short_rows(active)
    replay_targets = torch.tensor([row["target"] for row in rows])
    short_targets = torch.tensor([row["target"] for row in small])
    teacher = torch.full((96, 25), -2.0)
    teacher[torch.arange(96), replay_targets] = 2.0
    return {
        "images": torch.rand(128, 1, 64, 256, requires_grad=True),
        "targets": replay_targets,
        "sizes": torch.linspace(-0.3, 0.3, 96),
        "teacher": teacher,
        "rows": rows,
        "short_targets": short_targets,
        "short_sizes": torch.linspace(-0.2, 0.2, 32),
        "short_rows": small,
    }


def _group_gradients(model):
    result = {}
    for group in ("trunk", "style", "family_head", "size_head"):
        values = [parameter.grad for name, parameter in model.named_parameters()
                  if name.startswith(group + ".")]
        assert values and all(value is None or torch.isfinite(value).all() for value in values)
        result[group] = sum(float(value.abs().sum()) for value in values if value is not None)
    return result


def test_inactive_one_two_glyph_and_unknown_short_rows_have_exactly_zero_gradient():
    from wide_region_network import WideRegionFontClassifier

    torch.set_num_threads(4)
    torch.manual_seed(31)
    model = WideRegionFontClassifier(25).train()
    batch = _batch(active=False)
    total, parts = trainer.objective(model, batch)

    assert parts["short_eligible"].item() == 0
    assert parts["short_weighted"].item() == 0
    parts["short_weighted"].backward()
    assert batch["images"].grad is not None
    assert torch.count_nonzero(batch["images"].grad) == 0
    assert all(value == 0 for value in _group_gradients(model).values())
    assert torch.isfinite(total)


def test_active_three_four_glyph_short_loss_updates_every_wide_parameter_group_only_from_known_rows():
    from wide_region_network import WideRegionFontClassifier

    torch.set_num_threads(4)
    torch.manual_seed(32)
    model = WideRegionFontClassifier(25).train()
    batch = _batch(active=True)
    _, parts = trainer.objective(model, batch)

    assert parts["short_eligible"].item() == 24
    parts["short_weighted"].backward()
    gradients = _group_gradients(model)
    assert all(value > 0 for value in gradients.values())
    assert batch["images"].grad is not None
    assert torch.count_nonzero(batch["images"].grad[:96]) == 0
    assert torch.count_nonzero(batch["images"].grad[96:120]) > 0
    assert torch.count_nonzero(batch["images"].grad[120:]) == 0


def test_replay_component_is_exact_r22_full_loss_with_supplied_restored_teacher():
    from wide_region_network import WideRegionFontClassifier

    torch.set_num_threads(4)
    torch.manual_seed(33)
    model = WideRegionFontClassifier(25).train()
    batch = _batch(active=True)
    # Make the supplied teacher visibly distinct; objective must pass these bytes
    # through rather than selecting a teacher from prediction or short metadata.
    batch["teacher"] = torch.linspace(-3.0, 3.0, 96 * 25).reshape(96, 25)
    _, parts = trainer.objective(model, batch)

    font, size, features = r22.forward_with_features(model, batch["images"])
    expected = r22.full_losses(
        font[:96], size[:96], features[:96], model.family_head.weight,
        batch["targets"], batch["sizes"], batch["teacher"], batch["rows"], FAMILIES,
    )
    torch.testing.assert_close(parts["replay"], expected[0])
    torch.testing.assert_close(parts["replay_ce"], expected[1])
    torch.testing.assert_close(parts["teacher_kl"], expected[3])


def _metrics():
    return {"known_correct_coverage": 0.65, "named_precision": 0.95,
            "unknown_not_named_rate": 0.85, "passed": False,
            "per_domain": {"ios": {"known_correct_coverage": 0.75},
                           "android": {"known_correct_coverage": 0.55}}}


def _synthetic_replay(step):
    unknown_sources = [*short.SHORT_UNKNOWN_SOURCES,
                       "WenQuanYi Zen Hei", "Zhuque Fangsong"]
    rows = []
    for index in range(96):
        if index < 16:
            target = 24
            source = unknown_sources[(step * 16 + index) % len(unknown_sources)]
        else:
            target = (index - 16) % 24
            source = FAMILIES[target]
        rows.append({"family": FAMILIES[target], "target": target,
                     "source_font_family": source})
    return rows


def _synthetic_short(step):
    glyph_count = step + 1
    rows = [{"family": family, "target": target,
             "source_font_family": family, "glyph_count": glyph_count}
            for target, family in enumerate(FAMILIES[:-1])]
    rows += [{"family": "__unknown__", "target": 24,
              "source_font_family": short.SHORT_UNKNOWN_SOURCES[index],
              "glyph_count": glyph_count} for index in range(8)]
    return rows


class _ReportStub:
    def report(self):
        return {"synthetic": True}


def test_four_step_protocol_counts_long_rows_and_cal_pass_never_implies_release(
        monkeypatch, tmp_path):
    torch.set_num_threads(2)
    base = tmp_path / "r22"
    base.mkdir()
    np.savez(base / "CALIBRATION_OUTPUTS.npz",
             logits=np.zeros((2, 25), np.float32), log_em_ratio=np.zeros(2, np.float32))
    retention = tmp_path / "RETENTION_PLAN.json"
    retention.write_text("{}")
    preflight = tmp_path / "PREFLIGHT.json"
    preflight.write_text(json.dumps({"passed": True, "bindings": {}, "test_read": False}))
    args = SimpleNamespace(output=tmp_path / "run", plan=tmp_path / "PLAN.json",
                           teacher=tmp_path / "teacher", short=tmp_path / "short", device="cpu")
    monkeypatch.setattr(trainer, "BASE_RUN", base)
    monkeypatch.setattr(trainer, "RETENTION_PLAN", retention)
    monkeypatch.setattr(trainer, "STEPS", 4)
    monkeypatch.setattr(trainer, "files_bound", lambda args, datasets: {})

    model = torch.nn.Linear(1, 1, bias=False)
    baseline_metrics = _metrics()
    baseline_selection = {"selected": {"metrics": deepcopy(baseline_metrics)}}
    stub = _ReportStub()
    monkeypatch.setattr(trainer, "context", lambda args:
        (model, {}, {}, {}, stub, stub, baseline_selection, {"synthetic": True}))
    monkeypatch.setattr(trainer, "parameter_groups", lambda state:
        {"all": float(next(iter(state.values())).sum())})

    batches = [{"rows": _synthetic_replay(step), "short_rows": _synthetic_short(step)}
               for step in range(4)]
    batch_iter = iter(batches)
    monkeypatch.setattr(trainer, "inputs", lambda *args: next(batch_iter))

    def objective(model, batch):
        eligible = sum(row["target"] != 24 and row["glyph_count"] in (3, 4)
                       for row in batch["short_rows"])
        value = (model.weight ** 2).sum()
        parts = {name: value for name in
                 ("replay", "replay_ce", "teacher_kl", "short_weighted", "short_font", "short_size")}
        parts.update(short_eligible=value.new_tensor(eligible),
                     original_unknown_teacher_eligible=value.new_tensor(7),
                     supplement_unknown_teacher_eligible=value.new_tensor(5))
        return value, parts

    monkeypatch.setattr(trainer, "objective", objective)
    cal = {"rows": [{}, {}], "manifest_sha256": "manifest", "partition_sha256": "partition"}
    monkeypatch.setattr(trainer, "load_split", lambda *args: cal)
    record = {"metrics": deepcopy(baseline_metrics),
              "retention_checks": [{"passed": True}] * 46, "promotion_allowed": True}
    monkeypatch.setattr(trainer, "evaluate_outputs", lambda *args:
        (deepcopy(record), [], [{}, {}]))
    short_results = iter((
        {"known_views": 100, "correct_named": 50, "known_correct_coverage": 0.5,
         "unknown_views": 50, "unknown_falsely_named": 10},
        {"known_views": 100, "correct_named": 52, "known_correct_coverage": 0.52,
         "unknown_views": 50, "unknown_falsely_named": 10},
    ))
    monkeypatch.setattr(trainer, "short_population", lambda *args: next(short_results))
    monkeypatch.setattr(trainer, "infer", lambda *args:
        (np.zeros((2, 25), np.float32), np.zeros(2, np.float32)))

    plan = {**trainer.design(args, {}),
            "preflight": {"path": str(preflight), "sha256": trainer.sha(preflight)}}
    args.plan.write_text(json.dumps(plan))
    trainer.train(args)

    counts = json.loads((args.output / "TRAINING_COUNTS.json").read_text())
    selection = json.loads((args.output / "SELECTION.json").read_text())
    report = json.loads((args.output / "report.json").read_text())
    assert counts["replay_rows"] == 4 * 96 and counts["short_rows"] == 4 * 32
    assert counts["short_supervised_rows"] == 48
    assert counts["short_supervised_by_glyph_count"] == {"3": 24, "4": 24}
    assert counts["teacher_eligible_rows"] == {
        "original_unknown_teacher_eligible": 28,
        "supplement_unknown_teacher_eligible": 20,
    }
    assert len(counts["replay_unknown_by_source"]) == 11
    assert max(counts["replay_unknown_by_source"].values()) - min(
        counts["replay_unknown_by_source"].values()) <= 1
    assert selection["calibration_promotion_allowed"] is True
    assert selection["promotion_allowed"] is False
    assert selection["development_evaluated"] is False
    assert report["calibration_promotion_allowed"] is True
    assert report["promotion_allowed"] is False and report["exported"] is False
    assert report["status"] == "CAL_PASSED_AWAITING_EXPORT_AND_DEVELOPMENT"


def test_v3_plan_keeps_fixed_schedule_sampling_and_all_release_gates():
    args = SimpleNamespace(teacher="/teacher", short="/short")
    plan = trainer.design(args, {"bound": "sha"})
    assert plan["steps"] == plan["evaluation_step"] == 2400
    assert plan["checkpoint_selection"] == "fixed final step 2400; no intermediate CAL search"
    assert plan["objective"]["replay_batch"] == 96
    assert plan["objective"]["short_batch"] == 32
    assert plan["objective"]["short_loss"] == trainer.LONG_SHORT_CONTRACT
    assert plan["objective"]["inactive_short_rows"].startswith("1/2 glyph and all short unknown")
    assert plan["objective"]["original_full_correct_teacher_kl_preserved"] is True
    assert plan["short_sampling"] == short.SHORT_SAMPLING
    assert plan["additional_acceptance"] == short.EXTRA_ACCEPTANCE
    assert plan["additional_acceptance"]["original_46_checks_required"] is True
    assert plan["additional_acceptance"]["original_18_development_checks_required"] is True
    assert plan["additional_acceptance"]["r22_18_development_checks_also_required"] is True
    assert plan["test_read"] is False and plan["development_holdout_read"] is False
