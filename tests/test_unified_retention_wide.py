"""Pretraining integration checks for the fixed wide dual-teacher run."""
import hashlib
import importlib.machinery
import sys
import types

import numpy as np
import pytest

try:
    import onnxruntime  # noqa: F401
except ModuleNotFoundError:
    # These unit tests never construct a runtime session.
    fake_ort = types.ModuleType('onnxruntime')
    fake_ort.__spec__ = importlib.machinery.ModuleSpec('onnxruntime', loader=None)
    sys.modules['onnxruntime'] = fake_ort

from training import train_unified_retention_wide as m
from training import train_unified_retention_confidence_floor as narrow
from training.prepare_unified_regions import FAMILIES
from training.retention_confidence_floor_loss import unknown_floor_loss
from test_unified_retention_paired_known import sampler


torch = pytest.importorskip('torch')


def tensors():
    rows = sampler().batch(); targets = torch.tensor([row['target'] for row in rows])
    logits = torch.linspace(-1., 1., 96*25, dtype=torch.float64).reshape(96, 25).requires_grad_()
    ratios = torch.linspace(-.2, .4, 96, dtype=torch.float64).requires_grad_()
    sizes = torch.full((96,), .2, dtype=torch.float64)
    teacher = torch.full((96, 25), -2., dtype=torch.float64)
    teacher[torch.arange(96), targets] = 2.
    return rows, targets, logits, ratios, sizes, teacher


def test_full_loss_is_fixed_ce_unknown_size_and_selected_teacher_kl():
    rows, targets, logits, ratios, sizes, teacher = tensors()
    total, ce, size, kl, mask = m.full_losses(logits, ratios, targets, sizes, teacher, rows, FAMILIES)
    unknown = targets == 24; known = ~unknown
    weights = torch.tensor(m.core.native_core_weights(rows, targets.tolist(), FAMILIES), dtype=logits.dtype)
    known_ce = torch.nn.functional.cross_entropy(logits[known], targets[known],
        label_smoothing=.03, reduction='none')
    unknown_ce = unknown_floor_loss(logits[unknown], 24)
    expected_ce = ((known_ce*weights[known]).sum() + (unknown_ce*weights[unknown]).sum())/96
    expected_size = torch.nn.functional.smooth_l1_loss(ratios, sizes)
    expected_kl = torch.nn.functional.kl_div(logits.log_softmax(1), teacher.softmax(1),
        reduction='none').sum(1).mean()
    assert mask.all()
    torch.testing.assert_close(ce, expected_ce)
    torch.testing.assert_close(size, expected_size)
    torch.testing.assert_close(kl, expected_kl)
    torch.testing.assert_close(total, expected_ce + .2*expected_size + 2*expected_kl)
    total.backward()
    assert teacher.grad is None and bool(torch.isfinite(logits.grad).all()) and bool(torch.isfinite(ratios.grad).all())


def dataset(name):
    rows = [
        {'split': 'train', 'original_split': 'train', 'native_font_verified': True,
            'family': 'LXGW WenKai', 'target': 10, 'tile_start': 0, 'tile_count': 2},
        {'split': 'train', 'original_split': 'train', 'native_font_verified': True,
            'family': '__unknown__', 'target': 24, 'tile_start': 2, 'tile_count': 3},
        {'split': 'train', 'original_split': 'train', 'native_font_verified': True,
            'family': 'Roboto', 'target': 8, 'tile_start': 5, 'tile_count': 1},
    ]
    return {'rows': rows, 'tiles': np.zeros((6, 1, 64, 256), dtype=np.float32),
        'families': FAMILIES, 'partition': {'split': 'train'}, 'name': name}


def test_mixed_cache_uses_b08_for_known_and_r21_for_unknown_exact_tiles():
    datasets = {name: dataset(name) for name in ('original', 'supplement', 'known')}
    named_cache = {}; unknown_cache = {}
    for offset, name in enumerate(datasets):
        named = np.arange(6*25, dtype=np.float32).reshape(6, 25) + offset*1000
        unknown = -named - 5000
        named_cache[name] = {'base_logits': named}; unknown_cache[name] = unknown
    mixed, proof = m.mixed_teacher_cache(datasets, named_cache, unknown_cache)
    for name in datasets:
        expected = named_cache[name]['base_logits'].copy()
        expected[2:5] = unknown_cache[name][2:5]
        np.testing.assert_array_equal(mixed[name]['base_logits'], expected)
        labels = np.array([10, 10, 24, 24, 24, 8], dtype=np.int64)
        assert proof[name] == {'tiles': 6, 'named_teacher_tiles': 3, 'unknown_teacher_tiles': 3,
            'logits_sha256': hashlib.sha256(expected.tobytes()).hexdigest(),
            'labels_sha256': hashlib.sha256(labels.tobytes()).hexdigest()}


def test_mixed_cache_rejects_metadata_that_is_not_exact_tile_order():
    datasets = {name: dataset(name) for name in ('original', 'supplement', 'known')}
    datasets['original']['rows'] = list(reversed(datasets['original']['rows']))
    named = {name: {'base_logits': np.zeros((6, 25), dtype=np.float32)} for name in datasets}
    unknown = {name: np.ones((6, 25), dtype=np.float32) for name in datasets}
    with pytest.raises(ValueError):
        m.mixed_teacher_cache(datasets, named, unknown)


def test_6000_step_source_counts_are_exact_metadata_math():
    replay = sampler(); order = replay.unknown_source_order
    counts = m.expected_unknown_counts(order, m.STEPS*16)
    quotient, remainder = divmod(96000, 11)
    assert m.STEPS == 6000 and sum(counts.values()) == 96000
    assert list(counts.values()) == [quotient + int(i < remainder) for i in range(11)]
    extra = sum(counts[source] for source in m.NEW_SOURCES)
    assert extra == sum(m.expected_unknown_counts(order, 96000)[source] for source in m.NEW_SOURCES)
    assert m.STEPS*96 == 576000 and m.STEPS*2 == 12000
    assert m.STEPS*94 - extra == 564000 - extra


def test_short_counter_reconciles_dual_teacher_eligibility_and_sources():
    replay = sampler(); counts = m.FullSupplementCounts(FAMILIES, replay.unknown_source_order)
    for _ in range(11):
        rows = replay.batch(); predictions = [row['target'] for row in rows]
        counts.update(rows, [True]*96, predictions)
    report = counts.report()
    m.validate_training_counts(report, FAMILIES, 11)
    assert report['eligible_rows'] == 1056 and report['unknown_rows'] == 176
    assert report['known_supplement_rows'] == 22


def test_fixed_wide_protocol_has_no_stale_current_run_schedule_or_architecture():
    objective_plan = m.read_objective_plan(
        m.ROOT/'tests/fixtures/unified_retention_wide_objective_plan.json',
        'b82283b54b5e47f4639b0df62231544c0419f6299d10820d86dca96ef24094ff')
    assert (m.STEPS, m.EVAL_EVERY, m.SEED, m.LEARNING_RATE, m.MINIMUM_LEARNING_RATE) == (
        6000, 6000, 2026091414, 2e-5, 2e-6)
    assert m.FIXED_RUNTIME == {'temperature': 1.,
        'gates': {'min_score': .7, 'min_margin': .01, 'min_patch_agreement': 2/3},
        'max_size_relative_spread': .2}
    assert m.ARCHITECTURE == 'region-cnn64x256-unified-wide-v1'
    assert narrow.ARCHITECTURE == 'region-cnn64x256-unified-v1' != m.ARCHITECTURE
    assert m.OBJECTIVE['teacher_policy'] == m.TEACHER_POLICY
    assert m.OBJECTIVE['teacher_kl_weight'] == 2. and m.OBJECTIVE['teacher_temperature'] == 1.
    assert m.OBJECTIVE['teacher_distribution_kl'] is True
    assert objective_plan['checkpoint_selection'] == 'fixed final step 6000, no intermediate candidate selection'
    assert objective_plan['steps'] == objective_plan['eval_every'] == 6000
    args = m.parser().parse_args([])
    assert args.steps == 6000 and args.seed == 2026091414 and args.output.name == 'run-wide-transfer-v1'


def test_actual_wide_model_full_loss_updates_every_parameter_group():
    from training.wide_region_network import WideRegionFontClassifier
    torch.manual_seed(1414); torch.set_num_threads(4)
    model = WideRegionFontClassifier(25); before = m.parameter_groups(model.state_dict())
    rows = sampler().batch(); targets = torch.tensor([row['target'] for row in rows])
    teacher = torch.full((96, 25), -2.); teacher[torch.arange(96), targets] = 2.
    logits, ratios = model(torch.rand(96, 1, 64, 256))
    loss, *_ = m.full_losses(logits, ratios, targets, torch.full((96,), .2), teacher, rows, FAMILIES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    optimizer.zero_grad(set_to_none=True); loss.backward()
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters())
    optimizer.step(); after = m.parameter_groups(model.state_dict())
    assert all(after[group] != before[group] for group in ('trunk', 'style', 'family_head', 'size_head'))
