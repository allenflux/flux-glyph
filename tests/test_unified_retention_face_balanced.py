"""Pretraining checks for merged known data and face-balanced dual-teacher transfer."""
import hashlib
import importlib.machinery
import json
from copy import deepcopy
import sys
import types

import numpy as np
import pytest

try:
    import onnxruntime  # noqa: F401
except ModuleNotFoundError:
    fake_ort = types.ModuleType('onnxruntime')
    fake_ort.__spec__ = importlib.machinery.ModuleSpec('onnxruntime', loader=None)
    sys.modules['onnxruntime'] = fake_ort

from training import train_unified_retention_face_balanced as m
from training.prepare_unified_regions import FAMILIES
from training.retention_confidence_floor_loss import unknown_floor_loss
from test_retention_face_balanced_sampler import sampler


def production_loss_rows():
    rows = deepcopy(sampler().batch())
    for row in rows:
        if row['family'] in FAMILIES:
            row['target'] = FAMILIES.index(row['family'])
        else:
            row['family'] = FAMILIES[row['target']]
    return rows

def loss_tensors():
    torch = pytest.importorskip('torch')
    rows = production_loss_rows()
    targets = torch.tensor([row['target'] for row in rows])
    logits = torch.linspace(-1., 1., 96*25, dtype=torch.float64).reshape(96, 25).requires_grad_()
    ratios = torch.linspace(-.2, .4, 96, dtype=torch.float64).requires_grad_()
    sizes = torch.full((96,), .2, dtype=torch.float64)
    teacher = torch.full((96, 25), -2., dtype=torch.float64)
    teacher[torch.arange(96), targets] = 2.
    return rows, targets, logits, ratios, sizes, teacher


def test_full_loss_is_unchanged_known_ce_unknown_floor_size_and_eligible_kl():
    torch = pytest.importorskip('torch')
    rows, targets, logits, ratios, sizes, teacher = loss_tensors()
    total, ce, size, preservation, mask = m.full_losses(
        logits, ratios, targets, sizes, teacher, rows, FAMILIES)
    unknown = targets == 24; known = ~unknown
    weights = torch.tensor(m.core.native_core_weights(rows, targets.tolist(), FAMILIES),
                           dtype=logits.dtype)
    expected_ce = ((torch.nn.functional.cross_entropy(logits[known], targets[known],
        label_smoothing=.03, reduction='none') * weights[known]).sum()
        + (unknown_floor_loss(logits[unknown], 24) * weights[unknown]).sum()) / 96
    expected_size = torch.nn.functional.smooth_l1_loss(ratios, sizes)
    expected_kl = torch.nn.functional.kl_div(logits.log_softmax(1), teacher.softmax(1),
        reduction='none').sum(1).mean()
    assert mask.all()
    torch.testing.assert_close(total, expected_ce + .2*expected_size + 2*expected_kl)
    torch.testing.assert_close(ce, expected_ce)
    torch.testing.assert_close(size, expected_size)
    torch.testing.assert_close(preservation, expected_kl)
    total.backward()
    assert teacher.grad is None and bool(torch.isfinite(logits.grad).all())


def merged_fixture():
    rows = []
    offset = 0
    for family, target, face, count in (
            ('LXGW WenKai', 10, 'LXGWWenKai-Regular', 2),
            ('LXGW WenKai', 10, 'LXGWWenKai-Light', 1),
            ('LXGW WenKai', 10, 'LXGWWenKai-Medium', 2),
            ('WenQuanYi Micro Hei', 11, 'WenQuanYiMicroHei', 1)):
        rows.append({'family': family, 'target': target, 'font_face': face,
            'split': 'train', 'original_split': 'train', 'native_font_verified': True,
            'tile_start': offset, 'tile_count': count})
        offset += count
    return {'families': list(FAMILIES), 'rows': rows,
        'tiles': np.zeros((offset, 1, 64, 256), dtype=np.float32)}


def test_merged_known_substitution_uses_only_same_tile_b08_logits_and_true_labels():
    data = merged_fixture()
    logits = np.arange(len(data['tiles'])*25, dtype=np.float32).reshape(-1, 25)
    cache, proof = m.merged_known_teacher_cache(data, {'base_logits': logits})
    labels = np.array([10, 10, 10, 10, 10, 11], dtype=np.int64)
    assert cache['base_logits'] is logits
    assert proof == {'tiles': 6, 'named_teacher_tiles': 6, 'unknown_teacher_tiles': 0,
        'logits_sha256': hashlib.sha256(logits.tobytes()).hexdigest(),
        'labels_sha256': hashlib.sha256(labels.tobytes()).hexdigest()}
    with pytest.raises(ValueError):
        m.merged_known_teacher_cache(data, {'base_logits': logits, 'r21_logits': logits})


def tiny_data(prefix, identities, tile_count):
    return {'rows': [{'source_id': f'{prefix}-{source}', 'region_id': region}
                     for source, region in identities],
        'tiles': np.zeros((tile_count, 1, 64, 256), dtype=np.float32)}


def test_actual_merged_known_identity_is_separate_from_sources_and_supersets_history():
    training = tiny_data('original', [('a', 'r')], 2)
    supplement = tiny_data('unknown', [('b', 'r')], 3)
    historical = tiny_data('paired', [('c', 'r')], 4)
    known = tiny_data('paired', [('c', 'r'), ('d', 'r')], 7)
    counts = m.training_data_identity(training, supplement, known, historical)
    assert counts == {'original_views': 1, 'supplement_views': 1,
        'known_supplement_views': 2, 'combined_views': 4,
        'original_native_regions': 1, 'supplement_native_regions': 1,
        'known_supplement_native_regions': 2, 'combined_native_regions': 4,
        'original_tiles': 2, 'supplement_tiles': 3, 'known_supplement_tiles': 7,
        'combined_tiles': 12, 'calibration_unchanged': True,
        'derived_views_are_correlated': True}
    colliding = tiny_data('original', [('a', 'r')], 7)
    with pytest.raises(ValueError):
        m.training_data_identity(training, supplement, colliding, historical)


def test_6000_sampler_report_has_exact_face_and_total_row_contract():
    replay = sampler(m.SEED)
    for _ in range(m.STEPS):
        replay.batch()
    report = replay.report()
    face_names = {'LXGW WenKai-Light': 'LXGWWenKai-Light',
        'LXGW WenKai-Medium': 'LXGWWenKai-Medium',
        'LXGW WenKai-Regular': 'LXGWWenKai-Regular',
        'WenQuanYi Micro Hei-Regular': 'WenQuanYiMicroHei'}
    for key in ('known_supplement_face_rows', 'face_rows'):
        for row in report[key]:
            row['font_face'] = face_names.get(row['font_face'], row['font_face'])
    report['known_supplement_tile_count'] = 4820
    report = m.validate_sampling_report(report, m.STEPS)
    assert report['family_rows']['LXGW WenKai'] == 12000
    assert report['family_rows']['WenQuanYi Micro Hei'] == 12000
    assert sum(report['family_rows'].values()) == 576000


def test_short_counter_reconciles_true_target_masks_and_teacher_sources():
    replay = sampler(17)
    families = replay.base.families
    counts = m.FullSupplementCounts(families, replay.unknown_source_order)
    for _ in range(11):
        rows = replay.batch()
        counts.update(rows, [True]*96, [row['target'] for row in rows])
    report = m.validate_training_counts(counts.report(), families, 11)
    assert report['selected_teacher_true_target_rows'] == {'named_b08': 880, 'unknown_r21': 176}
    assert report['eligible_rows'] == 1056


def test_counter_rejects_sampled_teacher_index_or_partition_drift():
    replay = sampler(17);rows = replay.batch();families = replay.base.families
    indices = [('known' if row.get(m.KNOWN_SUPPLEMENT_MARKER) else
        'supplement' if row.get(m.SUPPLEMENT_MARKER) else 'original', row['tile_start'])
        for row in rows]
    known_position = next(index for index,row in enumerate(rows)
                          if row.get(m.KNOWN_SUPPLEMENT_MARKER))
    indices[known_position] = ('original', indices[known_position][1])
    counts = m.FullSupplementCounts(families, replay.unknown_source_order)
    with pytest.raises(ValueError, match='drifted'):
        counts.update(rows, [True]*96, [row['target'] for row in rows], indices)


def objective_plan(cache_sha):
    return {'schema': 'flux-glyph-wide-face-balanced-plan-v1',
        'architecture': m.ARCHITECTURE, 'design_changes': m.DESIGN_CHANGES,
        'source_initializer_sha256': m.STUDENT_CHECKPOINT_SHA,
        'teacher_cache_sha256': m.TEACHER_CACHE_SHA, 'objective': m.OBJECTIVE,
        'steps': m.STEPS, 'eval_every': m.EVAL_EVERY, 'seed': m.SEED,
        'weight_pairs_manifest_sha256': m.WEIGHT_PAIRS_MANIFEST_SHA,
        'weight_teacher_cache_manifest_sha256': cache_sha,
        'runtime': m.FIXED_RUNTIME, 'acceptance_plan_sha256': 'a'*64,
        'r21_inference_report_sha256': m.R21_REPORT_SHA, 'teacher_policy': m.TEACHER_POLICY,
        'checkpoint_selection': 'fixed final step 6000, no intermediate candidate selection',
        'calibration_used_in_prior_development': True, 'blind_test': False,
        'new_training_started': False, 'single_cause_claim': False}


def test_objective_plan_requires_concrete_merged_cache_pin(tmp_path):
    path = tmp_path/'PLAN.json'; cache_sha = m.WEIGHT_TEACHER_CACHE_MANIFEST_SHA
    path.write_text(json.dumps(objective_plan(cache_sha)))
    assert m.read_objective_plan(path, 'a'*64, cache_sha)['weight_teacher_cache_manifest_sha256'] == cache_sha
    with pytest.raises(ValueError):
        m.read_objective_plan(path, 'a'*64, 'b'*64)


def test_fixed_protocol_defaults_keep_legacy_witnesses_and_add_actual_merged_inputs():
    args = m.parser().parse_args([])
    assert (m.STEPS, m.EVAL_EVERY, m.SEED, m.LEARNING_RATE, m.MINIMUM_LEARNING_RATE) == (
        6000, 6000, 2026091414, 2e-5, 2e-6)
    assert m.ARCHITECTURE == 'region-cnn64x256-unified-wide-v1'
    assert args.output.name == 'run-wide-face-balanced-v1'
    assert args.objective_plan.parent.name == 'face-balanced-plan-v1'
    assert args.known == m.ROOT/'artifacts/unified-font-v2/paired-known-capture-v1/data'
    assert args.known_cache == m.ROOT/'artifacts/unified-font-v2/paired-known-capture-v1/cache'
    assert args.weight_pairs.name == 'merged-data'
    assert args.weight_teacher_cache.name == 'teacher-cache'
    assert m.EXPECTED_TRAINING_DATA_COUNTS['combined_views'] == 74926
    assert m.EXPECTED_TRAINING_DATA_COUNTS['combined_native_regions'] == 21003
    assert m.EXPECTED_TRAINING_DATA_COUNTS['combined_tiles'] == 138830
    assert m.WEIGHT_TEACHER_CACHE_MANIFEST_SHA == '17732d05f0102c6e384a4d234afdfc363adb94987ec300085485882e7593ef41'


def test_actual_wide_model_full_loss_updates_every_parameter_group():
    torch = pytest.importorskip('torch')
    from training.wide_region_network import WideRegionFontClassifier
    torch.manual_seed(1414);torch.set_num_threads(4)
    model = WideRegionFontClassifier(25);before = m.parameter_groups(model.state_dict())
    rows = production_loss_rows();targets = torch.tensor([row['target'] for row in rows])
    teacher = torch.full((96, 25), -2.);teacher[torch.arange(96), targets] = 2.
    logits, ratios = model(torch.rand(96, 1, 64, 256))
    loss, *_ = m.full_losses(logits, ratios, targets, torch.full((96,), .2), teacher, rows, FAMILIES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=m.LEARNING_RATE)
    optimizer.zero_grad(set_to_none=True);loss.backward()
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters())
    optimizer.step();after = m.parameter_groups(model.state_dict())
    assert all(after[group] != before[group] for group in ('trunk','style','family_head','size_head'))
