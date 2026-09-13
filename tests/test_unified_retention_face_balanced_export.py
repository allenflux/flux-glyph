"""Portable exporter checks for face-balanced merged-teacher provenance."""
from copy import deepcopy

import numpy as np
import pytest

from training import export_unified_retention_face_balanced as module
from training import train_unified_retention_face_balanced as trainer
from training.prepare_unified_regions import FAMILIES
from training.retention_face_balanced_sampler import FaceBalancedSampler
from test_retention_face_balanced_sampler import FAMILIES as PORTABLE_FAMILIES, fixtures


def merged_rows():
    faces = [('LXGW WenKai', 10, 'LXGWWenKai-Light'),
        ('LXGW WenKai', 10, 'LXGWWenKai-Medium'),
        ('LXGW WenKai', 10, 'LXGWWenKai-Regular'),
        ('WenQuanYi Micro Hei', 11, 'WenQuanYiMicroHei')]
    rows = []; offset = 0; position = 0
    while offset < 4820:
        family, target, face = faces[position % len(faces)]
        count = min(8, 4820-offset)
        rows.append({'family': family, 'target': target, 'font_face': face,
            'split': 'train', 'original_split': 'train', 'native_font_verified': True,
            'tile_start': offset, 'tile_count': count})
        offset += count; position += 1
    return rows


def replay_fixture():
    rows = merged_rows(); logits = np.full((4820, 25), -2., dtype=np.float32)
    histogram = []
    desired = [2000, 2000, 2000, 6000]
    face_rows = []
    for face_index, count in enumerate(desired):
        row_index = next(index for index,row in enumerate(rows)
                         if row['font_face'] == rows[face_index]['font_face'])
        tile_index = rows[row_index]['tile_start']
        logits[tile_index, rows[row_index]['target']] = 2.
        histogram.append({'tile_index': tile_index, 'rows': count})
        face_rows.append({'family': rows[row_index]['family'],
            'font_face': rows[row_index]['font_face'], 'rows': count, 'eligible_rows': count})
    counts = {'known_teacher_tile_index_counts': histogram,
        'known_supplement_face_rows': face_rows}
    return counts, rows, logits


def test_sampled_index_replay_recomputes_face_totals_and_same_tile_argmax():
    counts, rows, logits = replay_fixture()
    result = module.replay_known_teacher_indices(counts, rows, logits)
    assert result['sampled_rows'] == 12000
    assert result['unique_teacher_tiles'] == 4 and result['teacher_cache_tiles'] == 4820
    assert result['teacher'] == 'b08' and result['unknown_teacher_rows'] == 0
    assert result['face_rows'] == sorted(counts['known_supplement_face_rows'],
                                         key=lambda row: (row['family'], row['font_face']))


@pytest.mark.parametrize('fault', ['index', 'duplicate', 'count', 'label', 'logits'])
def test_sampled_index_replay_rejects_index_label_logit_and_count_drift(fault):
    counts, rows, logits = replay_fixture()
    if fault == 'index': counts['known_teacher_tile_index_counts'][0]['tile_index'] = 4820
    elif fault == 'duplicate':
        counts['known_teacher_tile_index_counts'][1]['tile_index'] = counts['known_teacher_tile_index_counts'][0]['tile_index']
    elif fault == 'count': counts['known_teacher_tile_index_counts'][0]['rows'] -= 1
    elif fault == 'label': rows[0]['target'] = 11
    else:
        index = counts['known_teacher_tile_index_counts'][0]['tile_index']
        logits[index] = -2.; logits[index, 0] = 2.
    with pytest.raises(ValueError):
        module.replay_known_teacher_indices(counts, rows, logits)


def selection_fixture():
    identity = {'path': '/portable/cache/CACHE_MANIFEST.json', 'sha256': '1'*64}
    historical_mix = {'original': {'tiles': 10, 'named_teacher_tiles': 7,
        'unknown_teacher_tiles': 3, 'logits_sha256': '0'*64, 'labels_sha256': '1'*64},
        'supplement': {'tiles': 5, 'named_teacher_tiles': 0,
            'unknown_teacher_tiles': 5, 'logits_sha256': '0'*64, 'labels_sha256': '1'*64},
        'known': {'tiles': 4, 'named_teacher_tiles': 4,
            'unknown_teacher_tiles': 0, 'logits_sha256': '0'*64, 'labels_sha256': '1'*64}}
    actual_mix = {'original': {'tiles': 10, 'named_teacher_tiles': 7,
        'unknown_teacher_tiles': 3, 'logits_sha256': '0'*64, 'labels_sha256': '1'*64},
        'supplement': {'tiles': 5, 'named_teacher_tiles': 0,
            'unknown_teacher_tiles': 5, 'logits_sha256': '0'*64, 'labels_sha256': '1'*64},
        'known': {'tiles': 4820,
        'named_teacher_tiles': 4820, 'unknown_teacher_tiles': 0,
        'logits_sha256': '2'*64, 'labels_sha256': '3'*64}}
    groups = dict.fromkeys(module.GROUPS, '4'*64)
    return {
        'families': list(FAMILIES), 'data_manifest_sha256': '9'*64,
        'historical_known_data': {'path': '/portable/historical/MANIFEST.json', 'sha256': '5'*64},
        'historical_known_cache': {'path': '/portable/historical-cache/CACHE_MANIFEST.json', 'sha256': '6'*64},
        'historical_known_data_used_for_sampling': False,
        'historical_known_cache_used_for_optimizer': False,
        'known_cache_kind': 'merged_known_b08_logits',
        'named_teacher_cache_used_partitions': ['original', 'supplement'],
        'unknown_teacher_used_partitions': ['original', 'supplement'],
        'objective_variant': trainer.OBJECTIVE_VARIANT, 'objective': trainer.OBJECTIVE,
        'unknown_floor_supervision': trainer.UNKNOWN_FLOOR_SUPERVISION,
        'sampling': trainer.SAMPLING, 'unknown_source_order': ['portable-source'],
        'design_changes': trainer.DESIGN_CHANGES, 'single_change_causal_attribution': False,
        'supplement_data': {'path': '/portable/supplement/MANIFEST.json', 'sha256': '7'*64},
        'supplement_cache': {'path': '/portable/supplement-cache/CACHE_MANIFEST.json', 'sha256': '8'*64},
        'known_data': {'path': '/portable/merged/MANIFEST.json', 'sha256': trainer.WEIGHT_PAIRS_MANIFEST_SHA},
        'known_cache': identity, 'merged_known_teacher_cache': identity,
        'student_initializer': {'state_sha256': '9'*64},
        'named_teacher_identity': {'state_sha256': '9'*64},
        'named_teacher_cache': {'path': '/portable/named/CACHE_MANIFEST.json', 'sha256': 'a'*64},
        'named_teacher_state_sha256': '9'*64,
        'unknown_teacher_identity': {'state_sha256': 'b'*64},
        'teacher_policy': trainer.TEACHER_POLICY, 'teacher_mix': actual_mix,
        'historical_teacher_mix': historical_mix,
        'widening': {'schema': 'portable'}, 'widening_parity': {'passed': True},
        'offline_teacher_count': 2, 'teacher_models_resident_during_optimizer': 0,
        'historical_core_caches_used_for_training': False,
        'training_data_counts': trainer.EXPECTED_TRAINING_DATA_COUNTS,
        'retention_plan': {'path': '/portable/RETENTION_PLAN.json', 'sha256': '8'*64},
        'objective_plan': {'path': '/portable/PLAN.json', 'sha256': 'c'*64},
        'base_checkpoint': {'path': '/portable/base.pth', 'sha256': 'd'*64},
        'base_selection': {'path': '/portable/SELECTION.json', 'sha256': 'e'*64},
        'base_state_sha256': 'f'*64, 'cache_manifest': {'path': '/portable/core-cache', 'sha256': '0'*64},
        'initial_state_sha256': '1'*64, 'state_before_sha256': '1'*64,
        'state_after_sha256': '2'*64,
        'initial_parameter_groups_sha256': groups,
        'selected_parameter_groups_sha256': dict.fromkeys(module.GROUPS, '5'*64),
        'final_parameter_groups_sha256': dict.fromkeys(module.GROUPS, '6'*64),
        'training_counts_sha256': '7'*64, 'optimizer_steps_executed': 6000,
        'source_optimizer_steps_executed': 3000, 'source_selected_step': 1500,
        'base_selected_step': 1000, 'core_optimizer_steps_executed': 1500,
        'training_device': 'mps', 'selected': {'step': 6000, 'metrics': {'passed': True}},
        'promotion_allowed': True,
        'checkpoint_selection': 'Fixed final step 6000; no intermediate CAL inference or checkpoint search',
        'calibration_reused_for_prior_development': True,
    }


def test_actual_metadata_and_parity_builders_preserve_compact_historical_and_merged_handoff():
    selection = selection_fixture()
    metadata = module.training_metadata(selection)
    report = module.build_parity_report(selection, 'a'*64, 'b'*64, 'c'*64, 'd'*64,
        {'exporter': 'e'*64}, {'selection': 'f'*64}, {'logits': 0., 'size': 0.}, 0.,
        46, 137, [{'batch_size': 1, 'passed': True}], {'metrics': {'passed': True}})
    for key in ('historical_known_data', 'historical_known_cache',
            'historical_known_data_used_for_sampling', 'historical_known_cache_used_for_optimizer',
            'known_data', 'known_cache', 'merged_known_teacher_cache', 'known_cache_kind',
            'historical_teacher_mix', 'teacher_mix', 'named_teacher_cache_used_partitions',
            'unknown_teacher_used_partitions'):
        assert metadata[key] == report[key] == selection[key]
    assert report['font_model_count'] == report['encoder_count'] == 1
    assert report['offline_teacher_count'] == 2
    assert 'bindings' not in metadata and 'bindings' not in report['known_cache']
    runtime = {'promotion_allowed': True, 'temperature': 1.,
        'gates': deepcopy(module.FIXED_RUNTIME['gates']), 'retention_checks': [],
        'retention_populations': {}, 'metrics': {'passed': True, 'named_precision': 1.,
            'known_correct_coverage': 1., 'unknown_not_named_rate': 1., 'checks': {}}}
    exported = module.metadata_for_export(selection,
        {'manifest': {'font_label_groups': {}}}, {'font_sources': {}},
        'c'*64, 'b'*64, 'a'*64, runtime)
    assert exported['validation']['kind'] == 'full_cnn_face_balanced_wide_transfer_calibration_only_at_export'
    assert exported['training']['merged_known_teacher_cache'] == selection['known_cache']


def test_teacher_mix_keeps_actual_merged_known_and_both_historical_teachers():
    mix = selection_fixture()['teacher_mix']
    module.validate_teacher_mix(mix, ('original', 'supplement', 'known'))
    assert mix['known']['named_teacher_tiles'] == 4820
    assert mix['known']['unknown_teacher_tiles'] == 0


def test_compact_handoff_rejects_embedded_cache_binding_closure():
    selection = selection_fixture()
    selection['known_cache'] = {**selection['known_cache'], 'bindings': {'source': 'a'*64}}
    with pytest.raises(ValueError, match='Compact TRAIN data/cache identity'):
        module.training_metadata(selection)


def test_family_quota_derivation_matches_real_6000_step_sampler_report():
    old, pools, new, known = fixtures()
    remaining = [family for family in FAMILIES if family not in PORTABLE_FAMILIES]
    placeholders = [family for family in PORTABLE_FAMILIES if family.startswith('Joint ')]
    mapping = dict(zip(placeholders, remaining))
    mapping.update({family: family for family in PORTABLE_FAMILIES if family not in mapping})
    for rows in (old, new, known):
        for row in rows:
            prior = row['family']; row['family'] = mapping[prior]
            row['target'] = FAMILIES.index(row['family'])
            if row['source_font_family'] == prior:row['source_font_family'] = row['family']
    replay = FaceBalancedSampler(old, FAMILIES, trainer.SEED, pools, new, known)
    for _ in range(6000):replay.batch()
    actual = replay.report()['family_rows']
    expected = module.expected_family_counts(FAMILIES, trainer.SAMPLING, 6000)
    assert actual == expected and sum(expected.values()) == 576000
    assert expected['PingFang'] == 114000
    assert expected['SF Pro'] == expected['Helvetica'] == 42000
    assert expected['Alipay Number'] == 18000
