"""Same-initializer teacher provenance and truth-correct TRAIN retention accounting."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from training import export_unified_retention_teacher_preserve as module
import train_unified_retention_teacher_preserve as trainer
import cache_unified_student_teacher as teacher_cache
import test_unified_retention_paired_known_export as paired
from prepare_unified_regions import FAMILIES


def fixture_student(monkeypatch, tmp_path):
    current, source = paired.fixture_student(monkeypatch, tmp_path)
    monkeypatch.setattr(trainer, 'STUDENT_CHECKPOINT_SHA', current['student_initializer']['checkpoint']['sha256'])
    monkeypatch.setattr(trainer, 'STUDENT_SELECTION_SHA', current['student_initializer']['selection']['sha256'])
    current.update(teacher_identity=deepcopy(current['student_initializer']),
        teacher_cache={'path': 'teacher/CACHE_MANIFEST.json', 'sha256': '9'*64},
        historical_core_caches_used_for_training=False,
        cached_teacher_state_sha256=current['student_initializer']['state_sha256'])
    return current, source


def test_student_and_actual_teacher_are_same_whole_prior_model(monkeypatch, tmp_path):
    selection, source = fixture_student(monkeypatch, tmp_path)
    assert module.validate_student_initializer(selection) == source
    assert selection['cached_teacher_state_sha256'] == selection['initial_state_sha256'] != selection['base_state_sha256']


@pytest.mark.parametrize('fault', ['core_teacher', 'different_teacher', 'historical_logits', 'wrong_source_step', 'wrong_group'])
def test_student_teacher_identity_cannot_be_substituted(monkeypatch, tmp_path, fault):
    selection, _ = fixture_student(monkeypatch, tmp_path)
    if fault == 'core_teacher': selection['cached_teacher_state_sha256'] = selection['base_state_sha256']
    elif fault == 'different_teacher': selection['teacher_identity']['state_sha256'] = '0'*64
    elif fault == 'historical_logits': selection['historical_core_caches_used_for_training'] = True
    elif fault == 'wrong_source_step': selection['student_initializer']['selected_step'] = 3000
    else: selection['initial_parameter_groups_sha256']['size_head'] = '0'*64
    with pytest.raises(ValueError): module.validate_student_initializer(selection)


def test_metadata_preserves_new_teacher_and_historical_evidence_separately(monkeypatch, tmp_path):
    selection, _ = fixture_student(monkeypatch, tmp_path)
    result = module.training_metadata(selection)
    for key in ('teacher_identity', 'teacher_cache', 'historical_core_caches_used_for_training',
        'student_initializer', 'base_checkpoint', 'cached_teacher_state_sha256', 'base_state_sha256',
        'known_data', 'known_cache', 'training_counts_sha256', 'initial_parameter_groups_sha256'):
        assert result[key] == selection[key]
    assert result['teacher_identity'] is not selection['teacher_identity']
    assert result['teacher_in_deployed_model'] is False and result['model_count'] == result['encoder_count'] == 1


def test_historical_known_core_cache_does_not_claim_to_be_the_new_teacher(monkeypatch, tmp_path):
    selection, data, unknown = paired.fixture_known(monkeypatch, tmp_path)
    monkeypatch.setattr(trainer, 'KNOWN_MANIFEST_SHA', selection['known_data']['sha256'])
    monkeypatch.setattr(trainer, 'KNOWN_CACHE_SHA', selection['known_cache']['sha256'])
    monkeypatch.setattr(trainer, 'load_known_cache', paired.trainer.load_known_cache)
    selection['cached_teacher_state_sha256'] = '9'*64
    rows, part = module.validate_known_supplement(selection, [], unknown)
    assert rows == data['rows'] and part == data['partition']


def fixture_cache(monkeypatch, tmp_path):
    folder = tmp_path/'teacher-cache'; folder.mkdir()
    (folder/'CACHE_FREEZE.json').write_text('{}')
    teacher = {'state_sha256': 'a'*64}
    identity = {'teacher': teacher, 'families': FAMILIES, 'teacher_logits_recomputed': True,
        'historical_logits_reused': False, 'bindings': {}, 'partitions': {}, 'total_tiles': 6}
    manifest = {'identity': identity, 'partitions': {}, 'cache_freeze_sha256': module.sha(folder/'CACHE_FREEZE.json')}
    selection = {'teacher_identity': deepcopy(teacher), 'student_initializer': deepcopy(teacher),
        'cached_teacher_state_sha256': 'a'*64, 'initial_state_sha256': 'a'*64, 'base_state_sha256': 'b'*64,
        'historical_core_caches_used_for_training': False, 'families': FAMILIES,
        'training_data_counts': {'combined_tiles': 6}, 'bindings': {}}
    arrays = {}; roots = {}
    for source in teacher_cache.PARTITIONS:
        root = tmp_path/source; (root/'train').mkdir(parents=True); roots[source] = root
        module.dump(root/'MANIFEST.json', {'families': FAMILIES})
        (root/'train/rows.json').write_text('[]'); (root/'train/tiles.raw').write_bytes(source.encode())
        part = {'split': 'train', 'families': FAMILIES, 'views': 1, 'tiles': 2, 'shape': [2, 1, 64, 256],
            'metadata': {'path': 'rows.json', 'sha256': module.sha(root/'train/rows.json')},
            'array': {'path': 'tiles.raw', 'sha256': module.sha(root/'train/tiles.raw')}}
        module.dump(root/'train/MANIFEST.json', part)
        descriptor = {'data_manifest': teacher_cache.entry(root/'MANIFEST.json'),
            'partition_manifest': teacher_cache.entry(root/'train/MANIFEST.json'), 'split': 'train',
            'region_count': 1, 'tile_count': 2, 'shape': part['shape'], 'order': teacher_cache.ORDER,
            'rows': teacher_cache.entry(root/'train/rows.json'), 'tiles': teacher_cache.entry(root/'train/tiles.raw')}
        identity['partitions'][source] = descriptor
        (folder/source).mkdir(); np.save(folder/source/'base_logits.npy', np.zeros((2, 25), dtype=np.float32))
        manifest['partitions'][source] = {'base_logits': {'path': source+'/base_logits.npy',
            'sha256': module.sha(folder/source/'base_logits.npy')}}
        arrays[source] = {'base_logits': np.zeros((2, 25), dtype=np.float32)}
        if source != 'original': selection['supplement_data' if source == 'supplement' else 'known_data'] = descriptor['data_manifest']
        for path in root.rglob('*'):
            if path.is_file(): identity['bindings'][str(path)] = module.sha(path)
    module.dump(folder/'CACHE_MANIFEST.json', manifest)
    selection['teacher_cache'] = teacher_cache.entry(folder/'CACHE_MANIFEST.json')
    selection['bindings'].update(identity['bindings'])
    for path in folder.rglob('*'):
        if path.is_file(): selection['bindings'][str(path)] = module.sha(path)
    monkeypatch.setattr(trainer, 'TEACHER_CACHE_SHA', selection['teacher_cache']['sha256'])
    monkeypatch.setattr(teacher_cache, 'load_cache', lambda path: (arrays, manifest))
    return selection, roots['original'], manifest


def test_b08_cache_partitions_match_actual_three_dataset_identities(monkeypatch, tmp_path):
    selection, data, manifest = fixture_cache(monkeypatch, tmp_path)
    assert module.validate_training_teacher(selection, data) == manifest


@pytest.mark.parametrize('fault', ['core_teacher', 'old_logits', 'old_training', 'missing_binding',
    'swapped_tiles', 'different_rows', 'bad_total', 'unbound_output'])
def test_teacher_cache_rejects_relabeling_wrong_source_order_or_missing_tiles(monkeypatch, tmp_path, fault):
    selection, data, manifest = fixture_cache(monkeypatch, tmp_path)
    if fault == 'core_teacher': selection['cached_teacher_state_sha256'] = selection['base_state_sha256']
    elif fault == 'old_logits': manifest['identity']['historical_logits_reused'] = True
    elif fault == 'old_training': selection['historical_core_caches_used_for_training'] = True
    elif fault == 'missing_binding': selection['bindings'].pop(next(iter(manifest['identity']['bindings'])))
    elif fault == 'swapped_tiles': manifest['identity']['partitions']['known']['tiles'] = manifest['identity']['partitions']['supplement']['tiles']
    elif fault == 'different_rows': manifest['identity']['partitions']['original']['rows']['sha256'] = '0'*64
    elif fault == 'bad_total': selection['training_data_counts']['combined_tiles'] += 1
    else:
        key = str(Path(selection['teacher_cache']['path']).parent/'original/base_logits.npy')
        selection['bindings'].pop(key)
    with pytest.raises(ValueError): module.validate_training_teacher(selection, data)


@pytest.fixture(scope='module')
def actual_counts():
    # Keep the exact executed sampler populations while making every same-tile
    # teacher prediction correct, including all native core and unknown rows.
    counts, known = deepcopy(paired.actual_counts.__wrapped__())
    counts.update(schema='flux-glyph-retention-teacher-preserve-counts-v1',
        objective_variant=trainer.OBJECTIVE_VARIANT, objective=trainer.OBJECTIVE,
        mask_rule=trainer.OBJECTIVE['teacher_mask'], teacher_mask_verified_against_same_tile_argmax=True,
        eligible_rows=288000, eligible_rows_per_step=[96]*3000,
        eligible_family_rows=deepcopy(counts['family_rows']), eligible_domain_rows=deepcopy(counts['domain_rows']))
    for row in counts['source_target_rows']: row['eligible_rows'] = row['rows']
    return counts, known


def test_truth_correct_core_and_unknown_rows_are_valid_kl_evidence(monkeypatch, tmp_path, actual_counts):
    run, data, selection, sampling, counts = paired.fixture_counts(monkeypatch, tmp_path, actual_counts)
    module.validate_counts(run, selection, data)
    assert counts['eligible_family_rows']['__unknown__'] == 48000
    assert counts['eligible_family_rows']['PingFang'] == counts['family_rows']['PingFang']
    assert counts['known_supplement_rows'] == 6000 and counts['original_rows'] == 273274


@pytest.mark.parametrize('fault', ['unchecked_argmax', 'inconsistent_unknown_count', 'old_objective', 'proposal_rows'])
def test_unverified_masks_and_inconsistent_actual_counts_are_rejected(monkeypatch, tmp_path, actual_counts, fault):
    run, data, selection, sampling, counts = paired.fixture_counts(monkeypatch, tmp_path, actual_counts)
    if fault == 'unchecked_argmax': counts['teacher_mask_verified_against_same_tile_argmax'] = False
    elif fault == 'inconsistent_unknown_count': counts['eligible_family_rows']['__unknown__'] -= 1
    elif fault == 'old_objective': counts['objective'] = paired.trainer.OBJECTIVE
    else: sampling['base']['known_rows'] = 144000
    module.dump(run/'TRAINING_COUNTS.json', counts); module.dump(run/'SAMPLING.json', sampling)
    selection['training_counts_sha256'] = module.sha(run/'TRAINING_COUNTS.json')
    selection['sampling_sha256'] = module.sha(run/'SAMPLING.json')
    with pytest.raises(ValueError): module.validate_counts(run, selection, data)
