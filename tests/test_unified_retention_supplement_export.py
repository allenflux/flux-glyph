"""Native TRAIN supplement release contracts with bound synthetic caches."""
import copy
import json
from collections import Counter

import numpy as np
import pytest

import test_unified_retention_export as previous
from training import export_unified_retention_supplement as module
import train_unified_retention_supplement as trainer
from flux_glyph.model_download import release_metadata


def fixture_run(monkeypatch, tmp_path):
    families = list(previous.module.retention.IOS_ANCHOR_FAMILIES)+[f'Extra {i}' for i in range(16)]+['__unknown__']
    monkeypatch.setattr(previous, 'FAMILIES', families)
    base_run, data, base, _, _, plan, parent_meta = previous.fixture_run(monkeypatch, tmp_path)
    root = base_run.parent; run = root/'supplement-run'; run.mkdir()
    monkeypatch.setattr(module, 'ROOT', root)
    monkeypatch.setattr(trainer.core, 'read_plan', previous.module.retention.read_plan)
    for split in ('train', 'calibration'):
        path = data/split/'MANIFEST.json'; part = trainer.read(path); part['tiles'] = 3
        module.dump(path, part); base['bindings'][str(path)] = module.sha(path)
    base.update(optimizer_steps_executed=1500, promotion_allowed=False)
    base['selected']['step'] = 1000
    module.dump(base_run/'SELECTION.json', base); (base_run/'model.pth').write_bytes(b'Frozen core checkpoint')
    monkeypatch.setattr(trainer, 'BASE_CHECKPOINT_SHA', module.sha(base_run/'model.pth'))
    monkeypatch.setattr(trainer, 'BASE_SELECTION_SHA', module.sha(base_run/'SELECTION.json'))
    bindings = dict(base['bindings'])
    names = ['training/train_unified_retention_supplement.py', 'training/retention_supplement_sampler.py',
             'training/prepare_unified_unknown_supplement.py', 'training/cache_unified_unknown_supplement.py',
             'training/train_unified_retention_adapter.py', 'training/retention_adapter_network.py',
             'training/train_unified_retention_core.py', 'training/evaluate_unified_retention_supplement.py']
    for name in names:
        path = root/name; path.write_text('Bound synthetic source '+name); bindings[str(path)] = module.sha(path)
    for name in ('model.pth', 'SELECTION.json', 'TRAINING_FREEZE.json', 'CALIBRATION_OUTPUTS.npz', 'CALIBRATION_DECISIONS.json'):
        bindings[str(base_run/name)] = module.sha(base_run/name)
    base_checkpoint = {'path': str(base_run/'model.pth'), 'sha256': module.sha(base_run/'model.pth')}
    base_selection = {'path': str(base_run/'SELECTION.json'), 'sha256': module.sha(base_run/'SELECTION.json')}
    parts = {}
    for split in ('train', 'calibration'):
        path = data/split/'MANIFEST.json'; part = trainer.read(path)
        parts[split] = {'partition_sha256': module.sha(path), 'tile_count': 3, 'rows': part['metadata'], 'tiles': part['array'],
            'order': 'Exact prepared tile array order; rows retain tile_start and tile_count.'}
    identity = {'architecture': module.ARCHITECTURE, 'base_checkpoint': base_checkpoint,
        'base_state_sha256': base['state_after_sha256'], 'feature_device': 'mps', 'families': families,
        'data_manifest_sha256': base['data_manifest_sha256'], 'partitions': parts, 'bindings': dict(bindings),
        'test_read': False, 'development_holdout_read': False}
    cache = run/'cache'
    module.dump(cache/'CACHE_FREEZE.json', {'schema': 'flux-glyph-retention-adapter-cache-freeze-v1',
        'identity': identity, 'feature_device': 'mps', 'batch_size': 128, 'optimizer_steps_executed': 0})
    logits, sizes = previous.sample_outputs(); partitions = {}
    for split in ('train', 'calibration'):
        (cache/split).mkdir(); partitions[split] = {}
        for name, array in [('features', np.zeros((3, 128), np.float32)), ('base_logits', logits), ('log_em_ratio', sizes)]:
            path = cache/split/(name+'.npy'); np.save(path, array)
            partitions[split][name] = {'path': f'{split}/{name}.npy', 'sha256': module.sha(path),
                'shape': list(array.shape), 'dtype': 'float32'}
            bindings[str(path)] = module.sha(path)
    module.dump(cache/'CACHE_MANIFEST.json', {'schema': 'flux-glyph-retention-adapter-cache-v1', 'identity': identity,
        'cache_freeze_sha256': module.sha(cache/'CACHE_FREEZE.json'), 'partitions': partitions})
    for name in ('CACHE_MANIFEST.json', 'CACHE_FREEZE.json'):bindings[str(cache/name)] = module.sha(cache/name)
    monkeypatch.setattr(trainer, 'CACHE_MANIFEST_SHA', module.sha(cache/'CACHE_MANIFEST.json'))
    initial = run/'INITIAL_RESIDUAL.pth'; initial.write_bytes(b'Bound zero-output head fixture'); bindings[str(initial)] = module.sha(initial)
    cal = module.calibration_metadata(data)
    baseline, outputs, _ = module.evaluate_outputs(logits, sizes, cal, plan, 0)
    module.dump(run/'BASELINE.json', baseline)
    module.dump(run/'BASELINE_CALIBRATION_DECISIONS.json', {'families': families, 'records': outputs})
    protocol = {'schema': 'flux-glyph-unified-retention-supplement-protocol-v1', 'architecture': module.ARCHITECTURE,
        'families': families, 'objective_variant': trainer.OBJECTIVE_VARIANT, 'objective': copy.deepcopy(trainer.OBJECTIVE),
        'sampling': trainer.SAMPLING, 'fixed_runtime': copy.deepcopy(module.FIXED_RUNTIME), 'steps': 2000, 'eval_every': 200,
        'batch_size': 96, 'learning_rate': trainer.LEARNING_RATE, 'minimum_learning_rate': trainer.MINIMUM_LEARNING_RATE,
        'weight_decay': 1e-4, 'gradient_clip_norm': 5., 'seed': 2026091407, 'training_device': 'cpu', 'feature_device': 'mps',
        'base_checkpoint': base_checkpoint, 'base_selection': base_selection, 'base_state_sha256': base['state_after_sha256'],
        'base_selected_step': 1000, 'source_optimizer_steps_executed': 1500, 'base_optimizer_steps': 0,
        'residual_optimizer_steps': 2000, 'optimizer_steps_executed': 2000, 'initial_state_sha256': 'c'*64,
        'residual_state_before_sha256': 'd'*64, 'cache_manifest': {'path': str(cache/'CACHE_MANIFEST.json'), 'sha256': module.sha(cache/'CACHE_MANIFEST.json')},
        'bindings': bindings, 'data_manifest_sha256': base['data_manifest_sha256'], 'retention_plan': base['retention_plan'],
        **{key: base[key] for key in ('parent_metadata', 'parent_checkpoint', 'parent_selection_sha256')},
        'baseline_bindings': {name: module.sha(run/name) for name in ('BASELINE.json', 'BASELINE_CALIBRATION_DECISIONS.json')},
        'initial_residual': {'path': str(initial), 'sha256': module.sha(initial)}, 'frozen_groups': list(module.BASE_GROUPS),
        'trainable_groups': ['residual_family_head'], 'all_parameters_trained': False, 'base_frozen': True,
        'size_head_frozen': True, 'residual_head_trained': True, 'model_count': 1, 'encoder_count': 1,
        'platform_routing': False, 'score_merging': False, 'external_teacher': False, 'test_read': False,
        'development_holdout_read': False, 'runtime_gates_searched': False, 'training_inputs': ['image_features_only'],
        'calibration_execution': 'Frozen MPS/CPU base logits + CPU residual(features); copied frozen size logits.'}
    add_supplement(monkeypatch, root, data, cache, base, bindings, protocol)
    history = []
    for step in range(200, 2001, 200):
        record, outputs, _ = module.evaluate_outputs(logits, sizes, cal, plan, step)
        folder = run/'checkpoints'/f'step{step:05d}'; folder.mkdir(parents=True)
        (folder/'model.pth').write_bytes(f'Residual step {step}'.encode())
        np.savez(folder/'CALIBRATION_OUTPUTS.npz', logits=logits, log_em_ratio=sizes)
        module.dump(folder/'CALIBRATION_DECISIONS.json', {'families': families, 'records': outputs})
        module.dump(folder/'METRICS.json', record)
        record['artifacts'] = {key: {'path': str((folder/name).relative_to(run)), 'sha256': module.sha(folder/name)} for key, name in
            [('checkpoint', 'model.pth'), ('outputs', 'CALIBRATION_OUTPUTS.npz'), ('decisions', 'CALIBRATION_DECISIONS.json'), ('metrics', 'METRICS.json')]}
        history.append(record)
    counts = {family: 2 for family in families}; counts['__unknown__'] = 16; counts['PingFang'] += 16
    for family in ('SF Pro', 'Helvetica'):counts[family] += 4
    for family in previous.module.retention.IOS_ANCHOR_FAMILIES:counts[family] += 1
    rows = [{'family': family, 'domain': 'android' if family == '__unknown__' else 'ios', 'source_font_family': family}
            for family, count in counts.items() for _ in range(count)]
    unknown_rows = [row for row in rows if row['family'] == '__unknown__']
    for index, row in enumerate(unknown_rows[:8]):
        row.update({trainer.SUPPLEMENT_MARKER: True, 'source_font_family': trainer.NEW_SOURCES[index//4]})
    counter = trainer.SupplementCounts()
    counter.update(rows, [r['family'] not in trainer.core.CORE_FAMILIES for r in rows],
        [2. if r['family'] in (*trainer.core.CORE_FAMILIES, '__unknown__') else 1. for r in rows])
    report = counter.report()
    for key in ('steps', 'rows', 'weight_two_rows', 'base_kl_rows', 'supplement_rows', 'original_rows'):report[key] *= 2000
    report['supplement_source_rows'] = {key: value*2000 for key, value in report['supplement_source_rows'].items()}
    for row in report['source_target_rows']:
        for key in ('rows', 'weight_two_rows', 'base_kl_rows'):row[key] *= 2000
    sampling = {'schema': 'flux-glyph-retention-supplement-sampling-v1',
        'slots': {key: trainer.SAMPLING[key]*2000 for key in
        ('base_known', 'base_unknown', 'supplement_unknown', 'ios_native_pingfang', 'ios_native_sfpro_helvetica', 'ios_native_original_eight')},
        'base': {'known_rows': 96000, 'unknown_rows': 16000}, 'family_rows': {key: value*2000 for key, value in counts.items()},
        'domain_rows': {'ios': 160000, 'android': 32000}, 'view_rows': {'native': 192000},
        'supplement_source_rows': dict(report['supplement_source_rows']), 'supplement_view_rows': {'native': 16000},
        'source_rows': [{key: row[key] for key in ('domain', 'source_font_family', 'target_family', 'rows')}
                        for row in report['source_target_rows']],
        'proposal_rows_discarded': 0, 'test_read': False, 'development_holdout_read': False}
    module.dump(run/'TRAINING_COUNTS.json', report); module.dump(run/'SAMPLING.json', sampling)
    selected = history[0]
    for key, name in [('outputs', 'CALIBRATION_OUTPUTS.npz'), ('decisions', 'CALIBRATION_DECISIONS.json')]:
        (run/name).write_bytes((run/selected['artifacts'][key]['path']).read_bytes())
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection = {**copy.deepcopy(protocol), 'schema': 'flux-glyph-unified-retention-supplement-selection-v1',
        'selected': selected, 'history': history, 'promotion_allowed': True, 'passed': False, 'calibration_passed': False,
        'training_protocol_sha256': module.sha(run/'TRAINING_FREEZE.json'), 'state_after_sha256': 'e'*64,
        'base_state_after_sha256': base['state_after_sha256'], 'residual_state_after_sha256': 'f'*64,
        'sampling_sha256': module.sha(run/'SAMPLING.json'), 'training_counts_sha256': module.sha(run/'TRAINING_COUNTS.json'),
        'calibration_outputs_sha256': module.sha(run/'CALIBRATION_OUTPUTS.npz'),
        'calibration_decisions_sha256': module.sha(run/'CALIBRATION_DECISIONS.json')}
    module.dump(run/'SELECTION.json', selection)
    return run, data, selection, protocol, cal, parent_meta



def add_supplement(monkeypatch, root, data, old_cache, base, bindings, protocol):
    import hashlib
    import cache_unified_unknown_supplement as supplement_cache
    monkeypatch.setattr(supplement_cache, 'ROOT', root)
    for name, value in [('BASE_CHECKPOINT_SHA', trainer.BASE_CHECKPOINT_SHA),
                        ('BASE_SELECTION_SHA', trainer.BASE_SELECTION_SHA),
                        ('BASE_STATE_SHA', base['state_after_sha256']),
                        ('OLD_CACHE_MANIFEST_SHA', module.sha(old_cache/'CACHE_MANIFEST.json')),
                        ('OLD_CACHE_FREEZE_SHA', module.sha(old_cache/'CACHE_FREEZE.json'))]:
        monkeypatch.setattr(supplement_cache, name, value)
    folder = root/'extra-data'; (folder/'train').mkdir(parents=True)
    proof = folder/'native-proof.json'; proof.write_text('Independent verified native capture')
    preparation = {'schema': 'flux-glyph-unified-unknown-supplement-freeze-v1', 'families': protocol['families'],
                   'only_split': 'train', 'bindings': {str(proof): module.sha(proof)}}
    module.dump(folder/'PREPARATION_FREEZE.json', preparation)
    manifest = {**preparation, 'schema': supplement_cache.DATA_SCHEMA,
                'preparation_freeze_sha256': module.sha(folder/'PREPARATION_FREEZE.json')}
    module.dump(folder/'MANIFEST.json', manifest)
    tiles = np.zeros((2, 1, 64, 256), np.float32); tiles.tofile(folder/'train/tiles.raw')
    rows = [{'split': 'train', 'family': '__unknown__', 'target': 24, 'domain': 'android',
             'native_font_verified': True, 'source_font_family': name,
             'source_id': 'extra-'+str(i), 'region_id': 'region-'+str(i), 'tile_start': i, 'tile_count': 1,
             'tiles_sha256': hashlib.sha256(tiles[i:i+1].tobytes()).hexdigest()}
            for i, name in enumerate(trainer.NEW_SOURCES)]
    module.dump(folder/'train/rows.json', rows)
    part = {'schema': supplement_cache.DATA_SCHEMA, 'split': 'train', 'families': protocol['families'],
        'root_manifest_sha256': module.sha(folder/'MANIFEST.json'), 'tiles': 2, 'views': 2, 'shape': [2, 1, 64, 256],
        'metadata': {'path': 'rows.json', 'sha256': module.sha(folder/'train/rows.json')},
        'array': {'path': 'tiles.raw', 'sha256': module.sha(folder/'train/tiles.raw')}}
    module.dump(folder/'train/MANIFEST.json', part)
    for path in [proof, folder/'MANIFEST.json', folder/'PREPARATION_FREEZE.json',
                 *[folder/'train'/name for name in ('MANIFEST.json', 'rows.json', 'tiles.raw')]]:
        bindings[str(path)] = module.sha(path)
    extra = {'tiles': tiles, 'rows': rows, 'families': protocol['families'], 'manifest': manifest, 'partition': part,
             'manifest_sha256': module.sha(folder/'MANIFEST.json'), 'partition_sha256': module.sha(folder/'train/MANIFEST.json')}
    identity = supplement_cache.cache_identity(extra, protocol['base_checkpoint'], base['state_after_sha256'],
        bindings, 'cpu', data_root=folder, old_cache=old_cache)
    cache = root/'extra-cache'; (cache/'train').mkdir(parents=True)
    supplement_cache.dump(cache/'CACHE_FREEZE.json', {'schema': supplement_cache.FREEZE_SCHEMA, 'identity': identity,
        'feature_device': 'cpu', 'batch_size': 128, 'optimizer_steps_executed': 0})
    arrays = {}
    for name, shape in [('features', (2, 128)), ('base_logits', (2, 25)), ('log_em_ratio', (2,))]:
        path = cache/'train'/(name+'.npy'); np.save(path, np.zeros(shape, np.float32)); bindings[str(path)] = module.sha(path)
        arrays[name] = {'path': 'train/'+name+'.npy', 'sha256': module.sha(path), 'dtype': 'float32', 'shape': list(shape)}
    supplement_cache.dump(cache/'CACHE_MANIFEST.json', {'schema': supplement_cache.CACHE_SCHEMA, 'identity': identity,
        'cache_freeze_sha256': module.sha(cache/'CACHE_FREEZE.json'), 'partitions': {'train': arrays},
        'base_state_before_sha256': base['state_after_sha256'], 'base_state_after_sha256': base['state_after_sha256'],
        'optimizer_steps_executed': 0})
    for name in ('CACHE_MANIFEST.json', 'CACHE_FREEZE.json'):bindings[str(cache/name)] = module.sha(cache/name)
    original_part = trainer.read(data/'train/MANIFEST.json'); original_rows = trainer.read(data/'train'/original_part['metadata']['path'])
    n = len({(row['source_id'], row['region_id']) for row in original_rows})
    protocol.update(feature_cache_reused=True, cached_feature_extraction_performed=False,
        supplement_data={'path': str(folder/'MANIFEST.json'), 'sha256': module.sha(folder/'MANIFEST.json')},
        supplement_cache={'path': str(cache/'CACHE_MANIFEST.json'), 'sha256': module.sha(cache/'CACHE_MANIFEST.json')},
        training_data_counts={'original_views': len(original_rows), 'supplement_views': 2, 'combined_views': len(original_rows)+2,
            'original_native_regions': n, 'supplement_native_regions': 2, 'combined_native_regions': n+2,
            'original_tiles': original_part['tiles'], 'supplement_tiles': 2, 'combined_tiles': original_part['tiles']+2,
            'calibration_unchanged': True, 'derived_views_are_correlated': True})
    monkeypatch.setattr(trainer, 'SUPPLEMENT_MANIFEST_SHA', module.sha(folder/'MANIFEST.json'))
    monkeypatch.setattr(trainer, 'SUPPLEMENT_CACHE_SHA', module.sha(cache/'CACHE_MANIFEST.json'))

def test_complete_residual_only_contract_uses_no_torch_or_holdout(monkeypatch, tmp_path):
    run, data, selection, *_ = fixture_run(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    assert module.validate(run, data) == selection
    assert selection['passed'] is False and selection['promotion_allowed'] is True
    assert not (data/'test').exists() and not (data/'development_holdout').exists()


@pytest.mark.parametrize('fault', ['all_parameters', 'base_optimizer', 'encoder_count', 'size_loss', 'seed', 'gate',
    'missing_history', 'wrong_selection', 'base_state', 'feature_bytes', 'size_bytes', 'cache_binding',
    'base_binding', 'counts', 'failed', 'stable', 'metadata_provenance'])
def test_no_release_after_contract_or_cached_evidence_changes(monkeypatch, tmp_path, fault):
    run, data, selection, protocol, *_ = fixture_run(monkeypatch, tmp_path)
    if fault in ('all_parameters', 'base_optimizer', 'encoder_count', 'seed'):
        key, value = {'all_parameters': ('all_parameters_trained', True), 'base_optimizer': ('base_optimizer_steps', 1),
            'encoder_count': ('encoder_count', 2), 'seed': ('seed', 99)}[fault]
        selection[key] = protocol[key] = value
    elif fault == 'size_loss':
        selection['objective']['size_loss_weight'] = protocol['objective']['size_loss_weight'] = .2
    elif fault == 'gate':selection['fixed_runtime']['gates']['min_score'] = protocol['fixed_runtime']['gates']['min_score'] = .5
    elif fault == 'missing_history':selection['history'].pop()
    elif fault == 'wrong_selection':selection['selected'] = copy.deepcopy(selection['history'][-1])
    elif fault == 'base_state':selection['base_state_after_sha256'] = '0'*64
    elif fault in ('feature_bytes', 'size_bytes'):
        name = 'features' if fault == 'feature_bytes' else 'log_em_ratio'
        (run/'cache/calibration'/(name+'.npy')).write_bytes(b'Changed cache')
    elif fault in ('cache_binding', 'base_binding'):
        key = str(run/'cache/CACHE_MANIFEST.json') if fault == 'cache_binding' else selection['base_checkpoint']['path']
        selection['bindings'].pop(key); protocol['bindings'].pop(key)
    elif fault == 'counts':selection['training_counts_sha256'] = '0'*64
    elif fault == 'failed':selection['promotion_allowed'] = False
    elif fault == 'stable':selection['passed'] = selection['calibration_passed'] = True
    elif fault == 'metadata_provenance':
        selection['parent_metadata']['sha256'] = protocol['parent_metadata']['sha256'] = '0'*64
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection['training_protocol_sha256'] = module.sha(run/'TRAINING_FREEZE.json'); module.dump(run/'SELECTION.json', selection)
    with pytest.raises(ValueError):module.validate(run, data)


def test_complete_adapter_metadata_is_one_cnn_with_only_residual_training(monkeypatch, tmp_path):
    _, _, selection, _, cal, parent = fixture_run(monkeypatch, tmp_path)
    metadata = module.metadata_for_export(selection, cal, parent, '0'*64, '1'*64, '2'*64, module.strip_artifacts(selection['selected']))
    assert metadata['network_architecture'] == module.ARCHITECTURE
    assert metadata['font_sources'] == parent['font_sources']
    assert metadata['families'] == selection['families'] and metadata['gates'] == module.FIXED_RUNTIME['gates']
    assert metadata['test_passed'] is metadata['stable_validation_passed'] is False
    assert metadata['training']['all_parameters_trained'] is False
    assert metadata['training']['base_frozen'] is metadata['training']['size_head_frozen'] is True
    assert metadata['training']['model_count'] == metadata['training']['encoder_count'] == 1
    assert metadata['training']['trainable_parameter_groups'] == ['residual_family_head']
    for key in ('objective_variant', 'objective', 'training_counts_sha256', 'base_checkpoint', 'base_selection', 'cache_manifest'):
        assert metadata['training'][key] == selection[key]
    assert release_metadata(metadata)['validation'] == metadata['validation']
    assert len(json.dumps(metadata).encode()) < 65536


def test_final_state_may_change_only_residual_keys(monkeypatch):
    monkeypatch.setattr(module, 'state_sha', lambda state: json.dumps(state, sort_keys=True))
    state = {'trunk.weight': 'frozen', **{module.RESIDUAL_PREFIX+key: 'trained' for key in ('0.weight', '0.bias', '2.weight', '2.bias')}}
    selection = {'families': ['F']*25, 'state_after_sha256': module.state_sha(state),
        'base_state_sha256': module.state_sha({'trunk.weight': 'frozen'}), 'base_state_after_sha256': module.state_sha({'trunk.weight': 'frozen'}),
        'residual_state_after_sha256': module.state_sha(trainer.residual_state(state)), 'residual_state_before_sha256': 'initial'}
    checkpoint = {'state_dict': state, 'families': selection['families'], 'architecture': module.ARCHITECTURE, 'selection_sha256': 'selected'}
    module.validate_checkpoint(checkpoint, selection, 'selected')
    state['trunk.weight'] = 'changed'; selection['state_after_sha256'] = module.state_sha(state)
    with pytest.raises(ValueError):module.validate_checkpoint(checkpoint, selection, 'selected')


@pytest.mark.parametrize('fault', ['new_data_digest', 'new_cache_digest', 'new_features', 'supplement_binding',
    'sampler_binding', 'dataset_counts', 'calibration_claim', 'old_schema', 'new_objective'])
def test_bound_supplement_identity_and_original_cal_scope_cannot_change(monkeypatch, tmp_path, fault):
    run, data, selection, protocol, *_ = fixture_run(monkeypatch, tmp_path)
    if fault == 'new_data_digest':monkeypatch.setattr(trainer, 'SUPPLEMENT_MANIFEST_SHA', '0'*64)
    elif fault == 'new_cache_digest':monkeypatch.setattr(trainer, 'SUPPLEMENT_CACHE_SHA', '0'*64)
    elif fault == 'new_features':(module.ROOT/'extra-cache/train/features.npy').write_bytes(b'Changed new features')
    elif fault in ('supplement_binding', 'sampler_binding'):
        path = selection['supplement_cache']['path'] if fault == 'supplement_binding' else str(module.ROOT/'training/retention_supplement_sampler.py')
        for document in (selection, protocol):document['bindings'].pop(path, None)
    elif fault in ('dataset_counts', 'calibration_claim'):
        for document in (selection, protocol):
            if fault == 'dataset_counts':document['training_data_counts']['supplement_views'] += 1
            else:document['training_data_counts']['calibration_unchanged'] = False
    elif fault == 'old_schema':selection['schema'] = 'flux-glyph-unified-retention-adapter-selection-v1'
    else:
        for document in (selection, protocol):document['objective']['base_kl_mask'] = 'Exclude unknown targets'
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection['training_protocol_sha256'] = module.sha(run/'TRAINING_FREEZE.json')
    module.dump(run/'SELECTION.json', selection)
    with pytest.raises(ValueError):module.validate(run, data)


@pytest.mark.parametrize('fault', ['old_unknown_quota', 'new_unknown_quota', 'new_source_balance',
    'discarded_proposals', 'actual_source_rows', 'new_view_total', 'supplement_total', 'original_total',
    'new_source_label', 'family_quota'])
def test_actual_replay_populations_are_verified_even_after_report_hashes_are_rebound(monkeypatch, tmp_path, fault):
    run, data, selection, *_ = fixture_run(monkeypatch, tmp_path)
    sampling = trainer.read(run/'SAMPLING.json'); counts = trainer.read(run/'TRAINING_COUNTS.json')
    if fault == 'old_unknown_quota':sampling['base']['unknown_rows'] = 32000
    elif fault == 'new_unknown_quota':sampling['slots']['supplement_unknown'] = 8000
    elif fault == 'new_source_balance':sampling['supplement_source_rows'][trainer.NEW_SOURCES[0]] -= 1
    elif fault == 'discarded_proposals':sampling['proposal_rows_discarded'] = 16000
    elif fault == 'actual_source_rows':sampling['source_rows'][0]['rows'] -= 1
    elif fault == 'new_view_total':sampling['supplement_view_rows']['native'] -= 1
    elif fault == 'supplement_total':counts['supplement_rows'] -= 1
    elif fault == 'original_total':counts['original_rows'] += 1
    elif fault == 'new_source_label':
        row = next(r for r in counts['source_target_rows'] if r['source_font_family'] == trainer.NEW_SOURCES[0])
        row['target_family'] = 'PingFang'
    else:
        sampling['family_rows']['PingFang'] -= 1
        sampling['family_rows']['SF Pro'] += 1
    module.dump(run/'SAMPLING.json', sampling); module.dump(run/'TRAINING_COUNTS.json', counts)
    selection['sampling_sha256'] = module.sha(run/'SAMPLING.json')
    selection['training_counts_sha256'] = module.sha(run/'TRAINING_COUNTS.json')
    module.dump(run/'SELECTION.json', selection)
    with pytest.raises(ValueError):module.validate(run, data)


def test_all_extra_training_provenance_fields_survive_export_metadata_exactly(monkeypatch, tmp_path):
    _, _, selection, _, cal, parent = fixture_run(monkeypatch, tmp_path)
    metadata = module.metadata_for_export(selection, cal, parent, '0'*64, '1'*64, '2'*64,
                                         module.strip_artifacts(selection['selected']))
    for key in ('supplement_data', 'supplement_cache', 'training_data_counts'):
        assert metadata['training'][key] == selection[key]
        assert metadata['training'][key] is not selection[key]
    assert metadata['training']['objective'] == trainer.OBJECTIVE
    assert metadata['training']['model_count'] == metadata['training']['encoder_count'] == 1
    assert metadata['training']['raw_size_branch_frozen'] is True
    assert metadata['validation']['kind'] == 'frozen_encoder_native_unknown_supplement_calibration_only_at_export'
    with pytest.raises(ValueError):
        module.metadata_for_export({**selection, 'promotion_allowed': False}, cal, parent,
                                   '0'*64, '1'*64, '2'*64, module.strip_artifacts(selection['selected']))
