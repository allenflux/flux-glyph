"""Region-mixup release contracts using synthetic caches, without CNN inference."""
import copy
import json
from collections import Counter

import numpy as np
import pytest

import test_unified_retention_export as previous
from training import export_unified_retention_region_mixup as module
import train_unified_retention_region_mixup as trainer
from flux_glyph.model_download import release_metadata


def fixture_run(monkeypatch, tmp_path):
    families = list(previous.module.retention.IOS_ANCHOR_FAMILIES)+[f'Extra {i}' for i in range(16)]+['__unknown__']
    monkeypatch.setattr(previous, 'FAMILIES', families)
    base_run, data, base, _, _, plan, parent_meta = previous.fixture_run(monkeypatch, tmp_path)
    root = base_run.parent; run = root/'region-mixup'; run.mkdir()
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
    names = ['training/train_unified_retention_region_mixup.py', 'training/retention_region_mixup_loss.py',
             'training/train_unified_retention_adapter.py', 'training/retention_adapter_network.py',
             'training/train_unified_retention_core.py', 'training/evaluate_unified_retention_region_mixup.py']
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
    protocol = {'schema': 'flux-glyph-unified-retention-region-mixup-protocol-v1', 'architecture': module.ARCHITECTURE,
        'families': families, 'objective_variant': trainer.OBJECTIVE_VARIANT, 'objective': copy.deepcopy(trainer.OBJECTIVE),
        'sampling': trainer.SAMPLING, 'fixed_runtime': copy.deepcopy(module.FIXED_RUNTIME),
        'region_tile_training': 'All tiles of each sampled region; mean-softmax supervised CE and known-only base KL.',
        'manifold_mixup': copy.deepcopy(trainer.MIXUP), 'feature_cache_reused': True, 'cached_feature_extraction_performed': False, 'steps': 2000, 'eval_every': 200,
        'batch_size': 96, 'learning_rate': trainer.LEARNING_RATE, 'minimum_learning_rate': trainer.MINIMUM_LEARNING_RATE,
        'weight_decay': 1e-4, 'gradient_clip_norm': 5., 'seed': 2026091406, 'training_device': 'cpu', 'feature_device': 'mps',
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
    rows = [{'family': family, 'target': families.index(family), 'domain': 'android' if family == '__unknown__' else 'ios', 'source_font_family': family}
            for family, count in counts.items() for _ in range(count)]
    counter = trainer.MixupCounts()
    counter.update(rows, [r['family'] not in (*trainer.core.CORE_FAMILIES, '__unknown__') for r in rows],
        [2. if r['family'] in (*trainer.core.CORE_FAMILIES, '__unknown__') else 1. for r in rows],
        [2]*96, .3, list(reversed(range(96))))
    report = counter.report()
    for key in ('steps', 'rows', 'weight_two_rows', 'base_kl_rows', 'full_region_tiles', 'mixup_batches',
                'mixup_examples', 'mixup_lambda_sum', 'mixup_lambda_square_sum', 'mixup_different_target_pairs', 'mixup_self_pairs'):
        report[key] *= 2000
    for name in ('full_region_tile_histogram', 'mixup_pairs_by_unknown_count'):
        report[name] = {key: value*2000 for key, value in report[name].items()}
    for row in report['source_target_rows']:
        for key in ('rows', 'weight_two_rows', 'base_kl_rows'):row[key] *= 2000
    sampling = {'slots': {key: trainer.SAMPLING[key]*2000 for key in
        ('base_known', 'base_unknown', 'ios_native_pingfang', 'ios_native_sfpro_helvetica', 'ios_native_original_eight')},
        'base': {'known_rows': 96000, 'unknown_rows': 32000}, 'family_rows': {key: value*2000 for key, value in counts.items()},
        'domain_rows': {'ios': 160000, 'android': 32000}, 'view_rows': {'native': 192000}}
    module.dump(run/'TRAINING_COUNTS.json', report); module.dump(run/'SAMPLING.json', sampling)
    selected = history[0]
    for key, name in [('outputs', 'CALIBRATION_OUTPUTS.npz'), ('decisions', 'CALIBRATION_DECISIONS.json')]:
        (run/name).write_bytes((run/selected['artifacts'][key]['path']).read_bytes())
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection = {**copy.deepcopy(protocol), 'schema': 'flux-glyph-unified-retention-region-mixup-selection-v1',
        'selected': selected, 'history': history, 'promotion_allowed': True, 'passed': False, 'calibration_passed': False,
        'training_protocol_sha256': module.sha(run/'TRAINING_FREEZE.json'), 'state_after_sha256': 'e'*64,
        'base_state_after_sha256': base['state_after_sha256'], 'residual_state_after_sha256': 'f'*64,
        'sampling_sha256': module.sha(run/'SAMPLING.json'), 'training_counts_sha256': module.sha(run/'TRAINING_COUNTS.json'),
        'calibration_outputs_sha256': module.sha(run/'CALIBRATION_OUTPUTS.npz'),
        'calibration_decisions_sha256': module.sha(run/'CALIBRATION_DECISIONS.json')}
    module.dump(run/'SELECTION.json', selection)
    return run, data, selection, protocol, cal, parent_meta


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


@pytest.mark.parametrize('fault', ['old_schema', 'logit_mean', 'mixup_weight', 'mixed_unknown_label',
    'mixup_deployed', 'fresh_cache', 'cache_digest', 'loss_binding', 'evaluator_binding'])
def test_new_region_mixup_protocol_cannot_be_replaced_with_another_objective(monkeypatch, tmp_path, fault):
    run, data, selection, protocol, *_ = fixture_run(monkeypatch, tmp_path)
    if fault == 'old_schema':
        selection['schema'] = 'flux-glyph-unified-retention-adapter-selection-v1'
    elif fault == 'logit_mean':
        protocol['objective']['region_probabilities'] = 'Softmax of mean raw tile logits'
        selection['objective'] = copy.deepcopy(protocol['objective'])
    elif fault in ('mixup_weight', 'mixed_unknown_label', 'mixup_deployed'):
        key, value = {'mixup_weight': ('loss_weight', 1.),
            'mixed_unknown_label': ('synthetic_unknown_labels', True), 'mixup_deployed': ('deployed', True)}[fault]
        for document in (selection, protocol):
            document['manifold_mixup'][key] = value
            document['objective']['mixup'][key] = value
    elif fault == 'fresh_cache':
        selection['feature_cache_reused'] = protocol['feature_cache_reused'] = False
    elif fault == 'cache_digest':
        monkeypatch.setattr(trainer, 'CACHE_MANIFEST_SHA', '0'*64)
    else:
        name = 'retention_region_mixup_loss.py' if fault == 'loss_binding' else 'evaluate_unified_retention_region_mixup.py'
        path = str(module.ROOT/'training'/name)
        for document in (selection, protocol):document['bindings'].pop(path, None)
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection['training_protocol_sha256'] = module.sha(run/'TRAINING_FREEZE.json')
    module.dump(run/'SELECTION.json', selection)
    with pytest.raises(ValueError):module.validate(run, data)


@pytest.mark.parametrize('fault', ['unknown_kl', 'tile_histogram', 'tile_total', 'pair_total',
    'unknown_pair_conservation', 'lambda_moments', 'synthetic_unknown', 'mixup_deployed', 'mixup_steps'])
def test_region_and_mixup_statistics_are_checked_after_rebinding_changed_report(monkeypatch, tmp_path, fault):
    run, data, selection, *_ = fixture_run(monkeypatch, tmp_path)
    report = trainer.read(run/'TRAINING_COUNTS.json')
    if fault == 'unknown_kl':
        row = next(r for r in report['source_target_rows'] if r['target_family'] == '__unknown__')
        row['base_kl_rows'] = 1; report['base_kl_rows'] += 1
    elif fault == 'tile_histogram':report['full_region_tile_histogram']['2'] -= 1
    elif fault == 'tile_total':report['full_region_tiles'] -= 1
    elif fault == 'pair_total':report['mixup_pairs_by_unknown_count']['0'] -= 1
    elif fault == 'unknown_pair_conservation':
        report['mixup_pairs_by_unknown_count']['0'] += 1
        report['mixup_pairs_by_unknown_count']['1'] -= 1
    elif fault == 'lambda_moments':report['mixup_lambda_square_sum'] = report['mixup_lambda_sum'] + 1
    elif fault == 'synthetic_unknown':report['synthetic_unknown_labels'] = True
    elif fault == 'mixup_deployed':report['mixup_deployed'] = True
    else:report['mixup_batches'] -= 1
    module.dump(run/'TRAINING_COUNTS.json', report)
    selection['training_counts_sha256'] = module.sha(run/'TRAINING_COUNTS.json')
    module.dump(run/'SELECTION.json', selection)
    with pytest.raises(ValueError):module.validate(run, data)


def test_mixup_provenance_survives_metadata_without_a_deployed_mixup_branch(monkeypatch, tmp_path):
    _, _, selection, _, cal, parent = fixture_run(monkeypatch, tmp_path)
    metadata = module.metadata_for_export(selection, cal, parent, '0'*64, '1'*64, '2'*64,
                                         module.strip_artifacts(selection['selected']))
    training = metadata['training']
    assert training['region_tile_training'] == selection['region_tile_training']
    assert training['manifold_mixup'] == trainer.MIXUP
    assert training['feature_cache_reused'] is True
    assert training['mixup_deployed'] is training['synthetic_unknown_labels'] is False
    assert training['model_count'] == training['encoder_count'] == 1
    assert metadata['validation']['kind'] == 'frozen_encoder_region_mixup_retention_calibration_only_at_export'
    with pytest.raises(ValueError):
        module.metadata_for_export({**selection, 'promotion_allowed': False}, cal, parent,
                                   '0'*64, '1'*64, '2'*64, module.strip_artifacts(selection['selected']))
