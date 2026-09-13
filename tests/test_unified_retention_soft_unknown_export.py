"""Soft-unknown full-CNN release contracts without training, export or holdout access."""
from copy import deepcopy
import json

import numpy as np
import pytest

import test_unified_retention_supplement_export as previous
from training import export_unified_retention_soft_unknown as module
import train_unified_retention_soft_unknown as trainer
from flux_glyph.model_download import release_metadata


def fixture_run(monkeypatch, tmp_path):
    run, data, old_selection, old_protocol, cal, parent = previous.fixture_run(monkeypatch, tmp_path)
    root = run.parent; monkeypatch.setattr(module, 'ROOT', root)
    for key in ('BASE_CHECKPOINT_SHA', 'BASE_SELECTION_SHA', 'CACHE_MANIFEST_SHA', 'SUPPLEMENT_MANIFEST_SHA', 'SUPPLEMENT_CACHE_SHA'):
        monkeypatch.setattr(trainer, key, getattr(previous.trainer, key))
    protocol = deepcopy(old_protocol)
    for key in ('initial_residual', 'residual_state_before_sha256', 'base_optimizer_steps', 'residual_optimizer_steps'):
        protocol.pop(key, None)
    for name in ('training/train_unified_retention_soft_unknown.py', 'training/evaluate_unified_retention_soft_unknown.py'):
        path = root/name; path.write_text('Frozen full-CNN source '+name); protocol['bindings'][str(path)] = module.sha(path)
    groups = {name: str(i+1)*64 for i, name in enumerate(module.GROUPS)}
    state = protocol['base_state_sha256']
    protocol.update(schema='flux-glyph-unified-retention-soft-unknown-protocol-v1', architecture=module.ARCHITECTURE,
        objective_variant=trainer.OBJECTIVE_VARIANT, objective=deepcopy(trainer.OBJECTIVE),
        soft_unknown_supervision=deepcopy(trainer.SOFT_UNKNOWN_SUPERVISION),
        sampling=deepcopy(trainer.SAMPLING), steps=3000, eval_every=500, seed=2026091409,
        learning_rate=2e-5, minimum_learning_rate=2e-6, optimizer_steps_executed=3000,
        initial_state_sha256=state, state_before_sha256=state, cached_teacher_state_sha256=state,
        initial_parameter_groups_sha256=groups,
        inheritance={'all_family_rows_inherited': True, 'all_parameters_inherited': True, 'all_parameters_trainable': True,
            'source_state_sha256': state, 'family_count': 25, 'new_random_output_rows': 0},
        all_parameters_trained=True, base_frozen=False, size_head_frozen=False, residual_head_trained=False,
        frozen_groups=[], trainable_groups=list(module.GROUPS), teacher_logits_used=True, second_model_resident=False,
        teacher_optimizer_steps=0, teacher_cache_deployed=False, feature_cache_reused=False, teacher_cache_reused=True,
        training_inputs=['image_tiles'],
        calibration_execution='Current complete CNN forward over all original CAL image tiles; current learned size head.')
    with np.load(run/'CALIBRATION_OUTPUTS.npz') as outputs:
        logits, sizes = outputs['logits'].copy(), outputs['log_em_ratio'].copy()
    plan = trainer.read(protocol['retention_plan']['path']); history = []
    for step in range(500, 3001, 500):
        record, predictions, _ = module.evaluate_outputs(logits, sizes, cal, plan, step)
        folder = run/'checkpoints'/f'step{step:05d}'; folder.mkdir(parents=True, exist_ok=True)
        (folder/'model.pth').write_bytes(f'Full CNN step {step}'.encode())
        np.savez(folder/'CALIBRATION_OUTPUTS.npz', logits=logits, log_em_ratio=sizes)
        module.dump(folder/'CALIBRATION_DECISIONS.json', {'families': protocol['families'], 'records': predictions})
        module.dump(folder/'METRICS.json', record)
        record['artifacts'] = {key: {'path': str((folder/name).relative_to(run)), 'sha256': module.sha(folder/name)}
            for key, name in [('checkpoint', 'model.pth'), ('outputs', 'CALIBRATION_OUTPUTS.npz'),
                              ('decisions', 'CALIBRATION_DECISIONS.json'), ('metrics', 'METRICS.json')]}
        history.append(record)
    original_counts = trainer.read(run/'TRAINING_COUNTS.json'); rows = []
    for row in original_counts['source_target_rows']:
        family = row['target_family']
        for _ in range(row['rows']//2000):
            values = {'family': family, 'target': protocol['families'].index(family), 'domain': row['domain'],
                'source_font_family': row['source_font_family'], 'view': 'native', 'split': 'train', 'native_font_verified': True}
            if row['source_font_family'] in trainer.NEW_SOURCES:values[trainer.SUPPLEMENT_MARKER] = True
            rows.append(values)
    counter = trainer.FullSupplementCounts(protocol['families'])
    counter.update(rows, [row['family'] not in (*trainer.core.CORE_FAMILIES, '__unknown__') for row in rows])
    counts = counter.report()
    for key in ('steps', 'rows', 'eligible_rows', 'native_core_weighted_rows', 'supplement_rows', 'original_rows'):
        counts[key] *= 3000
    for key in ('eligible_rows_per_step', 'native_core_weighted_rows_per_step'):counts[key] *= 3000
    for key in ('family_rows', 'eligible_family_rows', 'domain_rows', 'eligible_domain_rows',
                'native_core_weighted_family_rows', 'supplement_source_rows'):
        counts[key] = {name: value*3000 for name, value in counts[key].items()}
    for key in ('source_target_rows', 'native_core_source_rows'):
        for row in counts[key]:
            row['rows'] *= 3000
            if 'eligible_rows' in row:row['eligible_rows'] *= 3000
    sampling = trainer.read(run/'SAMPLING.json')
    for key in ('slots', 'family_rows', 'domain_rows', 'view_rows', 'supplement_source_rows', 'supplement_view_rows'):
        sampling[key] = {name: value*3//2 for name, value in sampling[key].items()}
    sampling['base'] = {'known_rows': 144000, 'unknown_rows': 24000}
    sampling['source_rows'] = [{key: row[key] for key in ('domain', 'source_font_family', 'target_family', 'rows')}
                               for row in counts['source_target_rows']]
    module.dump(run/'SAMPLING.json', sampling); module.dump(run/'TRAINING_COUNTS.json', counts)
    selected = history[0]
    for key, name in [('outputs', 'CALIBRATION_OUTPUTS.npz'), ('decisions', 'CALIBRATION_DECISIONS.json')]:
        (run/name).write_bytes((run/selected['artifacts'][key]['path']).read_bytes())
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection = {**deepcopy(protocol), 'schema': 'flux-glyph-unified-retention-soft-unknown-selection-v1',
        'selected': selected, 'history': history, 'promotion_allowed': True, 'passed': False, 'calibration_passed': False,
        'state_after_sha256': 'e'*64, 'selected_parameter_groups_sha256': dict.fromkeys(module.GROUPS, 'a'*64),
        'final_parameter_groups_sha256': dict.fromkeys(module.GROUPS, 'b'*64),
        'training_protocol_sha256': module.sha(run/'TRAINING_FREEZE.json'),
        'sampling_sha256': module.sha(run/'SAMPLING.json'), 'training_counts_sha256': module.sha(run/'TRAINING_COUNTS.json'),
        'calibration_outputs_sha256': module.sha(run/'CALIBRATION_OUTPUTS.npz'),
        'calibration_decisions_sha256': module.sha(run/'CALIBRATION_DECISIONS.json')}
    module.dump(run/'SELECTION.json', selection)
    return run, data, selection, protocol, cal, parent


def test_full_trainable_font_and_size_contract_with_cached_teacher_is_promotable(monkeypatch, tmp_path):
    run, data, selection, *_ = fixture_run(monkeypatch, tmp_path)
    assert module.validate(run, data) == selection
    assert selection['passed'] is False and selection['promotion_allowed'] is True
    assert not (data/'development_holdout').exists() and not (data/'test').exists()


@pytest.mark.parametrize('fault', ['frozen_base', 'frozen_size', 'residual', 'two_encoders', 'teacher_optimizer',
    'deployed_cache', 'feature_training', 'initial_state', 'cached_teacher_state', 'wrong_objective',
    'seed', 'steps', 'gate', 'incomplete_history', 'wrong_selected', 'selected_group_unchanged',
    'last_group_unchanged', 'missing_group', 'failed', 'size_cache', 'supplement_cache'])
def test_old_residual_claims_modified_objective_or_invalid_model_evidence_fail(monkeypatch, tmp_path, fault):
    run, data, selection, protocol, *_ = fixture_run(monkeypatch, tmp_path)
    changes = {'frozen_base': ('base_frozen', True), 'frozen_size': ('size_head_frozen', True),
        'residual': ('residual_head_trained', True), 'two_encoders': ('encoder_count', 2),
        'teacher_optimizer': ('teacher_optimizer_steps', 1), 'deployed_cache': ('teacher_cache_deployed', True),
        'feature_training': ('training_inputs', ['image_features_only']), 'initial_state': ('initial_state_sha256', '0'*64),
        'cached_teacher_state': ('cached_teacher_state_sha256', '0'*64), 'seed': ('seed', 99), 'steps': ('steps', 2000)}
    if fault in changes:
        key, value = changes[fault]; selection[key] = protocol[key] = value
    elif fault == 'wrong_objective':
        selection['objective']['unknown_ce_weight'] = protocol['objective']['unknown_ce_weight'] = 2.
    elif fault == 'gate':selection['fixed_runtime']['gates']['min_score'] = protocol['fixed_runtime']['gates']['min_score'] = .5
    elif fault == 'incomplete_history':selection['history'].pop()
    elif fault == 'wrong_selected':selection['selected'] = deepcopy(selection['history'][-1])
    elif fault in ('selected_group_unchanged', 'last_group_unchanged'):
        field = 'selected_parameter_groups_sha256' if fault == 'selected_group_unchanged' else 'final_parameter_groups_sha256'
        selection[field]['size_head'] = selection['initial_parameter_groups_sha256']['size_head']
    elif fault == 'missing_group':selection['selected_parameter_groups_sha256'].pop('trunk')
    elif fault == 'failed':selection['promotion_allowed'] = False
    elif fault == 'size_cache':(run/'cache/calibration/log_em_ratio.npy').write_bytes(b'Changed cached teacher size')
    else:monkeypatch.setattr(trainer, 'SUPPLEMENT_CACHE_SHA', '0'*64)
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection['training_protocol_sha256'] = module.sha(run/'TRAINING_FREEZE.json')
    module.dump(run/'SELECTION.json', selection)
    with pytest.raises(ValueError):module.validate(run, data)


@pytest.mark.parametrize('fault', ['old_unknown', 'new_quota', 'source_totals', 'wrong_kl', 'wrong_size_weight', 'core_teacher'])
def test_actual_training_counts_must_match_new_full_cnn_objective(monkeypatch, tmp_path, fault):
    run, data, selection, *_ = fixture_run(monkeypatch, tmp_path)
    sampling = trainer.read(run/'SAMPLING.json'); counts = trainer.read(run/'TRAINING_COUNTS.json')
    if fault == 'old_unknown':sampling['base']['unknown_rows'] = 48000
    elif fault == 'new_quota':counts['supplement_source_rows'][trainer.NEW_SOURCES[0]] -= 1
    elif fault == 'source_totals':sampling['source_rows'][0]['rows'] -= 1
    elif fault == 'wrong_kl':counts['objective']['teacher_kl_weight'] = 1.
    elif fault == 'wrong_size_weight':counts['objective']['size_smooth_l1_weight'] = 0.
    else:counts['eligible_family_rows']['PingFang'] = 1
    module.dump(run/'SAMPLING.json', sampling); module.dump(run/'TRAINING_COUNTS.json', counts)
    selection['sampling_sha256'] = module.sha(run/'SAMPLING.json')
    selection['training_counts_sha256'] = module.sha(run/'TRAINING_COUNTS.json')
    module.dump(run/'SELECTION.json', selection)
    with pytest.raises(ValueError):module.validate(run, data)


def test_metadata_exposes_the_actual_full_cnn_groups_and_no_deployed_teacher(monkeypatch, tmp_path):
    _, _, selection, _, cal, parent = fixture_run(monkeypatch, tmp_path)
    metadata = module.metadata_for_export(selection, cal, parent, '0'*64, '1'*64, '2'*64,
                                         module.strip_artifacts(selection['selected']))
    training = metadata['training']
    assert metadata['network_architecture'] == 'region-cnn64x256-unified-v1'
    for key in ('supplement_data', 'supplement_cache', 'training_data_counts', 'initial_parameter_groups_sha256',
                'selected_parameter_groups_sha256', 'final_parameter_groups_sha256', 'cached_teacher_state_sha256'):
        assert training[key] == selection[key]
    assert training['all_parameters_trained'] is training['teacher_logits_used'] is True
    assert training['base_frozen'] is training['size_head_frozen'] is training['residual_head_trained'] is False
    assert training['teacher_cache_deployed'] is training['teacher_in_deployed_model'] is False
    assert training['trainable_parameter_groups'] == list(module.GROUPS) and training['frozen_parameter_groups'] == []
    assert training['model_count'] == training['encoder_count'] == 1
    assert metadata['stable_validation_passed'] is metadata['test_passed'] is False
    assert release_metadata(metadata)['validation'] == metadata['validation']
    assert len(json.dumps(metadata).encode()) < 65536


def test_selected_checkpoint_requires_all_four_changed_groups_and_no_residual(monkeypatch):
    monkeypatch.setattr(module, 'state_sha', lambda state: json.dumps(state, sort_keys=True))
    state = {name+'.weight': 'trained' for name in module.GROUPS}
    selection = {'families': [str(i) for i in range(25)], 'state_after_sha256': module.state_sha(state),
        'initial_state_sha256': 'initial', 'initial_parameter_groups_sha256': dict.fromkeys(module.GROUPS, 'initial'),
        'selected_parameter_groups_sha256': {name: module.state_sha({name+'.weight': 'trained'}) for name in module.GROUPS}}
    checkpoint = {'state_dict': state, 'families': selection['families'], 'architecture': module.ARCHITECTURE, 'selection_sha256': 'selected'}
    module.validate_checkpoint(checkpoint, selection, 'selected')
    state['residual_family_head.0.weight'] = 'unexpected'; selection['state_after_sha256'] = module.state_sha(state)
    with pytest.raises(ValueError):module.validate_checkpoint(checkpoint, selection, 'selected')


@pytest.mark.parametrize('key,value', [('unknown_target_probability', 1.), ('named_target_probability', 0.),
    ('named_class_count', 25), ('known_label_smoothing', .1), ('labels_unchanged', False), ('inference_rule', True)])
def test_soft_unknown_target_cannot_be_replaced_or_used_as_an_inference_rule(monkeypatch, tmp_path, key, value):
    run, data, selection, protocol, *_ = fixture_run(monkeypatch, tmp_path)
    for document in (selection, protocol):
        document['soft_unknown_supervision'][key] = value
        document['objective']['soft_unknown_supervision'][key] = value
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection['training_protocol_sha256'] = module.sha(run/'TRAINING_FREEZE.json')
    module.dump(run/'SELECTION.json', selection)
    with pytest.raises(ValueError):module.validate(run, data)


@pytest.mark.parametrize('new_source', [False, True])
def test_neither_original_nor_new_unknown_may_receive_a_cached_teacher_target(monkeypatch, tmp_path, new_source):
    run, data, selection, *_ = fixture_run(monkeypatch, tmp_path)
    counts = trainer.read(run/'TRAINING_COUNTS.json')
    row = next(row for row in counts['source_target_rows'] if row['target_family'] == '__unknown__'
               and (row['source_font_family'] in trainer.NEW_SOURCES) == new_source)
    row['eligible_rows'] = 1
    counts['eligible_rows'] += 1; counts['eligible_rows_per_step'][0] += 1
    counts['eligible_family_rows']['__unknown__'] = 1
    counts['eligible_domain_rows'][row['domain']] = counts['eligible_domain_rows'].get(row['domain'], 0) + 1
    module.dump(run/'TRAINING_COUNTS.json', counts)
    selection['training_counts_sha256'] = module.sha(run/'TRAINING_COUNTS.json')
    module.dump(run/'SELECTION.json', selection)
    with pytest.raises(ValueError):module.validate(run, data)


def test_soft_distribution_and_unchanged_label_meaning_survive_metadata(monkeypatch, tmp_path):
    _, _, selection, _, cal, parent = fixture_run(monkeypatch, tmp_path)
    metadata = module.metadata_for_export(selection, cal, parent, '0'*64, '1'*64, '2'*64,
                                         module.strip_artifacts(selection['selected']))
    distribution = metadata['training']['soft_unknown_supervision']
    assert distribution == selection['soft_unknown_supervision'] == trainer.SOFT_UNKNOWN_SUPERVISION
    assert distribution is not selection['soft_unknown_supervision']
    assert distribution['unknown_target_probability'] == .5
    assert distribution['named_target_probability'] == .5/24 and distribution['named_class_count'] == 24
    assert distribution['labels_unchanged'] is True and distribution['inference_rule'] is False
    assert metadata['families'] == selection['families'] and metadata['families'][-1] == '__unknown__'
    assert metadata['gates'] == module.FIXED_RUNTIME['gates']
    assert metadata['training']['teacher_in_deployed_model'] is False
    assert metadata['validation']['kind'] == 'full_cnn_soft_unknown_supervision_calibration_only_at_export'
