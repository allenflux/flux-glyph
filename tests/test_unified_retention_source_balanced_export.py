"""Source-balanced full-CNN release contracts without training, export or holdout access."""
from copy import deepcopy
import json

import numpy as np
import pytest

import test_unified_retention_supplement_export as previous
from training import export_unified_retention_source_balanced as module
import train_unified_retention_source_balanced as trainer
from flux_glyph.model_download import release_metadata


def fixture_run(monkeypatch, tmp_path):
    run, data, old_selection, old_protocol, cal, parent = previous.fixture_run(monkeypatch, tmp_path)
    root = run.parent; monkeypatch.setattr(module, 'ROOT', root)
    for key in ('BASE_CHECKPOINT_SHA', 'BASE_SELECTION_SHA', 'CACHE_MANIFEST_SHA', 'SUPPLEMENT_MANIFEST_SHA', 'SUPPLEMENT_CACHE_SHA'):
        monkeypatch.setattr(trainer, key, getattr(previous.trainer, key))
    protocol = deepcopy(old_protocol)
    for key in ('initial_residual', 'residual_state_before_sha256', 'base_optimizer_steps', 'residual_optimizer_steps'):
        protocol.pop(key, None)
    for name in ('training/train_unified_retention_source_balanced.py', 'training/evaluate_unified_retention_source_balanced.py',
                 'training/retention_source_balanced_sampler.py'):
        path = root/name; path.write_text('Frozen full-CNN source '+name); protocol['bindings'][str(path)] = module.sha(path)
    source_order = sorted([*(f'Old TRAIN Negative {i}' for i in range(9)), *trainer.NEW_SOURCES])
    # The legacy cache fixture has three rows. Source derivation is tested separately
    # against real bound JSON below; this fixture isolates the full release contract.
    expected_order = tuple(source_order)
    monkeypatch.setattr(module, 'source_order_from_train', lambda data, selection: list(expected_order))
    groups = {name: str(i+1)*64 for i, name in enumerate(module.GROUPS)}
    state = protocol['base_state_sha256']
    protocol.update(schema='flux-glyph-unified-retention-source-balanced-protocol-v1', architecture=module.ARCHITECTURE,
        objective_variant=trainer.OBJECTIVE_VARIANT, objective=deepcopy(trainer.OBJECTIVE),
        unknown_binary_supervision=deepcopy(trainer.UNKNOWN_BINARY_SUPERVISION),
        unknown_source_order=source_order, design_changes=deepcopy(trainer.DESIGN_CHANGES),
        single_change_causal_attribution=False,
        sampling=deepcopy(trainer.SAMPLING), steps=3000, eval_every=500, seed=2026091410,
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
    counter = trainer.core.DistillationCounter(protocol['families'])
    counter.update(rows, [row['family'] not in (*trainer.core.CORE_FAMILIES, '__unknown__') for row in rows])
    counts = counter.report()
    for key in ('steps', 'rows', 'eligible_rows', 'native_core_weighted_rows'):
        counts[key] *= 3000
    for key in ('eligible_rows_per_step', 'native_core_weighted_rows_per_step'):counts[key] *= 3000
    for key in ('family_rows', 'eligible_family_rows', 'domain_rows', 'eligible_domain_rows',
                'native_core_weighted_family_rows'):
        counts[key] = {name: value*3000 for name, value in counts[key].items()}
    for key in ('source_target_rows', 'native_core_source_rows'):
        for row in counts[key]:
            row['rows'] *= 3000
            if 'eligible_rows' in row:row['eligible_rows'] *= 3000
    expected = trainer.expected_unknown_counts(source_order, 48000)
    extra = {source: expected[source] for source in trainer.NEW_SOURCES}
    new_total = sum(extra.values())
    counts['source_target_rows'] = [row for row in counts['source_target_rows'] if row['target_family'] != '__unknown__']
    counts['source_target_rows'].extend({'domain': 'android', 'source_font_family': source,
        'target_family': '__unknown__', 'rows': number, 'eligible_rows': 0} for source, number in expected.items())
    counts.update(schema='flux-glyph-retention-source-balanced-counts-v1',
        objective_variant=trainer.OBJECTIVE_VARIANT, objective=deepcopy(trainer.OBJECTIVE),
        unknown_binary_supervision=deepcopy(trainer.UNKNOWN_BINARY_SUPERVISION),
        mask_rule=trainer.OBJECTIVE['teacher_mask'], unknown_source_order=source_order,
        unknown_rows=48000, unknown_source_rows=expected, supplement_rows=new_total,
        supplement_source_rows=extra, original_rows=288000-new_total, teacher_logits_used=True,
        second_model_resident=False, teacher_optimizer_steps=0, teacher_cache_deployed=False)
    sampling = trainer.read(run/'SAMPLING.json')
    for key in ('family_rows', 'domain_rows', 'view_rows'):
        sampling[key] = {name: value*3//2 for name, value in sampling[key].items()}
    sampling.update(schema='flux-glyph-retention-source-balanced-sampling-v1', source_balanced=True,
        unknown_source_order=source_order, unknown_source_rows=expected, unknown_rows=48000,
        supplement_source_rows=extra, supplement_view_rows={'native': new_total},
        slots={'base_known': 144000, 'base_unknown': 48000-new_total, 'supplement_unknown': new_total,
            'ios_native_pingfang': 48000, 'ios_native_sfpro_helvetica': 24000, 'ios_native_original_eight': 24000},
        base={'known_rows': 144000, 'unknown_rows': 48000-new_total})
    sampling['source_rows'] = [{key: row[key] for key in ('domain', 'source_font_family', 'target_family', 'rows')}
                               for row in counts['source_target_rows']]
    module.dump(run/'SAMPLING.json', sampling); module.dump(run/'TRAINING_COUNTS.json', counts)
    selected = history[0]
    for key, name in [('outputs', 'CALIBRATION_OUTPUTS.npz'), ('decisions', 'CALIBRATION_DECISIONS.json')]:
        (run/name).write_bytes((run/selected['artifacts'][key]['path']).read_bytes())
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection = {**deepcopy(protocol), 'schema': 'flux-glyph-unified-retention-source-balanced-selection-v1',
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


@pytest.mark.parametrize('key,value', [('unknown_probability_target', 1.), ('named_conditional_distribution_target', .5/24),
    ('log_odds', 'z_unknown'), ('known_label_smoothing', .1), ('labels_unchanged', False), ('inference_rule', True)])
def test_binary_unknown_target_cannot_be_replaced_or_used_as_an_inference_rule(monkeypatch, tmp_path, key, value):
    run, data, selection, protocol, *_ = fixture_run(monkeypatch, tmp_path)
    for document in (selection, protocol):
        document['unknown_binary_supervision'][key] = value
        document['objective']['unknown_binary_supervision'][key] = value
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


def test_binary_supervision_source_cycle_and_unchanged_labels_survive_metadata(monkeypatch, tmp_path):
    _, _, selection, _, cal, parent = fixture_run(monkeypatch, tmp_path)
    metadata = module.metadata_for_export(selection, cal, parent, '0'*64, '1'*64, '2'*64,
                                         module.strip_artifacts(selection['selected']))
    distribution = metadata['training']['unknown_binary_supervision']
    assert distribution == selection['unknown_binary_supervision'] == trainer.UNKNOWN_BINARY_SUPERVISION
    assert distribution is not selection['unknown_binary_supervision']
    assert distribution['unknown_probability_target'] == .5
    assert distribution['named_conditional_distribution_target'] is None
    assert distribution['log_odds'] == 'z_unknown - logsumexp(z_named)'
    assert 'soft_unknown_supervision' not in metadata['training']
    for key in ('sampling', 'unknown_source_order', 'design_changes', 'single_change_causal_attribution'):
        assert metadata['training'][key] == selection[key]
    assert metadata['training']['single_change_causal_attribution'] is False
    assert metadata['training']['unknown_source_order'] is not selection['unknown_source_order']
    assert distribution['labels_unchanged'] is True and distribution['inference_rule'] is False
    assert metadata['families'] == selection['families'] and metadata['families'][-1] == '__unknown__'
    assert metadata['gates'] == module.FIXED_RUNTIME['gates']
    assert metadata['training']['teacher_in_deployed_model'] is False
    assert metadata['validation']['kind'] == 'full_cnn_source_balanced_binary_unknown_calibration_only_at_export'


def source_metadata(tmp_path):
    """Bound TRAIN JSON only; no pixel, CAL, development or TEST files exist."""
    folder = tmp_path/'data/train'; folder.mkdir(parents=True)
    families = list(previous.previous.FAMILIES)
    sources = [f'Original negative {i}' for i in range(9)]
    rows = [{'split': 'train', 'family': '__unknown__', 'target': families.index('__unknown__'),
             'source_font_family': source, 'domain': 'android'} for source in sources]
    # Duplicate views must not introduce extra source identities.
    rows.append(deepcopy(rows[0]))
    module.dump(folder/'rows.json', rows)
    part = {'split': 'train', 'families': families,
        'metadata': {'path': 'rows.json', 'sha256': module.sha(folder/'rows.json')}}
    module.dump(folder/'MANIFEST.json', part)
    selection = {'families': families, 'bindings': {str(folder/name): module.sha(folder/name)
                  for name in ('MANIFEST.json', 'rows.json')}}
    return folder.parent, selection, rows, sorted([*sources, *trainer.NEW_SOURCES])


def test_source_order_comes_from_bound_train_rows_without_reading_pixels(tmp_path):
    data, selection, _, expected = source_metadata(tmp_path)
    assert module.source_order_from_train(data, selection) == expected
    assert sorted(path.name for path in (data/'train').iterdir()) == ['MANIFEST.json', 'rows.json']
    assert not (data/'calibration').exists() and not (data/'test').exists()


@pytest.mark.parametrize('fault', ['unbound_rows', 'unbound_partition', 'modified_rows', 'wrong_split',
    'wrong_target', 'missing_source', 'overlap_new_source', 'only_eight', 'path_escape'])
def test_unknown_cycle_requires_exact_separate_original_train_sources(tmp_path, fault):
    data, selection, rows, _ = source_metadata(tmp_path)
    part_path = data/'train/MANIFEST.json'; rows_path = data/'train/rows.json'
    part = trainer.read(part_path)
    if fault == 'unbound_rows':selection['bindings'].pop(str(rows_path))
    elif fault == 'unbound_partition':selection['bindings'].pop(str(part_path))
    elif fault == 'modified_rows':rows_path.write_text('[]')
    else:
        if fault == 'wrong_split':rows[0]['split'] = 'calibration'
        elif fault == 'wrong_target':rows[0]['target'] = 0
        elif fault == 'missing_source':rows[0].pop('source_font_family')
        elif fault == 'overlap_new_source':rows[-2]['source_font_family'] = trainer.NEW_SOURCES[0]
        elif fault == 'only_eight':rows.pop(-2)
        elif fault == 'path_escape':part['metadata']['path'] = '../rows.json'; rows_path = data/'rows.json'
        module.dump(rows_path, rows)
        part['metadata']['sha256'] = module.sha(rows_path)
        module.dump(part_path, part)
        selection['bindings'][str(rows_path)] = module.sha(rows_path)
        selection['bindings'][str(part_path)] = module.sha(part_path)
    with pytest.raises(ValueError):module.source_order_from_train(data, selection)


def test_counts_use_continuous_cycle_remainder_and_dynamic_original_new_totals(monkeypatch, tmp_path):
    run, data, selection, *_ = fixture_run(monkeypatch, tmp_path)
    module.validate_counts(run, selection, data)
    sampling = trainer.read(run/'SAMPLING.json')
    order = selection['unknown_source_order']
    assert [sampling['unknown_source_rows'][source] for source in order] == [4364]*7 + [4363]*4
    assert sampling['supplement_source_rows'] == dict.fromkeys(trainer.NEW_SOURCES, 4363)
    assert sampling['slots']['supplement_unknown'] == 8726
    assert sampling['slots']['base_unknown'] == 39274
    assert sum(sampling['slots'].values()) == 288000


@pytest.mark.parametrize('fault', ['renamed_source', 'reset_each_batch', 'fixed_new_quota', 'duplicate_unknown_slot',
    'proposal_count', 'wrong_unknown_total', 'not_source_balanced', 'negative_views', 'extra_view',
    'sampling_source_order', 'changed_known_focus', 'single_change_claim'])
def test_balanced_source_and_actual_population_contract_rejects_drift(monkeypatch, tmp_path, fault):
    run, data, selection, protocol, *_ = fixture_run(monkeypatch, tmp_path)
    sampling = trainer.read(run/'SAMPLING.json'); counts = trainer.read(run/'TRAINING_COUNTS.json')
    if fault == 'renamed_source':
        # A internally consistent claimed cycle must still equal the bound TRAIN sources.
        for doc in (selection, protocol, sampling, counts):
            doc['unknown_source_order'][0] = 'Absent source'
        for doc in (sampling, counts):doc['unknown_source_rows']['Absent source'] = doc['unknown_source_rows'].pop('Old TRAIN Negative 0')
        for doc, key in ((sampling, 'source_rows'), (counts, 'source_target_rows')):
            for row in doc[key]:
                if row['source_font_family'] == 'Old TRAIN Negative 0':row['source_font_family'] = 'Absent source'
    elif fault == 'reset_each_batch':sampling['unknown_source_rows'][selection['unknown_source_order'][0]] = 6000
    elif fault == 'fixed_new_quota':sampling['supplement_source_rows'] = dict.fromkeys(trainer.NEW_SOURCES, 12000)
    elif fault == 'duplicate_unknown_slot':sampling['slots']['unknown'] = 48000
    elif fault == 'proposal_count':sampling['proposal_rows_discarded'] = 48000
    elif fault == 'wrong_unknown_total':sampling['unknown_rows'] = 48001
    elif fault == 'not_source_balanced':sampling['source_balanced'] = False
    elif fault == 'negative_views':sampling['supplement_view_rows']['half'] = -1; sampling['supplement_view_rows']['native'] += 1
    elif fault == 'extra_view':sampling['view_rows']['invented'] = 0
    elif fault == 'sampling_source_order':sampling['unknown_source_order'].reverse()
    elif fault == 'changed_known_focus':
        for doc in (sampling, counts):doc['family_rows']['PingFang'] -= 1; doc['family_rows']['SF Pro'] += 1
    else:
        selection['single_change_causal_attribution'] = protocol['single_change_causal_attribution'] = True
    module.dump(run/'SAMPLING.json', sampling); module.dump(run/'TRAINING_COUNTS.json', counts)
    selection['sampling_sha256'] = module.sha(run/'SAMPLING.json')
    selection['training_counts_sha256'] = module.sha(run/'TRAINING_COUNTS.json')
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection['training_protocol_sha256'] = module.sha(run/'TRAINING_FREEZE.json')
    module.dump(run/'SELECTION.json', selection)
    with pytest.raises(ValueError):module.validate(run, data)


@pytest.mark.parametrize('where', ['selection', 'protocol', 'counts', 'sampling'])
def test_obsolete_uniform_named_target_metadata_is_rejected(monkeypatch, tmp_path, where):
    run, data, selection, protocol, *_ = fixture_run(monkeypatch, tmp_path)
    sampling = trainer.read(run/'SAMPLING.json'); counts = trainer.read(run/'TRAINING_COUNTS.json')
    document = {'selection': selection, 'protocol': protocol, 'counts': counts, 'sampling': sampling}[where]
    document['soft_unknown_supervision'] = {'unknown_target_probability': .5, 'named_target_probability': .5/24}
    if where == 'protocol':selection['soft_unknown_supervision'] = document['soft_unknown_supervision']
    module.dump(run/'SAMPLING.json', sampling); module.dump(run/'TRAINING_COUNTS.json', counts)
    selection['sampling_sha256'] = module.sha(run/'SAMPLING.json')
    selection['training_counts_sha256'] = module.sha(run/'TRAINING_COUNTS.json')
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection['training_protocol_sha256'] = module.sha(run/'TRAINING_FREEZE.json')
    module.dump(run/'SELECTION.json', selection)
    with pytest.raises(ValueError):module.validate(run, data)
