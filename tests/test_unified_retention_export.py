"""Fixed-runtime retention export tests; synthetic arrays only, no model inference."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from training import export_unified_retention as module
from flux_glyph.model_download import release_metadata

FAMILIES = [f'Font {i}' for i in range(24)]+[module.UNKNOWN]


def sample_rows():
    return [{'tile_start': i, 'tile_count': 1, 'target': target, 'family': FAMILIES[target],
        'domain': domain, 'view': 'native', 'source_id': f'cal-{i}', 'region_id': f'row-{i}',
        'source_font_family': FAMILIES[target], 'ink_height_px': 40, 'font_size_px': 40,
        'split': 'calibration'} for i, (target, domain) in enumerate([(0, 'ios'), (1, 'android'), (24, 'android')])]


def sample_outputs():
    logits = np.zeros((3, 25), dtype=np.float32)
    logits[np.arange(3), [0, 1, 24]] = 8
    return logits, np.zeros(3, dtype=np.float32)


def test_status_and_rejection_reason_must_match_even_when_both_outputs_abstain():
    logits, ratios = sample_outputs()
    actual = module.region_outputs(logits, ratios, sample_rows(), 1.)
    assert module.compare_runtime_outputs(actual, copy.deepcopy(actual), sample_rows(), FAMILIES) == 0
    other = copy.deepcopy(actual)
    actual[0]['score'] = .69999999
    other[0]['score'] = .70000001
    actual[0]['margin'] = other[0]['margin'] = .005
    # Both are unconfirmed, but one fails score and the other fails separation.
    with pytest.raises(ValueError, match='rejection reason'):
        module.compare_runtime_outputs(actual, other, sample_rows(), FAMILIES)


@pytest.mark.parametrize('fault', ['name', 'unknown', 'size', 'numerical'])
def test_full_runtime_font_unknown_size_and_score_parity_is_required(fault):
    logits, ratios = sample_outputs()
    actual = module.region_outputs(logits, ratios, sample_rows(), 1.)
    other = copy.deepcopy(actual)
    if fault == 'name':other[0]['predicted'] = 1
    elif fault == 'unknown':other[0]['predicted'] = 24
    elif fault == 'size':other[0]['size_spread'] = .200000001
    else:other[0]['probabilities'][0] -= .01
    with pytest.raises((ValueError, AssertionError)):
        module.compare_runtime_outputs(actual, other, sample_rows(), FAMILIES)


@pytest.mark.parametrize('field,value', [('temperature', .5), ('temperature', True),
                                       ('min_score', .5), ('min_margin', 0.), ('min_patch_agreement', .5)])
def test_export_cannot_relax_any_fixed_runtime_gate(field, value):
    record = copy.deepcopy(module.retention.FIXED_RUNTIME)
    if field == 'temperature':record[field] = value
    else:record['gates'][field] = value
    with pytest.raises(ValueError, match='temperature or naming gates'):
        module.require_fixed_runtime(record)


def fixture_run(monkeypatch, tmp_path):
    root = (tmp_path/'repo').resolve(); run = root/'run'; data = root/'data'
    run.mkdir(parents=True)
    monkeypatch.setattr(module, 'ROOT', root)
    monkeypatch.setattr(module.retention, 'ROOT', root)
    names = ['training/train_unified_retention.py', 'training/train_unified_regions.py',
        'training/prepare_unified_regions.py', 'training/train_android_regions.py', 'training/train_regions.py',
        'training/region_network.py', 'training/network.py', 'training/evaluate_unified_retention.py',
        'src/flux_glyph/unified_font.py', 'src/flux_glyph/region_font.py']
    for name in names:
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
    module.dump(data/'MANIFEST.json', {'families': FAMILIES, 'font_label_groups': {}})
    rows = sample_rows()
    for split in ('train', 'calibration'):
        folder = data/split; folder.mkdir()
        module.dump(folder/'rows.json', [{**row, 'split': split} for row in rows])
        (folder/'tiles.raw').write_bytes(b'Not read as pixels by provenance validation')
        module.dump(folder/'MANIFEST.json', {'split': split, 'root_manifest_sha256': module.sha(data/'MANIFEST.json'),
            'families': FAMILIES, 'views': len(rows), 'rejected': [],
            'metadata': {'path': 'rows.json', 'sha256': module.sha(folder/'rows.json')},
            'array': {'path': 'tiles.raw', 'sha256': module.sha(folder/'tiles.raw')}})
    cal = module.calibration_metadata(data)
    plan = {'families': FAMILIES, 'masks': [],
            'constraints': [{'population': 'all', 'metric': 'correct_named', 'operator': 'ge', 'value': 2}],
            'bindings': {}}
    logits, ratios = sample_outputs()
    baseline, outputs, _ = module.retention.evaluate_outputs(logits, ratios, cal, plan, 0)
    groups = {name: 'a'*64 for name in module.GROUPS}
    final = {name: 'b'*64 for name in module.GROUPS}
    parent = root/'parent/model.pth'; parent.parent.mkdir(); parent.write_bytes(b'frozen parent checkpoint')
    parent_selection = {'schema': 'flux-glyph-unified-training-selection-v1', 'families': FAMILIES,
        'policy': module.POLICY, 'test_read': False, 'development_holdout_read': False,
        'data_manifest_sha256': module.sha(data/'MANIFEST.json'), 'final_parameter_groups_sha256': groups,
        'state_after_sha256': 'a'*64, 'optimizer_steps_executed': 6000,
        'selected': {**module.retention.FIXED_RUNTIME, 'step': 6000, 'metrics': baseline['metrics']}}
    np.savez(parent.parent/'CALIBRATION_OUTPUTS.npz', logits=logits, log_em_ratio=ratios)
    module.dump(parent.parent/'CALIBRATION_DECISIONS.json', {'families': FAMILIES, 'records': outputs})
    parent_selection.update(calibration_outputs_sha256=module.sha(parent.parent/'CALIBRATION_OUTPUTS.npz'),
        calibration_decisions_sha256=module.sha(parent.parent/'CALIBRATION_DECISIONS.json'))
    module.dump(parent.parent/'SELECTION.json', parent_selection)
    module.dump(parent.parent/'DEVELOPMENT_REGRESSION.json', {'not_a_blind_test': True})
    parent_model = root/'release/model.onnx'; parent_model.parent.mkdir(); parent_model.write_bytes(b'parent ONNX hash fixture')
    parent_meta = parent_model.parent/'metadata.json'
    metadata = {'schema': 'flux-glyph-unified-region-font-v1', 'font_mode': 'unified', 'families': FAMILIES,
        **module.retention.FIXED_RUNTIME, 'font_sources': {'Font 0': ['system']},
        'model': {'path': 'model.onnx', 'sha256': module.sha(parent_model)},
        'training': {'selection_sha256': module.sha(parent.parent/'SELECTION.json'), 'checkpoint_sha256': module.sha(parent)}}
    module.dump(parent_meta, metadata)
    module.dump(parent.parent/'PARITY.json', {'passed': True, 'model_sha256': module.sha(parent_model),
        'checkpoint_sha256': module.sha(parent), 'selection_sha256': module.sha(parent.parent/'SELECTION.json')})
    plan.update(parent_checkpoint={'path': 'parent/model.pth', 'sha256': module.sha(parent)},
        parent_selection_sha256=module.sha(parent.parent/'SELECTION.json'),
        parent_metadata={'path': 'release/metadata.json', 'sha256': module.sha(parent_meta)})
    plan_path = root/'RETENTION_PLAN.json'; module.dump(plan_path, plan)
    def read_plan(path, received_data, checkpoint):
        assert Path(path).resolve() == plan_path and Path(received_data).resolve() == data
        assert Path(checkpoint).resolve() == parent
        return json.loads(plan_path.read_text())
    monkeypatch.setattr(module.retention, 'read_plan', read_plan)
    inheritance = {'all_family_rows_inherited': True, 'all_parameters_inherited': True, 'all_parameters_trainable': True,
        'family_count': 25, 'new_random_output_rows': 0, 'source_state_sha256': 'a'*64,
        'checkpoint': {'path': str(parent), 'sha256': module.sha(parent)},
        'selection_sha256': module.sha(parent.parent/'SELECTION.json'), 'source_selected_step': 6000,
        'source_optimizer_steps_executed': 6000}
    module.dump(run/'BASELINE.json', baseline)
    for name in ('CALIBRATION_OUTPUTS.npz', 'CALIBRATION_DECISIONS.json'):
        (run/('BASELINE_'+name)).write_bytes((parent.parent/name).read_bytes())
    baseline_bindings = {name: module.sha(run/name) for name in
        ('BASELINE.json', 'BASELINE_CALIBRATION_OUTPUTS.npz', 'BASELINE_CALIBRATION_DECISIONS.json')}
    paths = [root/name for name in names]+[data/'MANIFEST.json', plan_path, parent, parent_meta, parent_model]
    paths += list(parent.parent.glob('*.json'))+[parent.parent/'CALIBRATION_OUTPUTS.npz']
    for split in ('train', 'calibration'):
        paths += [data/split/name for name in ('MANIFEST.json', 'rows.json', 'tiles.raw')]
    bindings = {str(path): module.sha(path) for path in paths}
    shared = {'architecture': module.ARCHITECTURE, 'policy': module.POLICY,
        'fixed_runtime': module.retention.FIXED_RUNTIME, 'families': FAMILIES, 'bindings': bindings,
        'parent_checkpoint': plan['parent_checkpoint'], 'parent_selection_sha256': plan['parent_selection_sha256'],
        'parent_metadata': plan['parent_metadata'], 'retention_plan': {'path': str(plan_path), 'sha256': module.sha(plan_path)},
        'initializer_evidence': inheritance, 'baseline_bindings': baseline_bindings,
        'test_read': False, 'development_holdout_read': False, 'model_count': 1,
        'platform_routing': False, 'score_merging': False, 'runtime_gates_searched': False}
    protocol = {**shared, 'schema': 'flux-glyph-unified-retention-training-protocol-v1', 'steps': module.retention.STEPS,
        'eval_every': module.retention.EVAL_EVERY, 'batch_size': module.retention.BATCH_SIZE,
        'learning_rate': module.retention.LEARNING_RATE, 'minimum_learning_rate': module.retention.MINIMUM_LEARNING_RATE,
        'sampling': module.retention.SAMPLING, 'objective': module.retention.OBJECTIVE,
        'training_inputs': ['image_tiles'], 'device': 'mps', 'initial_state_sha256': 'a'*64,
        'initial_parameter_groups_sha256': groups}
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    history = []
    for step in range(500, 3001, 500):
        record, outputs, _ = module.retention.evaluate_outputs(logits, ratios, cal, plan, step)
        folder = run/'checkpoints'/f'step{step:05d}'; folder.mkdir(parents=True)
        (folder/'model.pth').write_bytes(f'step {step} checkpoint fixture'.encode())
        np.savez(folder/'CALIBRATION_OUTPUTS.npz', logits=logits, log_em_ratio=ratios)
        module.dump(folder/'CALIBRATION_DECISIONS.json', {'families': FAMILIES, 'records': outputs})
        module.dump(folder/'METRICS.json', record)
        record['artifacts'] = {key: {'path': str((folder/name).relative_to(run)), 'sha256': module.sha(folder/name)}
            for key, name in [('checkpoint', 'model.pth'), ('outputs', 'CALIBRATION_OUTPUTS.npz'),
                              ('decisions', 'CALIBRATION_DECISIONS.json'), ('metrics', 'METRICS.json')]}
        history.append(record)
    selected = copy.deepcopy(history[0])
    for key, name in [('outputs', 'CALIBRATION_OUTPUTS.npz'), ('decisions', 'CALIBRATION_DECISIONS.json')]:
        (run/name).write_bytes((run/selected['artifacts'][key]['path']).read_bytes())
    count = module.retention.STEPS
    sampled = {'slots': {key: module.retention.SAMPLING[key]*count for key in
        ('base_known', 'base_unknown', 'ios_native_pingfang', 'ios_native_sfpro_helvetica', 'ios_native_original_eight')},
        'base': {'known_rows': 48*count, 'unknown_rows': 16*count},
        **{key: {'synthetic': 96*count} for key in ('family_rows', 'domain_rows', 'view_rows')}}
    module.dump(run/'SAMPLING.json', sampled)
    selection = {**shared, 'schema': 'flux-glyph-unified-retention-selection-v1',
        'selected': selected, 'history': history, 'passed': False, 'calibration_passed': False, 'promotion_allowed': True,
        'data_manifest_sha256': module.sha(data/'MANIFEST.json'),
        'training_protocol_sha256': module.sha(run/'TRAINING_FREEZE.json'), 'optimizer_steps_executed': 3000,
        'training_device': 'mps', 'state_before_sha256': 'a'*64, 'state_after_sha256': 'b'*64,
        'initial_parameter_groups_sha256': groups, 'final_parameter_groups_sha256': final,
        'calibration_outputs_sha256': module.sha(run/'CALIBRATION_OUTPUTS.npz'),
        'calibration_decisions_sha256': module.sha(run/'CALIBRATION_DECISIONS.json'),
        'sampling_sha256': module.sha(run/'SAMPLING.json')}
    module.dump(run/'SELECTION.json', selection)
    return run, data, selection, protocol, cal, plan, metadata


def test_promotable_retention_keeps_original_stable_failure_and_no_holdout_read(monkeypatch, tmp_path):
    run, data, selected, _, _, _, _ = fixture_run(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)  # Parent paths are relative to ROOT, never caller CWD.
    assert module.validate(run, data) == selected
    assert selected['passed'] is selected['calibration_passed'] is False
    assert not (data/'test').exists() and not (data/'development_holdout').exists()


@pytest.mark.parametrize('fault', ['promotion', 'stable', 'gate', 'search', 'test', 'scope', 'partial_steps',
                                  'unchanged_group', 'source_missing', 'tensor_binding_missing', 'checkpoint_sha',
                                  'cached_array', 'wrong_step_choice', 'history_missing', 'baseline', 'sampling'])
def test_provenance_mutations_and_forged_pass_are_rejected(monkeypatch, tmp_path, fault):
    run, data, selection, protocol, _, _, _ = fixture_run(monkeypatch, tmp_path)
    if fault == 'promotion':selection['promotion_allowed'] = False
    elif fault == 'stable':selection['passed'] = selection['calibration_passed'] = True
    elif fault == 'gate':selection['fixed_runtime'] = {**module.retention.FIXED_RUNTIME, 'temperature': .5}
    elif fault == 'search':selection['runtime_gates_searched'] = True
    elif fault == 'test':selection['test_read'] = True
    elif fault == 'scope':selection['model_count'] = 2
    elif fault == 'partial_steps':protocol['steps'] = selection['optimizer_steps_executed'] = 500
    elif fault == 'unchanged_group':selection['final_parameter_groups_sha256']['size_head'] = 'a'*64
    elif fault in ('source_missing', 'tensor_binding_missing'):
        key = str(module.ROOT/'training/train_regions.py') if fault == 'source_missing' else str(data/'train/tiles.raw')
        selection['bindings'].pop(key)
    elif fault == 'checkpoint_sha':selection['history'][0]['artifacts']['checkpoint']['sha256'] = '0'*64
    elif fault == 'cached_array':(run/'checkpoints/step03000/CALIBRATION_OUTPUTS.npz').write_bytes(b'changed')
    elif fault == 'wrong_step_choice':selection['selected'] = copy.deepcopy(selection['history'][-1])
    elif fault == 'history_missing':selection['history'].pop()
    elif fault == 'baseline':(run/'BASELINE.json').write_text('{}')
    elif fault == 'sampling':selection['sampling_sha256'] = '0'*64
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection['training_protocol_sha256'] = module.sha(run/'TRAINING_FREEZE.json')
    module.dump(run/'SELECTION.json', selection)
    with pytest.raises(ValueError):module.validate(run, data)


def test_same_bytes_at_another_data_root_are_not_an_authorized_input(monkeypatch, tmp_path):
    import shutil
    run, data, *_ = fixture_run(monkeypatch, tmp_path)
    copy_root = tmp_path/'copied-data'; shutil.copytree(data, copy_root)
    with pytest.raises(ValueError):module.validate(run, copy_root)


def test_cached_metrics_are_recomputed_not_accepted_from_the_passed_field(monkeypatch, tmp_path):
    run, data, selection, _, _, _, _ = fixture_run(monkeypatch, tmp_path)
    record = selection['history'][-1]
    record['retention_checks'][0]['value'] = 0
    record['retention_checks'][0]['actual'] = 100000
    path = run/record['artifacts']['metrics']['path']
    module.dump(path, module.strip_artifacts(record))
    record['artifacts']['metrics']['sha256'] = module.sha(path)
    module.dump(run/'SELECTION.json', selection)
    with pytest.raises(ValueError, match='reproduce'):
        module.validate(run, data)


def test_public_metadata_compacts_populations_and_never_claims_stable_acceptance(monkeypatch, tmp_path):
    _, _, selection, _, cal, _, parent_meta = fixture_run(monkeypatch, tmp_path)
    record = module.strip_artifacts(selection['selected'])
    record['retention_populations'] = {f'population:{i}': copy.deepcopy(record['metrics']) for i in range(40)}
    result = module.metadata_for_export(selection, cal, parent_meta, '0'*64, '1'*64, '2'*64, record)
    assert result['font_sources'] == parent_meta['font_sources']
    assert result['validation']['retention_promotion_allowed'] is True
    assert result['stable_validation_passed'] is result['test_passed'] is False
    assert result['validation']['calibration_passed'] is False
    for population in result['validation']['retention_populations'].values():
        assert 'per_domain' not in population and 'per_family' not in population and 'unknown_sources' not in population
    assert release_metadata(result)['validation'] == result['validation']
    assert len(json.dumps(result['validation']).encode()) < 65536
    record['promotion_allowed'] = False
    with pytest.raises(ValueError):module.metadata_for_export(selection, cal, parent_meta, '0'*64, '1'*64, '2'*64, record)


def test_checkpoint_family_state_and_every_parameter_group_are_bound(monkeypatch):
    monkeypatch.setattr(module, 'state_sha', lambda state: 'whole' if len(state) == 4 else next(iter(state.values())))
    state = {name+'.weight': name for name in module.GROUPS}
    selection = {'families': FAMILIES, 'state_after_sha256': 'whole',
                 'final_parameter_groups_sha256': {name: name for name in module.GROUPS}}
    checkpoint = {'families': FAMILIES, 'architecture': module.ARCHITECTURE,
                  'selection_sha256': 'selection', 'state_dict': state}
    module.validate_checkpoint(checkpoint, selection, 'selection')
    checkpoint['state_dict']['size_head.weight'] = 'changed'
    with pytest.raises(ValueError, match='parameter group'):
        module.validate_checkpoint(checkpoint, selection, 'selection')


def fixture_weighted_run(monkeypatch, tmp_path):
    import sys
    import train_unified_retention_balanced as balanced
    families = list(module.retention.IOS_ANCHOR_FAMILIES)+[f'Extra {i}' for i in range(16)]+[module.UNKNOWN]
    monkeypatch.setattr(sys.modules[__name__], 'FAMILIES', families)
    run, data, selection, protocol, cal, plan, parent_meta = fixture_run(monkeypatch, tmp_path)
    source = module.ROOT/'training/train_unified_retention_balanced.py'
    source.write_text('synthetic bound source; the pure contract uses the real imported implementation')
    selection['bindings'][str(source)] = module.sha(source)
    for document in (selection, protocol):
        document.update(objective_variant=module.WEIGHTED,
            objective=balanced.weighted_objective(families),
            class_prior_weighting=balanced.class_prior_weighting(families))
    sampled = json.loads((run/'SAMPLING.json').read_text())
    sampled['family_rows'] = {family: count*module.retention.STEPS
        for family, count in selection['class_prior_weighting']['sampled_counts'].items()}
    module.dump(run/'SAMPLING.json', sampled)
    selection['sampling_sha256'] = module.sha(run/'SAMPLING.json')
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection['training_protocol_sha256'] = module.sha(run/'TRAINING_FREEZE.json')
    module.dump(run/'SELECTION.json', selection)
    return run, data, selection, protocol, cal, plan, parent_meta


def test_weighted_training_has_one_explicit_objective_and_keeps_the_same_runtime(monkeypatch, tmp_path):
    run, data, selection, protocol, cal, _, parent_meta = fixture_weighted_run(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    assert module.validate(run, data) == selection
    contract = module.objective_contract(selection, protocol)
    weighting = contract['class_prior_weighting']
    assert weighting['sampled_counts']['PingFang'] == 19
    assert weighting['target_expected_counts']['PingFang'] == 77/24
    assert weighting['target_expected_counts'][module.UNKNOWN] == 19
    assert sum(weighting['sampled_counts'][f]*weighting['class_weights'][f] for f in FAMILIES) == pytest.approx(96)
    metadata = module.metadata_for_export(selection, cal, parent_meta, '0'*64, '1'*64, '2'*64,
                                         module.strip_artifacts(selection['selected']))
    assert metadata['training']['objective_variant'] == module.WEIGHTED
    assert metadata['training']['class_prior_loss_weighting'] is True
    assert metadata['training']['objective']['teacher_outputs'] is False
    assert metadata['training']['class_prior_weighting'] == weighting
    assert metadata['validation']['objective_variant'] == module.WEIGHTED
    assert metadata['validation']['class_prior_loss_weighting'] is True
    assert metadata['gates'] == module.retention.FIXED_RUNTIME['gates']
    assert metadata['stable_validation_passed'] is metadata['test_passed'] is False
    assert release_metadata(metadata)['validation'] == metadata['validation']


@pytest.mark.parametrize('fault', ['unknown_variant', 'selection_variant_missing', 'protocol_variant_missing',
    'variant_mismatch', 'objective_missing', 'cross_entropy', 'size_loss', 'teacher', 'target_prior',
    'class_weight', 'normalization', 'sampled_prior', 'source_binding_missing', 'actual_sampling',
    'promotion', 'runtime_gate'])
def test_weighted_objective_cannot_be_substituted_or_claimed_without_evidence(monkeypatch, tmp_path, fault):
    run, data, selection, protocol, *_ = fixture_weighted_run(monkeypatch, tmp_path)
    if fault == 'unknown_variant':selection['objective_variant'] = protocol['objective_variant'] = 'custom_relaxed_loss'
    elif fault == 'selection_variant_missing':selection.pop('objective_variant')
    elif fault == 'protocol_variant_missing':protocol.pop('objective_variant')
    elif fault == 'variant_mismatch':protocol['objective_variant'] = module.UNWEIGHTED
    elif fault == 'objective_missing':selection.pop('objective')
    elif fault in ('cross_entropy', 'size_loss', 'teacher'):
        key, value = {'cross_entropy': ('family_cross_entropy_label_smoothing', 0.),
            'size_loss': ('size_smooth_l1_weight', .1), 'teacher': ('teacher_outputs', True)}[fault]
        selection['objective'][key] = protocol['objective'][key] = value
    elif fault in ('target_prior', 'class_weight', 'normalization', 'sampled_prior'):
        for document in (selection, protocol):
            weighting = document['class_prior_weighting']
            if fault == 'target_prior':weighting['target_expected_counts']['PingFang'] = 19
            elif fault == 'class_weight':weighting['class_weights']['PingFang'] = 1.
            elif fault == 'normalization':weighting['normalization'] = 'Unnormalized sum'
            else:weighting['sampled_counts']['PingFang'] = 18
    elif fault == 'source_binding_missing':
        selection['bindings'].pop(str(module.ROOT/'training/train_unified_retention_balanced.py'))
    elif fault == 'actual_sampling':
        sampled = json.loads((run/'SAMPLING.json').read_text())
        sampled['family_rows']['PingFang'] -= 1
        sampled['family_rows']['SF Pro'] += 1
        module.dump(run/'SAMPLING.json', sampled)
        selection['sampling_sha256'] = module.sha(run/'SAMPLING.json')
    elif fault == 'promotion':selection['promotion_allowed'] = False
    elif fault == 'runtime_gate':
        selection['fixed_runtime'] = copy.deepcopy(selection['fixed_runtime'])
        selection['fixed_runtime']['gates']['min_score'] = .5
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection['training_protocol_sha256'] = module.sha(run/'TRAINING_FREEZE.json')
    module.dump(run/'SELECTION.json', selection)
    with pytest.raises(ValueError):module.validate(run, data)


def test_explicit_unweighted_variant_and_historical_omission_have_the_same_contract(monkeypatch, tmp_path):
    run, data, selection, protocol, *_ = fixture_run(monkeypatch, tmp_path)
    previous = module.objective_contract(selection, protocol)
    assert previous['variant'] == module.UNWEIGHTED and previous['class_prior_weighting'] is None
    selection['objective_variant'] = protocol['objective_variant'] = module.UNWEIGHTED
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection['training_protocol_sha256'] = module.sha(run/'TRAINING_FREEZE.json')
    module.dump(run/'SELECTION.json', selection)
    assert module.validate(run, data) == selection
    assert module.objective_contract(selection, protocol) == previous
    for document in (selection, protocol):document['class_prior_weighting'] = {}
    with pytest.raises(ValueError, match='Unweighted retention objective'):
        module.objective_contract(selection, protocol)
