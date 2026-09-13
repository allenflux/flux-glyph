"""Fixed mean release checks; synthetic tensors/JSON only, no source inference."""
from copy import deepcopy
import json

import numpy as np
import pytest

from training import export_unified_retention_source_average as module
import average_unified_retention_source_balanced as averaging
import test_unified_retention_source_balanced_export as previous
from flux_glyph.model_download import release_metadata


def save_evidence(run, selection, freeze, report):
    module.dump(run/'AVERAGING_FREEZE.json', freeze)
    selection['averaging_freeze_sha256'] = module.sha(run/'AVERAGING_FREEZE.json')
    module.dump(run/'SELECTION.json', selection)
    report['selection_sha256'] = module.sha(run/'SELECTION.json')
    report['averaging_freeze_sha256'] = selection['averaging_freeze_sha256']
    module.dump(run/'report.json', report)


def fixture_average(monkeypatch, tmp_path):
    # Source completion/rank validation is separately tested by the averaging
    # module. This fixture isolates the new candidate and release contracts.
    source_run, data, trained, _, cal, parent = previous.fixture_run(monkeypatch, tmp_path)
    core = averaging.read(trained['base_selection']['path'])
    core['objective'] = deepcopy(averaging.core.OBJECTIVE)
    plan = averaging.read(trained['retention_plan']['path'])
    plan['constraints'] = [deepcopy(plan['constraints'][0]) for _ in range(46)]
    plan_path = source_run.parent/'average-test-plan.json'; module.dump(plan_path, plan)
    trained['retention_plan'] = {'path': str(plan_path), 'sha256': module.sha(plan_path)}
    trained['bindings'][str(plan_path)] = module.sha(plan_path)
    monkeypatch.setattr(averaging, 'PLAN', plan_path)
    monkeypatch.setattr(averaging, 'PLAN_SHA', module.sha(plan_path))
    identities = []
    for index, selection in enumerate((core, trained)):
        folder = previous.module.Path(trained['base_checkpoint']['path']).parent if index == 0 else source_run
        identities.append({'role': averaging.METHOD['source_order'][index], 'run': str(folder),
            'checkpoint': {'path': str(folder/'model.pth'), 'sha256': str(index+1)*64},
            'selection': {'path': str(folder/'SELECTION.json'), 'sha256': str(index+3)*64},
            'training_freeze': {'path': str(folder/'TRAINING_FREEZE.json'), 'sha256': str(index+5)*64},
            'selected_step': (1000, 1500)[index], 'optimizer_steps_executed': (1500, 3000)[index],
            'state_sha256': selection['state_after_sha256'],
            'parameter_groups_sha256': dict.fromkeys(module.GROUPS, str(index+7)*64)})
    evidence = {key: deepcopy(trained[key]) for key in ('families', 'bindings', 'data_manifest_sha256', 'parent_metadata', 'parent_checkpoint')}
    evidence.update(plan=plan, source_checkpoints=identities, source_selections=[core, trained])
    monkeypatch.setattr(averaging, 'validate_sources', lambda data: deepcopy(evidence))
    freeze = {key: deepcopy(evidence[key]) for key in ('families', 'source_checkpoints', 'bindings', 'data_manifest_sha256', 'parent_metadata', 'parent_checkpoint')}
    freeze.update(schema='flux-glyph-unified-retention-source-averaging-freeze-v1', architecture=module.ARCHITECTURE,
        objective_variant=averaging.OBJECTIVE_VARIANT, method=deepcopy(averaging.METHOD),
        retention_plan=deepcopy(trained['retention_plan']),
        source_training_objectives=[deepcopy(item['objective']) for item in (core, trained)],
        fixed_runtime=deepcopy(module.FIXED_RUNTIME), runtime_gates_changed=False,
        state_sha256='f'*64, parameter_groups_sha256=dict.fromkeys(module.GROUPS, 'c'*64),
        optimizer_steps_executed=0, source_optimizer_steps_executed=3000, source_selected_step=1500,
        core_optimizer_steps_executed=1500, core_selected_step=1000, model_count=1, encoder_count=1,
        teacher_in_deployed_model=False, platform_routing=False, score_merging=False, calibration_device='mps',
        candidate_step=0, selection_search_performed=False, test_read=False, development_holdout_read=False,
        scope='Synthetic one fixed post-training average.')
    run = source_run.parent/'average'; run.mkdir()
    (run/'model.pth').write_bytes(b'Synthetic complete average checkpoint')
    with np.load(source_run/'CALIBRATION_OUTPUTS.npz') as saved:
        logits, sizes = saved['logits'].copy(), saved['log_em_ratio'].copy()
    record, outputs, _ = module.evaluate_outputs(logits, sizes, cal, plan, 0)
    np.savez(run/'CALIBRATION_OUTPUTS.npz', logits=logits, log_em_ratio=sizes)
    module.dump(run/'CALIBRATION_DECISIONS.json', {'families': trained['families'], 'records': outputs})
    module.dump(run/'RESULT.json', record)
    selection = {**deepcopy(freeze), 'schema': 'flux-glyph-unified-retention-source-average-selection-v1',
        'selected': record, 'history': [record], 'promotion_allowed': True,
        'passed': record['metrics']['passed'], 'calibration_passed': record['metrics']['passed'],
        'checkpoint_sha256': module.sha(run/'model.pth'), 'calibration_outputs_sha256': module.sha(run/'CALIBRATION_OUTPUTS.npz'),
        'calibration_decisions_sha256': module.sha(run/'CALIBRATION_DECISIONS.json'), 'result_sha256': module.sha(run/'RESULT.json')}
    report = {key: deepcopy(selection[key]) for key in ('objective_variant', 'method', 'optimizer_steps_executed',
        'source_optimizer_steps_executed', 'source_selected_step', 'core_optimizer_steps_executed', 'core_selected_step',
        'state_sha256', 'parameter_groups_sha256', 'promotion_allowed', 'calibration_passed', 'checkpoint_sha256',
        'calibration_outputs_sha256', 'calibration_decisions_sha256', 'result_sha256', 'fixed_runtime')}
    report.update(schema='flux-glyph-unified-retention-source-average-result-v1', status='PROMOTABLE_CANDIDATE',
        stable_validation_passed=False, test_passed=False, calibration_inference_completed=True,
        calibration_inference_passes=1, runtime_gates_changed=False, test_read=False, development_holdout_read=False,
        selected_step=0, retention_checks_passed=46, retention_checks_total=46)
    save_evidence(run, selection, freeze, report)
    return run, data, selection, freeze, report, cal, parent


def test_average_validates_one_cached_cal_record_without_source_training_or_holdout(monkeypatch, tmp_path):
    run, data, selection, *_ = fixture_average(monkeypatch, tmp_path)
    assert module.validate(run, data) == selection
    assert selection['optimizer_steps_executed'] == 0 and len(selection['history']) == 1
    assert selection['passed'] is False and selection['promotion_allowed'] is True
    assert not (data/'development_holdout').exists() and not (data/'test').exists()


@pytest.mark.parametrize('fault', ['weight_ratio', 'ratio_search', 'optimizer', 'source_steps', 'source_selected',
    'core_selected', 'class_order', 'source_groups', 'source_identity', 'candidate_group_missing',
    'same_as_source', 'two_encoders', 'gate', 'second_candidate', 'source_objective', 'result_bytes',
    'checkpoint_bytes', 'repeated_cal', 'test_claim', 'failed'])
def test_average_rejects_changed_arithmetic_sources_scope_or_cached_result(monkeypatch, tmp_path, fault):
    run, data, selection, freeze, report, *_ = fixture_average(monkeypatch, tmp_path)
    changes = {'optimizer': ('optimizer_steps_executed', 3000), 'source_steps': ('source_optimizer_steps_executed', 1500),
        'source_selected': ('source_selected_step', 3000), 'core_selected': ('core_selected_step', 1500),
        'two_encoders': ('encoder_count', 2)}
    if fault in changes:
        key, value = changes[fault]
        selection[key] = freeze[key] = value
    elif fault in ('weight_ratio', 'ratio_search'):
        key, value = ('weights', [.6, .4]) if fault == 'weight_ratio' else ('ratio_search', True)
        selection['method'][key] = freeze['method'][key] = value
    elif fault == 'class_order':selection['families'].reverse(); freeze['families'].reverse()
    elif fault == 'source_groups':
        for doc in (selection, freeze):doc['source_checkpoints'][1]['parameter_groups_sha256']['size_head'] = '0'*64
    elif fault == 'source_identity':
        for doc in (selection, freeze):doc['source_checkpoints'][1]['checkpoint']['sha256'] = '0'*64
    elif fault == 'candidate_group_missing':
        for doc in (selection, freeze):doc['parameter_groups_sha256'].pop('size_head')
    elif fault == 'same_as_source':selection['state_sha256'] = freeze['state_sha256'] = selection['source_checkpoints'][0]['state_sha256']
    elif fault == 'gate':
        for doc in (selection, freeze):doc['fixed_runtime']['gates']['min_score'] = .6
    elif fault == 'second_candidate':selection['history'].append(deepcopy(selection['selected']))
    elif fault == 'source_objective':
        for doc in (selection, freeze):doc['source_training_objectives'][1]['unknown_loss_weight'] = 2.
    elif fault == 'result_bytes':(run/'RESULT.json').write_text('{}')
    elif fault == 'checkpoint_bytes':(run/'model.pth').write_bytes(b'Changed weights')
    elif fault == 'repeated_cal':report['calibration_inference_passes'] = 2
    elif fault == 'test_claim':report['test_passed'] = True
    else:selection['promotion_allowed'] = False
    save_evidence(run, selection, freeze, report)
    with pytest.raises(ValueError):module.validate(run, data)


def test_average_metadata_states_zero_new_steps_and_two_actual_source_histories(monkeypatch, tmp_path):
    _, _, selection, _, _, cal, parent = fixture_average(monkeypatch, tmp_path)
    metadata = module.metadata_for_export(selection, cal, parent, '0'*64, '1'*64, '2'*64, selection['selected'])
    training = metadata['training']
    assert training['stage'] == 'post_training_parameter_average'
    assert training['optimizer_steps_executed'] == 0 and training['source_optimizer_steps_executed'] == 3000
    assert training['source_selected_step'] == 1500 and training['core_selected_step'] == 1000
    assert training['core_optimizer_steps_executed'] == 1500
    assert training['method']['weights'] == [.5, .5] and training['method']['candidate_count'] == 1
    assert training['method']['declared_after_inspecting_source_calibration'] is True
    assert training['source_checkpoints'] == selection['source_checkpoints']
    assert training['source_checkpoints'] is not selection['source_checkpoints']
    assert training['parameter_groups_sha256'] == selection['parameter_groups_sha256']
    assert 'all_parameters_trained' not in training
    assert training['all_parameters_averaged'] is True
    assert training['output_averaging'] is training['inference_ensemble'] is training['teacher_in_deployed_model'] is False
    assert metadata['validation']['kind'] == 'fixed_full_state_average_calibration_only_at_export'
    assert metadata['stable_validation_passed'] is metadata['test_passed'] is False
    assert release_metadata(metadata)['validation'] == metadata['validation']
    assert len(json.dumps(metadata).encode()) < 65536


def state_pair(torch):
    left = {name+'.weight': torch.tensor([1., 3., -9.], dtype=torch.float32) for name in module.GROUPS}
    right = {name+'.weight': torch.tensor([5., -5., 7.], dtype=torch.float32) for name in module.GROUPS}
    left['trunk.num_batches_tracked'] = torch.tensor(7, dtype=torch.int64)
    right['trunk.num_batches_tracked'] = torch.tensor(7, dtype=torch.int64)
    averaged = {name: ((value.double()+right[name].double())/2).to(value.dtype)
                if value.is_floating_point() else value.clone() for name, value in left.items()}
    return left, right, averaged


def test_exact_all_group_average_preserves_sources_and_integer_buffers():
    torch = pytest.importorskip('torch')
    left, right, averaged = state_pair(torch)
    before = tuple(module.state_sha(state) for state in (left, right, averaged))
    assert module.verify_average_state(left, right, averaged) == before[-1]
    assert tuple(module.state_sha(state) for state in (left, right, averaged)) == before
    assert averaged['trunk.num_batches_tracked'].dtype == torch.int64


def test_float64_accumulation_does_not_overflow_finite_float32_inputs():
    torch = pytest.importorskip('torch')
    value = torch.finfo(torch.float32).max
    left = {'family_head.weight': torch.tensor([value, -value])}
    right = {'family_head.weight': torch.tensor([value, -value])}
    average = {'family_head.weight': left['family_head.weight'].clone()}
    assert torch.isinf((left['family_head.weight']+right['family_head.weight'])/2).all()
    module.verify_average_state(left, right, average)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
def test_average_preserves_original_floating_dtype(dtype):
    torch = pytest.importorskip('torch')
    dtype = getattr(torch, dtype)
    left = {'style.weight': torch.tensor([1., -8.], dtype=dtype)}
    right = {'style.weight': torch.tensor([5., 2.], dtype=dtype)}
    average = {'style.weight': torch.tensor([3., -3.], dtype=dtype)}
    module.verify_average_state(left, right, average)


@pytest.mark.parametrize('fault', ['missing_key', 'added_key', 'source_shape', 'source_dtype', 'candidate_dtype',
    'source_nan', 'candidate_inf', 'different_integer', 'complex', 'wrong_weight', 'unaveraged_group', 'non_tensor'])
def test_average_rejects_nonexact_partial_or_incompatible_states(fault):
    torch = pytest.importorskip('torch')
    left, right, averaged = state_pair(torch)
    key = 'family_head.weight'
    if fault == 'missing_key':averaged.pop(key)
    elif fault == 'added_key':averaged['residual_family_head.0.weight'] = torch.tensor([0.])
    elif fault == 'source_shape':right[key] = torch.zeros(2)
    elif fault == 'source_dtype':right[key] = right[key].double()
    elif fault == 'candidate_dtype':averaged[key] = averaged[key].double()
    elif fault == 'source_nan':right[key][0] = float('nan')
    elif fault == 'candidate_inf':averaged[key][0] = float('inf')
    elif fault == 'different_integer':right['trunk.num_batches_tracked'] += 1
    elif fault == 'complex':
        for state in (left, right, averaged):state[key] = state[key].to(torch.complex64)
    elif fault == 'wrong_weight':averaged[key] = .6*left[key]+.4*right[key]
    elif fault == 'unaveraged_group':averaged['size_head.weight'] = left['size_head.weight'].clone()
    else:left[key] = [1., 3., -9.]
    with pytest.raises(ValueError):module.verify_average_state(left, right, averaged)


def test_equal_numeric_values_with_different_output_bytes_are_rejected():
    torch = pytest.importorskip('torch')
    left = {'style.weight': torch.tensor([0.])}
    right = {'style.weight': torch.tensor([0.])}
    candidate = {'style.weight': torch.tensor([-0.])}
    assert torch.equal(left['style.weight'], candidate['style.weight'])
    with pytest.raises(ValueError):module.verify_average_state(left, right, candidate)


def test_checkpoint_binds_average_freeze_and_zero_optimizer_without_selection_cycle():
    torch = pytest.importorskip('torch')
    _, _, state = state_pair(torch)
    selection = {'families': [str(i) for i in range(25)], 'state_sha256': module.state_sha(state),
        'parameter_groups_sha256': module.parameter_groups(state), 'averaging_freeze_sha256': 'a'*64}
    checkpoint = {'state_dict': state, 'families': selection['families'], 'architecture': module.ARCHITECTURE,
        'step': 0, 'optimizer_steps_executed': 0, 'source_optimizer_steps_executed': 3000,
        'averaging_freeze_sha256': selection['averaging_freeze_sha256']}
    module.validate_checkpoint(checkpoint, selection)
    assert 'selection_sha256' not in checkpoint
    for field, value in [('step', 1500), ('optimizer_steps_executed', 3000), ('averaging_freeze_sha256', 'b'*64)]:
        changed = {**checkpoint, field: value}
        with pytest.raises(ValueError):module.validate_checkpoint(changed, selection)
