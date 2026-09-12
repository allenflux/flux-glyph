"""Validate averaging provenance with temporary artifacts, never real datasets."""
import builtins
import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from training.system_finetune import assess_checkpoint


FAMILIES = ['HarmonyOS Sans SC', 'MiSans', 'Noto Sans CJK SC', 'OPPO Sans',
            'PingFang', 'SF Pro', 'Helvetica', 'Alipay Number']
SOURCE_FILES = ('training/train_regions.py', 'training/system_finetune.py',
                'training/region_network.py', 'training/network.py', 'training/region_labels.py',
                'training/capture/generate_scenes.py', 'src/flux_glyph/region_font.py')
FIXED_POLICY = {'temperature': .5, 'gates': {'min_score': .5, 'min_margin': .01, 'min_patch_agreement': .7},
                'max_size_relative_spread': .2}
ALPHAS = (.05, .1, .2, .35, .5, .75, 1.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')


def module():
    return importlib.import_module('training.average_system_finetune')


@pytest.fixture
def source(tmp_path, monkeypatch):
    averaging = module()
    root, run, data = tmp_path / 'repo', tmp_path / 'source-run', tmp_path / 'data'
    root.mkdir()
    run.mkdir()
    data.mkdir()
    monkeypatch.setattr(averaging, 'ROOT', root)
    code = {}
    for name in SOURCE_FILES:
        payload = ('# Temporary provenance fixture: ' + name + '\n').encode()
        for path in (root / name, run / 'frozen-code' / name):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        code[name] = sha(root / name)
    parent_checkpoint, parent_metadata, protocol = (tmp_path / name for name in ('parent.pth', 'parent.json', 'PROTOCOL.md'))
    parent_checkpoint.write_bytes(b'Fake parent checkpoint; validation must not deserialize it')
    (run / 'last-trained.pth').write_bytes(b'Fake trained checkpoint; validation must not deserialize it')
    protocol.write_text('Frozen system-focused continuation: CAL only, 4000 optimizer steps.\n')
    metadata = {'schema': 'flux-glyph-region-font-v1', 'algorithm': 'region-cnn64x256-v1',
                'families': FAMILIES, **copy.deepcopy(FIXED_POLICY), 'training': {'state_after_sha256': 'a' * 64}}
    data_rows = [{'family': family, 'target': target} for target, family in enumerate(FAMILIES)
                 for _ in range(2 if family in ('PingFang', 'SF Pro', 'Helvetica') else 1)]
    data_rows = [{'index': index, **row} for index, row in enumerate(data_rows)]
    partitions = {}
    for split in ('train', 'calibration', 'test'):
        array, rowfile = data / split / 'tiles.npy', data / split / 'metadata.json'
        if split != 'test':
            array.parent.mkdir(parents=True)
            array.write_bytes(('Not a NumPy array: ' + split).encode())
            dump(rowfile, {'schema': 'flux-glyph-native-region-training-data-v1', 'split': split,
                           'rows': data_rows})
        partitions[split] = {'regions': len(data_rows), 'tiles': len(data_rows),
                             'family_counts': {family: 2 if family in ('PingFang', 'SF Pro', 'Helvetica') else 1 for family in FAMILIES},
                             'array': {'path': f'{split}/tiles.npy', 'sha256': sha(array) if split != 'test' else 'c' * 64},
                             'metadata': {'path': f'{split}/metadata.json', 'sha256': sha(rowfile) if split != 'test' else 'd' * 64}}
    manifest = {'schema': 'flux-glyph-native-region-training-data-v1', 'families': FAMILIES,
                'source_kind': 'ios_simulator_screenshot', 'image_source': 'simctl_png',
                'accepted_native_verified_only': True, 'splits': partitions, 'test_history': 'reused',
                'font_label_groups': {'PingFang': ['PingFang SC', 'PingFang TC', 'PingFang HK']},
                'preprocessing': {'source_sha256': code['src/flux_glyph/region_font.py']}}
    parent_stats = {'system_correct': 3, 'overall_correct': 8, 'overall_wrong': 0,
                    'nonsystem_correct': 5, 'nonsystem_wrong': 0,
                    'systems': {name: {'rows': 2, 'correct': 1, 'wrong_true': 0, 'wrong_pred': 0, 'correct_coverage': .5}
                                for name in ('PingFang', 'SF Pro', 'Helvetica')},
                    'correct_rows': [0, 1, 2, 3, 4, 6, 8, 10], 'size_mean_relative_error': .1}
    baseline = {'schema': 'flux-glyph-parent-calibration-baseline-v1', 'optimizer_steps_executed': 0,
                'parent_state_sha256': 'a' * 64, 'fixed_policy': copy.deepcopy(FIXED_POLICY),
                'calibration_partition': copy.deepcopy(partitions['calibration']),
                'selection_stats': parent_stats, 'metrics': {}, 'test_arrays_opened': False}
    selection = {'source': 'calibration_only', 'global_wrong_must_not_increase': True,
                 'system_wrong_by_true_and_predicted_family_must_not_increase': True,
                 'nonsystem_correct_must_not_decrease': True, 'nonsystem_wrong_must_not_increase': True,
                 'preserve_all_parent_accepted_correct_rows': True, 'system_correct_must_increase': True,
                 'rank': ['system_correct', 'minimum_system_correct_coverage', 'overall_correct',
                          'negative_mean_size_relative_error'], 'tie': 'earliest_checkpoint', 'recalibrate': False}
    freeze = {'schema': 'flux-glyph-region-training-freeze-v1', 'families': FAMILIES,
              'seed': 2026091294, 'optimizer_steps_planned': 4000, 'batch_size': 64, 'evaluate_every_steps': 250,
              'optimizer': {'name': 'AdamW', 'learning_rate': 3e-5, 'weight_decay': .0001},
              'source_kind': 'ios_simulator_screenshot', 'test_history': 'reused', 'test_arrays_opened': False,
              'initial_state_sha256': 'a' * 64, 'code_sha256': code,
              'model_inputs': ['RGB-derived region tiles'], 'ocr_inputs': False, 'script_mask': False,
              'regions': {split: len(data_rows) for split in partitions}, 'selection': selection,
              'calibration': {**copy.deepcopy(FIXED_POLICY), 'policy_source': 'frozen_parent_metadata', 'recalibrate': False},
              'warm_start': {'path': str(parent_checkpoint), 'sha256': sha(parent_checkpoint), 'families': FAMILIES,
                             'preserved_size_head': True, 'new_family_initializers': {}},
              'training_region_sampling': {'system_family_weight': 2,
                                          'family_weights': {name: 2 if name in ('PingFang', 'SF Pro', 'Helvetica') else 1 for name in FAMILIES}},
              'regression_protocol': {'path': str(protocol), 'sha256': sha(protocol)},
              'system_focused': {'enabled': True, 'system_families': ['PingFang', 'SF Pro', 'Helvetica'],
                                 'system_family_weight': 2, 'parent_state_sha256': 'a' * 64,
                                 'fixed_policy': copy.deepcopy(FIXED_POLICY), 'selection_policy': copy.deepcopy(selection)}}
    history = [{'step': step, 'eligibility': assess_checkpoint(parent_stats, parent_stats),
                'selection_score': assess_checkpoint(parent_stats, parent_stats)['rank'],
                'selection_stats': copy.deepcopy(parent_stats), 'calibration': {}, 'unique_train_regions_sampled': len(data_rows)}
               for step in range(250, 4001, 250)]
    report = {'status': 'NO_PROMOTABLE_CHECKPOINT', 'optimizer_steps_executed': 4000,
              'state_before_sha256': 'a' * 64, 'state_after_sha256': 'b' * 64, 'parameters_changed': True,
              'last_checkpoint': {'path': 'last-trained.pth', 'sha256': sha(run / 'last-trained.pth')},
              'test_arrays_opened': False, 'test_history': 'reused'}
    weights = freeze['training_region_sampling']['family_weights']
    visits = {family: (4000 * 64 // sum(weights.values())) * weight for family, weight in weights.items()}
    visits[FAMILIES[0]] += 4000 * 64 - sum(visits.values())
    report['training_region_sampling'] = {'family_weights': copy.deepcopy(weights), 'family_visits': visits}
    case = SimpleNamespace(module=averaging, root=root, run=run, data=data, metadata_path=parent_metadata,
                           parent_checkpoint=parent_checkpoint, protocol=protocol, metadata=metadata,
                           manifest=manifest, baseline=baseline, freeze=freeze, history=history, report=report)

    def rebind():
        """Recompute file links so semantic-tampering tests get past SHA checks."""
        dump(parent_metadata, metadata)
        metadata_ref = {'path': str(parent_metadata), 'sha256': sha(parent_metadata)}
        baseline['parent_metadata'] = copy.deepcopy(metadata_ref)
        freeze['system_focused']['parent_metadata'] = copy.deepcopy(metadata_ref)
        dump(data / 'MANIFEST.json', manifest)
        for document in (baseline, freeze):
            document['data_manifest_sha256'] = sha(data / 'MANIFEST.json')
        dump(run / 'PARENT_CALIBRATION.json', baseline)
        baseline_ref = {'path': 'PARENT_CALIBRATION.json', 'sha256': sha(run / 'PARENT_CALIBRATION.json')}
        freeze['system_focused']['parent_calibration'] = copy.deepcopy(baseline_ref)
        report['parent_calibration'] = copy.deepcopy(baseline_ref)
        dump(run / 'TRAINING_FREEZE.json', freeze)
        report['training_freeze_sha256'] = sha(run / 'TRAINING_FREEZE.json')
        dump(run / 'history.json', history)
        report['history_sha256'] = sha(run / 'history.json')
        dump(run / 'NO_PROMOTABLE_CHECKPOINT.json', report)
        dump(run / 'report.json', report)

    case.rebind = rebind
    rebind()
    return case


def validate(source):
    return source.module.validate_source_run(source.run, source.data)


def test_valid_provenance_is_read_only_without_torch_numpy_arrays_or_test_files(source, monkeypatch):
    before = {str(path): path.read_bytes() for path in source.run.parent.rglob('*') if path.is_file()}
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split('.')[0] != 'torch', 'Provenance validation imported torch'
        return original_import(name, *args, **kwargs)

    def no_array_load(*args, **kwargs):
        pytest.fail('Provenance validation must not open NumPy arrays')

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(np, 'load', no_array_load)
    context = validate(source)
    assert context['source_run'] == source.run and context['data_dir'] == source.data
    assert context['families'] == FAMILIES and context['policy'] == FIXED_POLICY
    assert context['parent_checkpoint'] == source.parent_checkpoint
    assert context['trained_checkpoint'] == source.run / 'last-trained.pth'
    assert context['parent_metadata'] == source.metadata and context['manifest'] == source.manifest
    assert context['report'] == source.report and context['freeze'] == source.freeze and context['baseline'] == source.baseline
    assert context['hashes'] and all(entry['sha256'] == sha(entry['path']) for entry in context['hashes'].values())
    assert not (source.data / 'test').exists()
    assert before == {str(path): path.read_bytes() for path in source.run.parent.rglob('*') if path.is_file()}


@pytest.mark.parametrize('artifact', ['last-trained.pth', 'history.json', 'TRAINING_FREEZE.json',
                                    'PARENT_CALIBRATION.json', 'parent.json', 'parent.pth', 'PROTOCOL.md',
                                    'MANIFEST.json', 'live-code', 'frozen-code'])
def test_changed_bound_artifacts_are_rejected(source, artifact):
    paths = {'parent.json': source.metadata_path, 'parent.pth': source.parent_checkpoint, 'PROTOCOL.md': source.protocol,
             'MANIFEST.json': source.data / 'MANIFEST.json',
             'live-code': source.root / SOURCE_FILES[0], 'frozen-code': source.run / 'frozen-code' / SOURCE_FILES[0]}
    path = paths.get(artifact, source.run / artifact)
    path.write_bytes(path.read_bytes() + b'\nchanged after source training')
    with pytest.raises(ValueError):
        validate(source)


@pytest.mark.parametrize('owner', ['report', 'freeze', 'baseline'])
@pytest.mark.parametrize('opened', [True, 'false'])
def test_every_source_stage_must_explicitly_keep_test_unopened(source, owner, opened):
    getattr(source, owner)['test_arrays_opened'] = opened
    source.rebind()
    with pytest.raises(ValueError):
        validate(source)


@pytest.mark.parametrize('failure', ['promoted', 'report_differs', 'short_budget', 'short_run', 'baseline_trained',
                                    'unchanged', 'change_flag_false', 'state_binding', 'parent_state_binding',
                                    'history_missing', 'history_duplicate', 'history_late', 'history_eligible',
                                    'history_forged_ineligible', 'policy_drift', 'parent_family_gates',
                                    'calibration_binding', 'missing_code', 'missing_protocol', 'selection_present'])
def test_rehashed_but_inconsistent_source_claims_are_rejected(source, failure):
    if failure == 'promoted':
        source.report['status'] = 'PROMOTED'
    elif failure == 'short_budget':
        source.freeze['optimizer_steps_planned'] = 3999
    elif failure == 'short_run':
        source.report['optimizer_steps_executed'] = 3999
    elif failure == 'baseline_trained':
        source.baseline['optimizer_steps_executed'] = 1
    elif failure == 'unchanged':
        source.report['state_after_sha256'] = source.report['state_before_sha256']
    elif failure == 'change_flag_false':
        source.report['parameters_changed'] = False
    elif failure == 'state_binding':
        source.freeze['initial_state_sha256'] = 'e' * 64
    elif failure == 'parent_state_binding':
        source.metadata['training']['state_after_sha256'] = 'e' * 64
    elif failure == 'history_missing':
        source.history.pop(3)
    elif failure == 'history_duplicate':
        source.history[3]['step'] = source.history[2]['step']
    elif failure == 'history_late':
        source.history[-1]['step'] = 4250
    elif failure == 'history_eligible':
        source.history[-1]['eligibility']['eligible'] = True
    elif failure == 'history_forged_ineligible':
        stats = source.history[-1]['selection_stats']
        stats['system_correct'] += 1
        stats['overall_correct'] += 1
        stats['systems']['PingFang']['correct'] += 1
        stats['systems']['PingFang']['correct_coverage'] = 1.
        stats['correct_rows'].append(5)
        # Existing false eligibility is now a lie; validation must recompute it.
        assert assess_checkpoint(stats, source.baseline['selection_stats'])['eligible']
    elif failure == 'policy_drift':
        source.baseline['fixed_policy']['gates']['min_score'] = .4
    elif failure == 'parent_family_gates':
        source.metadata['family_gates'] = {'PingFang': {'min_score': .4, 'min_margin': .01, 'min_patch_agreement': .7}}
    elif failure == 'calibration_binding':
        source.baseline['calibration_partition']['array']['sha256'] = 'e' * 64
    elif failure == 'missing_code':
        source.freeze['code_sha256'].pop('training/system_finetune.py')
    elif failure == 'missing_protocol':
        source.freeze.pop('regression_protocol')
    source.rebind()
    if failure == 'report_differs':
        dump(source.run / 'report.json', {**source.report, 'parameters_changed': False})
    elif failure == 'selection_present':
        dump(source.run / 'SELECTION_FREEZE.json', {'test_arrays_opened': True})
    with pytest.raises(ValueError):
        validate(source)


def test_checkpoint_reference_cannot_escape_source_run_even_with_matching_hash(source):
    escaped = source.run.parent / 'outside.pth'
    escaped.write_bytes(b'Not an artifact from this source run')
    source.report['last_checkpoint'] = {'path': '../outside.pth', 'sha256': sha(escaped)}
    source.rebind()
    with pytest.raises(ValueError):
        validate(source)


def test_alpha_grid_is_fixed_and_parent_only_alpha_is_not_a_candidate():
    assert module().ALPHAS == ALPHAS and isinstance(module().ALPHAS, tuple)
    assert 0. not in module().ALPHAS


def candidate_records():
    return [{'alpha': alpha, 'eligibility': {'eligible': False, 'reasons': ['parent_correct_rows_lost'],
                                            'rank': [4, .5, 9, -.1]}} for alpha in ALPHAS]


def test_selection_ignores_ineligible_ranks_and_keeps_earlier_alpha_on_ties():
    records = candidate_records()
    records[0]['eligibility']['rank'][0] = 100  # Not eligible despite the large gain.
    for index in (1, 3, 4, 5, 6):
        records[index]['eligibility'].update(eligible=True, reasons=[])
    records[4]['eligibility']['rank'][0] = 6
    records[5]['eligibility']['rank'][0] = 6
    assert module().choose_candidate(records) is records[4]
    assert module().choose_candidate(candidate_records()) is None


@pytest.mark.parametrize('failure', ['zero', 'missing', 'extra', 'reordered', 'duplicate', 'refined'])
def test_selection_cannot_add_parent_baseline_or_change_fixed_grid(failure):
    records = candidate_records()
    if failure == 'zero':
        records.insert(0, {**copy.deepcopy(records[0]), 'alpha': 0.})
    elif failure == 'missing':
        records.pop()
    elif failure == 'extra':
        records.append(copy.deepcopy(records[-1]))
    elif failure == 'reordered':
        records.reverse()
    elif failure == 'duplicate':
        records[1]['alpha'] = records[0]['alpha']
    else:
        records[3]['alpha'] = .3
    with pytest.raises(ValueError, match='alpha'):
        module().choose_candidate(records)


def averaging_plan(source):
    context = validate(source)
    names = ('parent_checkpoint', 'parent_metadata', 'source_failure', 'source_training_freeze',
             'source_history', 'last_checkpoint', 'data_manifest')
    plan = {'schema': 'flux-glyph-system-averaging-fixed-regression-v1', 'operation': 'parameter_averaging',
            'alpha_grid': list(ALPHAS), 'alpha_zero_baseline_only': True, 'formula': source.module.FORMULA,
            'fixed_policy': copy.deepcopy(FIXED_POLICY), 'system_families': ['PingFang', 'SF Pro', 'Helvetica'],
            'additional_optimizer_steps_executed': 0, 'source_optimizer_steps_executed': 4000,
            'source_preconditions': {'source_status': 'NO_PROMOTABLE_CHECKPOINT', 'source_optimizer_steps_executed': 4000,
                                     'source_test_arrays_opened': False, 'source_parameters_changed': True},
            'selection': {'source': 'calibration_only', 'same_eligibility_as_system_finetuning': True,
                          'preserve_all_parent_accepted_correct_rows': True, 'rank': source.freeze['selection']['rank'],
                          'tie': 'earliest_alpha_in_fixed_grid', 'only_last_training_endpoint': True,
                          'single_selected_model': True, 'no_grid_refinement': True},
            'source_bindings': {name: copy.deepcopy(context['hashes'][name]) for name in names}}
    return context, plan


def test_independent_averaging_plan_is_bound_before_predictions(source):
    context, plan = averaging_plan(source)
    path = source.run.parent / 'AVERAGING_PLAN.json'
    dump(path, plan)
    before = path.read_bytes()
    assert source.module.validate_averaging_protocol(path, context) == {'path': str(path), 'sha256': sha(path)}
    assert path.read_bytes() == before


@pytest.mark.parametrize('failure', ['grid_zero', 'grid_extra', 'grid_reordered', 'policy', 'optimizer_steps',
                                    'test_source', 'loss_allowed', 'parent_is_candidate', 'source_binding',
                                    'endpoint', 'refinement', 'source_changed_flag'])
def test_independent_plan_cannot_relax_selection_or_change_sources(source, failure):
    context, plan = averaging_plan(source)
    if failure == 'grid_zero':
        plan['alpha_grid'].insert(0, 0.)
    elif failure == 'grid_extra':
        plan['alpha_grid'].append(.3)
    elif failure == 'grid_reordered':
        plan['alpha_grid'].reverse()
    elif failure == 'policy':
        plan['fixed_policy']['gates']['min_score'] = .4
    elif failure == 'optimizer_steps':
        plan['additional_optimizer_steps_executed'] = 1
    elif failure == 'test_source':
        plan['selection']['source'] = 'test'
    elif failure == 'loss_allowed':
        plan['selection']['preserve_all_parent_accepted_correct_rows'] = False
    elif failure == 'parent_is_candidate':
        plan['alpha_zero_baseline_only'] = False
    elif failure == 'source_binding':
        plan['source_bindings']['last_checkpoint']['sha256'] = 'f' * 64
    elif failure == 'endpoint':
        plan['selection']['only_last_training_endpoint'] = False
    elif failure == 'refinement':
        plan['selection']['no_grid_refinement'] = False
    elif failure == 'source_changed_flag':
        plan['source_preconditions']['source_parameters_changed'] = False
    path = source.run.parent / 'AVERAGING_PLAN.json'
    dump(path, plan)
    with pytest.raises(ValueError):
        source.module.validate_averaging_protocol(path, context)


def test_baseline_reproduction_allows_only_size_roundoff_not_row_replacements(source):
    expected = source.baseline['selection_stats']
    actual = copy.deepcopy(expected)
    actual['size_mean_relative_error'] += 1e-9
    source.module.verify_parent_stats(actual, expected)
    actual['correct_rows'][4] = 5
    with pytest.raises(ValueError, match='baseline'):
        source.module.verify_parent_stats(actual, expected)


def test_blending_cpu_float32_states_preserves_inputs_and_baseline_endpoint():
    torch = pytest.importorskip('torch')
    parent = {'weight': torch.tensor([[1., -2.]], dtype=torch.float32), 'bias': torch.tensor([3.], dtype=torch.float32)}
    trained = {'weight': torch.tensor([[5., 6.]], dtype=torch.float32), 'bias': torch.tensor([-1.], dtype=torch.float32)}
    snapshots = [{key: value.clone() for key, value in state.items()} for state in (parent, trained)]
    for alpha in (0., *ALPHAS):
        blended = module().blend_states(parent, trained, alpha)
        assert set(blended) == set(parent)
        for key in parent:
            torch.testing.assert_close(blended[key], (1 - alpha) * parent[key] + alpha * trained[key])
            assert blended[key].dtype == torch.float32 and blended[key].device.type == 'cpu'
            assert blended[key].data_ptr() not in (parent[key].data_ptr(), trained[key].data_ptr())
    for state, snapshot in zip((parent, trained), snapshots):
        for key in state:
            torch.testing.assert_close(state[key], snapshot[key], rtol=0, atol=0)


def test_identical_nonfloating_buffers_are_copied_without_interpolation():
    torch = pytest.importorskip('torch')
    parent = {'counter': torch.tensor([2], dtype=torch.int64)}
    trained = {'counter': torch.tensor([2], dtype=torch.int64)}
    blended = module().blend_states(parent, trained, .5)
    torch.testing.assert_close(blended['counter'], parent['counter'])
    assert blended['counter'].data_ptr() not in (parent['counter'].data_ptr(), trained['counter'].data_ptr())


@pytest.mark.parametrize('failure', ['keys', 'shape', 'dtype', 'integer', 'nan', 'infinity',
                                    'negative_alpha', 'excessive_alpha', 'nan_alpha', 'infinite_alpha', 'off_grid_alpha'])
def test_blend_rejects_incompatible_or_nonfinite_parameters_without_gpu(failure):
    torch = pytest.importorskip('torch')
    parent = {'weight': torch.ones(2, dtype=torch.float32)}
    trained = {'weight': torch.zeros(2, dtype=torch.float32)}
    alpha = .5
    if failure == 'keys':
        trained['extra'] = torch.ones(2)
    elif failure == 'shape':
        trained['weight'] = torch.zeros(3)
    elif failure == 'dtype':
        trained['weight'] = trained['weight'].to(torch.float64)
    elif failure == 'integer':
        parent['weight'] = parent['weight'].to(torch.int64)
        trained['weight'] = trained['weight'].to(torch.int64)
    elif failure == 'nan':
        parent['weight'][0] = float('nan')
    elif failure == 'infinity':
        trained['weight'][0] = float('inf')
    else:
        alpha = {'negative_alpha': -.1, 'excessive_alpha': 1.1, 'nan_alpha': float('nan'),
                 'infinite_alpha': float('inf'), 'off_grid_alpha': .3}[failure]
    with pytest.raises(ValueError):
        module().blend_states(parent, trained, alpha)
