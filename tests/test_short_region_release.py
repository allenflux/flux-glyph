"""Synthetic release-boundary tests for the short-region preview."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from scripts import prepare_short_region_release as release
from training import evaluate_unified_short_regions as evaluator


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    run, region = tmp_path / 'run', tmp_path / 'raw-region'
    run.mkdir(); region.mkdir()
    source = tmp_path / 'frozen-source.py'; source.write_text('frozen = True\n')
    cal = tmp_path / 'calibration.manifest'; cal.write_text('calibration identity\n')
    dev = tmp_path / 'development.manifest'; dev.write_text('development identity\n')
    checkpoint = run / 'model.pth'; checkpoint.write_bytes(b'checkpoint')
    model = region / 'model.onnx'; model.write_bytes(b'one synthetic cnn')
    source_bindings = {str(source.resolve()): release.sha(source)}
    selection = {'schema': release.SELECTION_SCHEMA, 'families': [f'f{i}' for i in range(24)] + ['__unknown__'],
        'calibration_promotion_allowed': True, 'promotion_allowed': False,
        'development_evaluated': False, 'exported': False, 'deployed': False,
        'selected': {'promotion_allowed': True,
                     'retention_checks': [{'passed': True} for _ in range(46)]},
        'r22_comparison': {'passed': True, 'checks': [{'passed': True} for _ in range(7)]},
        'bindings': source_bindings, 'runtime': deepcopy(release.FIXED_RUNTIME)}
    selection_path = run / 'SELECTION.json'; write_json(selection_path, selection)
    runtime = deepcopy(release.FIXED_RUNTIME)
    metadata = {'schema': release.METADATA_SCHEMA, 'model': {'path': 'model.onnx', 'sha256': release.sha(model)},
        'families': selection['families'], **runtime, 'release_tier': 'experimental',
        'stable_validation_passed': False, 'test_passed': False,
        'validation': {'calibration_promotion_allowed': True, 'promotion_allowed': False,
            'development_holdout_evaluated': False, 'original_retention_checks': 46,
            'short_regression_checks': 7, 'calibration_checks': 53,
            'retention_checks': [{'passed': True} for _ in range(46)],
            'short_comparison': {'passed': True, 'checks': [{'passed': True} for _ in range(7)]},
            'fixed_runtime': runtime, 'model_count': 1, 'encoder_count': 1,
            'platform_routing': False, 'blind_test_performed': False},
        'training': {'model_count': 1, 'selection_sha256': release.sha(selection_path),
            'checkpoint_sha256': release.sha(checkpoint), 'development_holdout_read': False,
            'test_read': False}}
    metadata_path = region / 'metadata.json'; write_json(metadata_path, metadata)
    parity = {'schema': release.PARITY_SCHEMA, 'passed': True,
        'calibration_promotion_allowed': True, 'promotion_allowed': False,
        'development_evaluated': False, 'calibration_regions': 18672, 'calibration_tiles': 37834,
        'original_retention_checks_passed': 46, 'short_checks_passed': 7,
        'calibration_checks_passed': 53, 'font_model_count': 1, 'encoder_count': 1,
        'output_family_count': 25, 'one_deployed_cnn': True, 'platform_routing': False,
        'score_merging': False, 'runtime_gates_changed': False,
        'calibration_font_decisions_identical': True,
        'calibration_runtime_signatures_identical': True, 'calibration_size_values_close': True,
        'cached_mps_outputs_close': True, 'original_torch_groupnorm_reference': True,
        'export_parameters_unchanged': True, 'teacher_cache_deployed': False,
        'batch_checks': [{'batch_size': size, 'passed': True, 'font_and_size_checked': True}
                         for size in (1, 7, 32, 128)],
        'stable_validation_passed': False, 'test_passed': False, 'test_read': False,
        'development_holdout_read': False, 'user_images_read': False,
        'selection_sha256': release.sha(selection_path),
        'checkpoint_sha256': release.sha(checkpoint), 'model_sha256': release.sha(model),
        'metadata_sha256': release.sha(metadata_path), 'fixed_runtime': runtime,
        'source_bindings': source_bindings,
        'calibration_bindings': {str(cal.resolve()): release.sha(cal)},
        'android_system_font_confusion': {'baseline_count': 27, 'maximum_count': 27, 'actual_count': 20,
            'passed': True, 'predicted_families': ['PingFang', 'SF Pro', 'Helvetica']}}
    if 'native-mobile' in release.PARITY_SCHEMA:
        from training.train_unified_native_mobile import ANDROID_SYSTEM_GUARD
        guard = {**deepcopy(ANDROID_SYSTEM_GUARD), 'actual': 20, 'passed': True}
        baseline = {**deepcopy(ANDROID_SYSTEM_GUARD), 'actual': 27, 'passed': True}
        parity.update(android_system_guard_policy=deepcopy(ANDROID_SYSTEM_GUARD),
                      android_system_guard=guard, android_system_guard_baseline=baseline)
        metadata['validation'].update(android_system_guard_policy=deepcopy(ANDROID_SYSTEM_GUARD),
                                      android_system_guard=guard)
        write_json(metadata_path, metadata)
        parity['metadata_sha256'] = release.sha(metadata_path)
    parity_path = run / 'PARITY.json'; write_json(parity_path, parity)
    evidence_bindings = {str(path.resolve()): release.sha(path) for path in
        (selection_path, checkpoint, parity_path, model, metadata_path, source, cal, dev)}
    freeze = {'schema': release.FREEZE_SCHEMA, 'bindings': evidence_bindings,
        'policies': [{**deepcopy(release.DEVELOPMENT_POLICY), 'baseline_version': version}
                     for version in ('r21-unified-font-v1-preview',
                                     'r22-unified-font-retention-v1-preview')],
        'calibration_checks_required': 53, 'development_checks_required': 36,
        'fixed_runtime': runtime, 'test_read': False, 'blind_test_performed': False,
        'used_to_select_training_checkpoint': False}
    if 'native-mobile' in release.PARITY_SCHEMA:
        freeze['android_system_guard_policy'] = deepcopy(parity['android_system_guard_policy'])
        freeze['android_system_guard'] = deepcopy(parity['android_system_guard'])
    report_path = tmp_path / 'DEVELOPMENT_REGRESSION.json'
    freeze_path = tmp_path / 'DEVELOPMENT_REGRESSION_FREEZE.json'; write_json(freeze_path, freeze)
    def comparison(version):
        return {'passed': True, 'checks': [{'passed': True} for _ in range(18)],
                'policy': {**deepcopy(release.DEVELOPMENT_POLICY),
                           'baseline_version': version}}
    report = {'schema': release.DEVELOPMENT_SCHEMA,
        'evaluation_kind': 'reused_development_regression', 'release_tier': 'preview',
        'promotion_allowed': True, 'calibration_promotion_allowed': True,
        'calibration_checks_passed': 53, 'development_checks_passed': 36,
        'development_checks_required': 36,
        'r21_comparison': comparison('r21-unified-font-v1-preview'),
        'r22_comparison': comparison('r22-unified-font-retention-v1-preview'),
        'bindings': evidence_bindings, 'freeze_sha256': release.sha(freeze_path),
        'model_sha256': release.sha(model), 'development_partition_sha256': release.sha(dev),
        'fixed_runtime': runtime, 'model_count': 1, 'encoder_count': 1,
        'platform_routing': False, 'score_merging': False,
        'stable_validation_passed': False, 'test_passed': False, 'test_read': False,
        'blind_test_performed': False, 'used_to_select_training_checkpoint': False,
        'proof_counts': {'calibration_regions': 18672, 'calibration_checks': 53,
            'development_views': 3504, 'development_tiles': 5967,
            'r21_development_checks': 18, 'r22_development_checks': 18,
            'development_checks': 36, 'deployed_cnns': 1}}
    if 'native-mobile' in release.PARITY_SCHEMA:
        report['android_system_guard_policy'] = deepcopy(parity['android_system_guard_policy'])
        report['android_system_guard'] = deepcopy(parity['android_system_guard'])
    if any(name in release.PARITY_SCHEMA for name in ('native-short', 'native-oe', 'native-ios', 'native-mobile')):
        trial = next(name for name in ('native-short', 'native-oe', 'native-ios', 'native-mobile')
                     if name in release.PARITY_SCHEMA)
        report['source_version_summaries'] = {'candidate': {
            'trial_id': trial + '-v1',
            'intended_release_version': release.VERSION,
            'model_sha256': release.sha(model), 'selection_sha256': release.sha(selection_path)}}
    write_json(report_path, report)
    return {'run': run, 'region': region, 'report_path': report_path, 'freeze_path': freeze_path,
            'report': report, 'parity_path': parity_path, 'selection_path': selection_path,
            'metadata_path': metadata_path, 'source': source, 'model': model, 'checkpoint': checkpoint}


@pytest.mark.parametrize('generation',['v1','v2','v3','native','native_oe','native_ios','native_mobile'])
def test_prepares_new_preview_without_mutating_raw_evidence(tmp_path,monkeypatch,generation):
    if generation in ('native','native_oe','native_ios','native_mobile'):
        for name in ('SELECTION_SCHEMA','PARITY_SCHEMA','DEVELOPMENT_SCHEMA','FREEZE_SCHEMA'):
            replacement = 'unified-native-short' if generation == 'native' else 'unified-' + generation.replace('_', '-')
            monkeypatch.setattr(release,name,getattr(release,name).replace('unified-short',replacement))
    elif generation!='v1':
        for name in ('SELECTION_SCHEMA','PARITY_SCHEMA','DEVELOPMENT_SCHEMA','FREEZE_SCHEMA'):
            monkeypatch.setattr(release,name,getattr(release,name).removesuffix('v1')+generation)
    f = fixture(tmp_path); output = tmp_path / 'release-ready'
    before = {name: release.sha(f[name]) for name in
              ('selection_path', 'metadata_path', 'parity_path', 'report_path',
               'freeze_path', 'model', 'checkpoint', 'source')}
    evidence = release.prepare_release(f['run'], f['region'], f['report_path'], output,
                                       f['freeze_path'])
    assert evidence['promotion_allowed'] is True
    assert evidence['calibration'] == {'regions': 18672, 'tiles': 37834,
        'original_checks': 46, 'short_checks': 7, 'checks': 53}
    assert evidence['development']['r21_checks'] == evidence['development']['r22_checks'] == 18
    assert evidence['development']['checks'] == 36 and evidence['model_count'] == 1
    assert evidence['stable_validation_passed'] is evidence['test_passed'] is False
    assert evidence['test_read'] is evidence['blind_test_performed'] is False
    assert {name: release.sha(f[name]) for name in before} == before
    assert (output / 'model.onnx').read_bytes() == f['model'].read_bytes()
    promoted = json.loads((output / 'metadata.json').read_text())
    assert promoted['version'] == release.VERSION and promoted['release_tier'] == 'experimental'
    assert promoted['validation']['promotion_allowed'] is True
    assert promoted['validation']['development_holdout_evaluated'] is True
    assert promoted['stable_validation_passed'] is promoted['test_passed'] is False
    assert json.loads((output / 'RELEASE_EVIDENCE.json').read_text()) == evidence


def test_rejects_mixed_training_and_parity_generations(tmp_path):
    f=fixture(tmp_path)
    selection=json.loads(f['selection_path'].read_text())
    selection['schema']='flux-glyph-unified-short-selection-v2'
    write_json(f['selection_path'],selection)
    with pytest.raises(ValueError,match='PARITY'):
        release.prepare_release(f['run'],f['region'],f['report_path'],tmp_path/'output',f['freeze_path'])


@pytest.mark.parametrize('field,value', [('trial_id','native-short-v1'),
    ('intended_release_version','r22-unified-font-retention-v1-preview'),
    ('model_sha256','0'*64), ('selection_sha256','0'*64)])
@pytest.mark.parametrize('generation', ['native-oe', 'native-ios', 'native-mobile'])
def test_native_oe_release_rejects_mixed_candidate_identity(tmp_path,monkeypatch,field,value,generation):
    for name in ('SELECTION_SCHEMA','PARITY_SCHEMA','DEVELOPMENT_SCHEMA','FREEZE_SCHEMA'):
        monkeypatch.setattr(release,name,getattr(release,name).replace('unified-short','unified-' + generation))
    f=fixture(tmp_path); report=json.loads(f['report_path'].read_text())
    report['source_version_summaries']['candidate'][field]=value
    write_json(f['report_path'],report)
    with pytest.raises(ValueError,match='candidate identity'):
        release.prepare_release(f['run'],f['region'],f['report_path'],tmp_path/'output',f['freeze_path'])


def test_release_policy_is_exactly_the_current_evaluator_policy():
    from training.train_unified_retention import FIXED_RUNTIME
    assert release.DEVELOPMENT_POLICY == evaluator.POLICY
    assert release.FIXED_RUNTIME == FIXED_RUNTIME


@pytest.mark.parametrize('key,value', [('actual_count', 28), ('actual_count', True),
    ('baseline_count', 28), ('maximum_count', 28), ('passed', False), ('predicted_families', ['PingFang'])])
def test_ios_trial_requires_android_confusion_guard_in_addition_to_cal53(tmp_path, monkeypatch, key, value):
    for name in ('SELECTION_SCHEMA','PARITY_SCHEMA','DEVELOPMENT_SCHEMA','FREEZE_SCHEMA'):
        monkeypatch.setattr(release,name,getattr(release,name).replace('unified-short','unified-native-ios'))
    f = fixture(tmp_path)
    selection, parity, metadata = (json.loads(f[k].read_text()) for k in ('selection_path','parity_path','metadata_path'))
    release.validate_calibration_boundary(selection, parity, metadata)
    parity['android_system_font_confusion'][key] = value
    with pytest.raises(ValueError, match='confusion guard'):
        release.validate_calibration_boundary(selection, parity, metadata)
    del parity['android_system_font_confusion']
    with pytest.raises(ValueError, match='confusion guard'):
        release.validate_calibration_boundary(selection, parity, metadata)


@pytest.mark.parametrize(('target', 'key', 'value'), [
    ('policy', 'maximum', 28), ('guard', 'actual', 28), ('guard', 'passed', False),
    ('baseline', 'actual', 26), ('metadata_guard', 'actual', 19)])
def test_mobile_trial_requires_separate_frozen_android_guard(tmp_path, monkeypatch, target, key, value):
    for name in ('SELECTION_SCHEMA','PARITY_SCHEMA','DEVELOPMENT_SCHEMA','FREEZE_SCHEMA'):
        monkeypatch.setattr(release, name,
            getattr(release, name).replace('unified-short', 'unified-native-mobile'))
    f = fixture(tmp_path)
    selection, parity, metadata = (json.loads(f[k].read_text())
        for k in ('selection_path', 'parity_path', 'metadata_path'))
    release.validate_calibration_boundary(selection, parity, metadata)
    locations = {'policy': parity['android_system_guard_policy'],
        'guard': parity['android_system_guard'],
        'baseline': parity['android_system_guard_baseline'],
        'metadata_guard': metadata['validation']['android_system_guard']}
    locations[target][key] = value
    with pytest.raises(ValueError, match='native-mobile Android system guard'):
        release.validate_calibration_boundary(selection, parity, metadata)


def test_mobile_release_rejects_guard_changed_after_dev_freeze(tmp_path, monkeypatch):
    for name in ('SELECTION_SCHEMA','PARITY_SCHEMA','DEVELOPMENT_SCHEMA','FREEZE_SCHEMA'):
        monkeypatch.setattr(release, name,
            getattr(release, name).replace('unified-short', 'unified-native-mobile'))
    f = fixture(tmp_path)
    report = json.loads(f['report_path'].read_text())
    report['android_system_guard']['actual'] -= 1
    write_json(f['report_path'], report)
    with pytest.raises(ValueError, match='Development evidence changed'):
        release.prepare_release(f['run'], f['region'], f['report_path'], tmp_path / 'output',
                                f['freeze_path'])


@pytest.mark.parametrize('field', ['gates', 'temperature', 'max_size_relative_spread'])
def test_cal_boundary_rejects_joint_export_runtime_changes(tmp_path, field):
    f = fixture(tmp_path)
    selection = json.loads(f['selection_path'].read_text())
    parity = json.loads(f['parity_path'].read_text())
    metadata = json.loads(f['metadata_path'].read_text())
    changed = deepcopy(release.FIXED_RUNTIME)
    changed[field] = {'min_score': .5, 'min_margin': .01, 'min_patch_agreement': 2/3} if field == 'gates' else .5
    parity['fixed_runtime'] = deepcopy(changed)
    metadata.update(deepcopy(changed))
    metadata['validation']['fixed_runtime'] = deepcopy(changed)
    with pytest.raises(ValueError, match='frozen runtime'):
        release.validate_calibration_boundary(selection, parity, metadata)
    selection['runtime'] = deepcopy(changed)
    with pytest.raises(ValueError, match='frozen runtime'):
        release.validate_calibration_boundary(selection, parity, metadata)


@pytest.mark.parametrize('key', [
    'calibration_tiles', 'calibration_regions', 'original_retention_checks_passed',
    'short_checks_passed', 'font_model_count', 'encoder_count', 'output_family_count',
    'one_deployed_cnn', 'platform_routing', 'score_merging', 'runtime_gates_changed',
    'calibration_font_decisions_identical', 'calibration_runtime_signatures_identical',
    'calibration_size_values_close', 'cached_mps_outputs_close',
    'original_torch_groupnorm_reference', 'export_parameters_unchanged',
    'teacher_cache_deployed', 'batch_checks',
])
@pytest.mark.parametrize('generation', ['native_short', 'native_oe', 'native_ios'])
def test_native_dev_entry_rejects_incomplete_parity_before_inference(tmp_path, monkeypatch, key, generation):
    from types import SimpleNamespace
    import importlib
    native = importlib.import_module('training.evaluate_unified_' + generation)
    f = fixture(tmp_path)
    selection = json.loads(f['selection_path'].read_text())
    selection['schema'] = 'flux-glyph-unified-' + generation.replace('_', '-') + '-selection-v1'
    parity = json.loads(f['parity_path'].read_text())
    parity['schema'] = native.PARITY_SCHEMA
    metadata = json.loads(f['metadata_path'].read_text())
    release.validate_calibration_boundary(selection, parity, metadata)
    del parity[key]
    write_json(f['parity_path'], parity)
    monkeypatch.setattr(native, 'validate', lambda *args: selection)
    def no_inference(*args, **kwargs):
        pytest.fail('Incomplete export proof reached model or DEV loading')
    monkeypatch.setattr(native, 'UnifiedFontClassifier', no_inference)
    monkeypatch.setattr(native, 'load_split', no_inference)
    args = SimpleNamespace(run=f['run'], data=tmp_path/'data', plan=tmp_path/'plan.json',
                           region=f['region'], output=tmp_path/'DEV.json')
    with pytest.raises(ValueError, match='PARITY'):
        native.evaluate(args)
    assert not args.output.exists()


@pytest.mark.parametrize('change', ['tile_count', 'batch_size', 'batch_failed', 'size_parity',
                                   'batch_size_unchecked', 'batch_size_missing'])
def test_shared_cal_boundary_rejects_changed_parity(tmp_path, change):
    f = fixture(tmp_path)
    selection = json.loads(f['selection_path'].read_text())
    parity = json.loads(f['parity_path'].read_text())
    metadata = json.loads(f['metadata_path'].read_text())
    if change == 'tile_count': parity['calibration_tiles'] = 37833
    elif change == 'batch_size': parity['batch_checks'][1]['batch_size'] = 8
    elif change == 'batch_failed': parity['batch_checks'][1]['passed'] = False
    elif change == 'batch_size_unchecked': parity['batch_checks'][1]['font_and_size_checked'] = False
    elif change == 'batch_size_missing': del parity['batch_checks'][1]['font_and_size_checked']
    else: parity['calibration_size_values_close'] = False
    with pytest.raises(ValueError, match='PARITY'):
        release.validate_calibration_boundary(selection, parity, metadata)


@pytest.mark.parametrize(('target', 'mutate'), [
    ('selection', lambda value: value['selected']['retention_checks'].pop()),
    ('parity', lambda value: value.__setitem__('calibration_tiles', 37833)),
    ('parity', lambda value: value.__setitem__('font_model_count', 2)),
    ('report', lambda value: value['r21_comparison']['checks'][0].__setitem__('passed', False)),
    ('report', lambda value: value.__setitem__('development_checks_passed', 35)),
])
def test_rejects_incomplete_cal_parity_or_dual_dev_proof(tmp_path, target, mutate):
    f = fixture(tmp_path)
    path = {'selection': f['selection_path'], 'parity': f['parity_path'],
            'report': f['report_path']}[target]
    value = json.loads(path.read_text()); mutate(value); write_json(path, value)
    with pytest.raises(ValueError):
        release.prepare_release(f['run'], f['region'], f['report_path'], tmp_path / 'output',
                                f['freeze_path'])


def test_rejects_changed_bound_source_and_extra_cnn(tmp_path):
    f = fixture(tmp_path); f['source'].write_text('changed after freeze\n')
    with pytest.raises(ValueError, match='binding changed'):
        release.prepare_release(f['run'], f['region'], f['report_path'], tmp_path / 'source-output',
                                f['freeze_path'])
    f = fixture(tmp_path / 'second'); (f['region'] / 'other.onnx').write_bytes(b'extra cnn')
    with pytest.raises(ValueError, match='exactly one'):
        release.prepare_release(f['run'], f['region'], f['report_path'], tmp_path / 'cnn-output',
                                f['freeze_path'])


def test_rejects_relaxed_development_policy(tmp_path):
    f = fixture(tmp_path)
    report = json.loads(f['report_path'].read_text())
    report['r22_comparison']['policy']['maximum_known_coverage_drop'] = .5
    write_json(f['report_path'], report)
    with pytest.raises(ValueError, match='policy or result differs'):
        release.prepare_release(f['run'], f['region'], f['report_path'], tmp_path / 'output',
                                f['freeze_path'])


def test_raw_promotion_is_rejected_and_existing_output_is_preserved(tmp_path):
    f = fixture(tmp_path)
    raw = json.loads(f['metadata_path'].read_text())
    raw['validation']['promotion_allowed'] = True; write_json(f['metadata_path'], raw)
    with pytest.raises(ValueError, match='pre-DEV'):
        release.prepare_release(f['run'], f['region'], f['report_path'], tmp_path / 'output',
                                f['freeze_path'])
    f = fixture(tmp_path / 'second'); output = tmp_path / 'exists'; output.mkdir()
    (output / 'keep').write_text('keep')
    with pytest.raises(ValueError, match='new release-ready'):
        release.prepare_release(f['run'], f['region'], f['report_path'], output, f['freeze_path'])
    assert (output / 'keep').read_text() == 'keep'
