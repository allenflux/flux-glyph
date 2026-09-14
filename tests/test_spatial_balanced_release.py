"""Synthetic release-boundary tests for the short-region preview."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from scripts import prepare_spatial_balanced_release as release


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
    selection.update(trial='balanced-low', android_system_guard={**release.NATIVE_MOBILE_ANDROID_GUARD_POLICY, 'actual':20,'passed':True}, android_system_guard_baseline={**release.NATIVE_MOBILE_ANDROID_GUARD_POLICY, 'actual':27,'passed':True})
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
    parity = {'schema': release.PARITY_SCHEMA, 'trial': 'balanced-low', 'passed': True,
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
    report['android_system_guard_policy'] = deepcopy(parity['android_system_guard_policy'])
    report['android_system_guard'] = deepcopy(parity['android_system_guard'])
    report['source_version_summaries'] = {'candidate': {
        'trial_id': 'balanced-low', 'intended_release_version': release.VERSION,
        'model_sha256': release.sha(model), 'selection_sha256': release.sha(selection_path)}}
    write_json(report_path, report)
    return {'run': run, 'region': region, 'report_path': report_path, 'freeze_path': freeze_path,
            'report': report, 'parity_path': parity_path, 'selection_path': selection_path,
            'metadata_path': metadata_path, 'source': source, 'model': model, 'checkpoint': checkpoint}



def test_valid_release_preserves_raw_artifacts_and_carries_all_checks(tmp_path):
    f = fixture(tmp_path)
    paths = ['selection_path','metadata_path','parity_path','report_path','freeze_path','model','checkpoint']
    before = {k: release.sha(f[k]) for k in paths}
    out = tmp_path / 'release'
    evidence = release.prepare_release(f['run'], f['region'], f['report_path'], out, f['freeze_path'])
    assert evidence['promotion_allowed'] is True
    assert evidence['calibration']['checks'] == 53 and evidence['development']['checks'] == 36
    assert evidence['test_read'] is evidence['stable_validation_passed'] is False
    assert before == {k: release.sha(f[k]) for k in paths}
    assert (out/'model.onnx').read_bytes() == f['model'].read_bytes()


@pytest.mark.parametrize('kind', ['guard', 'trial', 'cal', 'dev', 'model', 'source'])
def test_incomplete_or_mismatched_evidence_cannot_promote(tmp_path, kind):
    f = fixture(tmp_path)
    if kind in ('guard','trial','cal'):
        value = release.read(f['parity_path'])
        if kind == 'guard': value['android_system_guard']['actual'] = 28
        elif kind == 'trial': value['trial'] = 'balanced-high'
        else: value['calibration_checks_passed'] = 52
        write_json(f['parity_path'],value)
    elif kind == 'dev':
        value = release.read(f['report_path']); value['r22_comparison']['checks'][0]['passed'] = False
        write_json(f['report_path'],value)
    elif kind == 'model': f['model'].write_bytes(b'changed model')
    else: f['source'].write_text('changed source')
    with pytest.raises(ValueError):
        release.prepare_release(f['run'],f['region'],f['report_path'],tmp_path/'rejected',f['freeze_path'])
    assert not (tmp_path/'rejected').exists()
