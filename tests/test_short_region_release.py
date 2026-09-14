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
        'bindings': source_bindings}
    selection_path = run / 'SELECTION.json'; write_json(selection_path, selection)
    runtime = {'temperature': 1.0, 'gates': {'known': .5}, 'max_size_relative_spread': .2}
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
        'batch_checks': [{'batch_size': size, 'passed': True} for size in (1, 7, 32, 128)],
        'stable_validation_passed': False, 'test_passed': False, 'test_read': False,
        'development_holdout_read': False, 'user_images_read': False,
        'selection_sha256': release.sha(selection_path),
        'checkpoint_sha256': release.sha(checkpoint), 'model_sha256': release.sha(model),
        'metadata_sha256': release.sha(metadata_path), 'fixed_runtime': runtime,
        'source_bindings': source_bindings,
        'calibration_bindings': {str(cal.resolve()): release.sha(cal)}}
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
    write_json(report_path, report)
    return {'run': run, 'region': region, 'report_path': report_path, 'freeze_path': freeze_path,
            'report': report, 'parity_path': parity_path, 'selection_path': selection_path,
            'metadata_path': metadata_path, 'source': source, 'model': model, 'checkpoint': checkpoint}


@pytest.mark.parametrize('generation',['v1','v2','v3'])
def test_prepares_new_preview_without_mutating_raw_evidence(tmp_path,monkeypatch,generation):
    if generation!='v1':
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


def test_release_policy_is_exactly_the_current_evaluator_policy():
    assert release.DEVELOPMENT_POLICY == evaluator.POLICY


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
