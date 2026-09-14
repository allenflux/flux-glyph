#!/usr/bin/env python3
"""Prepare a new preview region directory after the short-region CAL and DEV gates."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile


VERSION = 'r23-unified-font-short-regions-v1-preview'
SELECTION_SCHEMA = 'flux-glyph-unified-short-selection-v1'
PARITY_SCHEMA = 'flux-glyph-unified-short-onnx-parity-v1'
METADATA_SCHEMA = 'flux-glyph-unified-region-font-v1'
DEVELOPMENT_SCHEMA = 'flux-glyph-unified-short-development-regression-v1'
FREEZE_SCHEMA = 'flux-glyph-unified-short-development-freeze-v1'
EVIDENCE_SCHEMA = 'flux-glyph-unified-short-release-evidence-v1'
SUPPORTED_SCHEMAS = {
    SELECTION_SCHEMA: (PARITY_SCHEMA, DEVELOPMENT_SCHEMA, FREEZE_SCHEMA),
    'flux-glyph-unified-short-selection-v2': (
        'flux-glyph-unified-short-onnx-parity-v2',
        'flux-glyph-unified-short-development-regression-v2',
        'flux-glyph-unified-short-development-freeze-v2'),
    'flux-glyph-unified-short-selection-v3': (
        'flux-glyph-unified-short-onnx-parity-v3',
        'flux-glyph-unified-short-development-regression-v3',
        'flux-glyph-unified-short-development-freeze-v3'),
    'flux-glyph-unified-native-short-selection-v1': (
        'flux-glyph-unified-native-short-onnx-parity-v1',
        'flux-glyph-unified-native-short-development-regression-v1',
        'flux-glyph-unified-native-short-development-freeze-v1'),
    'flux-glyph-unified-native-oe-selection-v1': (
        'flux-glyph-unified-native-oe-onnx-parity-v1',
        'flux-glyph-unified-native-oe-development-regression-v1',
        'flux-glyph-unified-native-oe-development-freeze-v1'),
    'flux-glyph-unified-native-ios-selection-v1': (
        'flux-glyph-unified-native-ios-onnx-parity-v1',
        'flux-glyph-unified-native-ios-development-regression-v1',
        'flux-glyph-unified-native-ios-development-freeze-v1'),
    'flux-glyph-unified-native-mobile-selection-v1': (
        'flux-glyph-unified-native-mobile-onnx-parity-v1',
        'flux-glyph-unified-native-mobile-development-regression-v1',
        'flux-glyph-unified-native-mobile-development-freeze-v1'),
}
CAL_REGIONS = 18672
CAL_TILES = 37834
CAL_ORIGINAL_CHECKS = 46
CAL_SHORT_CHECKS = 7
CAL_CHECKS = CAL_ORIGINAL_CHECKS + CAL_SHORT_CHECKS
DEV_CHECKS_PER_BASELINE = 18
DEV_CHECKS = DEV_CHECKS_PER_BASELINE * 2
DEV_VIEWS = 3504
DEV_TILES = 5967
FIXED_RUNTIME = {'temperature': 1.0,
    'gates': {'min_score': .7, 'min_margin': .01, 'min_patch_agreement': 2 / 3},
    'max_size_relative_spread': .2}
DEVELOPMENT_POLICY = {
    'maximum_named_precision_drop': .005,
    'maximum_known_coverage_drop': .01,
    'maximum_unknown_withholding_drop': .02,
    'maximum_size_median_ape': .10,
    'maximum_size_p90_ape': .25,
    'minimum_size_coverage_of_correct_names': .70,
    'populations': ['all', 'ios', 'android'],
    'is_blind_test': False,
    'used_to_select_training_checkpoint': False,
}
NATIVE_MOBILE_ANDROID_GUARD_POLICY = {
    'name': 'android_wrong_named_as_ios_system_family', 'domain': 'android',
    'predicted_families': ['PingFang', 'SF Pro', 'Helvetica'], 'metric': 'wrong_named',
    'operator': 'le', 'maximum': 27, 'r22_actual': 27,
    'population': 'all true CAL Android regions', 'original_53_checks_unchanged': True,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def _checks(value, count, label):
    require(isinstance(value, list) and len(value) == count
            and all(isinstance(row, dict) and row.get('passed') is True for row in value),
            f'{label} must contain exactly {count} passing checks')


def _verify_bindings(bindings, label):
    require(isinstance(bindings, dict) and bindings, f'{label} bindings are missing')
    for raw_path, digest in bindings.items():
        path = Path(raw_path)
        require(isinstance(raw_path, str) and isinstance(digest, str) and len(digest) == 64
                and path.is_file() and sha(path) == digest,
                f'{label} binding changed: {raw_path}')


def _bound(bindings, path, digest, label):
    resolved = Path(path).resolve()
    matches = [value for key, value in bindings.items() if Path(key).resolve() == resolved]
    require(matches == [digest], f'{label} does not bind {resolved}')


def _validate_cal(selection, parity, metadata):
    require(selection.get('runtime') == FIXED_RUNTIME
            and parity.get('fixed_runtime') == FIXED_RUNTIME
            and {key: metadata.get(key) for key in FIXED_RUNTIME} == FIXED_RUNTIME
            and metadata.get('validation', {}).get('fixed_runtime') == FIXED_RUNTIME,
            'CAL, ONNX metadata and PARITY must retain the frozen runtime')
    require(selection.get('schema') in SUPPORTED_SCHEMAS
            and selection.get('calibration_promotion_allowed') is True
            and selection.get('promotion_allowed') is False
            and selection.get('development_evaluated') is False
            and selection.get('exported') is False
            and selection.get('deployed') is False,
            'SELECTION must remain CAL-qualified and immutable at the pre-DEV boundary')
    selected = selection.get('selected', {})
    comparison = selection.get('r22_comparison', {})
    _checks(selected.get('retention_checks'), CAL_ORIGINAL_CHECKS, 'Original CAL retention')
    _checks(comparison.get('checks'), CAL_SHORT_CHECKS, 'Short-region CAL regression')
    require(selected.get('promotion_allowed') is True and comparison.get('passed') is True,
            'SELECTION does not pass the exact 46+7 CAL policy')

    require(parity.get('schema') == SUPPORTED_SCHEMAS[selection['schema']][0] and parity.get('passed') is True
            and parity.get('calibration_promotion_allowed') is True
            and parity.get('promotion_allowed') is False
            and parity.get('development_evaluated') is False
            and parity.get('calibration_regions') == CAL_REGIONS
            and parity.get('calibration_tiles') == CAL_TILES
            and parity.get('original_retention_checks_passed') == CAL_ORIGINAL_CHECKS
            and parity.get('short_checks_passed') == CAL_SHORT_CHECKS
            and parity.get('calibration_checks_passed') == CAL_CHECKS
            and parity.get('font_model_count') == 1
            and parity.get('encoder_count') == 1
            and parity.get('output_family_count') == 25
            and parity.get('one_deployed_cnn') is True
            and parity.get('platform_routing') is False
            and parity.get('score_merging') is False
            and parity.get('runtime_gates_changed') is False
            and parity.get('calibration_font_decisions_identical') is True
            and parity.get('calibration_runtime_signatures_identical') is True
            and parity.get('calibration_size_values_close') is True
            and parity.get('cached_mps_outputs_close') is True
            and parity.get('original_torch_groupnorm_reference') is True
            and parity.get('export_parameters_unchanged') is True
            and parity.get('teacher_cache_deployed') is False
            and parity.get('stable_validation_passed') is False
            and parity.get('test_passed') is False
            and parity.get('test_read') is False
            and parity.get('development_holdout_read') is False
            and parity.get('user_images_read') is False,
            'PARITY must prove the full 37,834-tile, one-CNN, 46+7 CAL boundary')
    if selection['schema'] == 'flux-glyph-unified-native-ios-selection-v1':
        guard = parity.get('android_system_font_confusion', {})
        require(guard.get('baseline_count') == guard.get('maximum_count') == 27
                and type(guard.get('actual_count')) is int and 0 <= guard['actual_count'] <= 27
                and guard.get('passed') is True
                and guard.get('predicted_families') == ['PingFang', 'SF Pro', 'Helvetica'],
                'PARITY Android system-font confusion guard must pass in addition to CAL53')
    if selection['schema'] == 'flux-glyph-unified-native-mobile-selection-v1':
        policy = parity.get('android_system_guard_policy')
        guard = parity.get('android_system_guard', {})
        baseline = parity.get('android_system_guard_baseline', {})
        require(policy == NATIVE_MOBILE_ANDROID_GUARD_POLICY
                and guard == {**policy, 'actual': guard.get('actual'), 'passed': guard.get('passed')}
                and type(guard.get('actual')) is int and 0 <= guard['actual'] <= 27
                and guard.get('passed') is True
                and baseline == {**policy, 'actual': 27, 'passed': True}
                and metadata.get('validation', {}).get('android_system_guard_policy') == policy
                and metadata.get('validation', {}).get('android_system_guard') == guard,
                'PARITY native-mobile Android system guard must pass in addition to CAL53')
    batches = parity.get('batch_checks')
    require(isinstance(batches, list) and len(batches) == 4
            and {row.get('batch_size') for row in batches} == {1, 7, 32, 128}
            and all(row.get('passed') is True and row.get('font_and_size_checked') is True
                    for row in batches),
            'PARITY batch checks are incomplete')

    validation = metadata.get('validation', {})
    training = metadata.get('training', {})
    require(metadata.get('schema') == METADATA_SCHEMA
            and metadata.get('release_tier') == 'experimental'
            and metadata.get('stable_validation_passed') is False
            and metadata.get('test_passed') is False
            and validation.get('calibration_promotion_allowed') is True
            and validation.get('promotion_allowed') is False
            and validation.get('development_holdout_evaluated') is False
            and validation.get('original_retention_checks') == CAL_ORIGINAL_CHECKS
            and validation.get('short_regression_checks') == CAL_SHORT_CHECKS
            and validation.get('calibration_checks') == CAL_CHECKS
            and validation.get('model_count') == 1
            and validation.get('encoder_count') == 1
            and validation.get('platform_routing') is False
            and validation.get('blind_test_performed') is False
            and training.get('model_count') == 1
            and training.get('development_holdout_read') is False
            and training.get('test_read') is False,
            'Raw metadata must remain a one-CNN experimental pre-DEV export')
    _checks(validation.get('retention_checks'), CAL_ORIGINAL_CHECKS, 'Metadata CAL retention')
    _checks(validation.get('short_comparison', {}).get('checks'), CAL_SHORT_CHECKS,
            'Metadata short-region CAL regression')
    require(validation['short_comparison'].get('passed') is True,
            'Raw metadata short-region comparison did not pass')


def validate_calibration_boundary(selection, parity, metadata):
    """Require complete CAL and export evidence before DEV can start."""
    _validate_cal(selection, parity, metadata)


def _validate_development(report, freeze, parity):
    native_trials = {
        'flux-glyph-unified-native-short-onnx-parity-v1': 'native-short-v1',
        'flux-glyph-unified-native-oe-onnx-parity-v1': 'native-oe-v1',
        'flux-glyph-unified-native-ios-onnx-parity-v1': 'native-ios-v1',
        'flux-glyph-unified-native-mobile-onnx-parity-v1': 'native-mobile-v1',
    }
    if parity.get('schema') in native_trials:
        candidate = report.get('source_version_summaries', {}).get('candidate', {})
        require(candidate.get('trial_id') == native_trials[parity['schema']]
                and candidate.get('intended_release_version') == VERSION
                and candidate.get('model_sha256') == parity.get('model_sha256')
                and candidate.get('selection_sha256') == parity.get('selection_sha256'),
                'Development candidate identity differs from the verified export')
    if parity.get('schema') == 'flux-glyph-unified-native-mobile-onnx-parity-v1':
        require(report.get('android_system_guard_policy') == freeze.get('android_system_guard_policy')
                == parity.get('android_system_guard_policy') == NATIVE_MOBILE_ANDROID_GUARD_POLICY
                and report.get('android_system_guard') == freeze.get('android_system_guard')
                == parity.get('android_system_guard'),
                'Development evidence changed the native-mobile Android system guard')
    matches = [schemas for schemas in SUPPORTED_SCHEMAS.values() if schemas[0] == parity.get('schema')]
    require(len(matches) == 1, 'Unknown short-region parity schema')
    _, development_schema, freeze_schema = matches[0]
    require(report.get('schema') == development_schema
            and report.get('evaluation_kind') == 'reused_development_regression'
            and report.get('release_tier') == 'preview'
            and report.get('promotion_allowed') is True
            and report.get('calibration_promotion_allowed') is True
            and report.get('calibration_checks_passed') == CAL_CHECKS
            and report.get('development_checks_passed') == DEV_CHECKS
            and report.get('development_checks_required') == DEV_CHECKS
            and report.get('model_count') == 1 and report.get('encoder_count') == 1
            and report.get('platform_routing') is False
            and report.get('score_merging') is False
            and report.get('stable_validation_passed') is False
            and report.get('test_passed') is False
            and report.get('test_read') is False
            and report.get('blind_test_performed') is False
            and report.get('used_to_select_training_checkpoint') is False,
            'Development evidence must be a passing 36-check preview report')
    for key, version in (('r21_comparison', 'r21-unified-font-v1-preview'),
                         ('r22_comparison', 'r22-unified-font-retention-v1-preview')):
        comparison = report.get(key, {})
        _checks(comparison.get('checks'), DEV_CHECKS_PER_BASELINE, key)
        require(comparison.get('passed') is True
                and comparison.get('policy') == {**DEVELOPMENT_POLICY,
                                                  'baseline_version': version},
                f'{key} policy or result differs')
    proof = report.get('proof_counts', {})
    require(proof.get('calibration_regions') == CAL_REGIONS
            and proof.get('calibration_checks') == CAL_CHECKS
            and proof.get('development_views') == DEV_VIEWS
            and proof.get('development_tiles') == DEV_TILES
            and proof.get('r21_development_checks') == DEV_CHECKS_PER_BASELINE
            and proof.get('r22_development_checks') == DEV_CHECKS_PER_BASELINE
            and proof.get('development_checks') == DEV_CHECKS
            and proof.get('deployed_cnns') == 1,
            'Development proof counts differ')
    require(freeze.get('schema') == freeze_schema
            and freeze.get('calibration_checks_required') == CAL_CHECKS
            and freeze.get('development_checks_required') == DEV_CHECKS
            and freeze.get('test_read') is False
            and freeze.get('blind_test_performed') is False
            and freeze.get('used_to_select_training_checkpoint') is False
            and freeze.get('fixed_runtime') == parity.get('fixed_runtime')
            and report.get('fixed_runtime') == parity.get('fixed_runtime'),
            'Development freeze or fixed runtime differs')
    policies = freeze.get('policies')
    require(isinstance(policies, list) and len(policies) == 2
            and policies == [{**DEVELOPMENT_POLICY,
                               'baseline_version': 'r21-unified-font-v1-preview'},
                              {**DEVELOPMENT_POLICY,
                               'baseline_version': 'r22-unified-font-retention-v1-preview'}],
            'Development freeze must contain the two explicit 18-check policies')


def validate_inputs(run, region, development_report, development_freeze):
    run, region = Path(run).resolve(), Path(region).resolve()
    development_report, development_freeze = (Path(development_report).resolve(),
                                               Path(development_freeze).resolve())
    paths = {'selection': run / 'SELECTION.json', 'checkpoint': run / 'model.pth',
             'parity': run / 'PARITY.json', 'model': region / 'model.onnx',
             'metadata': region / 'metadata.json', 'development_report': development_report,
             'development_freeze': development_freeze}
    require(all(path.is_file() for path in paths.values()), 'Required release evidence is missing')
    values = {name: read(path) for name, path in paths.items()
              if name not in ('checkpoint', 'model')}
    selection, parity, metadata = values['selection'], values['parity'], values['metadata']
    report, freeze = values['development_report'], values['development_freeze']
    _validate_cal(selection, parity, metadata)
    _validate_development(report, freeze, parity)

    model_relative = metadata.get('model', {}).get('path')
    require(model_relative == 'model.onnx'
            and [path.resolve() for path in region.rglob('*.onnx')] == [paths['model'].resolve()],
            'Raw region must contain exactly one model.onnx CNN')
    hashes = {name: sha(path) for name, path in paths.items()}
    require(metadata['model'].get('sha256') == hashes['model']
            and metadata.get('families') == selection.get('families')
            and isinstance(metadata.get('families'), list) and len(metadata['families']) == 25
            and len(set(metadata['families'])) == 25 and metadata['families'][-1] == '__unknown__'
            and metadata['validation'].get('fixed_runtime') == parity.get('fixed_runtime')
            and metadata['training'].get('selection_sha256') == hashes['selection']
            and metadata['training'].get('checkpoint_sha256') == hashes['checkpoint']
            and parity.get('selection_sha256') == hashes['selection']
            and parity.get('checkpoint_sha256') == hashes['checkpoint']
            and parity.get('model_sha256') == hashes['model']
            and parity.get('metadata_sha256') == hashes['metadata']
            and report.get('model_sha256') == hashes['model']
            and report.get('freeze_sha256') == hashes['development_freeze'],
            'Selection, checkpoint, model, metadata or evidence hashes differ')

    for label, bindings in (('SELECTION source', selection.get('bindings')),
                            ('PARITY source', parity.get('source_bindings')),
                            ('PARITY calibration', parity.get('calibration_bindings')),
                            ('DEV report', report.get('bindings')),
                            ('DEV freeze', freeze.get('bindings'))):
        _verify_bindings(bindings, label)
    require(report['bindings'] == freeze['bindings'], 'DEV report and freeze binding closures differ')
    for name in ('selection', 'checkpoint', 'parity', 'model', 'metadata'):
        _bound(report['bindings'], paths[name], hashes[name], 'DEV evidence')
    require(report.get('development_partition_sha256') in set(report['bindings'].values()),
            'DEV partition hash is absent from the evidence closure')
    return {'paths': paths, 'hashes': hashes, 'selection': selection, 'parity': parity,
            'metadata': metadata, 'report': report, 'freeze': freeze}


def prepare_release(run, region, development_report, output, development_freeze=None,
                    version=VERSION):
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', version) is not None,
            'Invalid release version')
    development_report = Path(development_report).resolve()
    if development_freeze is None:
        development_freeze = development_report.with_name(development_report.stem + '_FREEZE.json')
    state = validate_inputs(run, region, development_report, development_freeze)
    output = Path(output).resolve()
    require(not output.exists(), 'Choose a new release-ready region directory')
    output.parent.mkdir(parents=True, exist_ok=True)

    promoted = copy.deepcopy(state['metadata'])
    promoted['version'] = version
    promoted['release_tier'] = 'experimental'
    promoted['stable_validation_passed'] = False
    promoted['test_passed'] = False
    validation = promoted['validation']
    validation.update(kind='short_region_continuation_calibration_and_dual_development_preview',
        promotion_allowed=True, development_holdout_evaluated=True,
        development_checks=DEV_CHECKS, stable_validation_passed=False,
        test_passed=False, test_read=False, blind_test_performed=False,
        development_evidence={
            'report_sha256': state['hashes']['development_report'],
            'freeze_sha256': state['hashes']['development_freeze'],
            'r21_checks': DEV_CHECKS_PER_BASELINE, 'r22_checks': DEV_CHECKS_PER_BASELINE})
    metadata_bytes = encoded(promoted)
    promoted_metadata_sha = hashlib.sha256(metadata_bytes).hexdigest()
    final_model = output / 'model.onnx'; final_metadata = output / 'metadata.json'
    evidence = {'schema': EVIDENCE_SCHEMA, 'version': version, 'release_tier': 'preview',
        'promotion_allowed': True, 'calibration_promotion_allowed': True,
        'calibration': {'regions': CAL_REGIONS, 'tiles': CAL_TILES,
                        'original_checks': CAL_ORIGINAL_CHECKS,
                        'short_checks': CAL_SHORT_CHECKS, 'checks': CAL_CHECKS},
        'development': {'r21_checks': DEV_CHECKS_PER_BASELINE,
                        'r22_checks': DEV_CHECKS_PER_BASELINE, 'checks': DEV_CHECKS,
                        'views': DEV_VIEWS, 'tiles': DEV_TILES},
        'model_count': 1, 'encoder_count': 1, 'platform_routing': False,
        'score_merging': False, 'stable_validation_passed': False,
        'test_passed': False, 'test_read': False, 'blind_test_performed': False,
        'raw_inputs_immutable': True,
        'input_bindings': {str(state['paths'][name]): digest
                           for name, digest in state['hashes'].items()},
        'output_bindings': {str(final_model): state['hashes']['model'],
                            str(final_metadata): promoted_metadata_sha},
        'source_bindings_verified': True, 'parity_bindings_verified': True,
        'development_bindings_verified': True}

    temp = Path(tempfile.mkdtemp(prefix='.short-release-', dir=output.parent))
    try:
        shutil.copyfile(state['paths']['model'], temp / 'model.onnx')
        (temp / 'metadata.json').write_bytes(metadata_bytes)
        (temp / 'RELEASE_EVIDENCE.json').write_bytes(encoded(evidence))
        require(sha(temp / 'model.onnx') == state['hashes']['model']
                and sha(temp / 'metadata.json') == promoted_metadata_sha,
                'Release-ready model or metadata copy changed')
        require(all(sha(path) == state['hashes'][name]
                    for name, path in state['paths'].items()),
                'Raw selection, checkpoint, model, metadata or evidence changed during preparation')
        for label, bindings in (('SELECTION source', state['selection']['bindings']),
                                ('PARITY source', state['parity']['source_bindings']),
                                ('PARITY calibration', state['parity']['calibration_bindings']),
                                ('DEV evidence', state['report']['bindings'])):
            _verify_bindings(bindings, label)
        temp.rename(output)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--region', type=Path, required=True,
                        help='Raw experimental region export; it is never modified')
    parser.add_argument('--development-report', type=Path, required=True)
    parser.add_argument('--development-freeze', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', default=VERSION)
    args = parser.parse_args()
    result = prepare_release(args.run, args.region, args.development_report, args.output,
                             args.development_freeze, args.version)
    print(json.dumps({'directory': str(args.output.resolve()),
                      'version': result['version'],
                      'promotion_allowed': result['promotion_allowed']}, indent=2))
