#!/usr/bin/env python3
"""Run the two required 18-check DEV comparisons for a short-region ONNX preview."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'training')]
from train_regions import require, sha
from train_unified_regions import decisions, region_outputs
from prepare_unified_regions import load_split
from evaluate_unified_regions import summarize
from flux_glyph.unified_font import UnifiedFontClassifier
from scripts.prepare_short_region_release import validate_calibration_boundary
from export_unified_native_mobile import (CAL_REGIONS, METADATA_SCHEMA, PARITY_SCHEMA,
    STEPS, validate)

DEV_VIEWS = 3504
DEV_TILES = 5967
R21_BASELINE = ROOT / 'artifacts/unified-font-v1/run-v1/DEVELOPMENT_REGRESSION.json'
R22_BASELINE = ROOT / 'artifacts/unified-font-v3/run-wide-micro-recovery-v1/DEVELOPMENT_REGRESSION.json'
POLICY = {
    'maximum_named_precision_drop': .005, 'maximum_known_coverage_drop': .01,
    'maximum_unknown_withholding_drop': .02, 'maximum_size_median_ape': .10,
    'maximum_size_p90_ape': .25, 'minimum_size_coverage_of_correct_names': .70,
    'populations': ['all', 'ios', 'android'], 'is_blind_test': False,
    'used_to_select_training_checkpoint': False,
}


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def comparison(current, baseline, baseline_version):
    require(baseline_version in ('r21-unified-font-v1-preview', 'r22-unified-font-retention-v1-preview'),
            'Development baseline version must be explicit')
    checks = []
    groups = [('all', current, baseline)] + [(domain, current['per_domain'][domain],
              baseline['per_domain'][domain]) for domain in ('ios', 'android')]
    for population, actual, old in groups:
        for metric, tolerance in (
                ('named_precision', 'maximum_named_precision_drop'),
                ('known_correct_coverage', 'maximum_known_coverage_drop'),
                ('unknown_not_named_rate', 'maximum_unknown_withholding_drop')):
            reference, value = old.get(metric), actual.get(metric)
            require(reference is not None, 'Development baseline population is incomplete')
            minimum = reference - POLICY[tolerance]
            checks.append({'population': population, 'metric': metric, 'baseline': reference,
                           'minimum': minimum, 'actual': value,
                           'passed': value is not None and value + 1e-12 >= minimum})
        for metric, limit, direction in (
                ('median_ape', .10, 'maximum'), ('p90_ape', .25, 'maximum'),
                ('coverage_of_correct_names', .70, 'minimum')):
            value = actual['size'][metric]
            checks.append({'population': population, 'metric': 'size.' + metric,
                           direction: limit, 'actual': value,
                           'passed': value is not None and
                           (value <= limit if direction == 'maximum' else value >= limit)})
    policy = {**copy.deepcopy(POLICY), 'baseline_version': baseline_version}
    return {'passed': all(row['passed'] for row in checks), 'checks': checks, 'policy': policy}


def build_development_report(selection, parity, measured, by_view, confusion, denominators,
                             details, r21, r22, bindings, freeze_sha, partition_sha):
    require(selection.get('calibration_promotion_allowed') is True
            and selection.get('promotion_allowed') is False
            and selection.get('development_evaluated') is False
            and parity.get('passed') is True
            and parity.get('calibration_promotion_allowed') is True
            and parity.get('promotion_allowed') is False
            and parity.get('development_evaluated') is False
            and parity.get('calibration_checks_passed') == 53,
            'Development report requires the exact CAL-qualified raw-export boundary')
    require(len(r21.get('checks', [])) == len(r22.get('checks', [])) == 18
            and r21['policy']['baseline_version'] == 'r21-unified-font-v1-preview'
            and r22['policy']['baseline_version'] == 'r22-unified-font-retention-v1-preview'
            and r21.get('passed') is all(row.get('passed') is True for row in r21['checks'])
            and r22.get('passed') is all(row.get('passed') is True for row in r22['checks']),
            'Both explicit 18-check development comparisons are required')
    promotion = bool(r21['passed'] and r22['passed'])
    return {'schema': 'flux-glyph-unified-native-mobile-development-regression-v1',
        'evaluation_kind': 'reused_development_regression', 'release_tier': 'preview',
        'promotion_allowed': promotion, 'calibration_promotion_allowed': True,
        'calibration_checks_passed': 53, 'development_checks_passed':
            sum(row['passed'] for row in r21['checks']) + sum(row['passed'] for row in r22['checks']),
        'development_checks_required': 36, 'r21_comparison': r21, 'r22_comparison': r22,
        'metrics': measured, 'by_domain': measured['per_domain'], 'by_family': measured['per_family'],
        'by_view': by_view, 'confusion': confusion, 'denominators': denominators, 'rows': details,
        'bindings': bindings, 'freeze_sha256': freeze_sha, 'model_sha256': parity['model_sha256'],
        'development_partition_sha256': partition_sha, 'fixed_runtime': copy.deepcopy(parity['fixed_runtime']),
        'android_system_guard_policy': copy.deepcopy(parity.get('android_system_guard_policy')),
        'android_system_guard': copy.deepcopy(parity.get('android_system_guard')),
        'model_count': 1, 'encoder_count': 1, 'platform_routing': False, 'score_merging': False,
        'stable_validation_passed': False, 'test_passed': False, 'test_read': False,
        'blind_test_performed': False, 'used_to_select_training_checkpoint': False,
        'source_version_summaries': {
            'r21_baseline': {'version': 'r21-unified-font-v1-preview',
                             'report_sha256': bindings[str(R21_BASELINE.resolve())]},
            'r22_baseline': {'version': 'r22-unified-font-retention-v1-preview',
                             'report_sha256': bindings[str(R22_BASELINE.resolve())]},
            'candidate': {'trial_id': 'native-mobile-v1',
                          'intended_release_version': 'r23-unified-font-short-regions-v1-preview',
                          'selection_sha256': parity['selection_sha256'],
                          'model_sha256': parity['model_sha256']}},
        'proof_counts': {'calibration_regions': CAL_REGIONS, 'calibration_checks': 53,
                         'development_views': DEV_VIEWS, 'development_tiles': DEV_TILES,
                         'r21_development_checks': 18, 'r22_development_checks': 18,
                         'development_checks': 36, 'deployed_cnns': 1},
        'scope': 'The same 3,504-view development partition is reused twice: once against R21 and once against R22. This permits an experimental preview only; it is not blind TEST or stable validation.'}


def evaluate(args):
    run, data, plan, region, output = map(Path.resolve,
        map(Path, (args.run, args.data, args.plan, args.region, args.output)))
    selection = validate(run, data, plan)
    parity_path = run / 'PARITY.json'; parity = read(parity_path)
    metadata_path = region / 'metadata.json'; model_path = region / 'model.onnx'
    metadata = read(metadata_path)
    validate_calibration_boundary(selection, parity, metadata)
    require(parity.get('schema') == PARITY_SCHEMA and parity.get('passed') is True
            and parity.get('calibration_promotion_allowed') is True
            and parity.get('promotion_allowed') is False and parity.get('development_evaluated') is False
            and parity.get('calibration_checks_passed') == 53
            and parity.get('selection_sha256') == sha(run / 'SELECTION.json')
            and parity.get('checkpoint_sha256') == sha(run / 'model.pth')
            and parity.get('model_sha256') == sha(model_path)
            and parity.get('metadata_sha256') == sha(metadata_path)
            and metadata.get('schema') == METADATA_SCHEMA
            and metadata.get('release_tier') == 'experimental'
            and metadata.get('stable_validation_passed') is False
            and metadata.get('test_passed') is False
            and metadata.get('validation', {}).get('promotion_allowed') is False,
            'Short raw export or CAL parity differs')
    model = UnifiedFontClassifier(region)
    require(model.output_families == selection['families'] and parity['output_family_count'] == 25
            and all(model.meta[key] == value for key, value in parity['fixed_runtime'].items()),
            'Short ONNX family order or runtime changed')
    r21_path, r22_path = Path(args.r21_baseline).resolve(), Path(args.r22_baseline).resolve()
    r21_baseline, r22_baseline = read(r21_path), read(r22_path)
    partition_path = data / 'development_holdout/MANIFEST.json'
    partition_sha = sha(partition_path)
    require(r21_path == R21_BASELINE.resolve() and r22_path == R22_BASELINE.resolve()
            and r21_baseline.get('schema') == 'flux-glyph-unified-development-regression-v1'
            and r22_baseline.get('schema') == 'flux-glyph-retention-development-regression-v1'
            and r21_baseline.get('test_read') is False and r22_baseline.get('test_read') is False
            and r21_baseline.get('blind_test_performed') is False
            and r22_baseline.get('blind_test_performed') is False
            and r21_baseline.get('development_inference_completed') is True
            and r22_baseline.get('comparison_passed') is True
            and r21_baseline.get('development_partition_sha256') == partition_sha
            and r22_baseline.get('development_partition_sha256') == partition_sha,
            'R21/R22 baselines do not use the exact same development partition')
    require(selection['bindings'].get(str(r21_path)) == sha(r21_path)
            and selection['bindings'].get(str(r22_path)) == sha(r22_path)
            and parity.get('calibration_bindings', {}).get(str(r21_path)) == sha(r21_path)
            and parity.get('calibration_bindings', {}).get(str(r22_path)) == sha(r22_path)
            and all(sha(path) == digest for path, digest in parity.get('source_bindings', {}).items())
            and all(sha(path) == digest for path, digest in parity.get('calibration_bindings', {}).items()),
            'Export source closure or R21/R22 baseline binding changed before DEV')
    freeze_path = output.with_name(output.stem + '_FREEZE.json')
    require(not output.exists() and not freeze_path.exists(), 'Preserve previous development evidence')
    part = read(partition_path)
    dev_files = [data / 'MANIFEST.json', partition_path,
                 data / 'development_holdout' / part['metadata']['path'],
                 data / 'development_holdout' / part['array']['path']]
    source_files = [Path(__file__), ROOT / 'training/export_unified_native_mobile.py',
        ROOT / 'scripts/prepare_short_region_release.py',
        ROOT / 'training/evaluate_unified_regions.py', ROOT / 'training/train_unified_regions.py',
        ROOT / 'training/prepare_unified_regions.py', ROOT / 'src/flux_glyph/unified_font.py',
        ROOT / 'src/flux_glyph/region_font.py', run / 'SELECTION.json', run / 'model.pth',
        parity_path, model_path, metadata_path, r21_path, r22_path, *dev_files]
    bindings = {str(path.resolve()): sha(path) for path in source_files}
    freeze = {'schema': 'flux-glyph-unified-native-mobile-development-freeze-v1',
        'bindings': bindings, 'policies': [
            {**POLICY, 'baseline_version': 'r21-unified-font-v1-preview'},
            {**POLICY, 'baseline_version': 'r22-unified-font-retention-v1-preview'}],
        'calibration_checks_required': 53, 'development_checks_required': 36,
        'android_system_guard_policy': copy.deepcopy(parity['android_system_guard_policy']),
        'android_system_guard': copy.deepcopy(parity['android_system_guard']),
        'fixed_runtime': copy.deepcopy(parity['fixed_runtime']), 'test_read': False,
        'blind_test_performed': False, 'used_to_select_training_checkpoint': False}
    output.parent.mkdir(parents=True, exist_ok=True); write_new(freeze_path, freeze)
    freeze_sha = sha(freeze_path)
    development = load_split(data, 'development_holdout', allow_holdout=True)
    require(development['families'] == selection['families']
            and development['partition_sha256'] == partition_sha
            and len(development['rows']) == DEV_VIEWS and len(development['tiles']) == DEV_TILES,
            'Development population differs')
    logits = []; ratios = []
    for start in range(0, len(development['tiles']), 128):
        block = np.array(development['tiles'][start:start + 128], copy=True)
        font, size = model.session.run(['logits', 'log_em_ratio'], {'tiles': block})
        require(font.dtype == size.dtype == np.float32
                and font.shape == (len(block), 25) and size.shape == (len(block),)
                and np.isfinite(font).all() and np.isfinite(size).all() and (np.abs(size) <= 3).all(),
                'Invalid short preview DEV outputs')
        logits.append(font); ratios.append(size)
        if start == 0 or (start + len(block)) % 4096 < 128 or start + len(block) == DEV_TILES:
            print(json.dumps({'phase': 'development', 'tiles': start + len(block),
                              'total': DEV_TILES}), flush=True)
    outputs = region_outputs(np.concatenate(logits), np.concatenate(ratios), development['rows'],
                             parity['fixed_runtime']['temperature'])
    details = decisions(outputs, development['rows'], development['families'],
                        parity['fixed_runtime']['gates'])
    measured, by_view, confusion, denominators = summarize(
        details, development['families'], development['partition']['rejected'])
    r21 = comparison(measured, r21_baseline['metrics'], 'r21-unified-font-v1-preview')
    r22 = comparison(measured, r22_baseline['metrics'], 'r22-unified-font-retention-v1-preview')
    require(validate(run, data, plan) == selection and sha(freeze_path) == freeze_sha
            and all(sha(path) == digest for path, digest in bindings.items()),
            'Frozen short model or DEV evidence changed')
    report = build_development_report(selection, parity, measured, by_view, confusion,
        denominators, details, r21, r22, bindings, freeze_sha, partition_sha)
    write_new(output, report)
    print(json.dumps({'promotion_allowed': report['promotion_allowed'],
                      'r21_comparison_passed': r21['passed'], 'r22_comparison_passed': r22['passed'],
                      'failed_r21': [row for row in r21['checks'] if not row['passed']],
                      'failed_r22': [row for row in r22['checks'] if not row['passed']]}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--region', type=Path, required=True)
    parser.add_argument('--r21-baseline', type=Path, default=R21_BASELINE)
    parser.add_argument('--r22-baseline', type=Path, default=R22_BASELINE)
    parser.add_argument('--output', type=Path, required=True)
    evaluate(parser.parse_args())
