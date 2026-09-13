#!/usr/bin/env python3
"""Post-freeze development regression for the unified weight repair; never a blind test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'training')]
from train_regions import sha, dump, require
from train_unified_regions import region_outputs, decisions
from prepare_unified_regions import load_split
from evaluate_unified_regions import summarize
from flux_glyph.unified_font import UnifiedFontClassifier

# Chosen before the repair's training run. This source was evaluated for R21 and
# may have been seen by historical initialization: it is development evidence.
COMPARISON_POLICY = {
    'maximum_named_precision_drop': .005,
    'maximum_known_coverage_drop': .01,
    'maximum_unknown_withholding_drop': .02,
    'maximum_size_median_ape': .10,
    'maximum_size_p90_ape': .25,
    'minimum_size_coverage_of_correct_names': .70,
    'populations': ['all', 'ios', 'android'],
    'baseline_version': 'r21-unified-font-v1-preview',
    'is_blind_test': False,
    'used_to_select_training_checkpoint': False,
}


def comparison(current, baseline):
    checks = []
    groups = [('all', current, baseline)] + [
        (domain, current['per_domain'][domain], baseline['per_domain'][domain])
        for domain in ('ios', 'android')]
    for name, actual, old in groups:
        rules = [('named_precision', 'maximum_named_precision_drop'),
                 ('known_correct_coverage', 'maximum_known_coverage_drop'),
                 ('unknown_not_named_rate', 'maximum_unknown_withholding_drop')]
        for metric, tolerance in rules:
            reference, value = old.get(metric), actual.get(metric)
            require(reference is not None, 'Baseline population is missing: ' + name + '/' + metric)
            minimum = reference - COMPARISON_POLICY[tolerance]
            checks.append({'population': name, 'metric': metric, 'baseline': reference,
                           'minimum': minimum, 'actual': value,
                           'passed': value is not None and value + 1e-12 >= minimum})
        for metric, limit, direction in [('median_ape', .10, 'maximum'),
                                          ('p90_ape', .25, 'maximum'),
                                          ('coverage_of_correct_names', .70, 'minimum')]:
            value = actual['size'][metric]
            checks.append({'population': name, 'metric': 'size.' + metric,
                           direction: limit, 'actual': value,
                           'passed': value is not None and (value <= limit if direction == 'maximum' else value >= limit)})
    return {'passed': all(row['passed'] for row in checks), 'checks': checks, 'policy': COMPARISON_POLICY}


def evaluate(args):
    from export_unified_retention_confidence_floor import validate
    selection = validate(args.run, args.data)
    require(selection.get('promotion_allowed') is True, 'Repair has not passed its frozen CAL retention conditions')
    parity_path = args.run / 'PARITY.json'
    parity = json.loads(parity_path.read_text())
    require(parity.get('schema') == 'flux-glyph-unified-retention-onnx-parity-v1'
            and parity.get('passed') is True and parity.get('font_model_count') == 1
            and parity.get('selection_sha256') == sha(args.run / 'SELECTION.json')
            and parity.get('checkpoint_sha256') == sha(args.run / 'model.pth')
            and parity.get('model_sha256') == sha(args.region / 'model.onnx')
            and parity.get('metadata_sha256') == sha(args.region / 'metadata.json'),
            'Repair export provenance differs')
    model = UnifiedFontClassifier(args.region)
    require(model.output_families == selection['families'] and parity['output_family_count'] == len(model.output_families),
            'Repair family order differs')
    fixed = {'temperature': 1., 'gates': {'min_score': .7, 'min_margin': .01, 'min_patch_agreement': 2/3},
             'max_size_relative_spread': .2}
    require(all(model.meta[key] == value for key, value in fixed.items()), 'Inference gates changed during weight repair')
    baseline = json.loads(args.baseline.read_text())
    require(selection['bindings'].get(str(args.baseline.resolve())) == sha(args.baseline)
            and baseline.get('schema') == 'flux-glyph-unified-development-regression-v1'
            and baseline.get('blind_test_performed') is False
            and baseline['development_partition_sha256'] == sha(args.data / 'development_holdout/MANIFEST.json'),
            'R21 development comparison does not use the same partition')
    freeze_path = args.output.with_name(args.output.stem + '_FREEZE.json')
    require(not args.output.exists() and not freeze_path.exists(), 'Retain prior development evaluation evidence; do not silently repeat')
    bindings = {str(path.resolve()): sha(path) for path in [
        Path(__file__), args.run / 'SELECTION.json', args.run / 'model.pth', parity_path,
        args.region / 'model.onnx', args.region / 'metadata.json', args.baseline,
        ROOT / 'training/evaluate_unified_regions.py', ROOT / 'training/train_unified_regions.py',
        ROOT / 'training/prepare_unified_regions.py', ROOT / 'training/export_unified_retention_confidence_floor.py',
        ROOT / 'src/flux_glyph/region_font.py', ROOT / 'src/flux_glyph/unified_font.py',
        args.data / 'MANIFEST.json', args.data / 'development_holdout/MANIFEST.json']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with freeze_path.open('x') as stream:
        json.dump({'schema': 'flux-glyph-retention-development-freeze-v1', 'bindings': bindings,
                   'policy': COMPARISON_POLICY, 'fixed_runtime': fixed, 'test_read': False,
                   'blind_test_performed': False, 'used_to_select_training_checkpoint': False}, stream, indent=2)
        stream.write('\n')
    frozen_sha = sha(freeze_path)
    data = load_split(args.data, 'development_holdout', allow_holdout=True)
    require(data['families'] == selection['families'] and data['manifest_sha256'] == selection['data_manifest_sha256'],
            'Development data differs from the training registry')
    logits, ratios = [], []
    for start in range(0, len(data['tiles']), 128):
        block = np.array(data['tiles'][start:start+128], copy=True)
        family, size = model.session.run(['logits', 'log_em_ratio'], {'tiles': block})
        require(family.dtype == size.dtype == np.float32 and family.shape == (len(block), len(data['families']))
                and size.shape == (len(block),) and np.isfinite(family).all() and np.isfinite(size).all()
                and (np.abs(size) <= 3).all(), 'Invalid retention development model output')
        logits.append(family); ratios.append(size)
    outputs = region_outputs(np.concatenate(logits), np.concatenate(ratios), data['rows'], fixed['temperature'])
    details = decisions(outputs, data['rows'], data['families'], fixed['gates'])
    measured, by_view, confusion, denominators = summarize(details, data['families'], data['partition']['rejected'])
    compared = comparison(measured, baseline['metrics'])
    require(validate(args.run, args.data) == selection and sha(freeze_path) == frozen_sha
            and all(sha(path) == digest for path, digest in bindings.items()), 'Frozen model or source changed during development evaluation')
    report = {'schema': 'flux-glyph-retention-development-regression-v1', 'evaluation_kind': 'development_regression',
              'comparison_passed': compared['passed'], 'comparison': compared, 'metrics': measured,
              'by_domain': measured['per_domain'], 'by_family': measured['per_family'], 'by_view': by_view,
              'confusion': confusion, 'denominators': denominators, 'rows': details,
              'bindings': bindings, 'freeze_sha256': frozen_sha, 'model_sha256': parity['model_sha256'],
              'development_partition_sha256': data['partition_sha256'], 'fixed_runtime': fixed,
              'model_count': 1, 'test_read': False, 'blind_test_performed': False,
              'stable_validation_passed': False, 'test_passed': False, 'used_to_select_training_checkpoint': False,
              'scope': 'Repeated development regression on the same controlled native mobile regions and correlated views used for R21. Historical initialization may have seen images; neither a blind test nor a real-device accuracy estimate.'}
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'comparison_passed': compared['passed'], 'metrics': {key: measured[key] for key in
          ('views', 'named', 'correct_named', 'wrong_named', 'named_precision', 'known_correct_coverage', 'unknown_not_named_rate')},
          'failed_checks': [row for row in compared['checks'] if not row['passed']]}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'data', 'region', 'baseline', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    evaluate(parser.parse_args())
