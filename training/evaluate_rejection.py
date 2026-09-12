#!/usr/bin/env python3
"""One-time fixed test evaluation after checkpoint, threshold and export are frozen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'training')]
from flux_glyph.region_font import RegionFontClassifier, aggregate_predictions
from train_rejection import POLICY, SYSTEM, aggregate, annotate, load_gate, measures
from train_regions import RegionData, dump, require, sha


def evaluate(args):
    require(not args.output.exists(), 'test report already exists; retain its frozen results')
    selection = json.loads((args.run / 'SELECTION.json').read_text())
    parity = json.loads((args.run / 'PARITY.json').read_text())
    require(selection['calibration_passed'] and parity['passed'] and selection['test_read'] is False, 'selection/export must pass first')
    require(sha(args.data / 'MANIFEST.json') == selection['data_manifest_sha256']
            and sha(args.anchor / 'MANIFEST.json') == selection['anchor_manifest_sha256'], 'test data differs from frozen selection')
    policy = selection['policy']
    require(policy == POLICY, 'evaluation policy differs from frozen selection')
    classifier = RegionFontClassifier(args.region)
    config = classifier.meta['rejection']
    require(config['training']['selection_sha256'] == sha(args.run / 'SELECTION.json')
            and config['model']['sha256'] == parity['model_sha256'], 'frozen evaluation provenance changed')
    threshold = config['min_known_score']
    require(threshold == selection['selected']['min_known_score'], 'test threshold differs from calibration')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    rejector = ort.InferenceSession(str(args.region / config['model']['path']), sess_options=options, providers=['CPUExecutionProvider'])
    primary = ort.InferenceSession(str(args.region / classifier.meta['model']['path']), sess_options=options, providers=['CPUExecutionProvider'])
    native = annotate(load_gate(args.data, 'test', allow_test=True), 'new_native')
    anchor = annotate(RegionData(args.anchor, preprocessing_snapshot=args.snapshot).load('test'), 'r17_native', known=True)
    test_unknown = {r['family'] for r in native['rows'] if r['label'] == 0}
    require(not test_unknown.intersection(selection['unknown_train_families'] + selection['unknown_calibration_families']),
            'test unknown font families were used to train or calibrate')
    scores, rows, details = [], [], []
    for data in (native, anchor):
        gate_logits, family_logits, size_values = [], [], []
        for start in range(0, len(data['tiles']), 64):
            block = np.array(data['tiles'][start:start + 64], copy=True)
            gate_logits.append(rejector.run(['known_logits'], {'tiles': block})[0])
            family, sizes = primary.run(['logits', 'log_em_ratio'], {'tiles': block})
            family_logits.append(family)
            size_values.append(sizes)
        current = aggregate(np.concatenate(gate_logits), data['rows'])
        names, sizes = np.concatenate(family_logits), np.concatenate(size_values)
        require(np.isfinite(names).all() and np.isfinite(sizes).all() and np.all(np.abs(sizes) <= 3),
                'primary network returned invalid outputs; do not score them as valid font predictions')
        scores.extend(current.tolist())
        rows.extend(data['rows'])
        for index, row in enumerate(data['rows']):
            start, count = row['tile_start'], row['tile_count']
            result = aggregate_predictions(names[start:start + count], sizes[start:start + count], temperature=classifier.meta['temperature'])
            predicted = classifier.families[result['order'][0]]
            gates = classifier.meta['gates']
            accepted = (result['score'] >= gates['min_score'] and result['margin'] > 1e-8
                        and result['margin'] >= gates['min_margin'] and result['patch_agreement'] >= gates['min_patch_agreement'])
            details.append({'domain': row['domain'], 'source_id': row['source_id'], 'region_id': row['region_id'],
                            'family': row['family'], 'known_truth': row['label'] == 1,
                            'known_score': float(current[index]), 'rejected': bool(current[index] < threshold),
                            'primary_top1': predicted, 'primary_score': result['score'], 'primary_accepted': bool(accepted),
                            'primary_top1_correct': predicted == row['family'] if row['label'] == 1 else False})
        print(json.dumps({'evaluated_domain': data['rows'][0]['domain'], 'regions': len(data['rows'])}), flush=True)
    metrics = measures(np.asarray(scores), rows, threshold)
    checks = {'unknown_recall': metrics['unknown_recall'] >= policy['minimum_unknown_recall_test'],
              'known_retention_by_domain': all(m['known_false_rejection_rate'] <= policy['test_max_known_false_rejection_rate_per_domain']
                                               for m in metrics['by_domain'].values() if m['known']),
              'system_retention': metrics['system']['known_false_rejection_rate'] <= policy['test_max_system_false_rejection_rate'],
              'primary_onnx_unchanged': sha(args.region / classifier.meta['model']['path']) == config['base_model_sha256'],
              'unknown_families_disjoint': True}
    comparison = {}
    for domain in ('new_native', 'r17_native', 'system'):
        if domain == 'system':
            values = [row for row in details if row['family'] in SYSTEM and row['known_truth']]
        else:
            values = [row for row in details if row['domain'] == domain]
        comparison[domain] = {}
        for name, filtered in (('before', values), ('after', [row for row in values if not row['rejected']])):
            comparison[domain][name] = {
                'known_top_candidate_shown': sum(row['known_truth'] for row in filtered),
                'known_accepted_correct': sum(row['known_truth'] and row['primary_accepted'] and row['primary_top1_correct'] for row in filtered),
                'known_accepted_wrong': sum(row['known_truth'] and row['primary_accepted'] and not row['primary_top1_correct'] for row in filtered),
                'unknown_top_candidate_shown': sum(not row['known_truth'] for row in filtered),
                'unknown_accepted_as_known': sum(not row['known_truth'] and row['primary_accepted'] for row in filtered)}
    report = {'schema': 'flux-glyph-rejection-test-v1', 'passed': all(checks.values()), 'checks': checks, 'metrics': metrics,
              'comparison': comparison, 'rows': details, 'threshold': threshold, 'selection_sha256': sha(args.run / 'SELECTION.json'),
              'parity_sha256': sha(args.run / 'PARITY.json'), 'data_manifest_sha256': sha(args.data / 'MANIFEST.json'),
              'region_metadata_sha256': sha(args.region / 'metadata.json'), 'source_sha256': sha(__file__),
              'test_unknown_families': sorted(test_unknown), 'test_used_for_training_or_threshold': False,
              'scope': 'Native labeled region crops; new held-out unknown families in controlled iOS scenes plus reused R17 known regression. Not a detector or Android-native evaluation.'}
    dump(args.output, report)
    print(json.dumps({'passed': report['passed'], 'checks': checks, 'metrics': metrics, 'comparison': comparison}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'data', 'anchor', 'snapshot', 'region', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    evaluate(parser.parse_args())
