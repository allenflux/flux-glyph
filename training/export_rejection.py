#!/usr/bin/env python3
"""Export the frozen rejection checkpoint and verify against original PyTorch GN."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'training')]
from train_rejection import BASE_CHECKPOINT_SHA, BASE_DATA_SHA, BASE_ONNX_SHA, POLICY, aggregate, annotate, load_gate, measures
from train_regions import RegionData, dump, require, sha, state_sha

LOGITS_ATOL = LOGITS_RTOL = 2e-4
PROBABILITY_ATOL, PROBABILITY_RTOL = 2e-5, 1e-4
SNAPSHOT_SHA = '6e9ce2c812be94e41e0ca492c7555c671a5bd9d6bed34bec83200076f2202a25'
AUGMENTED_LINEAGE = ('supplement_manifest_sha256', 'warm_start_rejection_sha256',
                     'warm_start_selection_sha256', 'strategy_source_sha256')


def validate_frozen_inputs(args, checkpoint, selection, primary):
    """Bind the selected model, both CAL domains, and saved training decisions."""
    bindings = {}
    def bind(name, path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or digest == expected, 'rejection export source changed: ' + name)
        bindings[name] = {'path': str(path), 'sha256': digest}
        return path
    bind('selection', args.run / 'SELECTION.json', checkpoint['selection_sha256'])
    bind('checkpoint', args.run / 'rejection.pth')
    require(selection.get('schema') == 'flux-glyph-region-rejection-training-v1'
            and selection.get('calibration_passed') is True and selection.get('test_read') is False,
            'calibration selection must pass before export without reading test')
    require(selection.get('policy') == POLICY and selection['policy'].get('test_used_for_selection') is False,
            'rejection export policy differs from the frozen training policy')
    require(checkpoint.get('labels') == selection['policy']['labels'] == ['unknown', 'known']
            and type(checkpoint.get('temperature')) in (int, float)
            and checkpoint['temperature'] == selection['policy']['temperature'] == 1.,
            'rejection checkpoint labels/temperature differ from the frozen policy')
    threshold = checkpoint.get('min_known_score')
    require(type(threshold) in (int, float) and math.isfinite(threshold) and 0 <= threshold <= 1
            and threshold == selection['selected']['min_known_score'], 'rejection checkpoint threshold differs from selection')
    require(type(checkpoint.get('selected_step')) is int and 0 < checkpoint['selected_step'] <= selection['optimizer_steps_executed']
            and checkpoint['selected_step'] == selection['selected']['step'], 'rejection checkpoint selected step differs')
    require(selection['parent_checkpoint_sha256'] == BASE_CHECKPOINT_SHA
            and selection['parent_onnx_sha256'] == BASE_ONNX_SHA
            and selection['anchor_manifest_sha256'] == BASE_DATA_SHA, 'rejection parent lineage differs')
    require(primary.get('schema') == 'flux-glyph-region-font-v1' and primary.get('algorithm') == 'region-cnn64x256-v1'
            and 'rejection' not in primary and primary['families'] == checkpoint['families']
            and primary['model']['sha256'] == BASE_ONNX_SHA, 'primary classifier metadata changed')
    model_name = primary['model']['path']
    require(isinstance(model_name, str) and Path(model_name).name == model_name and '\\' not in model_name
            and model_name.endswith('.onnx') and model_name != 'rejection.onnx', 'invalid or colliding primary model filename')
    bind('primary_metadata', args.primary / 'metadata.json')
    model_path = (args.primary / primary['model']['path']).resolve()
    require(model_path.is_relative_to(args.primary.resolve()), 'primary model escapes its directory')
    bind('primary_onnx', model_path, BASE_ONNX_SHA)
    bind('gate_data_manifest', args.data / 'MANIFEST.json', selection['data_manifest_sha256'])
    bind('anchor_data_manifest', args.anchor / 'MANIFEST.json', selection['anchor_manifest_sha256'])
    bind('preprocessing_snapshot', args.snapshot, SNAPSHOT_SHA)
    allowed_trainers = [ROOT / 'training/train_rejection.py', ROOT / 'training/train_rejection_e2e.py',
                        ROOT / 'training/train_rejection_augmented.py']
    matching_trainers = [path for path in allowed_trainers if path.is_file() and sha(path) == selection['source_sha256']]
    require(len(matching_trainers) == 1, 'rejection export source changed: training_source must match one approved trainer')
    bind('training_source', matching_trainers[0], selection['source_sha256'])
    bind('network_source', ROOT / 'training/rejection_network.py', selection['network_source_sha256'])
    has_augmented_lineage = any(key in selection for key in AUGMENTED_LINEAGE)
    require(has_augmented_lineage or matching_trainers[0].name != 'train_rejection_augmented.py',
            'augmented training requires complete supplement and warm-start lineage')
    supplement, resume = getattr(args, 'supplement', None), getattr(args, 'resume_rejection', None)
    if has_augmented_lineage:
        require(all(type(selection.get(key)) is str and len(selection[key]) == 64
                    and all(char in '0123456789abcdef' for char in selection[key]) for key in AUGMENTED_LINEAGE),
                'augmented training requires complete SHA256 supplement and warm-start lineage')
        require(supplement is not None and resume is not None,
                'augmented export requires --supplement and --resume-rejection')
        bind('supplement_manifest', Path(supplement) / 'MANIFEST.json', selection['supplement_manifest_sha256'])
        bind('warm_start_rejection', resume, selection['warm_start_rejection_sha256'])
        warm_selection_path = bind('warm_start_selection', Path(resume).parent / 'SELECTION.json',
                                   selection['warm_start_selection_sha256'])
        bind('strategy_source', ROOT / 'training/train_rejection.py', selection['strategy_source_sha256'])
        warm_selection = json.loads(warm_selection_path.read_text())
        require(warm_selection.get('schema') == 'flux-glyph-region-rejection-training-v1'
                and warm_selection.get('test_read') is False and warm_selection.get('policy') == POLICY
                and all(warm_selection.get(key) == selection[key] for key in
                        ('parent_checkpoint_sha256', 'parent_onnx_sha256', 'data_manifest_sha256',
                         'anchor_manifest_sha256', 'network_source_sha256')),
                'warm-start selection has different policy, parent, data, network or test history')
    else:
        require(supplement is None and resume is None, 'extra augmented inputs are not bound by this selection')
    decisions_path = bind('calibration_decisions', args.run / 'CALIBRATION_DECISIONS.json', selection['calibration_decisions_sha256'])
    decisions = json.loads(decisions_path.read_text())
    require(decisions.get('schema') == 'flux-glyph-rejection-calibration-decisions-v1'
            and decisions.get('min_known_score') == threshold and isinstance(decisions.get('records'), list),
            'invalid frozen calibration decisions')
    return bindings, decisions


def calibration_records(decisions, rows, threshold):
    require(decisions['min_known_score'] == threshold and len(decisions['records']) == len(rows) > 0,
            'frozen calibration decision count or threshold differs')
    scores, seen = [], set()
    for record, row in zip(decisions['records'], rows):
        identity = tuple(row[key] for key in ('domain', 'source_id', 'region_id'))
        require(identity[0] in ('new_native', 'r17_native') and identity not in seen
                and identity == tuple(record.get(key) for key in ('domain', 'source_id', 'region_id')),
                'frozen calibration row identity/order differs')
        seen.add(identity)
        score = record.get('known_score')
        require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1
                and type(record.get('passed')) is bool and record['passed'] == (score >= threshold),
                'invalid frozen calibration score or decision')
        scores.append(score)
    return np.asarray(scores, dtype=np.float64)


def probabilities(logits):
    logits = np.asarray(logits, dtype=np.float64)
    require(logits.ndim == 2 and logits.shape[0] > 0 and logits.shape[1] == 2 and np.isfinite(logits).all(),
            'nonfinite or invalid rejection logits')
    values = np.exp(logits - logits.max(1, keepdims=True))
    return values / values.sum(1, keepdims=True)


def compare_logits(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, 'rejection logit shapes differ')
    actual_probability, expected_probability = probabilities(actual), probabilities(expected)
    np.testing.assert_allclose(actual, expected, atol=LOGITS_ATOL, rtol=LOGITS_RTOL, equal_nan=False)
    np.testing.assert_allclose(actual_probability, expected_probability,
                               atol=PROBABILITY_ATOL, rtol=PROBABILITY_RTOL, equal_nan=False)
    require(np.array_equal(actual.argmax(1), expected.argmax(1)), 'export changed binary winner')
    return {'max_logit_absolute_error': float(np.abs(actual.astype(np.float64) - expected).max()),
            'max_probability_absolute_error': float(np.abs(actual_probability - expected_probability).max()), 'passed': True}


def compare_calibration_scores(actual, expected, frozen, threshold):
    arrays = [np.asarray(values, dtype=np.float64) for values in (actual, expected, frozen)]
    require(all(values.ndim == 1 and values.shape == arrays[0].shape and values.size > 0
                and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all() for values in arrays),
            'invalid or nonfinite full calibration scores')
    actual, expected, frozen = arrays
    for left, right in ((actual, expected), (expected, frozen), (actual, frozen)):
        np.testing.assert_allclose(left, right, atol=PROBABILITY_ATOL, rtol=PROBABILITY_RTOL, equal_nan=False)
        require(np.array_equal(left >= threshold, right >= threshold), 'export changed frozen calibrated region rejection')
    return {'onnx_vs_torch_max_absolute_error': float(np.abs(actual - expected).max()),
            'torch_vs_training_max_absolute_error': float(np.abs(expected - frozen).max()),
            'onnx_vs_training_max_absolute_error': float(np.abs(actual - frozen).max()),
            'decisions_identical': True, 'passed': True}


def main(args):
    import onnx
    import onnxruntime as ort
    import torch
    from export_region_stable import replace_groupnorm
    from rejection_network import RegionRejector
    require(not args.output.exists(), 'export output must be new')
    require(not (args.run / 'PARITY.json').exists(), 'retain the existing parity report')
    checkpoint = torch.load(args.run / 'rejection.pth', map_location='cpu', weights_only=True)
    selection = json.loads((args.run / 'SELECTION.json').read_text())
    primary = json.loads((args.primary / 'metadata.json').read_text())
    bindings, decisions = validate_frozen_inputs(args, checkpoint, selection, primary)
    threshold = checkpoint['min_known_score']
    anchor = RegionData(args.anchor, preprocessing_snapshot=args.snapshot)
    partitions = [('new_native', annotate(load_gate(args.data, 'calibration'), 'new_native')),
                  ('r17_native', annotate(anchor.load('calibration'), 'r17_native', known=True))]
    cal_rows = [row for _, part in partitions for row in part['rows']]
    require(len(cal_rows) == selection['partitions']['calibration']['regions']
            and sum(len(part['tiles']) for _, part in partitions) == selection['partitions']['calibration']['tiles'],
            'full training calibration partitions differ')
    frozen_scores = calibration_records(decisions, cal_rows, threshold)
    require(measures(frozen_scores, cal_rows, threshold) == selection['selected']['metrics'],
            'saved training decisions differ from selected calibration metrics')
    source_code = {str(path.relative_to(ROOT)): sha(path) for path in
                   (Path(__file__), ROOT / 'training/export_region_stable.py', ROOT / 'training/rejection_network.py',
                    ROOT / 'training/region_network.py', ROOT / 'training/network.py',
                    ROOT / 'training/train_rejection.py', Path(bindings['training_source']['path']),
                    ROOT / 'src/flux_glyph/region_font.py')}
    reference = RegionRejector(len(checkpoint['families'])).eval()
    reference.load_state_dict(checkpoint['state_dict'], strict=True)
    exported = copy.deepcopy(reference)
    count = replace_groupnorm(exported, high_precision=True)
    require(count == 4 and state_sha(reference.state_dict()) == state_sha(exported.state_dict()), 'export changed parameters')
    args.output.mkdir(parents=True)
    output = args.output / 'rejection.onnx'
    torch.set_num_threads(4)
    torch.onnx.export(exported, torch.zeros(2, 1, 64, 256), output, input_names=['tiles'], output_names=['known_logits'],
                      dynamic_axes={'tiles': {0: 'batch'}, 'known_logits': {0: 'batch'}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output), sess_options=options, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(20260912)
    sampled = []
    sample_counts = {}
    for domain, data in partitions:
        indices = np.unique(np.rint(np.linspace(0, len(data['tiles']) - 1, min(512, len(data['tiles'])))).astype(int))
        sampled.append(np.asarray(data['tiles'][indices]))
        sample_counts[domain] = len(indices)
    tiles = np.concatenate([*sampled, rng.random((16, 1, 64, 256), dtype=np.float32)])
    reference_logits = []
    with torch.inference_mode():
        for start in range(0, len(tiles), 32):
            reference_logits.append(reference(torch.from_numpy(tiles[start:start + 32])).numpy())
    expected = np.concatenate(reference_logits)
    comparisons = []
    for batch in (1, 7, 32, 128):
        actual = np.concatenate([session.run(['known_logits'], {'tiles': tiles[start:start + batch]})[0]
                                 for start in range(0, len(tiles), batch)])
        comparisons.append({'batch': batch, **compare_logits(actual, expected)})
    full_calibration, cursor = [], 0
    for domain, data in partitions:
        cal_expected, cal_actual = [], []
        with torch.inference_mode():
            for start in range(0, len(data['tiles']), 64):
                block = np.array(data['tiles'][start:start + 64], copy=True)
                cal_expected.append(reference(torch.from_numpy(block)).numpy())
                cal_actual.append(session.run(['known_logits'], {'tiles': block})[0])
        expected_logits, actual_logits = np.concatenate(cal_expected), np.concatenate(cal_actual)
        numeric = compare_logits(actual_logits, expected_logits)
        expected_scores = aggregate(expected_logits, data['rows'])
        actual_scores = aggregate(actual_logits, data['rows'])
        count_rows = len(data['rows'])
        frozen = frozen_scores[cursor:cursor + count_rows]
        matching = compare_calibration_scores(actual_scores, expected_scores, frozen, threshold)
        full_calibration.append({'domain': domain, 'regions': count_rows, 'tiles': len(data['tiles']),
                                 'logits': numeric, 'scores': matching})
        cursor += count_rows
    require(cursor == len(frozen_scores) and set(anchor.loaded) == {'calibration'}, 'calibration coverage differs or another partition was opened')
    require(all(sha(item['path']) == item['sha256'] for item in bindings.values()), 'bound export inputs changed during verification')
    require(all(sha(ROOT / name) == expected for name, expected in source_code.items()), 'export/architecture source changed during verification')
    parity = {'schema': 'flux-glyph-rejection-onnx-parity-v1', 'passed': True, 'reference': 'original torch.nn.GroupNorm',
              'logits_atol': LOGITS_ATOL, 'logits_rtol': LOGITS_RTOL,
              'probability_atol': PROBABILITY_ATOL, 'probability_rtol': PROBABILITY_RTOL,
              'batches': comparisons, 'samples': len(tiles), 'sample_counts': sample_counts,
              'calibration_regions': len(cal_rows), 'full_calibration': full_calibration,
              'training_calibration_decisions_identical': True,
              'calibration_decisions_sha256': bindings['calibration_decisions']['sha256'],
              'source_bindings': bindings, 'code_sha256': source_code,
              'calibration_decisions_identical': True, 'test_read': False, 'model_sha256': sha(output),
              'checkpoint_sha256': sha(args.run / 'rejection.pth'), 'export_source_sha256': sha(__file__)}
    dump(args.run / 'PARITY.json', parity)
    shutil.copyfile(args.primary / primary['model']['path'], args.output / primary['model']['path'])
    primary['algorithm'] = 'region-cnn64x256-rejection-v2'
    primary['rejection'] = {
        'schema': 'flux-glyph-region-rejection-v1', 'algorithm': 'region-known-unknown-cnn64x256-v1',
        'model': {'path': 'rejection.onnx', 'sha256': sha(output)}, 'labels': ['unknown', 'known'],
        'known_families': checkpoint['families'], 'base_model_sha256': BASE_ONNX_SHA,
        'aggregation': 'mean_softmax_known_probability', 'temperature': checkpoint['temperature'], 'min_known_score': threshold,
        'training': {'selection_sha256': sha(args.run / 'SELECTION.json'), 'checkpoint_sha256': sha(args.run / 'rejection.pth'),
                     'optimizer_steps_executed': selection['optimizer_steps_executed'], 'selected_step': checkpoint['selected_step'],
                     'data_manifest_sha256': selection['data_manifest_sha256'], 'source_kind': 'ios_simulator_screenshot',
                     'anchor_manifest_sha256': selection['anchor_manifest_sha256'],
                     'calibration_decisions_sha256': selection['calibration_decisions_sha256'],
                     'primary_font_size_parameters_changed': False, 'unknown_test_read_for_selection': False},
        'scope': 'Learned unknown-font rejection from native controlled scenes; not universal unknown-font detection or Android device validation.'}
    primary['rejection']['training'].update({key: selection[key] for key in AUGMENTED_LINEAGE if key in selection})
    primary['scope'] = 'Image-only font/size CNN with independent learned known/unknown rejection; no OCR text input.'
    dump(args.output / 'metadata.json', primary)
    print(json.dumps({'output': str(args.output), 'parity': parity}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--anchor', required=True, type=Path)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--primary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--supplement', type=Path, help='Train-only supplement directory, required for augmented lineage')
    parser.add_argument('--resume-rejection', type=Path, help='Warm-start checkpoint, required for augmented lineage')
    main(parser.parse_args())
