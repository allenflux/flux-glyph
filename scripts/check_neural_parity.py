#!/usr/bin/env python3
"""Compare frozen PyTorch/ONNX font models on 128 fixed calibration glyphs.

Only the calibration partition is opened. Separate interpreters are supported
because the training and inference environments need not share Python versions:

  .venv-dev/bin/python scripts/check_neural_parity.py \
    --model-dir artifacts/ios-font-screenshots-v1/neural \
    --prepared artifacts/mobile-font-capture/ios-prepared-v1 \
    --torch-python /path/to/torch-env/bin/python \
    --ort-python .venv-dev/bin/python --output artifacts/parity.json

The check never selects weights, temperature or thresholds. A top-1 or frozen
gate decision mismatch fails. Probability error above 1e-4 also fails; raw logit
differences are reported rather than subjected to an overly strict allclose.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026091231
SAMPLES = 128
PROBABILITY_TOLERANCE = 1e-4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def state_sha(state):
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        result.update(key.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def local_file(root, name):
    require(isinstance(name, str), 'Asset path must be a string')
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), 'Asset escapes its directory or is missing')
    return path


def softmax(logits, indices, temperature):
    scores = logits[:, indices].astype(np.float64)
    scores = (scores - scores.max(axis=1, keepdims=True)) / temperature
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def sample_indices(rows, count=SAMPLES, seed=SEED):
    """Deterministic script/family stratification without replacement."""
    require(len(rows) >= count, f'Need at least {count} calibration glyphs')
    rng = np.random.default_rng(seed)
    cells = {}
    for i, row in enumerate(rows):
        require(row.get('script') in ('han', 'latin') and isinstance(row.get('family'), str), 'Invalid calibration script/family')
        cells.setdefault((row['script'], row['family']), []).append(i)
    queues = {key: list(rng.permutation(indices)) for key, indices in cells.items()}
    families = {script: sorted(key for key in cells if key[0] == script) for script in ('han', 'latin')}
    require(all(families.values()), 'Calibration must include both scripts')
    selected = []
    for script, number in [('han', count // 2), ('latin', count - count // 2)]:
        require(sum(len(queues[key]) for key in families[script]) >= number, f'Not enough {script} calibration glyphs')
        for i in range(number):
            candidates = families[script]
            start = i % len(candidates)
            for offset in range(len(candidates)):
                key = candidates[(start + offset) % len(candidates)]
                if queues[key]:
                    selected.append(int(queues[key].pop()))
                    break
    require(len(selected) == count and len(set(selected)) == count, 'Calibration sampling is incomplete or duplicated')
    return sorted(selected)


def calibration_batch(prepared, metadata):
    manifest_path = prepared / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get('schema') == 'flux-glyph-native-captured-glyphs-v1', 'Expected native captured data manifest')
    expected_manifest = metadata.get('training', {}).get('data_manifest_sha256')
    require(expected_manifest is None or sha(manifest_path) == expected_manifest,
            'Calibration data differs from the frozen training data manifest')
    # Deliberately access only this partition, never train/test array descriptors.
    partition = manifest['splits']['calibration']
    paths = {}
    for kind in ('metadata', 'glyphs'):
        descriptor = partition[kind]
        path = local_file(prepared, descriptor['path'])
        require(path.parent == (prepared / 'calibration').resolve(), 'Parity input must be the calibration partition')
        require(sha(path) == descriptor['sha256'], f'Calibration {kind} checksum differs')
        paths[kind] = path
    annotation = json.loads(paths['metadata'].read_text())
    require(annotation.get('schema') == 'flux-glyph-native-captured-metadata-v1'
            and annotation.get('split') == 'calibration', 'Expected calibration glyph metadata')
    rows = annotation['rows']
    require(len(rows) == partition['rows'], 'Calibration metadata count differs')
    x = np.load(paths['glyphs'], mmap_mode='r', allow_pickle=False)
    require(x.dtype == np.float32 and x.shape == (len(rows), 64, 64), 'Expected float32 calibration [N,64,64]')
    indices = sample_indices(rows)
    chosen = [rows[i] for i in indices]
    for i, row in zip(indices, chosen):
        require(row.get('array_row') == i and row['family'] in metadata['scripts'][row['script']],
                'Selected calibration row has a mismatched array row or model script mask')
        character = row.get('character')
        correct_script = (isinstance(character, str) and len(character) == 1 and
                          ('\u4e00' <= character <= '\u9fff' if row['script'] == 'han'
                           else character.isascii() and character.isalnum()))
        require(correct_script, 'Selected character/script differs')
    batch = np.ascontiguousarray(x[indices, None], dtype=np.float32)
    require(np.isfinite(batch).all() and batch.min() >= 0 and batch.max() <= 1, 'Invalid selected glyph pixels')
    for raster, row in zip(batch[:, 0], chosen):
        require(hashlib.sha256(raster.tobytes()).hexdigest() == row['glyph_sha256'], 'Selected glyph provenance SHA differs')
    audit = {'partition': 'calibration', 'seed': SEED, 'sample_count': len(indices),
             'array_rows': indices, 'script_counts': dict(Counter(row['script'] for row in chosen)),
             'family_counts': dict(Counter(row['family'] for row in chosen)),
             'manifest_sha256': sha(manifest_path), 'metadata_sha256': partition['metadata']['sha256'],
             'glyphs_sha256': partition['glyphs']['sha256'],
             'input_float32_sha256': hashlib.sha256(batch.tobytes()).hexdigest(),
             'test_partition_opened': False}
    return batch, chosen, audit


def torch_worker(args):
    import torch
    from torch import nn
    sys.path.insert(0, str(ROOT / 'training'))
    from network import FontClassifier
    metadata = json.loads((args.model_dir / 'metadata.json').read_text())
    checkpoint = torch.load(args.model_dir / 'model.pth', map_location='cpu', weights_only=True)
    names = checkpoint.get('families')
    require(isinstance(names, list) and len(names) == len(set(names)) and set(names) == set(metadata['families']),
            'Checkpoint and deployed model family labels differ')
    original_sha = state_sha(checkpoint['state_dict'])
    require(original_sha == metadata['training']['state_after_sha256'], 'Checkpoint state SHA differs from frozen metadata')
    head_rows = [names.index(name) for name in metadata['families']]
    model = FontClassifier()
    model.family_head = nn.Linear(model.family_head.in_features, len(head_rows))
    state = {key: value[head_rows].clone() if key.startswith('family_head.') else value
             for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state, strict=True)
    model.cpu().eval()
    torch.set_num_threads(4)
    x = np.load(args.input, allow_pickle=False)
    require(x.dtype == np.float32 and x.ndim == 4 and list(x.shape[1:]) == [1, 64, 64], 'Worker input is not float32 N16464')
    with torch.inference_mode():
        logits = model(torch.from_numpy(x)).cpu().numpy()
    np.save(args.logits, logits)
    write_json(args.worker_info, {'engine': 'pytorch', 'version': torch.__version__, 'device': 'cpu',
                                  'checkpoint_families': names, 'output_families': metadata['families'],
                                  'checkpoint_rows_in_output_order': head_rows, 'checkpoint_state_sha256': original_sha,
                                  'input_float32_sha256': hashlib.sha256(x.tobytes()).hexdigest()})


def runtime_decisions(classifier, logits, rows):
    """Use the deployed ranking and gate implementation for either engine."""
    result = [None] * len(rows)
    for script in ('han', 'latin'):
        positions = [i for i, row in enumerate(rows) if row['script'] == script]
        probabilities = softmax(logits[positions], classifier.indices[script], classifier.meta['temperature'][script])
        for index, probability in zip(positions, probabilities):
            candidates, score, margin = classifier._ranking(probability, script)
            reason = classifier._decision(score, margin, script)
            candidate = reason == 'neural_family_candidate'
            result[index] = {'top1': candidates[0]['family'], 'score': score, 'margin': margin,
                             'reason_code': reason, 'status': 'candidate' if candidate else 'uncertain',
                             'family': candidates[0]['family'] if candidate else None}
    return result


def ort_worker(args):
    import onnxruntime as ort
    sys.path.insert(0, str(ROOT / 'src'))
    from flux_glyph.neural_font import NeuralFontClassifier
    classifier = NeuralFontClassifier(args.model_dir)
    x = np.load(args.input, allow_pickle=False)
    rows = json.loads(args.samples.read_text())
    reference = np.load(args.reference_logits, allow_pickle=False)
    logits = classifier.session.run(['logits'], {'glyphs': x})[0]
    expected = (len(rows), len(classifier.families))
    require(logits.shape == expected and reference.shape == expected and logits.dtype == np.float32,
            'Worker logits have an invalid shape/dtype')
    require(np.isfinite(logits).all() and np.isfinite(reference).all(), 'Worker logits are nonfinite')
    np.save(args.logits, logits)
    write_json(args.worker_info, {'engine': 'onnxruntime', 'version': ort.__version__,
                                  'providers': classifier.session.get_providers(), 'output_families': classifier.families,
                                  'runtime_source_sha256': sha(ROOT / 'src/flux_glyph/neural_font.py'),
                                  'input_float32_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
                                  'torch_decisions': runtime_decisions(classifier, reference, rows),
                                  'ort_decisions': runtime_decisions(classifier, logits, rows)})


def error_summary(a, b):
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()), 'p99_abs': float(np.quantile(delta, .99))}


def compare_outputs(torch_logits, ort_logits, metadata, rows, torch_decisions, ort_decisions):
    require(torch_logits.shape == ort_logits.shape == (len(rows), len(metadata['families'])), 'Output shapes differ')
    require(np.isfinite(torch_logits).all() and np.isfinite(ort_logits).all(), 'Nonfinite model outputs')
    require(len(torch_decisions) == len(ort_decisions) == len(rows), 'Decision row counts differ')
    report = {'raw_logits': error_summary(torch_logits, ort_logits), 'scripts': {}, 'passed': True,
              'decision_level': 'individual preprocessed calibration glyph; deployed ranking and gate code',
              'criteria': {'script_top1_mismatches': 0, 'frozen_gate_decision_mismatches': 0,
                           'max_probability_abs_error': PROBABILITY_TOLERANCE,
                           'raw_logit_allclose_required': False}}
    for script in ('han', 'latin'):
        positions = [i for i, row in enumerate(rows) if row['script'] == script]
        require(positions, f'No {script} parity observations')
        columns = [i for i, family in enumerate(metadata['families']) if family in metadata['scripts'][script]]
        t = torch_logits[positions][:, columns]
        o = ort_logits[positions][:, columns]
        pt = softmax(torch_logits[positions], columns, metadata['temperature'][script])
        po = softmax(ort_logits[positions], columns, metadata['temperature'][script])
        probability_error = error_summary(pt, po)
        top1 = [i for i in positions if torch_decisions[i]['top1'] != ort_decisions[i]['top1']]
        gates = [i for i in positions if any(torch_decisions[i][key] != ort_decisions[i][key]
                                            for key in ('reason_code', 'status', 'family'))]
        mismatches = sorted(set(top1) | set(gates))
        passed = not mismatches and probability_error['max_abs'] <= PROBABILITY_TOLERANCE
        report['scripts'][script] = {'rows': len(positions), 'passed': passed,
                                      'centered_logits': error_summary(t - t.max(axis=1, keepdims=True), o - o.max(axis=1, keepdims=True)),
                                      'probabilities': probability_error, 'top1_mismatches': len(top1),
                                      'frozen_gate_decision_mismatches': len(gates),
                                      'torch_reason_counts': dict(Counter(torch_decisions[i]['reason_code'] for i in positions)),
                                      'ort_reason_counts': dict(Counter(ort_decisions[i]['reason_code'] for i in positions)),
                                      'mismatch_details': [{'sample_index': i, 'array_row': rows[i]['array_row'],
                                                            'character': rows[i]['character'], 'source_id': rows[i].get('source_id'),
                                                            'torch': torch_decisions[i], 'ort': ort_decisions[i]}
                                                           for i in mismatches]}
        report['passed'] = report['passed'] and passed
    return report


def run_worker(python, worker, args, temporary, reference=None):
    logits = temporary / (worker + '-logits.npy')
    info = temporary / (worker + '-info.json')
    command = [str(python), str(Path(__file__).resolve()), '--worker', worker, '--model-dir', str(args.model_dir.resolve()),
               '--input', str(temporary / 'input.npy'), '--samples', str(temporary / 'samples.json'),
               '--logits', str(logits), '--worker-info', str(info)]
    if reference:
        command += ['--reference-logits', str(reference)]
    process = subprocess.run(command, text=True, capture_output=True, timeout=180)
    require(process.returncode == 0, f'{worker} worker failed: {process.stderr[-4000:]}')
    return logits, json.loads(info.read_text())


def check(args):
    metadata_path = args.model_dir / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    require(metadata.get('schema') == 'flux-glyph-neural-font-v1' and metadata.get('algorithm') == 'glyph-cnn64-v1',
            'Unsupported neural model contract')
    selection_path = args.selection or args.model_dir.parent / 'SELECTION_FREEZE.json'
    require(selection_path.is_file() and sha(selection_path) == metadata['training']['selection_sha256'],
            'Frozen model selection is absent or its checksum differs')
    selection = json.loads(selection_path.read_text())
    require(selection.get('gates') == metadata.get('gates') and selection.get('temperatures') == metadata.get('temperature'),
            'Deployed thresholds/temperature differ from frozen selection')
    require(selection.get('state_after_sha256') == metadata['training']['state_after_sha256'],
            'Selected checkpoint state SHA differs from deployment metadata')
    onnx_path = local_file(args.model_dir, metadata['model']['path'])
    require(sha(onnx_path) == metadata['model']['sha256'], 'Exported ONNX checksum differs')
    assets = {'metadata': sha(metadata_path), 'checkpoint': sha(args.model_dir / 'model.pth'),
              'onnx': sha(onnx_path), 'selection': sha(selection_path)}
    batch, rows, sample_audit = calibration_batch(args.prepared.resolve(), metadata)
    with tempfile.TemporaryDirectory(prefix='flux-neural-parity-') as temp:
        temporary = Path(temp)
        np.save(temporary / 'input.npy', batch)
        write_json(temporary / 'samples.json', rows)
        torch_path, torch_info = run_worker(args.torch_python, 'torch', args, temporary)
        ort_path, ort_info = run_worker(args.ort_python, 'ort', args, temporary, torch_path)
        require(torch_info['input_float32_sha256'] == ort_info['input_float32_sha256'] == sample_audit['input_float32_sha256'],
                'The engines did not receive identical input pixels')
        result = compare_outputs(np.load(torch_path, allow_pickle=False), np.load(ort_path, allow_pickle=False), metadata, rows,
                                 ort_info.pop('torch_decisions'), ort_info.pop('ort_decisions'))
    require(assets == {'metadata': sha(metadata_path), 'checkpoint': sha(args.model_dir / 'model.pth'),
                       'onnx': sha(onnx_path), 'selection': sha(selection_path)}, 'Model or frozen selection changed during parity check')
    return {'schema': 'flux-glyph-neural-pytorch-onnx-parity-v1', 'status': 'passed' if result['passed'] else 'failed',
            'model_dir': str(args.model_dir.resolve()), 'model_asset_sha256': assets,
            'sample': sample_audit, 'families': metadata['families'], 'scripts': metadata['scripts'],
            'frozen_gates': metadata['gates'], 'frozen_temperature': metadata['temperature'],
            'engines': {'torch': torch_info, 'ort': ort_info}, 'comparison': result,
            'test_partition_opened': False, 'training_or_selection_performed': False,
            'scope': 'Export/runtime parity on fixed preprocessed calibration glyphs, not an accuracy estimate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--torch-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--ort-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', choices=['torch', 'ort'], help=argparse.SUPPRESS)
    for name in ('input', 'samples', 'logits', 'worker-info', 'reference-logits'):
        parser.add_argument('--' + name, type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        (torch_worker if args.worker == 'torch' else ort_worker)(args)
        return
    if not args.prepared or not args.output:
        parser.error('--prepared and --output are required')
    try:
        report = check(args)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        report = {'schema': 'flux-glyph-neural-pytorch-onnx-parity-v1', 'status': 'failed',
                  'error': str(error), 'test_partition_opened': False, 'training_or_selection_performed': False}
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
