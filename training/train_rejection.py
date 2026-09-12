#!/usr/bin/env python3
"""Train a local image-only rejection head with fixed known-font preservation limits.

Only train/calibration partitions are opened. Unknown font families in calibration
must be disjoint from training; the test split is consumed by a separate evaluator.
The R17 primary font/size model is never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'training')]
from train_regions import RegionData, dump, require, sha, state_sha, verify_preprocessing_snapshot

BASE_CHECKPOINT_SHA = 'aeda192187c90fa3e59afc23d879d8eda8d8f9fd987895bfd0d0d120c884649a'
BASE_ONNX_SHA = 'bff569708afbb127222885539914942f9d958f46c8c11bfb9ace80784f70916b'
BASE_DATA_SHA = 'ed75b831c6132284ad8e908f6a34723b0e0c1380a8a18be62dfbeefffdd35022'
POLICY = {
    'labels': ['unknown', 'known'], 'temperature': 1.0,
    'threshold_numeric_margin': 2e-5,
    'max_known_false_rejection_rate_per_domain': .01,
    'max_known_false_rejection_rate_per_family': .02,
    'max_system_false_rejection_rate': .01,
    'minimum_unknown_recall_calibration': .8,
    'minimum_unknown_recall_test': .8,
    'test_max_known_false_rejection_rate_per_domain': .02,
    'test_max_system_false_rejection_rate': .02,
    'selection': 'highest calibration unknown recall within known limits; then lowest known rejection; then earliest checkpoint',
    'test_used_for_selection': False,
}
SYSTEM = {'PingFang', 'SF Pro', 'Helvetica'}


def load_gate(directory, split, *, allow_test=False):
    from training.capture.prepare_unknown_regions import load_partition
    result = load_partition(directory, split, allow_test=allow_test)
    require(all(row.get('whole_width_covered') is True for row in result['rows']),
            'incomplete region coverage in rejection data')
    return result


def annotate(data, domain, known=False):
    return {**data, 'rows': [{**row, 'domain': domain, 'label': 1 if known else row['label']} for row in data['rows']]}


def aggregate(logits, rows):
    values = np.asarray(logits, dtype=np.float64)
    values -= values.max(1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(1, keepdims=True)
    return np.asarray([probabilities[row['tile_start']:row['tile_start'] + row['tile_count'], 1].mean() for row in rows])


def measures(scores, rows, threshold):
    def measure(indices):
        known = [i for i in indices if rows[i]['label'] == 1]
        unknown = [i for i in indices if rows[i]['label'] == 0]
        rejected_known = int(sum(scores[i] < threshold for i in known))
        rejected_unknown = int(sum(scores[i] < threshold for i in unknown))
        return {'known': len(known), 'known_rejected': int(rejected_known),
                'known_false_rejection_rate': rejected_known / len(known) if known else None,
                'unknown': len(unknown), 'unknown_rejected': int(rejected_unknown),
                'unknown_recall': rejected_unknown / len(unknown) if unknown else None}
    result = measure(range(len(rows)))
    result['by_domain'] = {domain: measure([i for i, row in enumerate(rows) if row['domain'] == domain])
                           for domain in sorted({row['domain'] for row in rows})}
    result['by_family'] = {family: measure([i for i, row in enumerate(rows) if row['family'] == family])
                           for family in sorted({row['family'] for row in rows})}
    result['system'] = measure([i for i, row in enumerate(rows) if row['label'] == 1 and row['family'] in SYSTEM])
    return result


def calibrate(scores, rows):
    # Reject when score < threshold. Taking the (allowed+1)-th lowest score
    # respects the bound even with ties; no evaluation labels enter this choice.
    bounds = []
    groups = [([i for i, r in enumerate(rows) if r['label'] == 1 and r['domain'] == domain], .01)
              for domain in sorted({r['domain'] for r in rows})]
    groups += [([i for i, r in enumerate(rows) if r['label'] == 1 and r['family'] == family], .02)
               for family in sorted({r['family'] for r in rows if r['label'] == 1})]
    groups += [([i for i, r in enumerate(rows) if r['label'] == 1 and r['family'] in SYSTEM], .01)]
    for indices, rate in groups:
        if indices:
            bounds.append(float(np.sort(scores[indices])[int(len(indices) * rate)]))
    require(bounds, 'calibration has no known families')
    threshold = max(0., min(bounds) - POLICY['threshold_numeric_margin'])
    return threshold, measures(scores, rows, threshold)


def extract(model, data, path, device):
    import torch
    model.eval().to(device)
    features = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(data['tiles']), 128))
    with torch.inference_mode():
        for start in range(0, len(features), 128):
            tensor = torch.from_numpy(np.array(data['tiles'][start:start + 128], copy=True)).to(device)
            features[start:start + len(tensor)] = model.features(tensor).cpu().numpy()
    features.flush()
    return features


def combine_features(parts):
    features, rows, cursor = [], [], 0
    for part, values in parts:
        features.append(values)
        rows.extend({**row, 'tile_start': row['tile_start'] + cursor} for row in part['rows'])
        cursor += len(values)
    return np.concatenate(features), rows


def main(args):
    import torch
    from rejection_network import RegionRejector
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    require(not args.output.exists(), 'use a new training directory')
    require(sha(args.checkpoint) == BASE_CHECKPOINT_SHA, 'primary parent checkpoint changed')
    require(sha(args.anchor / 'MANIFEST.json') == BASE_DATA_SHA, 'immutable R17 data manifest changed')
    verify_preprocessing_snapshot('6e9ce2c812be94e41e0ca492c7555c671a5bd9d6bed34bec83200076f2202a25', args.snapshot)
    args.output.mkdir(parents=True)
    dump(args.output / 'POLICY.json', {**POLICY, 'seed': args.seed, 'steps': args.steps,
                                      'evaluation_interval': 250, 'batch_regions': 256,
                                      'gate_data_sha256': sha(args.data / 'MANIFEST.json')})
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = RegionRejector(len(checkpoint['families']))
    model.initialize_encoder(checkpoint['state_dict'])
    encoder_before = state_sha({k: v for k, v in model.state_dict().items() if k.startswith(('trunk.', 'pool.', 'style.'))})
    anchor = RegionData(args.anchor, preprocessing_snapshot=args.snapshot)
    partitions = {}
    for split in ('train', 'calibration'):
        native = annotate(load_gate(args.data, split), 'new_native')
        original = annotate(anchor.load(split), 'r17_native', known=True)
        parts = []
        for name, data in (('gate', native), ('r17', original)):
            features = extract(model, data, args.output / f'{name}_{split}_features.npy', args.device)
            parts.append((data, features))
            print(json.dumps({'extracted': name, 'split': split, 'tiles': len(features), 'regions': len(data['rows'])}), flush=True)
        partitions[split] = combine_features(parts)
    train_x, train_rows = partitions['train']
    cal_x, cal_rows = partitions['calibration']
    train_unknown = {r['family'] for r in train_rows if r['label'] == 0}
    cal_unknown = {r['family'] for r in cal_rows if r['label'] == 0}
    require(train_unknown and cal_unknown and not train_unknown & cal_unknown, 'unknown families cross train/calibration')
    require(all(r['family'] not in checkpoint['families'] for r in train_rows + cal_rows if r['label'] == 0),
            'known family incorrectly labeled unknown')
    model.cpu()
    model.feature_mean.copy_(torch.from_numpy(train_x.mean(0)))
    model.feature_scale.copy_(torch.from_numpy(np.maximum(train_x.std(0), .05)))
    train_tensor = torch.from_numpy(train_x)
    cal_tensor = torch.from_numpy(cal_x)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=.001, weight_decay=.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=.0001)
    pools = {}
    for i, row in enumerate(train_rows):
        pools.setdefault((row['label'], row['domain'], row['family']), []).append(i)
    keys = {label: [key for key in pools if key[0] == label] for label in (0, 1)}
    best, history, started = None, [], time.monotonic()
    for step in range(1, args.steps + 1):
        chosen = []
        for label in (0, 1):
            for _ in range(128):
                key = keys[label][rng.integers(len(keys[label]))]
                index = int(rng.choice(pools[key]))
                row = train_rows[index]
                tile = row['tile_start'] + int(rng.integers(row['tile_count']))
                chosen.append((tile, label))
        model.train()
        logits = model.classify_features(train_tensor[[item[0] for item in chosen]])
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([item[1] for item in chosen]), label_smoothing=.02)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 250 == 0 or step == args.steps:
            model.eval()
            with torch.inference_mode():
                scores = aggregate(model.classify_features(cal_tensor).numpy(), cal_rows)
            threshold, metrics = calibrate(scores, cal_rows)
            record = {'step': step, 'loss': float(loss.detach()), 'min_known_score': threshold, 'metrics': metrics}
            history.append(record)
            key = (metrics['unknown_recall'], -metrics['known_rejected'], -step)
            if best is None or key > best[0]:
                best = (key, copy.deepcopy(model.state_dict()), record)
            print(json.dumps({'step': step, 'loss': float(loss.detach()), 'threshold': threshold,
                              'unknown_recall': metrics['unknown_recall'], 'known_rejected': metrics['known_rejected'],
                              'seconds': round(time.monotonic() - started, 2)}), flush=True)
    model.load_state_dict(best[1])
    encoder_after = state_sha({k: v for k, v in model.state_dict().items() if k.startswith(('trunk.', 'pool.', 'style.'))})
    require(encoder_before == encoder_after, 'frozen encoder changed')
    model.eval()
    with torch.inference_mode():
        final_scores = aggregate(model.classify_features(cal_tensor).numpy(), cal_rows)
    decisions = {'schema': 'flux-glyph-rejection-calibration-decisions-v1',
                 'min_known_score': best[2]['min_known_score'],
                 'records': [{'domain': row['domain'], 'source_id': row['source_id'], 'region_id': row['region_id'],
                              'known_score': float(score), 'passed': bool(score >= best[2]['min_known_score'])}
                             for row, score in zip(cal_rows, final_scores)]}
    dump(args.output / 'CALIBRATION_DECISIONS.json', decisions)
    passed = best[2]['metrics']['unknown_recall'] >= POLICY['minimum_unknown_recall_calibration']
    selected = {'schema': 'flux-glyph-region-rejection-training-v1', 'policy': POLICY, 'selected': best[2], 'history': history,
                'calibration_passed': passed, 'optimizer_steps_executed': args.steps, 'training_device': args.device,
                'training_scope': 'local frozen R17 image encoder plus learned binary MLP; primary font/size ONNX unchanged',
                'data_manifest_sha256': sha(args.data / 'MANIFEST.json'), 'anchor_manifest_sha256': BASE_DATA_SHA,
                'calibration_decisions_sha256': sha(args.output / 'CALIBRATION_DECISIONS.json'),
                'parent_checkpoint_sha256': BASE_CHECKPOINT_SHA, 'parent_onnx_sha256': BASE_ONNX_SHA,
                'encoder_before_sha256': encoder_before, 'encoder_after_sha256': encoder_after,
                'unknown_train_families': sorted(train_unknown), 'unknown_calibration_families': sorted(cal_unknown),
                'test_read': False, 'source_sha256': sha(__file__), 'network_source_sha256': sha(ROOT / 'training/rejection_network.py'),
                'partitions': {s: {'regions': len(rows), 'tiles': len(x), 'labels': dict(Counter(r['label'] for r in rows))}
                               for s, (x, rows) in partitions.items()}}
    dump(args.output / 'SELECTION.json', selected)
    torch.save({'state_dict': model.state_dict(), 'families': checkpoint['families'], 'labels': ['unknown', 'known'],
                'selected_step': best[2]['step'], 'min_known_score': best[2]['min_known_score'], 'temperature': 1.,
                'selection_sha256': sha(args.output / 'SELECTION.json')}, args.output / 'rejection.pth')
    print(json.dumps({'output': str(args.output), 'calibration_passed': passed, 'selected_step': best[2]['step']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260912)
    arguments = parser.parse_args()
    require(arguments.steps >= 250, 'at least 250 optimizer steps are required')
    main(arguments)
