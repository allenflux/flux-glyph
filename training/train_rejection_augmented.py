#!/usr/bin/env python3
"""Continue local rejector training with source-verified, train-only handwriting."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'training')]
from rejection_network import RegionRejector
from train_rejection import BASE_CHECKPOINT_SHA, BASE_DATA_SHA, BASE_ONNX_SHA, POLICY, aggregate, annotate, calibrate, load_gate
from train_regions import RegionData, dump, require, sha, state_sha, verify_preprocessing_snapshot


def main(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    require(not args.output.exists(), 'use a new training output directory')
    require(sha(args.checkpoint) == BASE_CHECKPOINT_SHA and sha(args.anchor / 'MANIFEST.json') == BASE_DATA_SHA,
            'primary checkpoint or known anchor dataset changed')
    verify_preprocessing_snapshot('6e9ce2c812be94e41e0ca492c7555c671a5bd9d6bed34bec83200076f2202a25', args.snapshot)
    args.output.mkdir(parents=True)
    dump(args.output / 'POLICY.json', {**POLICY, 'seed': args.seed, 'steps': args.steps, 'evaluation_interval': 500,
                                      'batch_regions': 64, 'gate_data_sha256': sha(args.data / 'MANIFEST.json'),
                                      'supplement_data_sha256': sha(args.supplement / 'MANIFEST.json'),
                                      'warm_start_rejection_sha256': sha(args.resume_rejection),
                                      'trainable_encoder': 'independent rejector only; primary font/size ONNX unchanged'})
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    resume = torch.load(args.resume_rejection, map_location='cpu', weights_only=True)
    resume_selection_path = args.resume_rejection.parent / 'SELECTION.json'
    resume_selection = json.loads(resume_selection_path.read_text())
    require(sha(resume_selection_path) == resume['selection_sha256'] and resume['families'] == checkpoint['families']
            and resume['labels'] == ['unknown', 'known'] and resume_selection['test_read'] is False
            and resume_selection['data_manifest_sha256'] == sha(args.data / 'MANIFEST.json')
            and resume_selection['anchor_manifest_sha256'] == BASE_DATA_SHA,
            'warm-start rejection checkpoint has different classes, data or test history')
    model = RegionRejector(len(checkpoint['families']))
    model.load_state_dict(resume['state_dict'], strict=True)
    model.requires_grad_(True)
    encoder_before = state_sha({k: v for k, v in model.state_dict().items() if k.startswith(('trunk.', 'pool.', 'style.'))})
    anchor = RegionData(args.anchor, preprocessing_snapshot=args.snapshot)
    partitions = {}
    for split in ('train', 'calibration'):
        partitions[split] = [annotate(load_gate(args.data, split), 'new_native'),
                             annotate(anchor.load(split), 'r17_native', known=True)]
        print(json.dumps({'loaded_split': split, 'regions': [len(part['rows']) for part in partitions[split]]}), flush=True)
    from training.capture.prepare_rejection_supplement import load_partition as load_supplement
    supplement = load_supplement(args.supplement)
    partitions['train'].append(annotate(supplement, 'supplement_native'))
    print(json.dumps({'loaded_supplement_train_regions': len(supplement['rows'])}), flush=True)
    unknown = {split: sorted({row['family'] for part in parts for row in part['rows'] if row['label'] == 0})
               for split, parts in partitions.items()}
    require(not set(unknown['train']) & set(unknown['calibration']), 'unknown font family crosses selection partitions')
    require(all(family not in checkpoint['families'] for families in unknown.values() for family in families), 'known family labeled unknown')
    pools = {}
    for source_index, part in enumerate(partitions['train']):
        for row in part['rows']:
            pools.setdefault((row['label'], row['domain'], row['family']), []).append((source_index, row))
    keys = {label: [key for key in pools if key[0] == label] for label in (0, 1)}
    model.to(args.device)
    optimizer = torch.optim.AdamW([
        {'params': [parameter for name, parameter in model.named_parameters() if not name.startswith('head.')], 'lr': .00015},
        {'params': model.head.parameters(), 'lr': .001}], weight_decay=.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=.00002)
    best, history, started = None, [], time.monotonic()
    best_scores = None
    cal_rows = [row for part in partitions['calibration'] for row in part['rows']]
    for step in range(1, args.steps + 1):
        images, labels = [], []
        for label in (0, 1):
            for _ in range(32):
                key = keys[label][rng.integers(len(keys[label]))]
                source_index, row = pools[key][rng.integers(len(pools[key]))]
                tile_index = row['tile_start'] + int(rng.integers(row['tile_count']))
                images.append(partitions['train'][source_index]['tiles'][tile_index])
                labels.append(label)
        model.train()
        logits = model(torch.from_numpy(np.stack(images)).to(args.device))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor(labels, device=args.device), label_smoothing=.02)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            print(json.dumps({'step': step, 'loss': float(loss.detach().cpu()), 'seconds': round(time.monotonic() - started, 2)}), flush=True)
        if step % 500 == 0 or step == args.steps:
            model.eval()
            scores = []
            with torch.inference_mode():
                for part in partitions['calibration']:
                    outputs = []
                    for start in range(0, len(part['tiles']), 128):
                        block = torch.from_numpy(np.array(part['tiles'][start:start + 128], copy=True)).to(args.device)
                        outputs.append(model(block).cpu().numpy())
                    scores.extend(aggregate(np.concatenate(outputs), part['rows']).tolist())
            scores = np.asarray(scores)
            threshold, metrics = calibrate(scores, cal_rows)
            record = {'step': step, 'loss': float(loss.detach().cpu()), 'min_known_score': threshold, 'metrics': metrics}
            history.append(record)
            key = (metrics['unknown_recall'], -metrics['known_rejected'], -step)
            if best is None or key > best[0]:
                best = (key, {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}, record)
                best_scores = scores.copy()
            dump(args.output / 'CALIBRATION_PROGRESS.json', history)
            print(json.dumps({'calibration_step': step, 'unknown_recall': metrics['unknown_recall'],
                              'known_rejected': metrics['known_rejected'], 'threshold': threshold}), flush=True)
    model.cpu().load_state_dict(best[1], strict=True)
    encoder_after = state_sha({k: v for k, v in model.state_dict().items() if k.startswith(('trunk.', 'pool.', 'style.'))})
    require(encoder_before != encoder_after, 'independent rejector encoder did not train')
    require(sha(args.checkpoint) == BASE_CHECKPOINT_SHA, 'training mutated the primary checkpoint')
    decisions = {'schema': 'flux-glyph-rejection-calibration-decisions-v1', 'min_known_score': best[2]['min_known_score'],
                 'records': [{'domain': row['domain'], 'source_id': row['source_id'], 'region_id': row['region_id'],
                              'known_score': float(score), 'passed': bool(score >= best[2]['min_known_score'])}
                             for row, score in zip(cal_rows, best_scores)]}
    dump(args.output / 'CALIBRATION_DECISIONS.json', decisions)
    selected = {'schema': 'flux-glyph-region-rejection-training-v1', 'policy': POLICY, 'selected': best[2], 'history': history,
                'calibration_passed': bool(best[2]['metrics']['unknown_recall'] >= POLICY['minimum_unknown_recall_calibration']),
                'optimizer_steps_executed': args.steps, 'training_device': args.device,
                'training_scope': 'local independent image CNN trained end to end; primary font/size ONNX unchanged',
                'supplement_manifest_sha256': sha(args.supplement / 'MANIFEST.json'),
                'warm_start_rejection_sha256': sha(args.resume_rejection),
                'warm_start_selection_sha256': sha(resume_selection_path),
                'strategy_source_sha256': sha(ROOT / 'training/train_rejection.py'),
                'data_manifest_sha256': sha(args.data / 'MANIFEST.json'), 'anchor_manifest_sha256': BASE_DATA_SHA,
                'calibration_decisions_sha256': sha(args.output / 'CALIBRATION_DECISIONS.json'),
                'parent_checkpoint_sha256': BASE_CHECKPOINT_SHA, 'parent_onnx_sha256': BASE_ONNX_SHA,
                'encoder_before_sha256': encoder_before, 'encoder_after_sha256': encoder_after,
                'unknown_train_families': unknown['train'], 'unknown_calibration_families': unknown['calibration'],
                'test_read': False, 'source_sha256': sha(__file__), 'network_source_sha256': sha(ROOT / 'training/rejection_network.py'),
                'partitions': {split: {'regions': sum(len(p['rows']) for p in parts), 'tiles': sum(len(p['tiles']) for p in parts),
                                      'labels': dict(Counter(row['label'] for part in parts for row in part['rows']))}
                               for split, parts in partitions.items()}}
    dump(args.output / 'SELECTION.json', selected)
    torch.save({'state_dict': model.state_dict(), 'families': checkpoint['families'], 'labels': ['unknown', 'known'],
                'selected_step': best[2]['step'], 'min_known_score': best[2]['min_known_score'], 'temperature': 1.,
                'selection_sha256': sha(args.output / 'SELECTION.json')}, args.output / 'rejection.pth')
    print(json.dumps({'output': str(args.output), 'calibration_passed': selected['calibration_passed'],
                      'selected_step': best[2]['step'], 'unknown_recall': best[2]['metrics']['unknown_recall']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'anchor', 'checkpoint', 'snapshot', 'output', 'supplement', 'resume-rejection'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260912)
    arguments = parser.parse_args()
    require(arguments.steps >= 500, 'at least 500 end-to-end steps required')
    main(arguments)
