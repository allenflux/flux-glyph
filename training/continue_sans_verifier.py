#!/usr/bin/env python3
"""Continue a CAL-selected verifier; preserve the original run and serving models."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'training')]
from train_regions import dump, require, sha, state_sha
from train_sans_verifier import POLICY, SEED, evaluate, summarize

VIEWS = ('native', 'half', 'three_quarters_jpeg75', 'jpeg75')
ENCODER_LR, HEAD_LR = .0001, .0005


def validate_resume(args, checkpoint):
    """Validate the complete old CAL cache before any new training or array load."""
    source = args.source_run.resolve()
    selection_path = source/'SELECTION.json'
    selection = json.loads(selection_path.read_text())
    require(selection.get('schema') == 'flux-glyph-sans-verifier-training-v1'
            and selection.get('test_read') is False and selection.get('policy') == POLICY,
            'resume requires the original unchanged CAL-only policy')
    require(checkpoint.get('selection_sha256') == sha(selection_path)
            and checkpoint.get('families') == selection.get('families'), 'resume checkpoint/selection differ')
    require(type(selection.get('optimizer_steps_executed')) is int and selection['optimizer_steps_executed'] > 0
            and type(selection.get('selected', {}).get('step')) is int
            and 0 < selection['selected']['step'] <= selection['optimizer_steps_executed']
            and selection['selected'] in selection.get('history', []), 'resume training step/history differs')
    bindings = selection.get('bindings')
    require(isinstance(bindings, dict) and bindings and all(isinstance(path, str)
            and str(Path(path).resolve()) == path and Path(path).is_file() and sha(path) == digest
            for path, digest in bindings.items()), 'resume source binding changed')
    required = [args.data/'MANIFEST.json', args.data/'train/MANIFEST.json', args.data/'calibration/MANIFEST.json',
                args.checkpoint, args.primary/'metadata.json', args.primary/'model.onnx', args.primary/'rejection.onnx',
                ROOT/'training/train_sans_verifier.py', ROOT/'training/prepare_sans_views.py',
                ROOT/'training/region_network.py', ROOT/'training/network.py']
    require(all(bindings.get(str(path.resolve())) == sha(path) for path in required),
            'resume data, primary or teacher differ from the cached baseline')
    require(sha(source/'BASELINE.json') == selection['baseline_sha256']
            and sha(source/'CALIBRATION_DECISIONS.json') == selection['calibration_decisions_sha256'],
            'resume baseline or CAL decisions changed')
    policy = json.loads((source/'POLICY.json').read_text())
    require(policy.get('policy') == POLICY and policy.get('test_read') is False
            and policy.get('bindings') == bindings and policy.get('steps') == selection['optimizer_steps_executed'],
            'resume POLICY and SELECTION disagree')
    primary = json.loads((args.primary/'metadata.json').read_text())
    data = json.loads((args.data/'MANIFEST.json').read_text())
    calibration = json.loads((args.data/'calibration/MANIFEST.json').read_text())
    baseline = json.loads((source/'BASELINE.json').read_text())
    decisions = json.loads((source/'CALIBRATION_DECISIONS.json').read_text())
    families = selection['families']
    require(isinstance(families, list) and len(families) == 9 and len(set(families)) == 9
            and families[-1] == 'Roboto' and data.get('families') == families
            and baseline.get('families') == primary.get('families') == families[:-1]
            and decisions.get('families') == families and decisions.get('policy') == POLICY,
            'resume class ordering or cached decisions policy differs')
    require(primary.get('algorithm') == 'region-cnn64x256-rejection-v2'
            and primary.get('model', {}).get('path') == 'model.onnx'
            and primary['model'].get('sha256') == sha(args.primary/'model.onnx')
            and primary.get('rejection', {}).get('model', {}).get('path') == 'rejection.onnx'
            and primary['rejection']['model'].get('sha256') == sha(args.primary/'rejection.onnx'),
            'resume must retain both original primary/rejection ONNX models')
    require(calibration.get('split') == 'calibration' and calibration.get('test_pixels_opened') is False
            and len(baseline.get('records', [])) == len(decisions.get('records', [])) == calibration.get('rows')
            and calibration['rows'] > 0, 'resume calibration cache shape/partition differs')
    # Inherit every old binding so old sources cannot be silently replaced.
    bindings = dict(bindings)
    for path in (selection_path, source/'verifier.pth', source/'CALIBRATION_DECISIONS.json',
                 source/'BASELINE.json', source/'POLICY.json', Path(__file__), ROOT/'training/train_regions.py'):
        bindings[str(path.resolve())] = sha(path)
    return selection, baseline, decisions, bindings


def training_pools(rows, family_count):
    pools = {}
    for row in rows:
        require(row.get('split') == 'train' and row.get('view') in VIEWS
                and row.get('domain') in ('anchor', 'new_native')
                and type(row.get('target')) is int and 0 <= row['target'] < family_count,
                'sampler accepts only valid training rows')
        pools.setdefault((row['target'], row['domain'], row['view']), []).append(row)
    require(all(any(key[0] == target and key[2] == view for key in pools)
                for target in range(family_count) for view in VIEWS), 'missing supervised family/view')
    return pools


def sample_rows(pools, family_count, step, rng):
    selected = []
    for i in range(64):
        target = int((step*64+i) % family_count)
        view = str(rng.choice(VIEWS, p=[.4, .2, .2, .2]))
        keys = [key for key in pools if key[0] == target and key[2] == view]
        key = keys[int(rng.integers(len(keys)))]
        row = pools[key][int(rng.integers(len(pools[key])))]
        tile = row['tile_start'] + int(rng.integers(row['tile_count']))
        selected.append((row, tile))
    return selected


def main(args):
    import torch
    from region_network import RegionFontClassifier
    from prepare_sans_views import load_views
    require(type(args.steps) is int and args.steps >= 500, 'at least 500 continuation steps')
    require(not args.output.exists(), 'continuation output must be new')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    resumed = torch.load(args.source_run/'verifier.pth', map_location='cpu', weights_only=True)
    source, baseline, previous_decisions, bindings = validate_resume(args, resumed)
    require(state_sha(resumed['state_dict']) == source['state_after_sha256'], 'resume parameters differ from selection')
    train, cal = load_views(args.data, 'train'), load_views(args.data, 'calibration')
    families = source['families']
    require(train['families'] == cal['families'] == families, 'continuation view class ordering differs')
    base, primary_families = baseline['records'], baseline['families']
    old_metrics, _ = summarize(cal['rows'], base, previous_decisions['records'], families, primary_families)
    require(old_metrics == source['selected']['metrics'], 'cached CAL baseline no longer reproduces source selection')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    require(checkpoint['families'] == families[:-1], 'teacher class order differs')
    model = RegionFontClassifier(9)
    model.load_state_dict(resumed['state_dict'], strict=True)
    before_sha = state_sha(model.state_dict())
    teacher = RegionFontClassifier(8).eval().requires_grad_(False)
    teacher.load_state_dict(checkpoint['state_dict'], strict=True)
    teacher.to(args.device)
    model.to(args.device)
    optimizer = torch.optim.AdamW([
        {'params': [parameter for name, parameter in model.named_parameters() if not name.startswith('family_head.')], 'lr': ENCODER_LR},
        {'params': model.family_head.parameters(), 'lr': HEAD_LR}], weight_decay=.0002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=1e-5)
    pools = training_pools(train['rows'], len(families))
    continuation = {'source_run': str(args.source_run.resolve()),
                    'source_selection_sha256': sha(args.source_run/'SELECTION.json'),
                    'source_checkpoint_sha256': sha(args.source_run/'verifier.pth'),
                    'source_selected_step': source['selected']['step'],
                    'source_optimizer_steps_executed': source['optimizer_steps_executed'],
                    'additional_optimizer_steps_planned': args.steps,
                    'cached_baseline_reused': True, 'cached_baseline_sha256': source['baseline_sha256']}
    optimizer_policy = {'name': 'AdamW', 'encoder_learning_rate': ENCODER_LR, 'family_head_learning_rate': HEAD_LR,
                        'weight_decay': .0002, 'scheduler': 'CosineAnnealingLR', 'minimum_learning_rate': 1e-5,
                        'label_smoothing': .03, 'size_loss_weight': .05, 'teacher_kl_weight': .5,
                        'teacher_temperature': 2., 'teacher_rows': 'anchor/native only', 'batch_regions': 64}
    args.output.mkdir(parents=True)
    dump(args.output/'POLICY.json', {'policy': POLICY, 'seed': args.seed, 'steps': args.steps, 'evaluation_interval': 500,
        'bindings': bindings, 'test_read': False, 'families': families, 'optimizer': optimizer_policy,
        'continuation': continuation, 'training_scope': 'local independent verifier continuation; primary/rejection unchanged',
        'sampling': 'balanced family; native views 40%, each degraded view 20%; balanced available domains'})
    shutil.copyfile(args.source_run/'BASELINE.json', args.output/'BASELINE.json')
    require(sha(args.output/'BASELINE.json') == source['baseline_sha256'], 'copied baseline bytes differ')
    print(json.dumps({'loaded_train_regions': len(train['rows']), 'calibration_regions': len(cal['rows']),
                      'train_family_counts': dict(Counter(row['family'] for row in train['rows'])),
                      'cached_baseline_reused': True, 'continuation': continuation}), flush=True)
    best, history, started, best_obs = None, [], time.monotonic(), None
    for step in range(1, args.steps+1):
        images, targets, sizes, anchored = [], [], [], []
        for row, tile in sample_rows(pools, len(families), step, rng):
            images.append(train['tiles'][tile])
            targets.append(row['target'])
            sizes.append(row['log_em_ratio'])
            anchored.append(row['domain'] == 'anchor' and row['view'] == 'native')
        tensor = torch.from_numpy(np.stack(images)).to(args.device)
        model.train()
        logits, ratio = model(tensor)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor(targets, device=args.device), label_smoothing=.03)
        loss = loss + .05*torch.nn.functional.smooth_l1_loss(ratio, torch.tensor(sizes, device=args.device, dtype=torch.float32))
        if any(anchored):
            mask = torch.tensor(anchored, dtype=torch.bool, device=args.device)
            with torch.no_grad():
                original, _ = teacher(tensor[mask])
            distribution = torch.cat([torch.softmax(original/2, dim=1), torch.zeros((len(original), 1), device=args.device)], dim=1)
            loss = loss + .5*4*torch.nn.functional.kl_div(torch.log_softmax(logits[mask]/2, dim=1), distribution, reduction='batchmean')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            print(json.dumps({'step': step, 'loss': float(loss.detach().cpu()),
                              'seconds': round(time.monotonic()-started, 1)}), flush=True)
        if step % 500 == 0 or step == args.steps:
            obs, nll = evaluate(model, cal, args.device)
            metrics, _ = summarize(cal['rows'], base, obs, families, primary_families)
            record = {'step': step, 'calibration_nll': nll, 'metrics': metrics}
            history.append(record)
            rank = (int(metrics['passed']), metrics['total']['before_wrong']-metrics['total']['after_wrong'],
                    metrics['total']['after_correct'], -nll, -step)
            if best is None or rank > best[0]:
                best = (rank, {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}, record)
                best_obs = copy.deepcopy(obs)
            dump(args.output/'CALIBRATION_PROGRESS.json', history)
            print(json.dumps({'calibration_step': step, 'metrics': metrics}), flush=True)
    model.cpu().load_state_dict(best[1], strict=True)
    after_sha = state_sha(model.state_dict())
    require(before_sha != after_sha, 'verifier parameters did not change during continuation')
    require(all(sha(path) == digest for path, digest in bindings.items()), 'continuation input changed')
    continuation.update(additional_optimizer_steps_executed=args.steps,
                        total_lineage_optimizer_steps_executed=source.get('continuation', {}).get(
                            'total_lineage_optimizer_steps_executed', source['optimizer_steps_executed']) + args.steps,
                        selected_step_in_continuation=best[2]['step'],
                        selected_checkpoint_lineage_optimizer_steps=source.get('continuation', {}).get(
                            'selected_checkpoint_lineage_optimizer_steps', source['selected']['step']) + best[2]['step'])
    dump(args.output/'CALIBRATION_DECISIONS.json', {'records': best_obs, 'families': families, 'policy': POLICY})
    selection = {'schema': 'flux-glyph-sans-verifier-training-v1', 'policy': POLICY, 'selected': best[2],
                 'calibration_passed': best[2]['metrics']['passed'], 'passed': best[2]['metrics']['passed'],
                 'test_read': False, 'history': history, 'families': families, 'bindings': bindings,
                 'optimizer_steps_executed': args.steps, 'training_device': args.device,
                 'state_before_sha256': before_sha, 'state_after_sha256': after_sha,
                 'calibration_decisions_sha256': sha(args.output/'CALIBRATION_DECISIONS.json'),
                 'baseline_sha256': sha(args.output/'BASELINE.json'), 'optimizer': optimizer_policy,
                 'continuation': continuation}
    dump(args.output/'SELECTION.json', selection)
    torch.save({'state_dict': model.state_dict(), 'families': families,
                'selection_sha256': sha(args.output/'SELECTION.json')}, args.output/'verifier.pth')
    print(json.dumps({'output': str(args.output), 'passed': selection['passed'], 'selected_step': best[2]['step'],
                      'additional_optimizer_steps_executed': args.steps}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'data', 'checkpoint', 'primary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    main(parser.parse_args())
