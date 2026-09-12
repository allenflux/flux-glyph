#!/usr/bin/env python3
"""Train an independent font verifier on native screenshots and source-isolated views.

The published primary font/size model and unknown-font rejector never change.
Selection uses calibration only; source test views cannot be opened here.
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
from train_regions import dump, require, sha, state_sha

SEED = 2026091301
SYSTEM = ('PingFang', 'SF Pro', 'Helvetica')
POLICY = {
    'schema': 'flux-glyph-sans-verifier-selection-v1',
    'temperature': 1.0,
    'gates': {'min_score': .5, 'min_margin': .01, 'min_patch_agreement': .7},
    'maximum_anchor_native_correct_loss_rate': .02,
    'maximum_system_family_native_correct_loss_rate': .05,
    'minimum_wrong_name_reduction': .20,
    'minimum_roboto_veto_recall': .80,
    'minimum_degraded_correct_retention': .85,
    'test_used_for_selection': False,
    'rank': ['wrong_names_removed', 'retained_correct', 'negative_calibration_nll', 'earliest_step'],
}


def observations(logits, rows, temperature, gates):
    from flux_glyph.region_font import aggregate_predictions
    logits = np.asarray(logits)
    require(logits.ndim == 2 and logits.shape[1] >= 2 and np.isfinite(logits).all(),
            'invalid or nonfinite family logits')
    require(rows and all(type(row['tile_start']) is int and type(row['tile_count']) is int
                        and row['tile_start'] >= 0 and row['tile_count'] > 0
                        and row['tile_start']+row['tile_count'] <= len(logits) for row in rows),
            'invalid observation tile mapping')
    result = []
    for row in rows:
        start, count = row['tile_start'], row['tile_count']
        value = aggregate_predictions(logits[start:start + count], np.zeros(count), temperature=temperature)
        passed = (value['score'] >= gates['min_score'] and value['margin'] >= gates['min_margin']
                  and value['margin'] > 1e-8 and value['patch_agreement'] >= gates['min_patch_agreement'])
        result.append({'predicted': int(value['order'][0]), 'score': value['score'],
                       'margin': value['margin'], 'agreement': value['patch_agreement'], 'passed': bool(passed),
                       'probabilities': value['probabilities'].tolist()})
    return result


def summarize(rows, baseline, verifier, families, primary_families):
    """Score the same naming decisions as the v3 runtime, including extra classes."""
    require(len(rows) == len(baseline) == len(verifier) > 0, 'unaligned calibration observations')
    details = []
    for row, before, check in zip(rows, baseline, verifier):
        predicted, verified = primary_families[before['predicted']], families[check['predicted']]
        accepted = before['passed'] and before['known']
        same = predicted == verified
        after = accepted and check['passed'] and same
        details.append({'domain': row['domain'], 'view': row['view'], 'family': row['family'],
                        'source_id': row['source_id'], 'region_id': row['region_id'],
                        'before_correct': accepted and predicted == row['family'],
                        'before_wrong': accepted and predicted != row['family'],
                        'after_correct': after and predicted == row['family'],
                        'after_wrong': after and predicted != row['family'],
                        'verifier_correct': verified == row['family'],
                        'roboto_veto': row['family'] == 'Roboto' and verified == 'Roboto' and check['passed']})

    def counts(values):
        out = {'regions': len(values)}
        for name in ('before_correct', 'before_wrong', 'after_correct', 'after_wrong', 'verifier_correct', 'roboto_veto'):
            out[name] = sum(int(row[name]) for row in values)
        out['correct_retention'] = out['after_correct'] / max(1, out['before_correct'])
        return out

    by_group = {f'{domain}/{view}': counts([r for r in details if r['domain'] == domain and r['view'] == view])
                for domain, view in sorted({(r['domain'], r['view']) for r in details})}
    anchor = [r for r in details if r['domain'] == 'anchor' and r['view'] == 'native']
    require(anchor, 'selection requires native anchor rows')
    native = counts(anchor)
    systems = {family: counts([r for r in anchor if r['family'] == family]) for family in SYSTEM}
    degraded = counts([r for r in details if r['view'] != 'native'])
    roboto = counts([r for r in details if r['family'] == 'Roboto'])
    total = counts(details)
    removed = total['before_wrong'] - total['after_wrong']
    reduction = removed / max(1, total['before_wrong'])
    roboto_recall = roboto['roboto_veto'] / max(1, roboto['regions'])
    checks = {
        'native_anchor_retention': native['correct_retention'] >= 1-POLICY['maximum_anchor_native_correct_loss_rate'],
        'native_system_retention': all(value['correct_retention'] >= 1-POLICY['maximum_system_family_native_correct_loss_rate']
                                       for value in systems.values() if value['before_correct']),
        'degraded_retention': degraded['correct_retention'] >= POLICY['minimum_degraded_correct_retention'],
        'wrong_name_reduction': reduction >= POLICY['minimum_wrong_name_reduction'],
        'roboto_veto': roboto_recall >= POLICY['minimum_roboto_veto_recall'],
        'no_new_wrong_names': total['after_wrong'] <= total['before_wrong'],
    }
    return {'passed': all(checks.values()), 'checks': checks, 'total': total, 'native_anchor': native,
            'systems': systems, 'degraded': degraded, 'roboto': roboto, 'by_group': by_group,
            'wrong_name_reduction': reduction, 'roboto_veto_recall': roboto_recall}, details


def baseline_observations(data, region):
    from flux_glyph.region_font import RegionFontClassifier
    classifier = RegionFontClassifier(region)
    logits, known = [], []
    for start in range(0, len(data['tiles']), 128):
        block = np.array(data['tiles'][start:start+128], copy=True)
        logits.append(classifier.session.run(['logits', 'log_em_ratio'], {'tiles': block})[0])
        values = classifier.rejection_session.run(['known_logits'], {'tiles': block})[0].astype(np.float64)
        require(values.shape == (len(block),2) and np.isfinite(values).all(), 'invalid baseline rejection output')
        values = np.exp((values - values.max(1, keepdims=True))/classifier.rejection_meta['temperature'])
        known.append(values[:, 1] / values.sum(1))
    result = observations(np.concatenate(logits), data['rows'], classifier.meta['temperature'], classifier.meta['gates'])
    known = np.concatenate(known)
    for row, value in zip(data['rows'], result):
        score = float(known[row['tile_start']:row['tile_start']+row['tile_count']].mean())
        value.update(known=score >= classifier.rejection_meta['min_known_score'], known_score=score)
    return result, classifier.families


def evaluate(model, data, device):
    import torch
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(data['tiles']), 128):
            outputs.append(model(torch.from_numpy(np.array(data['tiles'][start:start+128], copy=True)).to(device))[0].cpu().numpy())
    logits = np.concatenate(outputs)
    require(np.isfinite(logits).all(), 'nonfinite calibration logits')
    obs = observations(logits, data['rows'], POLICY['temperature'], POLICY['gates'])
    loss = float(np.mean([-np.log(max(1e-12, value['probabilities'][row['target']])) for value, row in zip(obs, data['rows'])]))
    return obs, loss


def main(args):
    import torch
    from region_network import RegionFontClassifier
    from prepare_sans_views import load_views
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    require(not args.output.exists(), 'training output must be new')
    train, cal = load_views(args.data, 'train'), load_views(args.data, 'calibration')
    families = train['families']
    require(cal['families'] == families and len(families) == 9 and families[-1] == 'Roboto', 'expected all nine native classes')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    require(checkpoint['families'] == families[:-1], 'primary family order differs')
    args.output.mkdir(parents=True)
    bindings = {str(path.resolve()): sha(path) for path in (
        args.data/'MANIFEST.json', args.data/'train/MANIFEST.json', args.data/'calibration/MANIFEST.json',
        args.checkpoint, args.primary/'metadata.json', args.primary/'model.onnx', args.primary/'rejection.onnx',
        Path(__file__), ROOT/'training/prepare_sans_views.py', ROOT/'training/region_network.py', ROOT/'training/network.py')}
    dump(args.output/'POLICY.json', {'policy': POLICY, 'seed': args.seed, 'steps': args.steps, 'evaluation_interval': 500,
                                   'bindings': bindings, 'test_read': False, 'families': families,
                                   'training_scope': 'local independent CNN; unchanged primary and rejection ONNX',
                                   'sampling': 'balanced family; native views 40%, each degraded view 20%; balanced available domains'})
    base, primary_families = baseline_observations(cal, args.primary)
    dump(args.output/'BASELINE.json', {'records': base, 'families': primary_families})
    print(json.dumps({'loaded_train_regions': len(train['rows']), 'calibration_regions': len(cal['rows']),
                      'train_family_counts': dict(Counter(r['family'] for r in train['rows']))}), flush=True)
    model = RegionFontClassifier(9)
    state = model.state_dict()
    for name, value in checkpoint['state_dict'].items():
        if name.startswith('family_head.'):
            state[name][:-1] = value
            state[name][-1] = value.mean(dim=0)
        else:
            state[name] = value.clone()
    state['family_head.bias'][-1] -= 2
    model.load_state_dict(state)
    before_sha = state_sha(model.state_dict())
    teacher = RegionFontClassifier(8).eval().requires_grad_(False)
    teacher.load_state_dict(checkpoint['state_dict'])
    teacher.to(args.device)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.0002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=1e-5)
    pools = {}
    for row in train['rows']:
        pools.setdefault((row['target'], row['domain'], row['view']), []).append(row)
    views = ['native', 'half', 'three_quarters_jpeg75', 'jpeg75']
    best, history, started = None, [], time.monotonic()
    best_obs = None
    for step in range(1, args.steps+1):
        images, targets, sizes, anchored = [], [], [], []
        for i in range(64):
            target = int((step*64+i) % len(families))
            view = str(rng.choice(views, p=[.4,.2,.2,.2]))
            keys = [key for key in pools if key[0] == target and key[2] == view]
            require(keys, 'missing supervised family/view')
            key = keys[int(rng.integers(len(keys)))]
            row = pools[key][int(rng.integers(len(pools[key])))]
            images.append(train['tiles'][row['tile_start']+int(rng.integers(row['tile_count']))])
            targets.append(target)
            sizes.append(row['log_em_ratio'])
            anchored.append(row['domain'] == 'anchor' and row['view'] == 'native')
        tensor = torch.from_numpy(np.stack(images)).to(args.device)
        model.train()
        logits, ratio = model(tensor)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor(targets, device=args.device), label_smoothing=.03)
        loss = loss + .05*torch.nn.functional.smooth_l1_loss(ratio, torch.tensor(sizes, device=args.device, dtype=torch.float32))
        mask = torch.tensor(anchored, dtype=torch.bool, device=args.device)
        if any(anchored):
            with torch.no_grad():
                original, original_size = teacher(tensor[mask])
            target_distribution = torch.cat([torch.softmax(original/2, dim=1), torch.zeros((len(original),1), device=args.device)], dim=1)
            loss = loss + .5*4*torch.nn.functional.kl_div(torch.log_softmax(logits[mask]/2, dim=1), target_distribution, reduction='batchmean')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            print(json.dumps({'step': step, 'loss': float(loss.detach().cpu()), 'seconds': round(time.monotonic()-started, 1)}), flush=True)
        if step % 500 == 0 or step == args.steps:
            obs, nll = evaluate(model, cal, args.device)
            metrics, _ = summarize(cal['rows'], base, obs, families, primary_families)
            record = {'step': step, 'calibration_nll': nll, 'metrics': metrics}
            history.append(record)
            rank = (int(metrics['passed']), metrics['total']['before_wrong']-metrics['total']['after_wrong'],
                    metrics['total']['after_correct'], -nll, -step)
            if best is None or rank > best[0]:
                best = (rank, {key: value.detach().cpu().clone() for key,value in model.state_dict().items()}, record)
                best_obs = copy.deepcopy(obs)
            dump(args.output/'CALIBRATION_PROGRESS.json', history)
            print(json.dumps({'calibration_step':step, 'metrics':metrics}), flush=True)
    model.cpu().load_state_dict(best[1])
    after_sha = state_sha(model.state_dict())
    require(before_sha != after_sha, 'verifier parameters did not train')
    require(all(sha(path)==digest for path,digest in bindings.items()), 'training input changed')
    dump(args.output/'CALIBRATION_DECISIONS.json', {'records': best_obs, 'families': families, 'policy': POLICY})
    selection = {'schema':'flux-glyph-sans-verifier-training-v1', 'policy':POLICY, 'selected': best[2],
                 'calibration_passed': best[2]['metrics']['passed'], 'passed': best[2]['metrics']['passed'],
                 'test_read':False, 'history':history, 'families':families, 'bindings':bindings,
                 'optimizer_steps_executed':args.steps, 'training_device':args.device,
                 'state_before_sha256':before_sha, 'state_after_sha256':after_sha,
                 'calibration_decisions_sha256':sha(args.output/'CALIBRATION_DECISIONS.json'),
                 'baseline_sha256':sha(args.output/'BASELINE.json')}
    dump(args.output/'SELECTION.json', selection)
    torch.save({'state_dict':model.state_dict(), 'families':families, 'selection_sha256':sha(args.output/'SELECTION.json')}, args.output/'verifier.pth')
    print(json.dumps({'output':str(args.output), 'passed':selection['passed'], 'selected_step':best[2]['step']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'checkpoint', 'primary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--learning-rate', type=float, default=.0001)
    parser.add_argument('--device', choices=('mps','cpu'), default='mps')
    args = parser.parse_args()
    require(args.steps >= 500, 'at least 500 training steps')
    main(args)
