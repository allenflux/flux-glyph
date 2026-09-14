#!/usr/bin/env python3
"""Train weak verified three/four-glyph supervision with original unknown-teacher restoration."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'training')]
from cache_r22_train_teacher import (BASE_RUN, BASE_SHA, BASE_SELECTION_SHA, BASE_STATE_SHA,
    ARCHITECTURE, SOURCES, load_base, load_training, read, write_new, source_bindings)
from prepare_unified_regions import FAMILIES, load_split
from train_regions import require, sha, state_sha
from train_unified_retention import FIXED_RUNTIME, evaluate_outputs, parameter_groups, infer
from train_unified_retention_micro_recovery import full_losses, forward_with_features, batch_source, pixel_batch
from retention_face_balanced_sampler import FaceBalancedSampler
from short_region_objective import (ShortSampler, SHORT_SAMPLING,
    EXTRA_ACCEPTANCE, short_population, compare_r22)
from long_short_supervision import CONTRACT as LONG_SHORT_CONTRACT, long_short_loss
from restore_unknown_train_teacher import load_replay_teacher
from retention_supplement_sampler import SUPPLEMENT_MARKER

RETENTION_PLAN = ROOT / 'artifacts/unified-font-v2/RETENTION_PLAN.json'
RETENTION_SHA = 'b82283b54b5e47f4639b0df62231544c0419f6299d10820d86dca96ef24094ff'
STEPS = 2400
SEED = 2026091401
LEARNING_RATE = 5e-6
MINIMUM_LEARNING_RATE = 5e-7
OBJECTIVE = {
    'replay_batch': 96, 'short_batch': 32,
    'replay_loss': 'unchanged R22 supervised, unknown-floor, angular, size and correct-teacher full KL2',
    'replay_teacher': {'original_true_unknown': 'original frozen R21 same-tile TRAIN logits',
                      'original_known': 'frozen R22 same-tile TRAIN logits',
                      'supplement_all': 'frozen R22 same-tile TRAIN logits',
                      'merged_known_all': 'frozen R22 same-tile TRAIN logits',
                      'selection_input': 'verified original TRAIN partition and true label only'},
    'short_loss': LONG_SHORT_CONTRACT, 'short_teacher': None,
    'short_weight': .125, 'replay_size_beta': 1.,
    'control_comparison': 'R22 initialization and exact128-row inputs/RNG match replay-only control',
    'rationale': 'Control mitigated the short regression but retained small precision/unknown losses; weak known3/4 supervision targets missing short gains while historical original-unknown teacher constrains false names.',
    'inactive_short_rows': '1/2 glyph and all short unknown: exactly zero font and size gradient',
    'label_sources': 'verified native TRAIN font/glyph proof, never predictions',
    'teacher_cache_deployed': False, 'second_model_resident_during_optimizer': False,
    'model_count': 1, 'all_parameters_trained': True,
    'ocr_recognition': False, 'runtime_changes': False,
    'arbitrary_confidence_boost': False, 'original_full_correct_teacher_kl_preserved': True,
}


def files_bound(args, datasets):
    bindings = dict(read(BASE_RUN / 'SELECTION.json')['bindings'])
    for additional in (source_bindings(datasets), read(args.short / 'MANIFEST.json')['bindings']):
        for path, digest in additional.items():
            require((path not in bindings or bindings[path] == digest) and sha(path) == digest,
                    'Native proof or training source conflicts with frozen evidence')
            bindings[path] = digest
    _, teacher_proof = load_replay_teacher(args.teacher, datasets)
    for path, digest in teacher_proof['bindings'].items():
        require(sha(path) == digest and (path not in bindings or bindings[path] == digest),
                'Restored teacher source conflicts with frozen evidence')
        bindings[path] = digest
    files = [Path(__file__), ROOT / 'training/short_region_objective.py',
             ROOT / 'training/train_unified_short_regions.py', ROOT / 'training/train_unified_short_control.py',
             ROOT / 'training/long_short_supervision.py', ROOT / 'training/restore_unknown_train_teacher.py',
             ROOT / 'artifacts/unified-font-v4/short-control-comparison-v1/REPORT.json',
             ROOT / 'artifacts/unified-font-v4/train-teacher-comparison-v1/REPORT.json',
             ROOT / 'training/prepare_short_region_supplement.py',
             BASE_RUN / 'CALIBRATION_OUTPUTS.npz', BASE_RUN / 'CALIBRATION_DECISIONS.json',
             BASE_RUN / 'DEVELOPMENT_REGRESSION.json', RETENTION_PLAN,
             args.teacher / 'CACHE_FREEZE.json', args.teacher / 'CACHE_MANIFEST.json',
             args.short / 'MANIFEST.json', args.short / 'train/MANIFEST.json']
    part = read(args.short / 'train/MANIFEST.json')
    files += [args.short / 'train' / part[key]['path'] for key in ('array', 'metadata')]
    for part in read(args.teacher / 'CACHE_MANIFEST.json')['partitions'].values():
        files += [args.teacher / value['path'] for value in part['files'].values()]
    for path in files:
        digest = sha(path)
        require(str(path.resolve()) not in bindings or bindings[str(path.resolve())] == digest,
                'New training source conflicts with frozen R22 evidence')
        bindings[str(path.resolve())] = digest
    return bindings


def context(args):
    from prepare_short_region_supplement import load_supplement
    require(sha(RETENTION_PLAN) == RETENTION_SHA, 'Original 46 retention rules changed')
    datasets = load_training()
    teacher, teacher_manifest = load_replay_teacher(args.teacher, datasets)
    short = load_supplement(args.short)
    require(short['families'] == FAMILIES and short['partition']['split'] == 'train', 'Short data is not verified TRAIN')
    model, base_selection = load_base()
    replay = {name: batch_source(data, {'base_logits': teacher[name]['logits']}) for name, data in datasets.items()}
    plan = read(RETENTION_PLAN)
    sampler = FaceBalancedSampler(datasets['original']['rows'], FAMILIES, SEED, plan['train_pools'],
                                  datasets['supplement']['rows'], datasets['known']['rows'])
    short_sampler = ShortSampler(short['rows'], SEED + 1)
    return model, datasets, replay, short, sampler, short_sampler, base_selection, teacher_manifest


def inputs(replay, short, sampler, short_sampler, device):
    import torch
    rows, small = sampler.batch(), short_sampler.batch()
    images, teacher, sizes, indices = pixel_batch(rows, replay['original'], replay['supplement'], replay['known'],
                                                 sampler.rng, return_indices=True)
    short_indices = [r['tile_start'] + int(short_sampler.rng.integers(r['tile_count'])) for r in small]
    require(all(short['rows'][r['index']] == r for r in small), 'Short sampler altered source truth')
    sx = np.array(short['tiles'][short_indices], copy=True)
    tensor = lambda values: torch.from_numpy(np.asarray(values)).to(device)
    return {'images': tensor(np.concatenate((images, sx))), 'teacher': tensor(teacher),
            'sizes': tensor(sizes), 'targets': tensor(np.array([r['target'] for r in rows], dtype=np.int64)),
            'short_sizes': tensor(np.array([r['log_em_ratio'] for r in small], dtype=np.float32)),
            'short_targets': tensor(np.array([r['target'] for r in small], dtype=np.int64)),
            'rows': rows, 'short_rows': small, 'indices': indices, 'short_indices': short_indices}


def objective(model, batch):
    font, size, features = forward_with_features(model, batch['images'])
    replay = full_losses(font[:96], size[:96], features[:96], model.family_head.weight,
                         batch['targets'], batch['sizes'], batch['teacher'], batch['rows'], FAMILIES)
    augment = long_short_loss(font[96:], size[96:], batch['short_targets'], batch['short_sizes'], batch['short_rows'])
    import torch
    old_unknown = torch.tensor([r['target'] == 24 and not r.get(SUPPLEMENT_MARKER, False)
                                for r in batch['rows']], device=font.device)
    new_unknown = torch.tensor([r['target'] == 24 and r.get(SUPPLEMENT_MARKER, False)
                                for r in batch['rows']], device=font.device)
    return replay[0] + augment[0], {'replay': replay[0], 'replay_ce': replay[1], 'teacher_kl': replay[3],
            'short_weighted': augment[0], 'short_font': augment[1], 'short_size': augment[2],
            'short_eligible': augment[3].sum(),
            'original_unknown_teacher_eligible': (old_unknown & replay[5]).sum(),
            'supplement_unknown_teacher_eligible': (new_unknown & replay[5]).sum()}



def design(args, bindings):
    return {'schema': 'flux-glyph-unified-short-training-plan-v3', 'architecture': ARCHITECTURE,
            'initializer_checkpoint_sha256': BASE_SHA, 'initializer_selection_sha256': BASE_SELECTION_SHA,
            'initializer_state_sha256': BASE_STATE_SHA, 'retention_plan_sha256': RETENTION_SHA,
            'families': FAMILIES, 'steps': STEPS, 'evaluation_step': STEPS,
            'seed': SEED, 'learning_rate': LEARNING_RATE, 'minimum_learning_rate': MINIMUM_LEARNING_RATE,
            'weight_decay': 1e-4, 'gradient_clip': 5., 'objective': OBJECTIVE,
            'short_sampling': SHORT_SAMPLING, 'runtime': FIXED_RUNTIME,
            'additional_acceptance': EXTRA_ACCEPTANCE, 'bindings': bindings,
            'teacher_cache': str(args.teacher), 'short_data': str(args.short),
            'checkpoint_selection': 'fixed final step 2400; no intermediate CAL search',
            'test_read': False, 'development_holdout_read': False,
            'calibration_reused_not_blind': True, 'source': 'local Mac native screenshots and derived TRAIN crops'}


def preflight(args):
    import torch
    require(not args.output.exists() and not args.plan.exists(), 'Keep earlier preflight and training plans')
    torch.set_num_threads(4); torch.manual_seed(SEED)
    model, datasets, replay, short, sampler, short_sampler, _, _ = context(args)
    bindings = files_bound(args, datasets)
    before = parameter_groups(model.state_dict())
    model.to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    traces = []
    for _ in range(4):
        batch = inputs(replay, short, sampler, short_sampler, args.device)
        loss, parts = objective(model, batch)
        require(bool(torch.isfinite(loss)), 'Nonfinite actual TRAIN preflight loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        gradients = {}
        for group in before:
            values = [p.grad for n, p in model.named_parameters() if n.startswith(group + '.')]
            require(values and all(v is not None and bool(torch.isfinite(v).all()) for v in values)
                    and any(bool(v.abs().sum() > 0) for v in values), 'Missing/nonfinite preflight gradients: ' + group)
            gradients[group] = True
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        traces.append({'loss': float(loss.detach()), 'parts': {k: float(v.detach()) for k, v in parts.items()},
                       'finite_nonzero_gradient_groups': gradients})
    model.cpu(); after = parameter_groups(model.state_dict())
    require(all(before[key] != after[key] for key in before)
            and sha(BASE_RUN / 'model.pth') == BASE_SHA
            and all(sha(p) == h for p, h in bindings.items()), 'Preflight mutated baseline or did not train every group')
    report = {'schema': 'flux-glyph-short-region-gradient-preflight-v3', 'passed': True, 'steps': 4,
              'device': args.device, 'replay_tiles': 384, 'short_tiles': 128, 'traces': traces,
              'short_forwarded_tiles': 128,
              'short_supervised_tiles': sum(int(t['parts']['short_eligible']) for t in traces),
              'all_groups_changed': True, 'baseline_unchanged': True, 'pixels': 'actual verified TRAIN only',
              'calibration_inference': False, 'development_read': False, 'test_read': False,
              'checkpoint_written': False, 'training_run_started': False, 'bindings': bindings}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(args.output, report)
    planned = {**design(args, bindings), 'preflight': {'path': str(args.output), 'sha256': sha(args.output)}}
    args.plan.parent.mkdir(parents=True, exist_ok=True); write_new(args.plan, planned)
    print(json.dumps({'passed': True, 'plan_sha256': sha(args.plan), 'training_started': False}), flush=True)


def train(args):
    import torch
    require(not args.output.exists(), 'Keep all old model runs; use a new output directory')
    torch.set_num_threads(4); torch.manual_seed(SEED)
    model, datasets, replay, short, sampler, short_sampler, base_selection, teacher_proof = context(args)
    bindings = files_bound(args, datasets)
    planned = read(args.plan)
    require(all(planned.get(k) == v for k, v in design(args, bindings).items()), 'Training plan changed after preflight')
    pre = planned['preflight']; proof = read(pre['path'])
    require(sha(pre['path']) == pre['sha256'] and proof['passed'] is True
            and proof['bindings'] == bindings and proof['test_read'] is False, 'TRAIN gradient preflight differs')
    bindings[str(args.plan)] = sha(args.plan); bindings[pre['path']] = pre['sha256']
    before = parameter_groups(model.state_dict())
    args.output.mkdir(parents=True)
    protocol = {**planned, 'schema': 'flux-glyph-unified-short-training-freeze-v3', 'bindings': bindings,
                'device': args.device, 'initial_parameter_groups_sha256': before,
                'training_teacher_proof': teacher_proof,
                'plan_sha256': sha(args.plan)}
    write_new(args.output / 'TRAINING_FREEZE.json', protocol)
    freeze_sha = sha(args.output / 'TRAINING_FREEZE.json')
    model.to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=MINIMUM_LEARNING_RATE)
    counts = Counter(); sources = Counter(); short_counts = Counter()
    supervised = Counter(); lengths = Counter(); teacher_eligible = Counter()
    started = time.monotonic()
    for step in range(1, STEPS + 1):
        batch = inputs(replay, short, sampler, short_sampler, args.device)
        loss, parts = objective(model, batch)
        require(bool(torch.isfinite(loss)), 'Nonfinite short-region optimizer loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        require(bool(torch.isfinite(norm)), 'Nonfinite optimizer gradients')
        optimizer.step(); scheduler.step()
        counts.update(r['family'] for r in batch['rows'])
        sources.update(r['source_font_family'] for r in batch['rows'] if r['family'] == '__unknown__')
        short_counts.update(r['family'] for r in batch['short_rows'])
        active = [r for r in batch['short_rows'] if r['target'] != 24 and r['glyph_count'] in (3, 4)]
        require(int(parts['short_eligible'].detach()) == len(active), 'Actual short supervision mask differs')
        supervised.update(r['family'] for r in active); lengths.update(str(r['glyph_count']) for r in active)
        for name in ('original_unknown_teacher_eligible', 'supplement_unknown_teacher_eligible'):
            teacher_eligible[name] += int(parts[name].detach())
        if step % 100 == 0:
            print(json.dumps({'step': step, 'loss': float(loss.detach()),
                              **{k: float(v.detach()) for k, v in parts.items()},
                              'seconds': round(time.monotonic() - started, 1)}), flush=True)
    require(sum(counts.values()) == STEPS * 96 and sum(short_counts.values()) == STEPS * 32
            and all(short_counts[f] == STEPS for f in FAMILIES[:-1])
            and short_counts['__unknown__'] == STEPS * 8
            and len(sources) == 11 and max(sources.values()) - min(sources.values()) <= 1,
            'Replay/short supervision row accounting differs from the plan')
    require(sum(supervised.values()) == STEPS * 12 and all(supervised[f] == STEPS // 2 for f in FAMILIES[:-1])
            and supervised['__unknown__'] == 0 and dict(lengths) == {'3': STEPS * 6, '4': STEPS * 6},
            'Known3/4 supervision budget differs from fixed per-family length cycle')
    model.cpu(); state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    after = parameter_groups(state)
    require(all(after[key] != before[key] for key in before), 'A CNN parameter group did not train')
    torch.save({'state_dict': state, 'architecture': ARCHITECTURE, 'families': FAMILIES,
                'step': STEPS, 'training_protocol_sha256': freeze_sha}, args.output / 'FINAL.pth')
    training_counts = {'optimizer_steps': STEPS, 'replay_rows': sum(counts.values()),
                       'short_rows': sum(short_counts.values()), 'short_supervised_rows': sum(supervised.values()),
                       'short_supervised_by_family': dict(supervised), 'short_supervised_by_glyph_count': dict(lengths),
                       'teacher_eligible_rows': dict(teacher_eligible), 'replay_by_family': dict(counts),
                       'replay_unknown_by_source': dict(sources), 'short_by_family': dict(short_counts),
                       'short_sampling': short_sampler.report(), 'replay_sampling': sampler.report()}
    write_new(args.output / 'TRAINING_COUNTS.json', training_counts)
    print(json.dumps({'optimizer_finished': True, 'step': STEPS, 'calibration_starting': True}), flush=True)
    cal = load_split(SOURCES['original'][0], 'calibration')
    retention = read(RETENTION_PLAN)
    with np.load(BASE_RUN / 'CALIBRATION_OUTPUTS.npz', allow_pickle=False) as cached:
        baseline, _, base_details = evaluate_outputs(cached['logits'], cached['log_em_ratio'], cal, retention, 0)
    require(baseline['metrics'] == base_selection['selected']['metrics'], 'R22 CAL baseline could not be reproduced')
    baseline_short = short_population(base_details, cal['rows'])
    model.to(args.device).eval()
    logits, ratios = infer(model, cal, args.device)
    record, outputs, details = evaluate_outputs(logits, ratios, cal, retention, STEPS)
    measured_short = short_population(details, cal['rows'])
    additional = compare_r22(record['metrics'], baseline['metrics'], measured_short, baseline_short)
    require(len(record['retention_checks']) == 46, 'Original CAL acceptance rule count changed')
    promotion = record['promotion_allowed'] and additional['passed']
    np.savez(args.output / 'CALIBRATION_OUTPUTS.npz', logits=logits, log_em_ratio=ratios)
    write_new(args.output / 'CALIBRATION_DECISIONS.json', {'families': FAMILIES, 'records': outputs})
    require(all(sha(path) == digest for path, digest in bindings.items())
            and sha(args.output / 'TRAINING_FREEZE.json') == freeze_sha, 'Frozen inputs changed during training')
    selection = {**protocol, 'schema': 'flux-glyph-unified-short-selection-v3', 'selected': record,
                 'history': [record], 'calibration_promotion_allowed': bool(promotion),
                 'promotion_allowed': False,
                 'passed': record['metrics']['passed'], 'calibration_passed': record['metrics']['passed'],
                 'training_protocol_sha256': freeze_sha, 'state_after_sha256': state_sha(state),
                 'selected_parameter_groups_sha256': after, 'short_baseline': baseline_short,
                 'short_metrics': measured_short, 'r22_comparison': additional,
                 'baseline_metrics': baseline['metrics'], 'optimizer_steps_executed': STEPS,
                 'final_checkpoint_sha256': sha(args.output / 'FINAL.pth'),
                 'training_counts_sha256': sha(args.output / 'TRAINING_COUNTS.json'),
                 'calibration_outputs_sha256': sha(args.output / 'CALIBRATION_OUTPUTS.npz'),
                 'calibration_decisions_sha256': sha(args.output / 'CALIBRATION_DECISIONS.json'),
                 'data_manifest_sha256': cal['manifest_sha256'],
                 'calibration_partition_sha256': cal['partition_sha256'],
                 'development_evaluated': False, 'exported': False, 'deployed': False}
    write_new(args.output / 'SELECTION.json', selection)
    torch.save({'state_dict': state, 'architecture': ARCHITECTURE, 'families': FAMILIES,
                'selection_sha256': sha(args.output / 'SELECTION.json')}, args.output / 'model.pth')
    report = {'status': 'CAL_PASSED_AWAITING_EXPORT_AND_DEVELOPMENT' if promotion else 'NOT_PROMOTABLE',
              'original_checks_passed': sum(v['passed'] for v in record['retention_checks']),
              'additional_checks_passed': sum(v['passed'] for v in additional['checks']),
              'original_checks': 46, 'additional_checks': 7, 'short_before': baseline_short,
              'short_after': measured_short, 'failed_original': [c for c in record['retention_checks'] if not c['passed']],
              'failed_additional': [c for c in additional['checks'] if not c['passed']],
              'checkpoint_sha256': sha(args.output / 'model.pth'),
              'calibration_promotion_allowed': bool(promotion), 'promotion_allowed': False,
              'test_read': False, 'exported': False, 'deployed': False}
    write_new(args.output / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'train'))
    for field in ('teacher', 'short', 'plan', 'output'):
        parser.add_argument('--' + field, type=lambda s: Path(s).resolve(), required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    args = parser.parse_args()
    (preflight if args.action == 'preflight' else train)(args)
