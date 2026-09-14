#!/usr/bin/env python3
"""Train verified native iOS short lines with frozen R22 replay and Android focus."""
from __future__ import annotations

import argparse
from collections import Counter
import json
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
from short_region_objective import EXTRA_ACCEPTANCE, short_population, compare_r22
from native_short_focus import FOCUS_KNOWN_FAMILIES, SAMPLING as FOCUS_SAMPLING, build_focus, NativeShortFocusSampler
from restore_unknown_train_teacher import load_replay_teacher
from native_unknown_oe import POLICY as UNKNOWN_OE_POLICY, unknown_named_uniformity
from ios_short_training import SAMPLING as IOS_SAMPLING, load_prepared, IOSShortSampler

RETENTION_PLAN = ROOT / 'artifacts/unified-font-v2/RETENTION_PLAN.json'
RETENTION_SHA = 'b82283b54b5e47f4639b0df62231544c0419f6299d10820d86dca96ef24094ff'
STEPS = 2400
SEED = 2026091401
LEARNING_RATE = 5e-6
MINIMUM_LEARNING_RATE = 5e-7
FOCUS_WEIGHT = .5
IOS_WEIGHT = .25
OE_MULTIPLIER = .25
EXPECTED_FOCUS_TEACHER_ELIGIBLE = 68712
OBJECTIVE = {
    'replay_batch': 96, 'focus_batch': 32, 'ios_batch': 24, 'focus_weight': FOCUS_WEIGHT,
    'ios_weight': IOS_WEIGHT, 'ios_sampling': IOS_SAMPLING, 'unknown_oe_multiplier': OE_MULTIPLIER,
    'effective_unknown_oe_coefficient': UNKNOWN_OE_POLICY['coefficient'] * OE_MULTIPLIER,
    'unknown_oe': UNKNOWN_OE_POLICY,
    'replay_loss': 'unchanged R22 supervised, unknown-floor, angular, size and correct-teacher full KL2',
    'focus_loss': 'same full R22 loss on complete native single-tile TRAIN regions',
    'replay_teacher': {'original_true_unknown': 'original frozen R21 same-tile TRAIN logits',
                      'original_known': 'frozen R22 same-tile TRAIN logits',
                      'supplement_all': 'frozen R22 same-tile TRAIN logits',
                      'merged_known_all': 'frozen R22 same-tile TRAIN logits',
                      'selection_input': 'verified original TRAIN partition and true label only'},
    'focus_teacher': 'same verified source/tile teacher as replay; no parent-crop teacher transfer',
    'focus_sampling': FOCUS_SAMPLING,
    'focus_normalization': '32 outputs repeated three times in the unchanged 96-row mean loss; no duplicated CNN forward or additional unique samples',
    'ios_loss': '.25 * (mean CE25 + .2 * mean smooth-L1 size beta .05); verified native labels only',
    'rationale': 'Joint native-iOS short supervision plus reduced replay-only unknown OE; not a causal control.',
    'replay_rng_unchanged': True, 'focus_rng_independent': True,
    'derived_short_crops_used': False, 'all_parameters_trained': True,
    'label_sources': 'verified native TRAIN font/glyph proof, never predictions',
    'teacher_cache_deployed': False, 'second_model_resident_during_optimizer': False,
    'model_count': 1, 'ocr_recognition': False, 'runtime_changes': False,
    'arbitrary_confidence_boost': False, 'original_full_correct_teacher_kl_preserved': True,
}


def files_bound(args, datasets):
    bindings = dict(read(BASE_RUN / 'SELECTION.json')['bindings'])
    _, teacher_proof = load_replay_teacher(args.teacher, datasets)
    _, focus_proof = build_focus(datasets)
    _, ios_proof = load_prepared(args.ios_data)
    for additional in (source_bindings(datasets), teacher_proof['bindings'], focus_proof['bindings'], ios_proof['bindings']):
        for path, digest in additional.items():
            require((path not in bindings or bindings[path] == digest) and sha(path) == digest,
                    'Native TRAIN or teacher proof conflicts with frozen evidence')
            bindings[path] = digest
    files = [Path(__file__), ROOT / 'training/native_short_focus.py', ROOT / 'training/ios_short_training.py',
        ROOT / 'training/native_unknown_oe.py', ROOT / 'training/train_unified_native_oe.py',
        ROOT / 'artifacts/unified-font-v4/run-native-oe-v1/SELECTION.json',
        ROOT / 'artifacts/unified-font-v4/NATIVE_OE_COMPLETION.json',
        ROOT / 'artifacts/unified-font-v4/native-unknown-train-probe-v1/REPORT.json',
        ROOT / 'artifacts/unified-font-v4/native-unknown-diagnosis-v1/REPORT.json',
        ROOT / 'artifacts/ios-native-short-train-v1/requests/REQUESTS.json',
        ROOT / 'artifacts/ios-native-short-train-v1/requests/Scenes.json',
        ROOT / 'artifacts/ios-native-short-train-v1/probe_r22_train.py',
        ROOT / 'artifacts/ios-native-short-train-v1/r22-train-probe-v1/REPORT.json',
        ROOT / 'artifacts/ios-native-short-train-v1/sampling-reconstruction-v1/REPORT.json',
        ROOT / 'artifacts/unified-font-v4/native-short-reconstruction-v1/REPORT.json',
        ROOT / 'training/capture/generate_ios_short_scenes.py',
        ROOT / 'training/train_unified_short_regions_v3.py', ROOT / 'training/restore_unknown_train_teacher.py',
        ROOT / 'training/short_region_objective.py',
        ROOT / 'artifacts/unified-font-v4/run-short-regions-v3/SELECTION.json',
        ROOT / 'artifacts/unified-font-v4/short-v3-comparison-v1/REPORT.json',
        BASE_RUN / 'CALIBRATION_OUTPUTS.npz', BASE_RUN / 'CALIBRATION_DECISIONS.json',
        BASE_RUN / 'DEVELOPMENT_REGRESSION.json', RETENTION_PLAN,
        args.teacher / 'CACHE_FREEZE.json', args.teacher / 'CACHE_MANIFEST.json']
    for part in read(args.teacher / 'CACHE_MANIFEST.json')['partitions'].values():
        files += [args.teacher / value['path'] for value in part['files'].values()]
    for path in files:
        digest = sha(path); key = str(path.resolve())
        require(key not in bindings or bindings[key] == digest,
                'New training source conflicts with frozen R22 evidence')
        bindings[key] = digest
    return bindings


def context(args):
    require(sha(RETENTION_PLAN) == RETENTION_SHA, 'Original 46 retention rules changed')
    datasets = load_training()
    teacher, teacher_proof = load_replay_teacher(args.teacher, datasets)
    focus, focus_proof = build_focus(datasets)
    model, base_selection = load_base()
    replay = {name: batch_source(data, {'base_logits': teacher[name]['logits']}) for name, data in datasets.items()}
    plan = read(RETENTION_PLAN)
    sampler = FaceBalancedSampler(datasets['original']['rows'], FAMILIES, SEED, plan['train_pools'],
                                  datasets['supplement']['rows'], datasets['known']['rows'])
    focus_sampler = NativeShortFocusSampler(focus, SEED + 1)
    ios, ios_proof = load_prepared(args.ios_data)
    ios_sampler = IOSShortSampler(ios, SEED + 2)
    return model, datasets, replay, sampler, focus_sampler, ios, ios_sampler, base_selection, teacher_proof, focus_proof, ios_proof


def inputs(replay, sampler, focus_sampler, ios, ios_sampler, device):
    import torch
    rows = sampler.batch()
    images, teacher, sizes, indices = pixel_batch(rows, replay['original'], replay['supplement'], replay['known'],
                                                 sampler.rng, return_indices=True)
    focus_rows = focus_sampler.batch()
    require(len(focus_rows) == 32 and all(r['tile_count'] == 1 for r in focus_rows),
            'Native focus requires exactly32 complete single-tile TRAIN rows')
    # Reuse the frozen image/teacher identity validator. Each focus tile has one
    # deterministic index; triplication does not select extra images or views.
    fx, ft, fs, fi = pixel_batch(focus_rows * 3, replay['original'], replay['supplement'], replay['known'],
                                focus_sampler.rng, return_indices=True)
    require(all(np.array_equal(x[:32], x[32:64]) and np.array_equal(x[:32], x[64:]) for x in (fx, ft, fs))
            and fi[:32] == fi[32:64] == fi[64:], 'Repeated focus validation changed the paired tile')
    ios_rows = ios_sampler.batch()
    require(len(ios_rows) == 24 and all(r['tile_count'] == 1 for r in ios_rows),
            'iOS batch must contain exactly24 verified single-tile TRAIN rows')
    ix = np.stack([ios['tiles'][r['tile_start']] for r in ios_rows])
    tensor = lambda values: torch.from_numpy(np.asarray(values)).to(device)
    return {'images': tensor(np.concatenate((images, fx[:32],ix))), 'teacher': tensor(teacher),
        'sizes': tensor(sizes), 'targets': tensor(np.array([r['target'] for r in rows], dtype=np.int64)),
        'focus_teacher': tensor(ft[:32]), 'focus_sizes': tensor(fs[:32]),
        'focus_targets': tensor(np.array([r['target'] for r in focus_rows], dtype=np.int64)),
        'rows': rows, 'focus_rows': focus_rows, 'ios_rows': ios_rows,
        'ios_targets': tensor(np.array([r['target'] for r in ios_rows], np.int64)),
        'ios_sizes': tensor(np.array([r['log_em_ratio'] for r in ios_rows], np.float32)),
        'indices': indices, 'focus_indices': fi[:32]}


def focus_full_loss(font, size, features, family_weight, targets, sizes, teacher, rows):
    require(len(rows) == 32 and font.shape == teacher.shape == (32, 25)
            and size.shape == sizes.shape == targets.shape == (32,) and features.shape[0] == 32,
            'Focus loss requires32 aligned native TRAIN outputs')
    return full_losses(font.repeat(3, 1), size.repeat(3), features.repeat(3, 1), family_weight,
                       targets.repeat(3), sizes.repeat(3), teacher.repeat(3, 1), rows * 3, FAMILIES)


def objective(model, batch):
    import torch.nn.functional as F
    font, size, features = forward_with_features(model, batch['images'])
    require(font.shape == (152, 25) and size.shape == (152,) and features.shape[0] == 152,
            'Joint optimizer requires one aligned 152-row CNN forward')
    replay = full_losses(font[:96], size[:96], features[:96], model.family_head.weight,
        batch['targets'], batch['sizes'], batch['teacher'], batch['rows'], FAMILIES)
    focus = focus_full_loss(font[96:128], size[96:128], features[96:128], model.family_head.weight,
        batch['focus_targets'], batch['focus_sizes'], batch['focus_teacher'], batch['focus_rows'])
    oe = unknown_named_uniformity(font[:96], batch['targets'], batch['rows'], FAMILIES)
    ios_ce = F.cross_entropy(font[128:152], batch['ios_targets'])
    ios_size = F.smooth_l1_loss(size[128:152], batch['ios_sizes'], beta=.05)
    ios_loss = IOS_WEIGHT * (ios_ce + .2 * ios_size)
    return replay[0] + FOCUS_WEIGHT * focus[0] + OE_MULTIPLIER * oe + ios_loss, {
        'unknown_oe': oe, 'unknown_oe_weighted': OE_MULTIPLIER * oe,
        'unknown_oe_rows': (batch['targets'] == 24).sum(),
        'replay': replay[0], 'replay_ce': replay[1], 'teacher_kl': replay[3],
        'focus_weighted': FOCUS_WEIGHT * focus[0], 'focus_ce': focus[1],
        'focus_teacher_kl': focus[3], 'focus_size': focus[2],
        'focus_teacher_eligible': focus[5][:32].sum(), 'ios_loss': ios_loss,
        'ios_ce': ios_ce, 'ios_size': ios_size}


def design(args, bindings):
    return {'schema': 'flux-glyph-unified-native-ios-training-plan-v1', 'architecture': ARCHITECTURE,
            'initializer_checkpoint_sha256': BASE_SHA, 'initializer_selection_sha256': BASE_SELECTION_SHA,
            'initializer_state_sha256': BASE_STATE_SHA, 'retention_plan_sha256': RETENTION_SHA,
            'families': FAMILIES, 'steps': STEPS, 'evaluation_step': STEPS,
            'seed': SEED, 'learning_rate': LEARNING_RATE, 'minimum_learning_rate': MINIMUM_LEARNING_RATE,
            'weight_decay': 1e-4, 'gradient_clip': 5., 'objective': OBJECTIVE,
            'focus_sampling': FOCUS_SAMPLING, 'ios_sampling': IOS_SAMPLING, 'ios_data': str(args.ios_data), 'runtime': FIXED_RUNTIME,
            'additional_acceptance': EXTRA_ACCEPTANCE, 'bindings': bindings,
            'teacher_cache': str(args.teacher), 'derived_crops_used': False,
            'checkpoint_selection': 'fixed final step 2400; no intermediate CAL search',
            'test_read': False, 'development_holdout_read': False,
            'calibration_reused_not_blind': True,
            'source': 'verified local native TRAIN screenshots: existing Android focus plus one-to-four-glyph iOS regions'}


def preflight(args):
    import torch
    require(not args.output.exists() and not args.plan.exists(), 'Keep earlier preflight and training plans')
    torch.set_num_threads(4); torch.manual_seed(SEED)
    model, datasets, replay, sampler, focus_sampler, ios, ios_sampler, _, _, focus_proof, ios_proof = context(args)
    bindings = files_bound(args, datasets)
    before = parameter_groups(model.state_dict())
    require(set(before) == {'trunk', 'style', 'family_head', 'size_head'},
            'Preflight requires the four frozen R22 parameter groups')
    model.to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    traces = []
    counts = Counter(); sources = Counter(); focus_counts = Counter(); ios_counts = Counter()
    for _ in range(4):
        batch = inputs(replay, sampler, focus_sampler, ios, ios_sampler, args.device)
        loss, parts = objective(model, batch)
        require(int(parts['unknown_oe_rows'].detach()) == 16,
                'OE preflight must use only the 16 replay unknown rows')
        require(bool(torch.isfinite(loss)), 'Nonfinite actual TRAIN preflight loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        gradients = {}
        for group in before:
            values = [p.grad for n, p in model.named_parameters() if n.startswith(group + '.')]
            require(values and all(v is not None and bool(torch.isfinite(v).all()) for v in values)
                    and any(bool(v.abs().sum() > 0) for v in values), 'Missing/nonfinite preflight gradients: ' + group)
            gradients[group] = True
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        counts.update(r['family'] for r in batch['rows'])
        sources.update(r['source_font_family'] for r in batch['rows'] if r['family'] == '__unknown__')
        focus_counts.update(r['family'] for r in batch['focus_rows'])
        ios_counts.update((r['family'], r['glyph_count']) for r in batch['ios_rows'])
        traces.append({'loss': float(loss.detach()), 'parts': {k: float(v.detach()) for k, v in parts.items()},
                       'finite_nonzero_gradient_groups': gradients})
    focus_report = focus_sampler.report(); ios_report = ios_sampler.report()
    require(sum(counts.values()) == 384 and len(sources) == 11
            and max(sources.values()) - min(sources.values()) <= 1
            and focus_report['steps'] == 4 and focus_report['rows'] == 128
            and focus_report['by_family'] == dict(focus_counts)
            and focus_report['by_glyph_count'] == {'3': 64, '4': 64}
            and ios_report['steps'] == 4 and ios_report['rows'] == 96
            and ios_report['by_family'] == {f: 32 for f in IOS_SAMPLING['families']}
            and ios_report['by_glyph_count'] == {str(n): 24 for n in (1, 2, 3, 4)}
            and sum(ios_counts.values()) == 96,
            'Preflight replay/focus/iOS source closure or row quotas differ')
    model.cpu(); after = parameter_groups(model.state_dict())
    require(all(before[key] != after[key] for key in before)
            and sha(BASE_RUN / 'model.pth') == BASE_SHA
            and all(sha(p) == h for p, h in bindings.items()), 'Preflight mutated baseline or did not train every group')
    report = {'schema': 'flux-glyph-native-ios-gradient-preflight-v1', 'passed': True, 'steps': 4,
              'device': args.device, 'replay_tiles': 384, 'focus_tiles': 128, 'ios_tiles': 96, 'traces': traces,
              'focus_supervised_tiles': 128, 'focus_sampling': focus_report,
              'focus_teacher_eligible_rows': sum(int(t['parts']['focus_teacher_eligible']) for t in traces),
              'unknown_oe_rows': sum(int(t['parts']['unknown_oe_rows']) for t in traces),
              'unknown_oe_policy': UNKNOWN_OE_POLICY,
              'unknown_oe_multiplier': OE_MULTIPLIER,
              'effective_unknown_oe_coefficient': UNKNOWN_OE_POLICY['coefficient'] * OE_MULTIPLIER,
              'replay_by_family': dict(counts), 'replay_unknown_by_source': dict(sources),
              'native_focus_proof': focus_proof, 'ios_short_proof': ios_proof, 'ios_sampling': ios_report,
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
    model, datasets, replay, sampler, focus_sampler, ios, ios_sampler, base_selection, teacher_proof, focus_proof, ios_proof = context(args)
    bindings = files_bound(args, datasets)
    planned = read(args.plan)
    require(all(planned.get(k) == v for k, v in design(args, bindings).items()), 'Training plan changed after preflight')
    pre = planned['preflight']; proof = read(pre['path'])
    require(sha(pre['path']) == pre['sha256'] and proof['passed'] is True
            and proof['bindings'] == bindings and proof['test_read'] is False, 'TRAIN gradient preflight differs')
    bindings[str(args.plan)] = sha(args.plan); bindings[pre['path']] = pre['sha256']
    before = parameter_groups(model.state_dict())
    args.output.mkdir(parents=True)
    protocol = {**planned, 'schema': 'flux-glyph-unified-native-ios-training-freeze-v1', 'bindings': bindings,
                'device': args.device, 'initial_parameter_groups_sha256': before,
                'training_teacher_proof': teacher_proof, 'native_focus_proof': focus_proof, 'ios_short_proof': ios_proof,
                'plan_sha256': sha(args.plan)}
    write_new(args.output / 'TRAINING_FREEZE.json', protocol)
    freeze_sha = sha(args.output / 'TRAINING_FREEZE.json')
    model.to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=MINIMUM_LEARNING_RATE)
    counts = Counter(); sources = Counter(); focus_counts = Counter()
    focus_teacher_eligible = 0
    unknown_oe_rows = 0; ios_counts = Counter()
    started = time.monotonic()
    for step in range(1, STEPS + 1):
        batch = inputs(replay, sampler, focus_sampler, ios, ios_sampler, args.device)
        loss, parts = objective(model, batch)
        require(int(parts['unknown_oe_rows'].detach()) == 16,
                'OE optimizer must use only the 16 replay unknown rows')
        require(bool(torch.isfinite(loss)), 'Nonfinite short-region optimizer loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        require(bool(torch.isfinite(norm)), 'Nonfinite optimizer gradients')
        optimizer.step(); scheduler.step()
        counts.update(r['family'] for r in batch['rows'])
        sources.update(r['source_font_family'] for r in batch['rows'] if r['family'] == '__unknown__')
        focus_counts.update(r['family'] for r in batch['focus_rows'])
        focus_teacher_eligible += int(parts['focus_teacher_eligible'].detach())
        unknown_oe_rows += int(parts['unknown_oe_rows'].detach())
        ios_counts.update((r['family'], r['glyph_count']) for r in batch['ios_rows'])
        if step % 100 == 0:
            print(json.dumps({'step': step, 'loss': float(loss.detach()),
                              **{k: float(v.detach()) for k, v in parts.items()},
                              'seconds': round(time.monotonic() - started, 1)}), flush=True)
    require(sum(counts.values()) == STEPS * 96 and sum(focus_counts.values()) == STEPS * 32
            and set(focus_counts) == set(FOCUS_KNOWN_FAMILIES) | {'__unknown__'}
            and all(focus_counts[f] == STEPS * 2 for f in FOCUS_KNOWN_FAMILIES)
            and focus_counts['__unknown__'] == STEPS * 4
            and len(sources) == 11 and max(sources.values()) - min(sources.values()) <= 1,
            'Replay/focus supervision row accounting differs from the fixed plan')
    require(unknown_oe_rows == STEPS * 16, 'Unknown OE must use exactly the original replay unknown rows')
    focus_report = focus_sampler.report()
    ios_report = ios_sampler.report()
    require(sum(ios_counts.values()) == STEPS * 24
            and ios_report['steps'] == STEPS and ios_report['rows'] == STEPS * 24
            and all(ios_report['by_family'][f] == STEPS * 8 for f in IOS_SAMPLING['families'])
            and all(ios_report['by_glyph_count'][str(n)] == STEPS * 6 for n in (1, 2, 3, 4)),
            'Native iOS family or length quota differs')
    require(focus_report['steps'] == STEPS and focus_report['rows'] == STEPS * 32
            and focus_report['by_family'] == dict(focus_counts)
            and focus_report['by_glyph_count'] == {'3': STEPS * 16, '4': STEPS * 16},
            'Complete native three/four-glyph budget differs')
    require(focus_teacher_eligible == EXPECTED_FOCUS_TEACHER_ELIGIBLE,
            'Native focus teacher eligibility differs from the frozen 68,712-row plan')
    model.cpu(); state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    after = parameter_groups(state)
    require(all(after[key] != before[key] for key in before), 'A CNN parameter group did not train')
    torch.save({'state_dict': state, 'architecture': ARCHITECTURE, 'families': FAMILIES,
                'step': STEPS, 'training_protocol_sha256': freeze_sha}, args.output / 'FINAL.pth')
    training_counts = {'optimizer_steps': STEPS, 'replay_rows': sum(counts.values()),
        'unknown_oe_rows': unknown_oe_rows, 'unknown_oe_policy': UNKNOWN_OE_POLICY,
        'unknown_oe_multiplier': OE_MULTIPLIER, 'effective_unknown_oe_coefficient': .125,
        'ios_rows': sum(ios_counts.values()), 'ios_by_family': ios_report['by_family'],
        'ios_by_glyph_count': ios_report['by_glyph_count'], 'ios_sampling': ios_report,
        'focus_rows': sum(focus_counts.values()), 'focus_supervised_rows': sum(focus_counts.values()),
        'focus_by_family': dict(focus_counts), 'focus_by_glyph_count': focus_report['by_glyph_count'],
        'focus_teacher_eligible_rows': focus_teacher_eligible,
        'replay_by_family': dict(counts), 'replay_unknown_by_source': dict(sources),
        'focus_sampling': focus_report, 'replay_sampling': sampler.report()}
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
    require(len(record['retention_checks']) == 46 and len(additional['checks']) == 7,
            'Original 46 plus 7 short-region CAL acceptance rules changed')
    promotion = record['promotion_allowed'] and additional['passed']
    np.savez(args.output / 'CALIBRATION_OUTPUTS.npz', logits=logits, log_em_ratio=ratios)
    write_new(args.output / 'CALIBRATION_DECISIONS.json', {'families': FAMILIES, 'records': outputs})
    require(all(sha(path) == digest for path, digest in bindings.items())
            and sha(args.output / 'TRAINING_FREEZE.json') == freeze_sha, 'Frozen inputs changed during training')
    selection = {**protocol, 'schema': 'flux-glyph-unified-native-ios-selection-v1', 'selected': record,
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
    for field in ('teacher', 'plan', 'output'):
        parser.add_argument('--' + field, type=lambda s: Path(s).resolve(), required=True)
    parser.add_argument('--ios-data', type=lambda s: Path(s).resolve(), default=(ROOT/'artifacts/ios-native-short-train-v1/prepared').resolve())
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    args = parser.parse_args()
    (preflight if args.action == 'preflight' else train)(args)
