#!/usr/bin/env python3
"""Fit zero-sum spatial features and the existing unknown row with verified-TRAIN repair losses."""
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
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'training')]
import train_unified_spatial_detail_v2 as parent
from spatial_rejection_network import (POLICY, add_residual, fold_residual, frozen_state,
    residual_check, CachedSpatialHead)
from cache_spatial_train_features import load_features
from spatial_balanced_objective import POLICY as REPAIR_POLICY, objective as repair_objective
from cache_r22_train_teacher import read, write_new
from train_regions import require, sha, state_sha

CONFIGS = {'balanced-low': 8e-5, 'balanced-medium': 8e-5, 'balanced-high': 8e-5}
REPAIR_WEIGHTS = {'balanced-low': .25, 'balanced-medium': .5, 'balanced-high': 1.}
STEPS = 2400
CACHE_MANIFEST_SHA = '9a881a03ab55e286e38fd7ede35ddd79e31390b452e3db1967acb1eb1cc59f52'
OBJECTIVE = {**parent.OBJECTIVE, 'all_parameters_trained': False,
             'trainable_policy': POLICY, 'repair_supervision': REPAIR_POLICY,
             'repair_confusion_weights': REPAIR_WEIGHTS,
             'rationale': 'Reduce extra Kaiti CE to 0.05 to avoid over-generalizing rare Kaiti examples; preserve all other verified-TRAIN and runtime contracts.'}


def bound_sources(args, datasets, feature_manifest):
    bindings = parent.files_bound(args, datasets)
    for path in [Path(__file__), ROOT / 'training/spatial_rejection_network.py',
                 ROOT / 'training/spatial_residual_network.py',
                 ROOT / 'training/spatial_rejection_objective.py',
                 ROOT / 'training/spatial_balanced_objective.py',
                 ROOT / 'training/train_spatial_rejection.py',
                 ROOT / 'training/cache_spatial_train_features.py',
                 args.cache / 'MANIFEST.json', args.cache / 'CACHE_FREEZE.json']:
        bindings[str(path.resolve())] = sha(path)
    bindings.update(feature_manifest['bindings'])
    for entry in feature_manifest['files'].values():
        bindings[str((args.cache / entry['path']).resolve())] = entry['sha256']
    return bindings


def feature_batch(raw, arrays, device):
    import torch
    indices = [*raw['indices'], *raw['focus_indices'],
        *((r['short_dataset'], r['tile_start']) for r in raw['mobile_rows']),
        *((r['short_dataset'], r['tile_start']) for r in raw['pair_rows'])]
    require(len(indices) == 180, 'Expected exactly the original 180 TRAIN image identities')
    pooled = np.stack([arrays[name][index] for name, index in indices])
    require(pooled.shape == (180, 8192) and np.isfinite(pooled).all(), 'Invalid frozen features')
    batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value)
             for key, value in raw.items() if key != 'images'}
    batch['images'] = torch.from_numpy(pooled).to(device)
    return batch


def next_batch(context, arrays, device):
    raw = parent.inputs(context[2], context[3], context[4], context[5], context[6], context[12], 'cpu')
    return raw, feature_batch(raw, arrays, device)


class Counts:
    def __init__(self):
        self.replay = Counter(); self.unknown = Counter(); self.focus = Counter()
        self.mobile = Counter(); self.sources = Counter(); self.platforms = Counter()
        self.steps = self.eligible = self.oe = 0

    def update(self, batch, parts):
        self.steps += 1
        self.replay.update(r['family'] for r in batch['rows'])
        self.unknown.update(r['source_font_family'] for r in batch['rows'] if r['target'] == 24)
        self.focus.update(r['family'] for r in batch['focus_rows'])
        parent._mobile_counts(batch['mobile_rows'], self.mobile, self.sources, self.platforms)
        self.eligible += int(parts['focus_teacher_eligible'].detach())
        self.oe += int(parts['unknown_oe_rows'].detach())

    def report(self, context):
        _, _, _, replay, focus, _, mobile, _, _, _, _, _, pairs, _, _, _ = context
        mobile_report, pair_report = mobile.report(), pairs.report()
        parent._require_mobile_budget(mobile_report, self.mobile, self.sources, self.platforms, self.steps)
        return {'optimizer_steps': self.steps, 'replay_rows': sum(self.replay.values()),
            'unknown_oe_rows': self.oe, 'unknown_oe_policy': parent.UNKNOWN_OE_POLICY,
            'unknown_oe_multiplier': parent.OE_MULTIPLIER, 'effective_unknown_oe_coefficient': .125,
            'mobile_rows': sum(self.mobile.values()), 'mobile_unknown_rows': self.mobile['__unknown__'],
            'mobile_by_family': dict(self.mobile), 'mobile_by_source': dict(self.sources),
            'mobile_by_platform': dict(self.platforms), 'mobile_sampling': mobile_report,
            'pair_count': self.steps * 16, 'pair_rows': self.steps * 32,
            'pair_policy': parent.PAIR_POLICY, 'pair_sampling': pair_report,
            'focus_rows': sum(self.focus.values()), 'focus_supervised_rows': sum(self.focus.values()),
            'focus_by_family': dict(self.focus), 'focus_by_glyph_count': focus.report()['by_glyph_count'],
            'focus_teacher_eligible_rows': self.eligible,
            'replay_by_family': dict(self.replay), 'replay_unknown_by_source': dict(self.unknown),
            'focus_sampling': focus.report(), 'replay_sampling': replay.report()}


def setup(args):
    import torch
    torch.set_num_threads(4); torch.manual_seed(parent.SEED)
    context = parent.context(args)
    require(sha(args.cache / 'MANIFEST.json') == CACHE_MANIFEST_SHA,
            'Only the pinned verified float32 TRAIN feature cache may be used')
    arrays, manifest = load_features(args.cache)
    require(manifest['sources'] == {'original':130854,'supplement':3156,'known':4820,'ios':2380,'android':4140}
            and manifest['feature_dtype'] == 'float32', 'TRAIN feature population or dtype changed')
    for name, entry in manifest['files'].items():
        data = context[1][name] if name in context[1] else context[5]['datasets'][name]
        path = str(Path(data['tiles'].filename).resolve())
        require(entry['source_pixels_path'] == path
                and entry['source_pixels_sha256'] == manifest['bindings'].get(path),
                'Features no longer correspond to the loaded TRAIN pixels')
    require(manifest['spatial_initial_state_sha256'] == state_sha(context[0].state_dict()),
            'Feature cache does not describe this exact inherited spatial model')
    bindings = bound_sources(args, context[1], manifest)
    model = add_residual(context[0])
    initial_folded = state_sha(context[0].state_dict())
    before = state_sha(frozen_state(model))
    model.to(args.device).train()
    return context, arrays, manifest, bindings, model, before, initial_folded


def optimizer_step(model, head, batch, optimizer, confusion_weight):
    import torch
    loss, parts = repair_objective(head, batch, confusion_weight)
    require(bool(torch.isfinite(loss)), 'Nonfinite residual objective')
    optimizer.zero_grad(set_to_none=True); loss.backward()
    trainable = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            require(name in POLICY['trainable'] and parameter.grad is not None
                    and bool(torch.isfinite(parameter.grad).all()), 'Invalid residual gradient')
            trainable.append(parameter)
        else:
            require(parameter.grad is None, 'An inherited parameter received gradients')
    require(len(trainable) == 3, 'Only the declared spatial and unknown-row residuals may train')
    norm = torch.nn.utils.clip_grad_norm_(trainable, 5.)
    require(bool(torch.isfinite(norm)), 'Nonfinite residual gradient norm')
    optimizer.step()
    return loss, parts, float(norm.detach())


def preflight(args):
    import torch
    require(not args.output.exists(), 'Preserve earlier residual preflights')
    args.output.mkdir(parents=True)
    plans = {}
    for trial, rate in CONFIGS.items():
        context, arrays, cache_manifest, bindings, model, before, initial = setup(args)
        head = CachedSpatialHead(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=rate, weight_decay=1e-4)
        traces = []; counts = Counts(); maximum = {'font': 0., 'size': 0.}
        for step in range(4):
            raw, batch = next_batch(context, arrays, args.device)
            with torch.no_grad():
                direct = model(raw['images'].to(args.device))
                cached = parent.forward_with_features(head, batch['images'])[:2]
                for key, a, b in zip(('font', 'size'), direct, cached):
                    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-5)
                    maximum[key] = max(maximum[key], float((a-b).abs().max().cpu()))
            with torch.no_grad():
                old_loss, old_parts = parent.objective(head, batch)
                _, new_parts = repair_objective(head, batch, REPAIR_WEIGHTS[trial])
                torch.testing.assert_close(old_loss, new_parts['original_total'], atol=0, rtol=0)
                for key in old_parts:
                    torch.testing.assert_close(old_parts[key], new_parts[key], atol=0, rtol=0)
            loss, parts, gradient = optimizer_step(model, head, batch, optimizer, REPAIR_WEIGHTS[trial])
            require(gradient > 0, 'Residual never received a learning signal')
            counts.update(batch, parts)
            traces.append({'step': step+1, 'loss': float(loss.detach()), 'gradient_norm': gradient,
                'parts': {k: float(v.detach()) for k,v in parts.items()}})
        model.cpu(); proof = residual_check(model)
        require(proof['max_absolute_residual'] > 0 and state_sha(frozen_state(model)) == before,
                'Preflight did not learn only the residual')
        folded = fold_residual(model).eval(); model.eval()
        with torch.no_grad():
            for a,b in zip(model(raw['images'][:8]), folded(raw['images'][:8])):
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        actual = counts.report(context)
        require(actual['mobile_sampling'] == context[11]['preflight']
                and actual['pair_sampling'] == context[14]['preflight']
                and actual['focus_teacher_eligible_rows'] == 114,
                'Residual preflight changed the original sampling trajectory')
        require(all(sha(p) == h for p,h in bindings.items()), 'Preflight mutated a frozen source')
        planned = {'schema': 'flux-glyph-spatial-balanced-plan-v1', 'trial': trial,
            'learning_rate': rate, 'minimum_learning_rate': rate / 10, 'steps': STEPS,
            'seed': parent.SEED, 'policy': POLICY, 'objective': OBJECTIVE,
            'architecture': parent.ARCHITECTURE, 'spatial_transfer': context[15],
            'initializer_checkpoint_sha256': parent.BASE_SHA,
            'initializer_selection_sha256': parent.BASE_SELECTION_SHA,
            'initializer_state_sha256': parent.BASE_STATE_SHA,
            'inherited_spatial_state_sha256': initial, 'frozen_parameter_state_sha256': before,
            'feature_manifest_sha256': sha(args.cache / 'MANIFEST.json'),
            'feature_cache': str(args.cache), 'runtime': parent.FIXED_RUNTIME,
            'retention_plan_sha256': parent.RETENTION_SHA,
            'additional_acceptance': parent.EXTRA_ACCEPTANCE,
            'android_system_guard_policy': parent.ANDROID_SYSTEM_GUARD,
            'prior_training_counts_sha256': parent.PRIOR_COUNTS_SHA,
            'selection_policy': 'fixed final step only; first fully CAL-qualified trial in declared ascending supervised confusion-penalty order',
            'grid': CONFIGS, 'repair_weights': REPAIR_WEIGHTS,
            'repair_confusion_weight': REPAIR_WEIGHTS[trial], 'bindings': bindings, 'test_read': False,
            'development_read': False, 'calibration_reused_not_blind': True,
            'preflight': {'passed': True, 'steps': 4, 'traces': traces, 'residual': proof,
                'frozen_parameters_unchanged': True, 'folding_parity_passed': True,
                'original_objective_equality_passed': True,
                'feature_reuse_maximum_errors': maximum, 'counts': actual}}
        path = args.output / (trial + '.json'); write_new(path, planned)
        plans[trial] = {'path': str(path), 'sha256': sha(path)}
        print({'preflight_passed': trial, 'residual': proof}, flush=True)
    write_new(args.output / 'GRID.json', {'schema': 'flux-glyph-spatial-balanced-grid-v1',
        'plans': plans, 'order': list(CONFIGS), 'selection_policy': planned['selection_policy'],
        'calibration_read': False, 'development_read': False, 'test_read': False})


def calibrate(model, run, protocol, counts, residual_report, state):
    import torch
    cal = parent.load_split(parent.SOURCES['original'][0], 'calibration')
    retention = read(parent.RETENTION_PLAN)
    with np.load(parent.BASE_RUN / 'CALIBRATION_OUTPUTS.npz', allow_pickle=False) as saved:
        baseline, _, base_details = parent.evaluate_outputs(saved['logits'], saved['log_em_ratio'], cal, retention, 0)
    require(baseline['metrics'] == read(parent.BASE_RUN / 'SELECTION.json')['selected']['metrics'],
            'R22 CAL baseline changed')
    base_short = parent.short_population(base_details, cal['rows'])
    base_guard = parent.android_system_guard(base_details)
    require(base_guard['actual'] == 27, 'R22 Android guard changed')
    font, size = parent.infer(model, cal, protocol['device'])
    record, outputs, details = parent.evaluate_outputs(font, size, cal, retention, STEPS)
    short = parent.short_population(details, cal['rows'])
    added = parent.compare_r22(record['metrics'], baseline['metrics'], short, base_short)
    guard = parent.android_system_guard(details)
    require(len(record['retention_checks']) == 46 and len(added['checks']) == 7, 'CAL rules changed')
    cal53 = bool(record['promotion_allowed'] and added['passed'])
    allowed = cal53 and guard['passed']
    np.savez(run / 'CALIBRATION_OUTPUTS.npz', logits=font, log_em_ratio=size)
    write_new(run / 'CALIBRATION_DECISIONS.json', {'families': parent.FAMILIES, 'records': outputs})
    selection = {**protocol, 'schema': 'flux-glyph-spatial-balanced-selection-v1',
        'selected': record, 'history': [record], 'short_metrics': short, 'short_baseline': base_short,
        'baseline_metrics': baseline['metrics'], 'r22_comparison': added,
        'android_system_guard': guard, 'android_system_guard_baseline': base_guard,
        'original_53_calibration_checks_passed': cal53, 'calibration_promotion_allowed': allowed,
        'promotion_allowed': False, 'development_evaluated': False, 'test_read': False,
        'exported': False, 'deployed': False, 'state_after_sha256': state_sha(state),
        'selected_parameter_groups_sha256': parent.parameter_groups(state),
        'optimizer_steps_executed': STEPS, 'residual_training': residual_report,
        'training_protocol_sha256': sha(run / 'TRAINING_FREEZE.json'),
        'training_counts_sha256': sha(run / 'TRAINING_COUNTS.json'),
        'final_checkpoint_sha256': sha(run / 'FINAL.pth'),
        'residual_checkpoint_sha256': sha(run / 'RESIDUAL_TRAINING.pth'),
        'calibration_outputs_sha256': sha(run / 'CALIBRATION_OUTPUTS.npz'),
        'calibration_decisions_sha256': sha(run / 'CALIBRATION_DECISIONS.json'),
        'data_manifest_sha256': cal['manifest_sha256'], 'calibration_partition_sha256': cal['partition_sha256']}
    require(all(sha(p) == h for p,h in protocol['bindings'].items()), 'A frozen training source changed')
    write_new(run / 'SELECTION.json', selection)
    torch.save({'state_dict': state, 'architecture': parent.ARCHITECTURE, 'families': parent.FAMILIES,
                'selection_sha256': sha(run / 'SELECTION.json')}, run / 'model.pth')
    report = {'status': 'CAL_PASSED_AWAITING_EXPORT_AND_DEVELOPMENT' if allowed else 'NOT_PROMOTABLE',
        'original_checks_passed': sum(c['passed'] for c in record['retention_checks']), 'original_checks': 46,
        'additional_checks_passed': sum(c['passed'] for c in added['checks']), 'additional_checks': 7,
        'android_system_guard': guard, 'short_before': base_short, 'short_after': short,
        'failed_original': [c for c in record['retention_checks'] if not c['passed']],
        'failed_additional': [c for c in added['checks'] if not c['passed']],
        'checkpoint_sha256': sha(run / 'model.pth'), 'calibration_promotion_allowed': allowed,
        'promotion_allowed': False, 'development_evaluated': False, 'test_read': False,
        'exported': False, 'deployed': False}
    write_new(run / 'report.json', report)
    print(json.dumps({'trial': protocol['trial'], **report}, ensure_ascii=False), flush=True)
    return allowed


def train(args):
    import torch
    require(not args.output.exists(), 'Preserve previous residual grids')
    grid = read(args.plan / 'GRID.json')
    require(grid['order'] == list(CONFIGS), 'Frozen trial order changed')
    args.output.mkdir(parents=True)
    selected = None; results = {}
    for trial, rate in CONFIGS.items():
        planned_path = Path(grid['plans'][trial]['path'])
        require(sha(planned_path) == grid['plans'][trial]['sha256'], 'Trial preflight/plan changed')
        plan = read(planned_path)
        context, arrays, manifest, bindings, model, before, initial = setup(args)
        require(plan['bindings'] == bindings and plan['policy'] == POLICY and plan['objective'] == OBJECTIVE
                and plan['learning_rate'] == rate and plan['steps'] == STEPS and plan['grid'] == CONFIGS
                and plan['repair_weights'] == REPAIR_WEIGHTS
                and plan['repair_confusion_weight'] == REPAIR_WEIGHTS[trial]
                and plan['runtime'] == parent.FIXED_RUNTIME and plan['preflight']['passed'] is True
                and plan['frozen_parameter_state_sha256'] == before
                and plan['inherited_spatial_state_sha256'] == initial
                and plan['feature_manifest_sha256'] == sha(args.cache / 'MANIFEST.json'),
                'Training differs from the actual preflight and frozen inputs')
        bindings[str(planned_path)] = sha(planned_path)
        bindings[str((args.plan / 'GRID.json').resolve())] = sha(args.plan / 'GRID.json')
        run = args.output / trial; run.mkdir()
        protocol = {**plan, 'schema': 'flux-glyph-spatial-balanced-training-freeze-v1',
            'bindings': bindings, 'device': args.device, 'families': parent.FAMILIES,
            'training_teacher_proof': context[8], 'native_focus_proof': context[9],
            'mobile_short_proof': context[10], 'native_font_pair_proof': context[13],
            'initial_parameter_groups_sha256': parent.parameter_groups(context[0].state_dict())}
        write_new(run / 'TRAINING_FREEZE.json', protocol)
        head = CachedSpatialHead(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=rate/10)
        counts = Counts(); started = time.monotonic()
        for step in range(1, STEPS+1):
            raw, batch = next_batch(context, arrays, args.device)
            loss, parts, gradient = optimizer_step(model, head, batch, optimizer, REPAIR_WEIGHTS[trial])
            scheduler.step(); counts.update(batch, parts)
            if step % 200 == 0:
                print(json.dumps({'trial': trial, 'step': step, 'loss': float(loss.detach()),
                    'gradient_norm': gradient, 'seconds': round(time.monotonic()-started, 1)}), flush=True)
        actual = counts.report(context)
        require(sha(parent.PRIOR_COUNTS) == parent.PRIOR_COUNTS_SHA and actual == read(parent.PRIOR_COUNTS),
                'Residual trial changed the original training sampling')
        model.cpu(); proof = residual_check(model)
        require(state_sha(frozen_state(model)) == before and proof['max_absolute_residual'] > 0,
                'An inherited parameter changed or the residual did not train')
        write_new(run / 'TRAINING_COUNTS.json', actual)
        torch.save({'state_dict': model.state_dict(), 'policy': POLICY,
                    'training_protocol_sha256': sha(run / 'TRAINING_FREEZE.json')}, run / 'RESIDUAL_TRAINING.pth')
        model.eval(); folded = fold_residual(model).eval()
        with torch.no_grad():
            for a,b in zip(model(raw['images'][:16]), folded(raw['images'][:16])):
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        state = {k:v.detach().clone() for k,v in folded.state_dict().items()}
        for key, value in context[0].state_dict().items():
            if key in ('family_head.weight', 'family_head.bias'):
                require(torch.equal(state[key][:24], value[:24]), 'A known classifier row changed: '+key)
            elif key != 'style.0.weight':
                require(torch.equal(state[key], value), 'Folded inherited parameter changed: '+key)
        error = float((state['style.0.weight'].reshape(384,128,4,4,4).sum(-1) -
                       context[0].state_dict()['style.0.weight'].reshape(384,128,4,4,4).sum(-1)).abs().max())
        require(error <= 1e-6, 'Folded coarse horizontal coefficients changed')
        proof.update(frozen_parameters_unchanged=True, folded_output_parity_passed=True,
            folded_coarse_weight_error=error, training_seconds=round(time.monotonic()-started,1))
        torch.save({'state_dict': state, 'architecture': parent.ARCHITECTURE, 'families': parent.FAMILIES,
            'step': STEPS, 'training_protocol_sha256': sha(run / 'TRAINING_FREEZE.json')}, run / 'FINAL.pth')
        del model, head, optimizer, scheduler
        print({'trial': trial, 'optimizer_finished': True, 'calibration_starting': True}, flush=True)
        allowed = calibrate(folded.to(args.device), run, protocol, actual, proof, state)
        results[trial] = {'path': str(run), 'report_sha256': sha(run / 'report.json'), 'calibration_passed': allowed}
        if allowed:
            selected = trial
            break
    write_new(args.output / 'GRID_RESULT.json', {'schema': 'flux-glyph-spatial-balanced-grid-result-v1',
        'selected': selected, 'results': results, 'grid_sha256': sha(args.plan / 'GRID.json'),
        'selection_policy': grid['selection_policy'], 'exported': False,
        'development_evaluated': False, 'test_read': False, 'deployed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight','train'])
    parser.add_argument('--output', type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument('--plan', type=lambda p: Path(p).resolve())
    parser.add_argument('--cache', type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument('--teacher', type=lambda p: Path(p).resolve(),
        default=(ROOT / 'artifacts/unified-font-v4/r22-train-teacher-v1').resolve())
    parser.add_argument('--ios-data', type=lambda p: Path(p).resolve(),
        default=(ROOT / 'artifacts/ios-native-short-train-v1/prepared').resolve())
    parser.add_argument('--android-data', type=lambda p: Path(p).resolve(),
        default=(ROOT / 'artifacts/android-native-short-train-v1/prepared').resolve())
    parser.add_argument('--android-manifest-sha', default='09584bd629c966148ed27a2c0a82698b3e09bc0f7b5e8c974f0bc0b20ceae872')
    parser.add_argument('--device', choices=['cpu','mps'], default='mps')
    args = parser.parse_args()
    (preflight if args.action == 'preflight' else train)(args)
