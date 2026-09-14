#!/usr/bin/env python3
"""Measure frozen font-training gradient conflicts on TRAIN, without an optimizer."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'training')]
from train_regions import require, sha, state_sha
from cache_r22_train_teacher import (load_training, source_bindings, BASE_RUN,
    BASE_SHA, BASE_SELECTION_SHA, BASE_STATE_SHA, ARCHITECTURE, read, write_new)
from restore_unknown_train_teacher import load_replay_teacher
from native_short_focus import build_focus, NativeShortFocusSampler
from native_mobile_short_training import load_mobile, NativeMobileShortSampler
from native_font_pairs import build_pairs, NativeFontPairSampler
from retention_face_balanced_sampler import FaceBalancedSampler
from train_unified_retention_micro_recovery import batch_source, SUPPLEMENT_MARKER, KNOWN_SUPPLEMENT_MARKER
from font_gradient_diagnostics import TERMS, GROUPS, MIN_NORM, decompose, gradients, summary
import train_unified_native_pairs as frozen

BASE = ROOT / 'artifacts/unified-font-v4'
CANDIDATE = BASE / 'run-native-pairs-v1'
CHECKPOINTS = {
    'r22': {'run': str(BASE_RUN), 'checkpoint_sha256': BASE_SHA,
            'selection_sha256': BASE_SELECTION_SHA, 'state_sha256': BASE_STATE_SHA},
    'native_pairs': {'run': str(CANDIDATE),
        'checkpoint_sha256': '5a13f76bae74b95545040162e23206e342558974bec7751f50950250ca1d9091',
        'selection_sha256': 'd8831e93070e5f2a280b9848bcfc202285d7b25414777d590ad27d978915ac3c',
        'state_sha256': '6235d342e684bf8db892d70906654e6d98fe4942377fd35b13f5e1d2b0060b80'},
}
COUNTS_SHA = '0902aaee5884070c11997778be840e5e254a7e56ccaf26fe12cf7de151eaa0c1'
ANDROID_SHA = '09584bd629c966148ed27a2c0a82698b3e09bc0f7b5e8c974f0bc0b20ceae872'
SAMPLE_STEPS = tuple(1 + round(2399 * index / 31) for index in range(32))
POLICY = {'schema': 'flux-glyph-font-gradient-conflict-policy-v1',
    'sample_steps': SAMPLE_STEPS, 'sampling_seed': frozen.SEED,
    'sampling': '32 fixed, evenly spaced batches from the exact prior 2400-step TRAIN stream',
    'checkpoint_selection': 'pinned R22 initializer and final rejected native-pairs checkpoint only',
    'terms': TERMS, 'terms_include_training_coefficients': True,
    'font_supervision_includes_true_unknown_loss': True,
    'reference_gradients': ['replay_font + focus_font + mobile_font', 'mobile_font'],
    'main_parameter_group': 'shared trunk and style; inspect output heads separately',
    'groups': GROUPS, 'undefined_cosine_below_norm': MIN_NORM,
    'negative_dot_interpretation': 'local first-order conflict only, not proof of harmful generalization',
    'ablation_interpretation': 'algebraic raw-SGD direction removal only; no AdamW or multi-step causal claim',
    'optimizer_steps': 0, 'checkpoint_written': False, 'runtime_changes': False,
    'teacher_models_loaded': False, 'models_resident_simultaneously': 1,
    'calibration_rows_read': False, 'calibration_pixels_read': False,
    'development_rows_read': False, 'development_pixels_read': False, 'test_read': False,
    'historical_checkpoint_identity_metadata_read': True,
    'independent_validation_accuracy_measured': False}


def merge_bindings(destination, values):
    for path, digest in values.items():
        key = str(Path(path).resolve())
        require(key not in destination or destination[key] == digest, 'Conflicting source proof')
        destination[key] = digest


def advance_unselected(sampler, focus_sampler, mobile_sampler, pair_sampler):
    """Consume the same RNG draws as pixel_batch, without decoding unused images."""
    rows = sampler.batch()
    for row in rows:
        sampler.rng.integers(row['tile_count'])
    focus = focus_sampler.batch()
    for row in focus * 3:
        focus_sampler.rng.integers(row['tile_count'])
    mobile_sampler.batch(); pair_sampler.batch()
    return focus


def batch_proof(batch, step):
    import torch
    def tensor_sha(value):
        array = value.detach().cpu().numpy()
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    identity = lambda row: [row.get('short_dataset', row.get('domain')),
        row['source_id'], row['region_id'], row['view'], row['target']]
    return {'training_stream_step': step,
        'tensor_sha256': {key: tensor_sha(value) for key, value in batch.items() if isinstance(value, torch.Tensor)},
        'indices': batch['indices'], 'focus_indices': batch['focus_indices'],
        'identities': {key: [identity(row) for row in batch[key]]
                       for key in ('rows', 'focus_rows', 'mobile_rows', 'pair_rows')}}


def prepare_inputs():
    require(sha(frozen.RETENTION_PLAN) == frozen.RETENTION_SHA, 'Frozen retention plan changed')
    datasets = load_training()
    teacher, teacher_proof = load_replay_teacher(BASE / 'r22-train-teacher-v1', datasets)
    focus, focus_proof = build_focus(datasets)
    mobile, mobile_proof = load_mobile(ROOT / 'artifacts/ios-native-short-train-v1/prepared',
        ROOT / 'artifacts/android-native-short-train-v1/prepared', ANDROID_SHA)
    pair_pool, pair_proof = build_pairs(mobile)
    frozen.validate_pair_proof(pair_proof)
    plan = read(frozen.RETENTION_PLAN)
    sampler = FaceBalancedSampler(datasets['original']['rows'], frozen.FAMILIES, frozen.SEED,
        plan['train_pools'], datasets['supplement']['rows'], datasets['known']['rows'])
    focus_sampler = NativeShortFocusSampler(focus, frozen.SEED + 1)
    mobile_sampler = NativeMobileShortSampler(mobile, frozen.SEED + 2)
    pair_sampler = NativeFontPairSampler(pair_pool, frozen.SEED + 3)
    replay = {key: batch_source(data, {'base_logits': teacher[key]['logits']}) for key, data in datasets.items()}
    batches = []; proofs = []; eligible = 0
    for step in range(1, 2401):
        if step in SAMPLE_STEPS:
            batch = frozen.inputs(replay, sampler, focus_sampler, mobile, mobile_sampler, pair_sampler, 'cpu')
            eligible += int((batch['focus_teacher'].argmax(1) == batch['focus_targets']).sum())
            batches.append(batch); proofs.append(batch_proof(batch, step))
        else:
            focus_rows = advance_unselected(sampler, focus_sampler, mobile_sampler, pair_sampler)
            for row in focus_rows:
                # Use the actual markers declared by pixel_batch rather than inferred row labels.
                source = 'known' if row.get(KNOWN_SUPPLEMENT_MARKER) else 'supplement' if row.get(SUPPLEMENT_MARKER) else 'original'
                eligible += int(teacher[source]['logits'][row['tile_start']].argmax() == row['target'])
    counts_path = CANDIDATE / 'TRAINING_COUNTS.json'
    require(sha(counts_path) == COUNTS_SHA, 'Prior completed training counts changed')
    counts = read(counts_path)
    reports = {'replay_sampling': sampler.report(), 'focus_sampling': focus_sampler.report(),
        'mobile_sampling': mobile_sampler.report(), 'pair_sampling': pair_sampler.report()}
    require(len(batches) == 32 and all(reports[key] == counts[key] for key in reports)
            and eligible == counts['focus_teacher_eligible_rows'] == 68712,
            'Sparse reconstruction did not reproduce the exact original sampling stream')
    bindings = source_bindings(datasets)
    for proof in (teacher_proof, focus_proof, mobile_proof, pair_proof):
        merge_bindings(bindings, proof['bindings'])
    for path in (counts_path, frozen.RETENTION_PLAN, frozen.MOBILE_RECONSTRUCTION, frozen.PAIR_RECONSTRUCTION):
        merge_bindings(bindings, {str(path): sha(path)})
    for identity in CHECKPOINTS.values():
        for filename, key in (('model.pth', 'checkpoint_sha256'), ('SELECTION.json', 'selection_sha256')):
            path = Path(identity['run']) / filename
            require(sha(path) == identity[key], 'Pinned checkpoint identity changed')
            merge_bindings(bindings, {str(path): identity[key]})
    source_proof = {'original_stream_reconstructed_exactly': True, 'stream_steps': 2400,
        'focus_teacher_eligible': eligible, 'measured_batches': len(batches),
        'measured_rows_per_checkpoint': 32 * 180, 'measured_rows_are_repeated_training_samples': True,
        'counts_sha256': COUNTS_SHA, 'sampling': reports,
        'teacher_proof': teacher_proof, 'mobile_proof': mobile_proof, 'pair_proof': pair_proof}
    return batches, proofs, bindings, source_proof


def load_checkpoint(identity, device):
    import torch
    from wide_region_network import WideRegionFontClassifier
    path = Path(identity['run']) / 'model.pth'
    require(sha(path) == identity['checkpoint_sha256'], 'Checkpoint bytes changed')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    selection = read(Path(identity['run']) / 'SELECTION.json')
    require(checkpoint['architecture'] == ARCHITECTURE
            and checkpoint['families'] == frozen.FAMILIES
            and checkpoint['selection_sha256'] == identity['selection_sha256']
            and state_sha(checkpoint['state_dict']) == selection['state_after_sha256'] == identity['state_sha256'],
            'Checkpoint state, family order or selection differs')
    model = WideRegionFontClassifier(25)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval()


def run(args):
    import torch
    require(not args.output.exists(), 'Preserve prior diagnostics; choose a new output folder')
    torch.set_num_threads(4); torch.manual_seed(frozen.SEED)
    if args.device == 'mps':
        require(torch.backends.mps.is_available(), 'Local MPS unavailable')
    batches, batch_proofs, bindings, source_proof = prepare_inputs()
    # Bind the actual local implementation dependencies, including the full loss chain.
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and any(path.is_relative_to(ROOT / folder) for folder in ('training', 'src')):
                merge_bindings(bindings, {str(path): sha(path)})
    merge_bindings(bindings, {str(Path(__file__).resolve()): sha(__file__)})
    require(all(sha(path) == digest for path, digest in bindings.items()), 'Diagnostic input changed')
    args.output.mkdir(parents=True)
    freeze = {'schema': 'flux-glyph-font-gradient-conflict-freeze-v1', 'policy': POLICY,
        'device': args.device, 'checkpoints': CHECKPOINTS, 'bindings': bindings,
        'batch_proofs': batch_proofs, 'source_proof': source_proof,
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_new(args.output / 'FREEZE.json', freeze)
    freeze_sha = sha(args.output / 'FREEZE.json')
    all_records = {}; summaries = {}; checkpoint_states = {}
    started = time.monotonic()
    for name, identity in CHECKPOINTS.items():
        model = load_checkpoint(identity, args.device); records = []
        for index, (step, cpu_batch) in enumerate(zip(SAMPLE_STEPS, batches), 1):
            batch = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value
                     for key, value in cpu_batch.items()}
            total, terms = decompose(model, batch)
            losses = {key: float(value.detach().cpu()) for key, value in terms.items()}
            original_loss = float(total.detach().cpu())
            groups, reconstruction = gradients(model, total, terms)
            records.append({'stream_step': step, 'losses': losses, 'original_total': original_loss,
                'groups': groups, 'gradient_reconstruction': reconstruction})
            del total, terms, batch
            if index % 4 == 0:
                print(json.dumps({'checkpoint': name, 'batches': index, 'total_batches': 32,
                    'seconds': round(time.monotonic() - started, 1)}), flush=True)
        model.cpu(); actual_state = state_sha(model.state_dict())
        require(actual_state == identity['state_sha256'] and all(p.grad is None for p in model.parameters()),
                'Read-only diagnostic changed checkpoint parameters or populated gradients')
        checkpoint_states[name] = {'before': identity['state_sha256'], 'after': actual_state,
                                   'weights_unchanged': True, 'optimizer_steps': 0}
        all_records[name] = records; summaries[name] = summary(records)
        del model
        if args.device == 'mps': torch.mps.empty_cache()
    require(all(batch_proof(batch, step) == proof for batch, step, proof in zip(batches, SAMPLE_STEPS, batch_proofs))
            and all(sha(path) == digest for path, digest in bindings.items())
            and sha(args.output / 'FREEZE.json') == freeze_sha, 'Diagnostic inputs mutated')
    report = {'schema': 'flux-glyph-font-gradient-conflict-report-v1', 'policy': POLICY,
        'freeze_sha256': freeze_sha, 'checkpoint_states': checkpoint_states,
        'summaries': summaries, 'measurements': all_records, 'source_bindings_unchanged': True,
        'input_batches_unchanged': True, 'optimizer_steps': 0, 'test_read': False,
        'calibration_inference': False, 'development_inference': False,
        'independent_validation_accuracy_measured': False,
        'seconds': round(time.monotonic() - started, 1)}
    write_new(args.output / 'REPORT.json', report)
    print(json.dumps({'status': 'COMPLETE', 'report_sha256': sha(args.output / 'REPORT.json'),
        'shared_parameter_summaries': {name: value['shared'] for name, value in summaries.items()}},
        ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    run(parser.parse_args())
