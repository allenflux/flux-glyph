#!/usr/bin/env python3
"""Export a CAL-qualified native-iOS continuation as one wide 25-output CNN."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'training')]
from train_regions import dump, require, sha, state_sha
from train_unified_retention import FIXED_RUNTIME, evaluate_outputs
from train_unified_regions import region_outputs
from export_unified_retention_core import (calibration_metadata, compare_runtime_outputs,
    require_fixed_runtime, cached_evaluation, strip_artifacts)
from short_region_objective import (EXTRA_ACCEPTANCE, SHORT_UNKNOWN_SOURCES, compare_r22, short_population)
from native_short_focus import FOCUS_KNOWN_FAMILIES, SAMPLING as FOCUS_SAMPLING
from native_unknown_oe import POLICY as UNKNOWN_OE_POLICY
from ios_short_training import (DEFAULT_ROOT as IOS_DATA_ROOT, IOS_FAMILIES, MANIFEST_SHA as IOS_MANIFEST_SHA,
    SAMPLING as IOS_SAMPLING, VIEWS as IOS_VIEWS)
from restore_unknown_train_teacher import POLICY as TEACHER_RESTORE_POLICY
from retention_face_balanced_sampler import SAMPLING as REPLAY_SAMPLING
from retention_source_balanced_sampler import expected_unknown_source_counts

ARCHITECTURE = 'region-cnn64x256-unified-wide-v1'
PARITY_SCHEMA = 'flux-glyph-unified-native-ios-onnx-parity-v1'
METADATA_SCHEMA = 'flux-glyph-unified-region-font-v1'
GROUPS = ('trunk', 'style', 'family_head', 'size_head')
CAL_REGIONS = 18672
CAL_TILES = 37834
STEPS = 2400
R22_RUN = ROOT / 'artifacts/unified-font-v3/run-wide-micro-recovery-v1'
R22_METADATA = ROOT / 'artifacts/unified-font-v3/region-wide-micro-recovery-v1/metadata.json'
IOS_RECONSTRUCTION = ROOT / 'artifacts/ios-native-short-train-v1/sampling-reconstruction-v1/REPORT.json'
NATIVE_RUN = ROOT / 'artifacts/unified-font-v4/run-native-ios-v1'
NATIVE_PLAN = ROOT / 'artifacts/unified-font-v4/native-ios-plan-v1/PLAN.json'
NATIVE_PREFLIGHT = ROOT / 'artifacts/unified-font-v4/native-ios-preflight-v1'
ADDITIONAL_GUARD = ROOT / 'artifacts/unified-font-v4/NATIVE_IOS_ADDITIONAL_RELEASE_GUARD.json'
ADDITIONAL_GUARD_SHA = '8261461ddb8a4f98fa6d18e240dcc59106a7b29f43a189e33c948be209b43a4f'
IOS_SYSTEM_FAMILIES = ('PingFang', 'SF Pro', 'Helvetica')


def read(path):
    return json.loads(Path(path).read_text())


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_v1_objective(objective):
    from train_unified_native_ios import OBJECTIVE as NATIVE_OBJECTIVE
    require(objective == NATIVE_OBJECTIVE
            and objective.get('focus_sampling') == NATIVE_OBJECTIVE.get('focus_sampling')
            and objective.get('focus_weight') == .5
            and objective.get('ios_batch') == 24
            and objective.get('ios_weight') == .25
            and objective.get('ios_sampling') == IOS_SAMPLING
            and objective.get('ios_loss') == '.25 * (mean CE25 + .2 * mean smooth-L1 size beta .05); verified native labels only'
            and objective.get('unknown_oe') == UNKNOWN_OE_POLICY
            and objective.get('unknown_oe_multiplier') == .25
            and objective.get('effective_unknown_oe_coefficient') == .125
            and objective.get('derived_short_crops_used') is False
            and objective.get('focus_normalization')
            and objective.get('original_full_correct_teacher_kl_preserved') is True
            and objective.get('runtime_changes') is False,
            'Native iOS/focus supervision or original-unknown teacher policy differs')


def validate_teacher_proof(proof, reconstructed, bindings):
    """Validate the actual reconstructed TRAIN teacher handoff, without inference."""
    require(proof == reconstructed
            and proof.get('schema') == 'flux-glyph-original-unknown-teacher-restoration-v1'
            and proof.get('policy') == TEACHER_RESTORE_POLICY
            and proof.get('model_inference') is False
            and proof.get('optimizer_steps') == 0
            and proof.get('calibration_read') is False
            and proof.get('development_read') is False
            and proof.get('test_read') is False,
            'NATIVE reconstructed training-teacher proof differs')
    parts = proof.get('partitions', {})
    require(set(parts) == {'original', 'supplement', 'known'}
            and parts['original'].get('tiles') == 130854
            and parts['original'].get('unknown_tiles') == 8072
            and parts['original'].get('restored_r21_tiles') == 8072
            and parts['supplement'].get('tiles') == 3156
            and parts['supplement'].get('restored_r21_tiles') == 0
            and parts['known'].get('tiles') == 4820
            and parts['known'].get('unknown_tiles') == 0
            and parts['known'].get('restored_r21_tiles') == 0
            and all(isinstance(parts[name].get(key), str) and len(parts[name][key]) == 64
                    for name in parts for key in
                    ('labels_sha256', 'r22_logits_sha256', 'mixed_logits_sha256',
                     'log_em_ratio_sha256')),
            'NATIVE teacher partition identity, scope or output digest differs')
    proof_bindings = proof.get('bindings')
    require(isinstance(proof_bindings, dict) and proof_bindings
            and all(bindings.get(path) == digest for path, digest in proof_bindings.items()),
            'NATIVE teacher reconstruction is absent from the frozen source closure')
    return proof


def validate_replay_counts(counts, families):
    require(counts.get('optimizer_steps') == STEPS and counts.get('replay_rows') == STEPS * 96,
            'Original replay step/row budget differs')
    replay = counts.get('replay_sampling', {})
    unknown_order = sorted([*SHORT_UNKNOWN_SOURCES, *REPLAY_SAMPLING['supplement_sources']])
    expected_unknown = expected_unknown_source_counts(unknown_order, STEPS)
    replay_unknown = counts.get('replay_unknown_by_source', {})
    require(replay.get('schema') == 'flux-glyph-retention-face-balanced-sampling-v1'
            and replay.get('unknown_source_order') == unknown_order
            and replay.get('unknown_source_rows') == replay_unknown == expected_unknown
            and replay.get('unknown_rows') == STEPS * 16,
            'R22 replay unknown-source cycle differs')
    expected_families = {name: STEPS * 2 for name in families[:-1]}
    expected_families['__unknown__'] = STEPS * 16
    expected_families['PingFang'] += STEPS * 16
    expected_families['SF Pro'] += STEPS * 4
    expected_families['Helvetica'] += STEPS * 4
    for name in REPLAY_SAMPLING['original_eight_families']:
        expected_families[name] += STEPS
    require(counts.get('replay_by_family') == replay.get('family_rows') == expected_families
            and replay.get('known_supplement_rows') == STEPS * 2
            and replay.get('known_supplement_source_rows') == {
                'LXGW WenKai': STEPS, 'WenQuanYi Micro Hei': STEPS},
            'R22 replay family or paired-known quota differs')
    face_rows = {}
    for row in replay.get('known_supplement_face_rows', []):
        key = (row.get('family'), row.get('font_face'))
        require(key not in face_rows and type(row.get('rows')) is int and row['rows'] > 0,
                'Invalid R22 paired-known face accounting')
        face_rows[key] = row['rows']
    require(face_rows == {
        ('LXGW WenKai', 'LXGWWenKai-Light'): STEPS // 3,
        ('LXGW WenKai', 'LXGWWenKai-Medium'): STEPS // 3,
        ('LXGW WenKai', 'LXGWWenKai-Regular'): STEPS // 3,
        ('WenQuanYi Micro Hei', 'WenQuanYiMicroHei'): STEPS},
        'R22 replay face cycle differs')
    return counts


def validate_counts(counts, families):
    validate_replay_counts(counts, families)
    require(counts.get('unknown_oe_rows') == STEPS * 16
            and counts.get('unknown_oe_policy') == UNKNOWN_OE_POLICY,
            'Unknown OE replay row budget or policy differs')
    require(counts.get('focus_rows') == counts.get('focus_supervised_rows') == STEPS * 32,
            'Native short focus row totals differ')
    by_family = counts.get('focus_by_family', {})
    require(set(by_family) == set(FOCUS_KNOWN_FAMILIES) | {'__unknown__'}
            and all(type(by_family[f]) is int and by_family[f] == STEPS * 2 for f in FOCUS_KNOWN_FAMILIES)
            and by_family['__unknown__'] == STEPS * 4
            and counts.get('focus_by_glyph_count') == {'3': STEPS * 16, '4': STEPS * 16},
            'Native focus family or glyph budget differs')
    require(counts.get('focus_teacher_eligible_rows') == 68712,
            'Native focus teacher eligibility differs from the frozen trajectory')
    report = counts.get('focus_sampling', {})
    from native_short_focus import FOCUS_UNKNOWN_SOURCES, VIEWS
    require(report.get('schema') == 'flux-glyph-native-short-focus-sampling-v1'
            and report.get('steps') == STEPS and report.get('rows') == STEPS * 32
            and report.get('sampling') == FOCUS_SAMPLING
            and report.get('by_family') == by_family
            and report.get('by_glyph_count') == counts['focus_by_glyph_count']
            and report.get('by_unknown_source') == {name: STEPS * 4 // len(FOCUS_UNKNOWN_SOURCES)
                                                   for name in FOCUS_UNKNOWN_SOURCES}
            and set(report.get('by_view', {})) == set(VIEWS)
            and all(type(n) is int and n > 0 for n in report['by_view'].values())
            and sum(report['by_view'].values()) == STEPS * 32,
            'Native focus sampling summary differs from the frozen contract')
    from collections import Counter
    face_totals = Counter(); seen = set()
    for row in report.get('by_face', []):
        key = (row.get('family'), row.get('font_face'))
        require(key not in seen and key[0] in by_family and isinstance(key[1], str)
                and key[1] and type(row.get('rows')) is int and row['rows'] > 0,
                'Native focus face accounting is invalid')
        seen.add(key); face_totals[key[0]] += row['rows']
    require(dict(face_totals) == by_family, 'Native focus face totals differ from its true families')
    require(counts.get('unknown_oe_multiplier') == .25
            and counts.get('effective_unknown_oe_coefficient') == .125,
            'Unknown OE multiplier or effective coefficient differs')
    ios = counts.get('ios_sampling', {})
    expected_ios = read(IOS_RECONSTRUCTION).get('full')
    require(counts.get('ios_rows') == STEPS * 24
            and counts.get('ios_by_family') == ios.get('by_family')
            == {name: STEPS * 8 for name in IOS_FAMILIES}
            and counts.get('ios_by_glyph_count') == ios.get('by_glyph_count')
            == {str(length): STEPS * 6 for length in (1, 2, 3, 4)}
            and ios == expected_ios
            and ios.get('schema') == 'flux-glyph-ios-native-short-sampling-v1'
            and ios.get('steps') == STEPS and ios.get('rows') == STEPS * 24
            and ios.get('sampling') == IOS_SAMPLING
            and set(ios.get('by_view', {})) == set(IOS_VIEWS)
            and sum(ios['by_view'].values()) == STEPS * 24
            and len(ios.get('by_face', {})) == 44
            and len(ios.get('by_identity', {})) == 595
            and sum(ios['by_face'].values()) == STEPS * 24
            and sum(ios['by_identity'].values()) == STEPS * 24,
            'Native iOS sampler trajectory, family/length quota, or proof differs')
    return counts


def validate_ios_proof(proof, reconstructed, bindings):
    require(proof == reconstructed
            and proof.get('schema') == 'flux-glyph-ios-native-short-proof-v1'
            and proof.get('sampling') == IOS_SAMPLING
            and proof.get('families') == list(IOS_FAMILIES)
            and proof.get('identity_count') == 595 and proof.get('views') == 2380
            and proof.get('by_family') == {'PingFang': 241, 'SF Pro': 252, 'Helvetica': 102}
            and proof.get('model_inference') is False
            and proof.get('calibration_read') is False
            and proof.get('development_read') is False and proof.get('test_read') is False,
            'Native iOS TRAIN proof differs from the frozen 595-region capture')
    proof_bindings = proof.get('bindings')
    require(isinstance(proof_bindings, dict) and proof_bindings
            and all(bindings.get(path) == digest for path, digest in proof_bindings.items()),
            'Native iOS source closure is unbound')
    manifest = str((ROOT / 'artifacts/ios-native-short-train-v1/prepared/MANIFEST.json').resolve())
    require(proof_bindings.get(manifest) == IOS_MANIFEST_SHA,
            'Native iOS prepared manifest differs')
    return proof


def validate_native_focus_proof(proof, reconstructed, bindings):
    require(proof == reconstructed and proof.get('schema') == 'flux-glyph-native-short-focus-proof-v1'
            and proof.get('sampling') == FOCUS_SAMPLING
            and proof.get('focus_known_families') == list(FOCUS_KNOWN_FAMILIES)
            and proof.get('model_inference') is False and proof.get('calibration_read') is False
            and proof.get('development_read') is False and proof.get('test_read') is False,
            'Native TRAIN focus proof is not the actual reconstructed complete-line pool')
    require(proof.get('bindings') and all(bindings.get(path) == digest
            for path, digest in proof['bindings'].items()), 'Native focus source closure is unbound')
    return proof


def reconstruct_sampling(datasets, pools, ios, teachers, steps):
    """Reproduce the fixed TRAIN RNG and teacher eligibility without reading pixels."""
    import train_unified_native_ios as trainer
    from ios_short_training import IOSShortSampler
    from native_short_focus import NativeShortFocusSampler
    from retention_face_balanced_sampler import FaceBalancedSampler
    from retention_paired_known_sampler import KNOWN_SUPPLEMENT_MARKER
    from retention_supplement_sampler import SUPPLEMENT_MARKER
    plan = read(trainer.RETENTION_PLAN)
    replay = FaceBalancedSampler(datasets['original']['rows'], trainer.FAMILIES, trainer.SEED,
        plan['train_pools'], datasets['supplement']['rows'], datasets['known']['rows'])
    focus = NativeShortFocusSampler(pools, trainer.SEED + 1); eligible = 0
    ios_sampler = IOSShortSampler(ios, trainer.SEED + 2)
    for _ in range(steps):
        rows = replay.batch()
        for row in rows:
            replay.rng.integers(row['tile_count'])
        rows = focus.batch()
        for row in rows * 3:
            focus.rng.integers(row['tile_count'])
        for row in rows:
            source = 'known' if row.get(KNOWN_SUPPLEMENT_MARKER, False) else 'supplement' if row.get(SUPPLEMENT_MARKER, False) else 'original'
            require(row['tile_count'] == 1, 'Focus teacher eligibility requires the exact single tile')
            eligible += int(np.argmax(teachers[source]['logits'][row['tile_start']]) == row['target'])
        ios_sampler.batch()
    return replay.report(), focus.report(), ios_sampler.report(), eligible


def calibration_contract(selection, measured, measured_short, r22_metrics, r22_short):
    require_fixed_runtime(measured)
    added = compare_r22(measured['metrics'], r22_metrics, measured_short, r22_short)
    checks = measured.get('retention_checks')
    cal_allowed = bool(measured.get('promotion_allowed') and added['passed'])
    require(isinstance(checks, list) and len(checks) == 46
            and all(row.get('passed') is True for row in checks)
            and len(added['checks']) == 7 and all(row['passed'] for row in added['checks'])
            and selection.get('selected') == measured
            and selection.get('short_metrics') == measured_short
            and selection.get('short_baseline') == r22_short
            and selection.get('baseline_metrics') == r22_metrics
            and selection.get('r22_comparison') == added
            and selection.get('calibration_promotion_allowed') is cal_allowed is True
            and selection.get('promotion_allowed') is False
            and selection.get('development_evaluated') is False,
            'Short continuation does not reproduce all 46+7 CAL gates or pre-DEV status')
    return added


def runtime_calibration_contract(measured, measured_short, r22_metrics, r22_short):
    """Check ONNX acceptance without requiring floating reports to equal saved MPS bytes."""
    require_fixed_runtime(measured)
    added = compare_r22(measured['metrics'], r22_metrics, measured_short, r22_short)
    checks = measured.get('retention_checks')
    require(measured.get('promotion_allowed') is True
            and isinstance(checks, list) and len(checks) == 46
            and all(row.get('passed') is True for row in checks)
            and added.get('passed') is True and len(added.get('checks', [])) == 7
            and all(row.get('passed') is True for row in added['checks']),
            'ONNX does not reproduce all 46+7 CAL acceptance gates')
    return added


def validate_preflight_loss_trace(parts):
    names = ('unknown_oe', 'unknown_oe_weighted', 'ios_loss', 'ios_ce', 'ios_size')
    require(parts.get('unknown_oe_rows') == 16
            and all(math.isfinite(parts.get(name, math.nan)) for name in names)
            and parts['unknown_oe'] >= 0 and parts['ios_loss'] >= 0
            and parts['ios_ce'] >= 0 and parts['ios_size'] >= 0
            and math.isclose(parts['unknown_oe_weighted'], .25 * parts['unknown_oe'],
                             rel_tol=1e-6, abs_tol=1e-8)
            and math.isclose(parts['ios_loss'], .25 * (parts['ios_ce'] + .2 * parts['ios_size']),
                             rel_tol=1e-6, abs_tol=1e-8),
            'Preflight loss trace does not execute the frozen OE/iOS coefficients')
    return parts


def validate_android_system_confusion(details, rows, baseline_details):
    """Apply the predeclared Android-to-iOS-system-font CAL regression guard."""
    def count(values):
        require(len(values) == len(rows)
                and all(detail.get('domain') == row.get('domain')
                        and detail.get('family') == row.get('family')
                        and detail.get('source_id') == row.get('source_id')
                        and detail.get('region_id') == row.get('region_id')
                        and type(detail.get('wrong_named')) is bool
                        for detail, row in zip(values, rows)),
                'Android system-font guard is not aligned to the frozen CAL rows')
        return sum(detail['wrong_named']
                   and row['domain'] == 'android'
                   and detail.get('predicted_family') in IOS_SYSTEM_FAMILIES
                   for detail, row in zip(values, rows))

    baseline_count = count(baseline_details)
    actual_count = count(details)
    result = {'baseline_count': baseline_count, 'maximum_count': 27,
              'actual_count': actual_count, 'passed': actual_count <= 27,
              'predicted_families': list(IOS_SYSTEM_FAMILIES),
              'scope': 'all Android CAL regions, including known truth and true unknowns; wrong named decisions only',
              'additional_to_original_53_cal_checks': True}
    require(sha(ADDITIONAL_GUARD) == ADDITIONAL_GUARD_SHA
            and baseline_count == 27 and result['passed'],
            'Native-iOS candidate increases Android wrong names into iOS system-font classes')
    return result


def validate(run, data, plan_path):
    """Validate frozen TRAIN/CAL evidence without importing Torch or reading CAL pixels."""
    import train_unified_native_ios as trainer
    run, data, plan_path = map(Path.resolve, (Path(run), Path(data), Path(plan_path)))
    require(run == NATIVE_RUN.resolve() and plan_path == NATIVE_PLAN.resolve(),
            'Only the frozen native-iOS run and plan may enter export')
    require(sha(ADDITIONAL_GUARD) == ADDITIONAL_GUARD_SHA,
            'Native-iOS additional Android confusion guard changed')
    selection = read(run / 'SELECTION.json')
    protocol = read(run / 'TRAINING_FREEZE.json')
    plan = read(plan_path)
    validate_v1_objective(protocol.get('objective', {}))
    require(selection.get('schema') == 'flux-glyph-unified-native-ios-selection-v1'
            and protocol.get('schema') == 'flux-glyph-unified-native-ios-training-freeze-v1'
            and plan.get('schema') == 'flux-glyph-unified-native-ios-training-plan-v1'
            and selection.get('architecture') == protocol.get('architecture') == plan.get('architecture') == ARCHITECTURE
            and selection.get('families') == protocol.get('families') == plan.get('families')
            and len(selection['families']) == 25 and selection['families'][-1] == '__unknown__',
            'Short continuation schema, architecture or class order differs')
    require(all(selection.get(key) == value for key, value in protocol.items()
                if key != 'schema'),
            'Selection no longer contains the exact frozen training protocol')
    bindings = selection.get('bindings')
    require(isinstance(bindings, dict) and bindings == protocol.get('bindings') and bindings
            and all(Path(path).is_file() and sha(path) == digest for path, digest in bindings.items()),
            'Short continuation source closure changed')
    require(sha(run / 'TRAINING_FREEZE.json') == selection.get('training_protocol_sha256')
            and sha(plan_path) == selection.get('plan_sha256')
            and bindings.get(str(plan_path)) == sha(plan_path),
            'Short plan or training freeze changed')
    for key, value in plan.items():
        if key not in ('schema', 'bindings'):
            require(protocol.get(key) == value, 'Preflight plan and training freeze differ: ' + key)
    require(all(bindings.get(path) == digest for path, digest in plan.get('bindings', {}).items()),
            'Plan source bindings are absent from training freeze')
    pre = plan.get('preflight', {})
    pre_path = Path(pre.get('path', '')).resolve()
    proof = read(pre_path)
    require(pre_path == NATIVE_PREFLIGHT.resolve()
            and bindings.get(str(pre_path)) == pre.get('sha256') == sha(pre_path)
            and proof.get('schema') == 'flux-glyph-native-ios-gradient-preflight-v1'
            and proof.get('passed') is True and proof.get('steps') == 4
            and proof.get('replay_tiles') == 384 and proof.get('focus_tiles') == 128
            and proof.get('ios_tiles') == 96
            and proof.get('focus_supervised_tiles') == 128
            and proof.get('unknown_oe_rows') == 64
            and proof.get('unknown_oe_policy') == UNKNOWN_OE_POLICY
            and proof.get('unknown_oe_multiplier') == .25
            and proof.get('effective_unknown_oe_coefficient') == .125
            and all(validate_preflight_loss_trace(row.get('parts', {}))
                    for row in proof.get('traces', []))
            and proof.get('all_groups_changed') is True and proof.get('baseline_unchanged') is True
            and proof.get('device') == protocol.get('device') == 'mps'
            and len(proof.get('traces', [])) == 4
            and all(isinstance(row.get('parts', {}).get('focus_teacher_eligible'), (int, float))
                    for row in proof.get('traces', []))
            and proof.get('bindings') == plan['bindings']
            and proof.get('calibration_inference') is False
            and proof.get('development_read') is False and proof.get('test_read') is False,
            'Short gradient preflight differs')
    datasets = trainer.load_training()
    teachers, reconstructed_teacher = trainer.load_replay_teacher(Path(protocol.get('teacher_cache', '')), datasets)
    validate_teacher_proof(protocol.get('training_teacher_proof', {}),
                           reconstructed_teacher, bindings)
    pools, reconstructed_focus = trainer.build_focus(datasets)
    validate_native_focus_proof(protocol.get('native_focus_proof', {}), reconstructed_focus, bindings)
    require(proof.get('native_focus_proof') == reconstructed_focus, 'Preflight native pool differs from full training')
    ios, reconstructed_ios = trainer.load_prepared(Path(protocol.get('ios_data', '')))
    validate_ios_proof(protocol.get('ios_short_proof', {}), reconstructed_ios, bindings)
    require(proof.get('ios_short_proof') == reconstructed_ios,
            'Preflight iOS pool differs from full training')
    ios_reconstruction = read(IOS_RECONSTRUCTION)
    ios_helper = ROOT / 'training/ios_short_training.py'
    require(ios_reconstruction.get('schema') == 'flux-glyph-ios-short-sampling-reconstruction-v1'
            and ios_reconstruction.get('seed') == trainer.SEED + 2
            and ios_reconstruction.get('helper_sha256') == sha(ios_helper)
            and ios_reconstruction.get('source_proof') == reconstructed_ios
            and ios_reconstruction.get('model_inference') is False
            and ios_reconstruction.get('calibration_read') is False
            and ios_reconstruction.get('development_read') is False
            and ios_reconstruction.get('test_read') is False
            and bindings.get(str(IOS_RECONSTRUCTION.resolve())) == sha(IOS_RECONSTRUCTION)
            and bindings.get(str(ios_helper.resolve())) == sha(ios_helper),
            'Independent iOS sampler reconstruction or helper binding differs')
    _, preflight_focus, preflight_ios, _ = reconstruct_sampling(datasets, pools, ios, teachers, 4)
    require(proof.get('focus_sampling') == preflight_focus
            and proof.get('ios_sampling') == preflight_ios
            and ios_reconstruction.get('preflight') == preflight_ios,
            'Actual preflight focus/iOS RNG or sample accounting differs')
    require(protocol.get('initializer_checkpoint_sha256') == trainer.BASE_SHA
            and protocol.get('initializer_selection_sha256') == trainer.BASE_SELECTION_SHA
            and protocol.get('initializer_state_sha256') == trainer.BASE_STATE_SHA
            and protocol.get('steps') == protocol.get('evaluation_step') == selection.get('optimizer_steps_executed') == STEPS
            and protocol.get('learning_rate') == trainer.LEARNING_RATE
            and protocol.get('minimum_learning_rate') == trainer.MINIMUM_LEARNING_RATE
            and protocol.get('seed') == trainer.SEED and protocol.get('objective') == trainer.OBJECTIVE
            and protocol.get('focus_sampling') == FOCUS_SAMPLING
            and protocol.get('ios_sampling') == IOS_SAMPLING
            and Path(protocol.get('ios_data', '')).resolve() == IOS_DATA_ROOT.resolve()
            and protocol.get('additional_acceptance') == EXTRA_ACCEPTANCE
            and protocol.get('runtime') == FIXED_RUNTIME
            and protocol.get('checkpoint_selection') == 'fixed final step 2400; no intermediate CAL search'
            and protocol.get('test_read') is False and protocol.get('development_holdout_read') is False,
            'Short optimizer, objective or no-holdout plan differs')
    require(selection.get('calibration_promotion_allowed') is True
            and selection.get('promotion_allowed') is False
            and selection.get('development_evaluated') is False
            and selection.get('exported') is False and selection.get('deployed') is False,
            'Only a CAL-qualified, pre-DEV checkpoint may enter export')
    initial = selection.get('initial_parameter_groups_sha256', {})
    final = selection.get('selected_parameter_groups_sha256', {})
    base_selection_path = Path(trainer.BASE_RUN / 'SELECTION.json').resolve()
    base_checkpoint_path = Path(trainer.BASE_RUN / 'model.pth').resolve()
    base_selection = read(base_selection_path)
    require(set(initial) == set(final) == set(GROUPS)
            and all(initial[name] != final[name] for name in GROUPS)
            and initial == base_selection.get('selected_parameter_groups_sha256')
            and selection.get('initializer_state_sha256') == base_selection.get('state_after_sha256') == trainer.BASE_STATE_SHA
            and bindings.get(str(base_selection_path)) == trainer.BASE_SELECTION_SHA == sha(base_selection_path)
            and bindings.get(str(base_checkpoint_path)) == trainer.BASE_SHA == sha(base_checkpoint_path)
            and selection.get('state_after_sha256') != trainer.BASE_STATE_SHA,
            'All four wide CNN groups must update from R22')
    counts_path = run / 'TRAINING_COUNTS.json'
    require(selection.get('training_counts_sha256') == sha(counts_path), 'Training counts changed')
    counts = validate_counts(read(counts_path), selection['families'])
    replay_report, focus_report, ios_report, focus_eligible = reconstruct_sampling(
        datasets, pools, ios, teachers, STEPS)
    require(counts['replay_sampling'] == replay_report and counts['focus_sampling'] == focus_report
            and counts['ios_sampling'] == ios_report
            and counts['focus_teacher_eligible_rows'] == focus_eligible,
            'Actual fixed replay/focus/iOS TRAIN trajectory is not reproducible')
    cal = calibration_metadata(data)
    require(cal['manifest_sha256'] == selection.get('data_manifest_sha256')
            and cal['partition'].get('tiles') == CAL_TILES and len(cal['rows']) == CAL_REGIONS
            and selection.get('calibration_partition_sha256') == sha(data / 'calibration/MANIFEST.json'),
            'Original CAL partition changed')
    retention_path = Path(trainer.RETENTION_PLAN).resolve()
    require(sha(retention_path) == trainer.RETENTION_SHA and bindings.get(str(retention_path)) == trainer.RETENTION_SHA,
            'Original 46-condition retention plan changed')
    retention = read(retention_path)
    current, _, details = cached_evaluation(run / 'CALIBRATION_OUTPUTS.npz',
        run / 'CALIBRATION_DECISIONS.json', cal, retention, STEPS)
    current_short = short_population(details, cal['rows'])
    r22, _, r22_details = cached_evaluation(R22_RUN / 'CALIBRATION_OUTPUTS.npz',
        R22_RUN / 'CALIBRATION_DECISIONS.json', cal, retention, 0)
    r22_short = short_population(r22_details, cal['rows'])
    calibration_contract(selection, current, current_short, r22['metrics'], r22_short)
    validate_android_system_confusion(details, cal['rows'], r22_details)
    require(selection.get('calibration_outputs_sha256') == sha(run / 'CALIBRATION_OUTPUTS.npz')
            and selection.get('calibration_decisions_sha256') == sha(run / 'CALIBRATION_DECISIONS.json')
            and bindings.get(str((R22_RUN / 'CALIBRATION_OUTPUTS.npz').resolve())) == sha(R22_RUN / 'CALIBRATION_OUTPUTS.npz')
            and bindings.get(str((R22_RUN / 'CALIBRATION_DECISIONS.json').resolve())) == sha(R22_RUN / 'CALIBRATION_DECISIONS.json')
            and bindings.get(str((R22_RUN / 'DEVELOPMENT_REGRESSION.json').resolve())) == sha(R22_RUN / 'DEVELOPMENT_REGRESSION.json'),
            'Current or R22 baseline evidence is unbound')
    return selection


def validate_checkpoint(checkpoint, final_checkpoint, selection, selection_sha256):
    require(checkpoint.get('architecture') == final_checkpoint.get('architecture') == ARCHITECTURE
            and checkpoint.get('families') == final_checkpoint.get('families') == selection['families']
            and checkpoint.get('selection_sha256') == selection_sha256
            and final_checkpoint.get('step') == STEPS
            and final_checkpoint.get('training_protocol_sha256') == selection['training_protocol_sha256']
            and state_sha(checkpoint['state_dict']) == state_sha(final_checkpoint['state_dict']) == selection['state_after_sha256'],
            'Short final checkpoint state differs')
    for name in GROUPS:
        group = {key: value for key, value in checkpoint['state_dict'].items() if key.startswith(name + '.')}
        require(group and state_sha(group) == selection['selected_parameter_groups_sha256'][name],
                'Short final parameter group differs: ' + name)


def validate_initializer_checkpoint(checkpoint, selection, base_state_sha256, base_selection_sha256):
    require(checkpoint.get('architecture') == ARCHITECTURE
            and checkpoint.get('families') == selection['families']
            and checkpoint.get('selection_sha256') == base_selection_sha256
            == selection['initializer_selection_sha256']
            and state_sha(checkpoint.get('state_dict', {})) == base_state_sha256
            == selection['initializer_state_sha256'],
            'Pinned R22 initializer state differs')
    for name in GROUPS:
        group = {key: value for key, value in checkpoint['state_dict'].items()
                 if key.startswith(name + '.')}
        require(group and state_sha(group) == selection['initial_parameter_groups_sha256'][name],
                'Initial parameter group does not come from the pinned R22 checkpoint: ' + name)


def training_summary(selection, selection_sha256, checkpoint_sha256):
    proof = selection['training_teacher_proof']
    ios_proof = selection['ios_short_proof']
    return {'kind': 'R22 continuation with retained Android focus, verified native iOS short lines, and reduced replay-only unknown OE',
        'steps': STEPS,
        'selection_sha256': selection_sha256, 'checkpoint_sha256': checkpoint_sha256,
        'training_protocol_sha256': selection['training_protocol_sha256'],
        'training_counts_sha256': selection['training_counts_sha256'],
        'initial_state_sha256': selection['initializer_state_sha256'],
        'final_state_sha256': selection['state_after_sha256'],
        'initial_parameter_groups_sha256': copy.deepcopy(selection['initial_parameter_groups_sha256']),
        'final_parameter_groups_sha256': copy.deepcopy(selection['selected_parameter_groups_sha256']),
        'objective': copy.deepcopy(selection['objective']), 'focus_sampling': copy.deepcopy(selection['focus_sampling']),
        'all_parameters_trained': True, 'model_count': 1, 'training_inputs': ['image_tiles'],
        'training_teacher_policy': copy.deepcopy(proof['policy']),
        'training_teacher_proof_sha256': json_sha(proof),
        'native_focus_proof_sha256': json_sha(selection['native_focus_proof']),
        'native_focus_identities': selection['native_focus_proof']['identity_count'],
        'ios_short_proof_sha256': json_sha(ios_proof),
        'ios_native_manifest_sha256': IOS_MANIFEST_SHA,
        'ios_native_identities': ios_proof['identity_count'], 'ios_native_views': ios_proof['views'],
        'ios_rows': STEPS * 24, 'ios_by_family': {name: STEPS * 8 for name in IOS_FAMILIES},
        'ios_by_glyph_count': {str(length): STEPS * 6 for length in (1, 2, 3, 4)},
        'ios_loss_weight': .25, 'ios_size_loss_weight': .2, 'ios_smooth_l1_beta': .05,
        'ios_label_source': 'verified real native iOS TRAIN glyphs and font sizes',
        'derived_crops_used': False, 'cnn_input_rows_per_step': 152,
        'cnn_input_groups_per_step': {'replay': 96, 'retained_android_focus': 32, 'native_ios': 24},
        'parameter_count': 1450938,
        'unknown_oe_policy': copy.deepcopy(UNKNOWN_OE_POLICY),
        'unknown_oe_rows': STEPS * 16, 'unknown_oe_multiplier': .25,
        'effective_unknown_oe_coefficient': .125,
        'r22_same_tile_teacher_cache_training_only': True,
        'r21_original_unknown_teacher_training_only': True, 'teacher_cache_deployed': False,
        'inference_rules_changed': False, 'platform_routing': False, 'score_merging': False,
        'test_read': False, 'development_holdout_read': False}


def metadata_for_export(selection, cal, parent_metadata, model_sha256, checkpoint_sha256,
                        selection_sha256, measured, added):
    require(selection['calibration_promotion_allowed'] is True and selection['promotion_allowed'] is False
            and len(measured.get('retention_checks', [])) == 46
            and all(row.get('passed') is True for row in measured['retention_checks'])
            and added.get('passed') is True and len(added.get('checks', [])) == 7
            and all(row.get('passed') is True for row in added['checks']),
            'Raw short export requires CAL qualification and must remain pre-DEV')
    return {'schema': METADATA_SCHEMA, 'algorithm': 'unified-region-cnn64x256-v1',
        'font_mode': 'unified', 'data_kind': 'native_mobile_screenshots',
        'network_architecture': ARCHITECTURE, 'model': {'path': 'model.onnx', 'sha256': model_sha256},
        'families': copy.deepcopy(selection['families']), **copy.deepcopy(FIXED_RUNTIME),
        'font_label_groups': copy.deepcopy(cal['manifest']['font_label_groups']),
        'font_sources': copy.deepcopy(parent_metadata.get('font_sources', {})),
        'release_tier': 'experimental', 'stable_validation_passed': False, 'test_passed': False,
        'pre_development': True,
        'validation': {'kind': 'native_ios_continuation_calibration_only_at_export',
            'calibration_promotion_allowed': True, 'promotion_allowed': False,
            'development_holdout_evaluated': False, 'original_retention_checks': 46,
            'short_regression_checks': 7, 'calibration_checks': 53,
            'retention_checks': copy.deepcopy(measured['retention_checks']),
            'short_comparison': copy.deepcopy(added), 'fixed_runtime': copy.deepcopy(FIXED_RUNTIME),
            'model_count': 1, 'encoder_count': 1, 'platform_routing': False,
            'scores_are_correctness_probabilities': False, 'blind_test_performed': False},
        'training': training_summary(selection, selection_sha256, checkpoint_sha256)}


def build_parity_report(selection, selection_sha256, checkpoint_sha256, model_sha256,
                        metadata_sha256, source_bindings, calibration_bindings, maximum,
                        max_size_px, batches, measured, added, android_system_confusion):
    require(selection.get('calibration_promotion_allowed') is True
            and selection.get('promotion_allowed') is False and len(measured['retention_checks']) == 46
            and all(row.get('passed') is True for row in measured['retention_checks'])
            and added.get('passed') is True and len(added.get('checks', [])) == 7
            and all(row.get('passed') is True for row in added['checks'])
            and android_system_confusion.get('baseline_count') == 27
            and android_system_confusion.get('maximum_count') == 27
            and android_system_confusion.get('passed') is True
            and android_system_confusion.get('actual_count', 28) <= 27
            and android_system_confusion.get('predicted_families') == list(IOS_SYSTEM_FAMILIES),
            'Parity cannot bypass the 46+7 CAL gate or pre-DEV state')
    return {'schema': PARITY_SCHEMA, 'passed': True, 'calibration_promotion_allowed': True,
        'promotion_allowed': False, 'development_evaluated': False, 'font_model_count': 1,
        'encoder_count': 1, 'output_family_count': 25, 'parameter_count': 1450938,
        'network_architecture': ARCHITECTURE,
        'selection_sha256': selection_sha256, 'checkpoint_sha256': checkpoint_sha256,
        'model_sha256': model_sha256, 'metadata_sha256': metadata_sha256,
        'source_bindings': copy.deepcopy(source_bindings),
        'calibration_bindings': copy.deepcopy(calibration_bindings),
        'fixed_runtime': copy.deepcopy(FIXED_RUNTIME), 'runtime_gates_changed': False,
        'calibration_regions': CAL_REGIONS, 'calibration_tiles': CAL_TILES,
        'original_retention_checks_passed': 46, 'short_checks_passed': 7,
        'calibration_checks_passed': 53, 'maximum_absolute_errors': copy.deepcopy(maximum),
        'max_size_pixel_error': max_size_px, 'batch_checks': copy.deepcopy(batches),
        'calibration_font_decisions_identical': True, 'calibration_runtime_signatures_identical': True,
        'calibration_size_values_close': True, 'cached_mps_outputs_close': True,
        'android_system_font_confusion': copy.deepcopy(android_system_confusion),
        'original_torch_groupnorm_reference': True, 'export_parameters_unchanged': True,
        'frozen_calibration_summary': strip_artifacts(selection['selected']),
        'runtime_calibration_summary': copy.deepcopy(measured), 'short_comparison': copy.deepcopy(added),
        'training': training_summary(selection, selection_sha256, checkpoint_sha256),
        'one_deployed_cnn': True, 'platform_routing': False, 'score_merging': False,
        'teacher_cache_deployed': False, 'stable_validation_passed': False, 'test_passed': False,
        'test_read': False, 'development_holdout_read': False, 'user_images_read': False}


def export(args):
    import onnx
    import onnxruntime as ort
    import torch
    from export_region_stable import replace_groupnorm
    from flux_glyph.unified_font import UnifiedFontClassifier
    from prepare_unified_regions import load_split
    from wide_region_network import WideRegionFontClassifier
    run, data, plan_path, output = map(Path.resolve, map(Path, (args.run, args.data, args.plan, args.output)))
    require(not output.exists() and not (run / 'PARITY.json').exists(),
            'Preserve prior native-iOS exports and parity evidence')
    selection = validate(run, data, plan_path)
    selection_sha = sha(run / 'SELECTION.json'); checkpoint_sha = sha(run / 'model.pth')
    parent_metadata_path = Path(args.parent_metadata).resolve()
    require(parent_metadata_path == R22_METADATA.resolve(), 'Only the pinned R22 metadata may supply source descriptions')
    sources = {str(path.resolve()): sha(path) for path in [Path(__file__),
        ROOT / 'scripts/prepare_short_region_release.py',
        ROOT / 'training/evaluate_unified_native_ios.py', ROOT / 'training/train_unified_native_ios.py',
        ROOT / 'training/ios_short_training.py', ROOT / 'training/prepare_ios_short_train.py',
        ROOT / 'artifacts/ios-native-short-train-v1/requests/REQUESTS.json',
        ROOT / 'artifacts/ios-native-short-train-v1/requests/Scenes.json', IOS_RECONSTRUCTION,
        ADDITIONAL_GUARD,
        ROOT / 'training/native_short_focus.py', ROOT / 'training/native_unknown_oe.py',
        ROOT / 'training/train_unified_native_oe.py',
        ROOT / 'training/restore_unknown_train_teacher.py',
        ROOT / 'artifacts/unified-font-v4/short-v3-comparison-v1/REPORT.json',
        ROOT / 'training/short_region_objective.py', ROOT / 'training/wide_region_network.py',
        ROOT / 'training/export_unified_retention_core.py', ROOT / 'training/export_region_stable.py',
        ROOT / 'src/flux_glyph/unified_font.py', ROOT / 'src/flux_glyph/region_font.py']}
    evidence = dict(selection['bindings'])
    for name in ('SELECTION.json', 'TRAINING_FREEZE.json', 'FINAL.pth', 'model.pth',
                 'CALIBRATION_OUTPUTS.npz', 'CALIBRATION_DECISIONS.json', 'TRAINING_COUNTS.json'):
        evidence[str((run / name).resolve())] = sha(run / name)
    evidence[str(parent_metadata_path)] = sha(parent_metadata_path)
    require(all(sha(path) == digest for path, digest in {**sources, **evidence}.items()),
            'Short export source or evidence changed before inference')
    checkpoint = torch.load(run / 'model.pth', map_location='cpu', weights_only=True)
    final_checkpoint = torch.load(run / 'FINAL.pth', map_location='cpu', weights_only=True)
    initializer = torch.load(__import__('train_unified_native_ios').BASE_RUN / 'model.pth',
                             map_location='cpu', weights_only=True)
    require(selection['final_checkpoint_sha256'] == sha(run / 'FINAL.pth'), 'Frozen FINAL checkpoint changed')
    validate_checkpoint(checkpoint, final_checkpoint, selection, selection_sha)
    validate_initializer_checkpoint(initializer, selection,
        __import__('train_unified_native_ios').BASE_STATE_SHA,
        __import__('train_unified_native_ios').BASE_SELECTION_SHA)
    reference = WideRegionFontClassifier(25).cpu().eval()
    reference.load_state_dict(checkpoint['state_dict'], strict=True)
    require(sum(parameter.numel() for parameter in reference.parameters()) == 1450938,
            'Wide 25-class CNN parameter count differs')
    converted = copy.deepcopy(reference)
    require(sum(isinstance(module, torch.nn.GroupNorm) for module in reference.modules()) == 4
            and replace_groupnorm(converted, high_precision=True) == 4
            and state_sha(converted.state_dict()) == selection['state_after_sha256'],
            'GroupNorm lowering changed short-region parameters')
    cal = load_split(data, 'calibration')
    retention = read(Path(__import__('train_unified_native_ios').RETENTION_PLAN))
    torch.set_num_threads(4); output.mkdir(parents=True)
    model_path = output / 'model.onnx'
    torch.onnx.export(converted, torch.zeros(2, 1, 64, 256), model_path,
        input_names=['tiles'], output_names=['logits', 'log_em_ratio'],
        dynamic_axes={'tiles': {0: 'batch'}, 'logits': {0: 'batch'}, 'log_em_ratio': {0: 'batch'}},
        opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(model_path))
    options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
    actual_logits = []; actual_sizes = []; reference_logits = []; reference_sizes = []
    maximum = {'logits': 0., 'size': 0.}
    with torch.inference_mode():
        for start in range(0, len(cal['tiles']), 128):
            block = np.array(cal['tiles'][start:start + 128], copy=True)
            expected_font, expected_size = reference(torch.from_numpy(block))
            expected_font, expected_size = expected_font.numpy(), expected_size.numpy()
            font, size = session.run(['logits', 'log_em_ratio'], {'tiles': block})
            for key, observed, expected in (('logits', font, expected_font), ('size', size, expected_size)):
                np.testing.assert_allclose(observed, expected, atol=2e-4, rtol=2e-4)
                maximum[key] = max(maximum[key], float(np.max(np.abs(observed - expected))))
            actual_logits.append(font); actual_sizes.append(size)
            reference_logits.append(expected_font); reference_sizes.append(expected_size)
            if start == 0 or (start + len(block)) % 4096 < 128 or start + len(block) == len(cal['tiles']):
                print(json.dumps({'phase': 'calibration_parity', 'tiles': start + len(block),
                                  'total': len(cal['tiles'])}), flush=True)
    actual_logits, actual_sizes = np.concatenate(actual_logits), np.concatenate(actual_sizes)
    reference_logits, reference_sizes = np.concatenate(reference_logits), np.concatenate(reference_sizes)
    with np.load(run / 'CALIBRATION_OUTPUTS.npz', allow_pickle=False) as saved:
        np.testing.assert_allclose(reference_logits, saved['logits'], atol=3e-4, rtol=3e-4)
        np.testing.assert_allclose(reference_sizes, saved['log_em_ratio'], atol=3e-4, rtol=3e-4)
        cached_outputs = region_outputs(saved['logits'], saved['log_em_ratio'], cal['rows'], 1.)
    actual_outputs = region_outputs(actual_logits, actual_sizes, cal['rows'], 1.)
    reference_outputs = region_outputs(reference_logits, reference_sizes, cal['rows'], 1.)
    max_size_px = max(compare_runtime_outputs(actual_outputs, other, cal['rows'], selection['families'])
                      for other in (reference_outputs, cached_outputs))
    measured, _, details = evaluate_outputs(actual_logits, actual_sizes, cal, retention, STEPS)
    measured_short = short_population(details, cal['rows'])
    added = runtime_calibration_contract(measured, measured_short,
                                         selection['baseline_metrics'], selection['short_baseline'])
    _, _, r22_details = cached_evaluation(R22_RUN / 'CALIBRATION_OUTPUTS.npz',
        R22_RUN / 'CALIBRATION_DECISIONS.json', cal, retention, 0)
    android_system_confusion = validate_android_system_confusion(details, cal['rows'], r22_details)
    indices = np.unique(np.linspace(0, len(cal['tiles']) - 1, min(256, len(cal['tiles']))).astype(int))
    samples = np.array(cal['tiles'][indices], copy=True); batches = []
    for count in (1, 7, 32, 128):
        values = [session.run(['logits', 'log_em_ratio'], {'tiles': samples[start:start + count]})
                  for start in range(0, len(samples), count)]
        font = np.concatenate([value[0] for value in values]); size = np.concatenate([value[1] for value in values])
        require(font.dtype == size.dtype == np.float32 and font.shape == (len(samples), 25)
                and size.shape == (len(samples),) and np.isfinite(font).all() and np.isfinite(size).all(),
                'Invalid dynamic short-region ONNX outputs')
        np.testing.assert_allclose(font, reference_logits[indices], atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(size, reference_sizes[indices], atol=2e-4, rtol=2e-4)
        batches.append({'batch_size': count, 'samples': len(samples), 'passed': True,
                        'font_and_size_checked': True})
    parent_metadata = read(parent_metadata_path)
    require(parent_metadata_path == R22_METADATA.resolve()
            and parent_metadata.get('schema') == METADATA_SCHEMA
            and parent_metadata.get('families') == selection['families'], 'R22 metadata family registry differs')
    metadata = metadata_for_export(selection, cal, parent_metadata, sha(model_path), checkpoint_sha,
                                   selection_sha, measured, added)
    dump(output / 'metadata.json', metadata)
    UnifiedFontClassifier(output)
    require(validate(run, data, plan_path) == selection
            and all(sha(path) == digest for path, digest in {**sources, **evidence}.items())
            and state_sha(reference.state_dict()) == state_sha(converted.state_dict()) == selection['state_after_sha256'],
            'Short export evidence or parameters changed')
    parity = build_parity_report(selection, selection_sha, checkpoint_sha, sha(model_path),
        sha(output / 'metadata.json'), sources, evidence, maximum, max_size_px, batches, measured, added,
        android_system_confusion)
    dump(run / 'PARITY.json', parity)
    print(json.dumps({'passed': True, 'calibration_promotion_allowed': True,
                      'promotion_allowed': False, 'model_sha256': parity['model_sha256'],
                      'calibration_tiles': CAL_TILES, 'maximum_errors': maximum}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent-metadata', type=Path, default=R22_METADATA)
    export(parser.parse_args())
