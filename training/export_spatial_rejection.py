#!/usr/bin/env python3
"""Export the first CAL-qualified spatial-rejection trial as one folded CNN."""
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
from export_unified_retention_core import (calibration_metadata, cached_evaluation,
    compare_runtime_outputs, strip_artifacts)
from short_region_objective import compare_r22, short_population

ARCHITECTURE = 'region-cnn64x256-unified-spatial4x16-v1'
PARITY_SCHEMA = 'flux-glyph-spatial-rejection-onnx-parity-v1'
METADATA_SCHEMA = 'flux-glyph-unified-region-font-v1'
PLAN_SCHEMA = 'flux-glyph-spatial-rejection-plan-v1'
POLICY_SCHEMA = 'flux-glyph-spatial-rejection-residual-v1'
CAL_REGIONS = 18672
CAL_TILES = 37834
STEPS = 2400
PARAMETERS = 3810234
TRAINABLE_PARAMETERS = 3145985
TRIALS = ('rejection-low', 'rejection-medium', 'rejection-high')
RATES = {name: 8e-5 for name in TRIALS}
REPAIR_WEIGHTS = dict(zip(TRIALS, (.25, .5, 1.)))
SELECTION_POLICY = ('fixed final step only; first fully CAL-qualified trial in declared '
                    'ascending supervised confusion-penalty order')
R22_RUN = ROOT / 'artifacts/unified-font-v3/run-wide-micro-recovery-v1'
R22_METADATA = ROOT / 'artifacts/unified-font-v3/region-wide-micro-recovery-v1/metadata.json'
R22_SELECTION_SHA = '49bd22a905bda9ffda11822323c81423482085e10adb477aed90089fdf43618e'
R22_CHECKPOINT_SHA = '53d73321c109825014fca96435f08a8c7a01ec6fe2abefa727855e1500d94c42'
R22_CAL_DECISIONS_SHA = 'b7c93734463255bd1f43481cbe3c332a3f83ff00d6f1d4c77c3e43a6ab26e55c'
FEATURE_MANIFEST_SHA = '9a881a03ab55e286e38fd7ede35ddd79e31390b452e3db1967acb1eb1cc59f52'
OBJECTIVE_SHA = '7aa6bf5a2afb1c40b18fa6fb060dd31dfe62fe1654862aede93fb17fb411a4ca'
RETENTION_PLAN_SHA = 'b82283b54b5e47f4639b0df62231544c0419f6299d10820d86dca96ef24094ff'
PRIOR_COUNTS = ROOT / 'artifacts/unified-font-v4/run-native-pairs-v1/TRAINING_COUNTS.json'
PRIOR_COUNTS_SHA = '0902aaee5884070c11997778be840e5e254a7e56ccaf26fe12cf7de151eaa0c1'
RETENTION_PLAN = ROOT / 'artifacts/unified-font-v2/RETENTION_PLAN.json'
GRID_ROOT = ROOT / 'artifacts/unified-font-v6/rejection-plans-v1'
RUN_ROOT = ROOT / 'artifacts/unified-font-v6/rejection-runs-v1'
ANDROID_GUARD = {'name': 'android_wrong_named_as_ios_system_family', 'domain': 'android',
    'predicted_families': ['PingFang', 'SF Pro', 'Helvetica'], 'metric': 'wrong_named',
    'operator': 'le', 'maximum': 27, 'r22_actual': 27,
    'population': 'all true CAL Android regions', 'original_53_checks_unchanged': True}
POLICY = {'schema': POLICY_SCHEMA,
    'trainable': ['style.0.residual_raw', 'family_head.unknown_weight_delta',
                  'family_head.unknown_bias_delta'],
    'constraint': 'zero-sum horizontal spatial residual; only existing unknown classifier row may adapt',
    'frozen': 'all inherited tensors; the 24 known classifier rows remain bitwise unchanged',
    'folded_architecture': ARCHITECTURE, 'model_count': 1, 'platform_routing': False,
    'extra_output_heads': False, 'runtime_score_changes': False}


def read(path):
    return json.loads(Path(path).read_text())


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _bindings(value, label):
    require(isinstance(value, dict) and value, label + ' bindings missing')
    require(all(isinstance(p, str) and isinstance(h, str) and len(h) == 64
                and Path(p).is_file() and sha(p) == h for p, h in value.items()),
            label + ' source closure changed')
    return value


def validate_android_system_guard(policy, measured, baseline):
    require(policy == ANDROID_GUARD
            and measured == {**ANDROID_GUARD, 'actual': measured.get('actual'),
                             'passed': measured.get('passed')}
            and type(measured.get('actual')) is int and 0 <= measured['actual'] <= 27
            and measured.get('passed') is True
            and baseline == {**ANDROID_GUARD, 'actual': 27, 'passed': True},
            'Android system-family guard policy, baseline, or result differs')
    return measured


def android_system_guard(details):
    require(isinstance(details, list) and all(d.get('domain') in ('ios', 'android')
            and isinstance(d.get('wrong_named'), bool) and isinstance(d.get('predicted_family'), str)
            for d in details), 'Android guard requires true CAL decision provenance')
    actual = sum(d.get('domain') == 'android' and d['wrong_named']
                 and d.get('predicted_family') in ANDROID_GUARD['predicted_families']
                 for d in details)
    return {**ANDROID_GUARD, 'actual': actual, 'passed': actual <= 27}


def validate_loss_trace(parts, weight):
    names = ('unknown_oe', 'unknown_oe_weighted', 'mobile_ce', 'mobile_size', 'mobile_loss',
             'pair_loss', 'pair_weighted', 'original_total', 'repair_unknown_ce',
             'repair_kaiti_ce', 'repair_system_confusion', 'repair_total')
    require(all(math.isfinite(parts.get(k, math.nan)) and parts[k] >= 0 for k in names)
            and parts.get('unknown_oe_rows') == 16 and parts.get('pair_count') == 16
            and parts.get('pair_rows') == 32
            and math.isclose(parts['unknown_oe_weighted'], .25 * parts['unknown_oe'], rel_tol=1e-5)
            and math.isclose(parts['mobile_loss'], .5 * (parts['mobile_ce'] + .2 * parts['mobile_size']), rel_tol=1e-5)
            and math.isclose(parts['pair_weighted'], .1 * parts['pair_loss'], rel_tol=1e-5)
            and math.isclose(parts['repair_total'], .25 * parts['repair_unknown_ce'] +
                             parts['repair_kaiti_ce'] + weight * parts['repair_system_confusion'],
                             rel_tol=1e-5, abs_tol=1e-7), 'Spatial rejection preflight loss differs')
    return parts


def validate_objective(objective):
    """Pin the complete nested 180-row objective without importing its Torch trainer."""
    require(json_sha(objective) == OBJECTIVE_SHA
            and objective.get('all_parameters_trained') is False
            and objective.get('trainable_policy') == POLICY
            and objective.get('repair_confusion_weights') == REPAIR_WEIGHTS
            and objective.get('repair_supervision', {}).get('original_objective_preserved') is True
            and objective.get('repair_supervision', {}).get('second_forward') is False
            and objective.get('repair_supervision', {}).get('runtime_changes') is False,
            'Complete spatial-rejection objective changed')
    return objective


def validate_preflight(preflight, trial):
    traces = preflight.get('traces', [])
    require(preflight.get('passed') is True and preflight.get('steps') == 4
            and len(traces) == 4 and [r.get('step') for r in traces] == [1, 2, 3, 4]
            and all(math.isfinite(r.get('loss', math.nan)) and r['loss'] >= 0
                    and math.isfinite(r.get('gradient_norm', math.nan)) and r['gradient_norm'] > 0
                    and validate_loss_trace(r.get('parts', {}), REPAIR_WEIGHTS[trial])
                    and math.isclose(r['loss'], r['parts']['original_total'] + r['parts']['repair_total'],
                                     rel_tol=1e-5, abs_tol=1e-7) for r in traces)
            and preflight.get('frozen_parameters_unchanged') is True
            and preflight.get('folding_parity_passed') is True
            and preflight.get('original_objective_equality_passed') is True,
            'Spatial rejection preflight failed or changed')
    counts = preflight.get('counts', {})
    require(counts.get('optimizer_steps') == 4 and counts.get('replay_rows') == 384
            and counts.get('unknown_oe_rows') == 64 and counts.get('focus_rows') == 128
            and counts.get('focus_teacher_eligible_rows') == 114
            and counts.get('mobile_rows') == 80 and counts.get('mobile_unknown_rows') == 12
            and counts.get('pair_count') == 64 and counts.get('pair_rows') == 128,
            'Spatial rejection preflight accounting differs')
    residual = preflight.get('residual', {})
    require(residual.get('trainable_parameter_count') == TRAINABLE_PARAMETERS
            and residual.get('max_absolute_residual', 0) > 0
            and residual.get('horizontal_sum_max_error', math.inf) <= residual.get('horizontal_sum_limit', -1),
            'Spatial rejection preflight residual proof differs')
    return preflight


def _selected_trial(grid_result):
    selected = grid_result.get('selected')
    require(selected in TRIALS and grid_result.get('selection_policy') == SELECTION_POLICY,
            'Grid has no CAL-qualified spatial-rejection trial')
    results = grid_result.get('results', {})
    index = TRIALS.index(selected)
    require(list(results) == list(TRIALS[:index + 1])
            and all(results[t].get('calibration_passed') is False for t in TRIALS[:index])
            and results[selected].get('calibration_passed') is True,
            'Grid did not select the first passing trial')
    return selected


def validate(run, data, plan_path):
    """Validate the frozen TRAIN/CAL boundary without importing Torch or reading pixels."""
    run, data, plan_path = map(Path.resolve, map(Path, (run, data, plan_path)))
    require(run.parent == RUN_ROOT.resolve() and run.name in TRIALS
            and plan_path.parent == GRID_ROOT.resolve() and plan_path.name == run.name + '.json',
            'Run and plan must be one declared spatial-rejection trial')
    selection, protocol, plan = map(read, (run/'SELECTION.json', run/'TRAINING_FREEZE.json', plan_path))
    grid_path = plan_path.parent/'GRID.json'; grid = read(grid_path)
    result_path = run.parent/'GRID_RESULT.json'; result = read(result_path)
    require(result.get('schema') == 'flux-glyph-spatial-rejection-grid-result-v1'
            and _selected_trial(result) == run.name and result.get('grid_sha256') == sha(grid_path)
            and result['results'][run.name].get('path') == str(run)
            and result['results'][run.name].get('report_sha256') == sha(run/'report.json'),
            'Run differs from the frozen first-passing grid result')
    require(grid.get('schema') == 'flux-glyph-spatial-rejection-grid-v1'
            and grid.get('order') == list(TRIALS) and grid.get('selection_policy') == SELECTION_POLICY
            and grid.get('calibration_read') is False and grid.get('development_read') is False
            and grid.get('test_read') is False
            and grid.get('plans', {}).get(run.name) == {'path': str(plan_path), 'sha256': sha(plan_path)},
            'Spatial rejection grid or selected plan changed')
    require(selection.get('schema') == 'flux-glyph-spatial-rejection-selection-v1'
            and protocol.get('schema') == 'flux-glyph-spatial-rejection-training-freeze-v1'
            and plan.get('schema') == PLAN_SCHEMA
            and all(selection.get(k) == v for k, v in protocol.items() if k != 'schema')
            and all(protocol.get(k) == v for k, v in plan.items() if k not in ('schema', 'bindings')),
            'Selection, TRAIN freeze, and preflight plan differ')
    bindings = _bindings(selection.get('bindings'), 'TRAIN')
    require(bindings == protocol.get('bindings')
            and all(bindings.get(k) == v for k, v in _bindings(plan.get('bindings'), 'plan').items())
            and bindings.get(str(plan_path)) == sha(plan_path)
            and bindings.get(str(grid_path.resolve())) == sha(grid_path), 'Plan is outside TRAIN closure')
    trial = run.name
    require(selection.get('trial') == trial and plan.get('trial') == trial
            and plan.get('architecture') == protocol.get('architecture') == selection.get('architecture') == ARCHITECTURE
            and plan.get('policy') == POLICY and plan.get('grid') == RATES
            and plan.get('repair_weights') == REPAIR_WEIGHTS
            and plan.get('learning_rate') == RATES[trial]
            and math.isclose(plan.get('minimum_learning_rate', -1), RATES[trial]/10)
            and plan.get('repair_confusion_weight') == REPAIR_WEIGHTS[trial]
            and plan.get('steps') == selection.get('optimizer_steps_executed') == STEPS
            and plan.get('selection_policy') == SELECTION_POLICY and plan.get('runtime') == FIXED_RUNTIME
            and plan.get('policy', {}).get('model_count') == 1
            and plan.get('test_read') is False and plan.get('development_read') is False,
            'Optimizer, residual policy, objective, or runtime changed')
    validate_objective(plan.get('objective', {}))
    validate_preflight(plan.get('preflight', {}), trial)
    require(sha(run/'TRAINING_FREEZE.json') == selection.get('training_protocol_sha256')
            and sha(run/'TRAINING_COUNTS.json') == selection.get('training_counts_sha256')
            and sha(run/'FINAL.pth') == selection.get('final_checkpoint_sha256')
            and sha(run/'RESIDUAL_TRAINING.pth') == selection.get('residual_checkpoint_sha256')
            and sha(run/'CALIBRATION_OUTPUTS.npz') == selection.get('calibration_outputs_sha256')
            and sha(run/'CALIBRATION_DECISIONS.json') == selection.get('calibration_decisions_sha256'),
            'Spatial checkpoint, counts, or CAL evidence changed')
    require(sha(PRIOR_COUNTS) == PRIOR_COUNTS_SHA
            and plan.get('prior_training_counts_sha256') == PRIOR_COUNTS_SHA
            and read(run/'TRAINING_COUNTS.json') == read(PRIOR_COUNTS),
            'Residual training changed the original 180-row sampling trajectory')
    cache = Path(plan.get('feature_cache', '')).resolve(); manifest_path = cache/'MANIFEST.json'
    manifest = read(manifest_path)
    require(plan.get('feature_manifest_sha256') == FEATURE_MANIFEST_SHA == sha(manifest_path)
            and manifest.get('schema') == 'flux-glyph-frozen-spatial-train-features-v1'
            and manifest.get('complete') is True and manifest.get('parameters_unchanged') is True
            and manifest.get('reuse_parity_passed') is True
            and manifest.get('calibration_read') is False and manifest.get('development_read') is False
            and manifest.get('test_read') is False and manifest.get('feature_dtype') == 'float32'
            and manifest.get('feature_width') == 8192
            and manifest.get('sources') == {'original':130854, 'supplement':3156,
                'known':4820, 'ios':2380, 'android':4140}
            and manifest.get('spatial_initial_state_sha256') == plan.get('inherited_spatial_state_sha256'),
            'Frozen TRAIN feature cache changed')
    expected_counts = manifest['sources']; files = manifest.get('files', {})
    require(set(files) == set(expected_counts)
            and all((cache / entry.get('path', '')).resolve().parent == cache
                    and sha(cache / entry['path']) == entry.get('sha256')
                    and entry.get('shape') == [expected_counts[name], 8192]
                    and entry.get('dtype') == 'float32'
                    and bindings.get(str(Path(entry.get('source_pixels_path', '')).resolve()))
                        == entry.get('source_pixels_sha256') == sha(entry['source_pixels_path'])
                    for name, entry in files.items())
            and bindings.get(str(manifest_path)) == FEATURE_MANIFEST_SHA
            and bindings.get(str((cache/'CACHE_FREEZE.json').resolve())) == sha(cache/'CACHE_FREEZE.json')
            == manifest.get('cache_freeze_sha256'), 'Frozen feature files or source pixels changed')
    require(plan.get('initializer_checkpoint_sha256') == R22_CHECKPOINT_SHA
            and plan.get('initializer_selection_sha256') == R22_SELECTION_SHA
            and sha(R22_RUN/'model.pth') == R22_CHECKPOINT_SHA
            and sha(R22_RUN/'SELECTION.json') == R22_SELECTION_SHA
            and sha(R22_RUN/'CALIBRATION_DECISIONS.json') == R22_CAL_DECISIONS_SHA,
            'Pinned R22 initializer or CAL baseline changed')
    require(plan.get('retention_plan_sha256') == RETENTION_PLAN_SHA == sha(RETENTION_PLAN)
            and bindings.get(str(RETENTION_PLAN.resolve())) == RETENTION_PLAN_SHA,
            'Original 46-check retention policy changed or is unbound')
    proof_schemas = {'training_teacher_proof':'flux-glyph-original-unknown-teacher-restoration-v1',
        'native_focus_proof':'flux-glyph-native-short-focus-proof-v1',
        'mobile_short_proof':'flux-glyph-native-mobile-short-pools-v1',
        'native_font_pair_proof':'flux-glyph-native-font-pair-pools-v1'}
    require(all(selection.get(key, {}).get('schema') == schema
                and selection[key].get('calibration_read') is False
                and selection[key].get('development_read') is False
                and selection[key].get('test_read') is False
                for key, schema in proof_schemas.items()),
            'Verified TRAIN teacher/focus/mobile/pair provenance differs')
    require(selection.get('calibration_promotion_allowed') is True
            and selection.get('original_53_calibration_checks_passed') is True
            and selection.get('promotion_allowed') is False
            and selection.get('development_evaluated') is False
            and selection.get('exported') is False and selection.get('deployed') is False
            and selection.get('test_read') is False,
            'Only a CAL-qualified pre-DEV trial may be exported')
    validate_android_system_guard(selection.get('android_system_guard_policy'),
                                  selection.get('android_system_guard'),
                                  selection.get('android_system_guard_baseline'))
    cal = calibration_metadata(data)
    require(cal['manifest_sha256'] == selection.get('data_manifest_sha256')
            and len(cal['rows']) == CAL_REGIONS and cal['partition'].get('tiles') == CAL_TILES
            and selection.get('calibration_partition_sha256') == sha(data/'calibration/MANIFEST.json'),
            'Original CAL partition changed')
    retention = read(RETENTION_PLAN)
    current, _, details = cached_evaluation(run/'CALIBRATION_OUTPUTS.npz',
        run/'CALIBRATION_DECISIONS.json', cal, retention, STEPS)
    baseline, _, base_details = cached_evaluation(R22_RUN/'CALIBRATION_OUTPUTS.npz',
        R22_RUN/'CALIBRATION_DECISIONS.json', cal, retention, 0)
    added = compare_r22(current['metrics'], baseline['metrics'], short_population(details, cal['rows']),
                        short_population(base_details, cal['rows']))
    require(selection.get('selected') == current and selection.get('r22_comparison') == added
            and len(current.get('retention_checks', [])) == 46
            and all(c.get('passed') is True for c in current['retention_checks'])
            and added.get('passed') is True and len(added.get('checks', [])) == 7
            and all(c.get('passed') is True for c in added['checks'])
            and selection.get('android_system_guard') == android_system_guard(details)
            and selection.get('android_system_guard_baseline') == android_system_guard(base_details),
            'Saved CAL outputs do not reproduce 46+7 and Android guard')
    return selection


def validate_folded_states(raw_checkpoint, final_checkpoint, checkpoint, selection, initial, raw, folded):
    """Validate the trainable raw residual and its ordinary-CNN folded state."""
    import torch
    from spatial_rejection_network import frozen_state, residual_check
    require(raw_checkpoint.get('policy') == POLICY
            and raw_checkpoint.get('training_protocol_sha256') == selection['training_protocol_sha256']
            and final_checkpoint.get('architecture') == checkpoint.get('architecture') == ARCHITECTURE
            and final_checkpoint.get('families') == checkpoint.get('families') == selection['families']
            and final_checkpoint.get('step') == STEPS
            and checkpoint.get('selection_sha256') == selection['_selection_sha256'],
            'Raw/final/model checkpoint envelope differs')
    require(state_sha(frozen_state(raw)) == selection['frozen_parameter_state_sha256'],
            'A frozen raw parameter changed')
    proof = residual_check(raw); saved = selection.get('residual_training', {})
    require(proof['trainable_parameter_count'] == TRAINABLE_PARAMETERS
            and proof['max_absolute_residual'] > 0
            and all(math.isclose(proof[k], saved[k], rel_tol=1e-6, abs_tol=1e-8)
                    for k in proof), 'Raw residual proof differs from selection')
    state = folded.state_dict()
    require(state_sha(state) == selection.get('state_after_sha256')
            and all(set(other) == set(state) and torch.equal(other[k], v)
                    for other in (final_checkpoint['state_dict'], checkpoint['state_dict']) for k,v in state.items()),
            'Folded FINAL/model state differs from raw residual')
    before = initial.state_dict()
    require(torch.equal(state['family_head.weight'][:24], before['family_head.weight'][:24])
            and torch.equal(state['family_head.bias'][:24], before['family_head.bias'][:24])
            and all(torch.equal(state[k], v) for k,v in before.items()
                    if k not in ('style.0.weight','family_head.weight','family_head.bias')),
            'Known rows or inherited tensors changed')
    error = float((state['style.0.weight'].reshape(384,128,4,4,4).sum(-1) -
                   before['style.0.weight'].reshape(384,128,4,4,4).sum(-1)).abs().max())
    require(error <= 1e-6 and saved.get('folded_coarse_weight_error', math.inf) <= 1e-6
            and saved.get('frozen_parameters_unchanged') is True
            and saved.get('folded_output_parity_passed') is True,
            'Folded coarse spatial coefficients changed')
    return error


def training_summary(selection, selection_sha256, checkpoint_sha256):
    return {'kind': 'R22 spatial residual with verified TRAIN rejection supervision',
        'trial': selection['trial'], 'steps': STEPS, 'selection_sha256': selection_sha256,
        'checkpoint_sha256': checkpoint_sha256,
        'training_protocol_sha256': selection['training_protocol_sha256'],
        'training_counts_sha256': selection['training_counts_sha256'],
        'initial_state_sha256': selection['initializer_state_sha256'],
        'inherited_spatial_state_sha256': selection['inherited_spatial_state_sha256'],
        'final_state_sha256': selection['state_after_sha256'],
        'objective': copy.deepcopy(selection['objective']), 'all_parameters_trained': False,
        'trainable_policy': copy.deepcopy(POLICY),
        'repair_confusion_weight': selection['repair_confusion_weight'],
        'parameter_count': PARAMETERS, 'trainable_parameter_count': TRAINABLE_PARAMETERS,
        'cnn_input_rows_per_step': 180,
        'cnn_input_groups_per_step': {'replay':96,'retained_android_focus':32,
            'native_mobile':20,'same_content_pairs':32},
        'feature_cache_training_only': True, 'teacher_cache_deployed': False,
        'data_manifest_sha256': selection['data_manifest_sha256'], 'ocr_text_used': False,
        'model_count': 1, 'encoder_count': 1, 'inference_rules_changed': False,
        'platform_routing': False, 'score_merging': False, 'platform_features': False,
        'test_read': False, 'development_holdout_read': False}


def metadata_for_export(selection, cal, parent_metadata, model_sha256, checkpoint_sha256,
                        selection_sha256, measured, added):
    require(len(measured.get('retention_checks', [])) == 46
            and all(c.get('passed') is True for c in measured['retention_checks'])
            and added.get('passed') is True and len(added.get('checks', [])) == 7,
            'Metadata requires passing 46+7 CAL')
    return {'schema': METADATA_SCHEMA, 'algorithm': 'unified-region-cnn64x256-v1',
        'font_mode': 'unified', 'data_kind': 'native_mobile_screenshots',
        'network_architecture': ARCHITECTURE, 'model': {'path':'model.onnx','sha256':model_sha256},
        'families': copy.deepcopy(selection['families']), **copy.deepcopy(FIXED_RUNTIME),
        'font_label_groups': copy.deepcopy(cal['manifest']['font_label_groups']),
        'font_sources': copy.deepcopy(parent_metadata.get('font_sources', {})),
        'release_tier': 'experimental', 'stable_validation_passed': False, 'test_passed': False,
        'pre_development': True,
        'validation': {'kind':'spatial_rejection_calibration_only_at_export',
            'calibration_promotion_allowed':True,'promotion_allowed':False,
            'development_holdout_evaluated':False,'original_retention_checks':46,
            'short_regression_checks':7,'calibration_checks':53,
            'retention_checks':copy.deepcopy(measured['retention_checks']),
            'short_comparison':copy.deepcopy(added),'fixed_runtime':copy.deepcopy(FIXED_RUNTIME),
            'model_count':1,'encoder_count':1,'platform_routing':False,
            'android_system_guard_policy':copy.deepcopy(ANDROID_GUARD),
            'android_system_guard':copy.deepcopy(selection['android_system_guard']),
            'scores_are_correctness_probabilities':False,'blind_test_performed':False},
        'training': training_summary(selection, selection_sha256, checkpoint_sha256)}


def build_parity_report(selection, selection_sha256, checkpoint_sha256, model_sha256,
                        metadata_sha256, source_bindings, calibration_bindings, maximum,
                        max_size_px, batches, measured, added, guard, raw_fold_error=0.):
    validate_android_system_guard(selection['android_system_guard_policy'], guard,
                                  selection['android_system_guard_baseline'])
    require(len(measured.get('retention_checks', [])) == 46
            and all(c.get('passed') is True for c in measured['retention_checks'])
            and added.get('passed') is True and len(added.get('checks', [])) == 7,
            'Parity requires passing 46+7 CAL')
    return {'schema':PARITY_SCHEMA,'trial':selection['trial'],'passed':True,
        'calibration_promotion_allowed':True,'promotion_allowed':False,'development_evaluated':False,
        'font_model_count':1,'encoder_count':1,'output_family_count':25,'parameter_count':PARAMETERS,
        'network_architecture':ARCHITECTURE,'selection_sha256':selection_sha256,
        'checkpoint_sha256':checkpoint_sha256,'model_sha256':model_sha256,
        'metadata_sha256':metadata_sha256,'source_bindings':copy.deepcopy(source_bindings),
        'calibration_bindings':copy.deepcopy(calibration_bindings),'fixed_runtime':copy.deepcopy(FIXED_RUNTIME),
        'runtime_gates_changed':False,'calibration_regions':CAL_REGIONS,'calibration_tiles':CAL_TILES,
        'original_retention_checks_passed':46,'short_checks_passed':7,'calibration_checks_passed':53,
        'maximum_absolute_errors':copy.deepcopy(maximum),'max_size_pixel_error':max_size_px,
        'raw_fold_maximum_error':raw_fold_error,'batch_checks':copy.deepcopy(batches),
        'calibration_font_decisions_identical':True,'calibration_runtime_signatures_identical':True,
        'calibration_size_values_close':True,'cached_mps_outputs_close':True,
        'android_system_guard_policy':copy.deepcopy(ANDROID_GUARD),'android_system_guard':copy.deepcopy(guard),
        'android_system_guard_baseline':copy.deepcopy(selection['android_system_guard_baseline']),
        'original_torch_groupnorm_reference':True,'export_parameters_unchanged':True,
        'frozen_calibration_summary':strip_artifacts(selection['selected']),
        'runtime_calibration_summary':copy.deepcopy(measured),'short_comparison':copy.deepcopy(added),
        'training':training_summary(selection,selection_sha256,checkpoint_sha256),
        'one_deployed_cnn':True,'platform_routing':False,'score_merging':False,
        'teacher_cache_deployed':False,'stable_validation_passed':False,'test_passed':False,
        'test_read':False,'development_holdout_read':False,'user_images_read':False}


def _validate_onnx_contract(model, session):
    require(len(model.opset_import) == 1 and model.opset_import[0].version == 17,
            'ONNX opset must be exactly 17')
    inputs, outputs = session.get_inputs(), session.get_outputs()
    require(len(inputs) == 1 and inputs[0].name == 'tiles'
            and inputs[0].shape[0] == 'batch' and inputs[0].shape[1:] == [1,64,256]
            and inputs[0].type == 'tensor(float)'
            and len(outputs) == 2 and [o.name for o in outputs] == ['logits','log_em_ratio']
            and outputs[0].shape == ['batch',25] and outputs[1].shape == ['batch'],
            'ONNX input/output or dynamic-batch contract differs')


def export(args):
    import onnx
    import onnxruntime as ort
    import torch
    from export_region_stable import replace_groupnorm
    from flux_glyph.unified_font import UnifiedFontClassifier
    from prepare_unified_regions import load_split
    from wide_region_network import WideRegionFontClassifier
    from spatial_region_network import expand_spatial_pool
    from spatial_rejection_network import add_residual, fold_residual

    run, data, plan_path, output = map(Path.resolve, map(Path,(args.run,args.data,args.plan,args.output)))
    require(not output.exists() and not (run/'PARITY.json').exists(), 'Preserve prior exports')
    selection = validate(run,data,plan_path); selection_sha = sha(run/'SELECTION.json')
    checkpoint_sha = sha(run/'model.pth'); selection['_selection_sha256'] = selection_sha
    parent_metadata_path = Path(args.parent_metadata).resolve()
    require(parent_metadata_path == R22_METADATA.resolve(), 'Only pinned R22 metadata may supply sources')
    source_files = [Path(__file__), ROOT/'training/evaluate_spatial_rejection.py',
        ROOT/'scripts/prepare_spatial_rejection_release.py', ROOT/'training/train_spatial_rejection.py',
        ROOT/'training/spatial_rejection_network.py', ROOT/'training/spatial_residual_network.py',
        ROOT/'training/spatial_rejection_objective.py', ROOT/'training/spatial_region_network.py',
        ROOT/'training/cache_spatial_train_features.py', ROOT/'training/export_unified_retention_core.py',
        ROOT/'training/export_region_stable.py', ROOT/'src/flux_glyph/unified_font.py',
        ROOT/'src/flux_glyph/region_font.py']
    sources = {str(p.resolve()):sha(p) for p in source_files}
    evidence = dict(selection['bindings'])
    for name in ('SELECTION.json','TRAINING_FREEZE.json','TRAINING_COUNTS.json','RESIDUAL_TRAINING.pth',
                 'FINAL.pth','model.pth','CALIBRATION_OUTPUTS.npz','CALIBRATION_DECISIONS.json','report.json'):
        evidence[str((run/name).resolve())] = sha(run/name)
    for p in (plan_path, plan_path.parent/'GRID.json', run.parent/'GRID_RESULT.json', parent_metadata_path):
        evidence[str(p.resolve())] = sha(p)
    require(all(sha(p)==h for p,h in {**sources,**evidence}.items()), 'Export evidence changed')
    raw_ckpt = torch.load(run/'RESIDUAL_TRAINING.pth',map_location='cpu',weights_only=True)
    final = torch.load(run/'FINAL.pth',map_location='cpu',weights_only=True)
    checkpoint = torch.load(run/'model.pth',map_location='cpu',weights_only=True)
    base = torch.load(R22_RUN/'model.pth',map_location='cpu',weights_only=True)
    wide = WideRegionFontClassifier(25).cpu().eval(); wide.load_state_dict(base['state_dict'])
    initial, _ = expand_spatial_pool(wide); initial.eval()
    require(state_sha(initial.state_dict()) == selection['inherited_spatial_state_sha256'],
            'Expanded R22 spatial state changed')
    raw = add_residual(initial); raw.load_state_dict(raw_ckpt['state_dict']); raw.eval()
    reference = fold_residual(raw).cpu().eval()
    coarse_error = validate_folded_states(raw_ckpt,final,checkpoint,selection,initial,raw,reference)
    require(sum(p.numel() for p in reference.parameters()) == PARAMETERS, 'Parameter count differs')
    converted=copy.deepcopy(reference)
    require(sum(isinstance(m,torch.nn.GroupNorm) for m in reference.modules())==4
            and replace_groupnorm(converted,high_precision=True)==4
            and state_sha(converted.state_dict())==selection['state_after_sha256'],
            'High-precision GroupNorm lowering changed parameters')
    cal=load_split(data,'calibration'); retention=read(RETENTION_PLAN)
    torch.set_num_threads(4); output.mkdir(parents=True); model_path=output/'model.onnx'
    torch.onnx.export(converted,torch.zeros(2,1,64,256),model_path,input_names=['tiles'],
        output_names=['logits','log_em_ratio'],dynamic_axes={'tiles':{0:'batch'},'logits':{0:'batch'},
        'log_em_ratio':{0:'batch'}},opset_version=17,dynamo=False)
    graph=onnx.load(model_path); onnx.checker.check_model(graph)
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model_path),sess_options=options,providers=['CPUExecutionProvider'])
    _validate_onnx_contract(graph,session)
    actual_f=[];actual_s=[];ref_f=[];ref_s=[];maximum={'logits':0.,'size':0.}; raw_fold=0.
    with torch.inference_mode():
        for start in range(0,len(cal['tiles']),128):
            block=np.array(cal['tiles'][start:start+128],copy=True); tensor=torch.from_numpy(block)
            expected=reference(tensor); unfurled=raw(tensor)
            for a,b in zip(expected,unfurled):
                np.testing.assert_allclose(a.numpy(),b.numpy(),atol=1e-6,rtol=1e-6)
                raw_fold=max(raw_fold,float((a-b).abs().max()))
            ef,es=(x.numpy() for x in expected); font,size=session.run(['logits','log_em_ratio'],{'tiles':block})
            for key,a,b in (('logits',font,ef),('size',size,es)):
                np.testing.assert_allclose(a,b,atol=2e-4,rtol=2e-4);maximum[key]=max(maximum[key],float(np.max(np.abs(a-b))))
            actual_f.append(font);actual_s.append(size);ref_f.append(ef);ref_s.append(es)
            completed = start + len(block)
            if start == 0 or completed % 4096 < 128 or completed == len(cal['tiles']):
                print(json.dumps({'phase':'calibration_parity','tiles':completed,
                                  'total':len(cal['tiles'])}),flush=True)
    actual_f,actual_s,ref_f,ref_s=map(np.concatenate,(actual_f,actual_s,ref_f,ref_s))
    with np.load(run/'CALIBRATION_OUTPUTS.npz',allow_pickle=False) as saved:
        np.testing.assert_allclose(ref_f,saved['logits'],atol=3e-4,rtol=3e-4)
        np.testing.assert_allclose(ref_s,saved['log_em_ratio'],atol=3e-4,rtol=3e-4)
        cached=region_outputs(saved['logits'],saved['log_em_ratio'],cal['rows'],1.)
    actual=region_outputs(actual_f,actual_s,cal['rows'],1.); expected=region_outputs(ref_f,ref_s,cal['rows'],1.)
    max_size=max(compare_runtime_outputs(actual,x,cal['rows'],selection['families']) for x in (expected,cached))
    measured,_,details=evaluate_outputs(actual_f,actual_s,cal,retention,STEPS)
    base,_,base_details=cached_evaluation(R22_RUN/'CALIBRATION_OUTPUTS.npz',R22_RUN/'CALIBRATION_DECISIONS.json',cal,retention,0)
    added=compare_r22(measured['metrics'],base['metrics'],short_population(details,cal['rows']),short_population(base_details,cal['rows']))
    guard=android_system_guard(details); validate_android_system_guard(ANDROID_GUARD,guard,android_system_guard(base_details))
    require(guard==selection['android_system_guard'], 'ONNX Android guard differs from saved CAL')
    indices=np.unique(np.linspace(0,len(cal['tiles'])-1,min(256,len(cal['tiles']))).astype(int));samples=np.array(cal['tiles'][indices],copy=True);batches=[]
    for count in (1,7,32,128):
        values=[session.run(['logits','log_em_ratio'],{'tiles':samples[s:s+count]}) for s in range(0,len(samples),count)]
        f=np.concatenate([v[0] for v in values]);z=np.concatenate([v[1] for v in values])
        np.testing.assert_allclose(f,ref_f[indices],atol=2e-4,rtol=2e-4);np.testing.assert_allclose(z,ref_s[indices],atol=2e-4,rtol=2e-4)
        require(f.dtype==z.dtype==np.float32 and f.shape==(len(samples),25) and z.shape==(len(samples),), 'Invalid dynamic output')
        batches.append({'batch_size':count,'samples':len(samples),'passed':True,'font_and_size_checked':True})
    parent_metadata=read(parent_metadata_path)
    require(parent_metadata.get('schema')==METADATA_SCHEMA and parent_metadata.get('families')==selection['families'],'R22 metadata differs')
    metadata=metadata_for_export(selection,cal,parent_metadata,sha(model_path),checkpoint_sha,selection_sha,measured,added)
    dump(output/'metadata.json',metadata);UnifiedFontClassifier(output)
    require(validate(run,data,plan_path).get('trial')==selection['trial']
            and all(sha(p)==h for p,h in {**sources,**evidence}.items()),'Evidence changed during export')
    parity=build_parity_report(selection,selection_sha,checkpoint_sha,sha(model_path),sha(output/'metadata.json'),
        sources,evidence,maximum,max_size,batches,measured,added,guard,max(raw_fold,coarse_error))
    dump(run/'PARITY.json',parity)
    print(json.dumps({'passed':True,'trial':selection['trial'],'model_sha256':parity['model_sha256'],
                      'calibration_tiles':CAL_TILES,'maximum_errors':maximum}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent-metadata',type=Path,default=R22_METADATA)
    export(parser.parse_args())
