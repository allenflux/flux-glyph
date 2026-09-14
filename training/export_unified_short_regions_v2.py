#!/usr/bin/env python3
"""Export a CAL-qualified short-region continuation as one wide 25-output CNN."""
from __future__ import annotations

import argparse
import copy
import json
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
from short_region_objective import (EXTRA_ACCEPTANCE, SHORT_SAMPLING, SHORT_UNKNOWN_SOURCES,
    SHORT_WEIGHT, compare_r22, short_population)
from short_confidence_retention import CONTRACT as CONFIDENCE_RETENTION_CONTRACT
from retention_face_balanced_sampler import SAMPLING as REPLAY_SAMPLING
from retention_source_balanced_sampler import expected_unknown_source_counts

ARCHITECTURE = 'region-cnn64x256-unified-wide-v1'
PARITY_SCHEMA = 'flux-glyph-unified-short-onnx-parity-v2'
METADATA_SCHEMA = 'flux-glyph-unified-region-font-v1'
GROUPS = ('trunk', 'style', 'family_head', 'size_head')
CAL_REGIONS = 18672
CAL_TILES = 37834
STEPS = 2400
R22_RUN = ROOT / 'artifacts/unified-font-v3/run-wide-micro-recovery-v1'
R22_METADATA = ROOT / 'artifacts/unified-font-v3/region-wide-micro-recovery-v1/metadata.json'


def read(path):
    return json.loads(Path(path).read_text())


def validate_confidence_contract(objective):
    require(objective.get('additional_confidence_retention') == CONFIDENCE_RETENTION_CONTRACT
            and objective.get('original_full_correct_teacher_kl_preserved') is True,
            'V2 confidence retention or original full teacher KL contract differs')


def validate_counts(counts, families):
    require(counts.get('optimizer_steps') == STEPS
            and counts.get('replay_rows') == STEPS * 96
            and counts.get('short_rows') == STEPS * 32,
            'Short continuation row totals differ')
    short_family = counts.get('short_by_family', {})
    require(set(short_family) == set(families)
            and all(short_family[name] == STEPS for name in families[:-1])
            and short_family['__unknown__'] == STEPS * 8,
            'Short continuation family quotas differ')
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
    report = counts.get('short_sampling', {})
    require(report.get('steps') == STEPS and report.get('rows') == STEPS * 32
            and report.get('unknown_source_order') == list(SHORT_UNKNOWN_SOURCES),
            'Short sampler report differs')
    entries = report.get('counts')
    require(isinstance(entries, list) and entries
            and sum(row.get('rows', 0) for row in entries) == STEPS * 32,
            'Short sampler detailed counts differ')
    unknown = {name: 0 for name in SHORT_UNKNOWN_SOURCES}
    for row in entries:
        require(row.get('family') in families and row.get('glyph_count') in (1, 2, 3, 4)
                and row.get('view') in SHORT_SAMPLING['views']
                and type(row.get('rows')) is int and row['rows'] > 0,
                'Short sampler count row differs')
        if row['family'] == '__unknown__':
            require(row.get('source_font_family') in unknown, 'Held-out unknown entered short supervision')
            unknown[row['source_font_family']] += row['rows']
    require(sum(unknown.values()) == STEPS * 8 and max(unknown.values()) - min(unknown.values()) <= 1,
            'Short unknown-source round robin differs')
    return counts


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


def validate(run, data, plan_path):
    """Validate frozen TRAIN/CAL evidence without importing Torch or reading CAL pixels."""
    import train_unified_short_regions_v2 as trainer
    run, data, plan_path = map(Path.resolve, (Path(run), Path(data), Path(plan_path)))
    selection = read(run / 'SELECTION.json')
    protocol = read(run / 'TRAINING_FREEZE.json')
    plan = read(plan_path)
    validate_confidence_contract(protocol.get('objective', {}))
    require(selection.get('schema') == 'flux-glyph-unified-short-selection-v2'
            and protocol.get('schema') == 'flux-glyph-unified-short-training-freeze-v2'
            and plan.get('schema') == 'flux-glyph-unified-short-training-plan-v2'
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
    require(bindings.get(str(pre_path)) == pre.get('sha256') == sha(pre_path)
            and proof.get('schema') == 'flux-glyph-short-region-gradient-preflight-v2'
            and proof.get('passed') is True and proof.get('steps') == 4
            and proof.get('bindings') == plan['bindings']
            and proof.get('calibration_inference') is False
            and proof.get('development_read') is False and proof.get('test_read') is False,
            'Short gradient preflight differs')
    require(protocol.get('initializer_checkpoint_sha256') == trainer.BASE_SHA
            and protocol.get('initializer_selection_sha256') == trainer.BASE_SELECTION_SHA
            and protocol.get('initializer_state_sha256') == trainer.BASE_STATE_SHA
            and protocol.get('steps') == protocol.get('evaluation_step') == selection.get('optimizer_steps_executed') == STEPS
            and protocol.get('learning_rate') == trainer.LEARNING_RATE
            and protocol.get('minimum_learning_rate') == trainer.MINIMUM_LEARNING_RATE
            and protocol.get('seed') == trainer.SEED and protocol.get('objective') == trainer.OBJECTIVE
            and protocol.get('short_sampling') == SHORT_SAMPLING
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
    validate_counts(read(counts_path), selection['families'])
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
    return {'kind': 'R22 short-region continuation', 'steps': STEPS,
        'selection_sha256': selection_sha256, 'checkpoint_sha256': checkpoint_sha256,
        'training_protocol_sha256': selection['training_protocol_sha256'],
        'training_counts_sha256': selection['training_counts_sha256'],
        'initial_state_sha256': selection['initializer_state_sha256'],
        'final_state_sha256': selection['state_after_sha256'],
        'initial_parameter_groups_sha256': copy.deepcopy(selection['initial_parameter_groups_sha256']),
        'final_parameter_groups_sha256': copy.deepcopy(selection['selected_parameter_groups_sha256']),
        'objective': copy.deepcopy(selection['objective']), 'short_sampling': copy.deepcopy(selection['short_sampling']),
        'all_parameters_trained': True, 'model_count': 1, 'training_inputs': ['image_tiles'],
        'r22_same_tile_teacher_cache_training_only': True, 'teacher_cache_deployed': False,
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
        'validation': {'kind': 'short_region_continuation_calibration_only_at_export',
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
                        max_size_px, batches, measured, added):
    require(selection.get('calibration_promotion_allowed') is True
            and selection.get('promotion_allowed') is False and len(measured['retention_checks']) == 46
            and all(row.get('passed') is True for row in measured['retention_checks'])
            and added.get('passed') is True and len(added.get('checks', [])) == 7
            and all(row.get('passed') is True for row in added['checks']),
            'Parity cannot bypass the 46+7 CAL gate or pre-DEV state')
    return {'schema': PARITY_SCHEMA, 'passed': True, 'calibration_promotion_allowed': True,
        'promotion_allowed': False, 'development_evaluated': False, 'font_model_count': 1,
        'encoder_count': 1, 'output_family_count': 25, 'network_architecture': ARCHITECTURE,
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
            'Preserve prior short-region exports and parity evidence')
    selection = validate(run, data, plan_path)
    selection_sha = sha(run / 'SELECTION.json'); checkpoint_sha = sha(run / 'model.pth')
    parent_metadata_path = Path(args.parent_metadata).resolve()
    require(parent_metadata_path == R22_METADATA.resolve(), 'Only the pinned R22 metadata may supply source descriptions')
    sources = {str(path.resolve()): sha(path) for path in [Path(__file__),
        ROOT / 'training/evaluate_unified_short_regions_v2.py', ROOT / 'training/train_unified_short_regions_v2.py',
        ROOT / 'training/short_confidence_retention.py',
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
    initializer = torch.load(__import__('train_unified_short_regions_v2').BASE_RUN / 'model.pth',
                             map_location='cpu', weights_only=True)
    require(selection['final_checkpoint_sha256'] == sha(run / 'FINAL.pth'), 'Frozen FINAL checkpoint changed')
    validate_checkpoint(checkpoint, final_checkpoint, selection, selection_sha)
    validate_initializer_checkpoint(initializer, selection,
        __import__('train_unified_short_regions_v2').BASE_STATE_SHA,
        __import__('train_unified_short_regions_v2').BASE_SELECTION_SHA)
    reference = WideRegionFontClassifier(25).cpu().eval()
    reference.load_state_dict(checkpoint['state_dict'], strict=True)
    converted = copy.deepcopy(reference)
    require(sum(isinstance(module, torch.nn.GroupNorm) for module in reference.modules()) == 4
            and replace_groupnorm(converted, high_precision=True) == 4
            and state_sha(converted.state_dict()) == selection['state_after_sha256'],
            'GroupNorm lowering changed short-region parameters')
    cal = load_split(data, 'calibration')
    retention = read(Path(__import__('train_unified_short_regions_v2').RETENTION_PLAN))
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
        sha(output / 'metadata.json'), sources, evidence, maximum, max_size_px, batches, measured, added)
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
