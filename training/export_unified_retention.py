#!/usr/bin/env python3
"""Export a promoted retention checkpoint with the unchanged unified runtime.

CAL parity retains original torch.nn.GroupNorm as its oracle. Neither export nor
release metadata may turn a failed stable validation into a passed one.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'training')]
from train_regions import sha, dump, require, state_sha
import train_unified_retention as retention
from train_unified_regions import ARCHITECTURE, POLICY, UNKNOWN, region_outputs
from export_unified_regions import compare_calibration_outputs, SIZE_ATOL_PX, SIZE_RTOL
from prepare_unified_regions import load_split

SCHEMA = 'flux-glyph-unified-retention-onnx-parity-v1'
GROUPS = ('trunk', 'style', 'family_head', 'size_head')
UNWEIGHTED = 'native_focus_unweighted'
WEIGHTED = 'r21_class_prior_weighted'


def objective_variant(selection):
    variant = selection.get('objective_variant', UNWEIGHTED)
    require(variant in (UNWEIGHTED, WEIGHTED), 'Unregistered retention objective variant')
    return variant


def objective_contract(selection, protocol):
    """Select a named training objective without mutating the frozen base trainer."""
    variant = objective_variant(selection)
    require(objective_variant(protocol) == variant
            and selection.get('objective_variant') == protocol.get('objective_variant'),
            'Retention objective variant differs from its freeze')
    if variant == UNWEIGHTED:
        require('class_prior_weighting' not in selection and 'class_prior_weighting' not in protocol
                and protocol.get('objective') == retention.OBJECTIVE
                and selection.get('objective', retention.OBJECTIVE) == retention.OBJECTIVE,
                'Unweighted retention objective differs from the frozen base objective')
        return {'variant': variant, 'objective': retention.OBJECTIVE,
                'source': ROOT/'training/train_unified_retention.py', 'class_prior_weighting': None}
    import train_unified_retention_balanced as balanced
    expected = balanced.class_prior_weighting(selection['families'])
    expected_objective = balanced.weighted_objective(selection['families'])
    require(selection.get('objective_variant') == protocol.get('objective_variant') == WEIGHTED
            and selection.get('objective') == protocol.get('objective') == expected_objective
            and selection.get('class_prior_weighting') == protocol.get('class_prior_weighting') == expected,
            'Weighted retention objective or R21 class-prior weights differ')
    return {'variant': variant, 'objective': expected_objective,
            'source': ROOT/'training/train_unified_retention_balanced.py', 'class_prior_weighting': expected}


def export_sources(selection):
    names = ['training/export_unified_retention.py', 'training/train_unified_retention.py',
             'training/export_unified_regions.py', 'training/export_region_stable.py',
             'training/train_unified_regions.py', 'training/train_android_regions.py',
             'training/prepare_unified_regions.py', 'training/train_regions.py',
             'training/region_network.py', 'training/network.py',
             'src/flux_glyph/unified_font.py', 'src/flux_glyph/region_font.py']
    if objective_variant(selection) == WEIGHTED:
        names.append('training/train_unified_retention_balanced.py')
    return {str((ROOT/name).resolve()): sha(ROOT/name) for name in names}


def require_fixed_runtime(record):
    require(type(record.get('temperature')) in (int, float)
            and record.get('temperature') == retention.FIXED_RUNTIME['temperature']
            and record.get('gates') == retention.FIXED_RUNTIME['gates'],
            'Retention export cannot change temperature or naming gates')


def runtime_signatures(outputs, families):
    """Include abstention reasons; close logits can still cross two reject gates."""
    gates = retention.FIXED_RUNTIME['gates']
    result = []
    for row in outputs:
        winner = families[row['predicted']]
        family = size = None
        status = 'uncertain'
        if winner == UNKNOWN:
            status, reason = 'out_of_scope', 'unknown_font_rejected'
        elif row['agreement'] < gates['min_patch_agreement']:
            reason = 'mixed_or_ambiguous_region'
        elif row['score'] < gates['min_score']:
            reason = 'below_score_gate'
        elif row['margin'] <= 1e-8 or row['margin'] < gates['min_margin']:
            reason = 'ambiguous_neural_families'
        else:
            status, reason, family = 'candidate', 'region_neural_family_candidate', winner
            size = row['size_spread'] <= retention.FIXED_RUNTIME['max_size_relative_spread']
        result.append((status, reason, family, size))
    return result


def compare_runtime_outputs(actual, expected, rows, families):
    require(runtime_signatures(actual, families) == runtime_signatures(expected, families),
            'ONNX changed a frozen runtime status, rejection reason or size availability')
    return compare_calibration_outputs(actual, expected, rows, families, retention.FIXED_RUNTIME['gates'])


def validate_checkpoint(checkpoint, selection, selection_sha256):
    require(checkpoint.get('families') == selection['families']
            and checkpoint.get('architecture') == ARCHITECTURE
            and checkpoint.get('selection_sha256') == selection_sha256
            and state_sha(checkpoint['state_dict']) == selection['state_after_sha256'],
            'Selected retention checkpoint binding differs')
    for name in GROUPS:
        group = {key: value for key, value in checkpoint['state_dict'].items() if key.startswith(name+'.')}
        require(group and state_sha(group) == selection['final_parameter_groups_sha256'][name],
                'Selected retention parameter group differs')


def checked_file(run, entry, expected):
    """Saved-step evidence must stay at its declared in-run canonical path."""
    require(isinstance(entry, dict) and entry.get('path') == expected,
            'Retention step evidence path differs')
    path = (run/expected).resolve()
    require(path.is_relative_to(run.resolve()) and path.is_file() and sha(path) == entry.get('sha256'),
            'Retention step evidence changed')
    return path


def calibration_metadata(data):
    """No Torch, tile loading or non-CAL partition reading is needed here."""
    manifest = json.loads((data/'MANIFEST.json').read_text())
    part = json.loads((data/'calibration/MANIFEST.json').read_text())
    require(part['split'] == 'calibration' and part['root_manifest_sha256'] == sha(data/'MANIFEST.json')
            and part['families'] == manifest['families'] and part['metadata']['path'] == 'rows.json',
            'Retention CAL metadata contract differs')
    path = data/'calibration/rows.json'
    require(sha(path) == part['metadata']['sha256'], 'Retention CAL row bytes changed')
    rows = json.loads(path.read_text())
    require(len(rows) == part['views'] and all(row['split'] == 'calibration' for row in rows),
            'Retention CAL row partition differs')
    return {'families': manifest['families'], 'manifest': manifest, 'partition': part, 'rows': rows,
            'manifest_sha256': sha(data/'MANIFEST.json')}


def strip_artifacts(record):
    return {key: value for key, value in record.items() if key != 'artifacts'}


def cached_evaluation(outputs_path, decisions_path, cal, plan, step):
    with np.load(outputs_path, allow_pickle=False) as saved:
        require(set(saved.files) == {'logits', 'log_em_ratio'}, 'Retention cached output schema differs')
        record, outputs, details = retention.evaluate_outputs(saved['logits'], saved['log_em_ratio'], cal, plan, step)
    cached = json.loads(decisions_path.read_text())
    require(cached == {'families': cal['families'], 'records': outputs},
            'Cached retention decisions do not reproduce frozen output arrays')
    require_fixed_runtime(record)
    return record, outputs, details


def validate(run, data):
    """Strict frozen release gate; intentionally imports no Torch or ONNX."""
    run, data = Path(run).resolve(), Path(data).resolve()
    selection = json.loads((run/'SELECTION.json').read_text())
    require(selection.get('schema') == 'flux-glyph-unified-retention-selection-v1'
            and selection.get('architecture') == ARCHITECTURE and selection.get('policy') == POLICY
            and selection.get('fixed_runtime') == retention.FIXED_RUNTIME
            and selection.get('promotion_allowed') is True,
            'Retention export requires a frozen promotable checkpoint and unchanged runtime')
    require(not (run/'NO_PROMOTABLE_CHECKPOINT.json').exists(), 'A failed promotion run cannot be exported')
    for key in ('test_read', 'development_holdout_read', 'platform_routing', 'score_merging', 'runtime_gates_searched'):
        require(selection.get(key) is False, 'Retention selection changed image-only/no-search scope: '+key)
    require(selection.get('model_count') == 1 and type(selection.get('passed')) is bool
            and selection.get('calibration_passed') is selection['passed'], 'Retention stable status differs')
    bindings = selection.get('bindings')
    require(isinstance(bindings, dict) and bindings, 'Missing retention source bindings')
    protocol = json.loads((run/'TRAINING_FREEZE.json').read_text())
    require(sha(run/'TRAINING_FREEZE.json') == selection['training_protocol_sha256']
            and protocol.get('schema') == 'flux-glyph-unified-retention-training-protocol-v1',
            'Retention training freeze changed')
    shared = ('architecture', 'policy', 'fixed_runtime', 'families', 'bindings', 'parent_checkpoint',
              'parent_selection_sha256', 'parent_metadata', 'retention_plan', 'initializer_evidence',
              'baseline_bindings', 'test_read', 'development_holdout_read', 'model_count',
              'platform_routing', 'score_merging', 'runtime_gates_searched')
    require(all(protocol.get(key) == selection.get(key) for key in shared), 'Retention freeze/selection differs')
    objective = objective_contract(selection, protocol)
    require(type(protocol.get('steps')) is int and protocol['steps'] == retention.STEPS
            and selection.get('optimizer_steps_executed') == retention.STEPS
            and protocol.get('eval_every') == retention.EVAL_EVERY
            and protocol.get('batch_size') == retention.BATCH_SIZE
            and protocol.get('learning_rate') == retention.LEARNING_RATE
            and protocol.get('minimum_learning_rate') == retention.MINIMUM_LEARNING_RATE
            and protocol.get('sampling') == retention.SAMPLING
            and protocol.get('training_inputs') == ['image_tiles'] and protocol.get('device') == selection.get('training_device')
            and protocol.get('device') in ('mps', 'cpu'), 'Retention optimizer or sampling protocol differs')
    parent_path = (ROOT/selection['parent_checkpoint']['path']).resolve()
    parent_meta_path = (ROOT/selection['parent_metadata']['path']).resolve()
    plan_path = Path(selection['retention_plan']['path']).resolve()
    required = [data/'MANIFEST.json', data/'train/MANIFEST.json', data/'calibration/MANIFEST.json',
        data/'train/rows.json', data/'calibration/rows.json',
        ROOT/'training/train_unified_retention.py', ROOT/'training/train_unified_regions.py',
        ROOT/'training/prepare_unified_regions.py', ROOT/'training/train_android_regions.py',
        ROOT/'training/train_regions.py', ROOT/'training/region_network.py', ROOT/'training/network.py',
        ROOT/'training/evaluate_unified_retention.py', ROOT/'src/flux_glyph/unified_font.py',
        ROOT/'src/flux_glyph/region_font.py', parent_path, parent_path.parent/'SELECTION.json',
        parent_path.parent/'PARITY.json', parent_path.parent/'DEVELOPMENT_REGRESSION.json',
        parent_path.parent/'CALIBRATION_OUTPUTS.npz', parent_path.parent/'CALIBRATION_DECISIONS.json',
        parent_meta_path, plan_path, objective['source']]
    for split in ('train', 'calibration'):
        part = json.loads((data/split/'MANIFEST.json').read_text())
        for key in ('array', 'metadata'):
            path = (data/split/part[key]['path']).resolve()
            require(path.parent == data/split and bindings.get(str(path)) == part[key]['sha256'],
                    'Retention tensor or row file is absent from exact-path source bindings')
            required.append(path)
    for path in required:
        require(str(path.resolve()) in bindings and sha(path) == bindings[str(path.resolve())],
                'Retention export input is absent from frozen exact-path bindings')
    retention.verify_bindings(bindings)
    require(selection['retention_plan']['sha256'] == sha(plan_path)
            and selection['parent_checkpoint']['sha256'] == sha(parent_path)
            and selection['parent_metadata']['sha256'] == sha(parent_meta_path)
            and selection['parent_selection_sha256'] == sha(parent_path.parent/'SELECTION.json')
            and selection['data_manifest_sha256'] == sha(data/'MANIFEST.json'), 'Retention input SHA differs')
    plan = retention.read_plan(plan_path, data, parent_path)
    for entry in plan['bindings'].values():
        require(sha(ROOT/entry['path']) == entry['sha256'], 'Retention population provenance changed')
    for key in ('parent_checkpoint', 'parent_selection_sha256', 'parent_metadata'):
        require(plan[key] == selection[key], 'Retention plan and frozen parent differ')
    cal = calibration_metadata(data)
    require(cal['families'] == selection['families'] == plan['families']
            and len(cal['families']) == 25 and cal['families'].count(UNKNOWN) == 1, 'Retention class order differs')
    parent = json.loads((parent_path.parent/'SELECTION.json').read_text())
    parent_metadata = json.loads(parent_meta_path.read_text())
    parent_parity = json.loads((parent_path.parent/'PARITY.json').read_text())
    require(parent.get('schema') == 'flux-glyph-unified-training-selection-v1'
            and parent.get('families') == selection['families'] and parent.get('policy') == POLICY
            and parent.get('test_read') is False and parent.get('development_holdout_read') is False
            and parent['data_manifest_sha256'] == selection['data_manifest_sha256'], 'Retention parent selection differs')
    require_fixed_runtime(parent['selected'])
    parent_model = parent_meta_path.parent/parent_metadata['model']['path']
    require(parent_metadata.get('schema') == 'flux-glyph-unified-region-font-v1'
            and parent_metadata.get('font_mode') == 'unified' and parent_metadata['families'] == selection['families']
            and parent_metadata['max_size_relative_spread'] == retention.FIXED_RUNTIME['max_size_relative_spread']
            and 'rejection' not in parent_metadata and 'verifier' not in parent_metadata
            and sha(parent_model) == parent_metadata['model']['sha256'] == parent_parity['model_sha256']
            and bindings.get(str(parent_model.resolve())) == sha(parent_model)
            and parent_parity.get('passed') is True
            and parent_parity['checkpoint_sha256'] == sha(parent_path)
            and parent_parity['selection_sha256'] == selection['parent_selection_sha256']
            and parent_metadata['training']['selection_sha256'] == selection['parent_selection_sha256']
            and parent_metadata['training']['checkpoint_sha256'] == sha(parent_path), 'R21 parent model provenance differs')
    require_fixed_runtime(parent_metadata)
    groups, final = selection['initial_parameter_groups_sha256'], selection['final_parameter_groups_sha256']
    require(set(groups) == set(final) == set(GROUPS) and groups == protocol['initial_parameter_groups_sha256']
            == parent['final_parameter_groups_sha256'] and all(groups[key] != final[key] for key in GROUPS)
            and selection['state_before_sha256'] == protocol['initial_state_sha256'] == parent['state_after_sha256']
            and selection['state_before_sha256'] != selection['state_after_sha256'], 'Retention did not inherit/train every parameter group')
    inheritance = selection['initializer_evidence']
    require(all(inheritance.get(key) is True for key in ('all_family_rows_inherited', 'all_parameters_inherited', 'all_parameters_trainable'))
            and inheritance.get('family_count') == 25 and inheritance.get('new_random_output_rows') == 0
            and inheritance.get('source_state_sha256') == selection['state_before_sha256']
            and Path(inheritance['checkpoint']['path']).resolve() == parent_path
            and inheritance['checkpoint']['sha256'] == sha(parent_path)
            and inheritance['selection_sha256'] == selection['parent_selection_sha256']
            and inheritance['source_selected_step'] == parent['selected']['step']
            and inheritance['source_optimizer_steps_executed'] == parent['optimizer_steps_executed'],
            'Retention all-state initialization evidence differs')
    baseline_names = {'BASELINE.json', 'BASELINE_CALIBRATION_OUTPUTS.npz', 'BASELINE_CALIBRATION_DECISIONS.json'}
    require(set(selection['baseline_bindings']) == baseline_names
            and all(sha(run/name) == digest for name, digest in selection['baseline_bindings'].items()),
            'Frozen parent baseline evidence changed')
    require(sha(run/'BASELINE_CALIBRATION_OUTPUTS.npz') == sha(parent_path.parent/'CALIBRATION_OUTPUTS.npz')
            == parent['calibration_outputs_sha256']
            and sha(run/'BASELINE_CALIBRATION_DECISIONS.json') == sha(parent_path.parent/'CALIBRATION_DECISIONS.json')
            == parent['calibration_decisions_sha256'], 'Baseline is not the frozen parent CAL cache')
    baseline, _, _ = cached_evaluation(run/'BASELINE_CALIBRATION_OUTPUTS.npz',
        run/'BASELINE_CALIBRATION_DECISIONS.json', cal, plan, 0)
    require(baseline == json.loads((run/'BASELINE.json').read_text())
            and baseline['metrics'] == parent['selected']['metrics'], 'Baseline retention metrics do not reproduce the parent')
    history = selection.get('history')
    require(isinstance(history, list) and [row['step'] for row in history]
            == list(range(retention.EVAL_EVERY, retention.STEPS+1, retention.EVAL_EVERY)),
            'Retention history must include every completed checkpoint')
    for record in history:
        require_fixed_runtime(record)
        step = record['step']; directory = f'checkpoints/step{step:05d}'
        require(set(record.get('artifacts', {})) == {'checkpoint', 'outputs', 'decisions', 'metrics'},
                'Incomplete retention checkpoint evidence')
        paths = {key: checked_file(run, record['artifacts'][key], directory+'/'+name) for key, name in
            [('checkpoint', 'model.pth'), ('outputs', 'CALIBRATION_OUTPUTS.npz'),
             ('decisions', 'CALIBRATION_DECISIONS.json'), ('metrics', 'METRICS.json')]}
        recomputed, _, _ = cached_evaluation(paths['outputs'], paths['decisions'], cal, plan, step)
        require(recomputed == strip_artifacts(record) == json.loads(paths['metrics'].read_text()),
                'Retention checkpoint metrics do not reproduce its cached fixed-runtime outputs')
    selected = selection['selected']
    require(selected in history and selected == max(history, key=retention.retention_rank)
            and selected['promotion_allowed'] is True and selection['promotion_allowed'] is True
            and selected['metrics']['passed'] is selection['passed'], 'Retention selection is not its promotable frozen winner')
    for key, name, field in [('outputs', 'CALIBRATION_OUTPUTS.npz', 'calibration_outputs_sha256'),
                             ('decisions', 'CALIBRATION_DECISIONS.json', 'calibration_decisions_sha256')]:
        require(sha(run/name) == selection[field] == selected['artifacts'][key]['sha256'],
                'Final CAL outputs are not the selected checkpoint bytes')
    require(sha(run/'SAMPLING.json') == selection['sampling_sha256'], 'Retention sampling evidence changed')
    sampled = json.loads((run/'SAMPLING.json').read_text())
    expected_slots = {key: retention.SAMPLING[key]*retention.STEPS for key in
                      ('base_known', 'base_unknown', 'ios_native_pingfang', 'ios_native_sfpro_helvetica', 'ios_native_original_eight')}
    require(sampled['slots'] == expected_slots and sampled['base']['known_rows'] == expected_slots['base_known']
            and sampled['base']['unknown_rows'] == expected_slots['base_unknown']
            and all(sum(sampled[key].values()) == retention.STEPS*retention.BATCH_SIZE
                    for key in ('family_rows', 'domain_rows', 'view_rows')),
            'Retention TRAIN sampling counts differ from the frozen protocol')
    if objective['variant'] == WEIGHTED:
        expected_counts = objective['class_prior_weighting']['sampled_counts']
        require(sampled['family_rows'] == {family: count*retention.STEPS for family, count in expected_counts.items()},
                'Weighted retention class-prior counts do not match actual TRAIN sampling')
    return selection


def calibration_evidence(run, data, selection):
    run, data = Path(run).resolve(), Path(data).resolve()
    part = json.loads((data/'calibration/MANIFEST.json').read_text())
    paths = [run/'SELECTION.json', run/'TRAINING_FREEZE.json', run/'model.pth',
             run/'CALIBRATION_DECISIONS.json', run/'CALIBRATION_OUTPUTS.npz', run/'SAMPLING.json',
             run/'BASELINE.json', run/'BASELINE_CALIBRATION_OUTPUTS.npz', run/'BASELINE_CALIBRATION_DECISIONS.json',
             data/'MANIFEST.json', data/'calibration/MANIFEST.json',
             Path(selection['retention_plan']['path'])]
    for key in ('array', 'metadata'):
        path = (data/'calibration'/part[key]['path']).resolve()
        require(path.parent == data/'calibration' and sha(path) == part[key]['sha256'], 'CAL parity asset differs')
        paths.append(path)
    for record in selection['history']:
        for entry in record['artifacts'].values():
            paths.append((run/entry['path']).resolve())
    return {str(path.resolve()): sha(path) for path in paths}


def metadata_for_export(selection, cal, parent_metadata, model_sha256, checkpoint_sha256,
                        selection_sha256, runtime_record):
    require(runtime_record['promotion_allowed'] is True, 'Retention promotion failed on actual ONNX outputs')
    require_fixed_runtime(runtime_record)
    measured = runtime_record['metrics']
    return {'schema': 'flux-glyph-unified-region-font-v1', 'algorithm': 'unified-region-cnn64x256-v1',
        'font_mode': 'unified', 'data_kind': 'native_mobile_screenshots',
        'model': {'path': 'model.onnx', 'sha256': model_sha256}, 'network_architecture': ARCHITECTURE,
        'families': selection['families'], **copy.deepcopy(retention.FIXED_RUNTIME),
        'font_label_groups': cal['manifest']['font_label_groups'],
        'font_sources': copy.deepcopy(parent_metadata.get('font_sources', {})),
        # No independent blind test was run. Passing a CAL retention protocol
        # does not make the original stable release criteria pass.
        'release_tier': 'experimental', 'stable_validation_passed': False, 'test_passed': False,
        'validation': {'kind': 'fixed_runtime_retention_calibration_only_at_export',
            'objective_variant': objective_variant(selection),
            'class_prior_loss_weighting': objective_variant(selection) == WEIGHTED,
            'calibration_passed': measured['passed'], 'retention_promotion_allowed': True,
            'retention_plan_sha256': selection['retention_plan']['sha256'],
            'fixed_runtime': copy.deepcopy(retention.FIXED_RUNTIME),
            'retention_checks': runtime_record['retention_checks'],
            'retention_populations': {name: {key: population[key] for key in
                ('views', 'named', 'correct_named', 'wrong_named', 'known_correct_coverage',
                 'named_precision', 'unknown_not_named_rate')}
                for name, population in runtime_record['retention_populations'].items()},
            'development_holdout_evaluated': False, 'blind_test_performed': False,
            'named_precision': measured['named_precision'], 'known_correct_coverage': measured['known_correct_coverage'],
            'unknown_not_named_rate': measured['unknown_not_named_rate'],
            'original_stable_calibration_checks': measured['checks'],
            'scores_are_correctness_probabilities': False, 'model_count': 1, 'platform_routing': False},
        'training': {'selection_sha256': selection_sha256, 'checkpoint_sha256': checkpoint_sha256,
            'objective_variant': objective_variant(selection),
            'objective': copy.deepcopy(selection.get('objective', retention.OBJECTIVE)),
            'class_prior_loss_weighting': objective_variant(selection) == WEIGHTED,
            'class_prior_weighting': copy.deepcopy(selection.get('class_prior_weighting')),
            'data_manifest_sha256': selection['data_manifest_sha256'],
            'retention_plan_sha256': selection['retention_plan']['sha256'],
            'fixed_runtime': copy.deepcopy(retention.FIXED_RUNTIME),
            'optimizer_steps_executed': selection['optimizer_steps_executed'],
            'selected_step': selection['selected']['step'], 'ocr_text_used': False,
            'training_device': selection['training_device'], 'model_count': 1, 'platform_routing': False,
            'score_merging': False, 'teacher_outputs_used': False, 'all_parameters_trained': True,
            'all_parent_class_rows_inherited': True, 'initializer_family_count': len(selection['families']),
            'parent_checkpoint_sha256': selection['parent_checkpoint']['sha256'],
            'parent_metadata_sha256': selection['parent_metadata']['sha256'],
            'development_holdout_is_blind_test': False}}


def export(args):
    import torch
    import onnx
    import onnxruntime as ort
    from region_network import RegionFontClassifier
    from export_region_stable import replace_groupnorm
    from flux_glyph.unified_font import UnifiedFontClassifier
    args.run, args.data, args.output = map(lambda path: Path(path).resolve(), (args.run, args.data, args.output))
    require(not args.output.exists() and not (args.run/'PARITY.json').exists(),
            'Retention export output must be new; preserve prior failure evidence')
    selection = validate(args.run, args.data)
    plan = retention.read_plan(Path(selection['retention_plan']['path']), args.data,
                               (ROOT/selection['parent_checkpoint']['path']).resolve())
    cal = load_split(args.data, 'calibration')
    require(cal['families'] == selection['families'], 'Retention CAL family order changed')
    checkpoint_sha = sha(args.run/'model.pth')
    selection_sha = sha(args.run/'SELECTION.json')
    checkpoint = torch.load(args.run/'model.pth', map_location='cpu', weights_only=True)
    validate_checkpoint(checkpoint, selection, selection_sha)
    require(sha(args.run/'model.pth') == checkpoint_sha, 'Checkpoint changed while loading')
    step_path = args.run/selection['selected']['artifacts']['checkpoint']['path']
    step_checkpoint = torch.load(step_path, map_location='cpu', weights_only=True)
    require(step_checkpoint.get('families') == selection['families']
            and step_checkpoint.get('architecture') == ARCHITECTURE
            and step_checkpoint.get('step') == selection['selected']['step']
            and step_checkpoint.get('training_protocol_sha256') == selection['training_protocol_sha256']
            and state_sha(step_checkpoint['state_dict']) == selection['state_after_sha256'],
            'Final weights differ from their selected saved-step checkpoint')
    sources = export_sources(selection)
    evidence = calibration_evidence(args.run, args.data, selection)
    reference = RegionFontClassifier(len(selection['families'])).cpu().eval()
    reference.load_state_dict(checkpoint['state_dict'], strict=True)
    require(sum(isinstance(module, torch.nn.GroupNorm) for module in reference.modules()) == 4,
            'Original Torch GroupNorm parity reference differs')
    converted = copy.deepcopy(reference)
    require(replace_groupnorm(converted, high_precision=True) == 4
            and state_sha(converted.state_dict()) == state_sha(reference.state_dict()) == selection['state_after_sha256'],
            'GroupNorm lowering changed selected weights')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    path = args.output/'model.onnx'
    torch.onnx.export(converted, torch.zeros(2, 1, 64, 256), path,
        input_names=['tiles'], output_names=['logits', 'log_em_ratio'],
        dynamic_axes={'tiles': {0: 'batch'}, 'logits': {0: 'batch'}, 'log_em_ratio': {0: 'batch'}},
        opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    actual_logits, actual_sizes, reference_logits, reference_sizes = [], [], [], []
    max_logit = max_size = 0.
    with torch.inference_mode():
        for start in range(0, len(cal['tiles']), 128):
            block = np.array(cal['tiles'][start:start+128], copy=True)
            logits, sizes = reference(torch.from_numpy(block))
            out, out_size = session.run(['logits', 'log_em_ratio'], {'tiles': block})
            np.testing.assert_allclose(out, logits.numpy(), atol=2e-4, rtol=2e-4)
            np.testing.assert_allclose(out_size, sizes.numpy(), atol=2e-4, rtol=2e-4)
            max_logit = max(max_logit, float(np.max(np.abs(out-logits.numpy()))))
            max_size = max(max_size, float(np.max(np.abs(out_size-sizes.numpy()))))
            actual_logits.append(out); actual_sizes.append(out_size)
            reference_logits.append(logits.numpy()); reference_sizes.append(sizes.numpy())
    actual_logits, actual_sizes = np.concatenate(actual_logits), np.concatenate(actual_sizes)
    reference_logits, reference_sizes = np.concatenate(reference_logits), np.concatenate(reference_sizes)
    selected = selection['selected']
    temperature = retention.FIXED_RUNTIME['temperature']
    actual = region_outputs(actual_logits, actual_sizes, cal['rows'], temperature)
    expected = region_outputs(reference_logits, reference_sizes, cal['rows'], temperature)
    stored_record, stored_outputs, _ = cached_evaluation(args.run/'CALIBRATION_OUTPUTS.npz',
        args.run/'CALIBRATION_DECISIONS.json', cal, plan, selected['step'])
    require(stored_record == strip_artifacts(selected), 'Selected cached retention metrics do not reproduce exactly')
    with np.load(args.run/'CALIBRATION_OUTPUTS.npz', allow_pickle=False) as saved:
        np.testing.assert_allclose(reference_logits, saved['logits'], atol=3e-4, rtol=3e-4)
        np.testing.assert_allclose(reference_sizes, saved['log_em_ratio'], atol=3e-4, rtol=3e-4)
    max_size_px = max(compare_runtime_outputs(actual, other, cal['rows'], cal['families'])
                      for other in (expected, stored_outputs))
    runtime_record, reproduced_outputs, _ = retention.evaluate_outputs(actual_logits, actual_sizes, cal, plan, selected['step'])
    require(reproduced_outputs == actual and runtime_record['promotion_allowed'] is True
            and runtime_record['metrics']['checks'] == selected['metrics']['checks']
            and runtime_record['metrics']['passed'] is selected['metrics']['passed']
            and [row['passed'] for row in runtime_record['retention_checks']]
                == [row['passed'] for row in selected['retention_checks']],
            'ONNX changed retention promotion or original stable CAL acceptance')
    require(all(runtime_record['metrics'][key] == selected['metrics'][key]
                for key in ('named', 'correct_named', 'wrong_named', 'unknown_wrongly_named')),
            'ONNX changed selected CAL counts')
    indices = np.unique(np.linspace(0, len(cal['tiles'])-1, min(256, len(cal['tiles']))).astype(int))
    sample = np.array(cal['tiles'][indices], copy=True)
    batch_checks = []
    for count in (1, 7, 32, 128):
        blocks = [session.run(['logits', 'log_em_ratio'], {'tiles': sample[start:start+count]})
                  for start in range(0, len(sample), count)]
        out, size = np.concatenate([value[0] for value in blocks]), np.concatenate([value[1] for value in blocks])
        require(out.dtype == size.dtype == np.float32 and out.shape == (len(sample), len(selection['families']))
                and size.shape == (len(sample),) and np.isfinite(out).all() and np.isfinite(size).all(),
                'Invalid dynamic-batch ONNX outputs')
        np.testing.assert_allclose(out, reference_logits[indices], atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(size, reference_sizes[indices], atol=2e-4, rtol=2e-4)
        batch_checks.append({'batch_size': count, 'samples': len(sample), 'passed': True, 'font_and_size_checked': True})
    parent_metadata = json.loads((ROOT/selection['parent_metadata']['path']).read_text())
    metadata = metadata_for_export(selection, cal, parent_metadata, sha(path), checkpoint_sha, selection_sha, runtime_record)
    dump(args.output/'metadata.json', metadata)
    UnifiedFontClassifier(args.output)
    require(validate(args.run, args.data) == selection and sha(args.run/'model.pth') == checkpoint_sha
            and state_sha(reference.state_dict()) == state_sha(converted.state_dict()) == selection['state_after_sha256']
            and all(sha(path) == digest for path, digest in {**sources, **evidence}.items()),
            'Frozen source, CAL evidence or selected model changed during export')
    report = {'schema': SCHEMA, 'passed': True, 'promotion_allowed': True, 'test_read': False,
        'development_holdout_read': False, 'user_images_read': False,
        'network_architecture': ARCHITECTURE, 'font_model_count': 1, 'output_family_count': len(selection['families']),
        'selection_sha256': selection_sha, 'checkpoint_sha256': checkpoint_sha, 'model_sha256': sha(path),
        'metadata_sha256': sha(args.output/'metadata.json'), 'source_bindings': sources, 'calibration_bindings': evidence,
        'retention_plan_sha256': selection['retention_plan']['sha256'],
        'objective_variant': objective_variant(selection),
        'class_prior_weighting': copy.deepcopy(selection.get('class_prior_weighting')),
        'fixed_runtime': retention.FIXED_RUNTIME, 'runtime_gates_changed': False,
        'calibration_regions': len(cal['rows']), 'calibration_tiles': len(cal['tiles']),
        'original_torch_groupnorm_reference': True, 'frozen_parameters_unchanged': True,
        'calibration_font_decisions_identical': True, 'calibration_runtime_reasons_identical': True,
        'calibration_scores_close': True, 'calibration_size_availability_identical': True,
        'calibration_size_values_close': True, 'max_logit_absolute_error': max_logit, 'max_size_absolute_error': max_size,
        'max_size_pixel_error': max_size_px, 'size_value_atol_px': SIZE_ATOL_PX, 'size_value_rtol': SIZE_RTOL,
        'cached_mps_outputs_reproduce_selected_metrics': True,
        'cached_output_execution': {'training_device': selection['training_device'], 'reference_device': 'cpu',
                                    'onnx_provider': 'CPUExecutionProvider'},
        'frozen_retention_summary': strip_artifacts(selected), 'runtime_retention_summary': runtime_record,
        'runtime_calibration_metrics': runtime_record['metrics'], 'batch_checks': batch_checks}
    dump(args.run/'PARITY.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'data', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    export(parser.parse_args())
