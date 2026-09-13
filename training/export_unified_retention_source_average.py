#!/usr/bin/env python3
"""Export one fixed full-state average as one image CNN, with exact source proof."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'training')]
from train_regions import require, sha, dump, state_sha
from train_unified_retention import FIXED_RUNTIME, evaluate_outputs, parameter_groups
from export_unified_retention_core import (compare_runtime_outputs, require_fixed_runtime,
    calibration_metadata, cached_evaluation, strip_artifacts)

ARCHITECTURE = 'region-cnn64x256-unified-v1'
SCHEMA = 'flux-glyph-unified-retention-onnx-parity-v1'
GROUPS = ('trunk', 'style', 'family_head', 'size_head')


def verify_average_state(core_state, source_state, candidate_state):
    """Independently reconstruct the one allowed mean; compare every output byte."""
    import torch
    states = (core_state, source_state, candidate_state)
    require(all(isinstance(state, dict) and state for state in states)
            and all(set(state) == set(core_state) for state in states), 'Average state keys differ')
    expected = {}
    for name, original in core_state.items():
        values = [state[name] for state in states]
        require(all(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                    and value.shape == original.shape and value.dtype == original.dtype
                    and not value.is_complex() and bool(torch.isfinite(value).all()) for value in values),
                'Average state tensor, shape, dtype, CPU placement or finiteness differs: '+name)
        if original.is_floating_point():
            expected[name] = ((original.detach().to(torch.float64)
                              + source_state[name].detach().to(torch.float64))/2.).to(original.dtype)
        else:
            require(torch.equal(original, source_state[name]), 'Nonfloating source buffers differ: '+name)
            expected[name] = original.detach().clone()
    require(state_sha(expected) == state_sha(candidate_state), 'Candidate is not the exact fixed CPU float64 50/50 full-state mean')
    return state_sha(expected)


def validate_checkpoint(checkpoint, selection):
    require(checkpoint.get('families') == selection['families'] and checkpoint.get('architecture') == ARCHITECTURE
            and checkpoint.get('step') == 0 and checkpoint.get('optimizer_steps_executed') == 0
            and checkpoint.get('source_optimizer_steps_executed') == 3000
            and checkpoint.get('averaging_freeze_sha256') == selection['averaging_freeze_sha256']
            and state_sha(checkpoint['state_dict']) == selection['state_sha256']
            and parameter_groups(checkpoint['state_dict']) == selection['parameter_groups_sha256']
            and not any(key.startswith('residual_family_head.') for key in checkpoint['state_dict']),
            'Averaged checkpoint, freeze, complete CNN groups or zero-optimizer identity differs')


def validate(run, data):
    """Validate completed source closures and the one cached average; no image forward."""
    import average_unified_retention_source_balanced as averaging
    run, data = Path(run).resolve(), Path(data).resolve()
    selection = averaging.read(run/'SELECTION.json')
    require(selection.get('schema') == 'flux-glyph-unified-retention-source-average-selection-v1'
            and selection.get('promotion_allowed') is True and not (run/'NO_PROMOTABLE_CHECKPOINT.json').exists(),
            'The fixed average has no promotable frozen candidate')
    freeze = averaging.read(run/'AVERAGING_FREEZE.json')
    require(freeze.get('schema') == 'flux-glyph-unified-retention-source-averaging-freeze-v1'
            and sha(run/'AVERAGING_FREEZE.json') == selection['averaging_freeze_sha256']
            and all(selection.get(key) == value for key, value in freeze.items() if key != 'schema'),
            'Average freeze and selection differ')
    expected = {'architecture': ARCHITECTURE, 'objective_variant': averaging.OBJECTIVE_VARIANT,
        'method': averaging.METHOD, 'fixed_runtime': FIXED_RUNTIME, 'runtime_gates_changed': False,
        'optimizer_steps_executed': 0, 'source_optimizer_steps_executed': 3000, 'source_selected_step': 1500,
        'core_optimizer_steps_executed': 1500, 'core_selected_step': 1000,
        'model_count': 1, 'encoder_count': 1, 'teacher_in_deployed_model': False,
        'platform_routing': False, 'score_merging': False, 'candidate_step': 0, 'selection_search_performed': False,
        'test_read': False, 'development_holdout_read': False}
    require(all(freeze.get(key) == value for key, value in expected.items())
            and freeze.get('calibration_device') in ('cpu', 'mps') and type(selection.get('passed')) is bool
            and selection.get('calibration_passed') is selection['passed'], 'Fixed full-state average contract differs')
    evidence = averaging.validate_sources(data)
    for key in ('families', 'source_checkpoints', 'bindings', 'data_manifest_sha256', 'parent_metadata', 'parent_checkpoint'):
        require(selection[key] == evidence[key], 'Average source closure differs: '+key)
    require(selection['source_training_objectives'] == [row['objective'] for row in evidence['source_selections']]
            and selection['retention_plan'] == {'path': str(averaging.PLAN), 'sha256': averaging.PLAN_SHA},
            'Averaging source objectives or the original retention plan changed')
    identities = selection['source_checkpoints']
    require(len(identities) == 2 and [row['role'] for row in identities] == averaging.METHOD['source_order']
            and [row['selected_step'] for row in identities] == [1000, 1500]
            and [row['optimizer_steps_executed'] for row in identities] == [1500, 3000]
            and all(selection['state_sha256'] != row['state_sha256'] for row in identities),
            'Average source order, optimizer lineage or distinct candidate state differs')
    for groups in [selection['parameter_groups_sha256'], *[row['parameter_groups_sha256'] for row in identities]]:
        require(set(groups) == set(GROUPS) and all(isinstance(value, str) and len(value) == 64 for value in groups.values()),
                'Incomplete full-CNN average parameter-group evidence')
    for name, field in [('model.pth', 'checkpoint_sha256'), ('CALIBRATION_OUTPUTS.npz', 'calibration_outputs_sha256'),
                        ('CALIBRATION_DECISIONS.json', 'calibration_decisions_sha256'), ('RESULT.json', 'result_sha256')]:
        require(sha(run/name) == selection[field], 'Saved average evidence changed: '+name)
    cal = calibration_metadata(data)
    require(cal['families'] == selection['families'] and cal['manifest_sha256'] == selection['data_manifest_sha256'],
            'Average calibration population or class order differs')
    actual, _, _ = cached_evaluation(run/'CALIBRATION_OUTPUTS.npz', run/'CALIBRATION_DECISIONS.json', cal, evidence['plan'], 0)
    require(actual == selection['selected'] == averaging.read(run/'RESULT.json')
            and selection['history'] == [actual] and actual['promotion_allowed'] is True
            and actual['metrics']['passed'] is selection['passed'] and len(actual['retention_checks']) == 46,
            'The one average does not reproduce its original 46-condition CAL result')
    report = averaging.read(run/'report.json')
    report_keys = ('objective_variant', 'method', 'optimizer_steps_executed', 'source_optimizer_steps_executed',
        'source_selected_step', 'core_optimizer_steps_executed', 'core_selected_step', 'state_sha256',
        'parameter_groups_sha256', 'promotion_allowed', 'calibration_passed', 'averaging_freeze_sha256',
        'checkpoint_sha256', 'calibration_outputs_sha256', 'calibration_decisions_sha256', 'result_sha256', 'fixed_runtime')
    require(all(report.get(key) == selection[key] for key in report_keys)
            and report.get('schema') == 'flux-glyph-unified-retention-source-average-result-v1'
            and report.get('status') == 'PROMOTABLE_CANDIDATE' and report.get('selection_sha256') == sha(run/'SELECTION.json')
            and report.get('stable_validation_passed') is False and report.get('test_passed') is False
            and report.get('calibration_inference_completed') is True and report.get('calibration_inference_passes') == 1
            and report.get('runtime_gates_changed') is False and report.get('test_read') is False
            and report.get('development_holdout_read') is False and report.get('selected_step') == 0
            and report.get('retention_checks_total') == report.get('retention_checks_passed') == 46,
            'Average result report changes candidate identity, inference count or evaluation scope')
    return selection


def averaging_identity(selection):
    keys = ('objective_variant', 'method', 'source_checkpoints', 'source_training_objectives',
        'state_sha256', 'parameter_groups_sha256', 'averaging_freeze_sha256', 'optimizer_steps_executed',
        'source_optimizer_steps_executed', 'source_selected_step', 'core_optimizer_steps_executed',
        'core_selected_step', 'calibration_device', 'candidate_step', 'selection_search_performed')
    return {key: copy.deepcopy(selection[key]) for key in keys}


def training_metadata(selection):
    return {**averaging_identity(selection), 'stage': 'post_training_parameter_average', 'selected_step': 0,
        'all_parameters_averaged': True, 'model_count': 1, 'encoder_count': 1,
        'teacher_in_deployed_model': False, 'teacher_cache_deployed': False,
        'output_averaging': False, 'inference_ensemble': False, 'ocr_text_used': False,
        'platform_routing': False, 'score_merging': False, 'development_holdout_is_blind_test': False}


def metadata_for_export(selection, cal, parent_metadata, model_sha256, checkpoint_sha256,
                        selection_sha256, runtime_record):
    require(selection['promotion_allowed'] is True and runtime_record['promotion_allowed'] is True,
            'A failed full-CNN candidate cannot be exported')
    require_fixed_runtime(runtime_record); measured = runtime_record['metrics']
    return {'schema': 'flux-glyph-unified-region-font-v1', 'algorithm': 'unified-region-cnn64x256-v1',
        'font_mode': 'unified', 'data_kind': 'native_mobile_screenshots', 'network_architecture': ARCHITECTURE,
        'model': {'path': 'model.onnx', 'sha256': model_sha256}, 'families': selection['families'],
        **copy.deepcopy(FIXED_RUNTIME), 'font_label_groups': cal['manifest']['font_label_groups'],
        'font_sources': copy.deepcopy(parent_metadata.get('font_sources', {})),
        'release_tier': 'experimental', 'stable_validation_passed': False, 'test_passed': False,
        'validation': {'kind': 'fixed_full_state_average_calibration_only_at_export',
            'calibration_passed': measured['passed'], 'retention_promotion_allowed': True,
            'retention_plan_sha256': selection['retention_plan']['sha256'],
            'fixed_runtime': copy.deepcopy(FIXED_RUNTIME), 'retention_checks': runtime_record['retention_checks'],
            'retention_populations': {name: {key: population[key] for key in
                ('views', 'named', 'correct_named', 'wrong_named', 'known_correct_coverage', 'named_precision', 'unknown_not_named_rate')}
                for name, population in runtime_record['retention_populations'].items()},
            'named_precision': measured['named_precision'], 'known_correct_coverage': measured['known_correct_coverage'],
            'unknown_not_named_rate': measured['unknown_not_named_rate'], 'original_stable_calibration_checks': measured['checks'],
            'model_count': 1, 'encoder_count': 1, 'teacher_in_deployed_model': False, 'platform_routing': False,
            'scores_are_correctness_probabilities': False, 'blind_test_performed': False, 'development_holdout_evaluated': False},
        'training': {**training_metadata(selection), 'selection_sha256': selection_sha256, 'checkpoint_sha256': checkpoint_sha256,
            'data_manifest_sha256': selection['data_manifest_sha256'], 'retention_plan_sha256': selection['retention_plan']['sha256'],
            'fixed_runtime': copy.deepcopy(FIXED_RUNTIME)}}


def export(args):
    import torch
    import onnx
    import onnxruntime as ort
    import average_unified_retention_source_balanced as averaging
    from region_network import RegionFontClassifier
    from export_region_stable import replace_groupnorm
    from prepare_unified_regions import load_split
    from train_unified_regions import region_outputs
    from flux_glyph.unified_font import UnifiedFontClassifier
    run, data, output = (Path(value).resolve() for value in (args.run, args.data, args.output))
    require(not output.exists() and not (run/'PARITY.json').exists(), 'Preserve earlier adapter export attempts')
    selection = validate(run, data); selection_sha = sha(run/'SELECTION.json'); checkpoint_sha = sha(run/'model.pth')
    sources = {str((ROOT/name).resolve()): sha(ROOT/name) for name in
        ('training/export_unified_retention_source_average.py', 'training/average_unified_retention_source_balanced.py',
         'training/export_unified_retention_source_balanced.py', 'training/train_unified_retention_source_balanced.py',
         'training/export_unified_retention_core.py', 'training/export_unified_regions.py',
         'training/export_region_stable.py', 'training/region_network.py',
         'src/flux_glyph/unified_font.py', 'src/flux_glyph/region_font.py')}
    evidence = dict(selection['bindings'])
    for name in ('SELECTION.json', 'AVERAGING_FREEZE.json', 'model.pth', 'CALIBRATION_OUTPUTS.npz',
                 'CALIBRATION_DECISIONS.json', 'RESULT.json', 'report.json'):
        evidence[str(run/name)] = sha(run/name)
    checkpoint = torch.load(run/'model.pth', map_location='cpu', weights_only=True)
    validate_checkpoint(checkpoint, selection)
    _, source_models = averaging.load_source_states(averaging.validate_sources(data))
    source_states = []
    for record, source in zip(selection['source_checkpoints'], source_models):
        require(source['families'] == selection['families'] and source['architecture'] == ARCHITECTURE
                and source.get('selection_sha256') == record['selection']['sha256']
                and state_sha(source['state_dict']) == record['state_sha256']
                and parameter_groups(source['state_dict']) == record['parameter_groups_sha256'],
                'Averaging source state or original parameter-group proof differs')
        source_states.append(source['state_dict'])
    require(verify_average_state(*source_states, checkpoint['state_dict']) == selection['state_sha256'],
            'Saved averaged state differs from its two complete source states')
    reference = RegionFontClassifier(25).cpu().eval(); reference.load_state_dict(checkpoint['state_dict'], strict=True)
    require(not any(name.startswith('residual_family_head.') for name in reference.state_dict())
            and sum(isinstance(module, torch.nn.GroupNorm) for module in reference.modules()) == 4,
            'Full-CNN reference must retain one original four-GroupNorm encoder and no residual head')
    converted = copy.deepcopy(reference)
    require(replace_groupnorm(converted, high_precision=True) == 4
            and state_sha(converted.state_dict()) == selection['state_sha256'], 'GroupNorm lowering changed adapter parameters')
    cal = load_split(data, 'calibration')
    plan = averaging.core.read_plan((ROOT/selection['retention_plan']['path']).resolve(), data, (ROOT/selection['parent_checkpoint']['path']).resolve())
    torch.set_num_threads(4); output.mkdir(parents=True); model_path = output/'model.onnx'
    torch.onnx.export(converted, torch.zeros(2, 1, 64, 256), model_path, input_names=['tiles'],
        output_names=['logits', 'log_em_ratio'], dynamic_axes={'tiles': {0: 'batch'}, 'logits': {0: 'batch'}, 'log_em_ratio': {0: 'batch'}},
        opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(model_path))
    options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
    actual_logits, actual_sizes, reference_logits, reference_sizes = [], [], [], []
    maximum = {'logits': 0., 'size': 0.}
    with torch.inference_mode():
        for start in range(0, len(cal['tiles']), 128):
            block = np.array(cal['tiles'][start:start+128], copy=True); stop = start+len(block)
            logits_tensor, sizes_tensor = reference(torch.from_numpy(block))
            logits, sizes = logits_tensor.numpy(), sizes_tensor.numpy()
            out, size = session.run(['logits', 'log_em_ratio'], {'tiles': block})
            for key, observed, expected in [('logits', out, logits), ('size', size, sizes)]:
                np.testing.assert_allclose(observed, expected, atol=2e-4, rtol=2e-4)
                maximum[key] = max(maximum[key], float(np.max(np.abs(observed-expected))))
            actual_logits.append(out); actual_sizes.append(size); reference_logits.append(logits); reference_sizes.append(sizes)
    actual_logits, actual_sizes = np.concatenate(actual_logits), np.concatenate(actual_sizes)
    reference_logits, reference_sizes = np.concatenate(reference_logits), np.concatenate(reference_sizes)
    with np.load(run/'CALIBRATION_OUTPUTS.npz', allow_pickle=False) as saved:
        np.testing.assert_allclose(reference_logits, saved['logits'], atol=3e-4, rtol=3e-4)
        np.testing.assert_allclose(reference_sizes, saved['log_em_ratio'], atol=3e-4, rtol=3e-4)
        cached_outputs = region_outputs(saved['logits'], saved['log_em_ratio'], cal['rows'], 1.)
    actual = region_outputs(actual_logits, actual_sizes, cal['rows'], 1.)
    expected = region_outputs(reference_logits, reference_sizes, cal['rows'], 1.)
    max_size_px = max(compare_runtime_outputs(actual, other, cal['rows'], selection['families']) for other in (expected, cached_outputs))
    measured, _, _ = evaluate_outputs(actual_logits, actual_sizes, cal, plan, selection['selected']['step'])
    require(measured['promotion_allowed'] is True and measured['metrics']['checks'] == selection['selected']['metrics']['checks']
            and measured['metrics']['passed'] is selection['passed']
            and all(measured['metrics'][key] == selection['selected']['metrics'][key]
                    for key in ('named', 'correct_named', 'wrong_named', 'unknown_wrongly_named')),
            'Actual complete ONNX changed retention or stable CAL acceptance')
    indices = np.unique(np.linspace(0, len(cal['tiles'])-1, min(256, len(cal['tiles']))).astype(int))
    samples = np.array(cal['tiles'][indices], copy=True); batches = []
    for count in (1, 7, 32, 128):
        values = [session.run(['logits', 'log_em_ratio'], {'tiles': samples[start:start+count]})
                  for start in range(0, len(samples), count)]
        out, size = np.concatenate([value[0] for value in values]), np.concatenate([value[1] for value in values])
        require(out.dtype == size.dtype == np.float32 and out.shape == (len(samples), 25) and size.shape == (len(samples),)
                and np.isfinite(out).all() and np.isfinite(size).all(), 'Invalid dynamic adapter output')
        np.testing.assert_allclose(out, reference_logits[indices], atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(size, reference_sizes[indices], atol=2e-4, rtol=2e-4)
        batches.append({'batch_size': count, 'samples': len(samples), 'passed': True, 'font_and_size_checked': True})
    parent_meta = averaging.read((ROOT/selection['parent_metadata']['path']).resolve())
    metadata = metadata_for_export(selection, cal, parent_meta, sha(model_path), checkpoint_sha, selection_sha, measured)
    dump(output/'metadata.json', metadata); UnifiedFontClassifier(output)
    require(validate(run, data) == selection and all(sha(path) == digest for path, digest in {**sources, **evidence}.items())
            and state_sha(reference.state_dict()) == state_sha(converted.state_dict()) == selection['state_sha256'],
            'Adapter evidence or weights changed during export')
    report = {'schema': SCHEMA, 'passed': True, 'promotion_allowed': True, 'font_model_count': 1, 'encoder_count': 1,
        'output_family_count': 25, 'network_architecture': ARCHITECTURE, 'selection_sha256': selection_sha,
        **averaging_identity(selection),
        'checkpoint_sha256': checkpoint_sha, 'model_sha256': sha(model_path), 'metadata_sha256': sha(output/'metadata.json'),
        'source_bindings': sources, 'calibration_bindings': evidence, 'fixed_runtime': FIXED_RUNTIME,
        'retention_plan_sha256': selection['retention_plan']['sha256'], 'runtime_gates_changed': False,
        'full_state_average_verified': True, 'all_parameters_averaged': True,
        'source_checkpoints_unchanged': True, 'export_parameters_unchanged': True,
        'original_torch_groupnorm_reference': True,
        'cached_average_outputs_reproduce_selected_metrics': True,
        'cached_full_model_runtime_parity_passed': True,
        'calibration_font_decisions_identical': True, 'calibration_runtime_reasons_identical': True,
        'calibration_size_availability_identical': True, 'calibration_size_values_close': True,
        'calibration_scores_close': True, 'max_absolute_errors': maximum, 'max_size_pixel_error': max_size_px,
        'calibration_regions': len(cal['rows']), 'calibration_tiles': len(cal['tiles']), 'batch_checks': batches,
        'frozen_retention_summary': strip_artifacts(selection['selected']), 'runtime_retention_summary': measured,
        'runtime_calibration_metrics': measured['metrics'], 'test_read': False, 'development_holdout_read': False,
        'user_images_read': False, 'training_cache_in_deployed_model': False,
        'teacher_in_deployed_model': False, 'teacher_cache_deployed': False,
        'output_averaging': False, 'inference_ensemble': False, 'platform_routing': False, 'score_merging': False}
    dump(run/'PARITY.json', report)
    print(json.dumps({'passed': True, 'promotion_allowed': True, 'model_sha256': report['model_sha256'],
                      'calibration_regions': len(cal['rows']), 'maximum_errors': maximum}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'data', 'output'):parser.add_argument('--'+name, type=Path, required=True)
    export(parser.parse_args())
