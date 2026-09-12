#!/usr/bin/env python3
"""CAL-only averaging of a failed, completed system-font fine-tune and R17.

This performs zero optimizer steps. It never opens test arrays or changes the
source run. The fixed alpha grid is considered once, with the original strict
parent-relative eligibility rules and unchanged runtime temperature/gates.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'training')]

from training.train_regions import (RegionData, dump, metrics, predict, region_observations,
                                    require, sha, state_sha)
from training.system_finetune import (MAX_SIZE_SPREAD, PARENT_GATES, PARENT_TEMPERATURE,
                                      SYSTEM_FAMILIES, assess_checkpoint, selection_stats)

ALPHAS = (.05, .1, .2, .35, .5, .75, 1.)
FORMULA = '(1-alpha)*R17_parent + alpha*last_trained; all floating tensors including size head; non-floating tensors must match exactly'
SOURCE_CODE = ('training/train_regions.py', 'training/system_finetune.py',
               'training/region_network.py', 'training/network.py', 'training/region_labels.py',
               'training/capture/generate_scenes.py', 'src/flux_glyph/region_font.py')


def read(path):
    return json.loads(Path(path).read_text())


def validate_source_run(source_run, data_dir):
    """Validate provenance without loading checkpoints or opening data arrays."""
    source_run, data_dir = Path(source_run).resolve(), Path(data_dir).resolve()
    hashes = {}

    def bind(name, path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        require(expected is None or actual == expected, 'averaging source SHA differs: ' + name)
        hashes[name] = {'path': str(path), 'sha256': actual}
        return path

    failure_path = bind('source_failure', source_run / 'NO_PROMOTABLE_CHECKPOINT.json')
    report = read(bind('source_report', source_run / 'report.json'))
    require(report == read(failure_path) and report.get('status') == 'NO_PROMOTABLE_CHECKPOINT',
            'averaging requires the unchanged failed source run report')
    require(report.get('test_arrays_opened') is False and 'test' not in report,
            'source run has inspected test arrays')
    require(not any((source_run / name).exists() for name in
                    ('SELECTION_FREEZE.json', 'region', 'best-calibration.pth')),
            'source run has a selected/exported checkpoint')
    require(report.get('optimizer_steps_executed') == 4000 and report.get('parameters_changed') is True
            and report.get('state_before_sha256') != report.get('state_after_sha256'),
            'source run must contain actual completed 4000-step parameter changes')
    freeze = read(bind('source_training_freeze', source_run / 'TRAINING_FREEZE.json', report['training_freeze_sha256']))
    require(freeze.get('schema') == 'flux-glyph-region-training-freeze-v1'
            and freeze.get('test_arrays_opened') is False and freeze.get('test_history') == 'reused',
            'source training freeze must precede test inspection')
    for key, value in {'seed': 2026091294, 'optimizer_steps_planned': 4000, 'batch_size': 64,
                       'evaluate_every_steps': 250}.items():
        require(freeze.get(key) == value, 'source training plan differs: ' + key)
    require(freeze['optimizer'] == {'name': 'AdamW', 'learning_rate': 3e-5, 'weight_decay': .0001},
            'source optimizer differs')
    focused = freeze['system_focused']
    require(focused.get('enabled') is True and focused.get('system_family_weight') == 2
            and focused.get('system_families') == list(SYSTEM_FAMILIES), 'source system training policy differs')
    families = freeze['families']
    require(len(families) == 8 and len(set(families)) == 8 and set(SYSTEM_FAMILIES).issubset(families),
            'all eight source classes must compete')
    expected_weights = {family: 2 if family in SYSTEM_FAMILIES else 1 for family in families}
    require(freeze['training_region_sampling']['family_weights'] == expected_weights
            and freeze['training_region_sampling']['system_family_weight'] == 2
            and report['training_region_sampling']['family_weights'] == expected_weights
            and sum(report['training_region_sampling']['family_visits'].values()) == 4000 * 64,
            'source sampling does not match the completed weighted training')
    policy = {'temperature': PARENT_TEMPERATURE, 'gates': dict(PARENT_GATES),
              'max_size_relative_spread': MAX_SIZE_SPREAD}
    require(focused['fixed_policy'] == policy and freeze['calibration'].get('recalibrate') is False
            and all(freeze['calibration'].get(key) == value for key, value in policy.items()),
            'source calibration policy changed')
    required_selection = {'source': 'calibration_only', 'global_wrong_must_not_increase': True,
                          'system_wrong_by_true_and_predicted_family_must_not_increase': True,
                          'nonsystem_correct_must_not_decrease': True, 'nonsystem_wrong_must_not_increase': True,
                          'preserve_all_parent_accepted_correct_rows': True, 'system_correct_must_increase': True,
                          'rank': ['system_correct', 'minimum_system_correct_coverage', 'overall_correct',
                                   'negative_mean_size_relative_error'], 'tie': 'earliest_checkpoint', 'recalibrate': False}
    require(focused['selection_policy'] == required_selection and freeze['selection'] == required_selection,
            'source checkpoint eligibility policy differs')
    parent_entry = focused['parent_metadata']
    parent_metadata = read(bind('parent_metadata', parent_entry['path'], parent_entry['sha256']))
    require(parent_metadata.get('schema') == 'flux-glyph-region-font-v1'
            and parent_metadata.get('algorithm') == 'region-cnn64x256-v1'
            and parent_metadata.get('families') == families and not parent_metadata.get('family_gates')
            and all(parent_metadata.get(key) == value for key, value in policy.items()),
            'parent runtime model or fixed policy differs')
    parent_state = parent_metadata['training']['state_after_sha256']
    require(report['state_before_sha256'] == freeze['initial_state_sha256'] == focused['parent_state_sha256'] == parent_state,
            'source parent trained-state lineage differs')
    require(freeze['warm_start']['preserved_size_head'] is True and not freeze['warm_start']['new_family_initializers']
            and freeze['warm_start']['families'] == families, 'source did not preserve the complete parent model')
    parent_checkpoint = bind('parent_checkpoint', freeze['warm_start']['path'], freeze['warm_start']['sha256'])
    last = report['last_checkpoint']
    require(last['path'] == 'last-trained.pth', 'source must use only its final trained checkpoint')
    trained_checkpoint = bind('last_checkpoint', source_run / last['path'], last['sha256'])
    require(trained_checkpoint != parent_checkpoint and hashes['last_checkpoint']['sha256'] != hashes['parent_checkpoint']['sha256'],
            'trained checkpoint cannot be the parent checkpoint')
    require(focused['parent_calibration'] == report['parent_calibration']
            and focused['parent_calibration']['path'] == 'PARENT_CALIBRATION.json', 'source parent CAL binding differs')
    baseline = read(bind('parent_calibration', source_run / 'PARENT_CALIBRATION.json', focused['parent_calibration']['sha256']))
    require(baseline.get('schema') == 'flux-glyph-parent-calibration-baseline-v1'
            and baseline.get('optimizer_steps_executed') == 0 and baseline.get('test_arrays_opened') is False
            and baseline.get('parent_state_sha256') == parent_state and baseline.get('parent_metadata') == parent_entry
            and baseline.get('fixed_policy') == policy, 'parent CAL baseline does not match the source model/policy')
    manifest = read(bind('data_manifest', data_dir / 'MANIFEST.json', freeze['data_manifest_sha256']))
    require(baseline['data_manifest_sha256'] == freeze['data_manifest_sha256']
            and manifest['families'] == families and baseline['calibration_partition'] == manifest['splits']['calibration'],
            'CAL source partition differs from the frozen baseline')
    code = freeze['code_sha256']
    require(set(code) == set(SOURCE_CODE), 'source training code inventory differs')
    for name, digest in code.items():
        bind('code:' + name, ROOT / name, digest)
        bind('frozen_code:' + name, source_run / 'frozen-code' / name, digest)
    require(manifest['preprocessing']['source_sha256'] == code['src/flux_glyph/region_font.py'],
            'prepared/runtime preprocessing identity differs')
    original_protocol = freeze.get('regression_protocol')
    require(isinstance(original_protocol, dict) and set(original_protocol) == {'path', 'sha256'},
            'source run has no bound regression protocol')
    bind('source_regression_protocol', original_protocol['path'], original_protocol['sha256'])
    history = read(bind('source_history', source_run / 'history.json', report['history_sha256']))
    require([record['step'] for record in history] == list(range(250, 4001, 250)), 'source calibration history is incomplete')
    for record in history:
        assessment = assess_checkpoint(record['selection_stats'], baseline['selection_stats'])
        require(assessment == record['eligibility'] and not assessment['eligible']
                and list(record['selection_score']) == assessment['rank'], 'source history contains an eligible or inconsistent checkpoint')
    return {'source_run': source_run, 'data_dir': data_dir, 'report': report, 'freeze': freeze,
            'baseline': baseline, 'parent_metadata': parent_metadata, 'families': families, 'policy': policy,
            'parent_checkpoint': parent_checkpoint, 'trained_checkpoint': trained_checkpoint,
            'hashes': hashes, 'manifest': manifest}


def validate_averaging_protocol(path, context):
    """Bind the independently predeclared plan before any CAL predictions."""
    path = Path(path).resolve()
    protocol = read(path)
    require(protocol.get('schema') == 'flux-glyph-system-averaging-fixed-regression-v1'
            and protocol.get('operation') == 'parameter_averaging'
            and protocol.get('alpha_grid') == list(ALPHAS) and protocol.get('alpha_zero_baseline_only') is True
            and protocol.get('formula') == FORMULA and protocol.get('fixed_policy') == context['policy']
            and protocol.get('system_families') == list(SYSTEM_FAMILIES)
            and protocol.get('additional_optimizer_steps_executed') == 0
            and protocol.get('source_optimizer_steps_executed') == 4000, 'independent averaging plan differs')
    require(protocol.get('source_preconditions') == {'source_status': 'NO_PROMOTABLE_CHECKPOINT',
            'source_optimizer_steps_executed': 4000, 'source_test_arrays_opened': False,
            'source_parameters_changed': True}, 'independent source preconditions differ')
    require(protocol.get('selection') == {'source': 'calibration_only', 'same_eligibility_as_system_finetuning': True,
            'preserve_all_parent_accepted_correct_rows': True, 'rank': context['freeze']['selection']['rank'],
            'tie': 'earliest_alpha_in_fixed_grid', 'only_last_training_endpoint': True,
            'single_selected_model': True, 'no_grid_refinement': True}, 'independent selection policy differs')
    names = {'parent_checkpoint', 'parent_metadata', 'source_failure', 'source_training_freeze',
             'source_history', 'last_checkpoint', 'data_manifest'}
    require(set(protocol.get('source_bindings', {})) == names
            and all(protocol['source_bindings'][name] == context['hashes'][name] for name in names),
            'independent averaging source bindings differ')
    return {'path': str(path), 'sha256': sha(path)}


def blend_states(parent, trained, alpha):
    """Use every matching floating tensor; alpha zero is only a baseline check."""
    import torch
    require(type(alpha) in (int, float) and math.isfinite(alpha) and (alpha == 0 or alpha in ALPHAS),
            'alpha must belong to the predeclared fixed grid or baseline zero')
    require(parent and set(parent) == set(trained), 'averaging checkpoint tensor names differ')
    result = {}
    for key, value in parent.items():
        other = trained[key]
        require(isinstance(value, torch.Tensor) and isinstance(other, torch.Tensor)
                and value.dtype == other.dtype and value.shape == other.shape,
                'averaging requires matching tensors: ' + key)
        left, right = value.detach().cpu(), other.detach().cpu()
        if not torch.is_floating_point(left):
            require(not torch.is_complex(left) and torch.equal(left, right), 'non-floating averaging tensors must match exactly: ' + key)
            result[key] = left.clone()
            continue
        require(left.dtype == torch.float32 and bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()),
                'averaging requires finite float32 parameters: ' + key)
        result[key] = left.clone() if alpha == 0 else right.clone() if alpha == 1 else (1 - alpha) * left + alpha * right
        require(bool(torch.isfinite(result[key]).all()), 'averaged tensor is nonfinite: ' + key)
    return result


def verify_parent_stats(actual, expected):
    """Counts/identities must be exact; allow only floating reduction roundoff."""
    without_size = lambda item: {key: value for key, value in item.items() if key != 'size_mean_relative_error'}
    require(without_size(actual) == without_size(expected)
            and math.isclose(actual['size_mean_relative_error'], expected['size_mean_relative_error'], rel_tol=1e-6, abs_tol=1e-8),
            'alpha-zero CAL does not reproduce the recorded parent baseline')


def choose_candidate(records):
    """Strict rank replacement preserves the smaller alpha when ranks tie."""
    require([row['alpha'] for row in records] == list(ALPHAS), 'all fixed nonzero alphas must be evaluated exactly once in order')
    best = None
    for record in records:
        if record['eligibility']['eligible'] and (best is None or record['eligibility']['rank'] > best['eligibility']['rank']):
            best = record
    return best


def average(args):
    import torch
    from region_network import RegionFontClassifier

    context = validate_source_run(args.source_run, args.data)
    output = Path(args.output).resolve()
    require(not output.exists(), 'averaging output must be a new directory')
    require(not output.is_relative_to(context['source_run']) and not output.is_relative_to(context['data_dir']),
            'averaging output must be independent of the source run and prepared data')
    independent_protocol = validate_averaging_protocol(args.regression_protocol or output.parent / 'REGRESSION_PROTOCOL.json', context)
    device = args.device
    require(device in ('mps', 'cpu') and (device != 'mps' or torch.backends.mps.is_available()), 'requested averaging device unavailable')
    torch.set_num_threads(4)
    torch.manual_seed(context['freeze']['seed'])
    parent = torch.load(context['parent_checkpoint'], map_location='cpu', weights_only=True)
    trained = torch.load(context['trained_checkpoint'], map_location='cpu', weights_only=True)
    require(parent['families'] == trained['families'] == context['families'] and trained['optimizer_steps'] == 4000,
            'checkpoint class order/source optimizer count differs')
    require(state_sha(parent['state_dict']) == context['report']['state_before_sha256']
            and state_sha(trained['state_dict']) == context['report']['state_after_sha256'],
            'actual checkpoint tensor hashes differ from completed training evidence')
    baseline_state = blend_states(parent['state_dict'], trained['state_dict'], 0)
    net = RegionFontClassifier(len(context['families']))
    net.load_state_dict(baseline_state, strict=True)
    net.to(device)
    data = RegionData(context['data_dir'])
    require(data.manifest_sha == context['freeze']['data_manifest_sha256'], 'prepared manifest changed before averaging')
    calibration = data.load('calibration')
    policy, families = context['policy'], context['families']
    code = {name: sha(ROOT / name) for name in SOURCE_CODE}
    code['training/average_system_finetune.py'] = sha(Path(__file__))
    bindings = copy.deepcopy(context['hashes'])
    bindings['averaging_regression_protocol'] = independent_protocol
    derivation = {'operation': 'parameter_averaging', 'formula': FORMULA,
                  'source_run': str(context['source_run']), 'source_optimizer_steps_executed': 4000,
                  'additional_optimizer_steps_executed': 0, 'source_bindings': bindings}
    protocol = {'schema': 'flux-glyph-system-averaging-protocol-v1', 'status_at_creation': 'before_any_averaging_cal_prediction',
                'operation': 'parameter_averaging', 'alpha_grid': list(ALPHAS), 'alpha_zero_baseline_only': True,
                'formula': FORMULA, 'fixed_policy': policy, 'eligibility': context['freeze']['selection'],
                'tie': 'smaller_alpha', 'no_grid_refinement': True, 'no_recalibration': True,
                'test_arrays_opened': False, 'derivation': derivation, 'code_sha256': code}
    output.mkdir(parents=True)
    dump(output / 'AVERAGING_PROTOCOL.json', protocol)
    freeze = {'schema': 'flux-glyph-region-averaging-freeze-v1', 'operation': 'parameter_averaging',
              'families': families, 'device': device, 'torch': torch.__version__,
              'optimizer_steps_executed': 0, 'source_optimizer_steps_executed': 4000,
              'alpha_grid': list(ALPHAS), 'fixed_policy': policy, 'selection': protocol['eligibility'],
              'tie': 'smaller_alpha', 'source_bindings': bindings, 'code_sha256': code,
              'data_manifest_sha256': data.manifest_sha, 'source_kind': 'ios_simulator_screenshot',
              'test_history': 'reused', 'test_arrays_opened': False,
              'protocol_sha256': sha(output / 'AVERAGING_PROTOCOL.json'), 'derivation': derivation}
    dump(output / 'AVERAGING_FREEZE.json', freeze)
    dump(output / 'TRAINING_FREEZE.json', freeze)
    for name in code:
        destination = output / 'frozen-code' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    obs = region_observations(*predict(net, calibration, device), calibration, policy['temperature'])
    parent_stats = selection_stats(obs, calibration, families, policy['gates'])
    verify_parent_stats(parent_stats, context['baseline']['selection_stats'])
    baseline = {'alpha': 0, 'candidate': False, 'state_sha256': state_sha(baseline_state),
                'selection_stats': parent_stats, 'metrics': metrics(obs, calibration, families, policy['gates']),
                'source_baseline_sha256': bindings['parent_calibration']['sha256'], 'test_arrays_opened': False}
    dump(output / 'PARENT_CALIBRATION.json', baseline)
    history = []
    for alpha in ALPHAS:
        state = blend_states(parent['state_dict'], trained['state_dict'], alpha)
        net.load_state_dict(state, strict=True)
        obs = region_observations(*predict(net, calibration, device), calibration, policy['temperature'])
        stats = selection_stats(obs, calibration, families, policy['gates'])
        record = {'alpha': alpha, 'state_sha256': state_sha(state),
                  'optimizer_steps_executed': 0, 'source_optimizer_steps_executed': 4000,
                  'selection_stats': stats, 'eligibility': assess_checkpoint(stats, parent_stats),
                  'metrics': metrics(obs, calibration, families, policy['gates']), 'test_arrays_opened': False}
        history.append(record)
        dump(output / 'history.json', history)
        print(json.dumps(record), flush=True)
    best = choose_candidate(history)
    require(set(data.loaded) == {'calibration'}, 'averaging must open only the calibration partition')
    require(all(sha(ROOT / name) == digest for name, digest in code.items()), 'averaging/runtime source changed during the frozen run')
    require(all(sha(entry['path']) == entry['sha256'] for entry in bindings.values()), 'averaging source provenance changed during the frozen run')
    report = {'schema': 'flux-glyph-system-averaging-report-v1', 'operation': 'parameter_averaging',
              'optimizer_steps_executed': 0, 'source_optimizer_steps_executed': 4000,
              'alpha_grid': list(ALPHAS), 'fixed_policy': policy, 'derivation': derivation,
              'averaging_freeze_sha256': sha(output / 'AVERAGING_FREEZE.json'),
              'history_sha256': sha(output / 'history.json'), 'test_arrays_opened': False, 'test_history': 'reused',
              'additional_training_performed': False, 'independent_parity_and_regression_required': True}
    if best is None:
        report['status'] = 'NO_PROMOTABLE_CHECKPOINT'
        dump(output / 'NO_PROMOTABLE_CHECKPOINT.json', report)
        dump(output / 'report.json', report)
        print(json.dumps({'finished': True, 'status': report['status'], 'output': str(output)}), flush=True)
        return
    selected_state = blend_states(parent['state_dict'], trained['state_dict'], best['alpha'])
    require(state_sha(selected_state) == best['state_sha256'] != context['report']['state_before_sha256'],
            'selected average differs from the evaluated state or equals the parent')
    neural = output / 'region'
    neural.mkdir()
    selected = {'state_dict': selected_state, 'families': families, 'optimizer_steps': 0,
                'source_optimizer_steps': 4000, 'alpha': best['alpha'], 'operation': 'parameter_averaging'}
    torch.save(selected, neural / 'model.pth')
    selected_entry = {'step': 0, 'alpha': best['alpha'], 'score': best['eligibility']['rank'],
                      'state_sha256': best['state_sha256'], 'checkpoint_sha256': sha(neural / 'model.pth')}
    chosen_derivation = {**derivation, 'selected_alpha': best['alpha'], 'state_sha256': best['state_sha256']}
    selection = {'schema': 'flux-glyph-region-selection-before-test-v1', 'operation': 'parameter_averaging',
                 'selected': selected_entry, 'optimizer_steps_executed': 0, 'source_optimizer_steps_executed': 4000,
                 'state_before_sha256': context['report']['state_before_sha256'], 'state_after_sha256': best['state_sha256'],
                 **policy, 'calibration': {'selection_stats': best['selection_stats'], 'metrics': best['metrics'],
                                         'eligibility': best['eligibility'], 'recalibrated': False},
                 'parent_calibration_sha256': sha(output / 'PARENT_CALIBRATION.json'),
                 'averaging_freeze_sha256': sha(output / 'AVERAGING_FREEZE.json'),
                 'history_sha256': sha(output / 'history.json'), 'derivation': chosen_derivation,
                 'test_arrays_opened': False, 'test_history': 'reused'}
    dump(output / 'SELECTION_FREEZE.json', selection)
    net.cpu().load_state_dict(selected_state, strict=True)
    net.eval()
    torch.onnx.export(net, torch.zeros(2, 1, 64, 256), neural / 'model.onnx', input_names=['tiles'],
                      output_names=['logits', 'log_em_ratio'], dynamic_axes={'tiles': {0: 'batch'},
                      'logits': {0: 'batch'}, 'log_em_ratio': {0: 'batch'}}, opset_version=17, dynamo=False)
    meta = copy.deepcopy(context['parent_metadata'])
    for key in ('export', 'validation', 'calibration', 'training', 'derivation'):
        meta.pop(key, None)
    meta['model'] = {'path': 'model.onnx', 'sha256': sha(neural / 'model.onnx')}
    meta['training'] = {'operation': 'parameter_averaging', 'optimizer_steps_executed': 0,
                        'source_optimizer_steps_executed': 4000, 'selected_step': 0,
                        'data_manifest_sha256': data.manifest_sha, 'source_kind': 'ios_simulator_screenshot',
                        'state_after_sha256': best['state_sha256'], 'selection_sha256': sha(output / 'SELECTION_FREEZE.json')}
    meta['derivation'] = chosen_derivation
    meta['calibration'] = {'gate_found': True, 'recalibrated': False, 'source': 'parent_metadata',
                           'parent_metadata_sha256': bindings['parent_metadata']['sha256'],
                           'selection_source': 'calibration_only_fixed_alpha_grid'}
    meta['release_status'] = 'experimental'
    meta['scope'] = 'CAL-selected parameter average; zero additional optimizer steps. Reused regression tests have not been opened by this run.'
    dump(neural / 'metadata.json', meta)
    report.update(status='CAL_ELIGIBLE_AVERAGE', selected=selected_entry, selection_sha256=sha(output / 'SELECTION_FREEZE.json'),
                  model_sha256=meta['model']['sha256'], derivation=chosen_derivation)
    dump(output / 'report.json', report)
    print(json.dumps({'finished': True, 'status': report['status'], 'alpha': best['alpha'], 'output': str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    parser.add_argument('--regression-protocol', type=Path, help='Read-only predeclared independent regression protocol to bind')
    average(parser.parse_args())


if __name__ == '__main__':
    main()
