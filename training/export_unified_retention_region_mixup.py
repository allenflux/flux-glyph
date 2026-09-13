#!/usr/bin/env python3
"""Export a region-supervised manifold-mixup residual CNN after complete CAL parity."""
from __future__ import annotations

import copy
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'training')]
from train_regions import require, sha, dump, state_sha
from train_unified_retention import FIXED_RUNTIME, evaluate_outputs, retention_rank
from export_unified_retention_core import (compare_runtime_outputs, require_fixed_runtime,
    calibration_metadata, checked_file, cached_evaluation, strip_artifacts)

ARCHITECTURE = 'region-cnn64x256-residual-head-v1'
SCHEMA = 'flux-glyph-unified-retention-onnx-parity-v1'
BASE_GROUPS = ('trunk', 'pool', 'style', 'family_head', 'size_head')
RESIDUAL_PREFIX = 'residual_family_head.'


def source_entry(entry, bindings):
    require(isinstance(entry, dict) and set(entry) == {'path', 'sha256'}, 'Incomplete adapter source identity')
    path = (ROOT/entry['path']).resolve()
    require(path.is_file() and sha(path) == entry['sha256'] == bindings.get(str(path)), 'Adapter source binding differs')
    return path


def validate_checkpoint(checkpoint, selection, selection_sha256):
    state = checkpoint['state_dict']
    base = {key: value for key, value in state.items() if not key.startswith(RESIDUAL_PREFIX)}
    residual = {key: value for key, value in state.items() if key.startswith(RESIDUAL_PREFIX)}
    require(checkpoint.get('families') == selection['families'] and checkpoint.get('architecture') == ARCHITECTURE
            and checkpoint.get('selection_sha256') == selection_sha256
            and state_sha(state) == selection['state_after_sha256']
            and state_sha(base) == selection['base_state_sha256'] == selection['base_state_after_sha256']
            and set(residual) == {RESIDUAL_PREFIX+name for name in ('0.weight', '0.bias', '2.weight', '2.bias')}
            and state_sha(residual) == selection['residual_state_after_sha256']
            != selection['residual_state_before_sha256'], 'Adapter changed the frozen base or selected residual identity')


def validate(run, data):
    """Read-only source/CAL-cache validation; imports no Torch and reads no holdout."""
    import train_unified_retention_region_mixup as trainer
    run, data = Path(run).resolve(), Path(data).resolve()
    selection = trainer.read(run/'SELECTION.json')
    require(selection.get('schema') == 'flux-glyph-unified-retention-region-mixup-selection-v1'
            and selection.get('architecture') == ARCHITECTURE and selection.get('promotion_allowed') is True
            and not (run/'NO_PROMOTABLE_CHECKPOINT.json').exists(), 'Adapter has no promotable frozen checkpoint')
    protocol = trainer.read(run/'TRAINING_FREEZE.json')
    require(protocol.get('schema') == 'flux-glyph-unified-retention-region-mixup-protocol-v1'
            and sha(run/'TRAINING_FREEZE.json') == selection['training_protocol_sha256']
            and all(selection.get(key) == value for key, value in protocol.items() if key != 'schema'),
            'Adapter protocol and selection differ')
    expected = {'architecture': ARCHITECTURE, 'objective_variant': trainer.OBJECTIVE_VARIANT,
        'objective': trainer.OBJECTIVE, 'sampling': trainer.SAMPLING, 'fixed_runtime': FIXED_RUNTIME,
        'region_tile_training': 'All tiles of each sampled region; mean-softmax supervised CE and known-only base KL.',
        'manifold_mixup': trainer.MIXUP, 'feature_cache_reused': True, 'cached_feature_extraction_performed': False,
        'steps': trainer.STEPS, 'eval_every': trainer.EVAL_EVERY, 'batch_size': trainer.BATCH_SIZE,
        'learning_rate': trainer.LEARNING_RATE, 'minimum_learning_rate': trainer.MINIMUM_LEARNING_RATE,
        'weight_decay': 1e-4, 'gradient_clip_norm': 5., 'training_device': 'cpu', 'seed': 2026091406,
        'base_selected_step': 1000, 'source_optimizer_steps_executed': 1500, 'base_optimizer_steps': 0,
        'residual_optimizer_steps': 2000, 'optimizer_steps_executed': 2000,
        'frozen_groups': ['trunk', 'pool', 'style', 'family_head', 'size_head'],
        'trainable_groups': ['residual_family_head'], 'all_parameters_trained': False, 'base_frozen': True,
        'size_head_frozen': True, 'residual_head_trained': True, 'model_count': 1, 'encoder_count': 1,
        'platform_routing': False, 'score_merging': False, 'external_teacher': False,
        'test_read': False, 'development_holdout_read': False, 'runtime_gates_searched': False,
        'training_inputs': ['image_features_only'],
        'calibration_execution': 'Frozen MPS/CPU base logits + CPU residual(features); copied frozen size logits.'}
    require(all(protocol.get(key) == value for key, value in expected.items())
            and protocol.get('feature_device') in ('cpu', 'mps') and type(selection.get('passed')) is bool
            and selection['calibration_passed'] is selection['passed'], 'Adapter training or single-encoder contract differs')
    bindings = selection['bindings']; trainer.core.verify_bindings(bindings)
    base_path = source_entry(selection['base_checkpoint'], bindings)
    base_selection_path = source_entry(selection['base_selection'], bindings)
    require(sha(base_path) == trainer.BASE_CHECKPOINT_SHA and sha(base_selection_path) == trainer.BASE_SELECTION_SHA
            and base_selection_path == base_path.parent/'SELECTION.json', 'Wrong frozen core initializer')
    base = trainer.read(base_selection_path)
    require(base.get('schema') == 'flux-glyph-unified-retention-selection-v1'
            and base['selected']['step'] == 1000 and base['optimizer_steps_executed'] == 1500
            and base['promotion_allowed'] is False and base['test_read'] is False and base['development_holdout_read'] is False
            and base['state_after_sha256'] == selection['base_state_sha256'] == selection['base_state_after_sha256']
            and base['families'] == selection['families'] and len(selection['families']) == 25
            and len(set(selection['families'])) == 25 and selection['families'].count('__unknown__') == 1
            and all(bindings.get(path) == digest for path, digest in base['bindings'].items()),
            'Adapter base state, classes or source closure differ')
    require_fixed_runtime(base['selected'])
    plan_path = source_entry(selection['retention_plan'], bindings)
    require(base['retention_plan']['sha256'] == sha(plan_path), 'Adapter retention conditions changed')
    plan = trainer.core.read_plan(plan_path, data, (ROOT/base['parent_checkpoint']['path']).resolve())
    cal = calibration_metadata(data)
    require(cal['manifest_sha256'] == selection['data_manifest_sha256'] == base['data_manifest_sha256']
            and cal['families'] == selection['families'] == plan['families'], 'Adapter data root or class order differs')
    for key in ('parent_metadata', 'parent_checkpoint', 'parent_selection_sha256'):
        require(selection[key] == base[key], 'Adapter parent provenance differs')
    parent_metadata = source_entry(selection['parent_metadata'], bindings)
    required = [ROOT/name for name in ('training/train_unified_retention_region_mixup.py', 'training/retention_region_mixup_loss.py',
        'training/train_unified_retention_adapter.py', 'training/retention_adapter_network.py',
        'training/train_unified_retention_core.py', 'training/train_unified_retention.py', 'training/train_unified_regions.py',
        'training/prepare_unified_regions.py', 'training/region_network.py', 'training/network.py', 'training/train_regions.py',
        'training/evaluate_unified_retention_region_mixup.py', 'src/flux_glyph/unified_font.py', 'src/flux_glyph/region_font.py')]
    required += [data/'MANIFEST.json', base_path.parent/'TRAINING_FREEZE.json',
                 base_path.parent/'CALIBRATION_OUTPUTS.npz', base_path.parent/'CALIBRATION_DECISIONS.json', parent_metadata]
    source_partitions = {}
    for split in ('train', 'calibration'):
        part_path = data/split/'MANIFEST.json'; part = trainer.read(part_path); required.append(part_path)
        require(part['split'] == split and part['families'] == selection['families']
                and part['root_manifest_sha256'] == cal['manifest_sha256'], 'Adapter source partition differs')
        for name in ('metadata', 'array'):
            path = (data/split/part[name]['path']).resolve()
            require(path.parent == data/split and bindings.get(str(path)) == part[name]['sha256'], 'Adapter requested data path is unbound')
            required.append(path)
        source_partitions[split] = {'partition_sha256': sha(part_path), 'tile_count': part['tiles'],
            'rows': part['metadata'], 'tiles': part['array'],
            'order': 'Exact prepared tile array order; rows retain tile_start and tile_count.'}
    require(all(bindings.get(str(path.resolve())) == sha(path) for path in required), 'Missing exact adapter source binding')
    cache_path = source_entry(selection['cache_manifest'], bindings)
    require(sha(cache_path) == trainer.CACHE_MANIFEST_SHA, 'Region mixup did not reuse the fixed base feature cache')
    cache_manifest = trainer.read(cache_path)
    identity = cache_manifest['identity']
    require(identity['architecture'] == ARCHITECTURE and identity['base_checkpoint'] == selection['base_checkpoint']
            and identity['base_state_sha256'] == selection['base_state_sha256'] and identity['families'] == selection['families']
            and identity['data_manifest_sha256'] == selection['data_manifest_sha256']
            and identity['partitions'] == source_partitions and identity['test_read'] is False
            and identity['development_holdout_read'] is False
            and all(bindings.get(path) == digest for path, digest in identity['bindings'].items()), 'Frozen feature cache input identity differs')
    cache, cache_manifest = trainer.load_cache(cache_path.parent, identity)
    freeze = trainer.read(cache_path.parent/'CACHE_FREEZE.json')
    require(freeze['schema'] == 'flux-glyph-retention-adapter-cache-freeze-v1' and freeze['optimizer_steps_executed'] == 0
            and freeze['feature_device'] == protocol['feature_device']
            and bindings.get(str(cache_path.parent/'CACHE_FREEZE.json')) == sha(cache_path.parent/'CACHE_FREEZE.json'),
            'Feature extraction provenance differs')
    for partition in cache_manifest['partitions'].values():
        for entry in partition.values():source_entry({'path': str(cache_path.parent/entry['path']), 'sha256': entry['sha256']}, bindings)
    initial_path = source_entry(selection['initial_residual'], bindings)
    require(initial_path == run/'INITIAL_RESIDUAL.pth' and selection['initial_state_sha256'] != selection['state_after_sha256']
            and selection['residual_state_before_sha256'] != selection['residual_state_after_sha256'], 'Residual head was not trained')
    baseline, outputs, _ = evaluate_outputs(cache['calibration']['base_logits'], cache['calibration']['log_em_ratio'], cal, plan, 0)
    require(set(selection['baseline_bindings']) == {'BASELINE.json', 'BASELINE_CALIBRATION_DECISIONS.json'}
            and all(sha(run/name) == digest for name, digest in selection['baseline_bindings'].items())
            and trainer.read(run/'BASELINE.json') == baseline
            and trainer.read(run/'BASELINE_CALIBRATION_DECISIONS.json') == {'families': selection['families'], 'records': outputs},
            'Adapter baseline does not reproduce the frozen base cache')
    history = selection['history']
    require([row['step'] for row in history] == list(range(200, 2001, 200)), 'Incomplete adapter checkpoint history')
    for record in history:
        directory = f"checkpoints/step{record['step']:05d}"
        require(set(record['artifacts']) == {'checkpoint', 'outputs', 'decisions', 'metrics'}, 'Incomplete adapter step evidence')
        paths = {key: checked_file(run, record['artifacts'][key], directory+'/'+name) for key, name in
            [('checkpoint', 'model.pth'), ('outputs', 'CALIBRATION_OUTPUTS.npz'),
             ('decisions', 'CALIBRATION_DECISIONS.json'), ('metrics', 'METRICS.json')]}
        actual, _, _ = cached_evaluation(paths['outputs'], paths['decisions'], cal, plan, record['step'])
        require(actual == strip_artifacts(record) == trainer.read(paths['metrics']), 'Adapter CAL metrics do not reproduce cached outputs')
        with np.load(paths['outputs'], allow_pickle=False) as saved:
            require(np.array_equal(saved['log_em_ratio'], cache['calibration']['log_em_ratio']), 'Adapter altered frozen size predictions')
    require(selection['selected'] == max(history, key=retention_rank) and selection['selected']['promotion_allowed'] is True
            and selection['selected']['metrics']['passed'] is selection['passed'], 'Adapter selected a different CAL candidate')
    for key, name, field in [('outputs', 'CALIBRATION_OUTPUTS.npz', 'calibration_outputs_sha256'),
                            ('decisions', 'CALIBRATION_DECISIONS.json', 'calibration_decisions_sha256')]:
        require(sha(run/name) == selection[field] == selection['selected']['artifacts'][key]['sha256'], 'Selected adapter CAL bytes differ')
    validate_counts(run, selection)
    return selection


def validate_counts(run, selection):
    from collections import Counter
    import train_unified_retention_region_mixup as trainer
    require(sha(run/'SAMPLING.json') == selection['sampling_sha256']
            and sha(run/'TRAINING_COUNTS.json') == selection['training_counts_sha256'], 'Adapter sampling counts changed')
    sampling = trainer.read(run/'SAMPLING.json'); counts = trainer.read(run/'TRAINING_COUNTS.json')
    trainer.validate_mixup_counts(counts, trainer.STEPS)
    slots = {key: trainer.SAMPLING[key]*2000 for key in
             ('base_known', 'base_unknown', 'ios_native_pingfang', 'ios_native_sfpro_helvetica', 'ios_native_original_eight')}
    require(sampling['slots'] == slots and sampling['base']['known_rows'] == 96000 and sampling['base']['unknown_rows'] == 32000
            and counts['schema'] == 'flux-glyph-retention-region-mixup-counts-v1' and counts['steps'] == 2000
            and counts['rows'] == 192000 and counts['objective'] == trainer.OBJECTIVE
            and counts['test_read'] is False and counts['development_holdout_read'] is False, 'Adapter supervised counts differ')
    family, domain, weighted, eligible, seen = Counter(), Counter(), 0, 0, set()
    for row in counts['source_target_rows']:
        key = (row['domain'], row['source_font_family'], row['target_family'])
        require(key not in seen and key[0] in ('ios', 'android') and key[2] in selection['families']
                and isinstance(key[1], str) and key[1]
                and all(type(row[name]) is int and 0 <= row[name] <= row['rows'] for name in ('rows', 'weight_two_rows', 'base_kl_rows')),
                'Invalid adapter source accounting')
        if key[2] in (*trainer.core.CORE_FAMILIES, '__unknown__'):
            require(row['base_kl_rows'] == 0, 'Unknown or core family received a base KL target')
        if key[2] == '__unknown__':require(row['weight_two_rows'] == row['rows'], 'Unknown supervised weighting differs')
        elif key[2] not in trainer.core.CORE_FAMILIES:require(row['weight_two_rows'] == 0, 'Noncore known class received native-core weighting')
        elif key[0] != 'ios':require(row['weight_two_rows'] == 0, 'Android region received iOS native weighting')
        seen.add(key); family[key[2]] += row['rows']; domain[key[0]] += row['rows']
        weighted += row['weight_two_rows']; eligible += row['base_kl_rows']
    require(dict(family) == sampling['family_rows'] and dict(domain) == sampling['domain_rows']
            and sum(family.values()) == sum(sampling['view_rows'].values()) == 192000
            and weighted == counts['weight_two_rows'] and eligible == counts['base_kl_rows']
            and 86000 <= weighted <= 98000, 'Adapter training/source aggregates differ')
    pairs = counts['mixup_pairs_by_unknown_count']
    require(pairs.get('1', 0) + 2*pairs.get('2', 0) == 2*family['__unknown__'],
            'Mixup permutation did not preserve both copies of the real unknown population')

def metadata_for_export(selection, cal, parent_metadata, model_sha256, checkpoint_sha256,
                        selection_sha256, runtime_record):
    require(selection['promotion_allowed'] is True and runtime_record['promotion_allowed'] is True,
            'A failed residual-head candidate cannot be exported')
    require_fixed_runtime(runtime_record)
    measured = runtime_record['metrics']
    return {'schema': 'flux-glyph-unified-region-font-v1', 'algorithm': 'unified-region-cnn64x256-v1',
        'font_mode': 'unified', 'data_kind': 'native_mobile_screenshots', 'network_architecture': ARCHITECTURE,
        'model': {'path': 'model.onnx', 'sha256': model_sha256}, 'families': selection['families'],
        **copy.deepcopy(FIXED_RUNTIME), 'font_label_groups': cal['manifest']['font_label_groups'],
        'font_sources': copy.deepcopy(parent_metadata.get('font_sources', {})),
        'release_tier': 'experimental', 'stable_validation_passed': False, 'test_passed': False,
        'validation': {'kind': 'frozen_encoder_region_mixup_retention_calibration_only_at_export',
            'calibration_passed': measured['passed'], 'retention_promotion_allowed': True,
            'retention_plan_sha256': selection['retention_plan']['sha256'],
            'fixed_runtime': copy.deepcopy(FIXED_RUNTIME), 'retention_checks': runtime_record['retention_checks'],
            'retention_populations': {name: {key: population[key] for key in
                ('views', 'named', 'correct_named', 'wrong_named', 'known_correct_coverage',
                 'named_precision', 'unknown_not_named_rate')}
                for name, population in runtime_record['retention_populations'].items()},
            'named_precision': measured['named_precision'], 'known_correct_coverage': measured['known_correct_coverage'],
            'unknown_not_named_rate': measured['unknown_not_named_rate'],
            'original_stable_calibration_checks': measured['checks'],
            'model_count': 1, 'encoder_count': 1, 'platform_routing': False,
            'scores_are_correctness_probabilities': False, 'blind_test_performed': False,
            'development_holdout_evaluated': False},
        'training': {'selection_sha256': selection_sha256, 'checkpoint_sha256': checkpoint_sha256,
            'objective_variant': selection['objective_variant'], 'objective': copy.deepcopy(selection['objective']),
            'region_tile_training': selection['region_tile_training'], 'manifold_mixup': copy.deepcopy(selection['manifold_mixup']),
            'feature_cache_reused': True, 'synthetic_unknown_labels': False, 'mixup_deployed': False,
            'training_counts_sha256': selection['training_counts_sha256'],
            'data_manifest_sha256': selection['data_manifest_sha256'],
            'retention_plan_sha256': selection['retention_plan']['sha256'], 'fixed_runtime': copy.deepcopy(FIXED_RUNTIME),
            'optimizer_steps_executed': selection['optimizer_steps_executed'],
            'base_optimizer_steps': 0, 'residual_optimizer_steps': selection['residual_optimizer_steps'],
            'source_optimizer_steps_executed': selection['source_optimizer_steps_executed'],
            'selected_step': selection['selected']['step'], 'base_checkpoint': copy.deepcopy(selection['base_checkpoint']),
            'base_selection': copy.deepcopy(selection['base_selection']), 'base_state_sha256': selection['base_state_sha256'],
            'cache_manifest': copy.deepcopy(selection['cache_manifest']),
            'residual_state_before_sha256': selection['residual_state_before_sha256'],
            'residual_state_after_sha256': selection['residual_state_after_sha256'],
            'all_parameters_trained': False, 'base_frozen': True, 'size_head_frozen': True,
            'raw_size_branch_frozen': True, 'residual_head_trained': True,
            'trainable_parameter_groups': ['residual_family_head'], 'frozen_parameter_groups': list(BASE_GROUPS),
            'training_inputs': selection['training_inputs'], 'model_count': 1, 'encoder_count': 1,
            'teacher_outputs_used': False, 'external_teacher': False, 'frozen_base_logits_used_for_training': True,
            'platform_routing': False, 'score_merging': False, 'ocr_text_used': False,
            'development_holdout_is_blind_test': False}}


def export(args):
    import torch
    import onnx
    import onnxruntime as ort
    import train_unified_retention_region_mixup as trainer
    from retention_adapter_network import RetentionAdapterClassifier
    from export_region_stable import replace_groupnorm
    from prepare_unified_regions import load_split
    from train_unified_regions import region_outputs
    from flux_glyph.unified_font import UnifiedFontClassifier
    run, data, output = (Path(value).resolve() for value in (args.run, args.data, args.output))
    require(not output.exists() and not (run/'PARITY.json').exists(), 'Preserve earlier adapter export attempts')
    selection = validate(run, data); selection_sha = sha(run/'SELECTION.json'); checkpoint_sha = sha(run/'model.pth')
    sources = {str((ROOT/name).resolve()): sha(ROOT/name) for name in
        ('training/export_unified_retention_region_mixup.py', 'training/train_unified_retention_region_mixup.py',
         'training/retention_region_mixup_loss.py', 'training/train_unified_retention_adapter.py',
         'training/retention_adapter_network.py', 'training/export_unified_retention_core.py',
         'training/export_unified_regions.py', 'training/export_region_stable.py',
         'src/flux_glyph/unified_font.py', 'src/flux_glyph/region_font.py')}
    evidence = dict(selection['bindings'])
    for name in ('SELECTION.json', 'TRAINING_FREEZE.json', 'model.pth', 'CALIBRATION_OUTPUTS.npz',
                 'CALIBRATION_DECISIONS.json', 'BASELINE.json', 'BASELINE_CALIBRATION_DECISIONS.json',
                 'SAMPLING.json', 'TRAINING_COUNTS.json'):
        evidence[str(run/name)] = sha(run/name)
    for record in selection['history']:
        for entry in record['artifacts'].values():evidence[str(run/entry['path'])] = entry['sha256']
    checkpoint = torch.load(run/'model.pth', map_location='cpu', weights_only=True)
    validate_checkpoint(checkpoint, selection, selection_sha)
    step = torch.load(run/selection['selected']['artifacts']['checkpoint']['path'], map_location='cpu', weights_only=True)
    require(step['architecture'] == ARCHITECTURE and step['families'] == selection['families']
            and step['step'] == selection['selected']['step'] and step['training_protocol_sha256'] == selection['training_protocol_sha256']
            and state_sha(step['state_dict']) == selection['state_after_sha256'], 'Final adapter differs from the selected saved checkpoint')
    base = torch.load((ROOT/selection['base_checkpoint']['path']).resolve(), map_location='cpu', weights_only=True)
    require(base['selection_sha256'] == selection['base_selection']['sha256'] and base['families'] == selection['families']
            and base['architecture'] == trainer.core.ARCHITECTURE
            and state_sha(base['state_dict']) == selection['base_state_sha256'], 'Frozen CNN initializer state differs')
    initial = torch.load((ROOT/selection['initial_residual']['path']).resolve(), map_location='cpu', weights_only=True)
    require(set(initial) == {RESIDUAL_PREFIX+name for name in ('0.weight', '0.bias', '2.weight', '2.bias')}
            and state_sha(initial) == selection['residual_state_before_sha256']
            and state_sha({**base['state_dict'], **initial}) == selection['initial_state_sha256']
            and torch.count_nonzero(initial[RESIDUAL_PREFIX+'2.weight']).item() == 0
            and torch.count_nonzero(initial[RESIDUAL_PREFIX+'2.bias']).item() == 0,
            'Residual classifier did not begin with an exactly zero output correction')
    reference = RetentionAdapterClassifier(25).cpu().eval(); reference.load_state_dict(checkpoint['state_dict'], strict=True)
    require(all(parameter.requires_grad == name.startswith(RESIDUAL_PREFIX) for name, parameter in reference.named_parameters())
            and sum(isinstance(module, torch.nn.GroupNorm) for module in reference.modules()) == 4,
            'Adapter reference must retain the original four GroupNorm layers and frozen base')
    converted = copy.deepcopy(reference)
    require(replace_groupnorm(converted, high_precision=True) == 4
            and state_sha(converted.state_dict()) == selection['state_after_sha256'], 'GroupNorm lowering changed adapter parameters')
    cache_path = (ROOT/selection['cache_manifest']['path']).resolve(); manifest = trainer.read(cache_path)
    cache, _ = trainer.load_cache(cache_path.parent, manifest['identity']); cached = cache['calibration']
    cal = load_split(data, 'calibration')
    plan = trainer.core.read_plan((ROOT/selection['retention_plan']['path']).resolve(), data, (ROOT/selection['parent_checkpoint']['path']).resolve())
    torch.set_num_threads(4); output.mkdir(parents=True); model_path = output/'model.onnx'
    torch.onnx.export(converted, torch.zeros(2, 1, 64, 256), model_path, input_names=['tiles'],
        output_names=['logits', 'log_em_ratio'], dynamic_axes={'tiles': {0: 'batch'}, 'logits': {0: 'batch'}, 'log_em_ratio': {0: 'batch'}},
        opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(model_path))
    options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
    actual_logits, actual_sizes, reference_logits, reference_sizes = [], [], [], []
    maximum = {'feature': 0., 'base_logits': 0., 'size_cache': 0., 'logits': 0., 'size': 0.}
    with torch.inference_mode():
        for start in range(0, len(cal['tiles']), 128):
            block = np.array(cal['tiles'][start:start+128], copy=True); stop = start+len(block)
            features = reference.features(torch.from_numpy(block)); old = reference.family_head(features).numpy()
            logits = reference.classify(features).numpy(); sizes = reference.size_head(features).squeeze(-1).numpy()
            for key, observed, expected in [('feature', features.numpy(), cached['features'][start:stop]),
                ('base_logits', old, cached['base_logits'][start:stop]), ('size_cache', sizes, cached['log_em_ratio'][start:stop])]:
                np.testing.assert_allclose(observed, expected, atol=3e-4, rtol=3e-4)
                maximum[key] = max(maximum[key], float(np.max(np.abs(observed-expected))))
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
    parent_meta = trainer.read((ROOT/selection['parent_metadata']['path']).resolve())
    metadata = metadata_for_export(selection, cal, parent_meta, sha(model_path), checkpoint_sha, selection_sha, measured)
    dump(output/'metadata.json', metadata); UnifiedFontClassifier(output)
    require(validate(run, data) == selection and all(sha(path) == digest for path, digest in {**sources, **evidence}.items())
            and state_sha(reference.state_dict()) == state_sha(converted.state_dict()) == selection['state_after_sha256'],
            'Adapter evidence or weights changed during export')
    report = {'schema': SCHEMA, 'passed': True, 'promotion_allowed': True, 'font_model_count': 1, 'encoder_count': 1,
        'output_family_count': 25, 'network_architecture': ARCHITECTURE, 'selection_sha256': selection_sha,
        'objective_variant': selection['objective_variant'], 'training_counts_sha256': selection['training_counts_sha256'],
        'region_tile_training': selection['region_tile_training'], 'manifold_mixup': copy.deepcopy(selection['manifold_mixup']),
        'feature_cache_reused': True, 'synthetic_unknown_labels': False, 'mixup_deployed': False,
        'checkpoint_sha256': checkpoint_sha, 'model_sha256': sha(model_path), 'metadata_sha256': sha(output/'metadata.json'),
        'source_bindings': sources, 'calibration_bindings': evidence, 'fixed_runtime': FIXED_RUNTIME,
        'retention_plan_sha256': selection['retention_plan']['sha256'], 'runtime_gates_changed': False,
        'base_checkpoint': selection['base_checkpoint'], 'base_selection': selection['base_selection'],
        'base_state_sha256': selection['base_state_sha256'], 'cache_manifest': selection['cache_manifest'],
        'base_parameters_unchanged': True, 'raw_size_branch_frozen': True, 'all_parameters_trained': False,
        'only_residual_head_trained': True, 'residual_head_trained': True, 'frozen_parameters_unchanged': True,
        'original_torch_groupnorm_reference': True, 'cached_training_outputs_reproduce_selected_metrics': True,
        'full_encoder_features_match_cache': True, 'full_base_logits_match_cache': True,
        'full_size_logits_match_cache': True, 'cached_full_model_runtime_parity_passed': True,
        'calibration_font_decisions_identical': True, 'calibration_runtime_reasons_identical': True,
        'calibration_size_availability_identical': True, 'calibration_size_values_close': True,
        'calibration_scores_close': True, 'max_absolute_errors': maximum, 'max_size_pixel_error': max_size_px,
        'calibration_regions': len(cal['rows']), 'calibration_tiles': len(cal['tiles']), 'batch_checks': batches,
        'frozen_retention_summary': strip_artifacts(selection['selected']), 'runtime_retention_summary': measured,
        'runtime_calibration_metrics': measured['metrics'], 'test_read': False, 'development_holdout_read': False,
        'user_images_read': False, 'training_cache_in_deployed_model': False}
    dump(run/'PARITY.json', report)
    print(json.dumps({'passed': True, 'promotion_allowed': True, 'model_sha256': report['model_sha256'],
                      'calibration_regions': len(cal['rows']), 'maximum_errors': maximum}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'data', 'output'):parser.add_argument('--'+name, type=Path, required=True)
    export(parser.parse_args())
