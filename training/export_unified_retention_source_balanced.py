#!/usr/bin/env python3
"""Export the source-balanced full CNN with binary unknown supervision and verified TRAIN source counts."""
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

ARCHITECTURE = 'region-cnn64x256-unified-v1'
SCHEMA = 'flux-glyph-unified-retention-onnx-parity-v1'
GROUPS = ('trunk', 'style', 'family_head', 'size_head')
RESIDUAL_PREFIX = 'residual_family_head.'


def source_entry(entry, bindings):
    require(isinstance(entry, dict) and set(entry) == {'path', 'sha256'}, 'Incomplete adapter source identity')
    path = (ROOT/entry['path']).resolve()
    require(path.is_file() and sha(path) == entry['sha256'] == bindings.get(str(path)), 'Adapter source binding differs')
    return path


def validate_checkpoint(checkpoint, selection, selection_sha256):
    state = checkpoint['state_dict']
    require(checkpoint.get('families') == selection['families'] and checkpoint.get('architecture') == ARCHITECTURE
            and checkpoint.get('selection_sha256') == selection_sha256
            and state_sha(state) == selection['state_after_sha256'] != selection['initial_state_sha256']
            and not any(key.startswith(RESIDUAL_PREFIX) for key in state), 'Selected full-CNN identity differs')
    for name in GROUPS:
        group = {key: value for key, value in state.items() if key.startswith(name+'.')}
        require(group and state_sha(group) == selection['selected_parameter_groups_sha256'][name]
                != selection['initial_parameter_groups_sha256'][name], 'Selected full-CNN group was not trained: '+name)


def validate(run, data):
    """Read-only source/CAL-cache validation; imports no Torch and reads no holdout."""
    import train_unified_retention_source_balanced as trainer
    run, data = Path(run).resolve(), Path(data).resolve()
    selection = trainer.read(run/'SELECTION.json')
    require(selection.get('schema') == 'flux-glyph-unified-retention-source-balanced-selection-v1'
            and selection.get('architecture') == ARCHITECTURE and selection.get('promotion_allowed') is True
            and not (run/'NO_PROMOTABLE_CHECKPOINT.json').exists(), 'Adapter has no promotable frozen checkpoint')
    protocol = trainer.read(run/'TRAINING_FREEZE.json')
    require(protocol.get('schema') == 'flux-glyph-unified-retention-source-balanced-protocol-v1'
            and sha(run/'TRAINING_FREEZE.json') == selection['training_protocol_sha256']
            and all(selection.get(key) == value for key, value in protocol.items() if key != 'schema'),
            'Adapter protocol and selection differ')
    require('soft_unknown_supervision' not in selection and 'soft_unknown_supervision' not in protocol,
            'Obsolete uniform named-class supervision cannot describe binary unknown training')
    expected = {'architecture': ARCHITECTURE, 'objective_variant': trainer.OBJECTIVE_VARIANT,
        'objective': trainer.OBJECTIVE, 'sampling': trainer.SAMPLING, 'fixed_runtime': FIXED_RUNTIME,
        'unknown_binary_supervision': trainer.UNKNOWN_BINARY_SUPERVISION,
        'design_changes': trainer.DESIGN_CHANGES, 'single_change_causal_attribution': False,
        'steps': 3000, 'eval_every': 500, 'batch_size': 96, 'learning_rate': 2e-5, 'minimum_learning_rate': 2e-6,
        'weight_decay': 1e-4, 'gradient_clip_norm': 5., 'seed': 2026091410,
        'base_selected_step': 1000, 'source_optimizer_steps_executed': 1500, 'optimizer_steps_executed': 3000,
        'frozen_groups': [], 'trainable_groups': list(GROUPS), 'all_parameters_trained': True,
        'base_frozen': False, 'size_head_frozen': False, 'residual_head_trained': False,
        'teacher_logits_used': True, 'second_model_resident': False,
        'teacher_optimizer_steps': 0, 'teacher_cache_deployed': False,
        'feature_cache_reused': False, 'teacher_cache_reused': True, 'cached_feature_extraction_performed': False,
        'training_inputs': ['image_tiles'],
        'calibration_execution': 'Current complete CNN forward over all original CAL image tiles; current learned size head.',
        'model_count': 1, 'encoder_count': 1, 'platform_routing': False, 'score_merging': False,
        'test_read': False, 'development_holdout_read': False, 'runtime_gates_searched': False}
    require(all(protocol.get(key) == value for key, value in expected.items())
            and protocol.get('training_device') in ('cpu', 'mps') and type(selection.get('passed')) is bool
            and selection['calibration_passed'] is selection['passed'], 'Full-CNN training or cached-teacher contract differs')
    for field in ('initial_parameter_groups_sha256', 'selected_parameter_groups_sha256', 'final_parameter_groups_sha256'):
        require(set(selection[field]) == set(GROUPS) and all(isinstance(value, str) and len(value) == 64
                for value in selection[field].values()), 'Incomplete full-CNN group evidence')
    require(all(selection['initial_parameter_groups_sha256'][name] != selection[field][name]
                for field in ('selected_parameter_groups_sha256', 'final_parameter_groups_sha256') for name in GROUPS),
            'Every selected and final CNN parameter group must have trained')
    bindings = selection['bindings']; trainer.core.verify_bindings(bindings)
    base_path = source_entry(selection['base_checkpoint'], bindings)
    base_selection_path = source_entry(selection['base_selection'], bindings)
    require(sha(base_path) == trainer.BASE_CHECKPOINT_SHA and sha(base_selection_path) == trainer.BASE_SELECTION_SHA
            and base_selection_path == base_path.parent/'SELECTION.json', 'Wrong frozen core initializer')
    base = trainer.read(base_selection_path)
    require(base.get('schema') == 'flux-glyph-unified-retention-selection-v1'
            and base['selected']['step'] == 1000 and base['optimizer_steps_executed'] == 1500
            and base['promotion_allowed'] is False and base['test_read'] is False and base['development_holdout_read'] is False
            and base['state_after_sha256'] == selection['base_state_sha256'] == selection['initial_state_sha256']
            == selection['state_before_sha256'] == selection['cached_teacher_state_sha256']
            and base['families'] == selection['families'] and len(selection['families']) == 25
            and len(set(selection['families'])) == 25 and selection['families'].count('__unknown__') == 1
            and all(bindings.get(path) == digest for path, digest in base['bindings'].items()),
            'Adapter base state, classes or source closure differ')
    require(selection['inheritance'] == {'all_family_rows_inherited': True, 'all_parameters_inherited': True,
            'all_parameters_trainable': True, 'source_state_sha256': selection['initial_state_sha256'],
            'family_count': 25, 'new_random_output_rows': 0}, 'Full-CNN parent inheritance differs')
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
    required = [ROOT/name for name in ('training/train_unified_retention_source_balanced.py', 'training/retention_source_balanced_sampler.py',
        'training/retention_supplement_sampler.py',
        'training/prepare_unified_unknown_supplement.py', 'training/cache_unified_unknown_supplement.py',
        'training/train_unified_retention_adapter.py', 'training/retention_adapter_network.py',
        'training/train_unified_retention_core.py', 'training/train_unified_retention.py', 'training/train_unified_regions.py',
        'training/prepare_unified_regions.py', 'training/region_network.py', 'training/network.py', 'training/train_regions.py',
        'training/evaluate_unified_retention_source_balanced.py', 'src/flux_glyph/unified_font.py', 'src/flux_glyph/region_font.py')]
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
    require(sha(cache_path) == trainer.CACHE_MANIFEST_SHA, 'Supplement run changed the original core feature cache')
    cache_manifest = trainer.read(cache_path)
    identity = cache_manifest['identity']
    require(identity['architecture'] == 'region-cnn64x256-residual-head-v1'
            and identity['base_checkpoint'] == selection['base_checkpoint']
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
    baseline, outputs, _ = evaluate_outputs(cache['calibration']['base_logits'], cache['calibration']['log_em_ratio'], cal, plan, 0)
    require(set(selection['baseline_bindings']) == {'BASELINE.json', 'BASELINE_CALIBRATION_DECISIONS.json'}
            and all(sha(run/name) == digest for name, digest in selection['baseline_bindings'].items())
            and trainer.read(run/'BASELINE.json') == baseline
            and trainer.read(run/'BASELINE_CALIBRATION_DECISIONS.json') == {'families': selection['families'], 'records': outputs},
            'Adapter baseline does not reproduce the frozen base cache')
    history = selection['history']
    require([row['step'] for row in history] == list(range(500, 3001, 500)), 'Incomplete adapter checkpoint history')
    for record in history:
        directory = f"checkpoints/step{record['step']:05d}"
        require(set(record['artifacts']) == {'checkpoint', 'outputs', 'decisions', 'metrics'}, 'Incomplete adapter step evidence')
        paths = {key: checked_file(run, record['artifacts'][key], directory+'/'+name) for key, name in
            [('checkpoint', 'model.pth'), ('outputs', 'CALIBRATION_OUTPUTS.npz'),
             ('decisions', 'CALIBRATION_DECISIONS.json'), ('metrics', 'METRICS.json')]}
        actual, _, _ = cached_evaluation(paths['outputs'], paths['decisions'], cal, plan, record['step'])
        require(actual == strip_artifacts(record) == trainer.read(paths['metrics']), 'Adapter CAL metrics do not reproduce cached outputs')
    require(selection['selected'] == max(history, key=retention_rank) and selection['selected']['promotion_allowed'] is True
            and selection['selected']['metrics']['passed'] is selection['passed'], 'Adapter selected a different CAL candidate')
    for key, name, field in [('outputs', 'CALIBRATION_OUTPUTS.npz', 'calibration_outputs_sha256'),
                            ('decisions', 'CALIBRATION_DECISIONS.json', 'calibration_decisions_sha256')]:
        require(sha(run/name) == selection[field] == selection['selected']['artifacts'][key]['sha256'], 'Selected adapter CAL bytes differ')
    validate_supplement(selection, data)
    validate_counts(run, selection, data)
    return selection


def validate_supplement(selection, data):
    """Validate bound extra TRAIN metadata/cache without re-running an encoder."""
    from collections import Counter
    import train_unified_retention_source_balanced as trainer
    bindings = selection['bindings']
    data_path = source_entry(selection['supplement_data'], bindings)
    cache_path = source_entry(selection['supplement_cache'], bindings)
    require(sha(data_path) == trainer.SUPPLEMENT_MANIFEST_SHA and sha(cache_path) == trainer.SUPPLEMENT_CACHE_SHA,
            'Extra native TRAIN data or feature cache differs from the fixed supplement')
    arrays, cache = trainer.load_supplement_cache(cache_path.parent)
    identity = cache['identity']; manifest = trainer.read(data_path)
    require(identity['data_manifest'] == selection['supplement_data']
            and identity['base_checkpoint'] == selection['base_checkpoint']
            and identity['base_selection'] == selection['base_selection']
            and identity['base_state_sha256'] == selection['base_state_sha256']
            and identity['old_cache_manifest'] == selection['cache_manifest']
            and identity['families'] == manifest['families'] == selection['families']
            and identity['partition']['split'] == 'train'
            and all(bindings.get(path) == digest for path, digest in identity['bindings'].items()),
            'Supplement was not derived from the same frozen base and separate TRAIN source')
    for name in ('data_manifest', 'preparation_freeze', 'partition_manifest', 'old_cache_manifest', 'old_cache_freeze'):
        source_entry(identity[name], bindings)
    source_entry({'path': str(cache_path.parent/'CACHE_FREEZE.json'), 'sha256': cache['cache_freeze_sha256']}, bindings)
    for item in cache['partitions']['train'].values():
        source_entry({'path': str((cache_path.parent/item['path']).resolve()), 'sha256': item['sha256']}, bindings)
    part_path = Path(identity['partition_manifest']['path']); part = trainer.read(part_path)
    for key in ('metadata', 'array'):
        source_entry({'path': str((part_path.parent/part[key]['path']).resolve()), 'sha256': part[key]['sha256']}, bindings)
    new_rows = trainer.read(part_path.parent/part['metadata']['path'])
    require(len(new_rows) == part['views'] == identity['partition']['region_count']
            and len(arrays['features']) == part['tiles'] == identity['partition']['tile_count']
            and all(row.get('split') == 'train' and row.get('family') == '__unknown__' and row.get('target') == 24
                and row.get('domain') == 'android' and row.get('native_font_verified') is True
                and row.get('source_font_family') in trainer.NEW_SOURCES for row in new_rows)
            and set(row['source_font_family'] for row in new_rows) == set(trainer.NEW_SOURCES),
            'Supplement contains a non-TRAIN or unverified/unmapped negative')
    original_part = trainer.read(Path(data)/'train/MANIFEST.json')
    original_rows = trainer.read(Path(data)/'train'/original_part['metadata']['path'])
    original_regions = {(row['source_id'], row['region_id']) for row in original_rows}
    new_regions = {(row['source_id'], row['region_id']) for row in new_rows}
    require(not original_regions & new_regions, 'Supplement overlaps an original TRAIN source region')
    expected = {'original_views': len(original_rows), 'supplement_views': len(new_rows),
        'combined_views': len(original_rows)+len(new_rows),
        'original_native_regions': len(original_regions), 'supplement_native_regions': len(new_regions),
        'combined_native_regions': len(original_regions)+len(new_regions),
        'original_tiles': original_part['tiles'], 'supplement_tiles': part['tiles'],
        'combined_tiles': original_part['tiles']+part['tiles'],
        'calibration_unchanged': True, 'derived_views_are_correlated': True}
    require(selection['training_data_counts'] == expected, 'Supplement dataset population counts differ')


def source_order_from_train(data, selection):
    """Derive the source cycle from bound TRAIN metadata, without opening pixels."""
    import train_unified_retention_source_balanced as trainer
    folder = Path(data).resolve()/'train'
    part_path = folder/'MANIFEST.json'; part = trainer.read(part_path)
    require(part.get('split') == 'train' and part.get('families') == selection['families']
            and selection['bindings'].get(str(part_path)) == sha(part_path), 'Unbound original TRAIN partition')
    path = (folder/part['metadata']['path']).resolve()
    require(path.parent == folder and selection['bindings'].get(str(path)) == part['metadata']['sha256'] == sha(path),
            'Unbound original TRAIN source metadata')
    rows = trainer.read(path); original = set()
    for row in rows:
        require(row.get('split') == 'train' and row.get('family') in selection['families']
                and type(row.get('target')) is int and row['target'] == selection['families'].index(row['family']),
                'Source-cycle derivation encountered invalid TRAIN labels')
        if row['family'] == '__unknown__':
            source = row.get('source_font_family')
            require(isinstance(source, str) and source and row.get('domain') in ('ios', 'android'),
                    'Unknown TRAIN source identity is missing')
            original.add(source)
    require(len(original) == 9 and not original.intersection(trainer.NEW_SOURCES),
            'Expected nine original TRAIN unknown sources disjoint from the supplement')
    order = sorted(original.union(trainer.NEW_SOURCES))
    trainer.expected_unknown_counts(order, 0)
    return order


def validate_counts(run, selection, data):
    from collections import Counter
    import train_unified_retention_source_balanced as trainer
    run = Path(run)
    require(sha(run/'SAMPLING.json') == selection['sampling_sha256']
            and sha(run/'TRAINING_COUNTS.json') == selection['training_counts_sha256'], 'Full-CNN training counts changed')
    sampling = trainer.read(run/'SAMPLING.json'); counts = trainer.read(run/'TRAINING_COUNTS.json')
    require('soft_unknown_supervision' not in counts and 'soft_unknown_supervision' not in sampling,
            'Obsolete uniform target metadata in source-balanced training evidence')
    trainer.validate_training_counts(counts, selection['families'], 3000)
    order = source_order_from_train(data, selection)
    expected_unknown = trainer.expected_unknown_counts(order, 48000)
    expected_new = {source: expected_unknown[source] for source in trainer.NEW_SOURCES}
    new_total = sum(expected_new.values()); old_unknown = 48000-new_total
    require(selection.get('unknown_source_order') == counts['unknown_source_order'] == sampling.get('unknown_source_order') == order
            and sampling.get('unknown_source_rows') == expected_unknown and sampling.get('unknown_rows') == 48000,
            'Reported unknown cycle differs from the bound TRAIN source families')
    require(counts['eligible_family_rows'].get('__unknown__', 0) == 0
            and all(row['eligible_rows'] == 0 for row in counts['source_target_rows']
                    if row['target_family'] == '__unknown__'), 'Unknown TRAIN rows received forbidden KL teacher targets')
    slots = {key: trainer.SAMPLING[key]*3000 for key in ('base_known',
        'ios_native_pingfang', 'ios_native_sfpro_helvetica', 'ios_native_original_eight')}
    slots.update(base_unknown=old_unknown, supplement_unknown=new_total)
    families = Counter({family: 6000 for family in selection['families'] if family != '__unknown__'})
    families.update({'__unknown__': 48000, 'PingFang': 48000, 'SF Pro': 12000, 'Helvetica': 12000})
    families.update({family: 3000 for family in trainer.SAMPLING['original_eight_families']})
    require(sampling['schema'] == 'flux-glyph-retention-source-balanced-sampling-v1'
            and sampling.get('source_balanced') is True
            and sampling['slots'] == slots and sampling['base']['known_rows'] == 144000
            and sampling['base']['unknown_rows'] == old_unknown and sampling['supplement_source_rows'] == expected_new
            and sampling['proposal_rows_discarded'] == 0 and sampling['test_read'] is False
            and sampling['development_holdout_read'] is False
            and counts['family_rows'] == sampling['family_rows'] == dict(families)
            and counts['domain_rows'] == sampling['domain_rows']
            and counts['supplement_rows'] == new_total and counts['original_rows'] == 288000-new_total
            and counts['supplement_source_rows'] == expected_new,
            'Full-CNN original/supplement replay populations differ')
    actual = {}
    for row in sampling['source_rows']:
        key = (row['domain'], row['source_font_family'], row['target_family'])
        require(key not in actual and type(row['rows']) is int and row['rows'] > 0, 'Invalid actual sampler source accounting')
        actual[key] = row['rows']
    sources = {(row['domain'], row['source_font_family'], row['target_family']): row['rows']
               for row in counts['source_target_rows']}
    require(actual == sources and all(sources.get(('android', source, '__unknown__')) == expected_new[source] for source in trainer.NEW_SOURCES)
            and all(type(n) is int and n >= 0 for group in ('view_rows', 'supplement_view_rows') for n in sampling[group].values())
            and sum(sampling['view_rows'].values()) == 288000
            and set(sampling['view_rows']) <= set(trainer.SAMPLING['supplement_view_weights'])
            and set(sampling['supplement_view_rows']) <= set(trainer.SAMPLING['supplement_view_weights'])
            and sum(sampling['supplement_view_rows'].values()) == new_total
            and all(n <= sampling['view_rows'].get(view, 0) for view, n in sampling['supplement_view_rows'].items()),
            'Full-CNN teacher/source accounting differs from actual sampled examples')


def training_metadata(selection):
    keys = ('objective_variant', 'objective', 'unknown_binary_supervision', 'sampling', 'unknown_source_order',
        'design_changes', 'single_change_causal_attribution', 'supplement_data', 'supplement_cache', 'training_data_counts',
        'base_checkpoint', 'base_selection', 'base_state_sha256', 'cache_manifest', 'initial_state_sha256',
        'state_before_sha256', 'state_after_sha256', 'cached_teacher_state_sha256',
        'initial_parameter_groups_sha256', 'selected_parameter_groups_sha256', 'final_parameter_groups_sha256',
        'training_counts_sha256', 'optimizer_steps_executed', 'source_optimizer_steps_executed', 'training_device')
    return {**{key: copy.deepcopy(selection[key]) for key in keys}, 'selected_step': selection['selected']['step'],
        'all_parameters_trained': True, 'base_frozen': False, 'size_head_frozen': False, 'residual_head_trained': False,
        'trainable_parameter_groups': list(GROUPS), 'frozen_parameter_groups': [],
        'model_count': 1, 'encoder_count': 1, 'teacher_logits_used': True, 'second_model_resident': False,
        'teacher_optimizer_steps': 0, 'teacher_cache_deployed': False, 'teacher_in_deployed_model': False,
        'feature_cache_reused': False, 'teacher_cache_reused': True, 'training_inputs': ['image_tiles'],
        'ocr_text_used': False, 'platform_routing': False, 'score_merging': False,
        'development_holdout_is_blind_test': False}


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
        'validation': {'kind': 'full_cnn_source_balanced_binary_unknown_calibration_only_at_export',
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
    import train_unified_retention_source_balanced as trainer
    from region_network import RegionFontClassifier
    from export_region_stable import replace_groupnorm
    from prepare_unified_regions import load_split
    from train_unified_regions import region_outputs
    from flux_glyph.unified_font import UnifiedFontClassifier
    run, data, output = (Path(value).resolve() for value in (args.run, args.data, args.output))
    require(not output.exists() and not (run/'PARITY.json').exists(), 'Preserve earlier adapter export attempts')
    selection = validate(run, data); selection_sha = sha(run/'SELECTION.json'); checkpoint_sha = sha(run/'model.pth')
    sources = {str((ROOT/name).resolve()): sha(ROOT/name) for name in
        ('training/export_unified_retention_source_balanced.py', 'training/train_unified_retention_source_balanced.py',
         'training/retention_source_balanced_sampler.py',
         'training/retention_supplement_sampler.py', 'training/prepare_unified_unknown_supplement.py',
         'training/cache_unified_unknown_supplement.py', 'training/train_unified_retention_adapter.py',
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
    require(state_sha(base['state_dict']) == selection['initial_state_sha256'], 'Full-CNN initialization was not exact')
    for name in GROUPS:
        group = {key: value for key, value in base['state_dict'].items() if key.startswith(name+'.')}
        require(state_sha(group) == selection['initial_parameter_groups_sha256'][name], 'Initial full-CNN group differs')
    last = torch.load(run/selection['history'][-1]['artifacts']['checkpoint']['path'], map_location='cpu', weights_only=True)
    require(last['step'] == 3000 and last['architecture'] == ARCHITECTURE and last['families'] == selection['families']
            and last['training_protocol_sha256'] == selection['training_protocol_sha256'], 'Final full-CNN step identity differs')
    for name in GROUPS:
        group = {key: value for key, value in last['state_dict'].items() if key.startswith(name+'.')}
        require(group and state_sha(group) == selection['final_parameter_groups_sha256'][name], 'Final full-CNN group differs')
    reference = RegionFontClassifier(25).cpu().eval(); reference.load_state_dict(checkpoint['state_dict'], strict=True)
    require(not any(name.startswith(RESIDUAL_PREFIX) for name in reference.state_dict())
            and sum(isinstance(module, torch.nn.GroupNorm) for module in reference.modules()) == 4,
            'Full-CNN reference must retain one original four-GroupNorm encoder and no residual head')
    converted = copy.deepcopy(reference)
    require(replace_groupnorm(converted, high_precision=True) == 4
            and state_sha(converted.state_dict()) == selection['state_after_sha256'], 'GroupNorm lowering changed adapter parameters')
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
    parent_meta = trainer.read((ROOT/selection['parent_metadata']['path']).resolve())
    metadata = metadata_for_export(selection, cal, parent_meta, sha(model_path), checkpoint_sha, selection_sha, measured)
    dump(output/'metadata.json', metadata); UnifiedFontClassifier(output)
    require(validate(run, data) == selection and all(sha(path) == digest for path, digest in {**sources, **evidence}.items())
            and state_sha(reference.state_dict()) == state_sha(converted.state_dict()) == selection['state_after_sha256'],
            'Adapter evidence or weights changed during export')
    report = {'schema': SCHEMA, 'passed': True, 'promotion_allowed': True, 'font_model_count': 1, 'encoder_count': 1,
        'output_family_count': 25, 'network_architecture': ARCHITECTURE, 'selection_sha256': selection_sha,
        'supplement_data': copy.deepcopy(selection['supplement_data']),
        'supplement_cache': copy.deepcopy(selection['supplement_cache']),
        'training_data_counts': copy.deepcopy(selection['training_data_counts']),
        'checkpoint_sha256': checkpoint_sha, 'model_sha256': sha(model_path), 'metadata_sha256': sha(output/'metadata.json'),
        'source_bindings': sources, 'calibration_bindings': evidence, 'fixed_runtime': FIXED_RUNTIME,
        'retention_plan_sha256': selection['retention_plan']['sha256'], 'runtime_gates_changed': False,
        'base_checkpoint': selection['base_checkpoint'], 'base_selection': selection['base_selection'],
        'base_state_sha256': selection['base_state_sha256'], 'cache_manifest': selection['cache_manifest'],
        'all_parameters_trained': True, 'base_frozen': False, 'size_head_frozen': False, 'residual_head_trained': False,
        'initial_state_sha256': selection['initial_state_sha256'],
        'state_before_sha256': selection['state_before_sha256'], 'state_after_sha256': selection['state_after_sha256'],
        'cached_teacher_state_sha256': selection['cached_teacher_state_sha256'],
        'objective_variant': selection['objective_variant'], 'training_counts_sha256': selection['training_counts_sha256'],
        'unknown_binary_supervision': copy.deepcopy(selection['unknown_binary_supervision']),
        'sampling': copy.deepcopy(selection['sampling']),
        'unknown_source_order': copy.deepcopy(selection['unknown_source_order']),
        'design_changes': copy.deepcopy(selection['design_changes']),
        'single_change_causal_attribution': selection['single_change_causal_attribution'],
        'initial_parameter_groups_sha256': copy.deepcopy(selection['initial_parameter_groups_sha256']),
        'selected_parameter_groups_sha256': copy.deepcopy(selection['selected_parameter_groups_sha256']),
        'final_parameter_groups_sha256': copy.deepcopy(selection['final_parameter_groups_sha256']),
        'teacher_logits_used': True, 'second_model_resident': False, 'teacher_optimizer_steps': 0,
        'teacher_cache_deployed': False, 'teacher_in_deployed_model': False,
        'feature_cache_reused': False, 'teacher_cache_reused': True,
        'export_parameters_unchanged': True, 'original_torch_groupnorm_reference': True,
        'teacher_cache_unchanged': True,
        'cached_training_outputs_reproduce_selected_metrics': True, 'cached_full_model_runtime_parity_passed': True,
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
