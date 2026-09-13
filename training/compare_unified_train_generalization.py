#!/usr/bin/env python3
"""Compare frozen R21, b08 and confidence-floor logits on identical TRAIN tiles."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from audit_unified_train_generalization import face_weight, read, sha, summarize


FOCUS = {'LXGW WenKai', 'WenQuanYi Micro Hei'}


def metric_rows(rows, fields, targets, predictions, probabilities, families, predicate=lambda row: True):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if predicate(row):
            groups[tuple(row[field] for field in fields)].append(index)
    return [{**dict(zip(fields, key)), **summarize(indices, targets, predictions, probabilities, families)}
        for key, indices in sorted(groups.items())]


def model_metrics(name, logits, rows, families):
    maximum = np.max(logits, axis=1, keepdims=True)
    values = np.exp(logits - maximum); probabilities = values / values.sum(axis=1, keepdims=True)
    predictions = np.argmax(logits, axis=1)
    targets = np.asarray([row['target'] for row in rows], dtype=np.int64)
    unknown_index = families.index('__unknown__'); known = targets != unknown_index; unknown = ~known
    focus = lambda row: row['family'] in FOCUS
    return {'model': name,
        'overall': summarize(range(len(rows)), targets, predictions, probabilities, families),
        'partition': metric_rows(rows, ['partition'], targets, predictions, probabilities, families),
        'native_target_kind': metric_rows(rows, ['target_kind'], targets, predictions, probabilities, families,
            lambda row: row['quality'] == 'native'),
        'focus_overall': metric_rows(rows, ['family'], targets, predictions, probabilities, families, focus),
        'focus_partition': metric_rows(rows, ['family', 'partition'], targets, predictions, probabilities, families, focus),
        'focus_quality': metric_rows(rows, ['family', 'quality'], targets, predictions, probabilities, families, focus),
        'focus_view': metric_rows(rows, ['family', 'view'], targets, predictions, probabilities, families, focus),
        'focus_face': metric_rows(rows, ['family', 'font_face', 'face_weight'], targets, predictions, probabilities, families, focus),
        'focus_script': metric_rows(rows, ['family', 'script'], targets, predictions, probabilities, families, focus),
        'unknown_source': metric_rows(rows, ['source_font_family'], targets, predictions, probabilities, families,
            lambda row: row['target_kind'] == 'unknown'),
        'unknown_script': metric_rows(rows, ['script'], targets, predictions, probabilities, families,
            lambda row: row['target_kind'] == 'unknown'),
        'collisions': {'known_predicted_unknown': {'tiles': int(np.sum(known & (predictions == unknown_index))),
                'rate': float(np.mean(predictions[known] == unknown_index))},
            'unknown_predicted_named': {'tiles': int(np.sum(unknown & (predictions != unknown_index))),
                'rate': float(np.mean(predictions[unknown] != unknown_index))},
            'unknown_to_focus': {family: {'tiles': int(np.sum(unknown & (predictions == families.index(family)))),
                    'rate': float(np.mean(predictions[unknown] == families.index(family)))} for family in sorted(FOCUS)},
            'focus_to_unknown': {family: {'tiles': int(np.sum((targets == families.index(family))
                        & (predictions == unknown_index))),
                    'rate': float(np.mean(predictions[targets == families.index(family)] == unknown_index))}
                for family in sorted(FOCUS)},
            'unknown_top_named': [{'family': family, 'tiles': count, 'rate': count / int(np.sum(unknown))}
                for family, count in Counter(families[int(value)] for value in predictions[unknown]
                    if value != unknown_index).most_common(10)]}}


def indexed(rows):
    return {(tuple((key, row[key]) for key in ('family', 'partition') if key in row)): row for row in rows}


def deltas(metrics, baseline, current):
    result = {}
    for section in ('partition', 'native_target_kind', 'focus_overall', 'focus_partition', 'focus_quality',
                    'focus_view', 'focus_face', 'focus_script', 'unknown_source', 'unknown_script'):
        old_rows = metrics[baseline][section]; new_rows = metrics[current][section]
        keys = [key for key in new_rows[0] if key not in ('tiles', 'correct', 'argmax_accuracy',
            'true_class_probability', 'predicted_unknown_rate', 'predicted_named_rate', 'top_wrong_predictions')]
        old = {tuple(row[key] for key in keys): row for row in old_rows}
        result[section] = [{**{key: row[key] for key in keys}, 'tiles': row['tiles'],
            'argmax_accuracy_delta': row['argmax_accuracy'] - old[tuple(row[key] for key in keys)]['argmax_accuracy'],
            'mean_true_probability_delta': row['true_class_probability']['mean']
                - old[tuple(row[key] for key in keys)]['true_class_probability']['mean'],
            'predicted_unknown_rate_delta': row['predicted_unknown_rate']
                - old[tuple(row[key] for key in keys)]['predicted_unknown_rate']}
            for row in new_rows]
    return result


def compare(args):
    source = args.source.resolve(); cache = args.cache.resolve(); output = args.output.resolve()
    if output.exists():
        raise ValueError('Retain prior comparison evidence; output already exists')
    inference = read(source/'REPORT.json'); freeze = read(source/'INFERENCE_FREEZE.json')
    if (inference.get('schema') != 'flux-glyph-train-inference-diagnostic-v1'
            or sha(source/'INFERENCE_FREEZE.json') != inference['freeze_sha256']
            or any(inference.get(key) is not False for key in ('calibration_read', 'development_read', 'test_read'))
            or inference.get('optimizer_steps') != 0):
        raise ValueError('Inference comparison is not frozen TRAIN-only evidence')
    families = freeze['families']; evidence = {
        str(source/'REPORT.json'): sha(source/'REPORT.json'),
        str(source/'INFERENCE_FREEZE.json'): sha(source/'INFERENCE_FREEZE.json')}

    rows = []; slices = {}; offset = 0
    for partition in ('original', 'supplement', 'known'):
        descriptor = freeze['partitions'][partition]
        rows_path = Path(descriptor['rows']['path']).resolve()
        if descriptor['split'] != 'train' or sha(rows_path) != descriptor['rows']['sha256']:
            raise ValueError('TRAIN row binding changed')
        evidence[str(rows_path)] = sha(rows_path)
        source_rows = read(rows_path); tile_rows = [None] * descriptor['tile_count']
        for row in source_rows:
            enriched = dict(row, partition=partition,
                target_kind='unknown' if row['family'] == '__unknown__' else 'known',
                quality='native' if row['view'] == 'native' else 'degraded',
                script=row.get('script', 'metadata_unavailable'), face_weight=face_weight(row['font_face']))
            for tile in range(row['tile_start'], row['tile_start'] + row['tile_count']):
                if not 0 <= tile < len(tile_rows) or tile_rows[tile] is not None:
                    raise ValueError('Invalid TRAIN row-to-tile order')
                tile_rows[tile] = enriched
        if any(row is None for row in tile_rows):
            raise ValueError('Incomplete TRAIN row-to-tile order')
        rows.extend(tile_rows); slices[partition] = slice(offset, offset + len(tile_rows)); offset += len(tile_rows)

    arrays = {}; cache_manifest = read(cache/'CACHE_MANIFEST.json')
    evidence[str(cache/'CACHE_MANIFEST.json')] = sha(cache/'CACHE_MANIFEST.json')
    b08 = []
    for partition in ('original', 'supplement', 'known'):
        entry = cache_manifest['partitions'][partition]['base_logits']; path = (cache/entry['path']).resolve()
        if sha(path) != entry['sha256']:
            raise ValueError('Frozen b08 cache changed')
        evidence[str(path)] = sha(path); b08.append(np.load(path, allow_pickle=False))
    arrays['b08'] = np.concatenate(b08)
    for model in ('r21', 'confidence_floor'):
        blocks = []
        for partition in ('original', 'supplement', 'known'):
            entry = inference['models'][model]['partitions'][partition]['logits']
            path = Path(entry['path']).resolve()
            if sha(path) != entry['sha256']:
                raise ValueError('Frozen diagnostic logits changed')
            value = np.load(path, allow_pickle=False)
            expected = freeze['partitions'][partition]['tile_count']
            if value.shape != (expected, len(families)) or not np.isfinite(value).all():
                raise ValueError('Invalid diagnostic logit shape')
            evidence[str(path)] = sha(path); blocks.append(value)
        arrays[model] = np.concatenate(blocks)
    if any(value.shape != (len(rows), len(families)) for value in arrays.values()):
        raise ValueError('Models do not cover identical TRAIN tile order')

    for model, descriptor in freeze['checkpoints'].items():
        for entry in descriptor.values():
            path = Path(entry['path']).resolve()
            if sha(path) != entry['sha256']:
                raise ValueError('Diagnostic model identity changed')
            evidence[str(path)] = sha(path)
    teacher = cache_manifest['identity']['teacher']
    for entry in teacher.values():
        if isinstance(entry, dict) and set(entry) == {'path', 'sha256'}:
            path = Path(entry['path']).resolve()
            if sha(path) != entry['sha256']:
                raise ValueError('b08 teacher identity changed')
            evidence[str(path)] = sha(path)

    metrics = {name: model_metrics(name, logits, rows, families) for name, logits in arrays.items()}
    report = {'schema': 'flux-glyph-train-generalization-comparison-v2',
        'scope': {'split': 'train', 'models': ['r21', 'b08', 'confidence_floor'],
            'unit': 'cached tile', 'pixel_or_tensor_image_arrays_read': False,
            'new_inference_performed_by_comparison': False, 'optimizer_steps': 0,
            'calibration_read': False, 'development_read': False, 'test_read': False,
            'interpretation_limit': 'Observed TRAIN fit only. Differences do not establish held-out generalization or causal effects.'},
        'model_identity': {'r21': freeze['checkpoints']['r21'],
            'b08': teacher, 'confidence_floor': freeze['checkpoints']['confidence_floor']},
        'families': families, 'total_tiles': len(rows), 'metrics': metrics,
        'deltas': {'confidence_floor_minus_b08': deltas(metrics, 'b08', 'confidence_floor'),
            'confidence_floor_minus_r21': deltas(metrics, 'r21', 'confidence_floor'),
            'b08_minus_r21': deltas(metrics, 'r21', 'b08')},
        'notes': {'native_known_vs_unknown': 'Reported on all native TRAIN tiles by true target kind.',
            'script_availability': 'Original prepared TRAIN metadata has no script; script comparisons cover paired captures only.',
            'face_weight': 'Parsed conservatively from font_face; unspecified means no encoded weight.',
            'correlation': 'Multiple views and tiles from one source region are correlated.'}}
    output.mkdir(parents=True)
    report_path = output/'COMPARISON_V2.json'
    with report_path.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')

    def find(section, **wanted):
        return next(row for row in section if all(row.get(key) == value for key, value in wanted.items()))
    lines = ['# R21 / b08 / confidence-floor TRAIN comparison', '',
        'All figures enumerate the same 137,230 bound TRAIN tiles. The comparison read frozen logits and row metadata only; it performed no inference and read no image arrays, CAL, development, or TEST data.', '',
        '## Focus families', '',
        '| Model | LXGW WenKai accuracy / mean p(true) | WenQuanYi Micro Hei accuracy / mean p(true) |',
        '|---|---:|---:|']
    for model in ('r21', 'b08', 'confidence_floor'):
        a = find(metrics[model]['focus_overall'], family='LXGW WenKai')
        b = find(metrics[model]['focus_overall'], family='WenQuanYi Micro Hei')
        lines.append(f"| {model} | {a['argmax_accuracy']:.2%} / {a['true_class_probability']['mean']:.3f} | "
            f"{b['argmax_accuracy']:.2%} / {b['true_class_probability']['mean']:.3f} |")
    lines += ['', '## Native known/unknown separation', '',
        '| Model | Native known accuracy / mean p(true) | Native unknown withholding / mean p(unknown) |',
        '|---|---:|---:|']
    for model in ('r21', 'b08', 'confidence_floor'):
        known = find(metrics[model]['native_target_kind'], target_kind='known')
        unknown = find(metrics[model]['native_target_kind'], target_kind='unknown')
        lines.append(f"| {model} | {known['argmax_accuracy']:.2%} / {known['true_class_probability']['mean']:.3f} | "
            f"{unknown['argmax_accuracy']:.2%} / {unknown['true_class_probability']['mean']:.3f} |")
    lines += ['', '## New-source unknown collisions', '']
    for model in ('r21', 'b08', 'confidence_floor'):
        z = find(metrics[model]['unknown_source'], source_font_family='Zhuque Fangsong')
        w = find(metrics[model]['unknown_source'], source_font_family='WenQuanYi Zen Hei')
        lines.append(f"- {model}: Zhuque withheld {z['argmax_accuracy']:.2%}; Zen Hei withheld {w['argmax_accuracy']:.2%}. "
            f"Across all unknown TRAIN tiles, {metrics[model]['collisions']['unknown_predicted_named']['rate']:.2%} were named.")
    b08_lx = find(metrics['b08']['focus_overall'], family='LXGW WenKai')
    now_lx = find(metrics['confidence_floor']['focus_overall'], family='LXGW WenKai')
    b08_micro = find(metrics['b08']['focus_overall'], family='WenQuanYi Micro Hei')
    now_micro = find(metrics['confidence_floor']['focus_overall'], family='WenQuanYi Micro Hei')
    b08_lx_pair = find(metrics['b08']['focus_partition'], family='LXGW WenKai', partition='known')
    now_lx_pair = find(metrics['confidence_floor']['focus_partition'], family='LXGW WenKai', partition='known')
    b08_micro_pair = find(metrics['b08']['focus_partition'], family='WenQuanYi Micro Hei', partition='known')
    now_micro_pair = find(metrics['confidence_floor']['focus_partition'], family='WenQuanYi Micro Hei', partition='known')
    now_faces = {row['face_weight']: row for row in metrics['confidence_floor']['focus_face']
        if row['family'] == 'LXGW WenKai'}
    b08_faces = {row['face_weight']: row for row in metrics['b08']['focus_face']
        if row['family'] == 'LXGW WenKai'}
    now_lx_native = find(metrics['confidence_floor']['focus_quality'], family='LXGW WenKai', quality='native')
    b08_micro_native = find(metrics['b08']['focus_quality'], family='WenQuanYi Micro Hei', quality='native')
    now_micro_native = find(metrics['confidence_floor']['focus_quality'], family='WenQuanYi Micro Hei', quality='native')
    b08_micro_degraded = find(metrics['b08']['focus_quality'], family='WenQuanYi Micro Hei', quality='degraded')
    now_micro_degraded = find(metrics['confidence_floor']['focus_quality'], family='WenQuanYi Micro Hei', quality='degraded')
    now_native_known = find(metrics['confidence_floor']['native_target_kind'], target_kind='known')
    now_native_unknown = find(metrics['confidence_floor']['native_target_kind'], target_kind='unknown')
    b08_native_known = find(metrics['b08']['native_target_kind'], target_kind='known')
    b08_native_unknown = find(metrics['b08']['native_target_kind'], target_kind='unknown')
    now_z = find(metrics['confidence_floor']['unknown_source'], source_font_family='Zhuque Fangsong')
    now_w = find(metrics['confidence_floor']['unknown_source'], source_font_family='WenQuanYi Zen Hei')
    b08_lx_han = find(metrics['b08']['focus_script'], family='LXGW WenKai', script='han')
    now_lx_han = find(metrics['confidence_floor']['focus_script'], family='LXGW WenKai', script='han')
    b08_micro_han = find(metrics['b08']['focus_script'], family='WenQuanYi Micro Hei', script='han')
    now_micro_han = find(metrics['confidence_floor']['focus_script'], family='WenQuanYi Micro Hei', script='han')
    lines += ['', '## Where the confidence-floor run moved TRAIN fit', '',
        f"Against b08, the confidence-floor checkpoint lowers LXGW WenKai TRAIN accuracy from {b08_lx['argmax_accuracy']:.2%} to {now_lx['argmax_accuracy']:.2%} and raises its predicted-unknown rate from {b08_lx['predicted_unknown_rate']:.2%} to {now_lx['predicted_unknown_rate']:.2%}. Its paired-known Regular subset is {now_lx_pair['argmax_accuracy']:.2%}, versus {b08_lx_pair['argmax_accuracy']:.2%} for b08. By available face, current accuracy is Light {now_faces['light']['argmax_accuracy']:.2%}, Medium {now_faces['medium']['argmax_accuracy']:.2%}, and Regular {now_faces['regular']['argmax_accuracy']:.2%}; the largest b08-to-current drop is Light, at {(b08_faces['light']['argmax_accuracy']-now_faces['light']['argmax_accuracy'])*100:.2f} percentage points.", '',
        f"WenQuanYi Micro Hei moves from {b08_micro['argmax_accuracy']:.2%} to {now_micro['argmax_accuracy']:.2%} overall. Its native accuracy is nearly preserved ({b08_micro_native['argmax_accuracy']:.2%} to {now_micro_native['argmax_accuracy']:.2%}), while degraded accuracy falls from {b08_micro_degraded['argmax_accuracy']:.2%} to {now_micro_degraded['argmax_accuracy']:.2%}. The paired-known subset is {now_micro_pair['argmax_accuracy']:.2%}, close to b08's {b08_micro_pair['argmax_accuracy']:.2%}, so the larger current decline occurs in the original TRAIN subset.", '',
        f"The confidence-floor run improves unknown withholding on TRAIN, including the new sources, but does not fit them completely: Zhuque Fangsong reaches {now_z['argmax_accuracy']:.2%} and WenQuanYi Zen Hei {now_w['argmax_accuracy']:.2%}. Across native tiles, it combines {now_native_known['argmax_accuracy']:.2%} known accuracy with {now_native_unknown['argmax_accuracy']:.2%} unknown withholding; b08 was {b08_native_known['argmax_accuracy']:.2%} and {b08_native_unknown['argmax_accuracy']:.2%}, respectively.", '',
        f"Script metadata exists only in paired captures. Relative to b08, current Han accuracy changes from {b08_lx_han['argmax_accuracy']:.2%} to {now_lx_han['argmax_accuracy']:.2%} for LXGW WenKai and from {b08_micro_han['argmax_accuracy']:.2%} to {now_micro_han['argmax_accuracy']:.2%} for WenQuanYi Micro Hei. Latin/numeric groups contain only 104–117 tiles per family; current differences there should be treated as small-sample descriptive results."]
    lines += ['', '## Interpretation', '',
        'The detailed JSON carries per-partition, native/degraded, view, available face/weight, and available-script results plus model-to-model deltas and top confusions.',
        '', 'These are TRAIN-fit measurements. A difference between these results and separately measured CAL behavior can locate a train/CAL gap, but this report does not read CAL and does not attribute that gap to a particular loss, script, face, or architecture.']
    summary_path = output/'SUMMARY_V2.md'; summary_path.write_text('\n'.join(lines) + '\n')
    evidence_report = {'schema': 'flux-glyph-train-generalization-comparison-evidence-v2',
        'inputs': evidence, 'source_sha256': sha(Path(__file__)),
        'comparison_sha256': sha(report_path), 'summary_sha256': sha(summary_path), **report['scope']}
    with (output/'EVIDENCE_V2.json').open('x') as stream:
        json.dump(evidence_report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'comparison': str(report_path), 'summary': str(summary_path),
        'focus': {model: metrics[model]['focus_overall'] for model in metrics},
        'native_target_kind': {model: metrics[model]['native_target_kind'] for model in metrics}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
        default=Path('artifacts/unified-font-v3/train-inference-audit-v1'))
    parser.add_argument('--cache', type=Path,
        default=Path('artifacts/unified-font-v2/teacher-preserve-cache-v1'))
    parser.add_argument('--output', type=Path,
        default=Path('artifacts/unified-font-v3/train-generalization-audit-v2'))
    compare(parser.parse_args())
