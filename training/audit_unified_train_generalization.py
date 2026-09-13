#!/usr/bin/env python3
"""Summarize frozen b08 logits over bound TRAIN metadata; read no pixels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


VIEW_WEIGHTS = {'native': .4, 'half': .2, 'three_quarters_jpeg75': .2, 'jpeg75': .2}
CORE_WEIGHTED = {'PingFang', 'SF Pro', 'Helvetica'}
FOCUS = {'LXGW WenKai', 'WenQuanYi Micro Hei'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def face_weight(face):
    normalized = face.lower().replace('_', '-').replace(' ', '-')
    for label in ('extra-black', 'semibold', 'demibold', 'medium', 'regular', 'normal',
                  'light', 'thin', 'bold', 'heavy', 'black'):
        if label in normalized:
            return label
    return 'unspecified'


def summarize(indices, targets, predictions, probabilities, families):
    idx = np.asarray(indices, dtype=np.int64)
    truth = targets[idx]; pred = predictions[idx]
    values = probabilities[idx, truth]
    correct = int(np.sum(pred == truth)); count = len(idx)
    wrong = Counter(families[int(value)] for value in pred[pred != truth])
    unknown_index = families.index('__unknown__')
    return {'tiles': count, 'correct': correct, 'argmax_accuracy': correct / count,
        'true_class_probability': {'mean': float(np.mean(values)), 'median': float(np.median(values)),
            'p10': float(np.quantile(values, .1)), 'p90': float(np.quantile(values, .9))},
        'predicted_unknown_rate': float(np.mean(pred == unknown_index)),
        'predicted_named_rate': float(np.mean(pred != unknown_index)),
        'top_wrong_predictions': [{'family': family, 'tiles': n, 'rate': n / count}
            for family, n in wrong.most_common(5)]}


def grouped(rows, dimensions, targets, predictions, probabilities, families):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[tuple(row[name] for name in dimensions)].append(index)
    result = []
    for key, indices in sorted(groups.items()):
        result.append({**dict(zip(dimensions, key)),
            **summarize(indices, targets, predictions, probabilities, families)})
    return result


def audit(args):
    cache = args.cache.resolve(); output = args.output.resolve()
    if output.exists():
        raise ValueError('Retain prior audit evidence; output already exists')
    manifest_path = cache/'CACHE_MANIFEST.json'; manifest = read(manifest_path)
    identity = manifest['identity']; families = identity['families']
    if (manifest.get('schema') != 'flux-glyph-unified-student-teacher-cache-v1'
            or identity.get('test_read') is not False
            or identity.get('development_holdout_read') is not False
            or identity.get('calibration_images_read') is not False
            or identity.get('teacher_logits_recomputed') is not True
            or identity.get('historical_logits_reused') is not False):
        raise ValueError('Cache is not the frozen TRAIN-only b08 teacher evidence')

    evidence = {}
    all_rows = []; logits_blocks = []
    partition_counts = {}
    for partition in ('original', 'supplement', 'known'):
        descriptor = identity['partitions'][partition]
        if descriptor['split'] != 'train':
            raise ValueError('Non-TRAIN partition requested')
        rows_path = Path(descriptor['rows']['path']).resolve()
        part_path = Path(descriptor['partition_manifest']['path']).resolve()
        logits_entry = manifest['partitions'][partition]['base_logits']
        logits_path = (cache/logits_entry['path']).resolve()
        for path, expected in ((rows_path, descriptor['rows']['sha256']),
                               (part_path, descriptor['partition_manifest']['sha256']),
                               (logits_path, logits_entry['sha256'])):
            actual = sha(path)
            if actual != expected:
                raise ValueError('Bound audit input changed: ' + str(path))
            evidence[str(path)] = actual
        rows = read(rows_path); logits = np.load(logits_path, allow_pickle=False)
        if logits.shape != (descriptor['tile_count'], len(families)) or not np.isfinite(logits).all():
            raise ValueError('Invalid cached b08 logits: ' + partition)
        tile_rows = [None] * len(logits)
        for row in rows:
            if row.get('split') != 'train' or row.get('native_font_verified') is not True:
                raise ValueError('Non-TRAIN or unverified row in audit')
            start, count = row['tile_start'], row['tile_count']
            if not 0 <= start < start + count <= len(logits):
                raise ValueError('Row escapes cached tile order')
            enriched = dict(row, partition=partition,
                target_kind='unknown' if row['family'] == '__unknown__' else 'known',
                script=row.get('script', 'metadata_unavailable'),
                face_weight=face_weight(row['font_face']),
                quality='native' if row['view'] == 'native' else 'degraded',
                sampling_view_weight=VIEW_WEIGHTS[row['view']],
                native_core_training_weight=2.0 if row['domain'] == 'ios' and row['view'] == 'native'
                    and row['family'] in CORE_WEIGHTED else 1.0)
            for index in range(start, start + count):
                if tile_rows[index] is not None:
                    raise ValueError('Overlapping row-to-tile mapping')
                tile_rows[index] = enriched
        if any(row is None for row in tile_rows):
            raise ValueError('Incomplete row-to-tile mapping')
        all_rows.extend(tile_rows); logits_blocks.append(logits)
        partition_counts[partition] = {'regions': len(rows), 'tiles': len(logits)}

    logits = np.concatenate(logits_blocks)
    maximum = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - maximum); probabilities = exp / np.sum(exp, axis=1, keepdims=True)
    predictions = np.argmax(logits, axis=1)
    targets = np.asarray([row['target'] for row in all_rows], dtype=np.int64)
    if any(families[target] != row['family'] for target, row in zip(targets, all_rows)):
        raise ValueError('TRAIN target registry mismatch')

    dimensions = {
        'partition': ['partition'], 'target_kind': ['target_kind'], 'true_family': ['family'],
        'source': ['source_font_family'], 'face': ['font_face'], 'face_weight': ['face_weight'],
        'script': ['script'], 'view': ['view'], 'quality': ['quality'],
        'true_family_source': ['family', 'source_font_family'],
        'true_family_quality': ['family', 'quality'],
        'true_family_view': ['family', 'view'],
        'true_family_script': ['family', 'script'],
        'full_available_stratum': ['partition', 'target_kind', 'family', 'source_font_family',
            'font_face', 'face_weight', 'script', 'view', 'quality',
            'sampling_view_weight', 'native_core_training_weight']}
    breakdowns = {name: grouped(all_rows, fields, targets, predictions, probabilities, families)
        for name, fields in dimensions.items()}

    focus_indices = [i for i, row in enumerate(all_rows) if row['family'] in FOCUS]
    focus = {'overall': grouped(all_rows, ['family'], targets, predictions, probabilities, families),
        'available_strata': grouped(all_rows, ['partition', 'family', 'source_font_family', 'font_face',
            'face_weight', 'script', 'view', 'quality'], targets, predictions, probabilities, families)}
    focus['overall'] = [row for row in focus['overall'] if row['family'] in FOCUS]
    focus['available_strata'] = [row for row in focus['available_strata'] if row['family'] in FOCUS]

    unknown_index = families.index('__unknown__')
    known = targets != unknown_index; unknown = ~known
    collisions = {
        'known_predicted_unknown': {'tiles': int(np.sum(known & (predictions == unknown_index))),
            'rate': float(np.mean(predictions[known] == unknown_index))},
        'unknown_predicted_named': {'tiles': int(np.sum(unknown & (predictions != unknown_index))),
            'rate': float(np.mean(predictions[unknown] != unknown_index)),
            'top_named_predictions': [{'family': family, 'tiles': n, 'rate': n / int(np.sum(unknown))}
                for family, n in Counter(families[int(value)] for value in predictions[unknown]
                    if value != unknown_index).most_common(10)]},
        'unknown_predicted_as_focus_family': {family: {'tiles': int(np.sum(unknown & (predictions == families.index(family)))),
            'rate': float(np.mean(predictions[unknown] == families.index(family)))} for family in sorted(FOCUS)},
        'focus_known_predicted_unknown': {family: {'tiles': int(np.sum((targets == families.index(family))
                & (predictions == unknown_index))),
            'rate': float(np.mean(predictions[targets == families.index(family)] == unknown_index))}
            for family in sorted(FOCUS)}}

    for path in (manifest_path, cache/'CACHE_FREEZE.json'):
        evidence[str(path)] = sha(path)
    for field in ('checkpoint', 'selection', 'training_freeze'):
        path = Path(identity['teacher'][field]['path']).resolve()
        if sha(path) != identity['teacher'][field]['sha256']:
            raise ValueError('Teacher identity changed')
        evidence[str(path)] = sha(path)

    report = {'schema': 'flux-glyph-b08-train-generalization-audit-v1',
        'scope': {'split': 'train', 'unit': 'cached tile', 'new_inference_performed': False,
            'pixel_or_tensor_image_arrays_read': False, 'cached_logits_read': True,
            'calibration_read': False, 'development_holdout_read': False, 'test_read': False,
            'interpretation_limit': 'Descriptive fit on observed TRAIN tiles only; no held-out or causal generalization claim.'},
        'teacher': identity['teacher'], 'families': families, 'partition_counts': partition_counts,
        'total_tiles': len(all_rows), 'overall': summarize(range(len(all_rows)), targets, predictions, probabilities, families),
        'breakdowns': breakdowns, 'focus_wenkai_microhei': focus, 'known_unknown_collisions': collisions,
        'notes': {'script_metadata': 'Available for paired known/new unknown captures; unavailable in original prepared TRAIN rows.',
            'face_weight': 'Parsed conservatively from font_face; unspecified means the face name did not encode weight.',
            'sampling_view_weight': VIEW_WEIGHTS,
            'native_core_training_weight': '2 only for iOS native PingFang/SF Pro/Helvetica; 1 otherwise.'}}
    output.mkdir(parents=True)
    with (output/'REPORT.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    evidence_report = {'schema': 'flux-glyph-b08-train-generalization-audit-evidence-v1',
        'inputs': evidence, 'cache_manifest_sha256': sha(manifest_path),
        'report_sha256': sha(output/'REPORT.json'), 'source_sha256': sha(Path(__file__)),
        'no_source_data_or_cache_writes': True, **report['scope']}
    with (output/'EVIDENCE.json').open('x') as stream:
        json.dump(evidence_report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'report': str(output/'REPORT.json'), 'evidence': str(output/'EVIDENCE.json'),
        'tiles': len(all_rows), 'focus': focus['overall'], 'collisions': collisions}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path,
        default=Path('artifacts/unified-font-v2/teacher-preserve-cache-v1'))
    parser.add_argument('--output', type=Path,
        default=Path('artifacts/unified-font-v3/train-generalization-audit-v1'))
    audit(parser.parse_args())
