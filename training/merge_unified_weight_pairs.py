#!/usr/bin/env python3
"""Merge verified Regular/Micro and Light/Medium paired TRAIN data without relabeling."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'flux-glyph-unified-known-weight-merged-v1'
FREEZE_SCHEMA = 'flux-glyph-unified-known-weight-merged-freeze-v1'
MAPPING_SCHEMA = 'flux-glyph-unified-known-weight-source-mapping-v1'
VIEWS = ('native', 'half', 'three_quarters_jpeg75', 'jpeg75')
OLD_FACES = {('LXGW WenKai', 'LXGWWenKai-Regular'),
             ('WenQuanYi Micro Hei', 'WenQuanYiMicroHei')}
NEW_FACES = {('LXGW WenKai', 'LXGWWenKai-Light'),
             ('LXGW WenKai', 'LXGWWenKai-Medium'),
             ('WenQuanYi Micro Hei', 'WenQuanYiMicroHei')}
KEPT_FACES = OLD_FACES | (NEW_FACES - {('WenQuanYi Micro Hei', 'WenQuanYiMicroHei')})
TARGETS = {'LXGW WenKai': 10, 'WenQuanYi Micro Hei': 11}
EXPECTED_FACE_NATIVE_REGIONS = {
    'LXGW WenKai/LXGWWenKai-Light': 120,
    'LXGW WenKai/LXGWWenKai-Medium': 120,
    'LXGW WenKai/LXGWWenKai-Regular': 240,
    'WenQuanYi Micro Hei/WenQuanYiMicroHei': 240,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def _validate_verified_source(data, allowed_faces, label):
    rows, tiles, families, part = data['rows'], data['tiles'], data['families'], data['partition']
    require(isinstance(families, list) and len(families) == len(set(families)) == 25
            and families[-1] == '__unknown__' and part.get('split') == 'train'
            and part.get('families') == families and isinstance(rows, list) and rows
            and isinstance(tiles, np.ndarray) and tiles.dtype == np.float32
            and tiles.ndim == 4 and tiles.shape[1:] == (1, 64, 256),
            label+' source is not an already-verified 25-class TRAIN tensor')
    offset = 0; identities = set(); found = set(); coverage = defaultdict(set)
    for row in rows:
        face = (row.get('family'), row.get('font_face')); count = row.get('tile_count')
        identity = (row.get('source_id'), row.get('region_id'), row.get('view'))
        require(face in allowed_faces and row.get('source_font_family') == row.get('family')
                and row.get('target') == TARGETS[row['family']]
                and row.get('split') == row.get('original_split') == 'train'
                and row.get('domain') == 'android' and row.get('native_font_verified') is True
                and row.get('source_dataset') == 'android_paired_known_supplement'
                and all(isinstance(value, str) and value for value in identity)
                and identity not in identities and row.get('tile_start') == offset
                and type(count) is int and 1 <= count <= 8 and offset+count <= len(tiles),
                label+' contains an invalid, relabeled, duplicate, or noncontiguous row')
        identities.add(identity); found.add(face); coverage[face].add(row['view']); offset += count
    require(offset == len(tiles) and found == allowed_faces
            and all(coverage[face] == set(VIEWS) for face in allowed_faces),
            label+' source does not cover its exact faces, tile order, and four views')
    return identities


def _select_verified(old, new):
    """Select rows from already-verified loaders and return exact source blocks."""
    old_ids = _validate_verified_source(old, OLD_FACES, 'old')
    new_ids = _validate_verified_source(new, NEW_FACES, 'new')
    require(old['families'] == new['families'], 'Source family registries differ')
    selected = []; excluded = []
    for label, data in (('old', old), ('new', new)):
        for index, row in enumerate(data['rows']):
            face = (row['family'], row['font_face'])
            keep = label == 'old' or face != ('WenQuanYi Micro Hei', 'WenQuanYiMicroHei')
            record = {'source': label, 'source_row_index': index,
                'source_tile_start': row['tile_start'], 'tile_count': row['tile_count'],
                'source_id': row['source_id'], 'region_id': row['region_id'], 'view': row['view'],
                'family': row['family'], 'font_face': row['font_face']}
            if keep:
                selected.append((record, row, data['tiles'][row['tile_start']:row['tile_start']+row['tile_count']]))
            else:
                excluded.append({**record, 'reason': 'duplicate_new_microhei_face_excluded'})
    kept_ids = [(row['source_id'], row['region_id'], row['view']) for _, row, _ in selected]
    old_source_ids = {row['source_id'] for row in old['rows']}
    new_source_ids = {row['source_id'] for row in new['rows']}
    old_regions = {(row['source_id'], row['region_id']) for row in old['rows']}
    new_regions = {(row['source_id'], row['region_id']) for row in new['rows']}
    require(len(kept_ids) == len(set(kept_ids)) and not old_ids.intersection(new_ids)
            and not old_source_ids.intersection(new_source_ids)
            and not old_regions.intersection(new_regions),
            'Old and new sources contain conflicting duplicate source rows')
    return selected, excluded


def _assemble_verified(old, new):
    """Pure merge for source structures that already passed the native proof loaders."""
    selected, excluded = _select_verified(old, new)
    rows = []; blocks = []; mapping = []; offset = 0
    for output_index, (source, original, block) in enumerate(selected):
        require(block.dtype == np.float32 and block.shape == (original['tile_count'], 1, 64, 256)
                and np.isfinite(block).all(), 'Invalid selected float32 source tile block')
        row = deepcopy(original); row['tile_start'] = offset
        require(all(row[key] == value for key, value in original.items() if key != 'tile_start'),
                'Merge changed source metadata')
        rows.append(row); blocks.append(block)
        mapping.append({**source, 'output_row_index': output_index, 'output_tile_start': offset})
        offset += len(block)
    tiles = np.concatenate(blocks, axis=0)
    require(tiles.dtype == np.float32 and tiles.shape == (offset, 1, 64, 256),
            'Merged tile tensor differs')
    return {'families': deepcopy(old['families']), 'rows': rows, 'tiles': tiles,
        'mapping': mapping, 'excluded': excluded}


def _source_identity(root, data):
    root = Path(root).resolve()
    paths = {'manifest': root/'MANIFEST.json', 'preparation_freeze': root/'PREPARATION_FREEZE.json',
             'partition': root/'train/MANIFEST.json', 'rows': root/'train/rows.json',
             'tiles': root/'train/tiles.raw'}
    require(all(path.is_file() for path in paths.values()), 'Verified source files are incomplete')
    identity = {name: {'path': str(path), 'sha256': sha(path)} for name, path in paths.items()}
    require(identity['manifest']['sha256'] == data['manifest_sha256']
            and identity['partition']['sha256'] == data['partition_sha256'],
            'Loaded source identity differs from its files')
    return identity


def _rejection_selection(old, new):
    kept = []; retained_audit = []; excluded_audit = []
    for label, data in (('old', old), ('new', new)):
        for index, row in enumerate(data['partition'].get('rejected', [])):
            face = (row.get('family'), row.get('font_face'))
            record = {'source': label, 'source_rejected_index': index,
                'source_id': row.get('source_id'), 'region_id': row.get('region_id'),
                'view': row.get('view'), 'family': row.get('family'),
                'font_face': row.get('font_face')}
            if label == 'old' or face != ('WenQuanYi Micro Hei', 'WenQuanYiMicroHei'):
                kept.append(deepcopy(row))
                retained_audit.append(record)
            else:
                excluded_audit.append({**record, 'reason': 'duplicate_new_microhei_face_excluded'})
    return kept, retained_audit, excluded_audit


def _merged_rejections(old, new):
    return _rejection_selection(old, new)[0]


def _coverage(rows, rejected):
    groups = defaultdict(lambda: defaultdict(set)); native = defaultdict(set)
    for status, values in (('accepted', rows), ('rejected', rejected)):
        for row in values:
            face = (row['family'], row['font_face']); region = (row['source_id'], row['region_id'])
            groups[face][region].add((row['view'], status)); native[face].add(region)
    require(set(groups) == KEPT_FACES and all(
        {view for view, _ in states} == set(VIEWS)
        for regions in groups.values() for states in regions.values()),
        'Every retained native region must account for all four requested views')
    accepted_views = defaultdict(set)
    for row in rows: accepted_views[(row['family'], row['font_face'])].add(row['view'])
    require(all(accepted_views[face] == set(VIEWS) for face in KEPT_FACES),
            'Each retained face/family needs accepted coverage in all four views')
    return {family+'/'+face: len(regions) for (family, face), regions in sorted(native.items())}


def merge(old_root, new_root, output):
    """Load both native-proof-verified roots and create one fresh merged TRAIN root."""
    from prepare_unified_known_supplement import load_supplement
    old_root, new_root, output = map(lambda value: Path(value).resolve(), (old_root, new_root, output))
    require(not output.exists() and old_root != new_root
            and output not in old_root.parents and output not in new_root.parents
            and old_root not in output.parents and new_root not in output.parents,
            'Merged output must be a separate new path')
    old, new = load_supplement(old_root), load_supplement(new_root)
    merged = _assemble_verified(old, new)
    rejected, retained_rejected, excluded_rejected = _rejection_selection(old, new)
    face_regions = _coverage(merged['rows'], rejected)
    require(face_regions == EXPECTED_FACE_NATIVE_REGIONS,
            'Expected exactly 720 retained native regions in the fixed face partition')
    identities = {'old': _source_identity(old_root, old), 'new': _source_identity(new_root, new)}
    bindings = {str(Path(__file__).resolve()): sha(__file__)}
    for data, identity in ((old, identities['old']), (new, identities['new'])):
        for item in identity.values(): bindings[item['path']] = item['sha256']
        for path, digest in data['manifest']['bindings'].items():
            require(path not in bindings or bindings[path] == digest, 'Source binding conflict')
            bindings[path] = digest
    require(all(sha(path) == digest for path, digest in bindings.items()), 'Bound merge source changed')

    output.mkdir(); (output/'train').mkdir()
    mapping = {'schema': MAPPING_SCHEMA, 'sources': identities,
        'selection': {'old': 'all Regular WenKai and MicroHei rows',
                      'new': 'Light and Medium WenKai rows only',
                      'excluded': 'all duplicated new MicroHei rows'},
        'retained_rows': merged['mapping'], 'excluded_rows': merged['excluded'],
        'retained_rejected_views': retained_rejected,
        'excluded_rejected_views': excluded_rejected,
        'relabeling_performed': False, 'augmentation_performed': False}
    dump(output/'train/SOURCE_MAPPING.json', mapping)
    dump(output/'train/rows.json', merged['rows'])
    (output/'train/tiles.raw').write_bytes(merged['tiles'].astype('<f4', copy=False).tobytes())
    freeze = {'schema': FREEZE_SCHEMA, 'families': merged['families'], 'only_split': 'train',
        'sources': identities, 'bindings': bindings, 'selection': mapping['selection'],
        'face_native_regions': face_regions, 'accepted_views': len(merged['rows']),
        'rejected_views': len(rejected), 'tiles': len(merged['tiles']),
        'old_data_modified': False, 'new_data_modified': False,
        'source_metadata_preserved_except_tile_start': True, 'exact_float32_tile_bytes_preserved': True,
        'relabeling_performed': False, 'augmentation_performed': False,
        'calibration_development_test_pixels_read': False, 'optimizer_steps': 0}
    dump(output/'PREPARATION_FREEZE.json', freeze)
    manifest = {**freeze, 'schema': SCHEMA, 'preparation_freeze_sha256': sha(output/'PREPARATION_FREEZE.json'),
        'source_kind': 'merged_verified_android_emulator_screenshots', 'model_inputs': ['image_tiles'],
        'ocr_performed': False, 'text_or_platform_features_used': False,
        'derived_views_are_correlated': True, 'native_font_verified': True}
    dump(output/'MANIFEST.json', manifest)
    counts = Counter(row['family'] for row in merged['rows'])
    faces = Counter(row['font_face'] for row in merged['rows'])
    part = {'schema': SCHEMA, 'split': 'train', 'families': merged['families'],
        'root_manifest_sha256': sha(output/'MANIFEST.json'), 'native_regions': sum(face_regions.values()),
        'views': len(merged['rows']), 'rejected_views': len(rejected), 'tiles': len(merged['tiles']),
        'shape': list(merged['tiles'].shape), 'counts': dict(counts), 'face_counts': dict(faces),
        'face_native_regions': face_regions, 'rejected': rejected,
        'array': {'path': 'tiles.raw', 'sha256': sha(output/'train/tiles.raw')},
        'metadata': {'path': 'rows.json', 'sha256': sha(output/'train/rows.json')},
        'source_mapping': {'path': 'SOURCE_MAPPING.json', 'sha256': sha(output/'train/SOURCE_MAPPING.json')}}
    dump(output/'train/MANIFEST.json', part)
    loaded = load_merged(output)
    print(json.dumps({'schema': SCHEMA, 'native_regions': part['native_regions'],
        'views': part['views'], 'rejected_views': part['rejected_views'], 'tiles': part['tiles'],
        'manifest_sha256': loaded['manifest_sha256']}, indent=2))
    return loaded


def load_merged(root):
    """Replay both native loaders and verify mapping, metadata, and every merged tile byte."""
    from prepare_unified_known_supplement import load_supplement
    root = Path(root).resolve(); manifest = read(root/'MANIFEST.json')
    freeze = read(root/'PREPARATION_FREEZE.json'); part = read(root/'train/MANIFEST.json')
    require(manifest.get('schema') == part.get('schema') == SCHEMA
            and freeze.get('schema') == FREEZE_SCHEMA and manifest.get('only_split') == part.get('split') == 'train'
            and manifest['preparation_freeze_sha256'] == sha(root/'PREPARATION_FREEZE.json')
            and all(manifest.get(key) == value for key, value in freeze.items() if key != 'schema')
            and part['root_manifest_sha256'] == sha(root/'MANIFEST.json'), 'Merged manifest/freeze differs')
    require(all(sha(path) == digest for path, digest in manifest['bindings'].items()), 'Merged source binding changed')
    for key in ('array', 'metadata', 'source_mapping'):
        require(sha(root/'train'/part[key]['path']) == part[key]['sha256'], 'Merged '+key+' changed')
    sources = {}
    for label, identity in manifest['sources'].items():
        require(all(sha(item['path']) == item['sha256'] for item in identity.values()), 'Merged source file changed')
        sources[label] = load_supplement(Path(identity['manifest']['path']).parent)
    expected = _assemble_verified(sources['old'], sources['new'])
    rejected, retained_rejected, excluded_rejected = _rejection_selection(
        sources['old'], sources['new'])
    mapping = read(root/'train'/part['source_mapping']['path']); rows = read(root/'train'/part['metadata']['path'])
    require(mapping == {'schema': MAPPING_SCHEMA, 'sources': manifest['sources'],
        'selection': freeze['selection'], 'retained_rows': expected['mapping'],
        'excluded_rows': expected['excluded'], 'relabeling_performed': False,
        'retained_rejected_views': retained_rejected,
        'excluded_rejected_views': excluded_rejected,
        'augmentation_performed': False} and rows == expected['rows'] and part['rejected'] == rejected,
        'Merged source mapping, selection, exclusion, or row metadata differs')
    path = root/'train'/part['array']['path']; count = part['tiles']
    require(path.stat().st_size == count*64*256*4 and part['shape'] == [count, 1, 64, 256],
            'Merged tensor shape/length differs')
    tiles = np.memmap(path, mode='r', dtype='<f4', shape=tuple(part['shape']))
    require(np.array_equal(tiles, expected['tiles']), 'Merged float32 tiles differ from selected source bytes')
    face_regions = _coverage(rows, rejected)
    require(part['native_regions'] == sum(face_regions.values()) == 720
            and part['views'] == len(rows) and part['rejected_views'] == len(rejected)
            and part['views']+part['rejected_views'] == 2880
            and part['face_native_regions'] == face_regions == EXPECTED_FACE_NATIVE_REGIONS
            and part['counts'] == dict(Counter(row['family'] for row in rows))
            and part['face_counts'] == dict(Counter(row['font_face'] for row in rows)),
            'Merged counts or actual view rejection accounting differs')
    return {'tiles': tiles, 'rows': rows, 'families': manifest['families'],
        'manifest': manifest, 'partition': part, 'mapping': mapping,
        'manifest_sha256': sha(root/'MANIFEST.json'), 'partition_sha256': sha(root/'train/MANIFEST.json')}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--old', type=Path, default=ROOT/'artifacts/unified-font-v2/paired-known-capture-v1/data')
    result.add_argument('--new', type=Path, default=ROOT/'artifacts/unified-font-v3/paired-wenkai-weight-capture-v1/data')
    result.add_argument('--output', type=Path, default=ROOT/'artifacts/unified-font-v3/paired-wenkai-weight-capture-v1/merged-data')
    return result


if __name__ == '__main__':
    args = parser().parse_args(); merge(args.old, args.new, args.output)
