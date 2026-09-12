#!/usr/bin/env python3
"""Prepare source-verified native region tiles for a separate binary rejector.

Labels are unknown=0, known=1. Text/script validate acquisition and isolation;
neither is an input feature. No existing test images are imported into training.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'training')]
from training.capture import prepare_captured as native
from training.capture.generate_unknown_scenes import KNOWN, UNKNOWN_SPLITS, COUNTS, canonical

SCHEMA = 'flux-glyph-native-unknown-region-data-v1'
SNAPSHOT_SHA = '6e9ce2c812be94e41e0ca492c7555c671a5bd9d6bed34bec83200076f2202a25'
SPLITS = ('train', 'calibration', 'test')

def require(value, message):
    if not value:
        raise ValueError('Unknown gate preparation: ' + message)

def sha(path):
    return native.sha(path)

def read(path):
    return json.loads(Path(path).read_text())

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def preprocessing_contract(path):
    nodes = []
    for node in ast.parse(Path(path).read_text()).body:
        if (isinstance(node, (ast.Import, ast.ImportFrom)) or
                isinstance(node, ast.FunctionDef) and node.name == 'preprocess_region' or
                isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('MAX_TILES', 'MAX_PIXELS') for t in node.targets)):
            nodes.append(ast.dump(node, include_attributes=False))
    require(any('preprocess_region' in node for node in nodes), 'missing preprocessing function')
    return nodes

def validate_family_label(family, split, label):
    require(type(label) is int and label in (0, 1), 'labels must be integer unknown=0 / known=1')
    require(split in SPLITS, 'unknown split')
    if label == 1:
        require(canonical(family) in KNOWN, 'unknown source mislabeled as known')
    else:
        require(family in UNKNOWN_SPLITS[split] and canonical(family) not in KNOWN,
                'unknown family leakage or known font mislabeled unknown')

def prepare(captures, output, *, scene_sources, snapshot):
    captures, output, scene_sources, snapshot = map(lambda p: Path(p).resolve(), (captures, output, scene_sources, snapshot))
    require(not output.exists(), 'use a new prepared directory')
    require(sha(snapshot) == SNAPSHOT_SHA, 'immutable R17 preprocessing snapshot changed')
    contract = preprocessing_contract(snapshot)
    require(contract == preprocessing_contract(ROOT/'src/flux_glyph/region_font.py'), 'current preprocessing differs from R17')
    spec = importlib.util.spec_from_file_location('unknown_data_frozen_r17_preprocessing', snapshot)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scenes_path = captures.parent/'Scenes.json'
    scenes, pages, isolation = native.load_scenes(scenes_path)
    source_manifest = read(scene_sources)
    require(sha(scenes_path) == source_manifest['scenes_sha256'], 'capture does not use frozen full scenes')
    require(scenes['gate_labels'] == source_manifest['labels'] == ['unknown', 'known'], 'wrong binary class order')
    require(scenes['gate_known_families'] == KNOWN, 'known family registry differs from R17')
    require(scenes['gate_unknown_family_splits'] == UNKNOWN_SPLITS == source_manifest['unknown_family_splits'],
            'unknown font family partition changed')
    require(dict(Counter(p['split'] for p in pages.values())) == COUNTS, 'full fixed native capture required')
    require(source_manifest['generator_sha256'] == sha(ROOT/'training/capture/generate_unknown_scenes.py'), 'scene generator changed after freeze')
    for path, expected in source_manifest['old_scenes_sha256'].items():
        require(sha(path) == expected, 'prior scene exclusion source changed')
        for page in read(path)['pages']:
            for region in page['regions']:
                # Metadata exclusion only: the old test pixels are never opened.
                old_text = native.normalized_text(region['text'])
                require(('normalized_region_text', old_text) not in isolation.values, 'old line text reused in new capture')
    labels_sha, scenes_sha, sources_sha = sha(captures), sha(scenes_path), sha(scene_sources)
    protocol_path, protocol = native.capture_protocol(captures.parent, scenes_sha)
    protocol_sha = sha(protocol_path)
    fonts = {f['id']: f for f in scenes['fonts']}
    require(len(fonts) == len(scenes['fonts']), 'duplicate font source ID')
    for font in fonts.values():
        if font['kind'] == 'asset':
            path = scene_sources.parent/'assets'/font['path']
            require(path.parent == scene_sources.parent/'assets' and sha(path) == font['sha256'], 'asset source hash differs')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.unknown-regions-', dir=output.parent) as temporary:
        stage = Path(temporary)/'prepared'
        stage.mkdir()
        metadata = {s: [] for s in SPLITS}
        tile_counts = Counter()
        rejected = Counter()
        sources, seen = [], set()
        streams = {}
        for split in SPLITS:
            (stage/split).mkdir()
            streams[split] = (stage/split/'tiles.raw').open('wb')
        try:
            with captures.open() as handle:
                for line in handle:
                    record = json.loads(line)
                    page, image_path, image, requests = native.native_source(record, scenes_sha, pages, captures.parent)
                    try:
                        require(record['page_id'] not in seen, 'duplicate source page')
                        seen.add(record['page_id'])
                        require(record['simulator_id'] == protocol['simulator_id'] and record['bundle_id'] == protocol['bundle_id'],
                                'native simulator or capture app changed')
                        split = record['split']
                        pixel_sha = native.pixels_sha(image)
                        identities = {'source_id': record['source_id'], 'source_file_sha256': record['source_sha256'],
                                      'decoded_pixel_sha256': pixel_sha, 'page_id': page['id'], 'content_group_id': page['content_group_id']}
                        for key, value in identities.items():
                            isolation.bind(key, value, split)
                        sources.append({**identities, 'split': split, 'image': str(image_path), 'source_kind': 'ios_simulator_screenshot'})
                        pairs = {}
                        for region in record['regions']:
                            request = requests[region['id']]
                            family, label = region['font_family'], request['gate_label']
                            validate_family_label(family, split, label)
                            if label == 0:
                                isolation.bind('unknown_font_family', family, split)
                                isolation.bind('unknown_font_source_sha256', fonts[request['font_id']]['sha256'], split)
                            reason = native.font_evidence(region, family, region['script'])
                            reason = reason or native.asset_evidence(region, request, fonts)
                            if reason:
                                rejected[f'{split}:{family}:native:{reason}'] += 1
                                continue
                            style = native.native_style(region, record['native'])
                            require(native.bounds(region.get('bbox'), image.size), 'invalid source region geometry')
                            require(region.get('bbox_points') == request['bbox_points'], 'requested geometry differs from native geometry')
                            pairs.setdefault(request['pair_id'], []).append((label, native.normalized_text(region['text']),
                                                                          request['font_size'], request['color'], region['script']))
                            crop = image.crop(region['bbox'])
                            crop_sha = native.pixels_sha(crop)
                            isolation.bind('region_rgb_sha256', crop_sha, split)
                            isolation.bind('normalized_region_text', native.normalized_text(region['text']), split)
                            processed = module.preprocess_region(crop)
                            if processed['status'] != 'ok' or processed.get('whole_width_covered') is not True:
                                rejected[f'{split}:{family}:preprocess:{processed["reason"]}'] += 1
                                continue
                            tiles = np.ascontiguousarray(processed['tiles'], dtype='<f4')
                            require(tiles.ndim == 4 and tiles.shape[1:] == (1, 64, 256) and 1 <= len(tiles) <= 8,
                                    'incorrect region tile contract')
                            require(np.isfinite(tiles).all() and tiles.min() >= 0 and tiles.max() <= 1, 'invalid image tile values')
                            raw = tiles.tobytes()
                            metadata[split].append({'index': len(metadata[split]), 'tile_start': tile_counts[split], 'tile_count': len(tiles),
                                'label': label, 'family': canonical(family), 'native_font_family': family,
                                'font_face': region['actual_font_postscript'], 'script_for_audit_only': region['script'],
                                'text_kind_for_audit_only': request['text_kind'], 'normalized_text_for_isolation_only': native.normalized_text(region['text']),
                                'source_id': record['source_id'], 'page_id': page['id'], 'region_id': region['id'],
                                'pair_id': request['pair_id'], 'content_group_id': page['content_group_id'], 'split': split,
                                'source_sha256': record['source_sha256'], 'decoded_pixel_sha256': pixel_sha,
                                'region_rgb_sha256': crop_sha, 'bbox': region['bbox'], 'tiles_sha256': hashlib.sha256(raw).hexdigest(),
                                'native_font_verified': True, 'whole_width_covered': True, 'ink_height_px': processed['ink_height_px'],
                                'native_font_proof': {'reason': region['reason'], 'postscript': region['actual_font_postscript'],
                                                      'font_source': region['font_source']}, **style})
                            streams[split].write(raw)
                            tile_counts[split] += len(tiles)
                        for pair in pairs.values():
                            require(len(pair) == 2 and sorted(r[0] for r in pair) == [0, 1] and pair[0][1:] == pair[1][1:],
                                    'native known/unknown pair lost equal script/style/text coverage')
                    finally:
                        image.close()
                    if len(seen) % 50 == 0:
                        print(json.dumps({'prepared_pages': len(seen), 'regions': {s: len(r) for s, r in metadata.items()}}), flush=True)
        finally:
            for stream in streams.values():
                stream.close()
        require(seen == set(pages), 'capture page registry incomplete')
        # Any native font rejection aborts the dataset, rather than making
        # font/script/style labels correlated through dropped samples.
        require(not rejected, 'native or preprocessing samples rejected: ' + str(dict(rejected)))
        splits = {}
        for split in SPLITS:
            folder = stage/split
            rows = metadata[split]
            require(len(rows) == COUNTS[split]*12, 'paired row budget not reached')
            require(Counter(r['label'] for r in rows) == {0: len(rows)//2, 1: len(rows)//2}, 'unequal binary class counts')
            require({r['family'] for r in rows if r['label'] == 1} == set(KNOWN), 'known family absent from split')
            require({r['family'] for r in rows if r['label'] == 0} == set(UNKNOWN_SPLITS[split]), 'unknown family absent from split')
            with (folder/'tiles.npy').open('wb') as dest, (folder/'tiles.raw').open('rb') as source:
                np.lib.format.write_array_header_2_0(dest, {'descr': '<f4', 'fortran_order': False, 'shape': (tile_counts[split], 1, 64, 256)})
                shutil.copyfileobj(source, dest)
            (folder/'tiles.raw').unlink()
            dump(folder/'metadata.json', {'schema': SCHEMA, 'split': split, 'labels': ['unknown', 'known'], 'rows': rows})
            splits[split] = {'regions': len(rows), 'tiles': tile_counts[split],
                             'family_counts': dict(Counter(r['family'] for r in rows)), 'label_counts': dict(Counter(r['label'] for r in rows)),
                             'array': {'path': f'{split}/tiles.npy', 'sha256': sha(folder/'tiles.npy')},
                             'metadata': {'path': f'{split}/metadata.json', 'sha256': sha(folder/'metadata.json')}}
        require(sha(captures) == labels_sha and sha(scenes_path) == scenes_sha and sha(scene_sources) == sources_sha
                and sha(protocol_path) == protocol_sha, 'capture or source contract changed during preparation')
        require(contract == preprocessing_contract(ROOT/'src/flux_glyph/region_font.py'), 'region preprocessing changed during preparation')
        manifest = {'schema': SCHEMA, 'labels': ['unknown', 'known'], 'known_families': KNOWN,
                    'unknown_family_splits': UNKNOWN_SPLITS, 'splits': splits, 'sources': sources,
                    'source_kind': 'ios_simulator_screenshot', 'image_source': 'simctl_png', 'accepted_native_verified_only': True,
                    'capture_domain': 'ios_simulator_controlled_scene', 'android_native_rendering_validated': False,
                    'ui_content_is_generated': True, 'network_inputs': ['image_tiles'], 'ocr_performed': False,
                    'script_features_used': False, 'text_features_used': False, 'character_segmentation_performed': False,
                    'input_labels': {'path': str(captures), 'sha256': labels_sha}, 'scenes': {'path': str(scenes_path), 'sha256': scenes_sha},
                    'capture_protocol': {'path': str(protocol_path), 'sha256': protocol_sha},
                    'scene_sources': {'path': str(scene_sources), 'sha256': sources_sha},
                    'preprocessing': {'function': 'preprocess_region', 'source_sha256': SNAPSHOT_SHA, 'snapshot_path': str(snapshot),
                                      'ast_contract_sha256': hashlib.sha256(json.dumps(contract).encode()).hexdigest(),
                                      'shape': [1, 64, 256], 'dtype': 'float32', 'max_tiles': 8},
                    'split_isolation': {k: 0 for k in ['source_id', 'source_file_sha256', 'decoded_pixel_sha256', 'page_id',
                                                     'content_group_id', 'region_rgb_sha256', 'normalized_region_text',
                                                     'unknown_font_family', 'unknown_font_source_sha256']},
                    'paired_known_unknown': 'identical native renderer, source page, text, script, color and size; random row order',
                    'old_test_pixels_used_for_training': False, 'user_screenshots_used_for_training': False,
                    'test_history': 'fresh capture and held-out unknown families for this gate; not proof of unseen backbone families',
                    'test_predictions_performed': False, 'rejected': dict(rejected), 'preparation_code_sha256': sha(__file__)}
        dump(stage/'MANIFEST.json', manifest)
        stage.rename(output)
    return manifest

def load_partition(directory, split, *, allow_test=False):
    """Validate provenance and load only the explicitly requested pixel partition."""
    require(split in SPLITS, 'invalid requested partition')
    require(split != 'test' or allow_test is True, 'test pixels require a separate post-selection evaluator')
    directory = Path(directory).resolve()
    manifest = read(directory/'MANIFEST.json')
    require(manifest['schema'] == SCHEMA and manifest['labels'] == ['unknown', 'known'], 'gate schema/class order differs')
    require(manifest['source_kind'] == 'ios_simulator_screenshot' and manifest['accepted_native_verified_only'] is True,
            'gate source is not verified native capture')
    require(manifest['known_families'] == KNOWN and manifest['unknown_family_splits'] == UNKNOWN_SPLITS,
            'family label registry or held-out families differ')
    require(manifest['network_inputs'] == ['image_tiles'] and manifest['ocr_performed'] is False
            and manifest['text_features_used'] is False and manifest['script_features_used'] is False,
            'gate data contract contains OCR or text inputs')
    require(manifest['preparation_code_sha256'] == sha(__file__), 'frozen preparation source changed')
    require(manifest['preprocessing']['source_sha256'] == SNAPSHOT_SHA, 'preprocessing contract changed')
    snapshot = Path(manifest['preprocessing']['snapshot_path'])
    require(sha(snapshot) == SNAPSHOT_SHA and preprocessing_contract(snapshot) == preprocessing_contract(ROOT/'src/flux_glyph/region_font.py'),
            'runtime differs from verified preprocessing snapshot')
    require(manifest['rejected'] == {} and all(v == 0 for v in manifest['split_isolation'].values()),
            'data has native rejection or source leakage')
    for key in ('input_labels', 'scenes', 'capture_protocol', 'scene_sources'):
        entry = manifest[key]
        require(sha(entry['path']) == entry['sha256'], 'source binding changed: ' + key)
    scenes, pages, _ = native.load_scenes(Path(manifest['scenes']['path']))
    registry = {font['id']: font for font in scenes['fonts']}
    source_manifest = read(manifest['scene_sources']['path'])
    require(source_manifest['scenes_sha256'] == manifest['scenes']['sha256'], 'scene source manifest differs')
    require(source_manifest['generator_sha256'] == sha(ROOT/'training/capture/generate_unknown_scenes.py'), 'scene generator changed')
    assets = Path(manifest['scene_sources']['path']).parent/'assets'
    for font in registry.values():
        if font['kind'] == 'asset':
            path = (assets/font['path']).resolve()
            require(path.parent == assets and sha(path) == font['sha256'], 'registered font source changed')
    item = manifest['splits'][split]
    paths = {}
    for key in ('array', 'metadata'):
        entry = item[key]
        path = (directory/entry['path']).resolve()
        require(path.is_relative_to(directory/split) and sha(path) == entry['sha256'], 'requested partition changed')
        paths[key] = path
    metadata = read(paths['metadata'])
    require(metadata['schema'] == SCHEMA and metadata['split'] == split and metadata['labels'] == ['unknown', 'known'],
            'partition metadata/class order differs')
    rows = metadata['rows']
    tiles = np.load(paths['array'], allow_pickle=False, mmap_mode='r')
    require(tiles.dtype == np.float32 and tiles.shape == (item['tiles'], 1, 64, 256), 'invalid gate tile contract')
    require(len(rows) == item['regions'] == COUNTS[split]*12, 'gate row budget differs')
    source_rows = {}
    selected_images_opened = 0
    captures = Path(manifest['input_labels']['path'])
    # Other splits are traversed as metadata only; their images/arrays are not opened.
    for line in captures.read_text().splitlines():
        record = json.loads(line)
        if record['split'] != split:
            continue
        page, _, image, requests = native.native_source(record, manifest['scenes']['sha256'], pages, captures.parent)
        try:
            selected_images_opened += 1
            pixel_sha = native.pixels_sha(image)
            for region in record['regions']:
                request = requests[region['id']]
                reason = native.font_evidence(region, region['font_family'], region['script']) or native.asset_evidence(region, request, registry)
                require(reason is None, 'native font/source evidence changed: ' + str(reason))
                key = (record['source_id'], region['id'])
                require(key not in source_rows, 'duplicate source/region identity')
                source_rows[key] = (record['source_sha256'], pixel_sha, region, request)
        finally:
            image.close()
    cursor, seen = 0, set()
    for i, row in enumerate(rows):
        require(row['split'] == split and row['index'] == i and row['tile_start'] == cursor and 1 <= row['tile_count'] <= 8,
                'invalid gate row/tile mapping')
        validate_family_label(row['native_font_family'], split, row['label'])
        require(row['family'] == canonical(row['native_font_family']), 'native font label differs')
        key = (row['source_id'], row['region_id'])
        require(key in source_rows and key not in seen, 'missing or duplicate source/region binding')
        seen.add(key)
        file_sha, pixel_sha, region, request = source_rows[key]
        require(row['label'] == request['gate_label'] and row['native_font_family'] == region['font_family']
                and row['font_face'] == region['actual_font_postscript'] and row['bbox'] == region['bbox'],
                'prepared label/font/geometry differs from native truth')
        require(row['source_sha256'] == file_sha and row['decoded_pixel_sha256'] == pixel_sha,
                'native source image identity differs')
        require(row['native_font_verified'] is True and row['native_font_proof']['font_source'] == region['font_source'],
                'prepared source proof changed')
        cursor += row['tile_count']
        block = tiles[row['tile_start']:cursor]
        require(np.isfinite(block).all() and block.min() >= 0 and block.max() <= 1
                and hashlib.sha256(block.tobytes()).hexdigest() == row['tiles_sha256'], 'invalid or modified gate tile bytes')
    require(cursor == len(tiles) and seen == set(source_rows), 'orphan native source/region or tile')
    require(selected_images_opened == COUNTS[split], 'incomplete requested source partition')
    return {'rows': rows, 'tiles': tiles, 'manifest': manifest,
            'audit': {'loaded_pixel_partition': split, 'source_images_verified': selected_images_opened,
                      'other_partition_pixels_opened': False, 'manifest_sha256': sha(directory/'MANIFEST.json')}}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scene-sources', required=True, type=Path)
    parser.add_argument('--snapshot', required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.captures, args.output, scene_sources=args.scene_sources, snapshot=args.snapshot)
    print(json.dumps({'splits': result['splits'], 'manifest_sha256': sha(args.output/'MANIFEST.json')}, ensure_ascii=False, indent=2))
