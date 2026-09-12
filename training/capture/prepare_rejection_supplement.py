#!/usr/bin/env python3
"""Prepare and verify a separate native, training-only handwriting supplement."""
from __future__ import annotations

import argparse
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
from training.capture.prepare_unknown_regions import require, read, dump, sha, preprocessing_contract, SNAPSHOT_SHA
from training.capture.generate_rejection_supplement import BASE, KNOWN_LATIN, NEW_UNKNOWN, PAGES, FROZEN_DATA_SHA

SCHEMA = 'flux-glyph-native-rejection-supplement-data-v1'

def validate_label(family, label, split='train'):
    require(split == 'train', 'supplement may not contain calibration or test rows')
    require(type(label) is int and label in (0, 1), 'binary label must be integer unknown=0/known=1')
    require(family in (KNOWN_LATIN if label else NEW_UNKNOWN), 'supplement family/label mismatch')

def source_contract(captures, scene_sources, snapshot):
    captures, scene_sources, snapshot = map(lambda p: Path(p).resolve(), (captures, scene_sources, snapshot))
    require(sha(snapshot) == SNAPSHOT_SHA and preprocessing_contract(snapshot) == preprocessing_contract(ROOT/'src/flux_glyph/region_font.py'),
            'R17 preprocessing contract changed')
    sources = read(scene_sources)
    require(sources['schema'] == 'flux-glyph-rejection-supplement-scene-source-v1' and sources['all_pages_training_only'] is True,
            'wrong supplement scene source schema')
    require(sources['old_data_manifest_sha256'] == sha(BASE/'data-v1/MANIFEST.json') == FROZEN_DATA_SHA, 'original gate data changed')
    require(sources['generator_sha256'] == sha(ROOT/'training/capture/generate_rejection_supplement.py')
            and sources['derivation_helper_sha256'] == sha(ROOT/'training/capture/generate_unknown_scenes.py'), 'scene generation code changed')
    scenes_path = captures.parent/'Scenes.json'
    require(sha(scenes_path) == sources['scenes_sha256'], 'supplement scene hash differs')
    scenes, pages, isolation = native.load_scenes(scenes_path)
    require(len(pages) == PAGES and all(p['split'] == 'train' for p in pages.values()), 'supplement must contain exactly the fixed training pages')
    require(scenes['gate_labels'] == ['unknown', 'known'] and scenes['gate_known_families'] == KNOWN_LATIN
            and scenes['training_only_unknown_families'] == NEW_UNKNOWN, 'supplement class registry changed')
    excluded_families = set(sources['excluded_calibration_test_families'])
    require(not set(NEW_UNKNOWN) & excluded_families, 'held-out calibration/test font moved into training')
    excluded_texts = set()
    for path, expected in sources['excluded_scene_sources'].items():
        require(sha(path) == expected, 'excluded scene source changed')
        excluded_texts.update(native.normalized_text(r['text']) for p in read(path)['pages'] for r in p['regions'])
    require(not excluded_texts & {native.normalized_text(r['text']) for p in pages.values() for r in p['regions']},
            'old training/calibration/test text reused in supplement')
    fonts = {f['id']: f for f in scenes['fonts']}
    require(len(fonts) == len(scenes['fonts']), 'duplicate source font ID')
    assets = scene_sources.parent/'assets'
    for f in fonts.values():
        if f['kind'] == 'asset':
            path = (assets/f['path']).resolve()
            require(path.parent == assets and sha(path) == f['sha256'], 'native asset source changed')
    protocol_path, protocol = native.capture_protocol(captures.parent, sha(scenes_path))
    return {'captures': captures, 'scene_sources': scene_sources, 'snapshot': snapshot, 'scenes_path': scenes_path,
            'pages': pages, 'fonts': fonts, 'isolation': isolation, 'protocol_path': protocol_path, 'protocol': protocol,
            'bindings': {key: {'path': str(p), 'sha256': sha(p)} for key, p in [('input_labels', captures), ('scenes', scenes_path),
                        ('scene_sources', scene_sources), ('capture_protocol', protocol_path), ('snapshot', snapshot)]}}

def validated_regions(context):
    """Yield verified native regions and crops from new training images only."""
    seen = set()
    for line in context['captures'].read_text().splitlines():
        record = json.loads(line)
        require(record['split'] == 'train', 'non-training image in supplement')
        page, image_path, image, requests = native.native_source(record, context['bindings']['scenes']['sha256'], context['pages'], context['captures'].parent)
        try:
            require(page['id'] not in seen, 'duplicate supplement page')
            seen.add(page['id'])
            require(record['simulator_id'] == context['protocol']['simulator_id'] and record['bundle_id'] == context['protocol']['bundle_id'],
                    'capture app/simulator changed')
            pixel_sha = native.pixels_sha(image)
            pairs = {}
            for region in record['regions']:
                request = requests[region['id']]
                validate_label(region['font_family'], request['gate_label'], record['split'])
                require(region['script'] == 'latin' and request['language'] == 'en', 'supplement must retain paired Latin script')
                reason = native.font_evidence(region, region['font_family'], 'latin') or native.asset_evidence(region, request, context['fonts'])
                require(reason is None, 'native supplement source rejected: ' + str(reason))
                require(native.bounds(region['bbox'], image.size) and region['bbox_points'] == request['bbox_points'], 'native geometry changed')
                pairs.setdefault(request['pair_id'], []).append((request['gate_label'], request['text'], request['font_size'], request['color']))
                yield record, region, request, image.crop(region['bbox']), pixel_sha
            require(len(pairs) == 6, 'supplement page does not have six pairs')
            for pair in pairs.values():
                require(len(pair) == 2 and sorted(x[0] for x in pair) == [0, 1] and pair[0][1:] == pair[1][1:], 'unmatched known/unknown style/text pair')
        finally:
            image.close()
    require(seen == set(context['pages']), 'supplement capture is incomplete')

def prepare(captures, output, *, scene_sources, snapshot):
    output = Path(output).resolve()
    require(not output.exists(), 'supplement data output must be new')
    context = source_contract(captures, scene_sources, snapshot)
    spec = importlib.util.spec_from_file_location('supplement_r17_preprocessing', context['snapshot'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.rejection-supplement-', dir=output.parent) as temp:
        stage = Path(temp)/'prepared'
        folder = stage/'train'
        folder.mkdir(parents=True)
        rows, cursor, sources = [], 0, {}
        with (folder/'tiles.raw').open('wb') as stream:
            for record, region, request, crop, pixel_sha in validated_regions(context):
                processed = module.preprocess_region(crop)
                require(processed['status'] == 'ok' and processed['whole_width_covered'], 'supplement preprocessing rejected a pair member')
                tiles = np.ascontiguousarray(processed['tiles'], dtype='<f4')
                require(tiles.ndim == 4 and tiles.shape[1:] == (1, 64, 256) and 1 <= len(tiles) <= 8,
                        'wrong supplement tile geometry')
                require(np.isfinite(tiles).all() and tiles.min() >= 0 and tiles.max() <= 1, 'invalid supplement tile values')
                raw = tiles.tobytes()
                rows.append({'index': len(rows), 'tile_start': cursor, 'tile_count': len(tiles), 'label': request['gate_label'],
                             'family': region['font_family'], 'native_font_family': region['font_family'], 'font_face': region['actual_font_postscript'],
                             'source_id': record['source_id'], 'page_id': record['page_id'], 'region_id': region['id'],
                             'pair_id': request['pair_id'], 'split': 'train', 'content_group_id': record['page_id'],
                             'source_sha256': record['source_sha256'], 'decoded_pixel_sha256': pixel_sha,
                             'region_rgb_sha256': native.pixels_sha(crop), 'bbox': region['bbox'],
                             'tiles_sha256': hashlib.sha256(raw).hexdigest(), 'native_font_verified': True,
                             'native_font_proof': {'postscript': region['actual_font_postscript'], 'font_source': region['font_source']},
                             'script_for_audit_only': 'latin', 'text_kind_for_audit_only': request['text_kind'],
                             'normalized_text_for_isolation_only': native.normalized_text(region['text']),
                             **native.native_style(region, record['native'])})
                stream.write(raw)
                cursor += len(tiles)
                sources[record['source_id']] = {'source_id': record['source_id'], 'source_sha256': record['source_sha256'],
                                               'decoded_pixel_sha256': pixel_sha, 'split': 'train', 'image': record['image']}
        require(len(rows) == PAGES*12 and Counter(r['label'] for r in rows) == {0: PAGES*6, 1: PAGES*6}, 'supplement paired budget differs')
        require({r['family'] for r in rows if r['label'] == 0} == set(NEW_UNKNOWN)
                and {r['family'] for r in rows if r['label'] == 1} == set(KNOWN_LATIN), 'font coverage missing')
        with (folder/'tiles.npy').open('wb') as dest, (folder/'tiles.raw').open('rb') as source:
            np.lib.format.write_array_header_2_0(dest, {'descr': '<f4', 'fortran_order': False, 'shape': (cursor, 1, 64, 256)})
            shutil.copyfileobj(source, dest)
        (folder/'tiles.raw').unlink()
        dump(folder/'metadata.json', {'schema': SCHEMA, 'split': 'train', 'labels': ['unknown', 'known'], 'rows': rows})
        manifest = {'schema': SCHEMA, 'source_kind': 'ios_simulator_screenshot', 'image_source': 'simctl_png',
                    'accepted_native_verified_only': True, 'training_only': True, 'labels': ['unknown', 'known'],
                    'known_families': KNOWN_LATIN, 'unknown_families': NEW_UNKNOWN, 'old_data_manifest_sha256': FROZEN_DATA_SHA,
                    'splits': {'train': {'regions': len(rows), 'tiles': cursor,
                              'label_counts': dict(Counter(r['label'] for r in rows)), 'family_counts': dict(Counter(r['family'] for r in rows)),
                              'array': {'path': 'train/tiles.npy', 'sha256': sha(folder/'tiles.npy')},
                              'metadata': {'path': 'train/metadata.json', 'sha256': sha(folder/'metadata.json')}}},
                    'sources': list(sources.values()), **context['bindings'],
                    'preparation_code_sha256': sha(__file__), 'preparation_helper_sha256': sha(ROOT/'training/capture/prepare_unknown_regions.py'),
                    'network_inputs': ['image_tiles'], 'ocr_performed': False, 'script_features_used': False, 'text_features_used': False,
                    'old_test_pixels_used_for_training': False, 'prior_calibration_test_font_overlap': 0, 'prior_text_overlap': 0,
                    'android_native_rendering_validated': False, 'rejected': {}}
        for bound in context['bindings'].values():
            require(sha(bound['path']) == bound['sha256'], 'supplement source changed during preparation')
        dump(stage/'MANIFEST.json', manifest)
        stage.rename(output)
    return manifest

def load_training_supplement(directory):
    """Strong loader with no option to consume this source as CAL or test."""
    directory = Path(directory).resolve()
    manifest = read(directory/'MANIFEST.json')
    require(manifest['schema'] == SCHEMA and manifest['training_only'] is True and set(manifest['splits']) == {'train'},
            'supplement is not the declared training-only source')
    require(manifest['labels'] == ['unknown', 'known'] and manifest['known_families'] == KNOWN_LATIN
            and manifest['unknown_families'] == NEW_UNKNOWN, 'supplement family/label contract differs')
    require(manifest['preparation_code_sha256'] == sha(__file__)
            and manifest['preparation_helper_sha256'] == sha(ROOT/'training/capture/prepare_unknown_regions.py'), 'frozen supplement preparation source changed')
    context = source_contract(manifest['input_labels']['path'], manifest['scene_sources']['path'], manifest['snapshot']['path'])
    require(all(context['bindings'][key] == manifest[key] for key in context['bindings']), 'supplement source binding changed')
    item, paths = manifest['splits']['train'], {}
    for key in ('array', 'metadata'):
        path = (directory/item[key]['path']).resolve()
        require(path.is_relative_to(directory/'train') and sha(path) == item[key]['sha256'], 'supplement data file changed')
        paths[key] = path
    metadata = read(paths['metadata'])
    require(metadata['schema'] == SCHEMA and metadata['split'] == 'train' and metadata['labels'] == ['unknown', 'known'], 'wrong supplement row schema')
    rows, tiles = metadata['rows'], np.load(paths['array'], allow_pickle=False, mmap_mode='r')
    require(len(rows) == item['regions'] == PAGES*12 and tiles.dtype == np.float32
            and tiles.shape == (item['tiles'], 1, 64, 256), 'supplement row/array budget differs')
    proofs = {}
    for record, region, request, crop, pixel_sha in validated_regions(context):
        key = (record['source_id'], region['id'])
        require(key not in proofs, 'duplicate native supplement source identity')
        proofs[key] = (record, region, request, pixel_sha, native.pixels_sha(crop))
    cursor, seen = 0, set()
    for i, row in enumerate(rows):
        key = (row['source_id'], row['region_id'])
        require(key in proofs and key not in seen and row['index'] == i and row['tile_start'] == cursor
                and 1 <= row['tile_count'] <= 8, 'supplement source/row/tile mapping differs')
        seen.add(key)
        record, region, request, pixel_sha, crop_sha = proofs[key]
        validate_label(row['family'], row['label'], row['split'])
        require(row['label'] == request['gate_label'] and row['family'] == row['native_font_family'] == region['font_family']
                and row['font_face'] == region['actual_font_postscript'], 'supplement prepared label differs from native truth')
        require(row['source_sha256'] == record['source_sha256'] and row['decoded_pixel_sha256'] == pixel_sha
                and row['region_rgb_sha256'] == crop_sha and row['bbox'] == region['bbox'], 'supplement source image differs')
        require(row['native_font_verified'] is True and row['native_font_proof']['font_source'] == region['font_source'], 'supplement font proof differs')
        cursor += row['tile_count']
        block = tiles[row['tile_start']:cursor]
        require(np.isfinite(block).all() and block.min() >= 0 and block.max() <= 1
                and hashlib.sha256(block.tobytes()).hexdigest() == row['tiles_sha256'], 'supplement tile values changed')
    require(cursor == len(tiles) and seen == set(proofs), 'orphan supplement source or tile')
    return {'rows': rows, 'tiles': tiles, 'manifest': manifest,
            'audit': {'loaded_pixel_partition': 'train', 'source_images_verified': PAGES,
                      'calibration_or_test_pixels_opened': False, 'manifest_sha256': sha(directory/'MANIFEST.json')}}

def load_partition(directory):
    """Trainer API: this source has one immutable training-only partition."""
    return load_training_supplement(directory)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scene-sources', required=True, type=Path)
    parser.add_argument('--snapshot', required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.captures, args.output, scene_sources=args.scene_sources, snapshot=args.snapshot)
    print(json.dumps({'splits': result['splits'], 'manifest_sha256': sha(args.output/'MANIFEST.json')}, ensure_ascii=False, indent=2))
