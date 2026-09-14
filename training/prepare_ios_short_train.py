#!/usr/bin/env python3
"""Prepare verified iOS native 1--4 character regions as an isolated TRAIN set."""
from __future__ import annotations
import argparse, hashlib, json, math, tempfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from flux_glyph.region_font import preprocess_region
try:
    from capture import prepare_captured as native
    from prepare_sans_views import apply_view, RECIPES
    from train_regions import dump, sha
except ImportError:
    from training.capture import prepare_captured as native
    from training.prepare_sans_views import apply_view, RECIPES
    from training.train_regions import dump, sha
from prepare_unified_regions import FAMILIES as UNIFIED_FAMILIES

SCHEMA = 'flux-glyph-ios-native-short-train-v1'
VIEWS = ('native', 'half', 'three_quarters_jpeg75', 'jpeg75')
FAMILIES = tuple(UNIFIED_FAMILIES)
TARGET_FAMILIES = {'PingFang': FAMILIES.index('PingFang'), 'SF Pro': FAMILIES.index('SF Pro'), 'Helvetica': FAMILIES.index('Helvetica')}

def require(ok, msg):
    if not ok: raise ValueError('iOS short TRAIN: ' + msg)

def canonical(name):
    return 'PingFang' if name in ('PingFang SC', 'PingFang TC', 'PingFang HK') else name

def verified_glyphs(region, image_size):
    text, glyphs, script = region['text'], region['glyphs'], region['script']
    require(script in ('han', 'latin') and all(native.script_of(c) == script for c in text),
            'only complete unmixed classification characters are accepted')
    require(len(glyphs) == len(text) and {g.get('text_index') for g in glyphs} == set(range(len(text))),
            'incomplete native short glyph identity')
    for g in glyphs:
        require(g.get('character') == text[g['text_index']] and g.get('visible') is True
                and g.get('font_match_verified') is True and g.get('font_classification_character') is True
                and g.get('font_family') == region['font_family']
                and g.get('font_postscript') == region['actual_font_postscript']
                and type(g.get('glyph_id')) is int and g['glyph_id'] > 0,
                'native glyph font/visibility/character differs')
        require(native.bounds(g.get('bbox'), image_size) and native.contained(g['bbox'], region['bbox'])
                and g.get('bbox_screen_px', g['bbox']) == g['bbox'], 'native glyph box outside its complete region')
    return len(glyphs)


def remove_conflicting_identities(rows, tensors):
    owners, members = defaultdict(set), defaultdict(set)
    for r in rows:
        identity = (r['source_id'], r['region_id'])
        for key in (('tile', r['tiles_sha256']), ('region', r['region_rgb_sha256'])):
            owners[key].add(r['family']); members[key].add(identity)
    excluded = set().union(*(members[k] for k, values in owners.items() if len(values) > 1))
    kept = [(r, t) for r, t in zip(rows, tensors) if (r['source_id'], r['region_id']) not in excluded]
    return [r for r, _ in kept], [t for _, t in kept], len(excluded)


def prepare(labels, output):
    labels, output = Path(labels).resolve(), Path(output).resolve()
    require(labels.name == 'labels.jsonl' and labels.is_file(), 'labels.jsonl is required')
    require(not output.exists(), 'refusing to overwrite an existing output')
    for line in labels.read_text().splitlines():
        if line.strip():
            require(json.loads(line).get('split') == 'train', 'TEST/CAL rows are forbidden')
    scenes_path = labels.parent / 'Scenes.json'; protocol_path = labels.parent / 'CAPTURE_PROTOCOL.json'
    require(scenes_path.is_file() and protocol_path.is_file(), 'Scenes.json and CAPTURE_PROTOCOL.json are required')
    bindings = {str(p): sha(p) for p in (labels, scenes_path, protocol_path)}
    for relative in ('training/prepare_ios_short_train.py', 'training/capture/prepare_captured.py',
                     'training/prepare_sans_views.py', 'training/train_regions.py',
                     'training/prepare_unified_regions.py', 'src/flux_glyph/region_font.py',
                     'training/capture/capture_ios.py', 'training/capture/ios/FluxFontCapture/AppDelegate.swift'):
        path = ROOT / relative; bindings[str(path)] = sha(path)
    scenes, pages, isolation = native.load_scenes(scenes_path)
    require(all(p['split'] == 'train' for p in pages.values()), 'new scene bundle must contain TRAIN only')
    scenes_sha = sha(scenes_path); _, protocol = native.capture_protocol(labels.parent, scenes_sha)
    registry = {f['id']: f for f in scenes.get('fonts', [])}
    require(len(registry) == len(scenes.get('fonts', [])) and registry, 'duplicate or missing font registry')
    require(protocol['capture_driver_sha256'] == bindings[str(ROOT/'training/capture/capture_ios.py')],
            'capture driver source differs')
    require(protocol.get('capture_app_source_sha256') == bindings[str(ROOT/'training/capture/ios/FluxFontCapture/AppDelegate.swift')]
            and native.valid_sha(protocol.get('capture_app_executable_sha256')), 'native app source/executable proof missing')
    records = []
    for line in labels.read_text().splitlines():
        if not line.strip(): continue
        record = json.loads(line); require(record.get('split') == 'train', 'TEST/CAL rows are forbidden')
        records.append(record)
    require(records, 'capture has no TRAIN rows')
    rows, tensors, rejected = [], [], Counter(); seen_pages = set(); sources = []
    for record in records:
        page, image_path, image, requests = native.native_source(record, scenes_sha, pages, labels.parent)
        try:
            require(record['page_id'] not in seen_pages, 'duplicate capture page')
            require(record.get('simulator_id') == protocol.get('simulator_id') and record.get('bundle_id') == protocol.get('bundle_id'), 'capture protocol mismatch')
            seen_pages.add(record['page_id'])
            bindings[str(image_path)] = record['source_sha256']
            frame_path = labels.parent/'frames'/(record['page_id'] + '.json')
            require(frame_path.is_file() and json.loads(frame_path.read_text()) == record,
                    'native captured frame record missing or differs')
            bindings[str(frame_path)] = sha(frame_path)
            decoded_sha = native.pixels_sha(image)
            sources.append({'source_id': record['source_id'], 'page_id': page['id'], 'image': str(image_path),
                            'source_sha256': record['source_sha256'], 'decoded_pixel_sha256': decoded_sha,
                            'frame_path': str(frame_path), 'frame_sha256': bindings[str(frame_path)], 'split': 'train'})
            for region in record.get('regions', []):
                text = region.get('text', ''); family = canonical(region.get('font_family'))
                if family not in TARGET_FAMILIES or not 1 <= len(text) <= 4:
                    rejected['family_or_length'] += 1; continue
                reason = native.font_evidence(region, region['font_family'], region.get('script'))
                reason = reason or native.asset_evidence(region, requests[region['id']], registry)
                if reason: rejected['native:' + reason] += 1; continue
                require(native.bounds(region.get('bbox'), image.size), 'invalid native region bounds')
                count = verified_glyphs(region, image.size)
                style = native.native_style(region, record['native'])
                crop = image.crop(region['bbox']); parent_sha = native.pixels_sha(crop)
                try: isolation.bind('region_rgb_sha256', parent_sha, 'train')
                except ValueError: rejected['cross_family_or_split_pixel_identity'] += 1; continue
                pending = []
                for view in VIEWS:
                    transformed, sy, sx = apply_view(crop, view)
                    processed = preprocess_region(transformed)
                    if not isinstance(processed, dict) or processed.get('status') != 'ok' or processed.get('whole_width_covered') is not True:
                        rejected['preprocess_region_rejected'] += 1; transformed.close(); continue
                    tiles = np.ascontiguousarray(processed['tiles'], dtype='<f4')
                    if tiles.ndim != 4 or tiles.shape[1:] != (1, 64, 256) or len(tiles) < 1 or not np.isfinite(tiles).all() or tiles.min() < 0 or tiles.max() > 1:
                        rejected['invalid_tile_shape_or_range'] += 1; transformed.close(); continue
                    if len(tiles) != 1:
                        rejected['not_single_tile'] += 1; transformed.close(); continue
                    ink = processed.get('ink_height_px'); require(type(ink) in (int,float) and math.isfinite(ink) and ink > 0, 'invalid ink height')
                    ratio = math.log(style['font_size_screen_px'] * sy / ink)
                    require(math.isfinite(ratio) and abs(ratio) <= 3, 'invalid pixel em/ink ratio')
                    tile_sha = hashlib.sha256(tiles.tobytes()).hexdigest()
                    row = {'index': len(rows), 'tile_start': len(tensors), 'tile_count': 1,
                           'family': family, 'target': TARGET_FAMILIES[family], 'font_face': region['actual_font_postscript'],
                           'native_font_family': region['font_family'], 'source_id': record['source_id'], 'page_id': page['id'],
                           'region_id': region['id'], 'parent_region_id': region['id'], 'view': view,
                           'view_is_derived': view != 'native', 'split': 'train', 'source_sha256': record['source_sha256'],
                           'decoded_pixel_sha256': decoded_sha, 'region_rgb_sha256': parent_sha,
                           'normalized_text_sha256': hashlib.sha256(native.normalized_text(text).encode()).hexdigest(),
                           'source_kind': 'ios_simulator_screenshot', 'domain': 'ios',
                           'glyph_count': count, 'script': region['script'], 'text': text,
                           'native_glyph_provenance': {'font_runs': region['font_runs'], 'glyph_coverage': region['glyph_coverage'], 'glyphs': region['glyphs']},
                           'native_font_source': region.get('font_source'), 'bbox': region['bbox'],
                           'source_font_family': region['font_family'], 'source_dataset': 'ios_native_short_train_v1',
                           'frame_path': str(frame_path), 'frame_sha256': bindings[str(frame_path)],
                           'native_font_size_px': style['font_size_screen_px'], 'view_scale_y': sy, 'view_scale_x': sx,
                           'font_size_px': style['font_size_screen_px'] * sy,
                           'ink_height_px': ink, 'font_size_screen_px': style['font_size_screen_px'] * sy,
                           'log_em_ratio': ratio, 'tiles_sha256': tile_sha, 'native_font_verified': True,
                           'whole_width_covered': True, 'actual_font_size_points': style['actual_font_size_points'],
                           'source_screen_scale': style['source_screen_scale'], 'text_color_hex': style['text_color_hex'],
                           'native_style': 'actual CTFont em size; pixels = points * screen scale'}
                    pending.append((row, tiles[0])); transformed.close()
                if len(pending) != len(VIEWS):
                    rejected['incomplete_view_set'] += 1
                else:
                    for row, tile in pending:
                        rows.append(row); tensors.append(tile)
                crop.close()
        finally: image.close()
    require(seen_pages == set(pages), 'all planned TRAIN capture pages must be present')
    rows, tensors, conflicts = remove_conflicting_identities(rows, tensors)
    rejected['cross_family_conflicting_native_identities'] = conflicts
    for index, (row, tile) in enumerate(zip(rows, tensors)):
        row['index'] = index; row['tile_start'] = index
    require(rows, 'no usable native short TRAIN regions')
    groups = defaultdict(list)
    for row in rows: groups[row['source_id'], row['region_id']].append(row)
    require(all(len(group) == 4 and {r['view'] for r in group} == set(VIEWS) for group in groups.values()),
            'accepted native identities must retain complete four-view sets')
    require(all(sha(path) == digest for path, digest in bindings.items()), 'bound native source changed during preparation')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ios-short-', dir=output.parent) as tmp:
        stage = Path(tmp); np.save(stage/'tiles.npy', np.asarray(tensors, dtype='<f4'))
        dump(stage/'rows.json', {'schema': 'flux-glyph-ios-native-short-rows-v1', 'split': 'train', 'rows': rows})
        manifest = {'schema': SCHEMA, 'families': list(FAMILIES), 'target_families': TARGET_FAMILIES, 'split': 'train', 'views': list(VIEWS),
                    'tiles': len(tensors), 'views_count': len(rows), 'native_regions': len(groups),
                    'regions': len(groups), 'captured_pages': len(sources), 'labels': str(labels), 'labels_sha256': bindings[str(labels)],
                    'scenes': str(scenes_path), 'scenes_sha256': scenes_sha, 'capture_protocol_sha256': sha(protocol_path),
                    'accepted_native_verified_only': True, 'test_read': False, 'calibration_read': False,
                    'rows_sha256': sha(stage/'rows.json'), 'array_sha256': sha(stage/'tiles.npy'),
                    'preprocessing': {'function': 'flux_glyph.region_font.preprocess_region', 'shape': [1,64,256], 'dtype': 'float32'},
                    'view_recipes': {v: RECIPES[v] for v in VIEWS}, 'rejected': dict(rejected), 'source_kind': 'ios_simulator_screenshot',
                    'bindings': bindings, 'sources': sources, 'model_count': 1,
                    'independent_region_counts': dict(Counter(group[0]['family'] for group in groups.values())),
                    'native_length_counts': dict(Counter(group[0]['glyph_count'] for group in groups.values())),
                    'related_views_are_independent_samples': False, 'ocr_performed': False}
        dump(stage/'MANIFEST.json', manifest); stage.rename(output)
    return manifest

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--labels', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); print(json.dumps(prepare(args.labels, args.output), indent=2))
