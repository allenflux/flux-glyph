#!/usr/bin/env python3
"""Create split-preserving image views of verified native font regions.

The root manifest is immutable. Preparing another split creates a new split
directory without rewriting any earlier training or calibration provenance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from flux_glyph.region_font import preprocess_region
from training.train_regions import RegionData, sha, dump

SCHEMA = 'flux-glyph-native-sans-views-v1'
SPLITS = ('train', 'calibration', 'test')
SEED = 2026091301
ANCHOR_PER_FAMILY = 384
RECIPES = {
    'native': {'scale': 1.0, 'resampling': None, 'jpeg_quality': None},
    'half': {'scale': .5, 'resampling': 'PIL.LANCZOS', 'jpeg_quality': None},
    'three_quarters_jpeg75': {'scale': .75, 'resampling': 'PIL.LANCZOS', 'jpeg_quality': 75},
    'jpeg75': {'scale': 1.0, 'resampling': None, 'jpeg_quality': 75},
}


def require(condition, message):
    if not condition:
        raise ValueError('Native sans views: ' + message)


def pixels_sha(image):
    return hashlib.sha256(f'RGB:{image.width}x{image.height}'.encode() + b'\0' + image.tobytes()).hexdigest()


def apply_view(image, view):
    """Return RGB pixels and actual vertical/horizontal raster scale factors."""
    require(view in RECIPES, 'unknown view recipe')
    recipe = RECIPES[view]
    size = tuple(max(1, round(value * recipe['scale'])) for value in image.size)
    result = image.convert('RGB')
    if size != image.size:
        result = result.resize(size, Image.Resampling.LANCZOS)
    if recipe['jpeg_quality'] is not None:
        stream = io.BytesIO()
        result.save(stream, format='JPEG', quality=recipe['jpeg_quality'], subsampling=2, optimize=False)
        stream.seek(0)
        with Image.open(stream) as compressed:
            result = compressed.convert('RGB')
    return result, result.height / image.height, result.width / image.width


def selection_evidence(path):
    require(path is not None, 'test preparation requires --frozen-selection before any test data is opened')
    path = Path(path).resolve()
    require(path.is_file(), 'frozen selection is missing')
    value = json.loads(path.read_text())
    require((value.get('passed') is True or value.get('calibration_passed') is True)
            and value.get('test_read') is False, 'frozen selection must pass calibration and declare test_read=false')
    return {'path': str(path), 'sha256': sha(path)}


def _open_dataset(directory, snapshot):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'MANIFEST.json').read_text())
    current_sha = sha(ROOT / 'src/flux_glyph/region_font.py')
    return RegionData(directory, preprocessing_snapshot=(None if manifest['preprocessing']['source_sha256'] == current_sha
                                                         else snapshot))


def _validate_cross_source_splits(datasets):
    bindings = {}
    for data in datasets.values():
        for source in data.sources.values():
            for key in ('source_id', 'source_sha256', 'decoded_pixel_sha256', 'page_id', 'content_group_id'):
                identity = (key, source[key])
                require(bindings.setdefault(identity, source['split']) == source['split'],
                        'native source/content crosses dataset splits: ' + key)
        # Identity metadata only: no held-out tile arrays or test image pixels
        # are opened here. This also catches identical crops across domains.
        for split in SPLITS:
            descriptor = data.manifest['splits'][split]['metadata']
            path = data.directory / descriptor['path']
            require(sha(path) == descriptor['sha256'], 'source region identity metadata changed')
            for row in json.loads(path.read_text())['rows']:
                require(row['split'] == split, 'source region metadata split differs')
                identity = ('region_rgb_sha256', row['region_rgb_sha256'])
                require(bindings.setdefault(identity, split) == split, 'native crop pixels cross dataset splits')


def select_rows(rows, split, domain):
    require(split in SPLITS and domain in ('anchor', 'new_native'), 'invalid source selection')
    if split != 'train' or domain != 'anchor':
        return list(rows)
    rng = np.random.default_rng(SEED)
    selected = []
    for family in sorted({row['family'] for row in rows}):
        pool = sorted((row for row in rows if row['family'] == family), key=lambda r: (r['source_id'], r['region_id']))
        indices = rng.permutation(len(pool))[:ANCHOR_PER_FAMILY]
        selected.extend(pool[int(i)] for i in indices)
    return sorted(selected, key=lambda row: (row['source_id'], row['region_id']))


def _root_manifest(datasets, snapshot):
    families = list(datasets['new_native'].families)
    require(len(families) == 9 and len(set(families)) == 9 and 'Roboto' in families,
            'new native dataset must contain nine ordered families including Roboto')
    require(set(datasets['anchor'].families) <= set(families), 'anchor family is absent from the new registry')
    paths = [Path(__file__), ROOT / 'src/flux_glyph/region_font.py', ROOT / 'training/train_regions.py']
    return {'schema': SCHEMA, 'families': families,
            'inputs': {name: {'path': str(data.directory), 'manifest_sha256': data.manifest_sha}
                       for name, data in datasets.items()},
            'snapshot': {'path': str(Path(snapshot).resolve()), 'sha256': sha(snapshot)},
            'splits': {split: {'manifest': f'{split}/MANIFEST.json'} for split in SPLITS},
            'view_recipes': RECIPES, 'seed': SEED, 'anchor_train_per_family': ANCHOR_PER_FAMILY,
            'selection': 'train anchor: seeded family selection without replacement; new native and evaluation: all rows',
            'code_sha256': {str(path.relative_to(ROOT)): sha(path) for path in paths},
            'libraries': {'numpy': np.__version__, 'pillow': PILLOW_VERSION},
            'preprocessing': {'function': 'flux_glyph.region_font.preprocess_region', 'shape': [1, 64, 256],
                              'source_sha256': sha(ROOT / 'src/flux_glyph/region_font.py')},
            'model_inputs': ['image_tiles'], 'ocr_performed': False, 'text_features_used': False,
            'script_features_used': False, 'platform_features_used': False,
            'source_kind': 'derived_from_verified_ios_simulator_screenshots',
            'size_target': 'log(native_font_size_screen_px * actual_vertical_resize_factor / measured_view_ink_height)',
            'split_policy': 'all derived views inherit original source partition; no cross-split source or crop identities',
            'test_policy': 'test pixels and arrays require a passed frozen calibration selection; identity metadata only may be audited earlier'}


def prepare_views(data, anchor, snapshot, output, split, frozen_selection=None):
    require(split in SPLITS, 'invalid partition')
    frozen = selection_evidence(frozen_selection) if split == 'test' else None
    output = Path(output).resolve()
    require(not (output / split).exists(), 'partition already exists; overwriting is forbidden')
    datasets = {'anchor': _open_dataset(anchor, snapshot), 'new_native': _open_dataset(data, snapshot)}
    _validate_cross_source_splits(datasets)
    expected = _root_manifest(datasets, snapshot)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        with tempfile.TemporaryDirectory(prefix='.sans-views-root-', dir=output.parent) as temporary:
            stage = Path(temporary) / 'root'
            stage.mkdir()
            dump(stage / 'MANIFEST.json', expected)
            stage.rename(output)
    require(output.is_dir() and json.loads((output / 'MANIFEST.json').read_text()) == expected,
            'existing root manifest differs; create a new output directory')
    root_sha = sha(output / 'MANIFEST.json')
    lock = output / '.prepare.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError('Native sans views: another partition preparation is active') from exc
    os.close(descriptor)
    try:
        require(not (output / split).exists(), 'partition already exists; overwriting is forbidden')
        with tempfile.TemporaryDirectory(prefix='.sans-views-', dir=output) as temporary:
            stage = Path(temporary) / split
            stage.mkdir()
            rows, sources, rejects, selected_sources, rejected_rows = [], {}, Counter(), {}, []
            tile_count = 0
            with (stage / 'tiles.raw').open('wb') as stream:
                for domain, dataset in datasets.items():
                    partition = dataset.load(split)
                    selected = select_rows(partition['rows'], split, domain)
                    selected_sources[domain] = {'native_rows': len(selected),
                        'family_counts': dict(Counter(row['family'] for row in selected)),
                        'partition': dataset.manifest['splits'][split]}
                    ordered = sorted(selected, key=lambda row: (row['source_id'], row['region_id']))
                    image, image_path = None, None
                    try:
                        for original in ordered:
                            source = dataset.sources[original['source_id']]
                            require(source['split'] == split, 'selected source crosses partition')
                            if image_path != source['image']:
                                if image is not None:
                                    image.close()
                                image_path = source['image']
                                require(sha(image_path) == source['source_sha256'], 'source PNG bytes changed')
                                with Image.open(image_path) as opened:
                                    image = opened.convert('RGB')
                                require(pixels_sha(image) == source['decoded_pixel_sha256'], 'source decoded pixels changed')
                            sources[(domain, source['source_id'])] = {'domain': domain, **source}
                            crop = image.crop(original['bbox'])
                            require(pixels_sha(crop) == original['region_rgb_sha256'], 'native crop pixels changed')
                            for view in RECIPES:
                                transformed, factor, factor_x = apply_view(crop, view)
                                prepared = preprocess_region(transformed)
                                if prepared.get('status') != 'ok' or prepared.get('whole_width_covered') is not True:
                                    rejects[f'{domain}:{view}:{prepared.get("reason", "unknown")}'] += 1
                                    rejected_rows.append({'domain': domain, 'view': view, 'source_id': original['source_id'],
                                                          'region_id': original['region_id'], 'reason': prepared.get('reason', 'unknown')})
                                    continue
                                tiles = np.ascontiguousarray(prepared['tiles'], dtype='<f4')
                                require(tiles.ndim == 4 and tiles.shape[1:] == (1, 64, 256)
                                        and 1 <= len(tiles) <= 8 and np.isfinite(tiles).all()
                                        and tiles.min() >= 0 and tiles.max() <= 1, 'invalid prepared view tensor')
                                size = original['font_size_screen_px'] * factor
                                log_ratio = math.log(size / prepared['ink_height_px'])
                                require(math.isfinite(log_ratio) and -3 <= log_ratio <= 3, 'invalid transformed size target')
                                raw = tiles.tobytes()
                                row = {key: original[key] for key in ('family', 'font_face', 'source_id', 'page_id', 'region_id',
                                    'content_group_id', 'source_sha256', 'decoded_pixel_sha256', 'region_rgb_sha256', 'bbox')}
                                row.update(index=len(rows), tile_start=tile_count, tile_count=len(tiles),
                                    target=expected['families'].index(original['family']), split=split, domain=domain, view=view,
                                    native_font_family=original.get('native_font_family', original['family']),
                                    native_font_verified=True, whole_width_covered=True,
                                    native_font_size_screen_px=original['font_size_screen_px'], font_size_screen_px=size,
                                    resize_factor=factor, resize_factor_x=factor_x, requested_resize_factor=RECIPES[view]['scale'],
                                    native_crop_size=list(crop.size), view_size=list(transformed.size),
                                    view_rgb_sha256=pixels_sha(transformed), ink_height_px=prepared['ink_height_px'],
                                    log_em_ratio=log_ratio, tiles_sha256=hashlib.sha256(raw).hexdigest())
                                rows.append(row)
                                stream.write(raw)
                                tile_count += len(tiles)
                    finally:
                        if image is not None:
                            image.close()
                    print(json.dumps({'split': split, 'domain': domain, 'native_regions': len(selected),
                                      'view_regions_so_far': len(rows), 'tiles_so_far': tile_count}), flush=True)
            require(rows and all(any(row['domain'] == domain and row['view'] == view for row in rows)
                                 for domain in datasets for view in RECIPES), 'partition lacks a required source/view stratum')
            with (stage / 'tiles.npy').open('wb') as destination, (stage / 'tiles.raw').open('rb') as incoming:
                np.lib.format.write_array_header_2_0(destination, {'descr': '<f4', 'fortran_order': False,
                                                               'shape': (tile_count, 1, 64, 256)})
                shutil.copyfileobj(incoming, destination, length=8 << 20)
            (stage / 'tiles.raw').unlink()
            dump(stage / 'metadata.json', {'schema': SCHEMA, 'split': split, 'rows': rows})
            partition_manifest = {'schema': SCHEMA, 'split': split, 'root_manifest_sha256': root_sha,
                'families': expected['families'], 'rows': len(rows), 'tiles': tile_count,
                'family_counts': dict(Counter(row['family'] for row in rows)),
                'stratum_counts': dict(Counter(row['domain'] + ':' + row['view'] for row in rows)),
                'array': {'path': 'tiles.npy', 'sha256': sha(stage / 'tiles.npy')},
                'metadata': {'path': 'metadata.json', 'sha256': sha(stage / 'metadata.json')},
                'selected_sources': selected_sources, 'sources': list(sources.values()),
                'rejected_views': dict(rejects), 'rejected_rows': rejected_rows,
                'test_pixels_opened': split == 'test', 'frozen_selection': frozen}
            for dataset in datasets.values():
                require(sha(dataset.manifest_path) == dataset.manifest_sha, 'input dataset manifest changed during preparation')
            require(sha(output / 'MANIFEST.json') == root_sha and _root_manifest(datasets, snapshot) == expected,
                    'view code or provenance changed during preparation')
            if frozen:
                require(sha(frozen['path']) == frozen['sha256'], 'frozen selection changed during preparation')
            dump(stage / 'MANIFEST.json', partition_manifest)
            stage.rename(output / split)
        return partition_manifest
    finally:
        lock.unlink()


def load_views(output, split):
    require(split in SPLITS, 'invalid partition')
    output = Path(output).resolve()
    root_path = output / 'MANIFEST.json'
    manifest = json.loads(root_path.read_text())
    require(manifest.get('schema') == SCHEMA and manifest.get('view_recipes') == RECIPES
            and set(manifest.get('splits', {})) == set(SPLITS), 'view manifest contract differs')
    require(manifest.get('model_inputs') == ['image_tiles'] and all(manifest.get(key) is False for key in
            ('ocr_performed', 'text_features_used', 'script_features_used', 'platform_features_used')), 'non-image input contract')
    families = manifest['families']
    require(len(families) == 9 and len(set(families)) == 9 and 'Roboto' in families, 'invalid family registry')
    for relative, expected in manifest['code_sha256'].items():
        require((ROOT / relative).resolve().is_relative_to(ROOT) and sha(ROOT / relative) == expected, 'view preparation code changed')
    snapshot = manifest['snapshot']
    require(sha(snapshot['path']) == snapshot['sha256'], 'preprocessing snapshot changed')
    source_maps, input_rows, source_bindings, input_partitions = {}, {}, {}, {}
    for domain, descriptor in manifest['inputs'].items():
        directory = Path(descriptor['path'])
        require(sha(directory / 'MANIFEST.json') == descriptor['manifest_sha256'], 'input dataset manifest changed')
        value = json.loads((directory / 'MANIFEST.json').read_text())
        input_partitions[domain] = value['splits'][split]
        source_maps[domain] = {source['source_id']: source for source in value['sources']}
        for source in value['sources']:
            for key in ('source_id', 'source_sha256', 'decoded_pixel_sha256', 'page_id', 'content_group_id'):
                require(source_bindings.setdefault((key, source[key]), source['split']) == source['split'],
                        'input source identities cross partitions')
        item = value['splits'][split]['metadata']
        require(sha(directory / item['path']) == item['sha256'], 'input region metadata changed')
        original_rows = json.loads((directory / item['path']).read_text())['rows']
        input_rows[domain] = {(row['source_id'], row['region_id']): row for row in original_rows}
        require(len(input_rows[domain]) == len(original_rows), 'duplicate source region identity')
    require(set(source_maps) == {'anchor', 'new_native'}, 'invalid source domains')
    relative = manifest['splits'][split]['manifest']
    require(relative == f'{split}/MANIFEST.json', 'partition manifest path differs')
    partition_path = output / relative
    part = json.loads(partition_path.read_text())
    require(part.get('schema') == SCHEMA and part.get('split') == split and part.get('families') == families
            and part.get('root_manifest_sha256') == sha(root_path), 'partition manifest differs')
    require(part.get('test_pixels_opened') is (split == 'test'), 'test access marker differs')
    if split == 'test':
        frozen = part.get('frozen_selection') or {}
        require(frozen == selection_evidence(frozen.get('path')), 'frozen test selection changed')
    expected_identities, expected_sources = set(), {}
    for domain, originals in input_rows.items():
        selected = select_rows(list(originals.values()), split, domain)
        summary = part['selected_sources'][domain]
        require(summary['native_rows'] == len(selected) and summary['family_counts'] == dict(Counter(row['family'] for row in selected))
                and summary['partition'] == input_partitions[domain], 'native source selection summary differs')
        for row in selected:
            for view in RECIPES:
                expected_identities.add((domain, row['source_id'], row['region_id'], view))
            source = source_maps[domain][row['source_id']]
            expected_sources[(domain, row['source_id'])] = {'domain': domain, **source}
    require({(source['domain'], source['source_id']): source for source in part['sources']} == expected_sources
            and len(part['sources']) == len(expected_sources), 'selected native source ledger differs')
    for source in expected_sources.values():
        require(source['split'] == split and sha(source['image']) == source['source_sha256'], 'selected native source PNG changed')
    paths = {}
    for kind, filename in (('array', 'tiles.npy'), ('metadata', 'metadata.json')):
        item = part[kind]
        require(item['path'] == filename, 'partition asset path differs')
        path = partition_path.parent / filename
        require(sha(path) == item['sha256'], 'partition asset SHA differs: ' + kind)
        paths[kind] = path
    metadata = json.loads(paths['metadata'].read_text())
    require(metadata.get('schema') == SCHEMA and metadata.get('split') == split, 'view row metadata contract differs')
    rows = metadata['rows']
    tiles = np.load(paths['array'], mmap_mode='r', allow_pickle=False)
    require(len(rows) == part['rows'] and tiles.dtype == np.float32 and tiles.shape == (part['tiles'], 1, 64, 256),
            'view array shape/count differs')
    next_tile, identities = 0, set()
    for index, row in enumerate(rows):
        require(row['index'] == index and row['split'] == split and row['tile_start'] == next_tile
                and type(row['tile_count']) is int and 1 <= row['tile_count'] <= 8, 'view tile indexing differs')
        require(row['family'] in families and row['target'] == families.index(row['family'])
                and row['domain'] in source_maps and row['view'] in RECIPES and row.get('native_font_verified') is True
                and row.get('whole_width_covered') is True, 'view family/label/source contract differs')
        require(not any(key in row for key in ('text', 'script', 'tokens', 'characters', 'platform')), 'view metadata contains model text/platform inputs')
        source = source_maps[row['domain']].get(row['source_id'])
        original = input_rows[row['domain']].get((row['source_id'], row['region_id']))
        require(source is not None and source['split'] == split and original is not None and original['split'] == split,
                'view source belongs to a different partition')
        for key in ('family', 'font_face', 'region_rgb_sha256', 'bbox', 'page_id', 'content_group_id',
                    'source_sha256', 'decoded_pixel_sha256'):
            require(row[key] == original[key], 'view native truth/source differs: ' + key)
        require(row['native_font_family'] == original.get('native_font_family', original['family'])
                and row['native_font_size_screen_px'] == original['font_size_screen_px'], 'view native font/size truth differs')
        identity = (row['domain'], row['source_id'], row['region_id'], row['view'])
        require(identity not in identities, 'duplicate native region view')
        identities.add(identity)
        native_size = row['native_crop_size']
        expected_size = [max(1, round(v * RECIPES[row['view']]['scale'])) for v in native_size]
        require(native_size == [row['bbox'][2] - row['bbox'][0], row['bbox'][3] - row['bbox'][1]]
                and row['view_size'] == expected_size and row['requested_resize_factor'] == RECIPES[row['view']]['scale']
                and row['resize_factor'] == expected_size[1] / native_size[1]
                and row['resize_factor_x'] == expected_size[0] / native_size[0], 'view resize geometry differs')
        require(math.isfinite(row['log_em_ratio']) and -3 <= row['log_em_ratio'] <= 3 and row['ink_height_px'] > 0
                and math.isclose(row['font_size_screen_px'], original['font_size_screen_px'] * row['resize_factor'], abs_tol=1e-9)
                and math.isclose(row['log_em_ratio'], math.log(row['font_size_screen_px'] / row['ink_height_px']), abs_tol=1e-9),
                'view size regression target differs')
        next_tile += row['tile_count']
        block = np.asarray(tiles[row['tile_start']:next_tile])
        require(np.isfinite(block).all() and block.min() >= 0 and block.max() <= 1
                and hashlib.sha256(block.tobytes()).hexdigest() == row['tiles_sha256'], 'view tensor pixels/hash differ')
    require(next_tile == len(tiles) and dict(Counter(row['family'] for row in rows)) == part['family_counts']
            and dict(Counter(row['domain'] + ':' + row['view'] for row in rows)) == part['stratum_counts'],
            'view summary counts differ')
    rejected_identities = set()
    for row in part['rejected_rows']:
        identity = (row['domain'], row['source_id'], row['region_id'], row['view'])
        require(identity not in identities and identity not in rejected_identities and identity in expected_identities
                and isinstance(row['reason'], str) and row['reason'], 'invalid rejected native view identity')
        rejected_identities.add(identity)
    require(identities | rejected_identities == expected_identities
            and dict(Counter(row['domain'] + ':' + row['view'] + ':' + row['reason'] for row in part['rejected_rows'])) == part['rejected_views'],
            'selected native views are missing or duplicated')
    return {'split': split, 'tiles': tiles, 'rows': rows, 'targets': np.asarray([row['target'] for row in rows], dtype=np.int64),
            'log_em_ratio': np.asarray([row['log_em_ratio'] for row in rows], dtype=np.float32), 'families': families,
            'manifest': manifest, 'manifest_sha256': sha(root_path), 'partition_manifest': part,
            'partition_manifest_sha256': sha(partition_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=SPLITS, required=True)
    parser.add_argument('--frozen-selection', type=Path)
    args = parser.parse_args()
    result = prepare_views(args.data, args.anchor, args.snapshot, args.output, args.split, args.frozen_selection)
    print(json.dumps({key: result[key] for key in ('split', 'rows', 'tiles', 'family_counts', 'stratum_counts', 'rejected_views')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
