#!/usr/bin/env python3
"""Join verified native mobile regions into one font vocabulary, without OCR inputs.

Original datasets stay immutable. New named families previously used as unknowns
receive page/text-connected development partitions before any joint training.
Those partitions are not blind tests of historical initialization checkpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'training')]
from train_regions import dump, require, sha
from training.capture.prepare_captured import normalized_text

SCHEMA = 'flux-glyph-unified-region-training-data-v1'
UNKNOWN = '__unknown__'
FAMILIES = [
    'HarmonyOS Sans SC', 'MiSans', 'Noto Sans CJK SC', 'OPPO Sans',
    'PingFang', 'SF Pro', 'Helvetica', 'Alipay Number', 'Roboto',
    'Noto Serif CJK SC', 'LXGW WenKai', 'WenQuanYi Micro Hei',
    'ZCOOL KuaiLe', 'ZCOOL XiaoWei', 'ZCOOL QingKe HuangYou', 'Ma Shan Zheng',
    'FandolHei', 'FandolKai', 'FandolSong', 'Long Cang', 'Zhi Mang Xing',
    'Kaiti SC', 'Songti SC', 'HanziPen SC', UNKNOWN,
]
SPLITS = ('train', 'calibration', 'development_holdout')
RECIPES = ('native', 'half', 'three_quarters_jpeg75', 'jpeg75')
INPUTS = {
    'ios_views': ROOT/'artifacts/font-sans-v1/views-v1',
    'ios_unknown': ROOT/'artifacts/font-unknown-gate-v1/data-v1',
    'ios_supplement': ROOT/'artifacts/font-unknown-gate-v1/supplement-data-v2',
    'android': ROOT/'artifacts/android-font-v1/data-v3',
}
PROMOTED = FAMILIES[16:-1]
SEED = 'unified-native-pages-20260913-v1'


def family_label(name):
    if name in ('PingFang SC', 'PingFang TC', 'PingFang HK'):
        return 'PingFang'
    return name if name in FAMILIES[:-1] else UNKNOWN


def connected_partitions(pages, identities, eligible):
    """Keep every source page and repeated text/crop in a single partition."""
    parent = {page: page for page in pages}

    def find(page):
        while parent[page] != page:
            parent[page] = parent[parent[page]]
            page = parent[page]
        return page

    for members in identities.values():
        members = sorted(set(members))
        for page in members[1:]:
            a, b = find(members[0]), find(page)
            if a != b:
                parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for page in pages:
        groups[find(page)].append(page)
    assignment, excluded, audit = {}, [], []
    movable = defaultdict(list)
    for members in groups.values():
        old_splits = {pages[p]['old_split'] for p in members}
        # Old independent calibration stays separate; do not move its text into
        # training or drop only the duplicated text while keeping its page.
        if len(old_splits) > 1:
            excluded.extend(members)
            continue
        old = next(iter(old_splits))
        for page in members:
            assignment[page] = old
        domains = {pages[p]['dataset'] for p in members}
        if old == 'train' and len(domains) == 1 and next(iter(domains)) in eligible:
            movable[next(iter(domains))].append(sorted(members))
    for dataset, components in sorted(movable.items()):
        budget = round(sum(map(len, components)) * .1)
        # Large connected components remain TRAIN; never split one to meet a
        # numeric quota. Remaining components have a fixed hash ordering.
        available = sorted(components, key=lambda c: hashlib.sha256((SEED+'|'+c[0]).encode()).hexdigest())
        counts = {}
        for split in ('development_holdout', 'calibration'):
            total, taken = 0, []
            for component in available:
                if len(component) <= budget-total:
                    for page in component:
                        assignment[page] = split
                    total += len(component)
                    taken.append(component)
            available = [c for c in available if c not in taken]
            counts[split] = total
        audit.append({'dataset': dataset, 'pages': sum(map(len, components)),
                      'components': len(components), 'largest_component': max(map(len, components)),
                      'target_pages_per_development_partition': budget, 'moved_pages': counts})
    return assignment, sorted(excluded), audit


def metadata_sources(bindings):
    """Read native truth/text for identity checks only; never create model features."""
    roots = [ROOT/'artifacts/ios-hant-region-v1/data', ROOT/'artifacts/font-sans-v1/data-v1',
             INPUTS['ios_unknown'], INPUTS['ios_supplement']]
    truth = {}
    for root in roots:
        manifest = json.loads((root/'MANIFEST.json').read_text())
        bindings[str(root/'MANIFEST.json')] = sha(root/'MANIFEST.json')
        entry = manifest['input_labels']
        require(sha(entry['path']) == entry['sha256'], 'native annotations changed')
        bindings[entry['path']] = entry['sha256']
        with Path(entry['path']).open() as source:
            for line in source:
                page = json.loads(line)
                if page['split'] not in ('train', 'calibration'):
                    continue
                for region in page['regions']:
                    identity = (page['source_id'], region['id'])
                    require(identity not in truth, 'duplicate native truth')
                    actual = region.get('font_source', {})
                    truth[identity] = {'text': normalized_text(region['text']),
                        'actual_family': region['font_family'], 'font_file_sha256': actual.get('sha256', ''),
                        'font_face': region['actual_font_postscript'],
                        'font_source_kind': actual.get('kind', 'system'),
                        'source_sha256': page['source_sha256'], 'native_split': page['split'],
                        'text_color_hex': region.get('actual_text_color_hex', region.get('text_color_hex'))}
    manifest = json.loads((INPUTS['android']/'MANIFEST.json').read_text())
    labels = Path(manifest['capture'])/'labels.jsonl'
    bindings[str(labels)] = sha(labels)
    with labels.open() as source:
        for line in source:
            row = json.loads(line)
            if row['split'] not in ('train', 'calibration'):
                continue
            identity = (row['source_id'], row['region_id'])
            require(identity not in truth and row['native_font_verified'] is True, 'invalid Android native truth')
            truth[identity] = {'text': normalized_text(row['text']), 'actual_family': row['font_family'],
                'font_file_sha256': row['font_file_sha256'], 'font_face': row['font_face'],
                'font_source_kind': 'asset', 'source_sha256': row['image_sha256'],
                'native_split': row['split'], 'text_color_hex': row['color']}
    return truth


def source_partition(dataset, split):
    root = INPUTS[dataset]
    if dataset == 'ios_views':
        from training.prepare_sans_views import load_views
        return load_views(root, split)
    if dataset == 'ios_unknown':
        from training.capture.prepare_unknown_regions import load_partition
        return load_partition(root, split)
    if dataset == 'ios_supplement':
        from training.capture.prepare_rejection_supplement import load_training_supplement
        from flux_glyph.region_font import preprocess_region
        from PIL import Image
        require(split == 'train', 'supplement only has training pixels')
        data = load_training_supplement(root)
        images = {r['source_id']: r['image'] for r in data['manifest']['sources']}
        previous, image = None, None
        for row in data['rows']:
            if row['source_id'] != previous:
                if image is not None:
                    image.close()
                with Image.open(images[row['source_id']]) as opened:
                    image = opened.convert('RGB')
                previous = row['source_id']
            processed = preprocess_region(image.crop(row['bbox']))
            require(processed['status'] == 'ok' and processed['whole_width_covered']
                    and hashlib.sha256(processed['tiles'].tobytes()).hexdigest() == row['tiles_sha256'],
                    'supplement remeasured crop differs from frozen native tiles')
            row['ink_height_px'] = processed['ink_height_px']
        if image is not None:
            image.close()
        return data
    from training.prepare_android_regions import load_split as load_android
    return load_android(root, split)


def normalize_row(row, dataset, old_split, truth):
    native = truth[(row['source_id'], row['region_id'])]
    actual = row.get('source_font_family', row.get('native_font_family', row['family']))
    require(actual == native['actual_family'] and row['font_face'] == native['font_face']
            and native['native_split'] == old_split, 'prepared font differs from native truth')
    if dataset != 'android':
        require(row.get('native_font_verified') is True, 'unverified iOS region')
    size = row.get('font_size_px', row.get('font_size_screen_px'))
    height = row['ink_height_px']
    require(type(size) in (int, float) and math.isfinite(size) and size > 0
            and type(height) in (int, float) and height > 0, 'missing native pixel size')
    ratio = math.log(size/height)
    require(-3 <= ratio <= 3 and math.isclose(row.get('log_em_ratio', ratio), ratio, abs_tol=1e-9),
            'native size target differs')
    family = family_label(actual)
    crop_sha = row.get('native_crop_sha256', row.get('region_rgb_sha256'))
    require(isinstance(crop_sha, str) and len(crop_sha) == 64, 'missing native crop identity')
    return {'family': family, 'target': FAMILIES.index(family), 'source_id': row['source_id'],
        'page_id': row['page_id'], 'region_id': row['region_id'], 'source_font_family': actual,
        'font_face': native['font_face'], 'font_file_sha256': native['font_file_sha256'],
        'font_source_kind': native['font_source_kind'], 'source_sha256': native['source_sha256'],
        'domain': 'android' if dataset == 'android' else 'ios', 'source_dataset': dataset,
        'original_split': old_split, 'view': row.get('view', 'native'),
        'native_crop_sha256': crop_sha, 'normalized_text_sha256': hashlib.sha256(native['text'].encode()).hexdigest(),
        'font_size_px': size, 'ink_height_px': height, 'log_em_ratio': ratio,
        'text_color_hex': native['text_color_hex'], 'tiles_sha256': row['tiles_sha256'],
        'tile_start': row['tile_start'], 'tile_count': row['tile_count'], 'native_font_verified': True}


def prepare(output):
    output = Path(output).resolve()
    require(not output.exists(), 'unified preparation output must be new')
    bound_code = [Path(__file__), ROOT/'src/flux_glyph/region_font.py', ROOT/'training/train_regions.py',
                  ROOT/'training/prepare_sans_views.py', ROOT/'training/prepare_android_regions.py',
                  ROOT/'training/capture/prepare_unknown_regions.py',
                  ROOT/'training/capture/prepare_rejection_supplement.py']
    bindings = {str(path.resolve()): sha(path) for path in bound_code}
    truth = metadata_sources(bindings)
    sources, all_rows, pages = {}, [], {}
    identities = defaultdict(list)
    seen = set()
    # Each existing loader verifies native screenshots, font evidence and all
    # requested tensor hashes. Their TEST-pixel gates are never enabled here.
    for dataset, root in INPUTS.items():
        bindings[str(root/'MANIFEST.json')] = sha(root/'MANIFEST.json')
        for split in ('train', 'calibration'):
            if dataset == 'ios_supplement' and split != 'train':
                continue
            original = source_partition(dataset, split)
            source_key = dataset+'/'+split
            sources[source_key] = original
            part_path = root/split/'MANIFEST.json'
            if part_path.exists():
                bindings[str(part_path)] = sha(part_path)
            if part_path.exists():
                m = original.get('partition') or json.loads(part_path.read_text())
            else:
                m = original['manifest']['splits'][split]
            # Bind the actual requested array and row file, without TEST arrays.
            for entry_name in ('array', 'metadata'):
                entry = m[entry_name]
                path = (root/split/entry['path']) if part_path.exists() else root/entry['path']
                require(sha(path) == entry['sha256'], 'requested source asset differs')
                bindings[str(path)] = entry['sha256']
            for row in original['rows']:
                item = normalize_row(row, dataset, split, truth)
                identity = (item['source_id'], item['region_id'], item['view'])
                require(identity not in seen, 'duplicate source region/view in union')
                seen.add(identity)
                item['_source_key'] = source_key
                all_rows.append(item)
                page = item['source_id']
                descriptor = {'old_split': split, 'dataset': dataset}
                require(pages.setdefault(page, descriptor) == descriptor, 'page source reused inconsistently')
                for kind in ('source_sha256', 'native_crop_sha256', 'normalized_text_sha256'):
                    identities[(kind, item[kind])].append(page)
            print(json.dumps({'loaded': source_key, 'verified_views': len(original['rows']),
                              'tiles': len(original['tiles'])}), flush=True)
    assignment, excluded, audit = connected_partitions(pages, identities, {'android', 'ios_unknown'})
    selected = {split: [] for split in SPLITS}
    for row in all_rows:
        if row['source_id'] not in assignment:
            continue
        row['split'] = assignment[row['source_id']]
        row['evaluation_role'] = ('historical_train_development' if row['split'] != row['original_split']
                                  else 'original_'+row['original_split'])
        selected[row['split']].append(row)
    for split, rows in selected.items():
        require(rows and set(r['family'] for r in rows) == set(FAMILIES), 'a unified class lacks '+split+' data')
    output.mkdir(parents=True)
    # Source partition loaders already count Android preprocessing rejections.
    # They are retained as abstentions in the corresponding new page partition.
    rejected = {split: [] for split in SPLITS}
    for key, source in sources.items():
        if not key.startswith('android/'):
            continue
        for row in source['partition'].get('rejected', []):
            page = row['source_id']
            if page not in assignment:
                continue
            native = truth[(page, row['region_id'])]
            rejected[assignment[page]].append({**row, 'family': family_label(native['actual_family']),
                'source_font_family': native['actual_family'], 'domain': 'android',
                'split': assignment[page], 'original_split': key.split('/')[1]})
    manifest = {'schema': SCHEMA, 'families': FAMILIES, 'promoted_native_families': PROMOTED,
        'source_kind': 'verified_native_mobile_screenshots', 'model_inputs': ['image_tiles'],
        'ocr_performed': False, 'text_or_platform_features_used': False, 'bindings': bindings,
        'preprocessing_shape': [1, 64, 256], 'derived_views_are_correlated': True,
        'input_roots': {k: str(v) for k, v in INPUTS.items()},
        'split_policy': {'seed': SEED, 'identity_keys': ['page', 'source_pixels', 'native_crop', 'normalized_text'],
            'moved_datasets': ['android', 'ios_unknown'], 'development_page_fraction': .1,
            'giant_components': 'keep indivisible; do not exceed each development page budget',
            'cross_original_split_conflicts': 'exclude entire connected components',
            'component_audit': audit, 'excluded_pages': excluded,
            'old_test_pixels_read': False, 'new_android_test_pixels_read': False,
            'development_holdout_is_blind_test': False,
            'initialization_history': 'Historical checkpoints may have seen development-holdout images under old labels; report as development regression only.'},
        'font_label_groups': {'PingFang': ['PingFang SC', 'PingFang TC', 'PingFang HK'],
            'Noto Sans CJK SC': ['Noto Sans CJK SC', 'Source Han Sans SC'],
            'Noto Serif CJK SC': ['Noto Serif CJK SC', 'Source Han Serif SC']},
        'split_counts': {s: {'views': len(rows), 'native_regions': len({(r['source_id'],r['region_id']) for r in rows}),
            'pages': len({r['source_id'] for r in rows}), 'families': dict(Counter(r['family'] for r in rows)),
            'platforms': dict(Counter(r['domain'] for r in rows)), 'rejected': len(rejected[s])} for s, rows in selected.items()}}
    dump(output/'MANIFEST.json', manifest)
    for split, rows in selected.items():
        folder = output/split
        folder.mkdir()
        offset = 0
        with (folder/'tiles.raw').open('wb') as stream:
            for row in rows:
                original = sources[row.pop('_source_key')]['tiles']
                start, count = row['tile_start'], row['tile_count']
                block = np.ascontiguousarray(original[start:start+count], dtype='<f4')
                require(hashlib.sha256(block.tobytes()).hexdigest() == row['tiles_sha256'], 'source tensor changed')
                stream.write(block.tobytes())
                row['tile_start'] = offset
                offset += count
        dump(folder/'rows.json', rows)
        dump(folder/'MANIFEST.json', {'schema': SCHEMA, 'split': split, 'families': FAMILIES,
            'root_manifest_sha256': sha(output/'MANIFEST.json'), 'views': len(rows), 'tiles': offset,
            'shape': [offset,1,64,256], 'counts': dict(Counter(r['family'] for r in rows)),
            'array': {'path': 'tiles.raw', 'sha256': sha(folder/'tiles.raw')},
            'metadata': {'path': 'rows.json', 'sha256': sha(folder/'rows.json')}, 'rejected': rejected[split]})
        print(json.dumps({'prepared': split, **manifest['split_counts'][split]}), flush=True)
    require(all(sha(path) == digest for path,digest in bindings.items()), 'source/preparation changed during union')
    return manifest


def load_split(root, split, *, allow_holdout=False):
    require(split in SPLITS and (split != 'development_holdout' or allow_holdout is True),
            'development holdout requires explicit post-selection evaluation')
    root = Path(root).resolve()
    manifest = json.loads((root/'MANIFEST.json').read_text())
    part = json.loads((root/split/'MANIFEST.json').read_text())
    require(manifest['schema'] == part['schema'] == SCHEMA and part['split'] == split
            and manifest['families'] == part['families'] == FAMILIES
            and part['root_manifest_sha256'] == sha(root/'MANIFEST.json'), 'unified source contract differs')
    require(all(sha(path) == digest for path,digest in manifest['bindings'].items()), 'frozen unified source changed')
    for name in ('array', 'metadata'):
        path = (root/split/part[name]['path']).resolve()
        require(path.parent == root/split and sha(path) == part[name]['sha256'], 'unified partition asset changed')
    rows = json.loads((root/split/part['metadata']['path']).read_text())
    count = part['tiles']
    path = root/split/part['array']['path']
    require(path.stat().st_size == count*64*256*4 and part['shape'] == [count,1,64,256]
            and len(rows) == part['views'], 'unified partition geometry differs')
    tiles = np.memmap(path, mode='r', dtype='<f4', shape=tuple(part['shape']))
    cursor = 0
    for row in rows:
        require(row['split'] == split and row['family'] in FAMILIES and row['target'] == FAMILIES.index(row['family'])
                and row['native_font_verified'] is True and row['domain'] in ('ios', 'android')
                and row['view'] in RECIPES and row['tile_start'] == cursor
                and type(row['tile_count']) is int and 1 <= row['tile_count'] <= 8, 'invalid unified row')
        cursor += row['tile_count']
        block = tiles[row['tile_start']:cursor]
        require(np.isfinite(block).all() and np.all((block>=0)&(block<=1))
                and hashlib.sha256(block.tobytes()).hexdigest() == row['tiles_sha256'], 'invalid unified tensor')
        require(math.isclose(math.log(row['font_size_px']/row['ink_height_px']),row['log_em_ratio'],abs_tol=1e-9),
                'unified regression target differs')
    require(cursor == count and Counter(r['family'] for r in rows) == part['counts'], 'unified coverage differs')
    return {'tiles': tiles, 'rows': rows, 'families': FAMILIES, 'manifest': manifest, 'partition': part,
            'manifest_sha256': sha(root/'MANIFEST.json'), 'partition_sha256': sha(root/split/'MANIFEST.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output)
