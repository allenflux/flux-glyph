#!/usr/bin/env python3
"""Prepare new verified Android native 1--4 glyph regions as isolated TRAIN data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import unicodedata

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'training')]

from flux_glyph.region_font import preprocess_region
try:
    from capture.generate_android_short_scenes import validate_short_scenes
    from prepare_android_regions import CROP_RULE, inside, source_crop_bbox
    from prepare_ios_short_train import remove_conflicting_identities
    from prepare_sans_views import RECIPES, apply_view, pixels_sha
    from prepare_unified_regions import FAMILIES as UNIFIED_FAMILIES, family_label
    from train_regions import dump, sha
except ImportError:
    from training.capture.generate_android_short_scenes import validate_short_scenes
    from training.prepare_android_regions import CROP_RULE, inside, source_crop_bbox
    from training.prepare_ios_short_train import remove_conflicting_identities
    from training.prepare_sans_views import RECIPES, apply_view, pixels_sha
    from training.prepare_unified_regions import FAMILIES as UNIFIED_FAMILIES, family_label
    from training.train_regions import dump, sha


SCHEMA = 'flux-glyph-android-native-short-train-v2'
ROWS_SCHEMA = 'flux-glyph-android-native-short-rows-v2'
NAMESPACE = 'android-native-short-train-v2'
VIEWS = ('native', 'half', 'three_quarters_jpeg75', 'jpeg75')
FAMILIES = tuple(UNIFIED_FAMILIES)


def require(condition, message):
    if not condition:
        raise ValueError('Android short TRAIN: ' + message)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _script_kind(text):
    """Classify only simple, visible Han, Latin-letter, or decimal-glyph strings."""
    require(isinstance(text, str) and 1 <= len(text) <= 4, 'text must contain 1--4 glyph characters')
    require(all(not character.isspace() and unicodedata.category(character)[0] in ('L', 'N') for character in text),
            'punctuation, whitespace, marks, and symbols are forbidden')

    def han(character):
        value = ord(character)
        return (0x3400 <= value <= 0x4DBF or 0x4E00 <= value <= 0x9FFF
                or 0xF900 <= value <= 0xFAFF or 0x20000 <= value <= 0x323AF)

    if all(han(character) for character in text):
        return 'han'
    if all(character.isdecimal() for character in text):
        return 'numeric'
    if all(unicodedata.category(character).startswith('L')
           and 'LATIN' in unicodedata.name(character, '') for character in text):
        return 'latin'
    raise ValueError('Android short TRAIN: text must be unmixed Han, Latin, or numeric glyphs')


def _descriptor(path_value, digest, purpose, bindings):
    require(isinstance(path_value, str) and path_value, purpose + ' path is missing')
    path = Path(path_value).resolve()
    require(path.is_file() and valid_sha(digest) and sha(path) == digest, purpose + ' SHA differs')
    bindings[str(path)] = digest
    return path


def _font_sources(scenes, requested_font_ids, requested_texts, bindings):
    """Validate all declared native source bytes, cmaps, licenses, and TRAIN eligibility."""
    fonts = scenes.get('fonts')
    require(isinstance(fonts, list) and fonts, 'font source registry is missing')
    registry = {font.get('id'): font for font in fonts}
    require(None not in registry and len(registry) == len(fonts), 'duplicate or missing font source ID')
    require(set(registry) == set(requested_font_ids), 'declared font sources must exactly match planned requests')
    declared_mapping = scenes.get('design', {}).get('source_family_to_unified_family')
    require(isinstance(declared_mapping, dict), 'source-to-unified family plan is missing')
    require(set(declared_mapping) == {font.get('family') for font in fonts},
            'source-to-unified family plan contains absent or missing sources')

    result, coverage = [], {}
    for font_id, font in sorted(registry.items()):
        for key in ('family', 'postscript', 'path', 'sha256', 'license'):
            require(isinstance(font.get(key), str) and font[key], 'font source field is missing: ' + key)
        require(type(font.get('ttc_index')) is int and font['ttc_index'] >= 0, 'invalid source TTC index')
        require(font.get('split') in ('train', 'all') and 'train' in font.get('allowed_splits', []),
                'declared source is not eligible for TRAIN')
        canonical = family_label(font['family'])
        require(canonical in FAMILIES and declared_mapping.get(font['family']) == canonical,
                'actual source family has no matching canonical label')
        if canonical == '__unknown__':
            require(font.get('training_family') == '__unknown__' and font.get('split') == 'train'
                    and font.get('allowed_splits') == ['train'], 'held-out unknown source is forbidden')
        else:
            require(font.get('training_family') in (canonical, '__unknown__'),
                    'capture registry class conflicts with actual source family')

        font_path = _descriptor(font['path'], font['sha256'], 'font file', bindings)
        source = font.get('source')
        require(isinstance(source, dict), 'font acquisition source is missing')
        source_path = source.get('path')
        source_sha = source.get('sha256')
        if source_path is not None or source_sha is not None:
            _descriptor(source_path, source_sha, 'font acquisition source', bindings)
        cmap = font.get('cmap')
        require(isinstance(cmap, dict), 'font cmap descriptor is missing')
        cmap_path = _descriptor(cmap.get('path'), cmap.get('sha256'), 'font cmap', bindings)
        codepoints = json.loads(cmap_path.read_text())
        require(isinstance(codepoints, list) and all(type(value) is int for value in codepoints),
                'font cmap data differs')
        coverage[font_id] = frozenset(codepoints)
        for text in requested_texts[font_id]:
            require(all(ord(character) in coverage[font_id] for character in text),
                    'planned text is absent from the declared source cmap')
        licenses = font.get('license_files')
        require(isinstance(licenses, list) and licenses, 'font license proof is missing')
        license_rows = []
        for item in licenses:
            require(isinstance(item, dict), 'font license descriptor differs')
            path = _descriptor(item.get('path'), item.get('sha256'), 'font license', bindings)
            license_rows.append({'path': str(path), 'sha256': item['sha256']})
        result.append({'font_id': font_id, 'family': font['family'], 'canonical_family': canonical,
                       'postscript': font['postscript'], 'font_file': str(font_path),
                       'font_file_sha256': font['sha256'], 'ttc_index': font['ttc_index'],
                       'split': font['split'], 'allowed_splits': font['allowed_splits'],
                       'cmap': {'path': str(cmap_path), 'sha256': cmap['sha256']},
                       'license': font['license'], 'license_files': license_rows,
                       'source_path': str(Path(source_path).resolve()) if source_path else None,
                       'source_sha256': source_sha})
    return registry, coverage, result


def _validate_plan(scenes, captured_rows):
    require(scenes.get('schema') == 'flux-glyph-android-scenes-v1', 'scene schema differs')
    pages = scenes.get('pages')
    require(isinstance(pages, list) and pages, 'planned pages are missing')
    require(all(page.get('split') == 'train' for page in pages), 'new scene bundle must contain TRAIN only')
    ids = [page.get('id') for page in pages]
    require(ids == [f'{NAMESPACE}-{index:05d}' for index in range(len(pages))],
            'planned page namespace/order differs')
    planned = [(page['id'], region.get('id')) for page in pages for region in page.get('regions', [])]
    require(all(region_id == f'{page_id}-r{index:02d}'
                for page_id, page in zip(ids, pages)
                for index, region_id in enumerate(region.get('id') for region in page.get('regions', []))),
            'planned region identity/order differs')
    require(len(planned) == len(set(planned)) and planned, 'duplicate or missing planned native region')
    design = scenes.get('design')
    require(isinstance(design, dict), 'short TRAIN design proof is missing')
    declared_pages = design.get('planned_pages', design.get('pages'))
    declared_regions = design.get('planned_regions', design.get('regions'))
    require(declared_pages == len(pages) == 128 and declared_regions == len(planned) == 1056,
            'declared plan counts differ')
    require(design.get('namespace') == design.get('protocol') == NAMESPACE,
            'short TRAIN v2 namespace/protocol differs')
    require(design.get('test_read') is False and design.get('all_pages_training_only') is True,
            'sealed TEST/TRAIN-only declarations are missing')
    require(design.get('matched_size_unit') == 'actual iOS native screen pixels'
            and design.get('ios_point_to_screen_scale') == 3,
            'scale-corrected native size provenance differs')
    groups = Counter(page.get('content_group_id') for page in pages)
    require(design.get('planned_content_groups') == len(groups) == 64
            and design.get('pages_per_content_group') == 2
            and set(groups.values()) == {2}, 'planned paired-page topology differs')
    actual = [(row.get('page_id'), row.get('region_id')) for row in captured_rows]
    require(len(actual) == len(planned) and len(actual) == len(set(actual)) and set(actual) == set(planned),
            'captured labels do not exactly cover every planned page/region')
    require({row.get('page_id') for row in captured_rows} == set(ids), 'all planned TRAIN pages must be complete')
    require(all(row.get('split') == 'train' for row in captured_rows), 'TEST/CAL rows are forbidden')
    return pages, planned


def _box(value, name, image_size=None):
    require(isinstance(value, list) and len(value) == 4
            and all(type(number) in (int, float) and math.isfinite(number) for number in value)
            and value[0] < value[2] and value[1] < value[3], 'invalid ' + name)
    if image_size is not None:
        require(0 <= value[0] < value[2] <= image_size[0]
                and 0 <= value[1] < value[3] <= image_size[1], name + ' lies outside the framebuffer')
    return value


def _native_glyph_proof(capture, native, font, cmap):
    """Recheck actual shaping identity, glyph alignment, coverage, and visible bounds."""
    proof_path = inside(capture, native['proof'])
    proof = json.loads(proof_path.read_text())
    require(proof.get('schema') == 'flux-glyph-android-render-proof-v1', 'native render proof schema differs')
    matches = [item for item in proof.get('regions', []) if item.get('id') == native['region_id']]
    require(len(matches) == 1, 'native proof lacks one exact region identity')
    actual = matches[0]
    text = native['text']
    kind = _script_kind(text)
    require(native.get('script') == ('han' if kind == 'han' else 'latin'), 'native script label differs from actual text')
    require(actual.get('text') == text and actual.get('font_id') == native['font_id'],
            'actual native text/font alignment differs')
    require(all(ord(character) in cmap for character in text), 'actual text is absent from source cmap')
    region_box = _box(actual.get('bbox'), 'native region bounds', (native['image_width'], native['image_height']))
    require(region_box == native['bbox'], 'native proof region bounds differ')
    ink_box = _box(actual.get('ink_bbox'), 'native ink bounds', (native['image_width'], native['image_height']))
    require(ink_box == native['ink_bbox'] and region_box[0] <= ink_box[0] < ink_box[2] <= region_box[2]
            and region_box[1] <= ink_box[1] < ink_box[3] <= region_box[3],
            'native ink proof is incomplete or outside its region')
    glyphs = actual.get('glyphs')
    require(isinstance(glyphs, list) and len(glyphs) == len(text),
            'native glyph count does not align one-for-one with complete text')
    for glyph in glyphs:
        evidence = glyph.get('font')
        require(isinstance(evidence, dict) and glyph.get('font_verified') is True
                and type(glyph.get('glyph_id')) is int and glyph['glyph_id'] > 0
                and evidence.get('sha256') == native['font_file_sha256'] == font['sha256']
                and evidence.get('postscript') == native['font_face'] == font['postscript']
                and evidence.get('ttc_index') == native['ttc_index'] == font['ttc_index'],
                'native fallback, zero glyph, or actual font face mismatch')
        box = _box(glyph.get('bbox'), 'native glyph bounds')
        require(region_box[0] - 1 <= box[0] < box[2] <= region_box[2] + 1
                and region_box[1] - 1 <= box[1] < box[3] <= region_box[3] + 1,
                'visible native glyph lies outside its complete region')
        position = glyph.get('position')
        require(isinstance(position, list) and len(position) == 2
                and all(type(value) in (int, float) and math.isfinite(value) for value in position),
                'native glyph position is invalid')
    require(actual.get('font_verified') is True
            and type(actual.get('advance_px')) in (int, float) and math.isfinite(actual['advance_px'])
            and actual['advance_px'] > 0, 'native font/advance proof is invalid')
    return {'proof_path': proof_path, 'proof_sha256': sha(proof_path), 'glyphs': glyphs,
            'glyph_count': len(glyphs), 'text_kind': kind,
            'binding_method': proof.get('binding_method'), 'typeface_weight': actual.get('typeface_weight'),
            'advance_px': actual['advance_px']}


def _tile_sha(tiles):
    return hashlib.sha256(np.asarray(tiles, dtype='<f4').tobytes()).hexdigest()


def prepare(capture, output):
    from training.capture.capture_android import load_capture, verify_region

    capture, output = Path(capture).resolve(), Path(output).resolve()
    require(capture.is_dir(), 'capture directory is missing')
    require(not output.exists(), 'refusing to overwrite an existing output')
    source = load_capture(capture)
    scenes_path = capture / 'Scenes.json'
    require(scenes_path.is_file(), 'captured Scenes.json is missing')
    scenes = json.loads(scenes_path.read_text())
    pages, planned = _validate_plan(scenes, source['rows'])
    requests = {region['id']: (page, region) for page in pages for region in page['regions']}

    bindings = {str(Path(path).resolve()): digest for path, digest in source['bindings'].items()}
    code_paths = [Path(__file__), ROOT/'training/capture/capture_android.py',
                  ROOT/'training/capture/android/CaptureActivity.java', ROOT/'training/prepare_android_regions.py',
                  ROOT/'training/capture/generate_android_short_scenes.py',
                  ROOT/'training/prepare_ios_short_train.py', ROOT/'training/prepare_sans_views.py',
                  ROOT/'training/prepare_unified_regions.py', ROOT/'training/train_regions.py',
                  ROOT/'src/flux_glyph/region_font.py']
    code_sha256 = {}
    for path in code_paths:
        digest = sha(path)
        bindings[str(path.resolve())] = digest
        code_sha256[str(path.relative_to(ROOT))] = digest
    require(all(Path(path).is_file() and sha(path) == digest for path, digest in bindings.items()),
            'bound capture/code input differs at start')

    requested_font_ids = {region['font_id'] for page in pages for region in page['regions']}
    requested_texts = defaultdict(set)
    for page in pages:
        for region in page['regions']:
            requested_texts[region['font_id']].add(region['text'])
    registry, cmaps, font_sources = _font_sources(
        scenes, requested_font_ids, requested_texts, bindings)
    validate_short_scenes(scenes, {font_id: set(cmap) for font_id, cmap in cmaps.items()})
    require(all(Path(path).is_file() and sha(path) == digest for path, digest in bindings.items()),
            'bound source/cmap/license input differs at start')

    rows, tensors, rejected = [], [], Counter()
    sources, images = {}, {}
    try:
        for native in source['rows']:
            verify_region(capture, native)
            font = registry[native['font_id']]
            request_page, request = requests[native['region_id']]
            require(native['font_family'] == font['family'] and native['font_face'] == font['postscript']
                    and native['font_file_sha256'] == font['sha256']
                    and native['ttc_index'] == font['ttc_index'] and native.get('native_font_verified') is True,
                    'capture label differs from declared actual font source')
            require(request['font_id'] == native['font_id'] and request['text'] == native['text']
                    and request['script'] == native['script']
                    and request['font_size_px'] == native['font_size_screen_px']
                    and request['ios_font_size_points'] * scenes['design']['ios_point_to_screen_scale']
                    == native['font_size_screen_px']
                    and request_page['content_group_id'] == native['content_group_id'],
                    'scale-corrected v2 request provenance differs from native capture')
            evidence = _native_glyph_proof(capture, native, font, cmaps[native['font_id']])
            image_path = inside(capture, native['image'])
            proof_path = evidence['proof_path']
            bindings[str(image_path)] = native['image_sha256']
            bindings[str(proof_path)] = evidence['proof_sha256']
            if image_path not in images:
                with Image.open(image_path) as opened:
                    image = opened.convert('RGB')
                require(list(image.size) == [native['image_width'], native['image_height']],
                        'native framebuffer dimensions differ')
                images[image_path] = image
                sources[native['source_id']] = {
                    'source_id': native['source_id'], 'page_id': native['page_id'], 'split': 'train',
                    'image': str(image_path), 'source_sha256': native['image_sha256'],
                    'decoded_pixel_sha256': pixels_sha(image)}
            image = images[image_path]
            bbox = source_crop_bbox(native)
            crop = image.crop(bbox)
            region_sha = pixels_sha(crop)
            family = family_label(native['font_family'])
            require(family in FAMILIES and scenes['design']['source_family_to_unified_family'][native['font_family']] == family,
                    'actual source font family label differs')
            pending = []
            for view in VIEWS:
                transformed, scale_y, scale_x = apply_view(crop, view)
                try:
                    processed = preprocess_region(transformed)
                    if (not isinstance(processed, dict) or processed.get('status') != 'ok'
                            or processed.get('whole_width_covered') is not True):
                        rejected['preprocess_region_rejected'] += 1
                        continue
                    tiles = np.ascontiguousarray(processed.get('tiles'), dtype='<f4')
                    if (tiles.ndim != 4 or tiles.shape != (1, 1, 64, 256)
                            or not np.isfinite(tiles).all() or tiles.min() < 0 or tiles.max() > 1):
                        rejected['invalid_or_not_single_tile'] += 1
                        continue
                    ink = processed.get('ink_height_px')
                    require(type(ink) in (int, float) and math.isfinite(ink) and ink > 0,
                            'invalid measured view ink height')
                    size = native['font_size_screen_px'] * scale_y
                    ratio = math.log(size / ink)
                    require(math.isfinite(size) and size > 0 and math.isfinite(ratio) and abs(ratio) <= 3,
                            'invalid native pixel em/ink ratio')
                    row = {
                        'index': len(rows) + len(pending), 'tile_start': len(tensors) + len(pending), 'tile_count': 1,
                        'family': family, 'target': FAMILIES.index(family), 'font_face': native['font_face'],
                        'source_id': native['source_id'], 'region_id': native['region_id'],
                        'parent_region_id': native['region_id'], 'page_id': native['page_id'], 'view': view,
                        'view_is_derived': view != 'native', 'source_sha256': native['image_sha256'],
                        'decoded_pixel_sha256': sources[native['source_id']]['decoded_pixel_sha256'],
                        'region_rgb_sha256': region_sha, 'tiles_sha256': _tile_sha(tiles),
                        'glyph_count': evidence['glyph_count'], 'script': native['script'], 'text': native['text'],
                        'text_kind': evidence['text_kind'], 'split': 'train', 'domain': 'android',
                        'source_kind': 'android_emulator_screenshot', 'source_dataset': NAMESPACE,
                        'content_group_id': native['content_group_id'],
                        'paired_style_id': request['paired_style_id'],
                        'request_origin': request['request_origin'],
                        'source_font_family': native['font_family'],
                        'capture_training_family': native['training_family'], 'font_id': native['font_id'],
                        'font_file_sha256': native['font_file_sha256'], 'ttc_index': native['ttc_index'],
                        'native_font_verified': True, 'native_glyph_provenance': {
                            'binding_method': evidence['binding_method'], 'glyphs': evidence['glyphs'],
                            'advance_px': evidence['advance_px'], 'typeface_weight': evidence['typeface_weight']},
                        'frame_path': str(proof_path), 'frame_sha256': evidence['proof_sha256'],
                        'proof_path': str(proof_path), 'proof_sha256': evidence['proof_sha256'],
                        'image_path': str(image_path), 'source_crop_bbox': bbox, 'bbox': native['bbox'],
                        'ink_bbox': native['ink_bbox'], 'native_font_size_px': native['font_size_screen_px'],
                        'ios_font_size_points': request['ios_font_size_points'],
                        'matched_ios_font_size_screen_px': request['font_size_px'],
                        'ios_source_page_id': request['ios_source_page_id'],
                        'ios_source_region_id': request['ios_source_region_id'],
                        'ios_capture_source_id': request['ios_capture_source_id'],
                        'ios_capture_source_sha256': request['ios_capture_source_sha256'],
                        'ios_capture_frame_path': request['ios_capture_frame_path'],
                        'ios_capture_frame_sha256': request['ios_capture_frame_sha256'],
                        'ios_actual_font_postscript': request['ios_actual_font_postscript'],
                        'ios_source_screen_scale': request['ios_source_screen_scale'],
                        'ios_actual_bbox_pixels': request['ios_actual_bbox_pixels'],
                        'ios_actual_ink_bbox_pixels': request['ios_actual_ink_bbox_pixels'],
                        'view_scale_y': scale_y, 'view_scale_x': scale_x, 'font_size_px': size,
                        'ink_height_px': ink, 'log_em_ratio': ratio, 'text_color_hex': native['color'],
                        'background_color_hex': native['background'], 'whole_width_covered': True,
                        'native_style': 'actual Android Canvas pixel text size; view scale uses transformed raster dimensions'}
                    pending.append((row, tiles[0]))
                finally:
                    transformed.close()
            if len(pending) != len(VIEWS):
                rejected['incomplete_view_set'] += 1
            else:
                for row, tile in pending:
                    rows.append(row)
                    tensors.append(tile)
            crop.close()
    finally:
        for image in images.values():
            image.close()

    rows, tensors, conflicts = remove_conflicting_identities(rows, tensors)
    rejected['cross_family_conflicting_native_identities'] = conflicts
    for index, row in enumerate(rows):
        row['index'] = index
        row['tile_start'] = index
    require(rows, 'no usable verified native short TRAIN regions')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['source_id'], row['region_id'])].append(row)
    require(all(len(group) == len(VIEWS) and {row['view'] for row in group} == set(VIEWS)
                for group in groups.values()), 'accepted identities must retain complete four-view sets')
    array = np.asarray(tensors, dtype='<f4')
    require(array.shape == (len(rows), 1, 64, 256) and np.isfinite(array).all()
            and np.all((array >= 0) & (array <= 1)), 'final TRAIN array shape/range differs')
    require(all(Path(path).is_file() and sha(path) == digest for path, digest in bindings.items()),
            'bound source/capture/code changed at end')

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.android-short-', dir=output.parent) as temporary:
        stage = Path(temporary)
        np.save(stage/'tiles.npy', array)
        dump(stage/'rows.json', {'schema': ROWS_SCHEMA, 'split': 'train', 'rows': rows})
        manifest = {
            'schema': SCHEMA, 'families': list(FAMILIES), 'split': 'train', 'views': list(VIEWS),
            'tiles': len(rows), 'views_count': len(rows), 'native_regions': len(groups),
            'planned_native_regions': len(planned), 'captured_pages': len(pages),
            'rows_sha256': sha(stage/'rows.json'), 'array_sha256': sha(stage/'tiles.npy'),
            'shape': [len(rows), 1, 64, 256], 'dtype': 'float32',
            'capture': str(capture), 'capture_manifest_sha256': source['manifest_sha256'],
            'scenes_sha256': sha(scenes_path), 'bindings': bindings, 'code_sha256': code_sha256,
            'font_sources': font_sources, 'sources': list(sources.values()),
            'accepted_native_verified_only': True, 'calibration_read': False, 'test_read': False,
            'preprocessing': {'function': 'flux_glyph.region_font.preprocess_region',
                              'shape': [1, 64, 256], 'dtype': 'float32'},
            'source_crop': CROP_RULE, 'view_recipes': {view: RECIPES[view] for view in VIEWS},
            'size_target': 'log(native pixel font size * actual vertical view scale / measured view ink height)',
            'independent_region_counts': dict(Counter(group[0]['family'] for group in groups.values())),
            'native_length_counts': dict(Counter(group[0]['glyph_count'] for group in groups.values())),
            'rejected': dict(rejected), 'related_views_are_independent_samples': False,
            'ocr_performed': False, 'text_features_used': False,
            'binding_closure': {'verified_at_start': True, 'verified_at_end': True,
                                'bound_files': len(bindings)}}
        dump(stage/'MANIFEST.json', manifest)
        require(not output.exists(), 'refusing to overwrite an existing output')
        stage.rename(output)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.capture, arguments.output), ensure_ascii=False, indent=2))
