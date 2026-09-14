#!/usr/bin/env python3
"""Build TRAIN-only 1--4 glyph crops from existing verified native regions.

Planning reads only unified TRAIN metadata plus matching TRAIN annotations/proofs.
It never reads source images. Materialization is a separate explicit operation.
The result is derived training augmentation, not an independent native capture.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unicodedata

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'training')]
from flux_glyph.region_font import preprocess_region
from prepare_sans_views import RECIPES, apply_view, pixels_sha
from prepare_unified_regions import FAMILIES, UNKNOWN, family_label
from train_regions import dump, require, sha

SCHEMA = 'flux-glyph-unified-short-region-supplement-v1'
PLAN_SCHEMA = 'flux-glyph-unified-short-region-plan-v1'
SEED = 'unified-short-region-train-20260914-v1'
MAX_PER_FAMILY_LENGTH = 64
MAX_PER_FAMILY = 256
MAX_SUBCROPS = 6400
MAX_OUTPUT_BYTES = 2_000_000_000
EXPECTED_NATIVE_IDENTITIES = 19803
VIEWS = tuple(RECIPES)
HELD_OUT_UNKNOWN = {'Yusei Magic', 'M PLUS 1p'}
ALLOWED_UNKNOWN_TRAIN = {
    'Apple Chancery', 'Bradley Hand', 'Courier', 'Lato', 'Liu Jian Mao Cao',
    'Noteworthy', 'Open Sans', 'Smiley Sans', 'Times',
}
DEFAULT_TRAIN = ROOT / 'artifacts/unified-font-v1/data-v1/train'
DEFAULT_OUTPUT = ROOT / 'artifacts/unified-font-v4/short-region-supplement-v2'
DEFAULT_SOURCES = {
    'ios_hant': ROOT / 'artifacts/mobile-font-capture/ios-hant-capture-v1/labels.jsonl',
    'ios_sans': ROOT / 'artifacts/font-sans-v1/capture-v1/labels.jsonl',
    'ios_unknown': ROOT / 'artifacts/font-unknown-gate-v1/capture-v1/labels.jsonl',
    'ios_supplement': ROOT / 'artifacts/font-unknown-gate-v1/supplement-capture-v2/labels.jsonl',
    'android': ROOT / 'artifacts/android-font-v1/capture-v3/labels.jsonl',
}
TRAIN_MARKER = re.compile(rb'"split"\s*:\s*"train"')
HEX64 = re.compile(r'[0-9a-f]{64}')


class CandidateRejected(Exception):
    """A verified parent that cannot prove an unambiguous short glyph mapping."""


def read(path):
    return json.loads(Path(path).read_text())


def text_sha(text):
    normalized = ''.join(unicodedata.normalize('NFKC', text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def is_han(character):
    value = ord(character)
    return (0x3400 <= value <= 0x4dbf or 0x4e00 <= value <= 0x9fff
            or 0xf900 <= value <= 0xfaff or 0x20000 <= value <= 0x3134f)


def classification_character(character):
    return is_han(character) or character.isascii() and character.isalnum()


def tile_sha(tiles):
    return hashlib.sha256(np.ascontiguousarray(tiles, dtype='<f4').tobytes()).hexdigest()


def inside_repo(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), 'source path is missing or outside repository')
    return path


def train_jsonl(path):
    """Decode only raw JSONL records bearing a top-level TRAIN marker."""
    with Path(path).open('rb') as source:
        for raw in source:
            if not TRAIN_MARKER.search(raw):
                continue
            value = json.loads(raw)
            if value.get('split') == 'train':
                yield value


def box(value, width, height, message='invalid glyph bbox'):
    require(isinstance(value, list) and len(value) == 4
            and all(type(item) in (int, float) and math.isfinite(item) for item in value), message)
    result = [math.floor(value[0]), math.floor(value[1]), math.ceil(value[2]), math.ceil(value[3])]
    require(0 <= result[0] < result[2] <= width and 0 <= result[1] < result[3] <= height, message)
    return result


def intersects(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def safe_crop(selected, unselected, parent, width, height):
    union = [min(item[0] for item in selected), min(item[1] for item in selected),
             max(item[2] for item in selected), max(item[3] for item in selected)]
    if not (0 <= parent[0] <= union[0] < union[2] <= parent[2] <= width
            and 0 <= parent[1] <= union[1] < union[3] <= parent[3] <= height):
        return None, None
    for padding in (2, 1, 0):
        candidate = [max(parent[0], union[0] - padding), max(parent[1], union[1] - padding),
                     min(parent[2], union[2] + padding), min(parent[3], union[3] + padding)]
        if (0 <= candidate[0] < candidate[2] <= width and 0 <= candidate[1] < candidate[3] <= height
                and not any(intersects(candidate, other) for other in unselected)):
            return candidate, padding
    return None, None


def rank(candidate_id):
    return int(hashlib.sha256((SEED + '|' + candidate_id).encode()).hexdigest(), 16)


def unified_train(path):
    path = Path(path).resolve()
    manifest_path, rows_path = path / 'MANIFEST.json', path / 'rows.json'
    manifest = read(manifest_path)
    require(manifest.get('split') == 'train' and manifest.get('families') == FAMILIES,
            'unified TRAIN class/split contract differs')
    require(manifest.get('metadata', {}).get('path') == 'rows.json'
            and manifest['metadata']['sha256'] == sha(rows_path), 'unified TRAIN rows binding differs')
    rows = read(rows_path)
    require(len(rows) == manifest['views'] and all(row.get('split') == 'train' for row in rows),
            'unified TRAIN row coverage differs')
    parents = {}
    views = Counter()
    fields = ('family', 'target', 'source_font_family', 'font_face', 'font_file_sha256',
              'source_sha256', 'domain', 'source_dataset', 'page_id', 'normalized_text_sha256')
    for row in rows:
        identity = (row['source_id'], row['region_id'])
        item = {key: row[key] for key in fields}
        item.update(source_id=row['source_id'], region_id=row['region_id'])
        require(row.get('native_font_verified') is True and row['target'] == FAMILIES.index(row['family']),
                'unverified or mislabeled unified TRAIN row')
        require(parents.setdefault(identity, item) == item, 'derived views disagree on native parent truth')
        views[identity] += 1
    require(len(parents) == EXPECTED_NATIVE_IDENTITIES,
            f'expected frozen {EXPECTED_NATIVE_IDENTITIES:,} unified TRAIN native identities')
    return parents, {
        str(manifest_path): sha(manifest_path), str(rows_path): sha(rows_path),
    }, dict(Counter(views.values()))


def same_font(parent, postscript, source_sha):
    return postscript == parent['font_face'] and (source_sha or '') == parent['font_file_sha256']


def ios_parent(parent, outer, region, label_path, label_sha):
    width, height = outer.get('pixel_size', [None, None])
    require(type(width) is int and type(height) is int and width > 0 and height > 0, 'invalid iOS image size')
    require(outer.get('split') == 'train' and region.get('status') == 'ok'
            and region.get('font_match_verified') is True and region.get('fallback_detected') is False,
            'iOS parent lacks accepted exact-font proof')
    require(region.get('font_family') == parent['source_font_family']
            and region.get('actual_font_postscript') == parent['font_face']
            and outer.get('source_sha256') == parent['source_sha256'], 'iOS native/unified truth differs')
    require(family_label(region.get('font_family')) == parent['family'], 'iOS unified family mapping differs')
    source_sha = region.get('font_source', {}).get('sha256', '') or ''
    require(source_sha == parent['font_file_sha256'], 'iOS font source SHA differs')
    text = region.get('text')
    require(isinstance(text, str) and text, 'iOS parent text is empty')
    require(text_sha(text) == parent['normalized_text_sha256'], 'iOS native text hash differs')
    if not all(ord(character) <= 0xffff for character in text):
        raise CandidateRejected('ios_non_bmp_text')
    image = inside_repo(outer['image'])
    parent_box = box(region.get('bbox_pixels', region.get('bbox')), width, height, 'invalid iOS parent bbox')
    glyphs = region.get('glyphs')
    require(isinstance(glyphs, list) and glyphs, 'iOS glyph proof missing')
    prepared = []
    previous = -1
    font_ids = set()
    for proof_index, glyph in enumerate(glyphs):
        index = glyph.get('text_index')
        if not (type(index) is int and 0 <= index < len(text) and index > previous
                and glyph.get('character') == text[index]):
            raise CandidateRejected('ios_ambiguous_glyph_text_index')
        previous = index
        require(type(glyph.get('glyph_id')) is int and glyph['glyph_id'] != 0
                and glyph.get('font_match_verified') is True, 'iOS glyph lacks exact-font proof')
        glyph_source_sha = glyph.get('font_source_sha256') or ''
        require(same_font(parent, glyph.get('font_postscript'), glyph_source_sha), 'iOS glyph font differs')
        font_ids.add((glyph.get('font_postscript'), glyph_source_sha))
        if glyph.get('visible') is True:
            glyph_box = box(glyph.get('bbox_screen_px', glyph.get('bbox')), width, height)
            prepared.append({'proof_index': proof_index, 'text_index': index, 'character': text[index],
                             'glyph_id': glyph['glyph_id'], 'bbox': glyph_box})
    require(len(font_ids) == 1, 'iOS parent contains multiple proven fonts')
    return {
        **parent, 'image': str(image), 'image_width': width, 'image_height': height,
        'parent_bbox': parent_box, 'native_font_size_px': region['font_size_screen_px'],
        'text_color_hex': region.get('actual_text_color_hex'), 'font_source_kind': region.get('font_source', {}).get('kind', 'ios_system'),
        'ttc_index': 0, 'glyphs': prepared, 'full_visible_bboxes': [item['bbox'] for item in prepared],
        'proof': str(label_path), 'proof_sha256': label_sha, 'mapping': 'ios_utf16_bmp_text_index',
    }


def android_parent(parent, label, label_path, proof_cache):
    width, height = label.get('image_width'), label.get('image_height')
    require(type(width) is int and type(height) is int and width > 0 and height > 0, 'invalid Android image size')
    require(label.get('split') == 'train' and label.get('native_font_verified') is True,
            'Android parent lacks native font verification')
    require(label.get('font_family') == parent['source_font_family'], 'Android source family differs')
    require(family_label(label.get('font_family')) == parent['family'], 'Android unified family mapping differs')
    require(label.get('font_face') == parent['font_face'], 'Android font face differs')
    require(label.get('font_file_sha256') == parent['font_file_sha256'], 'Android font file differs')
    require(label.get('image_sha256') == parent['source_sha256'], 'Android source image differs')
    text = label.get('text')
    require(isinstance(text, str) and text, 'Android parent text is empty')
    require(text_sha(text) == parent['normalized_text_sha256'], 'Android native text hash differs')
    if not all(ord(character) <= 0xffff for character in text):
        raise CandidateRejected('android_non_bmp_text')
    root = label_path.parent
    image = inside_repo(root / label['image'])
    proof_path = inside_repo(root / label['proof'])
    if proof_path not in proof_cache:
        proof_cache[proof_path] = (read(proof_path), sha(proof_path))
    proof, proof_sha = proof_cache[proof_path]
    require(proof.get('split') == 'train' and proof.get('page_id') == label['page_id'],
            'Android proof is not the matching TRAIN page')
    actual = next((item for item in proof.get('regions', []) if item.get('id') == label['region_id']), None)
    require(actual is not None and actual.get('text') == text and actual.get('font_id') == label['font_id']
            and actual.get('font_verified') is True, 'Android proof/label identity differs')
    glyphs = actual.get('glyphs')
    if not isinstance(glyphs, list) or len(glyphs) != len(text):
        raise CandidateRejected('android_bmp_text_glyph_count_mismatch')
    prepared, font_ids, previous_x = [], set(), -math.inf
    for index, (character, glyph) in enumerate(zip(text, glyphs)):
        evidence = glyph.get('font', {})
        require(type(glyph.get('glyph_id')) is int and glyph['glyph_id'] != 0
                and glyph.get('font_verified') is True
                and evidence.get('sha256') == label['font_file_sha256']
                and evidence.get('postscript') == label['font_face']
                and evidence.get('ttc_index') == label.get('ttc_index', 0), 'Android glyph font proof differs')
        position = glyph.get('position')
        if not (isinstance(position, list) and len(position) == 2 and type(position[0]) in (int, float)
                and math.isfinite(position[0]) and position[0] >= previous_x):
            raise CandidateRejected('android_glyph_order_cannot_be_proven')
        previous_x = position[0]
        glyph_box = box(glyph.get('bbox'), width, height)
        font_ids.add((evidence['postscript'], evidence['sha256'], evidence['ttc_index']))
        prepared.append({'proof_index': index, 'text_index': index, 'character': character,
                         'glyph_id': glyph['glyph_id'], 'bbox': glyph_box})
    require(len(font_ids) == 1, 'Android parent contains multiple proven fonts')
    parent_box = box(label.get('bbox'), width, height, 'invalid Android parent bbox')
    require(all(parent_box[0] <= item['bbox'][0] < item['bbox'][2] <= parent_box[2]
                and parent_box[1] <= item['bbox'][1] < item['bbox'][3] <= parent_box[3] for item in prepared),
            'Android glyph escapes parent region')
    return {
        **parent, 'image': str(image), 'image_width': width, 'image_height': height,
        'parent_bbox': parent_box, 'native_font_size_px': label['font_size_screen_px'],
        'text_color_hex': label.get('color'), 'font_source_kind': 'asset',
        'ttc_index': label.get('ttc_index', 0), 'glyphs': prepared,
        'full_visible_bboxes': [item['bbox'] for item in prepared],
        'proof': str(proof_path), 'proof_sha256': proof_sha,
        'mapping': 'android_bmp_equal_text_glyph_count_monotonic_proof_order_filter',
    }


def native_parents(parents, sources, bindings):
    matched = {}
    proof_cache = {}
    for source_name, source_path in sources.items():
        source_path = inside_repo(source_path)
        label_sha = sha(source_path)
        bindings[str(source_path)] = label_sha
        for record in train_jsonl(source_path):
            if source_name.startswith('ios_'):
                source_id = record.get('source_id')
                wanted = [region for region in record.get('regions', []) if (source_id, region.get('id')) in parents]
                for region in wanted:
                    identity = (source_id, region['id'])
                    require(identity not in matched, 'duplicate iOS native TRAIN identity')
                    try:
                        matched[identity] = ios_parent(parents[identity], record, region, source_path, label_sha)
                    except CandidateRejected as error:
                        matched[identity] = {**parents[identity], 'candidate_rejection': str(error)}
            elif source_name == 'android':
                identity = (record.get('source_id'), record.get('region_id'))
                if identity not in parents:
                    continue
                require(identity not in matched, 'duplicate Android native TRAIN identity')
                try:
                    matched[identity] = android_parent(parents[identity], record, source_path, proof_cache)
                except CandidateRejected as error:
                    matched[identity] = {**parents[identity], 'candidate_rejection': str(error)}
            else:
                raise ValueError('Short region supplement: unknown source kind ' + source_name)
    require(set(matched) == set(parents), 'not every unified TRAIN identity has matching TRAIN native proof')
    for item in matched.values():
        require(item['source_font_family'] not in HELD_OUT_UNKNOWN, 'held-out unknown family entered TRAIN supplement')
        if item['family'] == UNKNOWN:
            require(item['source_font_family'] in ALLOWED_UNKNOWN_TRAIN, 'unknown source is not one of the nine old TRAIN families')
    for path, (_, digest) in proof_cache.items():
        bindings[str(path)] = digest
    return matched


def candidate_rows(parent, rejected):
    glyphs = parent['glyphs']
    visible_boxes = parent['full_visible_bboxes']
    for length in range(1, 5):
        for start in range(0, len(glyphs) - length + 1):
            chosen = glyphs[start:start + length]
            if any(chosen[index + 1]['proof_index'] != chosen[index]['proof_index'] + 1
                   or chosen[index + 1]['text_index'] != chosen[index]['text_index'] + 1 for index in range(length - 1)):
                rejected['non_contiguous_glyph_or_text_index'] += 1
                continue
            if not any(classification_character(item['character']) for item in chosen):
                rejected['punctuation_only_fragment'] += 1
                continue
            selected_indices = {item['proof_index'] for item in chosen}
            unselected = [item['bbox'] for item in glyphs if item['proof_index'] not in selected_indices]
            crop, padding = safe_crop([item['bbox'] for item in chosen], unselected, parent['parent_bbox'],
                                      parent['image_width'], parent['image_height'])
            if crop is None:
                rejected['selected_crop_intersects_unselected_glyph'] += 1
                continue
            fragment = ''.join(item['character'] for item in chosen)
            candidate_id = hashlib.sha256(('|'.join((parent['source_id'], parent['region_id'],
                str(chosen[0]['proof_index']), str(length), ','.join(map(str, crop))))).encode()).hexdigest()
            yield {
                'candidate_id': candidate_id, 'split': 'train', 'family': parent['family'], 'target': parent['target'],
                'domain': parent['domain'], 'source_dataset': parent['source_dataset'],
                'source_font_family': parent['source_font_family'], 'font_face': parent['font_face'],
                'font_file_sha256': parent['font_file_sha256'], 'font_source_kind': parent['font_source_kind'],
                'ttc_index': parent['ttc_index'], 'source_id': parent['source_id'], 'region_id': 'short:' + candidate_id,
                'parent_source_id': parent['source_id'], 'parent_region_id': parent['region_id'],
                'parent_page_id': parent['page_id'], 'source_image': parent['image'],
                'source_image_sha256': parent['source_sha256'], 'source_image_size': [parent['image_width'], parent['image_height']],
                'parent_region_bbox': parent['parent_bbox'], 'source_crop_bbox': crop, 'crop_padding_px': padding,
                'selected_glyph_start': chosen[0]['proof_index'], 'selected_glyph_count': length,
                'selected_glyph_ids': [item['glyph_id'] for item in chosen],
                'selected_glyph_bboxes': [item['bbox'] for item in chosen],
                'selected_text_indices': [item['text_index'] for item in chosen],
                'normalized_text_sha256': text_sha(fragment), 'native_font_size_px': parent['native_font_size_px'],
                'text_color_hex': parent['text_color_hex'], 'native_font_verified': True,
                'glyph_mapping': parent['mapping'], 'native_proof': parent['proof'],
                'native_proof_sha256': parent['proof_sha256'], 'selection_rank': f'{rank(candidate_id):064x}',
            }


def choose(candidates):
    """Fixed-hash, round-robin domain/face stratification with bounded memory."""
    strata = defaultdict(list)
    totals = Counter()
    rejected = Counter()
    parent_rejected = Counter()
    for parent in candidates.values():
        if parent.get('candidate_rejection'):
            parent_rejected[parent['candidate_rejection']] += 1
            continue
        try:
            generated = candidate_rows(parent, rejected)
            seen = False
            for candidate in generated:
                seen = True
                key = (candidate['family'], candidate['selected_glyph_count'], candidate['domain'], candidate['font_face'])
                totals[(candidate['family'], candidate['selected_glyph_count'])] += 1
                item = (-int(candidate['selection_rank'], 16), candidate['candidate_id'], candidate)
                heap = strata[key]
                if len(heap) < MAX_PER_FAMILY_LENGTH:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
            if not seen:
                parent_rejected['no_safe_1_to_4_glyph_candidate'] += 1
        except ValueError as error:
            parent_rejected[str(error)] += 1
    selected = []
    for family in FAMILIES:
        family_total = 0
        for length in range(1, 5):
            keys = [key for key in strata if key[0] == family and key[1] == length]
            keys.sort(key=lambda key: hashlib.sha256((SEED + '|' + '|'.join(map(str, key))).encode()).hexdigest())
            queues = {key: sorted((item[2] for item in strata[key]), key=lambda item: item['selection_rank']) for key in keys}
            taken = 0
            while taken < MAX_PER_FAMILY_LENGTH and keys:
                remaining = []
                for key in keys:
                    if taken >= MAX_PER_FAMILY_LENGTH:
                        break
                    if queues[key]:
                        selected.append(queues[key].pop(0)); taken += 1; family_total += 1
                    if queues[key]:
                        remaining.append(key)
                keys = remaining
        require(family_total > 0, 'a family lacks any safe short TRAIN candidate: ' + family)
        require(family_total <= MAX_PER_FAMILY, 'family short-crop budget exceeded')
    require(len(selected) <= MAX_SUBCROPS, 'global short-crop budget exceeded')
    selected.sort(key=lambda item: (item['source_image'], item['candidate_id']))
    return selected, totals, rejected, parent_rejected


def build_plan(train=DEFAULT_TRAIN, sources=None):
    sources = DEFAULT_SOURCES if sources is None else sources
    parents, bindings, parent_views = unified_train(train)
    native = native_parents(parents, sources, bindings)
    selected, totals, rejected, parent_rejected = choose(native)
    selected_counts = Counter((item['family'], item['selected_glyph_count']) for item in selected)
    family_counts = Counter(item['family'] for item in selected)
    source_counts = Counter(item['source_font_family'] for item in selected if item['family'] == UNKNOWN)
    require(set(family_counts) == set(FAMILIES), 'selected short plan lacks a family')
    require(set(source_counts).issubset(ALLOWED_UNKNOWN_TRAIN) and not set(source_counts) & HELD_OUT_UNKNOWN,
            'short plan violates unknown-family isolation')
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    for dependency in (ROOT / 'training/prepare_sans_views.py', ROOT / 'training/prepare_unified_regions.py',
                       ROOT / 'training/train_regions.py', ROOT / 'src/flux_glyph/region_font.py'):
        bindings[str(dependency)] = sha(dependency)
    return {
        'schema': PLAN_SCHEMA, 'seed': SEED, 'only_split': 'train', 'artifact_kind': 'derived_training_augmentation',
        'parent_train': str(Path(train).resolve()), 'families': FAMILIES, 'views': list(VIEWS),
        'selection_policy': {
            'glyph_lengths': [1, 2, 3, 4], 'max_per_family_length': MAX_PER_FAMILY_LENGTH,
            'max_per_family': MAX_PER_FAMILY, 'max_subcrops': MAX_SUBCROPS,
            'strata': ['family', 'domain', 'font_face', 'glyph_length'], 'ordering': 'fixed_sha256_round_robin',
            'padding_px_max': 2, 'unselected_glyph_intersection_allowed': False,
            'required_content': 'at least one Han or ASCII letter/digit glyph; punctuation may accompany it',
        },
        'scope': {
            'parent_unified_train_native_identities': len(parents), 'parent_derived_views': parent_views,
            'calibration_development_test_read': False, 'source_images_read': False,
            'non_train_json_records_decoded': False, 'non_train_records_used': False,
            'ocr_performed': False, 'font_reference_matching_performed': False,
            'held_out_unknown_families_excluded': sorted(HELD_OUT_UNKNOWN),
            'allowed_unknown_train_source_families': sorted(ALLOWED_UNKNOWN_TRAIN),
        },
        'coverage': {
            'candidate_counts_by_family_length': {
                family: {str(length): totals[(family, length)] for length in range(1, 5)} for family in FAMILIES
            },
            'selected_by_family_length': {
                family: {str(length): selected_counts[(family, length)] for length in range(1, 5)} for family in FAMILIES
            },
            'selected_by_family': dict(family_counts),
            'selected_unknown_by_source_family': dict(source_counts),
            'selected_subcrops': len(selected), 'planned_views': len(selected) * len(VIEWS),
            'planned_bytes_if_one_tile_per_view': len(selected) * len(VIEWS) * 64 * 256 * 4,
            'candidate_rejections': dict(rejected), 'parent_rejections': dict(parent_rejected),
        },
        'bindings': dict(sorted(bindings.items())), 'candidates': selected,
    }


def write_plan(output, plan):
    output = Path(output).resolve()
    require(not output.exists(), 'plan output must be new')
    output.mkdir(parents=True)
    dump(output / 'PLAN.json', plan)
    return plan


def materialize(output, expected_plan):
    output = Path(output).resolve()
    plan_path = output / 'PLAN.json'
    require(output.is_dir() and plan_path.is_file(), 'run --plan-only before materialization')
    require(not (output / 'MANIFEST.json').exists() and not (output / 'train').exists(),
            'materialized supplement output must not be overwritten')
    frozen = read(plan_path)
    canonical_expected = json.loads(json.dumps(expected_plan, ensure_ascii=False, allow_nan=False))
    require(frozen == canonical_expected and frozen.get('schema') == PLAN_SCHEMA, 'frozen plan/source coverage changed')
    require(all(sha(path) == digest for path, digest in frozen['bindings'].items()), 'plan binding changed')
    image_bindings = {}
    rows, rejected, tile_count = [], [], 0
    with tempfile.TemporaryDirectory(prefix='.short-region-', dir=output.parent) as temporary:
        stage = Path(temporary)
        folder = stage / 'train'; folder.mkdir(parents=True)
        current_path, current_image = None, None
        try:
            with (folder / 'tiles.raw').open('wb') as stream:
                for candidate in frozen['candidates']:
                    path = inside_repo(candidate['source_image'])
                    if path != current_path:
                        if current_image is not None:
                            current_image.close()
                        require(sha(path) == candidate['source_image_sha256'], 'selected TRAIN source image changed')
                        current_image = Image.open(path).convert('RGB'); current_path = path
                        require(list(current_image.size) == candidate['source_image_size'], 'selected TRAIN image size differs')
                        image_bindings[str(path)] = candidate['source_image_sha256']
                    crop = current_image.crop(candidate['source_crop_bbox'])
                    prepared_views = []
                    reason = None
                    for view in VIEWS:
                        viewed, scale_y, scale_x = apply_view(crop, view)
                        processed = preprocess_region(viewed)
                        tiles_value = processed.get('tiles')
                        if (processed.get('status') != 'ok' or processed.get('whole_width_covered') is not True
                                or not isinstance(tiles_value, np.ndarray) or len(tiles_value) != 1):
                            reason = processed.get('reason', 'short_crop_requires_exactly_one_tile')
                            if processed.get('status') == 'ok' and len(processed['tiles']) != 1:
                                reason = 'short_crop_requires_exactly_one_tile'
                            break
                        tiles = processed['tiles']
                        height = processed['ink_height_px']; size = candidate['native_font_size_px'] * scale_y
                        ratio = math.log(size / height)
                        require(tiles.dtype == np.float32 and tiles.shape == (1, 1, 64, 256)
                                and np.isfinite(tiles).all() and np.all((tiles >= 0) & (tiles <= 1))
                                and math.isfinite(ratio) and -3 <= ratio <= 3, 'invalid short-region preprocessing')
                        prepared_views.append((view, tiles, scale_y, scale_x, height, size, ratio))
                    if reason is not None:
                        rejected.append({'candidate_id': candidate['candidate_id'], 'family': candidate['family'],
                                         'selected_glyph_count': candidate['selected_glyph_count'], 'reason': reason})
                        continue
                    crop_sha = pixels_sha(crop)
                    for view, tiles, scale_y, scale_x, height, size, ratio in prepared_views:
                        block = np.ascontiguousarray(tiles, dtype='<f4'); stream.write(block.tobytes())
                        rows.append({key: candidate[key] for key in (
                            'candidate_id', 'split', 'family', 'target', 'domain', 'source_dataset',
                            'source_font_family', 'font_face', 'font_file_sha256', 'font_source_kind', 'ttc_index',
                            'source_id', 'region_id', 'parent_source_id', 'parent_region_id', 'parent_page_id',
                            'source_image_sha256', 'parent_region_bbox', 'source_crop_bbox', 'crop_padding_px',
                            'selected_glyph_start', 'selected_glyph_count', 'selected_glyph_ids',
                            'selected_glyph_bboxes', 'selected_text_indices', 'normalized_text_sha256',
                            'native_font_size_px', 'text_color_hex', 'native_font_verified', 'glyph_mapping',
                            'native_proof', 'native_proof_sha256') } | {
                            'index': len(rows), 'glyph_count': candidate['selected_glyph_count'],
                            'evaluation_role': 'derived_train_augmentation', 'view': view,
                            'native_crop_sha256': crop_sha, 'actual_view_scale_y': scale_y,
                            'actual_view_scale_x': scale_x, 'tile_start': tile_count, 'tile_count': 1,
                            'tiles_sha256': tile_sha(block), 'ink_height_px': height,
                            'font_size_px': size, 'log_em_ratio': ratio,
                        })
                        tile_count += 1
            require((folder / 'tiles.raw').stat().st_size == tile_count * 64 * 256 * 4
                    and (folder / 'tiles.raw').stat().st_size <= MAX_OUTPUT_BYTES, 'short supplement exceeds byte budget')
            require(rows and all(any(row['family'] == family and row['view'] == 'native' for row in rows)
                                 for family in FAMILIES), 'materialized supplement lacks a family native view')
            dump(folder / 'rows.json', rows)
            manifest = {
                'schema': SCHEMA, 'only_split': 'train', 'artifact_kind': 'derived_training_augmentation',
                'families': FAMILIES, 'parent_train': frozen['parent_train'], 'plan': str(plan_path),
                'plan_sha256': sha(plan_path), 'bindings': {**frozen['bindings'], **image_bindings},
                'views': VIEWS, 'model_inputs': ['image_tiles'], 'ocr_performed': False,
                'text_features_used': False, 'script_features_used': False, 'platform_features_used': False,
                'font_reference_matching_performed': False, 'native_font_verified': True,
                'old_data_modified': False, 'calibration_development_test_read': False,
                'size_target': 'log(native font px * actual vertical view scale / measured view ink height)',
                'materialization_policy': 'all four views accepted together; preprocessing failure is counted and never replenished from CAL',
            }
            dump(stage / 'MANIFEST.json', manifest)
            part = {
                'schema': SCHEMA, 'split': 'train', 'families': FAMILIES,
                'root_manifest_sha256': sha(stage / 'MANIFEST.json'), 'selected_candidates': len(frozen['candidates']),
                'native_subcrops': len(rows) // len(VIEWS), 'views': len(rows), 'tiles': tile_count,
                'shape': [tile_count, 1, 64, 256], 'bytes': (folder / 'tiles.raw').stat().st_size,
                'counts': dict(Counter(row['family'] for row in rows)),
                'native_counts': dict(Counter(row['family'] for row in rows if row['view'] == 'native')),
                'glyph_length_counts': dict(Counter(str(row['selected_glyph_count']) for row in rows if row['view'] == 'native')),
                'unknown_source_counts': dict(Counter(row['source_font_family'] for row in rows
                                                     if row['view'] == 'native' and row['family'] == UNKNOWN)),
                'rejected': rejected, 'array': {'path': 'tiles.raw', 'sha256': sha(folder / 'tiles.raw')},
                'metadata': {'path': 'rows.json', 'sha256': sha(folder / 'rows.json')},
            }
            dump(folder / 'MANIFEST.json', part)
            require(sha(plan_path) == manifest['plan_sha256']
                    and all(sha(path) == digest for path, digest in manifest['bindings'].items()),
                    'source or frozen plan changed during materialization')
            shutil.move(str(stage / 'MANIFEST.json'), str(output / 'MANIFEST.json'))
            shutil.move(str(folder), str(output / 'train'))
        finally:
            if current_image is not None:
                current_image.close()
    return manifest


def prepare(train=DEFAULT_TRAIN, output=DEFAULT_OUTPUT, *, sources=None, plan_only=False):
    plan = build_plan(train, sources)
    if plan_only:
        return write_plan(output, plan)
    return materialize(output, plan)


def load_supplement(root):
    root = Path(root).resolve(); manifest = read(root / 'MANIFEST.json'); part = read(root / 'train/MANIFEST.json')
    require(manifest.get('schema') == part.get('schema') == SCHEMA and manifest.get('only_split') == part.get('split') == 'train'
            and manifest.get('artifact_kind') == 'derived_training_augmentation'
            and manifest.get('model_inputs') == ['image_tiles'] and manifest.get('families') == part.get('families') == FAMILIES
            and part.get('root_manifest_sha256') == sha(root / 'MANIFEST.json'), 'short supplement contract differs')
    require(manifest.get('plan_sha256') == sha(root / 'PLAN.json') and all(sha(path) == digest for path, digest in manifest['bindings'].items()),
            'short supplement provenance changed')
    for key in ('array', 'metadata'):
        path = (root / 'train' / part[key]['path']).resolve()
        require(path.parent == root / 'train' and sha(path) == part[key]['sha256'], 'short supplement file changed')
    rows = read(root / 'train' / part['metadata']['path']); count = part['tiles']
    path = root / 'train' / part['array']['path']
    require(len(rows) == part['views'] == count and part['shape'] == [count, 1, 64, 256]
            and path.stat().st_size == count * 64 * 256 * 4 <= MAX_OUTPUT_BYTES, 'short supplement shape/size differs')
    tiles = np.memmap(path, mode='r', dtype='<f4', shape=tuple(part['shape']))
    plan = read(root / 'PLAN.json'); candidates = {item['candidate_id']: item for item in plan['candidates']}
    truth_fields = ('family', 'target', 'domain', 'source_dataset', 'source_font_family', 'font_face',
                    'font_file_sha256', 'font_source_kind', 'ttc_index', 'source_id', 'region_id',
                    'parent_source_id', 'parent_region_id', 'parent_page_id', 'source_image_sha256',
                    'parent_region_bbox', 'source_crop_bbox', 'crop_padding_px', 'selected_glyph_start',
                    'selected_glyph_count', 'selected_glyph_ids', 'selected_glyph_bboxes',
                    'selected_text_indices', 'normalized_text_sha256', 'native_font_size_px',
                    'text_color_hex', 'native_font_verified', 'glyph_mapping', 'native_proof',
                    'native_proof_sha256')
    for offset, row in enumerate(rows):
        candidate = candidates.get(row.get('candidate_id'))
        require(candidate is not None and row.get('split') == 'train' and row.get('native_font_verified') is True
                and row.get('family') in FAMILIES and row.get('target') == FAMILIES.index(row['family'])
                and row.get('domain') in ('ios', 'android') and row.get('view') in VIEWS
                and row.get('index') == offset and row.get('tile_start') == offset and row.get('tile_count') == 1
                and row.get('glyph_count') == row.get('selected_glyph_count') in (1, 2, 3, 4)
                and row.get('parent_source_id') == candidate['parent_source_id']
                and row.get('parent_region_id') == candidate['parent_region_id']
                and row.get('selected_glyph_start') == candidate['selected_glyph_start']
                and row.get('selected_glyph_count') == candidate['selected_glyph_count'], 'short supplement row identity differs')
        require(all(row.get(key) == candidate.get(key) for key in truth_fields),
                'short supplement row differs from frozen candidate truth')
        require(math.isclose(math.log(row['font_size_px'] / row['ink_height_px']), row['log_em_ratio'], abs_tol=1e-9)
                and np.isfinite(tiles[offset]).all() and np.all((tiles[offset] >= 0) & (tiles[offset] <= 1))
                and tile_sha(tiles[offset:offset + 1]) == row['tiles_sha256'], 'short supplement tensor/size target differs')
    require(part['counts'] == dict(Counter(row['family'] for row in rows))
            and part['native_counts'] == dict(Counter(row['family'] for row in rows if row['view'] == 'native')),
            'short supplement coverage differs')
    return {'families': FAMILIES, 'rows': rows, 'tiles': tiles, 'partition': part, 'manifest': manifest,
            'manifest_sha256': sha(root / 'MANIFEST.json'), 'partition_sha256': sha(root / 'train/MANIFEST.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=DEFAULT_TRAIN)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    result = prepare(args.train, args.output, plan_only=args.plan_only)
    print(json.dumps({'schema': result['schema'], 'output': str(args.output),
                      'selected_subcrops': result.get('coverage', {}).get('selected_subcrops'),
                      'source_images_read': not args.plan_only}, ensure_ascii=False), flush=True)
