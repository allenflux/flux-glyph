#!/usr/bin/env python3
"""Author TRAIN-only short text requests; font truth comes from native capture."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import unicodedata

from capture.generate_scenes import ios_fonts

NAMESPACE = 'ios-native-short-train-v1'
SEED = 2026091402
# Authored independently of screenshot/CAL text and labels. Full-text hashes
# from reused CAL/DEV may only remove collisions; sealed TEST is never opened.
HAN = {
    'simplified': '山水风云日月星光春夏秋冬花草树木书画诗词城市乡村桥路湖海家门衣食学友明暗高低长短新旧左右东西南北',
    'traditional': '山水風雲日月星光春夏秋冬花草樹木書畫詩詞城市鄉村橋路湖海家門衣食學友明暗高低長短新舊左右東西南北',
}
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
WORDS = {1: list(LETTERS), 2: 'go up in on at by if we us ox my no'.split(),
         3: 'air sky oak sun sea ink map owl fox red dry joy'.split(),
         4: 'dawn dusk leaf blue bird moon sand snow wind rain hill lake'.split()}


def text_sha(text):
    normalized = ''.join(unicodedata.normalize('NFKC', text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corpus(kind, length, count, rng, forbidden):
    result, seen = [], set()
    for attempt in range(10000):
        if len(result) == count:
            return result
        if kind in HAN:
            text = ''.join(rng.choice(HAN[kind]) for _ in range(length))
        elif kind == 'numeric':
            text = ''.join(rng.choice('0123456789') for _ in range(length))
        else:
            text = (WORDS[length][attempt % len(WORDS[length])] if attempt < len(WORDS[length])
                    else ''.join(rng.choice(LETTERS) for _ in range(length)))
        digest = text_sha(text)
        if digest not in seen and digest not in forbidden:
            seen.add(digest)
            result.append(text)
    raise ValueError(f'Insufficient distinct non-heldout text for {kind}/{length}')


def generate(forbidden=()):
    forbidden = frozenset(forbidden)
    text_rng, order_rng = random.Random(SEED), random.Random(SEED + 1)
    fonts = [f for f in ios_fonts() if f['family'] in {'PingFang SC', 'SF Pro', 'Helvetica'}]
    # Same authored corpus and style plan across faces; text/size/color must
    # never encode a font label. Digits have only ten unique one-char strings.
    pools = {(kind, length): corpus(kind, length, 8, text_rng, forbidden)
             for kind in (*HAN, 'english', 'numeric') for length in range(1, 5)}
    items = []
    sizes = [12, 15, 18, 22, 27, 32, 36, 40]
    colors = ['#111111', '#333333', '#555555', '#999999', '#005BBB', '#B42318', '#1D6B44', '#303A45']
    for font in fonts:
        kinds = list(HAN) if font['family'] == 'PingFang SC' else ['english', 'numeric']
        for kind in kinds:
            for length in range(1, 5):
                for index, text in enumerate(pools[kind, length]):
                    han = kind in HAN
                    items.append({'text': text, 'script': 'han' if han else 'latin',
                                  'text_kind': 'han' if han else kind,
                                  'han_orthography': kind if han else None,
                                  'language': ('zh-Hant' if kind == 'traditional' else 'zh-Hans') if han else 'en',
                                  'font_id': font['id'], 'font_postscript': font['postscript'],
                                  'font_family': font['family'], 'weight': font['weight'],
                                  'font_size': sizes[(index + length) % len(sizes)],
                                  'color': colors[(index + length * 3) % len(colors)]})
    order_rng.shuffle(items)
    pages = []
    for start in range(0, len(items), 12):
        page_id = f'{NAMESPACE}-{len(pages):05d}'
        regions = []
        for i, item in enumerate(items[start:start + 12]):
            regions.append({**item, 'id': f'{page_id}-r{i:02d}',
                            'bbox_points': [24, 66 + i * 64, 382, 118 + i * 64]})
        pages.append({'id': page_id, 'source_id': page_id, 'content_group_id': page_id,
                      'split': 'train', 'background': '#FFFFFF', 'regions': regions})
    return {'schema': 'flux-glyph-capture-scenes-v1', 'platform': 'ios',
            'canvas_points': [402, 874], 'seed': SEED, 'fonts': fonts, 'pages': pages,
            'label_status': 'requests only; actual CTFont/CTRun proof must pass before TRAIN import',
            'ui_content_is_generated': True, 'intended_capture_kind': 'ios_simulator_controlled_scene',
            'split_unit': 'new TRAIN-only scene namespace; no new evaluation partition',
            'characters_may_overlap_across_splits': True, 'test_read': False,
            'heldout_policy': 'remove full normalized-text hash collisions with reused CAL/DEV only; TEST remains sealed',
            'style_policy': 'same text, size and color plan for every face within each script',
            'short_region_characters': [1, 2, 3, 4], 'source_capture_required': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reused-data', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing scene bundle')
    forbidden, bindings = set(), {}
    for split in ('calibration', 'development_holdout'):
        path = args.reused_data / split / 'rows.json'
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not rows:
            raise ValueError('Expected existing CAL/DEV row metadata')
        for row in rows:
            value = row.get('normalized_text_sha256')
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError('Missing full normalized-text hash')
            forbidden.add(value)
        bindings[str(path.resolve())] = sha(path)
    scenes = generate(forbidden)
    rows = [r for p in scenes['pages'] for r in p['regions']]
    assert not {text_sha(r['text']) for r in rows}.intersection(forbidden)
    args.output.mkdir(parents=True)
    scene_path = args.output / 'Scenes.json'
    scene_path.write_text(json.dumps(scenes, ensure_ascii=False, indent=2) + '\n')
    report = {'schema': 'flux-glyph-ios-short-train-requests-v1', 'pages': len(scenes['pages']),
              'regions': len(rows), 'family_counts': dict(Counter(r['font_family'] for r in rows)),
              'face_counts': dict(Counter(r['font_id'] for r in rows)),
              'length_counts': dict(Counter(len(r['text']) for r in rows)),
              'kind_counts': dict(Counter(r['text_kind'] for r in rows)),
              'full_text_cal_dev_hash_overlap': 0, 'metadata_bindings': bindings,
              'generator_sha256': sha(__file__), 'font_request_registry_sha256': sha(Path(__file__).with_name('generate_scenes.py')),
              'scenes_sha256': sha(scene_path), 'test_read': False, 'native_font_labels_verified': False,
              'samples_used_in_training': False, 'source_kind': 'authored_native_capture_requests'}
    (args.output / 'REQUESTS.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
