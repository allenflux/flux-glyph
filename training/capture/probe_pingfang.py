"""Prepare/score small native SC/TC/HK pixel probes, never training examples."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from generate_scenes import SCHEMA, ios_fonts, sha, write_json

PROBES = ['支付成功交易完成', '賬單詳情餘額轉帳', '臺灣香港網絡設定',
          '骨直令雨言衣辶示', '內戶告起青者返道', '線綫裏裡兌說説龍']


def document(inventory):
    fonts = [font for font in ios_fonts(inventory, include_traditional=True) if font['family'].startswith('PingFang ')]
    lookup = {(font['family'], font['weight']): font for font in fonts}
    if set(lookup) != {(family, weight) for family in ('PingFang SC', 'PingFang TC', 'PingFang HK') for weight in (300, 400, 500, 600)}:
        raise ValueError('Probe requires SC/TC/HK Light/Regular/Medium/Semibold in the actual inventory')
    regions = []
    for weight in (300, 400, 500, 600):
        for language in ('zh-Hant', 'zh-Hant-HK'):
            for text_index, text in enumerate(PROBES):
                group = f'w{weight}-{language}-text{text_index}'
                for family in ('PingFang SC', 'PingFang TC', 'PingFang HK'):
                    font = lookup[family, weight]
                    regions.append({'text': text, 'script': 'han', 'han_orthography': 'traditional',
                                    'language': language, 'probe_group': group, 'font_id': font['id'],
                                    'font_postscript': font['postscript'], 'font_family': family,
                                    'font_size': 24, 'weight': weight, 'color': '#111111'})
    pages = []
    for offset in range(0, len(regions), 12):
        identifier = f'ios-pingfang-region-probe-{offset//12:02d}'
        rows = []
        for i, source in enumerate(regions[offset:offset+12]):
            rows.append(dict(source, id=f'{identifier}-r{i:02d}', bbox_points=[24, 66+i*64, 382, 118+i*64]))
        pages.append({'id': identifier, 'source_id': identifier, 'content_group_id': 'pingfang-visual-identity-probe',
                      'split': 'train', 'background': '#FFFFFF', 'diagnostic_only': True, 'regions': rows})
    return {'schema': SCHEMA, 'platform': 'ios', 'canvas_points': [402, 874], 'fonts': fonts, 'pages': pages,
            'diagnostic_only': True, 'training_eligible': False,
            'purpose': 'Native family/fallback proof and same-content regional pixel identity; never use for model selection',
            'inventory': {'path': str(Path(inventory).resolve()), 'sha256': sha(inventory)}}


def analyze(labels, output):
    from prepare_captured import font_evidence, native_source, load_scenes
    from flux_glyph.region_font import preprocess_region
    labels = Path(labels).resolve()
    scenes_path = labels.parent / 'Scenes.json'
    scenes, pages, _ = load_scenes(scenes_path)
    if scenes.get('training_eligible') is not False or scenes.get('diagnostic_only') is not True:
        raise ValueError('Expected diagnostic-only native probe scenes')
    groups, verified, rejected = defaultdict(dict), [], []
    for line in labels.read_text().splitlines():
        record = json.loads(line)
        page, image_path, image, requests = native_source(record, sha(scenes_path), pages, labels.parent)
        try:
            for region in record['regions']:
                request = requests[region['id']]
                reason = font_evidence(region, region['font_family'], 'han')
                if reason:
                    rejected.append({'page': page['id'], 'region': region['id'], 'reason': reason,
                                     'requested': request['font_postscript'], 'actual': region.get('actual_font_postscript')})
                    continue
                pixels = np.asarray(image.crop(region['bbox']), dtype=np.uint8).copy()
                groups[request['probe_group']][region['font_family']] = (pixels, region)
                verified.append({'page': page['id'], 'region': region['id'], 'family': region['font_family'],
                                 'requested': region['requested_font_postscript'], 'actual': region['actual_font_postscript'],
                                 'actual_family': region['actual_font_family'], 'text': region['text'],
                                 'language': region.get('requested_language'), 'fallback': region['fallback_detected'],
                                 'zero_glyphs': region['glyph_coverage']['zero_run_glyph_count'],
                                 'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()})
        finally:
            image.close()
    comparisons = []
    for name, group in sorted(groups.items()):
        for first, second in [('PingFang SC', 'PingFang TC'), ('PingFang TC', 'PingFang HK'), ('PingFang SC', 'PingFang HK')]:
            if first not in group or second not in group:
                continue
            a, ar = group[first]
            b, br = group[second]
            if a.shape != b.shape:
                raise ValueError('Probe crop sizes differ')
            diff = np.abs(a.astype(np.int16)-b.astype(np.int16))
            ink = np.any(a < 230, axis=2) | np.any(b < 230, axis=2)
            changed = np.any(diff > 0, axis=2)
            ap, bp = preprocess_region(Image.fromarray(a)), preprocess_region(Image.fromarray(b))
            valid = ap['status'] == bp['status'] == 'ok' and ap['tiles'].shape == bp['tiles'].shape
            comparisons.append({'probe_group': name, 'first': first, 'second': second, 'text': ar['text'],
                                'identical_pixels': bool(np.array_equal(a,b)),
                                'max_rgb_byte_difference': int(diff.max()),
                                'ink_union_pixels': int(ink.sum()), 'changed_ink_pixels': int((changed&ink).sum()),
                                'mean_absolute_rgb_difference_on_ink': float(diff[ink].mean()) if ink.any() else 0.,
                                'normalized_input_identical': bool(np.array_equal(ap['tiles'],bp['tiles'])) if valid else False,
                                'normalized_input_max_absolute_difference': float(np.abs(ap['tiles']-bp['tiles']).max()) if valid else None,
                                'actual_postscript': [ar['actual_font_postscript'],br['actual_font_postscript']]})
    summary = {}
    for pair in sorted({(row['first'], row['second']) for row in comparisons}):
        subset = [row for row in comparisons if (row['first'], row['second']) == pair]
        summary[' / '.join(pair)] = {'comparisons': len(subset), 'identical_pixels': sum(row['identical_pixels'] for row in subset),
                                     'identical_normalized_inputs': sum(row['normalized_input_identical'] for row in subset),
                                     'max_rgb_byte_difference': max(row['max_rgb_byte_difference'] for row in subset),
                                     'max_normalized_input_difference': max(row['normalized_input_max_absolute_difference'] or 0 for row in subset)}
    result = {'schema': 'flux-glyph-pingfang-native-visual-probe-v1', 'source_kind': 'ios_simulator_screenshot',
              'training_eligible': False, 'ocr_performed': False, 'labels_sha256': sha(labels),
              'scenes_sha256': sha(scenes_path), 'native_verified_regions': len(verified),
              'native_family_counts': dict(Counter(row['family'] for row in verified)),
              'rejected': rejected, 'same_content_comparisons': summary, 'regions': verified, 'comparisons': comparisons,
              'interpretation': 'Same pixels with distinct native font names are not visually identifiable; do not force a regional prediction.'}
    write_json(Path(output), result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if args.labels:
        result=analyze(args.labels,args.output)
        print(json.dumps({key:result[key] for key in ('native_verified_regions','native_family_counts','same_content_comparisons','rejected')},ensure_ascii=False,indent=2))
    elif args.inventory:
        result=document(args.inventory)
        write_json(args.output,result)
        print(json.dumps({'pages':len(result['pages']),'regions':sum(len(p['regions']) for p in result['pages'])}))
    else:
        parser.error('Supply --inventory to generate, or --labels to analyze')


if __name__ == '__main__':
    main()
