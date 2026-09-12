#!/usr/bin/env python3
"""Verify every native screenshot source and fixed split before preparation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'training')]
from training.capture import prepare_captured as native


def audit(directory, previous_scenes=()):
    directory = Path(directory).resolve()
    labels = directory / 'labels.jsonl'
    scenes_path = directory / 'Scenes.json'
    scenes_sha = native.sha(scenes_path)
    scenes, pages, isolation = native.load_scenes(scenes_path)
    protocol_path, protocol = native.capture_protocol(directory, scenes_sha)
    require = native.require
    require(scenes.get('diagnostic_only') is not True and scenes.get('training_eligible') is not False,
            'diagnostic probe captures cannot become a training cohort')
    fonts = {font['id']: font for font in scenes['fonts']}
    planned_regions = sum(len(page['regions']) for page in pages.values())
    expected_text = {native.normalized_text(region['text']) for page in pages.values() for region in page['regions']}
    expected_groups = {page.get('content_group_id', page['id']) for page in pages.values()}
    exclusion = []
    for path in previous_scenes:
        path = Path(path).resolve()
        old = json.loads(path.read_text())
        old_text = {native.normalized_text(region['text']) for page in old['pages'] for region in page['regions']}
        old_groups = {page.get('content_group_id', page['id']) for page in old['pages']}
        require(not old_text & expected_text, 'new/prior NFKC-casefold-whitespace text identity overlaps')
        require(not old_groups & expected_groups, 'new/prior content groups overlap')
        exclusion.append({'path': str(path), 'sha256': native.sha(path), 'text_overlap': 0, 'content_group_overlap': 0})
    rows = [json.loads(line) for line in labels.read_text().splitlines() if line.strip()]
    require(len(rows) == len(pages) and {row['page_id'] for row in rows} == set(pages), 'capture does not cover every planned page exactly once')
    require(len({row['source_id'] for row in rows}) == len(rows), 'duplicate source IDs')
    require(len(list((directory / 'frames').glob('*.json'))) == len(rows), 'frame records do not match source rows')
    summary = json.loads((directory / 'CAPTURE_SUMMARY.json').read_text())
    require(summary['captured_pages'] == len(rows) and summary['labels_sha256'] == native.sha(labels), 'capture summary differs from actual labels')
    counts = {split: defaultdict(Counter) for split in native.SPLITS}
    source_counts, source_kinds, seen_regions = Counter(), Counter(), 0
    for index, row in enumerate(rows):
        page, path, image, requests = native.native_source(row, scenes_sha, pages, directory)
        try:
            require(json.loads((directory / 'frames' / (page['id'] + '.json')).read_text()) == row,
                    'frame record and assembled labels differ')
            require(row['simulator_id'] == protocol['simulator_id'] and row['bundle_id'] == protocol['bundle_id'],
                    'source app/simulator differs from frozen capture protocol')
            split = row['split']
            source_counts[split] += 1
            source_kinds[row['source_kind']] += 1
            for kind, value in {'source_id': row['source_id'], 'page_id': page['id'],
                                'source_file_sha256': row['source_sha256'], 'decoded_pixel_sha256': native.pixels_sha(image),
                                'content_group_id': page.get('content_group_id', page['id'])}.items():
                isolation.bind(kind, value, split)
            for region in row['regions']:
                request = requests[region['id']]
                reason = native.font_evidence(region, region['font_family'], region['script'])
                reason = reason or native.asset_evidence(region, request, fonts)
                require(reason is None, 'unverified native font in final cohort: ' + str(reason))
                native.native_style(region, row['native'])
                require(native.bounds(region['bbox'], image.size), 'native region bbox outside screenshot')
                isolation.bind('region_rgb_sha256', native.pixels_sha(image.crop(region['bbox'])), split)
                isolation.bind('normalized_region_text', native.normalized_text(region['text']), split)
                seen_regions += 1
                for key, value in {'native_family': region['font_family'], 'script': region['script'],
                                   'text_kind': request['text_kind'], 'orthography': request.get('han_orthography') or 'latin',
                                   'font_source': fonts[request['font_id']]['kind']}.items():
                    counts[split][key][value] += 1
        finally:
            image.close()
        if (index + 1) % 100 == 0:
            print(json.dumps({'audited_sources': index + 1, 'native_verified_regions': seen_regions}), flush=True)
    require(seen_regions == planned_regions, 'native region count differs from planned scenes')
    require(dict(source_counts) == dict(Counter(page['split'] for page in pages.values())), 'native source split counts differ')
    return {'schema': 'flux-glyph-native-capture-source-audit-v1', 'all_planned_sources_verified': True,
            'source_counts': dict(source_counts), 'source_kinds': dict(source_kinds),
            'native_verified_regions': seen_regions, 'native_rejected_regions': 0,
            'source_file_sha256_verified': len(rows), 'source_decoded_pixel_hashes_checked': len(rows),
            'scene_sha256': scenes_sha, 'labels_sha256': native.sha(labels),
            'capture_protocol_sha256': native.sha(protocol_path), 'audit_code_sha256': native.sha(__file__),
            'text_normalization': 'NFKC + casefold + remove whitespace', 'prior_scene_exclusion': exclusion,
            'split_overlap': {key: 0 for key in ('source_id', 'page_id', 'source_file_sha256', 'decoded_pixel_sha256',
                                                'content_group_id', 'region_rgb_sha256', 'normalized_region_text')},
            'splits': {split: {key: dict(value) for key,value in items.items()} for split,items in counts.items()},
            'ocr_performed': False, 'model_predictions_or_test_metrics_computed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', required=True, type=Path)
    parser.add_argument('--exclude-scenes', action='append', type=Path, default=[])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.captures, args.exclude_scenes)
    destination = args.output or args.captures / 'SOURCE_AUDIT.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'verified': True, 'sources': result['source_counts'],
                      'regions': result['native_verified_regions'], 'output': str(destination)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
