#!/usr/bin/env python3
"""Audit the sans capture plan before the shared native region preparation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from training.capture.generate_sans_scenes import COUNTS, canonical
from training.capture.generate_scenes import normalized_text, sha, write_json
from training.train_regions import prepare

MAX_PREPROCESS_REJECTION_FRACTION = .005


def prepare_sans(captures, output, scene_sources, *, verify_prepared=False):
    captures, output, scene_sources = map(lambda p: Path(p).resolve(), (captures, output, scene_sources))
    source = json.loads(scene_sources.read_text())
    scenes_path = captures.parent/'Scenes.json'
    scenes = json.loads(scenes_path.read_text())
    assert source['scenes_sha256'] == sha(scenes_path)
    assert source['generator_sha256'] == sha(ROOT/'training/capture/generate_sans_scenes.py')
    assert dict(Counter(p['split'] for p in scenes['pages'])) == source['split_counts'] == COUNTS
    assert not source['android_native_rendering_validated'] and not source['old_test_pixels_opened']
    current_text = {normalized_text(r['text']) for p in scenes['pages'] for r in p['regions']}
    for path, expected in source['old_scene_sources'].items():
        assert sha(path) == expected, 'Old text exclusion source changed'
        prior = json.loads(Path(path).read_text())
        assert not current_text & {normalized_text(r['text']) for p in prior.get('pages', []) for r in p['regions']}
    for font in scenes['fonts']:
        if font['kind'] == 'asset':
            assert Path(font['path']).name == font['path']
            assert sha(scene_sources.parent/'assets'/font['path']) == font['sha256']
    report = (json.loads((output/'MANIFEST.json').read_text()) if verify_prepared else
              prepare(captures, output, group_pingfang=True, test_history='fresh'))
    assert report['input_labels']['sha256'] == sha(captures)
    assert report['scenes']['sha256'] == sha(scenes_path)
    assert report['preparation_code_sha256'] == sha(ROOT/'training/train_regions.py')
    assert report['test_history'] == 'fresh'
    if set(report['rejected']) - {'preprocess_region_rejected'}:
        raise ValueError('Sans capture contains failed native proof: ' + str(report['rejected']))
    # Keep the existing inference preprocessor's quality rejection. A tiny thin
    # glyph can have correct native truth while supplying too little usable ink.
    # Never lower that pixel threshold just to satisfy a planned region count.
    accepted, exclusions = set(), []
    for split, count in COUNTS.items():
        part = report['splits'][split]
        for binding in ('array', 'metadata'):
            assert sha(output/part[binding]['path']) == part[binding]['sha256']
        rows = json.loads((output/part['metadata']['path']).read_text())['rows']
        assert len(rows) == part['regions']
        accepted.update(r['region_id'] for r in rows)
        assert Counter(r['family'] for r in rows) == part['family_counts']
        assert 0 <= count*12-part['regions'] <= count*12*MAX_PREPROCESS_REJECTION_FRACTION
    from training.capture.prepare_captured import font_evidence, asset_evidence
    fonts = {f['id']: f for f in scenes['fonts']}
    requests = {r['id']: r for p in scenes['pages'] for r in p['regions']}
    seen = set()
    for line in captures.read_text().splitlines():
        frame = json.loads(line)
        for region in frame['regions']:
            rid = region['id']
            assert rid in requests and rid not in seen
            seen.add(rid)
            reason = font_evidence(region, region['font_family'], region['script'])
            reason = reason or asset_evidence(region, requests[rid], fonts)
            assert reason is None, reason
            if rid not in accepted:
                exclusions.append({'region_id': rid, 'split': frame['split'],
                                   'family': canonical(region['font_family']),
                                   'font_face': region['actual_font_postscript'],
                                   'font_size_points': region['actual_font_size_points'],
                                   'reason': 'existing_preprocess_region_rejected'})
    assert seen == set(requests)
    assert len(exclusions) == report['rejected'].get('preprocess_region_rejected', 0)
    audit = {'schema': 'flux-glyph-sans-native-data-audit-v1', 'passed': True,
             'manifest_sha256': sha(output/'MANIFEST.json'), 'scene_sources_sha256': sha(scene_sources),
             'scenes_sha256': sha(scenes_path), 'labels_sha256': sha(captures),
             'preparer_sha256': sha(__file__), 'same_renderer_controlled_native_data': True,
             'android_native_rendering_validated': False, 'user_pixels_used': False,
             'old_test_pixels_used_for_training': False, 'old_normalized_text_overlap': 0,
             'native_rejections': 0, 'native_verified_regions': len(seen),
             'preprocessing_exclusions': exclusions,
             'max_preprocessing_exclusion_fraction_per_split': MAX_PREPROCESS_REJECTION_FRACTION,
             'regions_by_split': {s: report['splits'][s]['regions'] for s in COUNTS},
             'families': report['families']}
    write_json(output/'SOURCE_AUDIT.json', audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scene-sources', required=True, type=Path)
    parser.add_argument('--verify-prepared', action='store_true')
    args = parser.parse_args()
    prepare_sans(args.captures, args.output, args.scene_sources, verify_prepared=args.verify_prepared)
