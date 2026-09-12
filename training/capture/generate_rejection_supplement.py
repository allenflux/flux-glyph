#!/usr/bin/env python3
"""Plan new training-only native Latin handwriting pairs; render no pixels."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.capture.generate_unknown_scenes import canonical, derive_conflicting_face, dump, sha
from training.capture.generate_scenes import normalized_text, text_candidate

SEED = 2026091296
PAGES = 60
KNOWN_LATIN = ['HarmonyOS Sans SC', 'MiSans', 'OPPO Sans', 'SF Pro', 'Helvetica', 'Alipay Number']
NEW_UNKNOWN = ['Bradley Hand', 'Noteworthy', 'Apple Chancery']
SOURCES = [
    ('Bradley Hand', Path('/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf'),
     'ea54acf02ee9cc276e0e44ce6b2bf60e8eee0a4517468f0e989db55fc1ffe19c', ['BradleyHandITCTT-Bold']),
    ('Noteworthy', Path('/System/Library/Fonts/Noteworthy.ttc'),
     '7785163d9b841cc586d1e696fb4c16a657e16049b092e488a51003e74be8ec21', ['Noteworthy-Light', 'Noteworthy-Bold']),
    ('Apple Chancery', Path('/System/Library/Fonts/Supplemental/Apple Chancery.ttf'),
     'a9145a5dd2ab5b8f1813d22688de78ac1efee135d1c7f3bb30b5d8f30a311d80', ['Apple-Chancery']),
]
BASE = ROOT/'artifacts/font-unknown-gate-v1'
FROZEN_SCENES_SHA = '9fb9c9bc0410c71ffd97d1cacf6f9a80253137ce15478ade912caa1b23a5e5db'
FROZEN_DATA_SHA = '18e8f3238b1e2a9be62e7502d432624987c08e158579348a17209dc0e6cd3e02'

def generate(output):
    from fontTools.ttLib import TTFont, TTCollection
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Supplement output must be new')
    source_path = BASE/'scenes-v4/Scenes.json'
    assert sha(source_path) == FROZEN_SCENES_SHA and sha(BASE/'data-v1/MANIFEST.json') == FROZEN_DATA_SHA
    old = json.loads(source_path.read_text())
    assert not set(NEW_UNKNOWN) & {f['family'] for f in old['fonts']}
    assets = output/'assets'
    assets.mkdir(parents=True)
    fonts = [dict(f) for f in old['fonts'] if canonical(f['family']) in KNOWN_LATIN]
    cmaps, derivations = {}, []
    for f in fonts:
        if f['kind'] == 'asset':
            path = BASE/'scenes-v4/assets'/f['path']
            assert sha(path) == f['sha256']
            shutil.copyfile(path, assets/f['path'])
            face = TTFont(path)
            cmaps[f['id']] = set(face.getBestCmap())
            face.close()
    for family, source, expected, selected in SOURCES:
        assert sha(source) == expected
        collection = TTCollection(source, lazy=False) if source.suffix == '.ttc' else None
        faces = collection.fonts if collection else [TTFont(source)]
        available = {f['name'].getDebugName(6): (i, f) for i, f in enumerate(faces)}
        for ps in selected:
            index, face = available[ps]
            target, actual_ps, proof = derive_conflicting_face(source, index, family, ps, assets)
            derivations.append(proof)
            key = 'supplement_' + ps.lower()
            cmaps[key] = set(face.getBestCmap())
            assert all(ord(c) in cmaps[key] for c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
            fonts.append({'id': key, 'family': family, 'postscript': actual_ps, 'kind': 'asset',
                          'path': target.name, 'sha256': sha(target), 'source_path': str(source), 'original_postscript': ps,
                          'source_derivation': proof, 'ttc_index': index, 'scripts': ['latin'],
                          'weight': int(face['OS/2'].usWeightClass), 'numeric_only': False})
        if collection:
            collection.close()
        else:
            faces[0].close()
    exclusions = [source_path, ROOT/'artifacts/mobile-font-capture/ios-capture-v1/Scenes.json',
                  ROOT/'artifacts/mobile-font-capture/ios-hant-capture-v1/Scenes.json']
    seen = {normalized_text(r['text']) for path in exclusions for page in json.loads(path.read_text())['pages'] for r in page['regions']}
    by_family = {family: [f for f in fonts if f['family'] == family] for family in KNOWN_LATIN+NEW_UNKNOWN}
    rng, pages = random.Random(SEED), []
    for i in range(PAGES):
        page_id = f'ios-unknown-latin-supplement-v1-{i:04d}'
        dark = rng.randrange(5) == 0
        background = rng.choice(['#1C1C1E', '#242426'] if dark else ['#FFFFFF', '#F5F5F7', '#FAFAF8'])
        known_order = KNOWN_LATIN.copy()
        rng.shuffle(known_order)
        regions = []
        for pair, family in enumerate(known_order):
            unknown_family = NEW_UNKNOWN[(i*6+pair) % 3]
            known_font, unknown_font = rng.choice(by_family[family]), rng.choice(by_family[unknown_family])
            numeric = family == 'Alipay Number' or rng.randrange(2) == 0
            for attempt in range(10000):
                text = text_candidate(rng, 'latin', 'bill', numeric, 'simplified', True)
                # Both pair members receive the same numeric prefix. Some
                # handwriting digits (7/9) have negative initial side bearings;
                # the unchanged native collector begins at the region edge.
                # Alipay Number has no space glyph, so a blank inset would
                # introduce fallback. Zero preserves native coverage while
                # interior 7/9 glyphs remain present throughout the corpus.
                if numeric:
                    text = '0' + text
                if normalized_text(text) in seen:
                    continue
                if not all(all(ord(c) in cmaps[f['id']] for c in text) for f in (known_font, unknown_font) if f['id'] in cmaps):
                    continue
                seen.add(normalized_text(text))
                break
            else:
                raise ValueError('No fresh covered Latin text found')
            size = min(rng.choice([13, 15, 17, 19, 21, 23, 25, 27]), int(328/(len(text)*.75+1)))
            color = rng.choice(['#FFFFFF', '#ECECEF', '#8CC8FF', '#FFB4AB', '#81D8B0'] if dark else ['#111111', '#333333', '#555555', '#005BBB', '#B42318', '#1D6B44'])
            for label, font in [(1, known_font), (0, unknown_font)]:
                regions.append({'text': text, 'script': 'latin', 'font_family': font['family'], 'font_postscript': font['postscript'],
                                'font_id': font['id'], 'font_size': size, 'color': color, 'weight': font['weight'], 'language': 'en',
                                'han_orthography': None, 'text_kind': 'numeric' if numeric else 'english', 'gate_label': label,
                                'pair_id': f'{page_id}-pair-{pair}'})
        rng.shuffle(regions)
        for j, r in enumerate(regions):
            top = 66+j*64
            r.update(id=f'{page_id}-r{j:02d}', bbox_points=[24, top, 382, top+52])
        pages.append({'id': page_id, 'content_group_id': page_id, 'split': 'train', 'background': background, 'regions': regions})
    document = {'schema': 'flux-glyph-capture-scenes-v1', 'platform': 'ios', 'canvas_points': [402, 874], 'seed': SEED,
                'fonts': fonts, 'pages': pages, 'gate_labels': ['unknown', 'known'], 'gate_known_families': KNOWN_LATIN,
                'training_only_unknown_families': NEW_UNKNOWN, 'all_pages_training_only': True,
                'paired_rendering': 'same native renderer/page/text/script/color/font size; shuffled row order',
                'android_native_rendering_validated': False, 'ui_content_is_generated': True,
                'numeric_pair_prefix': '0; applied to known and unknown equally to avoid negative initial glyph side bearing'}
    dump(output/'Scenes.json', document)
    required = {f['id'] for f in fonts if f['family'] in NEW_UNKNOWN}
    smoke = []
    while required:
        page = max(pages, key=lambda p: len(required & {r['font_id'] for r in p['regions']}))
        covered = required & {r['font_id'] for r in page['regions']}
        assert covered
        smoke.append(page)
        required -= covered
    dump(output/'SmokeScenes.json', {**document, 'pages': smoke})
    manifest = {'schema': 'flux-glyph-rejection-supplement-scene-source-v1', 'all_pages_training_only': True,
                'seed': SEED, 'planned_pages': PAGES, 'known_families': KNOWN_LATIN, 'unknown_families': NEW_UNKNOWN,
                'labels': ['unknown', 'known'], 'old_data_manifest_sha256': FROZEN_DATA_SHA,
                'excluded_scene_sources': {str(p): sha(p) for p in exclusions},
                'excluded_calibration_test_families': old['gate_unknown_family_splits']['calibration'] + old['gate_unknown_family_splits']['test'],
                'scenes_sha256': sha(output/'Scenes.json'), 'name_only_derivations': derivations,
                'generator_sha256': sha(__file__), 'derivation_helper_sha256': sha(ROOT/'training/capture/generate_unknown_scenes.py'),
                'requested_family_regions': dict(Counter(r['font_family'] for p in pages for r in p['regions'])),
                'renderer': 'planned UIView.draw + CTLineDraw + simctl screenshot', 'old_test_pixels_used_for_training': False,
                'source_family_overlap_with_prior_gate_data': 0, 'normalized_text_overlap_with_excluded_scenes': 0}
    dump(output/'SOURCE_MANIFEST.json', manifest)
    print(json.dumps({'output': str(output), 'pages': len(pages), 'smoke_pages': len(smoke),
                      'scenes_sha256': manifest['scenes_sha256'], 'family_counts': manifest['requested_family_regions']}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    generate(parser.parse_args().output)
