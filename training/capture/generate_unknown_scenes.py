#!/usr/bin/env python3
"""Prepare paired native known/unknown font scenes; this tool renders no pixels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.capture.generate_scenes import normalized_text, text_candidate

SEED = 2026091295
KNOWN = ['HarmonyOS Sans SC', 'MiSans', 'Noto Sans CJK SC', 'OPPO Sans',
         'PingFang', 'SF Pro', 'Helvetica', 'Alipay Number']
UNKNOWN_HAN = {'train': ['Kaiti SC', 'Songti SC', 'HanziPen SC'],
               'calibration': ['Baoli SC', 'Weibei SC'],
               'test': ['Wawati SC', 'Libian SC']}
UNKNOWN_LATIN = {'train': ['Times', 'Courier'], 'calibration': ['Lucida Grande', 'Comic Sans MS'],
                 'test': ['Seravek', 'Chalkboard']}
UNKNOWN_SPLITS = {s: UNKNOWN_HAN[s] + UNKNOWN_LATIN[s] for s in UNKNOWN_HAN}
COUNTS = {'train': 240, 'calibration': 60, 'test': 60}
FONT8 = Path('/System/Library/AssetsV2/com_apple_MobileAsset_Font8')
UNKNOWN = [
    ('Kaiti SC', FONT8/'88d6cc32a907955efa1d014207889413890573be.asset/AssetData/Kaiti.ttc',
     'cfd20ab9514ad926699811cdfc92963478adc65b370965657b56dd1338686570', ['STKaitiSC-Regular', 'STKaitiSC-Bold']),
    ('Songti SC', Path('/System/Library/Fonts/Supplemental/Songti.ttc'),
     '6873ac2ccab5c2e74d87d6b690f3773098dd6a6238805363a3b3567f2caf6f47', ['STSongti-SC-Light', 'STSongti-SC-Regular', 'STSongti-SC-Bold']),
    ('HanziPen SC', FONT8/'a3c69464b629577766c23bcdb12ffbfe3759b923.asset/AssetData/Hanzipen.ttc',
     'cec58b8918ec1468ebb29e9c6d0c096f893e5a9c416c6ed63e3098900389213a', ['HanziPenSC-W3', 'HanziPenSC-W5']),
    ('Baoli SC', FONT8/'70875a1270987c17cfd6f40cd3d755ec04d03b33.asset/AssetData/Baoli.ttc',
     '63bf43fc567d4d28d66e64b904a72f9c2cebd3e3e594cfa37d801a441128bab8', ['STBaoliSC-Regular']),
    ('Weibei SC', FONT8/'c745f84f5eb15b1f594d3769dc86146fccee61ff.asset/AssetData/WeibeiSC-Bold.otf',
     '4d1a6fc6fd5bbafbd50d7931d185436a3c59f1fbfbb4404807004d57728afb6a', ['WeibeiSC-Bold']),
    ('Wawati SC', FONT8/'37618f984b05fb22375ca0987912653aba363389.asset/AssetData/WawaSC-Regular.otf',
     '3dd67c3f9e9a391aa7ce4d5656e6102a87b7f8ea73697e4d9f498689dab1dec3', ['DFWaWaSC-W5']),
    ('Libian SC', FONT8/'a304e3396d019087ab67af77f5e398977529007d.asset/AssetData/Libian.ttc',
     'db436cfe5b8d2cd1a6400eda6724f117c9b2b87fa1d0ffdd41d4cc35d5a7e824', ['STLibianSC-Regular']),
    ('Times', Path('/System/Library/Fonts/Times.ttc'),
     '20e3dc89912f4b37f2c29add764855164efafb4c66ca32551d9eb52c7411c7c7', ['Times-Roman', 'Times-Bold']),
    ('Courier', Path('/System/Library/Fonts/Courier.ttc'),
     '449a068ddd88f179b50b4d6dbb86aa08a30981148b510d5cc30d4ff0cb29e9a6', ['Courier', 'Courier-Bold']),
    ('Lucida Grande', Path('/System/Library/Fonts/LucidaGrande.ttc'),
     'd1030f2b3c967b740d236f2a59d38b69705eda36e0633c88eaef06d8b0833ab6', ['LucidaGrande', 'LucidaGrande-Bold']),
    ('Comic Sans MS', Path('/System/Library/Fonts/Supplemental/Comic Sans MS.ttf'),
     'a6f559bc059fe6d0e68826e2daee0db31194844ed6145d5cc31a610c4e2700d2', ['ComicSansMS']),
    ('Seravek', Path('/System/Library/Fonts/Supplemental/Seravek.ttc'),
     'aee4895b60f5e1512a61277cfca2267a0e3445261196c2701d383c83670e9ec9', ['Seravek', 'Seravek-Light', 'Seravek-Bold']),
    ('Chalkboard', Path('/System/Library/Fonts/Supplemental/Chalkboard.ttc'),
     'c1ce8c38124fff79fa2de8e4e9003208a41821a254d2eeee5162672dbc3f3c6c', ['Chalkboard', 'Chalkboard-Bold']),
]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def canonical(family):
    return 'PingFang' if family in ('PingFang SC', 'PingFang TC', 'PingFang HK') else family

def derive_conflicting_face(source, index, family, original_ps, assets):
    """Change names only; bind every other font table to the original TTC face."""
    from fontTools.ttLib import TTFont
    font = TTFont(source, fontNumber=index, recalcTimestamp=False)
    tags = set(font.reader.keys())
    unchanged = {tag: hashlib.sha256(font.reader[tag]).hexdigest() for tag in tags - {'name', 'head', 'DSIG'}}
    original_head = font.reader['head']
    derived_ps = 'FluxGate-' + original_ps
    family_name = 'FluxGate-' + family.replace(' ', '')
    changes = {1: family_name, 3: derived_ps + '-' + sha(source)[:16], 4: derived_ps,
               6: derived_ps, 16: family_name, 18: derived_ps, 21: family_name}
    for record in font['name'].names:
        if record.nameID in changes:
            record.string = changes[record.nameID].encode(record.getEncoding())
    for name_id, value in changes.items():
        font['name'].setName(value, name_id, 3, 1, 0x409)
        font['name'].setName(value, name_id, 1, 0, 0)
    dsig_removed = 'DSIG' in font
    if dsig_removed:
        del font['DSIG']
    target = assets/(derived_ps + '.ttf')
    font.save(target)
    font.close()
    check = TTFont(target, recalcTimestamp=False)
    assert set(check.reader.keys()) == tags - ({'DSIG'} if dsig_removed else set())
    for tag, expected in unchanged.items():
        assert hashlib.sha256(check.reader[tag]).hexdigest() == expected, f'Derivation changed glyph/metrics table {tag}'
    derived_head = check.reader['head']
    assert original_head[:8] + original_head[12:] == derived_head[:8] + derived_head[12:], 'Unexpected head change'
    assert check['name'].getDebugName(6) == derived_ps
    check.close()
    proof = {'operation': 'name_only_unique_PostScript_derivation', 'original_source': str(source),
             'original_source_sha256': sha(source), 'original_face_index': index, 'original_postscript': original_ps,
             'derived_file': target.name, 'derived_sha256': sha(target), 'derived_postscript': derived_ps,
             'unchanged_table_sha256': unchanged, 'head_only_checksum_adjustment_changed': True,
             'DSIG_removed_after_name_change': dsig_removed, 'glyph_outlines_cmap_metrics_byte_identical': True}
    return target, derived_ps, proof

def generate(output):
    from fontTools.ttLib import TTFont, TTCollection
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Scene output must be new')
    assets = output/'assets'
    assets.mkdir(parents=True)
    old_scenes = [ROOT/'artifacts/mobile-font-capture'/folder/'Scenes.json'
                  for folder in ('ios-capture-v1', 'ios-hant-capture-v1')]
    old = [json.loads(path.read_text()) for path in old_scenes]
    fonts = [dict(f) for f in old[1]['fonts']]
    cmaps = {}
    sources, derivations = [], []
    for f in fonts:
        if f['kind'] == 'asset':
            source = ROOT/'artifacts/mobile-font-capture/ios-hant-scenes-v1/assets'/f['path']
            assert sha(source) == f['sha256']
            if not (assets/f['path']).exists():
                shutil.copyfile(source, assets/f['path'])
            font = TTFont(source)
            assert font['name'].getDebugName(6) == f['postscript']
            cmaps[f['id']] = set(font.getBestCmap())
            font.close()
    for family, source, expected_sha, selected in UNKNOWN:
        assert sha(source) == expected_sha, str(source)
        shutil.copyfile(source, assets/source.name)
        collection = TTCollection(source, lazy=False) if source.suffix == '.ttc' else None
        faces = collection.fonts if collection else [TTFont(source)]
        available = {f['name'].getDebugName(6): (i, f) for i, f in enumerate(faces)}
        for ps in selected:
            index, font = available[ps]
            key = 'unknown_' + ps.lower()
            cmaps[key] = set(font.getBestCmap())
            scripts = ['han', 'latin'] if family in sum(UNKNOWN_HAN.values(), []) else ['latin']
            probe = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' + ('字体识别' if 'han' in scripts else '')
            assert all(ord(c) in cmaps[key] for c in probe)
            target, actual_ps, proof = (source, ps, None)
            if family in ('Courier', 'Seravek'):
                target, actual_ps, proof = derive_conflicting_face(source, index, family, ps, assets)
                derivations.append(proof)
            fonts.append({'id': key, 'family': family, 'postscript': actual_ps, 'kind': 'asset',
                          'path': target.name, 'sha256': sha(target), 'source_path': str(source),
                          'original_postscript': ps, 'source_derivation': proof,
                          'ttc_index': index, 'scripts': scripts,
                          'weight': int(font['OS/2'].usWeightClass) if 'OS/2' in font else (700 if font['head'].macStyle & 1 else 400),
                          'numeric_only': False})
        sources.append({'family': family, 'path': str(source), 'sha256': expected_sha,
                        'source_kind': 'locally_installed_Apple_font_asset', 'postscript_names': selected,
                        'unknown_partition': next(s for s, fs in UNKNOWN_SPLITS.items() if family in fs)})
        if collection:
            collection.close()
        else:
            faces[0].close()
    rng = random.Random(SEED)
    seen = {normalized_text(r['text']) for doc in old for p in doc['pages'] for r in p['regions']}
    forbidden_count = len(seen)
    by_family = {}
    for f in fonts:
        by_family.setdefault(canonical(f['family']), []).append(f)
    pages = []
    counts = {s: Counter() for s in COUNTS}
    for split, page_count in COUNTS.items():
        pair_index = 0
        script_indices = Counter()
        known_order = KNOWN.copy()
        rng.shuffle(known_order)
        for page_index in range(page_count):
            page_id = f'ios-unknown-v1-{split}-{page_index:04d}'
            dark = rng.randrange(5) == 0
            background = rng.choice(['#1C1C1E', '#242426'] if dark else ['#FFFFFF', '#F5F5F7', '#FAFAF8'])
            regions = []
            for pair in range(6):
                family = known_order[pair_index % len(known_order)]
                pair_index += 1
                known_font = rng.choice(by_family[family])
                script = rng.choice(known_font['scripts'])
                unknown_pool = UNKNOWN_HAN[split] if script == 'han' else UNKNOWN_LATIN[split]
                unknown = unknown_pool[script_indices[script] % len(unknown_pool)]
                script_indices[script] += 1
                unknown_font = rng.choice(by_family[unknown])
                numeric = script == 'latin' and (family == 'Alipay Number' or rng.randrange(2) == 0)
                orthography = rng.choice(['simplified', 'traditional'])
                for attempt in range(10000):
                    value = text_candidate(rng, script, rng.choice(['bill', 'settings', 'chat']), numeric, orthography, True)
                    if normalized_text(value) in seen:
                        continue
                    if not all(all(ord(c) in cmaps[f['id']] for c in value) for f in [known_font, unknown_font] if f['id'] in cmaps):
                        continue
                    seen.add(normalized_text(value))
                    break
                else:
                    raise ValueError('No split-isolated fully covered text found')
                size = min(rng.choice([13, 15, 17, 19, 21, 23, 25, 27]), int(328/(len(value)*(1.0 if script == 'han' else .75)+1)))
                color = rng.choice(['#FFFFFF', '#ECECEF', '#8CC8FF', '#FFB4AB', '#81D8B0'] if dark else ['#111111', '#333333', '#555555', '#005BBB', '#B42318', '#1D6B44'])
                for label, font in [(1, known_font), (0, unknown_font)]:
                    regions.append({'text': value, 'script': script, 'font_family': font['family'],
                                    'font_postscript': font['postscript'], 'font_id': font['id'],
                                    'font_size': size, 'color': color, 'weight': font['weight'],
                                    'language': ('zh-Hant' if orthography == 'traditional' else 'zh-Hans') if script == 'han' else 'en',
                                    'han_orthography': orthography if script == 'han' else None,
                                    'text_kind': 'numeric' if numeric else ('english' if script == 'latin' else 'han'),
                                    'gate_label': label, 'pair_id': f'{page_id}-pair-{pair}'})
                    counts[split][font['family']] += 1
            rng.shuffle(regions)
            for i, region in enumerate(regions):
                top = 66 + i*64
                region.update(id=f'{page_id}-r{i:02d}', bbox_points=[24, top, 382, top+52])
            pages.append({'id': page_id, 'content_group_id': page_id, 'split': split,
                          'background': background, 'regions': regions})
    document = {'schema': 'flux-glyph-capture-scenes-v1', 'platform': 'ios', 'canvas_points': [402, 874],
                'seed': SEED, 'fonts': fonts, 'pages': pages, 'gate_known_families': KNOWN,
                'gate_unknown_family_splits': UNKNOWN_SPLITS,
                'gate_labels': ['unknown', 'known'],
                'label_policy': '0 source-verified family outside R17 eight classes; 1 known R17 family',
                'text_isolation': 'new normalized line combinations disjoint from both prior native datasets; paired text only within one split',
                'paired_rendering': 'each known/unknown pair shares text, size, color and background; randomized row order',
                'ui_content_is_generated': True, 'android_native_rendering_validated': False}
    dump(output/'Scenes.json', document)
    smoke = dict(document)
    # Check source registration for every new unknown face; no classifier is run.
    needed = {f['id'] for f in fonts if f['family'] in sum(UNKNOWN_SPLITS.values(), [])}
    smoke['pages'] = []
    while needed:
        best = max(pages, key=lambda p: len(needed & {r['font_id'] for r in p['regions']}))
        covered = needed & {r['font_id'] for r in best['regions']}
        assert covered, 'Unknown source face was never assigned a capture region'
        smoke['pages'].append(best)
        needed -= covered
    dump(output/'SmokeScenes.json', smoke)
    manifest = {'schema': 'flux-glyph-native-unknown-scene-source-v1', 'seed': SEED,
                'scenes_sha256': sha(output/'Scenes.json'), 'smoke_scenes_sha256': sha(output/'SmokeScenes.json'),
                'generator_sha256': sha(__file__), 'old_scenes_sha256': {str(p): sha(p) for p in old_scenes},
                'old_normalized_text_count': forbidden_count, 'new_unique_text_count': len(seen)-forbidden_count,
                'known_families': KNOWN, 'labels': ['unknown', 'known'],
                'unknown_family_splits': UNKNOWN_SPLITS, 'unknown_sources': sources, 'name_only_derivations': derivations,
                'planned_pages': COUNTS, 'planned_family_regions': counts,
                'heldout_family_scope': 'unseen to this binary gate; historical backbone exposure is not claimed absent',
                'test_metadata_read_for_text_exclusion_only': True, 'old_test_pixels_used_for_training': False,
                'user_screenshots_used_for_training': False, 'renderer': 'planned UIView.draw + CTLineDraw + simctl screenshot'}
    dump(output/'SOURCE_MANIFEST.json', manifest)
    print(json.dumps({'output': str(output), 'pages': len(pages), 'regions': sum(sum(x.values()) for x in counts.values()),
                      'scenes_sha256': manifest['scenes_sha256'], 'family_counts': counts}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    generate(parser.parse_args().output)
