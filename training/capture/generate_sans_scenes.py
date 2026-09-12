#!/usr/bin/env python3
"""Plan paired native sans-font captures with explicit source identities.

This creates requests, not font truth or Android screenshots. The existing
native collector must verify actual fonts, glyph coverage and screenshot bytes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.capture.generate_scenes import ios_fonts, normalized_text, text_candidate, sha, write_json

SEED = 2026091299
COUNTS = {'train': 300, 'calibration': 60, 'test': 60}
HAN = ['HarmonyOS Sans SC', 'MiSans', 'Noto Sans CJK SC', 'OPPO Sans', 'PingFang']
LATIN = ['HarmonyOS Sans SC', 'MiSans', 'OPPO Sans', 'SF Pro', 'Helvetica', 'Alipay Number', 'Roboto']
BASE = ROOT/'artifacts/font-sans-v1'
OLD = ROOT/'artifacts/mobile-font-capture/ios-hant-capture-v1'
OEM = ROOT.parent/'alipay-ai-inference/runs/font-family-identity-v1-20260908/android-source'
ROBOTO_SHA = 'd7598e12c5dbef095ff8272cfc55da0250bd07fbdecbac8a530b9b277872a134'


def canonical(family):
    return 'PingFang' if family.startswith('PingFang ') else family


def numeric_text(rng):
    """Content choices are shared across competing families, including Alipay."""
    kind = rng.randrange(4)
    if kind == 0:
        return f'-{rng.randrange(10, 9900)}.{rng.randrange(100):02d}'
    if kind == 1:
        return f'{rng.randrange(2000, 2040)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}'
    if kind == 2:
        return f'{rng.randrange(24):02d}:{rng.randrange(60):02d}:{rng.randrange(60):02d}'
    return ''.join(rng.choice('0123456789') for _ in range(rng.randrange(7, 12)))


def generate(output):
    import fontTools
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new scene output directory')
    assets = output/'assets'
    assets.mkdir(parents=True)
    old_scene = json.loads((OLD/'Scenes.json').read_text())
    fonts = [dict(f) for f in old_scene['fonts'] if canonical(f['family']) in HAN+LATIN]
    cmaps, sources, derivations = {}, [], []

    def add_file(source, family, provenance):
        font = TTFont(source)
        ps = font['name'].getDebugName(6)
        target = assets/source.name
        if target.exists() and sha(target) != sha(source):
            raise ValueError('Conflicting font filename')
        shutil.copyfile(source, target)
        row = {'id': 'sans_' + ps, 'family': family, 'postscript': ps, 'kind': 'asset',
               'path': target.name, 'sha256': sha(target), 'source_path': str(source),
               'ttc_index': 0, 'weight': int(font['OS/2'].usWeightClass), 'variable': False,
               'scripts': ['han'] if family == 'Noto Sans CJK SC' else ['latin'] if family == 'Roboto' else ['han', 'latin'],
               'numeric_only': False, 'source_provenance': provenance}
        fonts.append(row)
        cmaps[row['id']] = set(font.getBestCmap())
        sources.append({'font_id': row['id'], 'sha256': row['sha256'], **provenance})
        font.close()

    for row in fonts:
        if row['kind'] == 'asset':
            source = Path(row.get('source_path') or row['path'])
            if not source.is_file():
                source = ROOT/'artifacts/mobile-font-capture/ios-hant-scenes-v1/assets'/Path(row['path']).name
            assert sha(source) == row['sha256']
            target = assets/Path(row['path']).name
            shutil.copyfile(source, target)
            row['path'] = target.name
            with TTFont(target) as face:
                cmaps[row['id']] = set(face.getBestCmap())

    archive = OEM/'HarmonyOS-Sans.zip'
    assert sha(archive) == 'fb02c86e358cd9aad8d4dfa957ee502381e7ee2e94499a9133add4324b6ce69a'
    for weight in ('Thin', 'Light'):
        member = f'HarmonyOS Sans/HarmonyOS_Sans_SC/HarmonyOS_Sans_SC_{weight}.ttf'
        target = BASE/'sources'/Path(member).name
        with zipfile.ZipFile(archive) as package:
            raw = package.read(member)  # zipfile also validates the member CRC.
            target.write_bytes(raw)
        add_file(target, 'HarmonyOS Sans SC', {'archive': str(archive), 'archive_sha256': sha(archive),
                                             'archive_member': member, 'source_manifest': str(OEM/'source-manifest.json'),
                                             'source_manifest_sha256': sha(OEM/'source-manifest.json')})
    downloads = BASE/'sources/DOWNLOAD_MANIFEST.json'
    download_manifest = json.loads(downloads.read_text())
    for row in download_manifest['files']:
        source = BASE/'sources'/row['file']
        assert sha(source) == row['sha256']
        add_file(source, 'MiSans' if source.name.startswith('MiSans') else 'Noto Sans CJK SC',
                 {'download_manifest': str(downloads), 'download_manifest_sha256': sha(downloads), **row})

    roboto = ROOT/'artifacts/neural-font-v1/downloads/roboto-google-fonts/Roboto[wdth,wght].ttf'
    assert sha(roboto) == ROBOTO_SHA
    for width in (75, 100):
        for weight in (300, 400, 500, 700, 900):
            with TTFont(roboto, recalcTimestamp=False) as variable:
                static = instantiateVariableFont(variable, {'wdth': width, 'wght': weight}, inplace=False)
            ps = f'FluxSans-Roboto-W{weight}-D{width}'
            # Keep true family and shapes; assign a unique PS name so registration
            # cannot silently select a previously registered variable default.
            for record in static['name'].names:
                if record.nameID in (3, 4, 6):
                    record.string = ps.encode(record.getEncoding())
            static.recalcTimestamp = False
            target = BASE/'sources'/f'{ps}.ttf'
            static.save(target)
            static.close()
            proof = {'source': str(roboto), 'source_sha256': ROBOTO_SHA,
                     'source_manifest': str(roboto.parent/'SOURCE_MANIFEST.json'),
                     'source_manifest_sha256': sha(roboto.parent/'SOURCE_MANIFEST.json'),
                     'operation': 'fontTools varLib.instantiateVariableFont then unique name IDs 3/4/6',
                     'axes': {'wdth': width, 'wght': weight}, 'fonttools_version': fontTools.__version__,
                     'derived_sha256': sha(target), 'derived_postscript': ps,
                     'glyphs_are_original_variable_font_instances': True}
            add_file(target, 'Roboto', proof)
            derivations.append(proof)

    # Read text metadata only, across all historical splits and abandoned plans.
    exclusions = sorted(p for p in (ROOT/'artifacts').rglob('Scenes.json')
                        if 'font-sans-v1' not in p.parts)
    excluded_text = {normalized_text(r['text']) for p in exclusions
                     for page in json.loads(p.read_text()).get('pages', []) for r in page['regions']}
    seen = set(excluded_text)
    by_family = {family: [f for f in fonts if canonical(f['family']) == family] for family in HAN+LATIN}
    rng, pages = random.Random(SEED), []
    for split, count in COUNTS.items():
        for index in range(count):
            pid = f'ios-sans-v1-{split}-{index:04d}'
            weight = (300, 400, 500, 600, 700, 200, 900)[index % 7]
            selected = {}
            for family in sorted(set(HAN+LATIN)):
                choices = by_family[family]
                delta = min(abs(f['weight']-weight) for f in choices)
                selected[family] = rng.choice([f for f in choices if abs(f['weight']-weight) == delta])
            dark = index % 5 == 0
            background = rng.choice(['#19191C', '#242426'] if dark else ['#FFFFFF', '#F5F5F7', '#FAFAF8'])
            color = rng.choice(['#FFFFFF', '#ECECEF', '#8CC8FF', '#FFB4AB'] if dark else ['#111111', '#333333', '#555555', '#005BBB', '#B42318'])
            regions = []

            def fresh(script, relevant, numeric=False, traditional=False):
                for _ in range(20000):
                    value = numeric_text(rng) if numeric else text_candidate(rng, script, 'bill', False,
                                          'traditional' if traditional else 'simplified', True)
                    if normalized_text(value) in seen:
                        continue
                    if not all(all(ord(c) in cmaps[f['id']] for c in value)
                               for f in relevant if f['id'] in cmaps):
                        continue
                    seen.add(normalized_text(value))
                    return value
                raise ValueError('Could not choose fresh covered text')

            traditional = index % 3 == 0
            han_text = fresh('han', [selected[f] for f in HAN], traditional=traditional)
            numeric = index % 3 != 0
            shared_latin = [f for f in LATIN if numeric or f != 'Alipay Number']
            latin_text = fresh('latin', [selected[f] for f in shared_latin], numeric=numeric)
            alipay_text = latin_text if numeric else fresh('latin', [selected['Alipay Number']], numeric=True)
            han_size = min(rng.choice([14, 17, 20, 23, 26, 29, 32]), int(326/(len(han_text)+.8)))
            latin_size = min(rng.choice([14, 18, 22, 26, 30, 34, 38]), int(326/(len(latin_text)*.72+1)))
            for script, families, value, size in [('han', HAN, han_text, han_size), ('latin', LATIN, latin_text, latin_size)]:
                for family in families:
                    font = selected[family]
                    text = alipay_text if family == 'Alipay Number' else value
                    regions.append({'text': text, 'script': script, 'font_family': font['family'],
                                    'font_postscript': font['postscript'], 'font_id': font['id'],
                                    'font_size': size, 'color': color, 'weight': font['weight'],
                                    'language': 'zh-Hant' if script == 'han' and traditional else 'zh-Hans' if script == 'han' else 'en',
                                    'han_orthography': ('traditional' if traditional else 'simplified') if script == 'han' else None,
                                    'text_kind': 'numeric' if family == 'Alipay Number' or script == 'latin' and numeric else 'english' if script == 'latin' else 'han',
                                    'comparison_group': f'{pid}-{script}'})
            rng.shuffle(regions)
            for i, row in enumerate(regions):
                top = 62+i*64
                row.update(id=f'{pid}-r{i:02d}', bbox_points=[28, top, 378, top+54])
            pages.append({'id': pid, 'content_group_id': pid, 'split': split, 'background': background, 'regions': regions})
    document = {'schema': 'flux-glyph-capture-scenes-v1', 'platform': 'ios', 'canvas_points': [402, 874],
                'seed': SEED, 'fonts': fonts, 'pages': pages, 'ui_content_is_generated': True,
                'renderer': 'controlled UIView.draw + CTLineDraw + simctl screenshot',
                'paired_content': 'five Han and seven numeric families share text, size, weight target, color, page; shuffled rows',
                'android_native_rendering_validated': False, 'font_is_not_platform_truth': True}
    write_json(output/'Scenes.json', document)
    required = {f['id'] for f in fonts}
    # Smoke uses training pages only: fresh TEST pixels stay unopened for modeling.
    smoke = []
    while required:
        page = max((p for p in pages if p['split'] == 'train'), key=lambda p: len(required & {r['font_id'] for r in p['regions']}))
        covered = required & {r['font_id'] for r in page['regions']}
        if not covered:
            raise ValueError('Font registry contains an unplanned face')
        smoke.append(page)
        required -= covered
    write_json(output/'SmokeScenes.json', {**document, 'pages': smoke})
    manifest = {'schema': 'flux-glyph-sans-scenes-source-v1', 'scenes_sha256': sha(output/'Scenes.json'),
                'generator_sha256': sha(__file__), 'seed': SEED, 'split_counts': COUNTS,
                'old_scene_sources': {str(p): sha(p) for p in exclusions},
                'old_text_count': len(excluded_text), 'old_text_overlap': 0,
                'old_test_pixels_opened': False, 'android_native_rendering_validated': False,
                'base_scene_sha256': sha(OLD/'Scenes.json'), 'new_font_sources': sources,
                'variable_font_derivations': derivations, 'font_family_counts': dict(Counter(canonical(r['font_family']) for p in pages for r in p['regions'])),
                'font_face_counts': dict(Counter(r['font_id'] for p in pages for r in p['regions']))}
    write_json(output/'SOURCE_MANIFEST.json', manifest)
    print(json.dumps({'pages': len(pages), 'smoke_pages': len(smoke), 'fonts': len(fonts),
                      'scenes_sha256': manifest['scenes_sha256'], 'family_counts': manifest['font_family_counts']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    generate(parser.parse_args().output)
