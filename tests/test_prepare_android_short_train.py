import copy
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from training.capture import capture_android as capture_native
from training import prepare_android_short_train as m


REQUESTS = m.ROOT/'artifacts/android-native-short-train-v1/requests-v2/Scenes.json'


def test_real_request_bundle_is_exact_train_only_and_source_bound():
    scenes = json.loads(REQUESTS.read_text())
    labels = [{'page_id': page['id'], 'region_id': region['id'], 'split': 'train'}
              for page in scenes['pages'] for region in page['regions']]
    pages, planned = m._validate_plan(scenes, labels)
    requested = {region['font_id'] for page in pages for region in page['regions']}
    texts = {}
    for font_id in requested:
        texts[font_id] = {region['text'] for page in pages for region in page['regions']
                          if region['font_id'] == font_id}
    bindings = {}
    registry, coverage, sources = m._font_sources(scenes, requested, texts, bindings)
    assert len(pages) == 128 and len(planned) == 1056
    assert scenes['design']['namespace'] == m.NAMESPACE == 'android-native-short-train-v2'
    assert {region['font_size_px'] for page in pages for region in page['regions']} == {36, 45, 54, 66, 81, 96, 108, 120}
    assert all(region['font_size_px'] == region['ios_font_size_points'] * 3
               for page in pages for region in page['regions'])
    assert len(registry) == len(coverage) == len(sources) == 45
    assert all(source['split'] in ('train', 'all') and 'train' in source['allowed_splits']
               for source in sources)
    assert {source['canonical_family'] for source in sources} <= set(m.FAMILIES)
    assert all(Path(path).is_file() and m.sha(path) == digest for path, digest in bindings.items())


def test_request_plan_rejects_missing_rows_and_nontrain_pages():
    scenes = json.loads(REQUESTS.read_text())
    labels = [{'page_id': page['id'], 'region_id': region['id'], 'split': 'train'}
              for page in scenes['pages'] for region in page['regions']]
    with pytest.raises(ValueError, match='exactly cover'):
        m._validate_plan(scenes, labels[:-1])
    changed = copy.deepcopy(scenes)
    changed['pages'][0]['split'] = 'test'
    with pytest.raises(ValueError, match='TRAIN only'):
        m._validate_plan(changed, labels)
    changed = copy.deepcopy(scenes)
    changed['design']['namespace'] = changed['design']['protocol'] = 'android-native-short-train-v1'
    with pytest.raises(ValueError, match='v2 namespace'):
        m._validate_plan(changed, labels)


def native_capture_fixture(tmp_path, glyph_mutation=None):
    font_path = tmp_path/'FixturePS.ttf'
    font_path.write_bytes(b'fixture font bytes; the native proof binds this exact payload')
    license_path = tmp_path/'OFL.txt'
    license_path.write_text('fixture license')
    cmap_path = tmp_path/'FixturePS.cmap.json'
    cmap_path.write_text(json.dumps([ord('A')]))
    font = {
        'id': 'fixture', 'family': 'Roboto', 'training_family': 'Roboto',
        'native_family': 'Roboto', 'postscript': 'FixturePS', 'path': str(font_path),
        'sha256': capture_native.sha(font_path), 'ttc_index': 0, 'split': 'all',
        'allowed_splits': ['train', 'calibration', 'test'], 'scripts': ['latin'],
        'license': 'OFL-1.1',
        'license_files': [{'path': str(license_path), 'sha256': capture_native.sha(license_path)}],
        'cmap': {'path': str(cmap_path), 'sha256': capture_native.sha(cmap_path)},
        'source': {'path': str(font_path), 'sha256': capture_native.sha(font_path)},
    }
    requested = {'id': f'{m.NAMESPACE}-00000-r00', 'font_id': 'fixture', 'text': 'A',
                 'script': 'latin', 'font_size_px': 40, 'color': '#333333',
                 'bbox': [20, 20, 120, 80]}
    page = {'id': f'{m.NAMESPACE}-00000', 'content_group_id': f'{m.NAMESPACE}-g00000',
            'split': 'train', 'background': '#FFFFFF', 'regions': [requested]}
    scenes = {'schema': 'flux-glyph-android-scenes-v1', 'canvas_px': [1080, 2400],
              'families': ['Roboto'] + [f'Family{i}' for i in range(8)] + ['__unknown__'],
              'fonts': [font], 'pages': [page], 'bindings': {},
              'design': {'source_family_to_unified_family': {'Roboto': 'Roboto'}}}
    capture_native.dump(tmp_path/'Scenes.json', scenes)
    (tmp_path/'screenshots').mkdir()
    image_path = tmp_path/'screenshots'/f'{page["id"]}.png'
    image = Image.new('RGB', (1080, 2400), '#FFFFFF')
    ImageDraw.Draw(image).rectangle([32, 32, 52, 62], fill='#333333')
    image.save(image_path)
    (tmp_path/'proofs').mkdir()
    glyph = {'glyph_id': 7, 'font_verified': True, 'position': [32.0, 62.0],
             'bbox': [32, 32, 52, 62],
             'font': {'sha256': font['sha256'], 'postscript': 'FixturePS', 'ttc_index': 0}}
    glyphs = [glyph]
    if glyph_mutation == 'zero':
        glyph['glyph_id'] = 0
    elif glyph_mutation == 'count':
        glyphs.append(copy.deepcopy(glyph))
    elif glyph_mutation == 'bounds':
        glyph['bbox'] = [-10, 32, 52, 62]
    actual = {'id': requested['id'], 'font_id': 'fixture', 'text': 'A',
              'bbox': requested['bbox'], 'ink_bbox': [32, 32, 52, 62],
              'font_size_px': 40, 'color': '#333333', 'font_verified': True,
              'glyphs': glyphs, 'advance_px': 22.0, 'typeface_weight': 400}
    proof = {'schema': 'flux-glyph-android-render-proof-v1', 'canvas_px': [1080, 2400],
             'build_fingerprint': 'fixture', 'sdk_int': 35, 'page_id': page['id'],
             'split': 'train', 'binding_method': 'fixture of native proof schema',
             'regions': [actual]}
    proof_path = tmp_path/'proofs'/f'{page["id"]}.json'
    capture_native.dump(proof_path, proof)
    row = {'schema': 'flux-glyph-android-native-region-v1', 'split': 'train',
           'source_id': 'android:'+page['id'], 'page_id': page['id'],
           'content_group_id': page['content_group_id'], 'region_id': requested['id'],
           'image': f'screenshots/{page["id"]}.png', 'image_sha256': capture_native.sha(image_path),
           'image_width': 1080, 'image_height': 2400, 'bbox': requested['bbox'],
           'ink_bbox': actual['ink_bbox'], 'text': 'A', 'script': 'latin', 'font_id': 'fixture',
           'font_family': 'Roboto', 'training_family': 'Roboto', 'font_face': 'FixturePS',
           'font_file_sha256': font['sha256'], 'ttc_index': 0, 'font_size_screen_px': 40,
           'color': '#333333', 'background': '#FFFFFF', 'native_font_verified': True,
           'native_rendering': 'Android Canvas + TextRunShaper',
           'proof': f'proofs/{page["id"]}.json'}
    (tmp_path/'labels.jsonl').write_text(json.dumps(row)+'\n')
    manifest = {'schema': capture_native.SCHEMA,
                'families': scenes['families'], 'fonts': scenes['fonts'], 'bindings': {},
                'scenes': {'path': 'Scenes.json', 'sha256': capture_native.sha(tmp_path/'Scenes.json')},
                'labels': {'path': 'labels.jsonl', 'sha256': capture_native.sha(tmp_path/'labels.jsonl')},
                'files': [{'path': row['image'], 'sha256': row['image_sha256']},
                          {'path': row['proof'], 'sha256': capture_native.sha(proof_path)}],
                'pages': 1, 'regions': 1, 'split_counts': {'train': 1},
                'device': {'build_fingerprint': 'fixture', 'sdk_int': 35},
                'all_native_fonts_verified': True}
    capture_native.dump(tmp_path/'CAPTURE_MANIFEST.json', manifest)
    return row, font


def test_actual_capture_loader_and_native_proof_contract(tmp_path):
    row, font = native_capture_fixture(tmp_path)
    loaded = capture_native.load_capture(tmp_path)
    capture_native.verify_region(tmp_path, loaded['rows'][0])
    evidence = m._native_glyph_proof(tmp_path, row, font, {ord('A')})
    assert evidence['glyph_count'] == 1 and evidence['text_kind'] == 'latin'
    crop = Image.open(tmp_path/row['image']).convert('RGB').crop(m.source_crop_bbox(row))
    try:
        for view in m.VIEWS:
            transformed, _, _ = m.apply_view(crop, view)
            try:
                processed = m.preprocess_region(transformed)
                assert processed['status'] == 'ok'
                assert processed['whole_width_covered'] is True
                assert processed['tiles'].shape == (1, 1, 64, 256)
            finally:
                transformed.close()
    finally:
        crop.close()


@pytest.mark.parametrize('mutation,match', [
    ('zero', 'zero glyph'), ('count', 'one-for-one'), ('bounds', 'outside'),
])
def test_native_proof_mutations_fail_closed(tmp_path, mutation, match):
    row, font = native_capture_fixture(tmp_path, mutation)
    loaded = capture_native.load_capture(tmp_path)
    if mutation == 'zero':
        with pytest.raises(ValueError, match='fallback or missing glyph'):
            capture_native.verify_region(tmp_path, loaded['rows'][0])
    else:
        capture_native.verify_region(tmp_path, loaded['rows'][0])
        with pytest.raises(ValueError, match=match):
            m._native_glyph_proof(tmp_path, row, font, {ord('A')})


def test_only_short_unmixed_visible_classification_text_is_accepted():
    assert [m._script_kind(value) for value in ('字', 'Ab', '09')] == ['han', 'latin', 'numeric']
    for invalid in ('A1', 'A!', 'A B', 'abcde'):
        with pytest.raises(ValueError):
            m._script_kind(invalid)


def test_cross_family_pixel_conflict_drops_all_views_of_both_identities():
    rows, tensors = [], []
    for source, family in (('s1', 'Roboto'), ('s2', '__unknown__')):
        for view in m.VIEWS:
            rows.append({'source_id': source, 'region_id': 'r', 'family': family,
                         'view': view, 'tiles_sha256': 'same', 'region_rgb_sha256': 'same-region'})
            tensors.append(source+view)
    rows.append({'source_id': 's3', 'region_id': 'r', 'family': 'FandolKai',
                 'view': 'native', 'tiles_sha256': 'unique', 'region_rgb_sha256': 'unique-region'})
    tensors.append('kept')
    kept, kept_tensors, conflicts = m.remove_conflicting_identities(rows, tensors)
    assert conflicts == 2 and [(row['source_id'], row['family']) for row in kept] == [('s3', 'FandolKai')]
    assert kept_tensors == ['kept']
