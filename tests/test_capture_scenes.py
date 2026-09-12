"""Check capture-request splitting and labels before expensive native capture."""
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('generate_scenes', Path(__file__).parents[1] / 'training/capture/generate_scenes.py')
scenes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scenes)


def test_thousand_pages_are_disjoint_bounded_and_reproducible():
    content = scenes.content_pages()
    document = scenes.scene_document('ios', scenes.ios_fonts(), content)
    assert document == scenes.scene_document('ios', scenes.ios_fonts(), scenes.content_pages())
    assert Counter(page['split'] for page in document['pages']) == {'train': 800, 'calibration': 100, 'test': 100}
    seen = set()
    sources = set()
    font_text = defaultdict(set)
    for page in document['pages']:
        assert page['source_id'] not in sources
        sources.add(page['source_id'])
        assert 10 <= len(page['regions']) <= 12
        for region in page['regions']:
            text = ''.join(region['text'].split())
            assert text not in seen
            seen.add(text)
            assert all('\u4e00' <= c <= '\u9fff' for c in text) if region['script'] == 'han' else text.isascii() and text.isalnum()
            left, top, right, bottom = region['bbox_points']
            assert 0 <= left < right <= 402 and 0 <= top < bottom <= 874
            assert 10 <= region['font_size'] <= 27
            font_text[(page['split'], region['font_id'])].add(text)
    # Every face sees many distinct combinations even in held-out splits.
    assert len(font_text) == len(document['fonts']) * 3
    assert min(map(len, font_text.values())) >= 20


def test_inventory_removes_unavailable_named_fonts(tmp_path):
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps({'postscript_names': ['PingFangSC-Regular', 'Helvetica']}))
    fonts = scenes.ios_fonts(inventory)
    assert {font['postscript'] for font in fonts} == {'PingFangSC-Regular', 'Helvetica', '-system'}
    assert len(fonts) == 7
    assert all(font['sha256'] is None for font in fonts)


def test_numeric_asset_does_not_label_english():
    fonts = scenes.ios_fonts()
    fonts.append({'id': 'alipay', 'family': 'Alipay Number', 'postscript': 'AlipayNumber-Regular',
                  'kind': 'asset', 'path': '/sample/font.ttf', 'sha256': 'a' * 64,
                  'scripts': ['latin'], 'weight': 400, 'numeric_only': True,
                  'cmap_codepoints': list(range(ord('0'), ord('9') + 1))})
    document = scenes.scene_document('ios', fonts, scenes.content_pages(100))
    rows = [region for page in document['pages'] for region in page['regions'] if region['font_id'] == 'alipay']
    assert rows and all(row['text'].isdigit() and row['text_kind'] == 'numeric' for row in rows)
    fonts[0]['family'] = 'DeviceBrandIsNotAFont'
    with pytest.raises(ValueError, match='FAMILIES'):
        scenes.scene_document('ios', fonts, scenes.content_pages(100))


def test_traditional_registry_is_append_only_and_inventory_verified(tmp_path):
    assert scenes.capture_families() == scenes.families() + ['PingFang TC', 'PingFang HK']
    with pytest.raises(ValueError, match='inventory'):
        scenes.ios_fonts(include_traditional=True)
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps({'postscript_names': ['PingFangSC-Regular', 'PingFangTC-Regular', 'Helvetica']}))
    fonts = scenes.ios_fonts(inventory, include_traditional=True)
    named = {font['postscript']: font['family'] for font in fonts}
    assert named['PingFangTC-Regular'] == 'PingFang TC'
    assert not any(font['family'] == 'PingFang HK' for font in fonts)


def test_mixed_traditional_content_is_independent_of_font_and_old_groups(tmp_path):
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps({'postscript_names': [f'PingFang{variant}-{suffix}' for variant in ('SC','TC','HK')
                                                        for suffix in ('Light','Regular','Medium','Semibold')] + ['Helvetica']}))
    old = scenes.content_pages(100)
    forbidden = {r['text'] for page in old for r in page['regions']}
    content = scenes.content_pages(1000, 2026091291, 'mixed', 'hant-v2', forbidden)
    document = scenes.scene_document('ios', scenes.ios_fonts(inventory, include_traditional=True), content, 2026091291)
    assert not {p['content_group_id'] for p in content} & {p['content_group_id'] for p in old}
    new_text = {r['text'] for page in content for r in page['regions']}
    assert not new_text & forbidden
    groups = defaultdict(set)
    traditional_text = ''
    for page in document['pages']:
        for row in page['regions']:
            if row['script'] == 'han':
                groups[(page['split'], row['font_id'])].add(row['han_orthography'])
                assert row['language'] == ('zh-Hant' if row['han_orthography']=='traditional' else 'zh-Hans')
                if row['han_orthography']=='traditional':
                    traditional_text += row['text']
    assert all(values == {'simplified','traditional'} for values in groups.values())
    assert all(c in traditional_text for c in '帳單詳餘轉網絡醫療關閉語設')


def test_smoke_scales_past_twenty_four_faces():
    fonts = scenes.ios_fonts()
    for index in range(20):
        fonts.append(dict(fonts[0], id=f'additional-{index}'))
    document = scenes.scene_document('ios', fonts, scenes.content_pages(1000))
    smoke = scenes.smoke_document(document)
    assert len(smoke['pages']) == (len(fonts)+11)//12
    assert {r['font_id'] for p in smoke['pages'] for r in p['regions']} == {f['id'] for f in fonts}
    assert all(len(p['regions']) == 12 for p in smoke['pages'])


def test_second_thousand_pages_preserve_english_numeric_without_text_reuse():
    old = scenes.content_pages(1000)
    forbidden = {''.join(r['text'].split()) for p in old for r in p['regions']}
    new = scenes.content_pages(1000, 2026091291, 'mixed', 'ios-hant-v1', forbidden, extended_english=True)
    current = {''.join(r['text'].split()) for p in new for r in p['regions']}
    assert not current & forbidden
    assert len(current) == sum(len(p['regions']) for p in new)
    for page in new:
        assert Counter(r['text_kind'] for r in page['regions'])['english'] == 2
        assert Counter(r['text_kind'] for r in page['regions'])['numeric'] == 2
        for row in page['regions']:
            if row['text_kind']=='numeric':
                assert row['text'].isdigit() and row['text'].isascii()


def test_prior_text_exclusion_uses_nfkc_casefold_and_whitespace():
    old = scenes.content_pages(100)
    forbidden = [r['text'].upper().replace(' ', '\t') for p in old for r in p['regions']]
    current = scenes.content_pages(100, namespace='different', forbidden_texts=forbidden)
    assert not ({scenes.normalized_text(r['text']) for p in current for r in p['regions']}
                & {scenes.normalized_text(text) for text in forbidden})
    assert scenes.normalized_text(' ＡＢＣ\t１２３ ') == scenes.normalized_text('abc123')
