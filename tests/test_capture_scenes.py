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
