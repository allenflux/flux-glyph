"""Joint data must retain real family truth and source-connected partitions."""
from collections import Counter
import math

import pytest

from training.prepare_unified_regions import (
    FAMILIES, UNKNOWN, connected_partitions, family_label, normalize_row,
)


def test_promoted_real_fonts_are_named_and_other_families_stay_unknown():
    assert len(FAMILIES) == 25 and len(set(FAMILIES)) == 25
    for font in ('FandolKai', 'Long Cang', 'Zhi Mang Xing', 'Kaiti SC', 'Songti SC', 'HanziPen SC'):
        assert family_label(font) == font
    assert family_label('PingFang TC') == family_label('PingFang HK') == 'PingFang'
    assert family_label('FandolKai') != family_label('Kaiti SC')
    assert family_label('Unknown theme font') == UNKNOWN


def test_connected_pages_never_cross_partitions_and_giant_group_stays_training():
    pages = {f'page-{i:03d}': {'old_split': 'train', 'dataset': 'android'} for i in range(100)}
    identities = {('text', 'shared'): [f'page-{i:03d}' for i in range(35)]}
    assigned, excluded, audit = connected_partitions(pages, identities, {'android'})
    assert not excluded
    assert {assigned[p] for p in identities[('text', 'shared')]} == {'train'}
    assert Counter(assigned.values()) == {'train': 80, 'calibration': 10, 'development_holdout': 10}
    assert audit[0]['largest_component'] == 35
    assert (assigned, excluded, audit) == connected_partitions(dict(reversed(list(pages.items()))), identities, {'android'})


def test_cross_original_split_content_excludes_entire_connected_component():
    pages = {p: {'old_split': s, 'dataset': 'android'} for p,s in [('a','train'),('b','train'),('c','calibration'),('d','calibration')]}
    assigned, excluded, _ = connected_partitions(pages, {('text','x'): ['a','b'], ('crop','y'): ['b','c']}, {'android'})
    assert excluded == ['a','b','c']
    assert assigned == {'d':'calibration'}


def test_untouched_training_source_is_not_repartitioned():
    pages = {str(i): {'old_split':'train', 'dataset':'ios_supplement'} for i in range(20)}
    assigned, excluded, audit = connected_partitions(pages, {}, {'android','ios_unknown'})
    assert set(assigned.values()) == {'train'} and not excluded and not audit


def native_fixture():
    row = {'source_id':'android:p','page_id':'p','region_id':'r','family':UNKNOWN,
           'source_font_family':'FandolKai','font_face':'FandolKai-Regular',
           'font_size_px':40.,'ink_height_px':32,'log_em_ratio':math.log(1.25),
           'native_crop_sha256':'a'*64,'tiles_sha256':'b'*64,'tile_start':0,'tile_count':1,'view':'native'}
    truth = {('android:p','r'): {'text':'训练文字','actual_family':'FandolKai',
             'font_file_sha256':'c'*64,'font_face':'FandolKai-Regular','font_source_kind':'asset',
             'source_sha256':'d'*64,'native_split':'train','text_color_hex':'#333333'}}
    return row, truth


def test_promote_uses_native_family_even_when_old_target_was_unknown():
    row, truth = native_fixture()
    out = normalize_row(row, 'android', 'train', truth)
    assert out['family'] == 'FandolKai' and out['target'] == FAMILIES.index('FandolKai')
    assert out['font_file_sha256'] == 'c'*64 and out['font_size_px'] == 40
    assert 'text' not in out and out['normalized_text_sha256']
    assert out['domain'] == 'android'


@pytest.mark.parametrize('change', [
    {'source_font_family':'Kaiti SC'}, {'font_face':'FandolKai-Bold'},
    {'font_size_px':float('nan')}, {'log_em_ratio':0.0}, {'native_crop_sha256':None},
])
def test_mislabeled_or_invalid_native_targets_fail(change):
    row, truth = native_fixture()
    with pytest.raises(ValueError):
        normalize_row({**row, **change}, 'android', 'train', truth)


def test_holdout_is_not_opened_without_post_selection_access(tmp_path):
    from training.prepare_unified_regions import load_split
    with pytest.raises(ValueError, match='holdout'):
        load_split(tmp_path, 'development_holdout')
