from collections import Counter
import copy
import json

import pytest

from ios_short_training import IOSShortSampler, IOS_FAMILIES, VIEWS, load_prepared


def pools():
    result = {}
    for family in IOS_FAMILIES:
        for length in (1, 2, 3, 4):
            result[family, length] = {}
            for face, count in (('face_a', 1), ('face_b', 7)):
                result[family, length][face] = {}
                for n in range(count):
                    identity = (family, f'{length}-{face}-{n}')
                    result[family, length][face][identity] = {
                        view: {'family': family, 'glyph_count': length, 'font_face': face,
                               'source_id': identity[0], 'region_id': identity[1], 'view': view}
                        for view in VIEWS}
    return {'pools': result}


def test_face_round_robin_is_independent_of_pool_multiplicity():
    data = pools()
    original = copy.deepcopy(data)
    sampler = IOSShortSampler(data, 42)
    counters = Counter()
    for _ in range(40):
        rows = sampler.batch()
        assert len(rows) == 24
        assert Counter(r['family'] for r in rows) == dict.fromkeys(IOS_FAMILIES, 8)
        assert Counter(r['glyph_count'] for r in rows) == dict.fromkeys((1, 2, 3, 4), 6)
        counters.update((r['family'], r['glyph_count'], r['font_face']) for r in rows)
    assert set(counters.values()) == {40}
    assert data == original
    report = sampler.report()
    assert report['rows'] == 960 and sum(report['by_identity'].values()) == 960
    assert sum(report['by_face'].values()) == sum(report['by_view'].values()) == 960


def test_identity_sampling_is_random_reproducible_and_rows_are_copies():
    data = pools()
    one, two = IOSShortSampler(data, 1), IOSShortSampler(data, 1)
    rows1, rows2 = [], []
    for _ in range(80):
        rows1.extend(one.batch()); rows2.extend(two.batch())
    assert rows1 == rows2 and one.report() == two.report()
    samples = {r['region_id'] for r in rows1 if r['family'] == 'PingFang' and r['glyph_count'] == 1 and r['font_face'] == 'face_b'}
    assert len(samples) == 7
    rows1[0]['family'] = 'wrong'
    assert all(r['family'] != 'wrong' for faces in data['pools'].values() for ids in faces.values() for views in ids.values() for r in views.values())


def test_nonfrozen_capture_rejected_before_arrays_are_opened(tmp_path):
    (tmp_path/'MANIFEST.json').write_text(json.dumps({'schema': 'flux-glyph-ios-native-short-train-v1'}))
    with pytest.raises(ValueError, match='frozen capture'):
        load_prepared(tmp_path)
