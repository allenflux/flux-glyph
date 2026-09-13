from collections import Counter
from copy import deepcopy
import pytest

from training.retention_supplement_sampler import SupplementSampler, NEW_SOURCES, VIEW_WEIGHTS, SUPPLEMENT_MARKER
from test_unified_retention import sampler_data, FAMILIES


def supplemental_rows():
    return [{'family': '__unknown__', 'target': 24, 'split': 'train', 'domain': 'android',
        'native_font_verified': True, 'source_font_family': source, 'font_face': source + '-Regular',
        'font_file_sha256': str(index) * 64, 'view': view, 'tile_start': index * 4 + offset,
        'tile_count': 1, 'source_id': source + str(offset), 'region_id': str(offset)}
        for index, source in enumerate(NEW_SOURCES) for offset, view in enumerate(VIEW_WEIGHTS)]


def test_original_known_replay_and_actual_negative_populations_are_preserved():
    rows, pools = sampler_data()
    extra = supplemental_rows()
    original, new = deepcopy(rows), deepcopy(extra)
    sampler = SupplementSampler(rows, FAMILIES, 2026091407, pools, extra)
    for _ in range(200):
        batch = sampler.batch()
        assert len(batch) == 96
        assert Counter(r['family'] for r in batch)['__unknown__'] == 16
        assert Counter(r['source_font_family'] for r in batch if r.get(SUPPLEMENT_MARKER)) == dict.fromkeys(NEW_SOURCES, 4)
        assert sum(not r.get(SUPPLEMENT_MARKER) and r['family'] == '__unknown__' for r in batch) == 8
    report = sampler.report()
    assert report['base']['known_rows'] == 9600
    assert report['base']['unknown_rows'] == 1600
    assert report['supplement_source_rows'] == dict.fromkeys(NEW_SOURCES, 800)
    assert report['family_rows']['PingFang'] == 19 * 200
    assert report['family_rows']['SF Pro'] == report['family_rows']['Helvetica'] == 7 * 200
    assert report['family_rows']['__unknown__'] == 16 * 200
    assert sum(report['slots'].values()) == sum(report['family_rows'].values()) == 19200
    assert sum(r['rows'] for r in report['source_rows']) == 19200
    # Statistical view check uses 1600 fixed-seed draws, not a mirrored implementation.
    assert .34 < report['supplement_view_rows']['native'] / 1600 < .46
    assert rows == original and extra == new
    assert report['proposal_rows_discarded'] == 0


def test_supplement_replay_is_reproducible():
    rows, pools = sampler_data()
    a = SupplementSampler(rows, FAMILIES, 17, pools, supplemental_rows())
    b = SupplementSampler(rows, FAMILIES, 17, pools, supplemental_rows())
    for _ in range(5):
        assert a.batch() == b.batch()
    assert a.report() == b.report()


@pytest.mark.parametrize('key,value', [('split', 'calibration'), ('split', 'test'),
    ('split', 'development_holdout'), ('family', 'PingFang'), ('target', 4),
    ('native_font_verified', False), ('domain', 'ios'), ('source_font_family', 'Yusei Magic'),
    ('tile_count', 0), ('font_file_sha256', ''), (SUPPLEMENT_MARKER, True)])
def test_invalid_or_heldout_supplement_is_rejected(key, value):
    rows, pools = sampler_data()
    extra = supplemental_rows()
    extra[0][key] = value
    with pytest.raises(ValueError):
        SupplementSampler(rows, FAMILIES, 17, pools, extra)


def test_missing_source_view_is_rejected():
    rows, pools = sampler_data()
    with pytest.raises(ValueError):
        SupplementSampler(rows, FAMILIES, 17, pools, supplemental_rows()[1:])
