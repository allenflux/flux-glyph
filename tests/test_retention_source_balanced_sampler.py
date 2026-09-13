"""Check actual emitted rows, provenance and persistent source balance."""
from collections import Counter
from copy import deepcopy

import pytest

from training.retention_source_balanced_sampler import (
    SourceBalancedSampler, SAMPLING, expected_unknown_source_counts,
)
from training.retention_supplement_sampler import NEW_SOURCES, SUPPLEMENT_MARKER, VIEW_WEIGHTS
from test_retention_supplement_sampler import supplemental_rows
from test_unified_retention import FAMILIES, sampler_data


def source_balanced_data():
    rows, pools = sampler_data()
    # The original fixture's unknown rows are last, so known anchor indices stay fixed.
    rows = [row for row in rows if row['family'] != '__unknown__']
    for index in range(9):
        source = f'Old unknown {index}'
        for domain in ('ios', 'android'):
            for face in (0, 1):
                for view in VIEW_WEIGHTS:
                    rows.append({'family': '__unknown__', 'target': 24, 'split': 'train',
                        'domain': domain, 'view': view, 'source_font_family': source,
                        'font_face': f'{source}-{face}', 'font_file_sha256': str(face) * 64,
                        'native_font_verified': True, 'tile_start': len(rows), 'tile_count': 1,
                        'source_id': str(len(rows)), 'region_id': 'native'})
    return rows, pools


def test_emitted_rows_keep_96_slots_and_exact_continuous_11_source_balance():
    rows, pools = source_balanced_data()
    extra = supplemental_rows()
    before, extra_before = deepcopy(rows), deepcopy(extra)
    sampler = SourceBalancedSampler(rows, FAMILIES, 2026091410, pools, extra)
    source_counts, families, views = Counter(), Counter(), Counter()
    calls, focused = [], []
    base_draw, focus_draw = sampler.base._row, sampler._focus_row

    def base(group):
        row = base_draw(group)
        calls.append((group, row))
        return row

    def focus(family, pool):
        row = focus_draw(family, pool)
        focused.append((pool, row))
        return row

    sampler.base._row, sampler._focus_row = base, focus
    for step in range(1, 100):
        prior_calls, prior_focus = len(calls), len(focused)
        batch = sampler.batch()
        unknown = [r for r in batch if r['family'] == '__unknown__']
        new_count = sum(r.get(SUPPLEMENT_MARKER, False) for r in unknown)
        assert len(batch) == 96 and len(unknown) == 16 and 2 <= new_count <= 4
        assert len(calls) - prior_calls == 48 + 16 - new_count
        assert len(focused) - prior_focus == 32
        assert all(r['split'] == 'train' and r['native_font_verified'] for r in batch)
        assert all(bool(r.get(SUPPLEMENT_MARKER)) == (r['source_font_family'] in NEW_SOURCES) for r in batch)
        source_counts.update(r['source_font_family'] for r in unknown)
        families.update(r['family'] for r in batch)
        views.update(r['view'] for r in batch if r.get(SUPPLEMENT_MARKER))
        assert source_counts == expected_unknown_source_counts(sampler.unknown_sources, step)
        assert max(source_counts.values()) - min(source_counts.values()) <= 1
        if step % 11 == 0:
            assert set(source_counts.values()) == {16 * step // 11}
        current = Counter(r['family'] for r in batch)
        assert current['PingFang'] == 19 and current['SF Pro'] == current['Helvetica'] == 7
        assert all(current[f] == (3 if f in FAMILIES[:8] else 2)
                   for f in FAMILIES[:-1] if f not in ('PingFang', 'SF Pro', 'Helvetica'))

    report = sampler.report()
    assert report['source_balanced'] and SAMPLING['source_balanced']
    assert SAMPLING['unknown'] == 16 and SAMPLING['base_unknown'] is None
    assert SAMPLING['supplement_unknown'] is None
    assert report['unknown_source_order'] == sampler.unknown_source_order == sorted(source_counts)
    assert report['unknown_rows'] == 1584
    assert report['unknown_source_rows'] == dict.fromkeys(source_counts, 144)
    assert report['base']['known_rows'] == 4752
    assert report['base']['unknown_rows'] == report['slots']['base_unknown'] == 1296
    assert report['slots']['supplement_unknown'] == 288
    assert report['supplement_source_rows'] == dict.fromkeys(NEW_SOURCES, 144)
    assert report['supplement_view_rows'] == dict(views)
    assert report['family_rows'] == dict(families)
    assert sum(report['slots'].values()) == sum(families.values()) == 9504
    assert sum(r['rows'] for r in report['source_rows']) == 9504
    assert report['proposal_rows_discarded'] == 0
    assert report['test_read'] is False and report['development_holdout_read'] is False
    assert rows == before and extra == extra_before
    anchor = set(pools[2]['indices'])
    assert all(r['domain'] == 'ios' and r['view'] == 'native' for _, r in focused)
    assert all(r['tile_start'] in anchor for p, r in focused if p == 'ios_native_anchor_original8')
    pf_faces = Counter(r['font_face'] for p, r in focused if p == 'ios_native_pingfang')
    assert set(pf_faces.values()) == {792}
    # Original unknown source/domain/face cycling remains balanced without discarded draws.
    old_faces = Counter((r['source_font_family'], r['domain'], r['font_face'])
                        for group, r in calls if group[0] == 'unknown')
    assert len(old_faces) == 36 and set(old_faces.values()) == {36}


def test_sampling_reproducible_and_new_sources_keep_weighted_views():
    rows, pools = source_balanced_data()
    a = SourceBalancedSampler(rows, FAMILIES, 71, pools, supplemental_rows())
    b = SourceBalancedSampler(rows, FAMILIES, 71, pools, supplemental_rows())
    for _ in range(11):
        assert a.batch() == b.batch()
    assert a.report() == b.report()
    detached = a.unknown_source_order
    detached.clear()
    assert len(a.unknown_source_order) == 11
    for _ in range(539):
        a.batch()
    report = a.report()
    assert report['supplement_source_rows'] == dict.fromkeys(NEW_SOURCES, 800)
    assert sum(report['supplement_view_rows'].values()) == 1600
    for view, expected in VIEW_WEIGHTS.items():
        assert abs(report['supplement_view_rows'][view] / 1600 - expected) < .045


@pytest.mark.parametrize('fault', ['old_cal', 'new_holdout', 'unverified', 'old_marker',
    'new_marker', 'missing_view', 'eight_sources', 'ten_sources', 'source_overlap'])
def test_invalid_sources_and_nontrain_rows_are_rejected(fault):
    rows, pools = source_balanced_data()
    extra = supplemental_rows()
    if fault == 'old_cal':
        rows[-1]['split'] = 'calibration'
    elif fault == 'new_holdout':
        extra[0]['split'] = 'development_holdout'
    elif fault == 'unverified':
        extra[0]['native_font_verified'] = False
    elif fault == 'old_marker':
        rows[0][SUPPLEMENT_MARKER] = True
    elif fault == 'new_marker':
        extra[0][SUPPLEMENT_MARKER] = True
    elif fault == 'missing_view':
        extra.pop()
    elif fault == 'eight_sources':
        rows = [r for r in rows if r['source_font_family'] != 'Old unknown 8']
    elif fault == 'ten_sources':
        rows.append({**rows[-1], 'source_font_family': 'Old unknown 9'})
    else:
        rows[-1]['source_font_family'] = NEW_SOURCES[0]
    with pytest.raises(ValueError):
        SourceBalancedSampler(rows, FAMILIES, 71, pools, extra)


@pytest.mark.parametrize('steps', [-1, True, 1.5, '1', None])
def test_expected_counts_require_integer_nonnegative_steps(steps):
    order = sorted([f'Old unknown {i}' for i in range(9)] + list(NEW_SOURCES))
    with pytest.raises(ValueError):
        expected_unknown_source_counts(order, steps)


def test_expected_counts_validate_order_and_long_run_arithmetic():
    order = sorted([f'Old unknown {i}' for i in range(9)] + list(NEW_SOURCES))
    assert expected_unknown_source_counts(order, 0) == dict.fromkeys(order, 0)
    assert expected_unknown_source_counts(order, 3000) == {
        source: 4364 if i < 7 else 4363 for i, source in enumerate(order)}
    for invalid in (order[::-1], order[:-1], [*order[:-1], order[0]],
                    sorted([*order[:-1], 'Unapproved new source'])):
        with pytest.raises(ValueError):
            expected_unknown_source_counts(invalid, 1)
