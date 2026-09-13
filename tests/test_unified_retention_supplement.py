from collections import Counter
import numpy as np
import pytest

from training.train_unified_retention_supplement import feature_batch, SupplementCounts
from training.retention_supplement_sampler import SupplementSampler, SUPPLEMENT_MARKER, NEW_SOURCES
from test_unified_retention import sampler_data, FAMILIES
from test_retention_supplement_sampler import supplemental_rows


def test_coincident_tile_indices_select_their_own_cache_without_blending():
    old = {'features': np.full((2, 128), 3, np.float32), 'base_logits': np.full((2, 25), 7, np.float32)}
    new = {'features': np.full((2, 128), -5, np.float32), 'base_logits': np.full((2, 25), -11, np.float32)}
    rows = [{'tile_start': 0, 'tile_count': 2}, {'tile_start': 0, 'tile_count': 2, SUPPLEMENT_MARKER: True}]
    features, logits = feature_batch(rows, old, new, np.random.default_rng(4))
    assert features.dtype == logits.dtype == np.float32
    np.testing.assert_array_equal(features, np.array([[3]*128, [-5]*128], np.float32))
    np.testing.assert_array_equal(logits, np.array([[7]*25, [-11]*25], np.float32))


def test_tile_bounds_apply_to_selected_cache():
    old = {'features': np.zeros((5, 128)), 'base_logits': np.zeros((5, 25))}
    new = {'features': np.zeros((1, 128)), 'base_logits': np.zeros((1, 25))}
    with pytest.raises(ValueError):
        feature_batch([{'tile_start': 2, 'tile_count': 1, SUPPLEMENT_MARKER: True}], old, new, np.random.default_rng(4))


def test_training_counts_measure_actual_old_and_new_updates():
    rows, pools = sampler_data()
    sampler = SupplementSampler(rows, FAMILIES, 2026091407, pools, supplemental_rows())
    counts = SupplementCounts()
    for _ in range(5):
        batch = sampler.batch()
        weights = [2. if row['family'] == '__unknown__' else 1. for row in batch]
        counts.update(batch, [False]*96, weights)
    report = counts.report()
    assert report['rows'] == 480
    assert report['original_rows'] == 440
    assert report['supplement_rows'] == 40
    assert report['supplement_source_rows'] == dict.fromkeys(NEW_SOURCES, 20)
    assert Counter({r['source_font_family']: r['rows'] for r in report['source_target_rows']
        if r['source_font_family'] in NEW_SOURCES}) == dict.fromkeys(NEW_SOURCES, 20)


def test_missing_new_samples_cannot_be_reported_as_supplement_training():
    rows, pools = sampler_data()
    sampler = SupplementSampler(rows, FAMILIES, 1, pools, supplemental_rows())
    batch = sampler.batch()
    next(row for row in batch if row.get(SUPPLEMENT_MARKER)).pop(SUPPLEMENT_MARKER)
    with pytest.raises(ValueError):
        SupplementCounts().update(batch, [False]*96, [1.]*96)
