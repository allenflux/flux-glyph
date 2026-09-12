import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from train_rejection import aggregate, calibrate
from train_regions import sha, verify_preprocessing_snapshot


def test_rejection_calibration_preserves_known_domains_and_family():
    rows = [{'label': 1, 'domain': 'r17_native', 'family': 'PingFang'} for _ in range(100)]
    rows += [{'label': 1, 'domain': 'new_native', 'family': 'Helvetica'} for _ in range(20)]
    rows += [{'label': 0, 'domain': 'new_native', 'family': 'Heldout Handwriting'} for _ in range(20)]
    scores = np.array([.1] + [.9] * 99 + [.7] * 20 + [.01] * 19 + [.8])
    threshold, result = calibrate(scores, rows)
    assert .69 < threshold < .7
    assert result['unknown_recall'] == .95
    assert result['by_domain']['r17_native']['known_rejected'] == 1
    assert result['by_domain']['new_native']['known_rejected'] == 0
    assert result['system']['known_false_rejection_rate'] <= .01


def test_rejection_is_mean_tile_probability_not_font_score_or_mean_logits():
    rows = [{'tile_start': 0, 'tile_count': 2}, {'tile_start': 2, 'tile_count': 1}]
    logits = np.array([[0., 100.], [0., -1.], [0., 0.]])
    result = aggregate(logits, rows)
    assert .63 < result[0] < .64
    assert result[1] == .5


def test_immutable_preprocessing_snapshot_checks_contents(tmp_path):
    source = ROOT / 'src/flux_glyph/region_font.py'
    snapshot = tmp_path / 'snapshot.py'
    snapshot.write_bytes(source.read_bytes())
    verify_preprocessing_snapshot(sha(source), snapshot)
    snapshot.write_text(snapshot.read_text().replace('contrast < 24', 'contrast < 30'))
    with pytest.raises(ValueError, match='snapshot differs'):
        verify_preprocessing_snapshot(sha(source), snapshot)
    with pytest.raises(ValueError, match='current region preprocessing differs'):
        verify_preprocessing_snapshot(sha(snapshot), snapshot)
