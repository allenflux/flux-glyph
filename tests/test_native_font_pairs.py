"""Check matched-font training truth, reproducibility and the bias-free objective."""
import copy
import hashlib
from collections import Counter

import numpy as np
import pytest

from training import native_font_pairs as pairs


def mobile_fixture():
    pools = {}
    for index, source in enumerate(pairs.SOURCES):
        family = source if source in pairs.KNOWN else '__unknown__'
        target = pairs.FAMILIES.index(family)
        platform = 'ios' if source in ('PingFang', 'SF Pro', 'Helvetica') else 'android'
        groups = {}
        for number, text in enumerate(('字体', '文字')):
            identity = (platform, f'page-{index}', f'region-{number}')
            row = {'text': text, 'normalized_text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'script': 'han', 'glyph_count': 2, 'native_font_size_px': 54.,
                'text_color_hex': '#333333FF', 'split': 'train', 'native_font_verified': True,
                'family': family, 'target': target, 'source_font_family': source,
                'short_dataset': platform, 'font_face': source + '-Regular', 'tile_count': 1,
                'source_id': identity[1], 'region_id': identity[2]}
            groups[identity] = {view: {**row, 'view': view, 'tiles_sha256': f'{index}-{number}-{view}'}
                                for view in pairs.VIEWS}
        pools[source, 'han', 2] = {source + '-Regular': groups}
    return {'pools': pools}


def test_sampler_matches_actual_text_style_and_view_with_balanced_independent_sources():
    mobile = mobile_fixture(); before = copy.deepcopy(mobile)
    pool, proof = pairs.build_pairs(mobile)
    assert proof['native_identities'] == proof['pair_eligible_native_identities'] == 42
    assert proof['cohorts_with_different_targets'] == 2
    first, second = pairs.NativeFontPairSampler(pool, 42), pairs.NativeFontPairSampler(pool, 42)
    external_rng = np.random.default_rng(7); state = copy.deepcopy(external_rng.bit_generator.state)
    for _ in range(21):
        batch = first.batch()
        assert batch == second.batch()
        for a, b in batch:
            assert pairs.cohort_key(a) == pairs.cohort_key(b)
            assert a['target'] != b['target'] and a['view'] == b['view']
            a['text'] = 'mutated copy'
    report = first.report()
    assert report == second.report()
    assert report['anchor_by_source'] == dict.fromkeys(pairs.SOURCES, 16)
    assert sum(report['rows_by_source'].values()) == 672
    assert sum(report['pairs_by_view'].values()) == 336
    assert external_rng.bit_generator.state == state and mobile == before


def test_casefold_equal_hash_does_not_pair_different_native_text():
    a = next(iter(next(iter(mobile_fixture()['pools'].values())).values()))
    row = copy.deepcopy(next(iter(a.values()))['native'])
    row.update(text='A', normalized_text_sha256='same', script='latin', glyph_count=1)
    other = {**row, 'text': 'a'}
    assert pairs.cohort_key(row) != pairs.cohort_key(other)


@pytest.mark.parametrize('fault', ['missing_view', 'non_train', 'false_truth', 'different_style'])
def test_pool_rejects_unverified_or_inconsistent_native_truth(fault):
    data = mobile_fixture()
    groups = next(iter(next(iter(data['pools'].values())).values()))
    views = next(iter(groups.values()))
    if fault == 'missing_view': del views['half']
    elif fault == 'non_train': views['native']['split'] = 'calibration'
    elif fault == 'false_truth': views['native']['native_font_verified'] = False
    else: views['half']['native_font_size_px'] = 99.
    with pytest.raises(ValueError): pairs.build_pairs(data)


def test_pair_margin_cancels_class_bias_and_is_symmetric_with_correct_gradients():
    torch = pytest.importorskip('torch')
    torch.manual_seed(73)
    logits = torch.randn(32, 25, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([4, 2] * 15 + [24, 5])
    bias = torch.randn(25, dtype=torch.float64, requires_grad=True)
    base = pairs.pair_margin_loss(logits, targets)
    shifted = pairs.pair_margin_loss(logits + bias, targets)
    torch.testing.assert_close(base, shifted)
    shifted.backward()
    torch.testing.assert_close(bias.grad, torch.zeros_like(bias), atol=1e-15, rtol=0)
    assert torch.all(logits.grad[::2, 4][:15] < 0)
    assert torch.all(logits.grad[1::2, 2][:15] < 0)
    order = torch.arange(32).reshape(16, 2).flip(1).flatten()
    torch.testing.assert_close(base, pairs.pair_margin_loss(logits[order], targets[order]))
    correct = torch.zeros(32, 25, dtype=torch.float64)
    correct[torch.arange(32), targets] = 2.
    assert pairs.pair_margin_loss(correct, targets) < pairs.pair_margin_loss(-correct, targets)


def test_pair_margin_rejects_unknown_unknown_and_nonfinite_logits():
    torch = pytest.importorskip('torch')
    values = torch.zeros(32, 25); targets = torch.tensor([4, 24] * 16)
    assert torch.isfinite(pairs.pair_margin_loss(values, targets))
    targets[0] = 24
    with pytest.raises(ValueError): pairs.pair_margin_loss(values, targets)
    targets[0] = 4; values[0, 0] = float('nan')
    with pytest.raises(ValueError): pairs.pair_margin_loss(values, targets)
