"""Verify the native-pairs trainer keeps old losses exact and isolates pair rows."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from training import train_unified_native_pairs as trainer
from training import train_unified_retention_micro_recovery as r22
from training.prepare_unified_regions import FAMILIES


def replay_rows():
    rows = []
    for index in range(96):
        target = index % 24 if index < 80 else 24
        family = FAMILIES[target]
        rows.append({'split': 'train', 'native_font_verified': True, 'target': target,
            'family': family, 'source_font_family': family if target < 24 else 'Smiley Sans',
            'domain': 'android', 'view': 'native', 'tile_start': index, 'tile_count': 1,
            'log_em_ratio': index / 1000, 'source_id': f's{index}', 'region_id': f'r{index}'})
    return rows


def mobile_rows():
    rows = []; indices = {'ios': 0, 'android': 0}
    for family in trainer.MOBILE_KNOWN:
        platform = 'ios' if family in ('PingFang', 'SF Pro', 'Helvetica') else 'android'
        index = indices[platform]; indices[platform] += 1
        rows.append({'target': FAMILIES.index(family), 'family': family,
            'source_font_family': family, 'short_dataset': platform, 'tile_start': index,
            'tile_count': 1, 'log_em_ratio': (index - 8) / 100})
    for offset in range(3):
        index = indices['android']; indices['android'] += 1
        rows.append({'target': 24, 'family': '__unknown__',
            'source_font_family': trainer.MOBILE_UNKNOWN_SOURCES[offset],
            'short_dataset': 'android', 'tile_start': index, 'tile_count': 1,
            'log_em_ratio': (index - 8) / 100})
    return rows


def pair_rows():
    pairs = []
    for index in range(16):
        shared = {'text': f'pair-{index}', 'normalized_text_sha256': f'{index:064x}',
            'script': 'han' if index % 2 else 'latin', 'glyph_count': index % 4 + 1,
            'native_font_size_px': float(30 + index), 'text_color_hex': '#334455FF',
            'view': ('native', 'half', 'three_quarters_jpeg75', 'jpeg75')[index % 4],
            'tile_count': 1, 'split': 'train', 'native_font_verified': True}
        a = {**shared, 'target': 4, 'family': 'PingFang', 'source_font_family': 'PingFang SC',
             'short_dataset': 'ios', 'source_id': f'ia{index}', 'region_id': f'ira{index}',
             'tile_start': index % 3}
        b = {**shared, 'target': 2, 'family': 'Noto Sans CJK SC',
             'source_font_family': 'Noto Sans CJK SC', 'short_dataset': 'android',
             'source_id': f'ab{index}', 'region_id': f'arb{index}', 'tile_start': index % 18}
        pairs.append((a, b))
    return pairs


def batch():
    rows = replay_rows(); focus = deepcopy(rows[:32]); mobile = mobile_rows()
    targets = torch.tensor([row['target'] for row in rows])
    teacher = torch.full((96, 25), -2.)
    teacher[torch.arange(96), targets] = 2.
    pairs = pair_rows(); flat = [row for pair in pairs for row in pair]
    return {'images': torch.rand(180, 1, 64, 256, requires_grad=True),
        'targets': targets, 'sizes': torch.linspace(-.3, .3, 96),
        'teacher': teacher, 'rows': rows, 'focus_rows': focus,
        'focus_targets': targets[:32], 'focus_sizes': torch.linspace(-.2, .2, 32),
        'focus_teacher': teacher[:32], 'mobile_rows': mobile,
        'mobile_targets': torch.tensor([row['target'] for row in mobile]),
        'mobile_sizes': torch.tensor([row['log_em_ratio'] for row in mobile]),
        'pair_rows': flat, 'pair_targets': torch.tensor([row['target'] for row in flat])}


def test_objective_preserves_native_mobile_terms_and_adds_only_weighted_pair_margin():
    from wide_region_network import WideRegionFontClassifier
    torch.set_num_threads(2); torch.manual_seed(81)
    model = WideRegionFontClassifier(25).eval(); values = batch()
    total, parts = trainer.objective(model, values)
    font, size, features = r22.forward_with_features(model, values['images'])
    replay = r22.full_losses(font[:96], size[:96], features[:96], model.family_head.weight,
        values['targets'], values['sizes'], values['teacher'], values['rows'], FAMILIES)
    focus = trainer.focus_full_loss(font[96:128], size[96:128], features[96:128],
        model.family_head.weight, values['focus_targets'], values['focus_sizes'],
        values['focus_teacher'], values['focus_rows'])
    mobile_ce = torch.nn.functional.cross_entropy(font[128:148], values['mobile_targets'])
    mobile_size = torch.nn.functional.smooth_l1_loss(size[128:148], values['mobile_sizes'], beta=.05)
    mobile_loss = .5 * (mobile_ce + .2 * mobile_size)
    pair_loss = trainer.pair_margin_loss(font[148:180], values['pair_targets'])
    expected = replay[0] + .5 * focus[0] + .25 * parts['unknown_oe'] + mobile_loss + .1 * pair_loss
    torch.testing.assert_close(total, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(parts['replay'], replay[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(parts['mobile_loss'], mobile_loss)
    torch.testing.assert_close(parts['pair_loss'], pair_loss)
    torch.testing.assert_close(parts['pair_weighted'], .1 * pair_loss)
    assert parts['pair_count'].item() == 16 and parts['pair_rows'].item() == 32
    assert parts['unknown_oe_rows'].item() == 16


def test_pair_term_uses_only_last_32_images_and_never_size_head():
    from wide_region_network import WideRegionFontClassifier
    torch.set_num_threads(2); torch.manual_seed(82)
    model = WideRegionFontClassifier(25).train(); values = batch()
    _, parts = trainer.objective(model, values)
    parts['pair_weighted'].backward()
    affected = values['images'].grad.flatten(1).abs().sum(1)
    assert torch.count_nonzero(affected[:148]) == 0
    assert torch.all(affected[148:] > 0)
    for group in ('trunk', 'style', 'family_head'):
        gradients = [p.grad for name, p in model.named_parameters() if name.startswith(group + '.')]
        assert gradients and any(g is not None and g.abs().sum() > 0 for g in gradients)
    size_gradients = [p.grad for name, p in model.named_parameters() if name.startswith('size_head.')]
    assert size_gradients and all(g is None or torch.count_nonzero(g) == 0 for g in size_gradients)


@pytest.mark.parametrize('fault', ['same_target', 'different_text', 'different_size', 'different_color',
                                   'different_view', 'same_identity', 'not_tuple'])
def test_pair_rows_fail_closed_on_alignment_or_target_changes(fault):
    pairs = pair_rows()
    if fault == 'not_tuple': pairs[0] = list(pairs[0])
    else:
        a, b = pairs[0]
        if fault == 'same_target': b['target'] = a['target']
        elif fault == 'different_text': b['text'] += 'x'
        elif fault == 'different_size': b['native_font_size_px'] += 1
        elif fault == 'different_color': b['text_color_hex'] = '#000000'
        elif fault == 'different_view': b['view'] = 'native' if a['view'] != 'native' else 'half'
        else: b['short_dataset'], b['source_id'], b['region_id'] = trainer._pair_identity(a)
    with pytest.raises(ValueError):
        trainer.validate_pair_rows(pairs)


class RowsSampler:
    def __init__(self, rows, seed=0):
        self.rows = rows; self.rng = np.random.default_rng(seed)
    def batch(self): return deepcopy(self.rows)


class PairSampler:
    def __init__(self, pairs): self.pairs = pairs
    def batch(self): return deepcopy(self.pairs)


def sources():
    rows = replay_rows()
    images = np.stack([np.full((1, 64, 256), index / 100, np.float32) for index in range(96)])
    teacher = np.stack([np.full(25, index / 10, np.float32) for index in range(96)])
    source = r22.batch_source({'rows': rows, 'tiles': images, 'partition': {'split': 'train'}},
                              {'base_logits': teacher})
    return rows, {'original': source, 'supplement': source, 'known': source}


def mobile_data():
    return {'datasets': {
        'ios': {'tiles': np.stack([np.full((1, 64, 256), 5 + i / 100, np.float32)
                                   for i in range(3)])},
        'android': {'tiles': np.stack([np.full((1, 64, 256), 7 + i / 100, np.float32)
                                       for i in range(18)])}}}


def test_inputs_append_interleaved_pair_tiles_without_changing_old_rng_streams():
    rows, replay = sources(); focus = list(reversed(rows[:32])); data = mobile_data()
    original = RowsSampler(rows, 401); expected = RowsSampler(rows, 401)
    mobile_sampler = RowsSampler(mobile_rows(), 403)
    pair_sampler = PairSampler(pair_rows())
    expected_images, expected_teacher, expected_size = r22.pixel_batch(
        expected.batch(), replay['original'], replay['supplement'], replay['known'], expected.rng)
    value = trainer.inputs(replay, original, RowsSampler(focus, 402), data,
                           mobile_sampler, pair_sampler, 'cpu')
    assert value['images'].shape == (180, 1, 64, 256)
    np.testing.assert_array_equal(value['images'][:96].numpy(), expected_images)
    np.testing.assert_array_equal(value['teacher'].numpy(), expected_teacher)
    np.testing.assert_array_equal(value['sizes'].numpy(), expected_size)
    assert original.rng.bit_generator.state == expected.rng.bit_generator.state
    expected_pair = np.stack([data['datasets'][row['short_dataset']]['tiles'][row['tile_start']]
                              for pair in pair_rows() for row in pair])
    np.testing.assert_array_equal(value['images'][148:].numpy(), expected_pair)
    assert value['pair_targets'].tolist() == [row['target'] for pair in pair_rows() for row in pair]


def pair_report(steps):
    return {'schema': trainer.PAIR_SAMPLING['schema'], 'steps': steps,
        'pairs': steps * 16, 'rows': steps * 32, 'anchor_by_source': {'PingFang': 1},
        'counterpart_by_source': {'Noto Sans CJK SC': 1},
        'rows_by_source': {'PingFang': 1}, 'rows_by_family': {'PingFang': 1},
        'pairs_by_source': {'pair': 1},
        'anchor_by_script_length': {'slot': 1}, 'pairs_by_view': {'native': 1},
        'rows_by_identity': {'identity': 1}, 'rows_by_face': {'face': 1}}


def test_plan_and_pair_budget_are_exact_and_fail_closed():
    assert trainer.validate_pair_contract() == trainer.PAIR_POLICY
    args = SimpleNamespace(teacher='/teacher', ios_data='/ios', android_data='/android',
                           android_manifest_sha='a' * 64)
    plan = trainer.design(args, {'bound': 'sha'})
    assert plan['schema'] == 'flux-glyph-unified-native-pairs-training-plan-v1'
    assert plan['steps'] == plan['evaluation_step'] == 2400 and plan['seed'] == 2026091401
    assert plan['objective']['replay_batch'] == 96 and plan['objective']['focus_batch'] == 32
    assert plan['objective']['mobile_batch'] == 20 and plan['objective']['mobile_weight'] == .5
    assert plan['objective']['pair_rows_have_ce'] is False
    assert plan['objective']['pair_rows_have_size_loss'] is False
    assert plan['objective']['cnn_input_rows_per_step'] == 180
    assert plan['objective']['one_cnn_forward_per_step'] is True
    assert plan['expected_pair_budget'] == {'pairs': 38400, 'rows': 76800}
    assert plan['runtime'] == trainer.FIXED_RUNTIME and plan['android_system_guard_policy']['maximum'] == 27
    report = pair_report(4)
    assert trainer._require_pair_budget(report, deepcopy(report), 4) is report
    changed = deepcopy(report); changed['rows'] -= 1
    with pytest.raises(ValueError): trainer._require_pair_budget(changed, changed, 4)
    with pytest.raises(ValueError): trainer._require_pair_budget(report, {**report, 'pairs': 63}, 4)
