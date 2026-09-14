"""Verify the joint replay, native-focus, and native-iOS training boundary."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from training import train_unified_native_ios as trainer
from training import train_unified_retention_micro_recovery as r22
from training.prepare_unified_regions import FAMILIES


def replay_rows():
    rows = []
    for i in range(96):
        target = i % 24 if i < 80 else 24
        family = FAMILIES[target]
        rows.append({'split': 'train', 'native_font_verified': True, 'target': target,
                     'family': family, 'source_font_family': family if target < 24 else 'Smiley Sans',
                     'domain': 'android', 'view': 'native', 'tile_start': i, 'tile_count': 1,
                     'log_em_ratio': i / 1000, 'source_id': f's{i}', 'region_id': f'r{i}'})
    return rows


def ios_rows():
    rows = []
    for family in trainer.IOS_SAMPLING['families']:
        for glyph_count in (1, 2, 3, 4):
            for repeat in range(2):
                rows.append({'split': 'train', 'target': FAMILIES.index(family), 'family': family,
                             'source_font_family': family, 'domain': 'ios', 'view': 'native',
                             'tile_start': len(rows), 'tile_count': 1, 'glyph_count': glyph_count,
                             'log_em_ratio': (len(rows) - 12) / 100, 'source_id': f'i{len(rows)}',
                             'region_id': f'ir{len(rows)}', 'repeat': repeat})
    return rows


def batch():
    rows = replay_rows(); focus = deepcopy(rows[:32]); ios = ios_rows()
    targets = torch.tensor([r['target'] for r in rows])
    teacher = torch.full((96, 25), -2.)
    teacher[torch.arange(96), targets] = 2.
    return {'images': torch.rand(152, 1, 64, 256, requires_grad=True),
            'targets': targets, 'sizes': torch.linspace(-.3, .3, 96),
            'teacher': teacher, 'rows': rows, 'focus_rows': focus,
            'focus_targets': targets[:32], 'focus_sizes': torch.linspace(-.2, .2, 32),
            'focus_teacher': teacher[:32], 'ios_rows': ios,
            'ios_targets': torch.tensor([r['target'] for r in ios]),
            'ios_sizes': torch.tensor([r['log_em_ratio'] for r in ios], dtype=torch.float32)}


def test_objective_uses_exact_slices_and_weights_for_ios_and_oe():
    from wide_region_network import WideRegionFontClassifier
    torch.set_num_threads(4); torch.manual_seed(61)
    model = WideRegionFontClassifier(25).eval(); values = batch()
    total, parts = trainer.objective(model, values)
    font, size, features = r22.forward_with_features(model, values['images'])
    replay = r22.full_losses(font[:96], size[:96], features[:96], model.family_head.weight,
        values['targets'], values['sizes'], values['teacher'], values['rows'], FAMILIES)
    focus = trainer.focus_full_loss(font[96:128], size[96:128], features[96:128],
        model.family_head.weight, values['focus_targets'], values['focus_sizes'],
        values['focus_teacher'], values['focus_rows'])
    expected_ios_ce = torch.nn.functional.cross_entropy(font[128:152], values['ios_targets'])
    expected_ios_size = torch.nn.functional.smooth_l1_loss(
        size[128:152], values['ios_sizes'], beta=.05)
    expected_ios = .25 * (expected_ios_ce + .2 * expected_ios_size)
    expected = replay[0] + .5 * focus[0] + .25 * parts['unknown_oe'] + expected_ios
    torch.testing.assert_close(total, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(parts['replay'], replay[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(parts['focus_weighted'], .5 * focus[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(parts['ios_ce'], expected_ios_ce)
    torch.testing.assert_close(parts['ios_size'], expected_ios_size)
    torch.testing.assert_close(parts['ios_loss'], expected_ios)
    assert parts['unknown_oe_rows'].item() == 16


def test_ios_and_oe_gradients_are_partitioned_to_their_input_rows():
    from wide_region_network import WideRegionFontClassifier
    torch.set_num_threads(4); torch.manual_seed(62)
    model = WideRegionFontClassifier(25).train(); values = batch()
    _, parts = trainer.objective(model, values)
    parts['ios_loss'].backward(retain_graph=True)
    affected = values['images'].grad.flatten(1).abs().sum(1)
    assert torch.count_nonzero(affected[:128]) == 0
    assert torch.all(affected[128:152] > 0)
    for group in ('trunk', 'style', 'family_head', 'size_head'):
        gradients = [p.grad for name, p in model.named_parameters() if name.startswith(group + '.')]
        assert gradients and any(g is not None and g.abs().sum() > 0 for g in gradients)

    model.zero_grad(set_to_none=True); values['images'].grad.zero_()
    _, parts = trainer.objective(model, values)
    parts['unknown_oe'].backward()
    affected = values['images'].grad.flatten(1).abs().sum(1)
    assert torch.count_nonzero(affected[96:]) == 0
    assert torch.count_nonzero(affected[:80]) == 0
    assert torch.all(affected[80:96] > 0)
    assert model.size_head.weight.grad is None


class RowsSampler:
    def __init__(self, rows, seed=0):
        self.rows = rows; self.rng = np.random.default_rng(seed)

    def batch(self):
        return deepcopy(self.rows)


def sources():
    rows = replay_rows()
    images = np.stack([np.full((1, 64, 256), i / 100, np.float32) for i in range(96)])
    teacher = np.stack([np.full(25, i / 10, np.float32) for i in range(96)])
    source = r22.batch_source({'rows': rows, 'tiles': images, 'partition': {'split': 'train'}},
                             {'base_logits': teacher})
    return rows, {'original': source, 'supplement': source, 'known': source}


def test_inputs_keep_replay_and_focus_unchanged_and_append_24_ios_rows():
    rows, replay = sources(); focus = list(reversed(rows[:32])); ios = ios_rows()
    ios_tiles = np.stack([np.full((1, 64, 256), 5 + i / 100, np.float32) for i in range(24)])
    original = RowsSampler(rows, 301); expected = RowsSampler(rows, 301)
    expected_images, expected_teacher, expected_size = r22.pixel_batch(
        expected.batch(), replay['original'], replay['supplement'], replay['known'], expected.rng)
    value = trainer.inputs(replay, original, RowsSampler(focus, 302),
                           {'tiles': ios_tiles}, RowsSampler(ios), 'cpu')
    assert value['images'].shape == (152, 1, 64, 256)
    np.testing.assert_array_equal(value['images'][:96].numpy(), expected_images)
    np.testing.assert_array_equal(value['teacher'].numpy(), expected_teacher)
    np.testing.assert_array_equal(value['sizes'].numpy(), expected_size)
    assert original.rng.bit_generator.state == expected.rng.bit_generator.state
    for i, row in enumerate(focus):
        index = row['tile_start']
        np.testing.assert_array_equal(value['images'][96 + i].numpy(), replay['original']['tiles'][index])
        np.testing.assert_array_equal(value['focus_teacher'][i].numpy(), replay['original']['teacher_logits'][index])
        assert value['focus_indices'][i] == ('original', index)
    np.testing.assert_array_equal(value['images'][128:].numpy(), ios_tiles)
    assert value['ios_targets'].tolist() == [r['target'] for r in ios]
    np.testing.assert_allclose(value['ios_sizes'].numpy(), [r['log_em_ratio'] for r in ios])


def test_plan_freezes_trial_and_release_boundaries():
    args = SimpleNamespace(teacher='/teacher', ios_data='/ios/prepared')
    plan = trainer.design(args, {'bound': 'sha'})
    assert plan['schema'] == 'flux-glyph-unified-native-ios-training-plan-v1'
    assert plan['steps'] == plan['evaluation_step'] == 2400
    assert plan['seed'] == 2026091401
    assert plan['learning_rate'] == 5e-6 and plan['minimum_learning_rate'] == 5e-7
    assert plan['objective']['replay_batch'] == 96
    assert plan['objective']['focus_batch'] == 32 and plan['objective']['focus_weight'] == .5
    assert plan['objective']['ios_batch'] == 24 and plan['objective']['ios_weight'] == .25
    assert plan['objective']['unknown_oe_multiplier'] == .25
    assert plan['objective']['effective_unknown_oe_coefficient'] == .125
    assert plan['additional_acceptance']['original_46_checks_required'] is True
    assert plan['additional_acceptance']['original_18_development_checks_required'] is True
    assert plan['test_read'] is False and plan['development_holdout_read'] is False


def test_completed_training_records_exact_counts_but_never_self_promotes(monkeypatch, tmp_path):
    torch.set_num_threads(2)
    base = tmp_path / 'r22'; base.mkdir()
    np.savez(base / 'CALIBRATION_OUTPUTS.npz', logits=np.zeros((2, 25), np.float32),
             log_em_ratio=np.zeros(2, np.float32))
    retention = tmp_path / 'retention.json'; retention.write_text('{}')
    pre = tmp_path / 'preflight.json'
    pre.write_text(json.dumps({'passed': True, 'bindings': {}, 'test_read': False}))
    args = SimpleNamespace(output=tmp_path / 'run', plan=tmp_path / 'plan.json',
                           teacher=tmp_path / 'teacher', ios_data=tmp_path / 'ios', device='cpu')
    monkeypatch.setattr(trainer, 'BASE_RUN', base); monkeypatch.setattr(trainer, 'RETENTION_PLAN', retention)
    monkeypatch.setattr(trainer, 'STEPS', 4); monkeypatch.setattr(trainer, 'EXPECTED_FOCUS_TEACHER_ELIGIBLE', 80)
    monkeypatch.setattr(trainer, 'files_bound', lambda *args: {})
    metrics = {'known_correct_coverage': .65, 'named_precision': .95, 'unknown_not_named_rate': .85,
               'passed': False, 'per_domain': {'ios': {'known_correct_coverage': .75},
                                              'android': {'known_correct_coverage': .55}}}
    model = torch.nn.Linear(1, 1, bias=False)
    focus = [{'family': f, 'target': FAMILIES.index(f), 'source_font_family': f}
             for f in trainer.FOCUS_KNOWN_FAMILIES for _ in range(2)]
    focus += [{'family': '__unknown__', 'target': 24, 'source_font_family': 'Smiley Sans'} for _ in range(4)]
    focus_counts = {f: 8 for f in trainer.FOCUS_KNOWN_FAMILIES}; focus_counts['__unknown__'] = 16

    class FocusStub:
        def report(self):
            return {'steps': 4, 'rows': 128, 'by_family': focus_counts,
                    'by_glyph_count': {'3': 64, '4': 64}}

    class IOSStub:
        def report(self):
            return {'steps': 4, 'rows': 96,
                    'by_family': {f: 32 for f in trainer.IOS_SAMPLING['families']},
                    'by_glyph_count': {str(n): 24 for n in (1, 2, 3, 4)}, 'by_view': {}}

    focus_stub = FocusStub(); ios_stub = IOSStub()
    monkeypatch.setattr(trainer, 'context', lambda *args:
        (model, {}, {}, focus_stub, focus_stub, {}, ios_stub,
         {'selected': {'metrics': deepcopy(metrics)}}, {'synthetic': 'teacher'},
         {'synthetic': 'focus'}, {'synthetic': 'ios'}))
    monkeypatch.setattr(trainer, 'parameter_groups',
                        lambda state: {'all': float(next(iter(state.values())).sum())})
    from short_region_objective import SHORT_UNKNOWN_SOURCES
    names = [*SHORT_UNKNOWN_SOURCES, 'WenQuanYi Zen Hei', 'Zhuque Fangsong']
    batches = []
    for step in range(4):
        rows = [{'family': '__unknown__', 'target': 24,
                 'source_font_family': names[(step * 16 + i) % 11]} for i in range(16)]
        rows += [{'family': FAMILIES[i % 24], 'target': i % 24,
                  'source_font_family': FAMILIES[i % 24]} for i in range(80)]
        batches.append({'rows': rows, 'focus_rows': focus, 'ios_rows': ios_rows()})
    iterator = iter(batches); monkeypatch.setattr(trainer, 'inputs', lambda *args: next(iterator))

    def objective(model, batch):
        value = model.weight.square().sum()
        return value, {'focus_teacher_eligible': value.new_tensor(20),
                       'unknown_oe_rows': value.new_tensor(16)}

    monkeypatch.setattr(trainer, 'objective', objective)
    cal = {'rows': [{}, {}], 'manifest_sha256': 'm', 'partition_sha256': 'p'}
    monkeypatch.setattr(trainer, 'load_split', lambda *args: cal)
    record = {'metrics': deepcopy(metrics), 'retention_checks': [{'passed': True}] * 46,
              'promotion_allowed': True}
    monkeypatch.setattr(trainer, 'evaluate_outputs',
                        lambda *args: (deepcopy(record), [], [{}, {}]))
    measured = iter([{'known_views': 100, 'correct_named': n, 'known_correct_coverage': n / 100,
                     'unknown_views': 50, 'unknown_falsely_named': 10} for n in (50, 52)])
    monkeypatch.setattr(trainer, 'short_population', lambda *args: next(measured))
    monkeypatch.setattr(trainer, 'infer',
                        lambda *args: (np.zeros((2, 25), np.float32), np.zeros(2, np.float32)))
    args.plan.write_text(json.dumps({**trainer.design(args, {}),
                                    'preflight': {'path': str(pre), 'sha256': trainer.sha(pre)}}))
    trainer.train(args)
    counts = json.loads((args.output / 'TRAINING_COUNTS.json').read_text())
    selection = json.loads((args.output / 'SELECTION.json').read_text())
    report = json.loads((args.output / 'report.json').read_text())
    assert counts['replay_rows'] == 384 and counts['unknown_oe_rows'] == 64
    assert counts['focus_rows'] == 128 and counts['focus_teacher_eligible_rows'] == 80
    assert counts['ios_rows'] == 96
    assert counts['ios_by_family'] == {f: 32 for f in trainer.IOS_SAMPLING['families']}
    assert counts['ios_by_glyph_count'] == {str(n): 24 for n in (1, 2, 3, 4)}
    assert counts['effective_unknown_oe_coefficient'] == .125
    assert selection['calibration_promotion_allowed'] is True
    assert selection['promotion_allowed'] is False and selection['development_evaluated'] is False
    assert selection['exported'] is False and selection['deployed'] is False
    assert report['promotion_allowed'] is False and report['exported'] is False and report['deployed'] is False
    assert selection['ios_short_proof'] == {'synthetic': 'ios'}
