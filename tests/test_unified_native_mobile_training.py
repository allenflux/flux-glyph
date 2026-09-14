"""Verify the unified replay, focus, and native-mobile training boundary."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from training import train_unified_native_mobile as trainer
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


def mobile_rows(step=0):
    rows = []; indices = {'ios': 0, 'android': 0}
    for family in trainer.MOBILE_KNOWN:
        platform = 'ios' if family in ('PingFang', 'SF Pro', 'Helvetica') else 'android'
        index = indices[platform]; indices[platform] += 1
        rows.append({'split': 'train', 'native_font_verified': True, 'target': FAMILIES.index(family),
                     'family': family, 'source_font_family': family, 'short_dataset': platform,
                     'domain': platform, 'view': 'native', 'tile_start': index, 'tile_count': 1,
                     'glyph_count': index % 4 + 1, 'log_em_ratio': (index - 8) / 100,
                     'source_id': f'{platform}-{step}-{index}', 'region_id': f'r-{platform}-{step}-{index}'})
    for offset in range(3):
        source = trainer.MOBILE_UNKNOWN_SOURCES[(step * 3 + offset) % 4]
        index = indices['android']; indices['android'] += 1
        rows.append({'split': 'train', 'native_font_verified': True, 'target': 24,
                     'family': '__unknown__', 'source_font_family': source, 'short_dataset': 'android',
                     'domain': 'android', 'view': 'native', 'tile_start': index, 'tile_count': 1,
                     'glyph_count': index % 4 + 1, 'log_em_ratio': (index - 8) / 100,
                     'source_id': f'android-{step}-{index}', 'region_id': f'r-android-{step}-{index}'})
    return rows


def batch():
    rows = replay_rows(); focus = deepcopy(rows[:32]); mobile = mobile_rows()
    targets = torch.tensor([r['target'] for r in rows])
    teacher = torch.full((96, 25), -2.)
    teacher[torch.arange(96), targets] = 2.
    return {'images': torch.rand(148, 1, 64, 256, requires_grad=True),
            'targets': targets, 'sizes': torch.linspace(-.3, .3, 96),
            'teacher': teacher, 'rows': rows, 'focus_rows': focus,
            'focus_targets': targets[:32], 'focus_sizes': torch.linspace(-.2, .2, 32),
            'focus_teacher': teacher[:32], 'mobile_rows': mobile,
            'mobile_targets': torch.tensor([r['target'] for r in mobile]),
            'mobile_sizes': torch.tensor([r['log_em_ratio'] for r in mobile], dtype=torch.float32)}


def test_objective_preserves_full_losses_and_adds_exact_mobile_term():
    from wide_region_network import WideRegionFontClassifier
    torch.set_num_threads(2); torch.manual_seed(71)
    model = WideRegionFontClassifier(25).eval(); values = batch()
    total, parts = trainer.objective(model, values)
    font, size, features = r22.forward_with_features(model, values['images'])
    replay = r22.full_losses(font[:96], size[:96], features[:96], model.family_head.weight,
        values['targets'], values['sizes'], values['teacher'], values['rows'], FAMILIES)
    focus = trainer.focus_full_loss(font[96:128], size[96:128], features[96:128],
        model.family_head.weight, values['focus_targets'], values['focus_sizes'],
        values['focus_teacher'], values['focus_rows'])
    mobile_ce = torch.nn.functional.cross_entropy(font[128:], values['mobile_targets'])
    mobile_size = torch.nn.functional.smooth_l1_loss(size[128:], values['mobile_sizes'], beta=.05)
    mobile_loss = .5 * (mobile_ce + .2 * mobile_size)
    expected = replay[0] + .5 * focus[0] + .25 * parts['unknown_oe'] + mobile_loss
    torch.testing.assert_close(total, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(parts['replay'], replay[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(parts['teacher_kl'], replay[3], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(parts['mobile_ce'], mobile_ce)
    torch.testing.assert_close(parts['mobile_size'], mobile_size)
    torch.testing.assert_close(parts['mobile_loss'], mobile_loss)
    assert parts['unknown_oe_rows'].item() == 16


def test_mobile_gradient_uses_only_mobile_pixels_and_all_20_verified_labels():
    from wide_region_network import WideRegionFontClassifier
    torch.set_num_threads(2); torch.manual_seed(72)
    model = WideRegionFontClassifier(25).train(); values = batch()
    _, parts = trainer.objective(model, values)
    parts['mobile_loss'].backward()
    affected = values['images'].grad.flatten(1).abs().sum(1)
    assert torch.count_nonzero(affected[:128]) == 0
    assert torch.all(affected[128:] > 0)
    assert values['mobile_targets'][-3:].tolist() == [24, 24, 24]
    for group in ('trunk', 'style', 'family_head', 'size_head'):
        gradients = [p.grad for name, p in model.named_parameters() if name.startswith(group + '.')]
        assert gradients and any(g is not None and g.abs().sum() > 0 for g in gradients)


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


def mobile_data():
    pools = {}; indices = {'ios': 0, 'android': 0}
    for source in (*trainer.MOBILE_KNOWN, *trainer.MOBILE_UNKNOWN_SOURCES):
        known = source in trainer.MOBILE_KNOWN
        platform = 'ios' if known and source in ('PingFang', 'SF Pro', 'Helvetica') else 'android'
        index = indices[platform]; indices[platform] += 1
        family = source if known else '__unknown__'
        base = {'split': 'train', 'native_font_verified': True, 'target': FAMILIES.index(family),
                'family': family, 'source_font_family': source, 'short_dataset': platform,
                'domain': platform, 'tile_start': index, 'tile_count': 1, 'glyph_count': 1,
                'log_em_ratio': (index - 8) / 100, 'source_id': f'{platform}-{index}',
                'region_id': f'r-{platform}-{index}'}
        identity = (platform, base['source_id'], base['region_id'])
        views = {view: {**base, 'view': view} for view in trainer.MOBILE_SAMPLING['view_weights']}
        pools[(source, 'latin', 1)] = {source + '-face': {identity: views}}
    ios_tiles = np.stack([np.full((1, 64, 256), 5 + i / 100, np.float32) for i in range(3)])
    android_tiles = np.stack([np.full((1, 64, 256), 7 + i / 100, np.float32) for i in range(18)])
    return {'pools': pools, 'datasets': {'ios': {'tiles': ios_tiles}, 'android': {'tiles': android_tiles}}}


def test_inputs_append_exact_dataset_tiles_without_changing_replay_rng():
    rows, replay = sources(); focus = list(reversed(rows[:32])); data = mobile_data()
    original = RowsSampler(rows, 301); expected = RowsSampler(rows, 301)
    mobile_sampler = trainer.NativeMobileShortSampler(data, 303)
    expected_mobile_sampler = trainer.NativeMobileShortSampler(data, 303)
    mobile = expected_mobile_sampler.batch()
    expected_images, expected_teacher, expected_size = r22.pixel_batch(
        expected.batch(), replay['original'], replay['supplement'], replay['known'], expected.rng)
    value = trainer.inputs(replay, original, RowsSampler(focus, 302), data, mobile_sampler, 'cpu')
    assert value['images'].shape == (148, 1, 64, 256)
    np.testing.assert_array_equal(value['images'][:96].numpy(), expected_images)
    np.testing.assert_array_equal(value['teacher'].numpy(), expected_teacher)
    np.testing.assert_array_equal(value['sizes'].numpy(), expected_size)
    assert original.rng.bit_generator.state == expected.rng.bit_generator.state
    assert mobile_sampler.rng.bit_generator.state == expected_mobile_sampler.rng.bit_generator.state
    assert [(r['source_id'], r['view']) for r in value['mobile_rows']] == [(r['source_id'], r['view']) for r in mobile]
    expected_mobile = np.stack([data['datasets'][r['short_dataset']]['tiles'][r['tile_start']] for r in mobile])
    np.testing.assert_array_equal(value['images'][128:].numpy(), expected_mobile)
    assert value['mobile_targets'].tolist() == [r['target'] for r in mobile]


def test_plan_and_exact_mobile_budgets_are_frozen():
    args = SimpleNamespace(teacher='/teacher', ios_data='/ios', android_data='/android',
                           android_manifest_sha='a' * 64)
    plan = trainer.design(args, {'bound': 'sha'})
    assert plan['schema'] == 'flux-glyph-unified-native-mobile-training-plan-v1'
    assert plan['steps'] == plan['evaluation_step'] == 2400
    assert plan['seed'] == 2026091401
    assert plan['objective']['replay_batch'] == 96 and plan['objective']['focus_batch'] == 32
    assert plan['objective']['mobile_batch'] == 20 and plan['objective']['mobile_weight'] == .5
    assert plan['objective']['unknown_oe_multiplier'] == .25
    assert plan['expected_mobile_sampling']['rows'] == 48000
    assert plan['expected_mobile_sampling']['unknown_rows'] == 7200
    assert plan['expected_mobile_sampling']['by_platform'] == {'ios': 7200, 'android': 40800}
    assert set(plan['expected_mobile_sampling']['known_by_family'].values()) == {2400}
    assert set(plan['expected_mobile_sampling']['unknown_by_source'].values()) == {1800}
    assert plan['android_system_guard_policy']['maximum'] == plan['android_system_guard_policy']['r22_actual'] == 27
    assert plan['test_read'] is False and plan['development_holdout_read'] is False
    families = {**{f: 4 for f in trainer.MOBILE_KNOWN}, '__unknown__': 12}
    sources = {**{f: 4 for f in trainer.MOBILE_KNOWN},
               **{s: 3 for s in trainer.MOBILE_UNKNOWN_SOURCES}}
    report = {'steps': 4, 'rows': 80, 'by_source': sources}
    trainer._require_mobile_budget(report, trainer.Counter(families), trainer.Counter(sources),
                                   trainer.Counter({'android': 68, 'ios': 12}), 4)
    with pytest.raises((ValueError, RuntimeError)):
        trainer._require_mobile_budget(report, trainer.Counter(families), trainer.Counter(sources),
                                       trainer.Counter({'android': 67, 'ios': 13}), 4)


def test_android_guard_counts_wrong_named_target_families_only_on_true_android_domain():
    details = [
        {'domain': 'android', 'wrong_named': True, 'predicted_family': 'PingFang'},
        {'domain': 'android', 'wrong_named': True, 'predicted_family': 'SF Pro'},
        {'domain': 'android', 'wrong_named': True, 'predicted_family': 'Roboto'},
        {'domain': 'android', 'wrong_named': False, 'predicted_family': 'Helvetica'},
        {'domain': 'ios', 'wrong_named': True, 'predicted_family': 'Helvetica'},
    ]
    result = trainer.android_system_guard(details)
    assert result['actual'] == 2 and result['passed'] is True
    failed = trainer.android_system_guard(details[:2] * 14)
    assert failed['actual'] == 28 and failed['passed'] is False


def test_final_report_keeps_original_53_and_android_guard_separate_and_requires_dev(monkeypatch, tmp_path):
    torch.set_num_threads(2)
    base = tmp_path / 'r22'; base.mkdir()
    np.savez(base / 'CALIBRATION_OUTPUTS.npz', logits=np.zeros((30, 25), np.float32),
             log_em_ratio=np.zeros(30, np.float32))
    retention = tmp_path / 'retention.json'; retention.write_text('{}')
    pre = tmp_path / 'preflight.json'; pre.write_text(json.dumps({'passed': True, 'bindings': {}, 'test_read': False}))
    args = SimpleNamespace(output=tmp_path / 'run', plan=tmp_path / 'plan.json', teacher=tmp_path / 'teacher',
                           ios_data=tmp_path / 'ios', android_data=tmp_path / 'android',
                           android_manifest_sha='a' * 64, device='cpu')
    monkeypatch.setattr(trainer, 'BASE_RUN', base); monkeypatch.setattr(trainer, 'RETENTION_PLAN', retention)
    monkeypatch.setattr(trainer, 'STEPS', 4); monkeypatch.setattr(trainer, 'EXPECTED_FOCUS_TEACHER_ELIGIBLE', 80)
    monkeypatch.setattr(trainer, 'files_bound', lambda *args: {})
    metrics = {'known_correct_coverage': .65, 'named_precision': .95, 'unknown_not_named_rate': .85,
               'passed': True, 'per_domain': {'ios': {'known_correct_coverage': .75},
                                             'android': {'known_correct_coverage': .55}}}
    model = torch.nn.Linear(1, 1, bias=False)
    focus = [{'family': f, 'target': FAMILIES.index(f), 'source_font_family': f}
             for f in trainer.FOCUS_KNOWN_FAMILIES for _ in range(2)]
    focus += [{'family': '__unknown__', 'target': 24, 'source_font_family': 'Smiley Sans'} for _ in range(4)]
    focus_counts = {f: 8 for f in trainer.FOCUS_KNOWN_FAMILIES}; focus_counts['__unknown__'] = 16
    mobile_source_counts = {**{f: 4 for f in trainer.MOBILE_KNOWN},
                            **{s: 3 for s in trainer.MOBILE_UNKNOWN_SOURCES}}

    class ReplayStub:
        def report(self): return {'synthetic': 'replay'}

    class FocusStub:
        def report(self):
            return {'steps': 4, 'rows': 128, 'by_family': focus_counts,
                    'by_glyph_count': {'3': 64, '4': 64}}

    class MobileStub:
        def report(self): return {'steps': 4, 'rows': 80, 'by_source': mobile_source_counts}

    mobile_report = MobileStub().report()
    monkeypatch.setattr(trainer, 'context', lambda *args:
        (model, {}, {}, ReplayStub(), FocusStub(), {}, MobileStub(),
         {'selected': {'metrics': deepcopy(metrics)}}, {'synthetic': 'teacher'},
         {'synthetic': 'focus'}, {'synthetic': 'mobile'}, {'full': mobile_report}))
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
        batches.append({'rows': rows, 'focus_rows': focus, 'mobile_rows': mobile_rows(step)})
    iterator = iter(batches); monkeypatch.setattr(trainer, 'inputs', lambda *args: next(iterator))

    def objective(model, batch):
        value = model.weight.square().sum()
        return value, {'focus_teacher_eligible': value.new_tensor(20),
                       'unknown_oe_rows': value.new_tensor(16)}

    monkeypatch.setattr(trainer, 'objective', objective)
    cal = {'rows': [{} for _ in range(30)], 'manifest_sha256': 'm', 'partition_sha256': 'p'}
    monkeypatch.setattr(trainer, 'load_split', lambda *args: cal)
    record = {'metrics': deepcopy(metrics), 'retention_checks': [{'passed': True}] * 46,
              'promotion_allowed': True}
    baseline_details = [{'domain': 'android', 'wrong_named': i < 27,
                         'predicted_family': 'PingFang' if i < 27 else 'Roboto'} for i in range(30)]
    trial_details = deepcopy(baseline_details)
    evaluations = iter([(deepcopy(record), [], baseline_details), (deepcopy(record), [], trial_details)])
    monkeypatch.setattr(trainer, 'evaluate_outputs', lambda *args: next(evaluations))
    measured = iter([{'known_views': 100, 'correct_named': n, 'known_correct_coverage': n / 100,
                     'unknown_views': 50, 'unknown_falsely_named': 10} for n in (50, 52)])
    monkeypatch.setattr(trainer, 'short_population', lambda *args: next(measured))
    monkeypatch.setattr(trainer, 'infer',
                        lambda *args: (np.zeros((30, 25), np.float32), np.zeros(30, np.float32)))
    args.plan.write_text(json.dumps({**trainer.design(args, {}),
                                    'preflight': {'path': str(pre), 'sha256': trainer.sha(pre)}}))
    trainer.train(args)
    counts = json.loads((args.output / 'TRAINING_COUNTS.json').read_text())
    selection = json.loads((args.output / 'SELECTION.json').read_text())
    report = json.loads((args.output / 'report.json').read_text())
    assert counts['replay_rows'] == 384 and counts['unknown_oe_rows'] == 64
    assert counts['mobile_rows'] == 80 and counts['mobile_unknown_rows'] == 12
    assert counts['mobile_by_platform'] == {'android': 68, 'ios': 12}
    assert selection['original_53_calibration_checks_passed'] is True
    assert selection['android_system_guard']['actual'] == 27
    assert selection['calibration_promotion_allowed'] is True
    assert selection['promotion_allowed'] is False and selection['development_evaluated'] is False
    assert selection['exported'] is False and selection['deployed'] is False
    assert report['original_checks'] == 46 and report['additional_checks'] == 7
    assert report['promotion_allowed'] is False and report['development_evaluated'] is False
