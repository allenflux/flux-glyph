"""Verify native focus tile/teacher pairing, gradients and release boundaries."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from training import train_unified_native_short as trainer
from training import train_unified_retention_micro_recovery as r22
from training.prepare_unified_regions import FAMILIES


def replay_rows():
    return [{'split': 'train', 'native_font_verified': True, 'target': i % 25,
             'family': FAMILIES[i % 25], 'source_font_family': FAMILIES[i % 25],
             'domain': 'android', 'view': 'native', 'tile_start': i, 'tile_count': 1,
             'log_em_ratio': i / 1000, 'source_id': f's{i}', 'region_id': f'r{i}'}
            for i in range(96)]


def batch():
    rows = replay_rows(); focus = deepcopy(rows[:32])
    targets = torch.tensor([r['target'] for r in rows])
    teacher = torch.full((96, 25), -2.)
    teacher[torch.arange(96), targets] = 2.
    return {'images': torch.rand(128, 1, 64, 256, requires_grad=True),
            'targets': targets, 'sizes': torch.linspace(-.3, .3, 96),
            'teacher': teacher, 'rows': rows, 'focus_rows': focus,
            'focus_targets': targets[:32], 'focus_sizes': torch.linspace(-.2, .2, 32),
            'focus_teacher': teacher[:32]}


def test_focus_gradients_reach_all_model_groups_and_only_focus_input_pixels():
    from wide_region_network import WideRegionFontClassifier
    torch.set_num_threads(4); torch.manual_seed(51)
    model = WideRegionFontClassifier(25).train(); values = batch()
    total, parts = trainer.objective(model, values)
    assert torch.isfinite(total)
    parts['focus_weighted'].backward()
    assert values['images'].grad is not None
    assert torch.count_nonzero(values['images'].grad[:96]) == 0
    assert torch.count_nonzero(values['images'].grad[96:]) > 0
    for group in ('trunk', 'style', 'family_head', 'size_head'):
        grads = [p.grad for name, p in model.named_parameters() if name.startswith(group + '.')]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)


def test_original_replay_loss_is_preserved_with_focus_in_the_same_cnn_batch():
    from wide_region_network import WideRegionFontClassifier
    torch.set_num_threads(4); torch.manual_seed(52)
    model = WideRegionFontClassifier(25).train(); values = batch()
    _, parts = trainer.objective(model, values)
    font, size, features = r22.forward_with_features(model, values['images'][:96])
    reference = r22.full_losses(font, size, features, model.family_head.weight,
        values['targets'], values['sizes'], values['teacher'], values['rows'], FAMILIES)
    torch.testing.assert_close(parts['replay'], reference[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(parts['teacher_kl'], reference[3], atol=1e-6, rtol=1e-6)


class RowsSampler:
    def __init__(self, rows, seed):
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


def test_focus_uses_exact_same_tile_teacher_and_keeps_replay_rng():
    rows, replay = sources()
    focus_rows = list(reversed(rows[:32]))
    original = RowsSampler(rows, 301); expected = RowsSampler(rows, 301)
    expected_images, expected_teacher, expected_size = r22.pixel_batch(
        expected.batch(), replay['original'], replay['supplement'], replay['known'], expected.rng)
    value = trainer.inputs(replay, original, RowsSampler(focus_rows, 302), 'cpu')
    assert value['images'].shape == (128, 1, 64, 256)
    np.testing.assert_array_equal(value['images'][:96].numpy(), expected_images)
    np.testing.assert_array_equal(value['teacher'].numpy(), expected_teacher)
    np.testing.assert_array_equal(value['sizes'].numpy(), expected_size)
    assert original.rng.bit_generator.state == expected.rng.bit_generator.state
    for i, row in enumerate(focus_rows):
        index = row['tile_start']
        np.testing.assert_array_equal(value['images'][96 + i].numpy(), replay['original']['tiles'][index])
        np.testing.assert_array_equal(value['focus_teacher'][i].numpy(), replay['original']['teacher_logits'][index])
        assert value['focus_indices'][i] == ('original', index)
        assert value['focus_targets'][i].item() == row['target']
        assert value['focus_sizes'][i].item() == pytest.approx(row['log_em_ratio'])


@pytest.mark.parametrize('change', ['target', 'source_id', 'view', 'split', 'tile_count'])
def test_focus_rejects_relabelled_or_misrouted_rows_before_forward(change):
    rows, replay = sources(); focus = deepcopy(rows[:32])
    focus[0][change] = {'target': 2, 'source_id': 'other', 'view': 'half',
                        'split': 'calibration', 'tile_count': 2}[change]
    with pytest.raises((ValueError, RuntimeError)):
        trainer.inputs(replay, RowsSampler(rows, 1), RowsSampler(focus, 2), 'cpu')


def test_plan_preserves_all_fixed_acceptance_rules():
    plan = trainer.design(SimpleNamespace(teacher='/teacher'), {'bound': 'sha'})
    assert plan['steps'] == plan['evaluation_step'] == 2400
    assert plan['objective']['replay_batch'] == 96 and plan['objective']['focus_batch'] == 32
    assert plan['objective']['focus_weight'] == .5
    assert plan['objective']['derived_short_crops_used'] is False
    assert plan['objective']['runtime_changes'] is False
    assert plan['additional_acceptance']['original_46_checks_required'] is True
    assert plan['additional_acceptance']['original_18_development_checks_required'] is True
    assert plan['additional_acceptance']['r22_18_development_checks_also_required'] is True
    assert plan['test_read'] is False and plan['development_holdout_read'] is False


def test_completed_cal_pass_still_requires_export_and_both_development_comparisons(monkeypatch, tmp_path):
    torch.set_num_threads(2)
    base = tmp_path / 'r22'; base.mkdir()
    np.savez(base / 'CALIBRATION_OUTPUTS.npz', logits=np.zeros((2, 25), np.float32),
             log_em_ratio=np.zeros(2, np.float32))
    retention = tmp_path / 'retention.json'; retention.write_text('{}')
    pre = tmp_path / 'preflight.json'; pre.write_text(json.dumps({'passed': True, 'bindings': {}, 'test_read': False}))
    args = SimpleNamespace(output=tmp_path / 'run', plan=tmp_path / 'plan.json', teacher=tmp_path / 'teacher', device='cpu')
    monkeypatch.setattr(trainer, 'BASE_RUN', base); monkeypatch.setattr(trainer, 'RETENTION_PLAN', retention)
    monkeypatch.setattr(trainer, 'STEPS', 4); monkeypatch.setattr(trainer, 'files_bound', lambda *args: {})
    metrics = {'known_correct_coverage': .65, 'named_precision': .95, 'unknown_not_named_rate': .85,
               'passed': False, 'per_domain': {'ios': {'known_correct_coverage': .75},
                                              'android': {'known_correct_coverage': .55}}}
    model = torch.nn.Linear(1, 1, bias=False)
    focus = [{'family': f, 'target': FAMILIES.index(f), 'source_font_family': f}
             for f in trainer.FOCUS_KNOWN_FAMILIES for _ in range(2)]
    focus += [{'family': '__unknown__', 'target': 24, 'source_font_family': 'Smiley Sans'} for _ in range(4)]
    counts = {f: 8 for f in trainer.FOCUS_KNOWN_FAMILIES}; counts['__unknown__'] = 16
    class Stub:
        def report(self):
            return {'steps': 4, 'rows': 128, 'by_family': counts, 'by_glyph_count': {'3': 64, '4': 64}}
    stub = Stub()
    monkeypatch.setattr(trainer, 'context', lambda *args:
        (model, {}, {}, stub, stub, {'selected': {'metrics': deepcopy(metrics)}},
         {'synthetic': True}, {'synthetic': True}))
    monkeypatch.setattr(trainer, 'parameter_groups', lambda state: {'all': float(next(iter(state.values())).sum())})
    from short_region_objective import SHORT_UNKNOWN_SOURCES
    names = [*SHORT_UNKNOWN_SOURCES, 'WenQuanYi Zen Hei', 'Zhuque Fangsong']
    batches = []
    for step in range(4):
        rows = [{'family': '__unknown__', 'target': 24, 'source_font_family': names[(step * 16 + i) % 11]}
                for i in range(16)]
        rows += [{'family': FAMILIES[i % 24], 'target': i % 24, 'source_font_family': FAMILIES[i % 24]}
                 for i in range(80)]
        batches.append({'rows': rows, 'focus_rows': focus})
    iterator = iter(batches); monkeypatch.setattr(trainer, 'inputs', lambda *args: next(iterator))
    def objective(model, batch):
        value = model.weight.square().sum()
        return value, {'focus_teacher_eligible': value.new_tensor(20)}
    monkeypatch.setattr(trainer, 'objective', objective)
    cal = {'rows': [{}, {}], 'manifest_sha256': 'm', 'partition_sha256': 'p'}
    monkeypatch.setattr(trainer, 'load_split', lambda *args: cal)
    record = {'metrics': deepcopy(metrics), 'retention_checks': [{'passed': True}] * 46, 'promotion_allowed': True}
    monkeypatch.setattr(trainer, 'evaluate_outputs', lambda *args: (deepcopy(record), [], [{}, {}]))
    measured = iter([{'known_views': 100, 'correct_named': n, 'known_correct_coverage': n / 100,
                     'unknown_views': 50, 'unknown_falsely_named': 10} for n in (50, 52)])
    monkeypatch.setattr(trainer, 'short_population', lambda *args: next(measured))
    monkeypatch.setattr(trainer, 'infer', lambda *args: (np.zeros((2, 25), np.float32), np.zeros(2, np.float32)))
    args.plan.write_text(json.dumps({**trainer.design(args, {}), 'preflight': {'path': str(pre), 'sha256': trainer.sha(pre)}}))
    trainer.train(args)
    counts = json.loads((args.output / 'TRAINING_COUNTS.json').read_text())
    selection = json.loads((args.output / 'SELECTION.json').read_text())
    assert counts['replay_rows'] == 384 and counts['focus_rows'] == counts['focus_supervised_rows'] == 128
    assert counts['focus_by_glyph_count'] == {'3': 64, '4': 64}
    assert counts['focus_teacher_eligible_rows'] == 80
    assert selection['calibration_promotion_allowed'] is True
    assert selection['promotion_allowed'] is False and selection['development_evaluated'] is False
    assert selection['native_focus_proof'] == {'synthetic': True}
