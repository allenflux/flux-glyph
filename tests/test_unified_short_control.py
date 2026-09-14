import copy
import pytest

torch = pytest.importorskip('torch')
from training import train_unified_short_control as control
from training import train_unified_retention_micro_recovery as base
from training.prepare_unified_regions import FAMILIES


def _rows():
    return [{'split': 'train', 'native_font_verified': True, 'family': FAMILIES[i % 25],
             'target': i % 25, 'tile_start': 0, 'tile_count': 1,
             'log_em_ratio': 0.0, 'source_id': str(i), 'region_id': str(i),
             'domain': 'ios', 'view': 'native'}
            for i in range(96)]


def _batch(model):
    rows = _rows()
    images = torch.rand(128, 1, 64, 256, requires_grad=True)
    targets = torch.tensor([r['target'] for r in rows], dtype=torch.int64)
    sizes = torch.linspace(-.4, .4, 96)
    teacher = torch.full((96, 25), -2.)
    teacher[torch.arange(96), targets] = 2.
    return {'images': images, 'targets': targets, 'sizes': sizes, 'teacher': teacher,
            'rows': rows, 'short_rows': rows[:32]}


def test_control_objective_matches_full_replay_and_declares_zero_short_supervision():
    from wide_region_network import WideRegionFontClassifier
    torch.manual_seed(7)
    model = WideRegionFontClassifier(25)
    batch = _batch(model)
    got, parts = control.objective(model, batch)
    # A separate 96-row forward checks absence of cross-sample coupling.
    reference = copy.deepcopy(model)
    font, size, features = base.forward_with_features(reference, batch['images'][:96])
    expected = base.full_losses(font, size, features, reference.family_head.weight,
                                batch['targets'], batch['sizes'], batch['teacher'], batch['rows'], FAMILIES)[0]
    torch.testing.assert_close(got, expected)
    assert parts['short_supervised'].item() == 0.0 and parts['short_forward_zero'].item() == 0.0
    got.backward()
    expected.backward()
    for (name, actual), (reference_name, wanted) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == reference_name
        torch.testing.assert_close(actual.grad, wanted.grad, atol=2e-6, rtol=2e-4)


def test_short_forward_has_zero_gradient_while_replay_trains_all_groups():
    from wide_region_network import WideRegionFontClassifier
    torch.manual_seed(8)
    model = WideRegionFontClassifier(25)
    batch = _batch(model)
    loss, _ = control.objective(model, batch)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.parameters())
    assert batch['images'].grad is not None
    assert batch['images'].grad[96:].abs().sum() == 0
    assert batch['images'].grad[:96].abs().sum() > 0


def test_control_protocol_is_replay_only_and_fixed():
    assert control.STEPS == 2400 and control.SEED == 2026091401
    assert control.LEARNING_RATE == 5e-6 and control.MINIMUM_LEARNING_RATE == 5e-7
    assert control.OBJECTIVE['short_supervised_rows'] == 0
    assert control.OBJECTIVE['short_forwarded_rows'] == 32
    assert control.OBJECTIVE['original_full_correct_teacher_kl_preserved'] is True
    assert control.OBJECTIVE['runtime_changes'] is False and control.OBJECTIVE['model_count'] == 1


def test_corrupt_short_targets_and_sizes_cannot_change_control_loss():
    from wide_region_network import WideRegionFontClassifier
    torch.manual_seed(9)
    model = WideRegionFontClassifier(25)
    first = _batch(model)
    second = copy.deepcopy(first)
    second['short_targets'] = torch.full((32,), 24)
    second['short_sizes'] = torch.randn(32) * 100
    first_loss, _ = control.objective(model, first)
    second_loss, _ = control.objective(model, second)
    torch.testing.assert_close(first_loss, second_loss)


def test_control_design_has_v1_control_schema_and_no_release_state():
    plan = control.design(type('Args', (), {'teacher': '/teacher', 'short': '/short'})(), {})
    assert plan['schema'] == 'flux-glyph-unified-short-control-plan-v1'
    assert plan['steps'] == plan['evaluation_step'] == 2400
    assert plan['test_read'] is False and plan['development_holdout_read'] is False


def test_control_cal_success_still_requires_export_and_development(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).with_name('test_unified_short_training.py')
    spec = importlib.util.spec_from_file_location('control_cal_gate_fixture', path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    monkeypatch.setattr(fixture, 'trainer', control)
    fixture.test_calibration_success_is_recorded_but_never_becomes_release_promotion(monkeypatch, tmp_path)
