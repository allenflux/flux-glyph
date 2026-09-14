"""Numerical contracts for the frozen native-pairs loss diagnostics."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from training import font_gradient_diagnostics as diagnostics
from training import train_unified_native_pairs as trainer


def _frozen_native_pairs_fixture():
    """Reuse the existing synthetic 180-row fixture without loading artifacts."""
    path = Path(__file__).with_name('test_unified_native_pairs_training.py')
    spec = importlib.util.spec_from_file_location('_native_pairs_test_fixture', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.batch()


def _state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_decompose_matches_frozen_objective_and_gradients_without_mutation():
    from training.wide_region_network import WideRegionFontClassifier

    torch.set_num_threads(2)
    torch.manual_seed(20260914)
    model = WideRegionFontClassifier(25).eval()
    batch = _frozen_native_pairs_fixture()
    before = _state(model)
    assert all(parameter.grad is None for parameter in model.parameters())

    frozen_total, _ = trainer.objective(model, batch)
    diagnostic_total, terms = diagnostics.decompose(model, batch)
    torch.testing.assert_close(diagnostic_total, frozen_total, rtol=2e-6, atol=2e-6)
    assert tuple(terms) == diagnostics.TERMS
    torch.testing.assert_close(sum(terms.values()), diagnostic_total, rtol=2e-6, atol=2e-6)

    groups, error = diagnostics.gradients(model, diagnostic_total, terms)
    assert error['relative_l2_error'] <= 5e-5
    assert error['maximum_absolute_error'] <= 2e-4

    size_head = groups['size_head']['weighted_norms']
    assert size_head['size'] > 1e-8
    assert all(size_head[name] <= 1e-12 for name in diagnostics.TERMS if name != 'size')
    family_head = groups['family_head']['weighted_norms']
    assert family_head['size'] <= 1e-12
    assert any(family_head[name] > 1e-8 for name in ('replay_font', 'focus_font', 'mobile_font'))

    assert all(parameter.grad is None for parameter in model.parameters())
    after = _state(model)
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)


def test_geometry_uses_weighted_gradient_dot_cosines_and_none_for_zero_norms():
    weighted = np.zeros((len(diagnostics.TERMS), 3), dtype=np.float64)
    weighted[0] = (2.0, 4.0, 0.0)       # 2 * (1, 2, 0)
    weighted[1] = (.75, 1.0, 0.0)       # .25 * (3, 4, 0)
    result = diagnostics.geometry(weighted @ weighted.T)

    expected_dot = 5.5
    expected_cosine = expected_dot / (np.sqrt(20.0) * 1.25)
    assert result['weighted_norms']['replay_font'] == pytest.approx(np.sqrt(20.0))
    assert result['weighted_norms']['focus_font'] == pytest.approx(1.25)
    assert result['cosines']['replay_font']['focus_font'] == pytest.approx(expected_cosine)
    assert result['cosines']['focus_font']['replay_font'] == pytest.approx(expected_cosine)
    for name in diagnostics.TERMS[2:]:
        assert result['weighted_norms'][name] == 0.0
        assert result['cosines']['replay_font'][name] is None
        assert result['cosines'][name]['replay_font'] is None


def test_reference_alignment_reports_full_and_removed_size_teacher_cases():
    # Terms 0, 1, 2 are the font-supervision reference; size and teacher
    # intentionally share its first basis direction.
    vectors = np.zeros((len(diagnostics.TERMS), 3), dtype=np.float64)
    vectors[0] = (1.0, 0.0, 0.0)
    vectors[1] = (0.0, 1.0, 0.0)
    vectors[2] = (0.0, 0.0, 1.0)
    vectors[3] = (1.0, 0.0, 0.0)
    vectors[4] = (1.0, 0.0, 0.0)
    result = diagnostics.geometry(vectors @ vectors.T)
    reference = result['references']['font_supervision']

    assert reference['full_descent_alignment'] == pytest.approx(5.0 / 3.0)
    assert reference['without_size_descent_alignment'] == pytest.approx(4.0 / 3.0)
    assert reference['without_teacher_descent_alignment'] == pytest.approx(4.0 / 3.0)
    assert reference['term_dot_relative_to_reference_squared']['size'] == pytest.approx(1.0 / 3.0)
    assert reference['term_dot_relative_to_reference_squared']['teacher'] == pytest.approx(1.0 / 3.0)


def test_gradient_diagnostic_rejects_preexisting_parameter_gradients():
    from training.wide_region_network import WideRegionFontClassifier

    torch.manual_seed(20260915)
    model = WideRegionFontClassifier(25).eval()
    batch = _frozen_native_pairs_fixture()
    total, terms = diagnostics.decompose(model, batch)
    next(model.parameters()).grad = torch.zeros_like(next(model.parameters()))
    with pytest.raises(ValueError, match='untouched, trainable parameter gradients'):
        diagnostics.gradients(model, total, terms)
