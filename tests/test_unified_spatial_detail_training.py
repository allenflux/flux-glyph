"""Independent checks that the spatial-detail trainer preserves native-pairs policy."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from training import train_unified_native_pairs as frozen
from training import train_unified_spatial_detail_v2 as spatial
from training.spatial_region_network import expand_spatial_pool
from training.wide_region_network import WideRegionFontClassifier


def _oracle_fixture():
    """Load the existing native-pairs synthetic fixture as the sampler oracle."""
    path = 'tests/test_unified_native_pairs_training.py'
    spec = importlib.util.spec_from_file_location('native_pairs_test_oracle', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source():
    torch.manual_seed(20260917)
    model = WideRegionFontClassifier(25).cpu().eval()
    # Keep size parity observable; the production initializer happens to use a
    # zero size head, which would make a broken transfer look correct.
    with torch.no_grad():
        model.size_head.weight.copy_(torch.linspace(-.02, .02, model.size_head.weight.numel())
                                     .reshape_as(model.size_head.weight))
        model.size_head.bias.fill_(.13)
    return model


def test_sampling_and_objective_match_frozen_native_pairs_oracle():
    oracle = _oracle_fixture()
    rows, replay = oracle.sources()
    focus = list(reversed(rows[:32]))
    mobile = oracle.mobile_data()

    def sam(seed, values):
        return oracle.RowsSampler(values, seed)

    old = frozen.inputs(replay, sam(401, rows), sam(402, focus), mobile,
                        sam(403, oracle.mobile_rows()), oracle.PairSampler(oracle.pair_rows()), 'cpu')
    new = spatial.inputs(replay, sam(401, rows), sam(402, focus), mobile,
                         sam(403, oracle.mobile_rows()), oracle.PairSampler(oracle.pair_rows()), 'cpu')
    assert old.keys() == new.keys()
    for key in old:
        if torch.is_tensor(old[key]):
            torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
        else:
            assert old[key] == new[key]

    torch.set_num_threads(2)
    source = _source()
    target, transfer = expand_spatial_pool(source)
    assert transfer['loss_and_sampling_changed'] is False
    values = oracle.batch()
    old_total, old_parts = frozen.objective(source, values)
    new_total, new_parts = spatial.objective(target, values)
    torch.testing.assert_close(new_total, old_total, rtol=2e-5, atol=2e-6)
    assert old_parts.keys() == new_parts.keys()
    for key in old_parts:
        if torch.is_tensor(old_parts[key]):
            torch.testing.assert_close(new_parts[key], old_parts[key], rtol=2e-5, atol=2e-6)
        else:
            assert new_parts[key] == old_parts[key]


def test_loss_policy_constants_and_plan_fields_are_frozen_except_graph_schema():
    for name in ('STEPS', 'SEED', 'LEARNING_RATE', 'MINIMUM_LEARNING_RATE',
                 'FOCUS_WEIGHT', 'MOBILE_WEIGHT', 'OE_MULTIPLIER', 'PAIR_COUNT',
                 'PAIR_ROWS', 'PAIR_WEIGHT', 'EXPECTED_FOCUS_TEACHER_ELIGIBLE',
                 'EXPECTED_PREFLIGHT_FOCUS_TEACHER_ELIGIBLE'):
        assert getattr(spatial, name) == getattr(frozen, name)
    assert spatial.OBJECTIVE == frozen.OBJECTIVE

    args = SimpleNamespace(teacher='/teacher', ios_data='/ios', android_data='/android',
                           android_manifest_sha='a' * 64)
    transfer = {'schema': 'synthetic-transfer-v1'}
    plan = spatial.design(args, {'bound': 'sha'}, transfer)
    assert plan['reference_trial'] == 'native-pairs-v1'
    assert plan['spatial_transfer'] is transfer
    assert plan['steps'] == plan['evaluation_step'] == frozen.STEPS
    assert plan['objective'] == frozen.OBJECTIVE
    assert plan['runtime'] == frozen.FIXED_RUNTIME
    assert plan['expected_pair_budget'] == {'pairs': frozen.STEPS * frozen.PAIR_COUNT,
                                            'rows': frozen.STEPS * frozen.PAIR_ROWS}
    assert plan['expected_mobile_sampling'] == frozen.EXPECTED_MOBILE_SAMPLING
    assert plan['test_read'] is False and plan['development_holdout_read'] is False


def test_spatial_coefficients_can_diverge_after_one_objective_update_without_source_mutation():
    oracle = _oracle_fixture()
    source = _source()
    source_before = {name: value.detach().clone() for name, value in source.state_dict().items()}
    target, _ = expand_spatial_pool(source)
    assert spatial.horizontal_spread(target) == 0.

    target.train()
    loss, _ = spatial.objective(target, oracle.batch())
    loss.backward()
    gradient = target.style[0].weight.grad.detach().reshape(384, 128, 4, 4, 4)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[..., 1:] - gradient[..., :1]) > 0

    with torch.no_grad():
        target.style[0].weight.add_(-1e-3 * target.style[0].weight.grad)
    assert spatial.horizontal_spread(target) > 0.
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, source_before[name], rtol=0, atol=0)


def test_spatial_trainer_rejects_noncompliant_input_before_forward():
    source = _source()
    target, _ = expand_spatial_pool(source)
    with pytest.raises(ValueError, match='Spatial font model requires'):
        target(torch.zeros(1, 1, 32, 256))
    with pytest.raises(ValueError, match='Spatial font model requires'):
        target(torch.zeros(1, 64, 256))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64], ids=['float32', 'float64'])
def test_verify_transfer_proves_both_precision_paths_and_preserves_source(monkeypatch, dtype):
    source = _source().to(dtype=dtype)
    target, transfer = expand_spatial_pool(source)
    source_before = {name: value.detach().clone() for name, value in source.state_dict().items()}
    source_dtypes = {name: value.dtype for name, value in source.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    monkeypatch.setattr(spatial, 'BASE_STATE_SHA', spatial.state_sha(source.state_dict()))
    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    rng = np.random.default_rng(20260918)
    datasets = {'original': {'tiles': rng.normal(size=(8, 1, 64, 256)).astype(np_dtype)}}
    mobile = {'datasets': {
        'ios': {'tiles': rng.normal(size=(8, 1, 64, 256)).astype(np_dtype)},
        'android': {'tiles': rng.normal(size=(8, 1, 64, 256)).astype(np_dtype)},
    }}

    report = spatial.verify_transfer(source, target, datasets, mobile, transfer)
    evidence = report['initial_parity']
    assert evidence['verified'] is True
    assert evidence['float64_maximum_logit_error'] <= 1e-10
    assert evidence['float64_maximum_size_error'] <= 1e-10
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert {name: value.dtype for name, value in source.state_dict().items()} == source_dtypes
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, source_before[name], rtol=0, atol=0)


def test_verify_transfer_rejects_target_weight_perturbation(monkeypatch):
    source = _source()
    target, transfer = expand_spatial_pool(source)
    monkeypatch.setattr(spatial, 'BASE_STATE_SHA', spatial.state_sha(source.state_dict()))
    rng = np.random.default_rng(20260919)
    datasets = {'original': {'tiles': rng.normal(size=(8, 1, 64, 256)).astype(np.float32)}}
    mobile = {'datasets': {
        'ios': {'tiles': rng.normal(size=(8, 1, 64, 256)).astype(np.float32)},
        'android': {'tiles': rng.normal(size=(8, 1, 64, 256)).astype(np.float32)},
    }}
    with torch.no_grad():
        target.style[0].weight[0, 0] += .01
    with pytest.raises(ValueError, match='Spatial transfer changed initial TRAIN'):
        spatial.verify_transfer(source, target, datasets, mobile, transfer)
