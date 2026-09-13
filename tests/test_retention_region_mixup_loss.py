"""Verify probability aggregation, unknown masking and manifold-mixup gradients."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from retention_adapter_network import RetentionAdapterClassifier
from retention_region_mixup_loss import region_mixup_loss
from test_unified_retention import FAMILIES


def fixture():
    torch.set_num_threads(2)
    torch.manual_seed(1406)
    model = RetentionAdapterClassifier().double().train()
    counts = [1 + i % 3 for i in range(96)]
    indices = []
    offset = 0
    for count in counts:
        indices.append(offset)
        offset += count
    features = torch.randn(offset, 128, dtype=torch.float64)
    targets = torch.arange(96) % 25
    rows = [{'target': int(target), 'family': FAMILIES[int(target)], 'split': 'train',
             'domain': 'ios' if i % 2 == 0 else 'android', 'view': 'native',
             'native_font_verified': True} for i, target in enumerate(targets)]
    # Frozen base is deliberately correct for every region, including unknown.
    base = torch.full((offset, 25), -1., dtype=torch.float64)
    for start, count, target in zip(indices, counts, targets):
        base[start:start + count, target] = 2.
    order = list(reversed(range(96)))
    return dict(model=model, features=features, base_logits=base, tile_counts=counts,
                targets=targets, rows=rows, families=FAMILIES, mix_indices=indices,
                permutation=order, lam=.3)


def explicit(args):
    model = args['model'];features = args['features'];base = args['base_logits']
    targets = args['targets'];counts = args['tile_counts'];order = args['permutation'];lam = args['lam']
    p = torch.stack([part.mean(0) for part in (base + model.residual_family_head(features)).softmax(1).split(counts)])
    bp = torch.stack([part.mean(0) for part in base.softmax(1).split(counts)])
    onehot = torch.nn.functional.one_hot(targets, 25).double()
    weights = torch.tensor([2. if r['family'] == '__unknown__' or r['domain'] == 'ios'
                            and r['family'] in ('PingFang', 'SF Pro', 'Helvetica') else 1.
                            for r in args['rows']], dtype=torch.float64)
    ce = (-((onehot * .97 + .03 / 25) * p.log()).sum(1) * weights).sum() / 96
    mask = torch.tensor([r['family'] not in ('PingFang', 'SF Pro', 'Helvetica', '__unknown__')
                         for r in args['rows']]) & (bp.argmax(1) == targets)
    kl = (bp[mask] * (bp[mask].log() - p[mask].log())).sum(1).mean()
    selected = features[args['mix_indices']];mixed = lam * selected + (1 - lam) * selected[order]
    logits = model.family_head(mixed) + model.residual_family_head(mixed)
    mix = -((lam * onehot + (1 - lam) * onehot[order]) * logits.log_softmax(1)).sum(1).mean()
    return p, ce, kl, mix, weights, mask


def test_region_probability_loss_and_every_residual_gradient_match_explicit_reference():
    args = fixture()
    with torch.no_grad():
        args['model'].residual_family_head[2].weight.normal_(0, .04)
    reference_model = deepcopy(args['model'])
    actual = region_mixup_loss(**args)
    p, ce, kl, mix, weights, mask = explicit({**args, 'model': reference_model})
    for a, b in [(actual['region_probabilities'], p), (actual['region_ce'], ce),
                 (actual['base_kl'], kl), (actual['mixup_ce'], mix)]:
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    assert actual['ce_weights'] == weights.tolist() and torch.equal(actual['kl_mask'], mask)
    actual['loss'].backward();(ce + kl + .5 * mix).backward()
    for (name, param), (_, other) in zip(args['model'].named_parameters(), reference_model.named_parameters()):
        if name.startswith('residual_family_head.'):
            torch.testing.assert_close(param.grad, other.grad, rtol=1e-10, atol=1e-12)
            assert param.grad.abs().sum() > 0
        else:
            assert param.grad is None and other.grad is None
    assert args['features'].grad is None and args['base_logits'].grad is None


def test_mean_tile_probabilities_are_not_softmax_of_mean_tile_logits():
    args = fixture();start = args['mix_indices'][1]
    args['base_logits'][start:start + 2].fill_(-30)
    args['base_logits'][start, :2] = torch.tensor([8., 0.])
    args['base_logits'][start + 1, :2] = torch.tensor([0., 2.])
    result = region_mixup_loss(**args)
    block = args['base_logits'][start:start + 2]
    torch.testing.assert_close(result['region_probabilities'][1], block.softmax(1).mean(0))
    assert not torch.allclose(result['region_probabilities'][1], block.mean(0).softmax(0), atol=.1)


@pytest.mark.parametrize('lam', [0., 1.])
def test_mixup_endpoints_are_ordinary_unsmoothed_ce(lam):
    args = fixture();args['lam'] = lam
    result = region_mixup_loss(**args)
    features = args['features'][args['mix_indices']]
    expected = torch.nn.functional.cross_entropy(args['model'].classify(features), args['targets'])
    torch.testing.assert_close(result['mixup_ce'], expected, rtol=1e-12, atol=1e-12)


def test_unknown_and_core_are_excluded_from_kl_even_when_base_correct():
    args = fixture();result = region_mixup_loss(**args)
    for i, row in enumerate(args['rows']):
        assert bool(result['kl_mask'][i]) == (row['family'] not in ('__unknown__', 'PingFang', 'SF Pro', 'Helvetica'))
        if row['family'] == '__unknown__':
            assert result['ce_weights'][i] == 2.


def test_empty_kl_mask_is_differentiable_zero():
    args = fixture();args['targets'].fill_(24)
    for row in args['rows']:
        row.update(target=24, family='__unknown__')
    result = region_mixup_loss(**args)
    assert not result['kl_mask'].any() and result['base_kl'].item() == 0
    result['base_kl'].backward(retain_graph=True)
    assert not args['model'].residual_family_head[2].weight.grad.any()
    args['model'].zero_grad();result['loss'].backward()
    assert args['model'].residual_family_head[2].weight.grad.abs().sum() > 0


def test_integer_tensor_indices_and_counts_are_accepted():
    args = fixture();expected = region_mixup_loss(**args)
    for key in ('tile_counts', 'mix_indices', 'permutation'):
        args[key] = torch.tensor(args[key], dtype=torch.int64)
    actual = region_mixup_loss(**args)
    torch.testing.assert_close(actual['loss'], expected['loss'], rtol=0, atol=0)


@pytest.mark.parametrize('fault', ['cal_row', 'unverified', 'identity', 'target_range', 'target_dtype',
    'feature_grad', 'base_grad', 'nan_feature', 'nan_base', 'count_zero', 'count_nine', 'count_total',
    'mixed_region', 'permutation_duplicate', 'permutation_range', 'permutation_float', 'lam_nan',
    'lam_negative', 'lam_high', 'lam_bool', 'base_trainable', 'base_training', 'dtype_mismatch'])
def test_invalid_or_nontrain_inputs_are_rejected(fault):
    args = fixture()
    if fault == 'cal_row':args['rows'][0]['split'] = 'calibration'
    elif fault == 'unverified':args['rows'][0]['native_font_verified'] = False
    elif fault == 'identity':args['rows'][0]['family'] = 'PingFang'
    elif fault == 'target_range':args['targets'][0] = 25
    elif fault == 'target_dtype':args['targets'] = args['targets'].float()
    elif fault == 'feature_grad':args['features'].requires_grad_(True)
    elif fault == 'base_grad':args['base_logits'].requires_grad_(True)
    elif fault == 'nan_feature':args['features'][0, 0] = float('nan')
    elif fault == 'nan_base':args['base_logits'][0, 0] = float('nan')
    elif fault == 'count_zero':args['tile_counts'][0] = 0
    elif fault == 'count_nine':args['tile_counts'][0] = 9
    elif fault == 'count_total':args['features'] = args['features'][:-1]
    elif fault == 'mixed_region':args['mix_indices'][0] = 1
    elif fault == 'permutation_duplicate':args['permutation'][0] = args['permutation'][1]
    elif fault == 'permutation_range':args['permutation'][0] = 96
    elif fault == 'permutation_float':args['permutation'][0] = 95.
    elif fault == 'lam_nan':args['lam'] = float('nan')
    elif fault == 'lam_negative':args['lam'] = -.1
    elif fault == 'lam_high':args['lam'] = 1.1
    elif fault == 'lam_bool':args['lam'] = True
    elif fault == 'base_trainable':args['model'].family_head.requires_grad_(True)
    elif fault == 'base_training':args['model'].family_head.train()
    else:args['features'] = args['features'].float()
    with pytest.raises(ValueError):region_mixup_loss(**args)
