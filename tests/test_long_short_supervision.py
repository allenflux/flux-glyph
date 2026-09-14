import copy
import pytest

torch = pytest.importorskip('torch')
from long_short_supervision import long_short_loss, CONTRACT
from prepare_unified_regions import FAMILIES


def batch():
    target = torch.tensor(list(range(24)) + [24] * 8)
    rows = [{'target': int(t), 'family': FAMILIES[int(t)], 'split': 'train',
             'native_font_verified': True, 'glyph_count': i % 4 + 1} for i,t in enumerate(target)]
    logits = torch.linspace(-2.,2.,800,dtype=torch.float64).reshape(32,25).requires_grad_()
    ratios = torch.linspace(-.2,.3,32,dtype=torch.float64).requires_grad_()
    return logits, ratios, target, torch.zeros(32,dtype=torch.float64), rows


def test_exact_known_long_only_weight_fixed_denominator_and_gradient():
    from torch.nn import functional as F
    logits,ratios,target,sizes,rows=batch()
    loss,font,size,mask=long_short_loss(logits,ratios,target,sizes,rows)
    expected=torch.tensor([i<24 and i%4>=2 for i in range(32)])
    assert torch.equal(mask,expected) and mask.sum()==12
    expected_font=F.cross_entropy(logits[mask],target[mask],label_smoothing=.03,reduction='sum')/32
    expected_size=F.smooth_l1_loss(ratios[mask],sizes[mask],beta=.05,reduction='sum')/32
    torch.testing.assert_close(font,expected_font);torch.testing.assert_close(size,expected_size)
    torch.testing.assert_close(loss,.125*(expected_font+.2*expected_size))
    loss.backward()
    assert torch.count_nonzero(logits.grad[~mask])==0 and torch.count_nonzero(ratios.grad[~mask])==0
    assert torch.all(logits.grad[mask].abs().sum(1)>0) and torch.all(ratios.grad[mask].abs()>0)


def test_no_eligible_rows_returns_finite_differentiable_zero():
    logits,ratios,target,sizes,rows=batch()
    for row in rows:row['glyph_count']=1
    loss,_,_,mask=long_short_loss(logits,ratios,target,sizes,rows)
    assert not mask.any() and loss==0
    loss.backward();assert torch.count_nonzero(logits.grad)==torch.count_nonzero(ratios.grad)==0


@pytest.mark.parametrize('fault',['label','family','split','unverified','nan','shape','dtype','count','bool_count'])
def test_rejects_changed_truth_or_invalid_tensors(fault):
    logits,ratios,target,sizes,rows=batch()
    if fault=='label':rows[0]['target']=1
    elif fault=='family':rows[0]['family']='__unknown__'
    elif fault=='split':rows[0]['split']='development'
    elif fault=='unverified':rows[0]['native_font_verified']=False
    elif fault=='nan':sizes[0]=float('nan')
    elif fault=='shape':sizes=sizes[:-1]
    elif fault=='dtype':target=target.float()
    elif fault=='count':rows[0]['glyph_count']=5
    else:rows[0]['glyph_count']=True
    with pytest.raises(ValueError):long_short_loss(logits,ratios,target,sizes,rows)
