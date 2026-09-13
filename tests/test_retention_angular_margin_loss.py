"""Focused CPU checks for the known-only angular-margin auxiliary."""
import math

import pytest

torch = pytest.importorskip('torch')

from training.prepare_unified_regions import FAMILIES
from training.retention_angular_margin_loss import (
    ANGULAR_MARGIN, COEFFICIENT, MARGIN_RADIANS, SCALE, angular_margin_loss, monotone_angular_target,
)


def rows(targets):
    result=[]
    for index,target in enumerate(targets):
        family=FAMILIES[target]
        result.append({'family':family,'target':target,'split':'train','native_font_verified':True,
            'domain':'ios' if index == 0 else 'android','view':'native' if index == 0 else 'half',
            'source_font_family':family})
    return result


def test_public_objective_spec_matches_executed_defaults():
    assert ANGULAR_MARGIN['weight'] == COEFFICIENT == .10
    assert ANGULAR_MARGIN['scale'] == SCALE == 16.
    assert ANGULAR_MARGIN['angle_radians'] == MARGIN_RADIANS == .10
    assert ANGULAR_MARGIN['denominator'] == 96 and ANGULAR_MARGIN['monotone_guard'] is True


def test_hand_computed_orthogonal_known_row_and_native_weight():
    target=FAMILIES.index('PingFang');targets=torch.full((96,),24,dtype=torch.int64);targets[0]=target
    features=torch.zeros(96,256,dtype=torch.float64);features[:,255]=1.
    weight=torch.zeros(25,256,dtype=torch.float64);weight[torch.arange(25),torch.arange(25)]=1.
    actual=angular_margin_loss(features,weight,targets,rows(targets.tolist()),FAMILIES)
    target_logit=-SCALE*math.sin(MARGIN_RADIANS)*math.sqrt(1+1e-12)
    expected=COEFFICIENT*2/96*(math.log(math.exp(target_logit)+24)-target_logit)
    torch.testing.assert_close(actual,torch.tensor(expected,dtype=torch.float64),rtol=1e-12,atol=1e-12)


def test_unknown_rows_have_zero_feature_gradient_while_known_and_prototypes_train():
    generator=torch.Generator().manual_seed(1414)
    features=torch.randn(96,256,generator=generator,dtype=torch.float64,requires_grad=True)
    weight=torch.randn(25,256,generator=generator,dtype=torch.float64,requires_grad=True)
    targets=torch.full((96,),24,dtype=torch.int64);targets[:3]=torch.tensor([4,10,11])
    loss=angular_margin_loss(features,weight,targets,rows(targets.tolist()),FAMILIES);loss.backward()
    assert loss.ndim == 0 and float(loss.detach())>0
    assert bool((features.grad[:3].abs().sum(1)>0).all())
    assert torch.count_nonzero(features.grad[3:]) == 0
    assert bool(torch.isfinite(weight.grad).all()) and float(weight.grad.abs().sum())>0


def test_all_unknown_is_differentiable_exact_zero():
    features=torch.zeros(96,256,dtype=torch.float32,requires_grad=True)
    weight=torch.zeros(25,256,dtype=torch.float32,requires_grad=True)
    targets=torch.full((96,),24,dtype=torch.int64)
    loss=angular_margin_loss(features,weight,targets,rows(targets.tolist()),FAMILIES);loss.backward()
    assert float(loss.detach())==0. and torch.count_nonzero(features.grad)==torch.count_nonzero(weight.grad)==0


def test_monotone_guard_has_no_raw_cosine_reversal_near_pi():
    threshold=-math.cos(MARGIN_RADIANS)
    values=torch.tensor([-1.,threshold-.002,threshold,threshold+.002,0.,1.],dtype=torch.float64)
    corrected=monotone_angular_target(values)
    assert bool((corrected[1:]-corrected[:-1]>=0).all())
    assert corrected[0] < corrected[2] < corrected[-1]
    raw_at_pi=math.cos(math.pi+MARGIN_RADIANS)
    assert raw_at_pi > float(corrected[0])
    endpoints=torch.tensor([-1.,1.],dtype=torch.float64,requires_grad=True)
    monotone_angular_target(endpoints).sum().backward()
    assert bool(torch.isfinite(endpoints.grad).all())


@pytest.mark.parametrize('fault',['nan_features','nan_weight','zero_feature','zero_weight','shape',
    'target_dtype','target_range','row_provenance','scale','margin','coefficient'])
def test_invalid_tensor_norm_and_train_contracts_are_rejected(fault):
    features=torch.randn(96,256);weight=torch.randn(25,256);targets=torch.zeros(96,dtype=torch.int64)
    batch=rows(targets.tolist());kwargs={}
    if fault=='nan_features':features[0,0]=float('nan')
    elif fault=='nan_weight':weight[0,0]=float('nan')
    elif fault=='zero_feature':features[0].zero_()
    elif fault=='zero_weight':weight[0].zero_()
    elif fault=='shape':features=features[:95]
    elif fault=='target_dtype':targets=targets.float()
    elif fault=='target_range':targets[0]=25
    elif fault=='row_provenance':batch[0]['split']='calibration'
    elif fault=='scale':kwargs['scale']=0.
    elif fault=='margin':kwargs['margin']=math.pi/2
    else:kwargs['coefficient']=-.1
    with pytest.raises(ValueError):angular_margin_loss(features,weight,targets,batch,FAMILIES,**kwargs)


def test_combined_objective_reaches_all_wide_parameter_groups():
    from training.wide_region_network import WideRegionFontClassifier
    torch.manual_seed(17);torch.set_num_threads(2)
    model=WideRegionFontClassifier(25).double()
    images=torch.rand(96,1,64,256,dtype=torch.float64)
    features=model.style(model.pool(model.trunk(images)))
    logits=model.family_head(features);ratios=model.size_head(features).squeeze(-1)
    targets=torch.arange(96,dtype=torch.int64)%25
    batch=rows(targets.tolist())
    base=torch.nn.functional.cross_entropy(logits,targets)
    size=torch.nn.functional.smooth_l1_loss(ratios,torch.linspace(-.2,.3,96,dtype=torch.float64))
    auxiliary=angular_margin_loss(features,model.family_head.weight,targets,batch,FAMILIES)
    (base+.2*size+auxiliary).backward()
    groups={'trunk':model.trunk,'style':model.style,'family':model.family_head,'size':model.size_head}
    assert all(any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum())>0
                   for p in group.parameters()) for group in groups.values())
