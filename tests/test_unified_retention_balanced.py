"""Class-prior correction changes supervised gradients, never runtime gates."""
from collections import Counter
from copy import deepcopy
import math
import numpy as np
import pytest
from training import train_unified_retention_balanced as module
from test_unified_retention import FAMILIES, sampler_data


def test_prior_weights_restore_original_known_and_unknown_mass():
    result=module.class_prior_weighting(FAMILIES)
    counts=result['sampled_counts'];weights=result['class_weights']
    assert counts['PingFang']==19 and counts['SF Pro']==counts['Helvetica']==7
    assert counts[module.UNKNOWN]==16
    assert all(counts[f]==3 for f in module.IOS_ANCHOR_FAMILIES if f not in ('PingFang','SF Pro','Helvetica'))
    assert all(counts[f]==2 for f in FAMILIES if f not in module.IOS_ANCHOR_FAMILIES and f!=module.UNKNOWN)
    assert sum(counts.values())==96
    assert sum(counts[f]*weights[f] for f in FAMILIES)==pytest.approx(96)
    for family in FAMILIES:
        expected=19 if family==module.UNKNOWN else 77/24
        assert counts[family]*weights[family]==pytest.approx(expected)
    assert module.class_prior_weighting(FAMILIES[::-1])==result


def test_actual_frozen_sampler_satisfies_exact_weighted_batch_composition():
    rows,pools=sampler_data();sampler=module.RetentionSampler(rows,FAMILIES,2026091403,pools)
    result=module.class_prior_weighting(FAMILIES)
    for _ in range(20):
        batch=sampler.batch()
        assert dict(Counter(row['family'] for row in batch))==result['sampled_counts']
        assert math.isclose(sum(result['class_weights'][r['family']] for r in batch),96)
    assert sampler.report()['family_rows']=={family:count*20 for family,count in result['sampled_counts'].items()}


@pytest.mark.parametrize('families',[FAMILIES[:-1],FAMILIES[:-1]+['Other'],FAMILIES+['Extra']])
def test_changed_registry_cannot_reuse_frozen_prior(families):
    with pytest.raises(ValueError):module.class_prior_weighting(families)


def tensor_batch():
    torch=pytest.importorskip('torch');torch.set_num_threads(4);torch.manual_seed(10)
    weighting=module.class_prior_weighting(FAMILIES)
    targets=torch.tensor([index for index,family in enumerate(FAMILIES)
                          for _ in range(weighting['sampled_counts'][family])])
    logits=torch.randn(96,25,dtype=torch.float64,requires_grad=True)
    ratios=torch.linspace(-1.5,1.5,96,dtype=torch.float64,requires_grad=True)
    sizes=torch.full((96,),.1,dtype=torch.float64)
    return torch,logits,ratios,targets,sizes,weighting


def test_weighted_ce_and_size_values_and_gradients_match_explicit_prior():
    torch,logits,ratios,targets,sizes,weighting=tensor_batch()
    before=logits.detach().clone()
    total,family_loss,size_loss=module.weighted_losses(logits,ratios,targets,sizes,FAMILIES,weighting)
    weights=torch.tensor([weighting['class_weights'][FAMILIES[t]] for t in targets],dtype=torch.float64)
    smooth=torch.nn.functional.one_hot(targets,25).double()*.97+.03/25
    expected_family=(-(smooth*logits.log_softmax(1)).sum(1)*weights).sum()/96
    residual=ratios-sizes
    expected_size=(torch.where(residual.abs()<1,.5*residual.square(),residual.abs()-.5)*weights).sum()/96
    torch.testing.assert_close(family_loss,expected_family)
    torch.testing.assert_close(size_loss,expected_size)
    torch.testing.assert_close(total,expected_family+.2*expected_size)
    total.backward()
    torch.testing.assert_close(logits.grad,(logits.detach().softmax(1)-smooth)*weights[:,None]/96)
    torch.testing.assert_close(ratios.grad,.2*residual.detach().clamp(-1,1)*weights/96)
    assert torch.equal(logits.detach(),before)
    assert bool(torch.isfinite(logits.grad).all()) and bool((logits.grad.abs().sum(0)>0).all())


def test_float32_device_normalization_and_both_heads_receive_gradients():
    torch,logits,ratios,targets,sizes,weighting=tensor_batch()
    logits=logits.detach().float().requires_grad_();ratios=ratios.detach().float().requires_grad_()
    loss,_,_=module.weighted_losses(logits,ratios,targets,sizes.float(),FAMILIES,weighting)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad.abs().sum()>0 and ratios.grad.abs().sum()>0


@pytest.mark.parametrize('fault',['counts','nan_weight','negative_weight','wrong_sum','nan_logits','shape','target_dtype'])
def test_invalid_batch_or_loss_evidence_fails_before_optimizer(fault):
    torch,logits,ratios,targets,sizes,weighting=tensor_batch();weighting=deepcopy(weighting)
    if fault=='counts':targets[0]=24
    elif fault=='nan_weight':weighting['class_weights']['PingFang']=float('nan')
    elif fault=='negative_weight':weighting['class_weights']['PingFang']=-1
    elif fault=='wrong_sum':weighting['class_weights']['PingFang']*=2
    elif fault=='nan_logits':logits=logits.detach().clone();logits[0,0]=float('nan')
    elif fault=='shape':ratios=ratios[:-1]
    else:targets=targets.float()
    with pytest.raises(ValueError):module.weighted_losses(logits,ratios,targets,sizes,FAMILIES,weighting)


def test_variant_is_explicit_and_fixed_runtime_sampling_and_selection_unchanged():
    objective=module.weighted_objective(FAMILIES)
    assert objective['variant']=='r21_class_prior_weighted'
    assert objective['class_prior_weighting']==module.class_prior_weighting(FAMILIES)
    assert objective['teacher_outputs'] is False and objective['all_parameters_trainable'] is True
    from train_unified_retention import evaluate_outputs,retention_rank,RetentionSampler,FIXED_RUNTIME
    assert module.evaluate_outputs is evaluate_outputs and module.retention_rank is retention_rank
    assert module.RetentionSampler is RetentionSampler and module.FIXED_RUNTIME is FIXED_RUNTIME
    args=module.parser().parse_args(['--data','d','--output','o','--plan','p'])
    assert args.steps==3000 and args.seed==2026091403
    assert module.FIXED_RUNTIME['gates']=={'min_score':.7,'min_margin':.01,'min_patch_agreement':2/3}
    assert module.FIXED_RUNTIME['temperature']==1. and not hasattr(args,'gates')
