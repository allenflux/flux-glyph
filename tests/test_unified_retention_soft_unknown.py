"""Explicit soft unknown targets change training probabilities, never native labels."""
from copy import deepcopy
import numpy as np
import pytest
from training import train_unified_retention_soft_unknown as m
from test_unified_retention import FAMILIES
from test_unified_retention_full_supplement import loss_batch,samples


def test_unknown_half_mass_and_named_uniform_targets_leave_labels_unchanged():
    torch,rows,targets,_,_,_,_=loss_batch();before=targets.clone();original=deepcopy(rows)
    q=m.supervision_targets(targets,FAMILIES,torch.float64);unknown=targets==24
    torch.testing.assert_close(q.sum(1),torch.ones(96,dtype=torch.float64))
    assert torch.all(q[unknown,24]==.5) and torch.all(q[unknown,:24]==.5/24)
    known=torch.nn.functional.one_hot(targets[~unknown],25).double()*.97+.03/25
    torch.testing.assert_close(q[~unknown],known)
    assert torch.equal(targets,before) and rows==original
    assert m.SOFT_UNKNOWN_SUPERVISION['labels_unchanged'] is True


def test_soft_unknown_supervision_and_known_only_kl_exact_value_and_gradients():
    torch,rows,targets,logits,ratios,sizes,teacher=loss_batch()
    total,ce,size,kl,mask=m.full_losses(logits,ratios,targets,sizes,teacher,rows,FAMILIES)
    weights=torch.tensor(m.core.native_core_weights(rows,targets.tolist(),FAMILIES),dtype=torch.float64)
    q=torch.nn.functional.one_hot(targets,25).double()*.97+.03/25
    q[targets==24]=.5/24;q[targets==24,24]=.5
    expected_mask=torch.tensor([r['family'] not in (*m.core.CORE_FAMILIES,m.core.UNKNOWN) for r in rows])
    assert torch.equal(mask,expected_mask) and not mask[targets==24].any()
    torch.testing.assert_close(ce,(-(q*logits.log_softmax(1)).sum(1)*weights).sum()/96)
    torch.testing.assert_close(total,ce+.2*size+2*kl)
    total.backward();gradient=(logits.detach().softmax(1)-q)*weights[:,None]/96
    gradient[mask]+=2*(logits.detach()[mask].softmax(1)-teacher[mask].softmax(1))/mask.sum()
    torch.testing.assert_close(logits.grad,gradient)
    torch.testing.assert_close(ratios.grad,.2*torch.clamp(ratios.detach()-sizes,-1,1)/96)
    assert teacher.grad is None and sizes.grad is None


def test_unknown_teacher_outputs_cannot_pull_the_new_unknown_objective():
    torch,rows,targets,logits,ratios,sizes,teacher=loss_batch()
    before=m.full_losses(logits,ratios,targets,sizes,teacher,rows,FAMILIES)
    changed=teacher.clone();changed[targets==24]=-100;changed[targets==24,4]=100
    after=m.full_losses(logits,ratios,targets,sizes,changed,rows,FAMILIES)
    for left,right in zip(before,after):torch.testing.assert_close(left,right)


def test_empty_known_teacher_mask_is_differentiable_and_leaves_soft_ce_and_size():
    torch,rows,targets,logits,ratios,sizes,teacher=loss_batch()
    teacher.fill_(-2);teacher[torch.arange(96),(targets+1)%25]=2
    teacher[targets==24,24]=10  # Correct unknown teacher still never enters KL.
    total,ce,size,kl,mask=m.full_losses(logits,ratios,targets,sizes,teacher,rows,FAMILIES)
    assert not mask.any() and kl.item()==0 and kl.requires_grad
    torch.testing.assert_close(total,ce+.2*size);total.backward()
    assert logits.grad.abs().sum()>0 and ratios.grad.abs().sum()>0


@pytest.mark.parametrize('fault',['teacher_grad','teacher_nan','size_grad','label_dtype','bad_label','family_order','holdout','native_false'])
def test_invalid_supervision_fails_closed(fault):
    torch,rows,targets,logits,ratios,sizes,teacher=loss_batch();families=FAMILIES.copy()
    if fault=='teacher_grad':teacher.requires_grad_(True)
    elif fault=='teacher_nan':teacher[0,0]=float('nan')
    elif fault=='size_grad':sizes.requires_grad_(True)
    elif fault=='label_dtype':targets=targets.double()
    elif fault=='bad_label':targets[0]=25
    elif fault=='family_order':families[0],families[1]=families[1],families[0]
    elif fault=='holdout':rows[0]['split']='development_holdout'
    else:rows[0]['native_font_verified']=False
    with pytest.raises(ValueError):m.full_losses(logits,ratios,targets,sizes,teacher,rows,families)


def counts():
    _,_,_,rows=samples();counter=m.FullSupplementCounts(FAMILIES)
    mask=[r['family'] not in (*m.core.CORE_FAMILIES,m.core.UNKNOWN) for r in rows]
    counter.update(rows,mask);return counter.report(),rows,mask


def test_unknown_training_keeps_eight_new_samples_but_zero_teacher_rows():
    report,_,_=counts();m.validate_training_counts(report,FAMILIES,1)
    assert report['family_rows'][m.core.UNKNOWN]==16 and report['eligible_family_rows'].get(m.core.UNKNOWN,0)==0
    assert report['supplement_rows']==8 and report['soft_unknown_supervision']==m.SOFT_UNKNOWN_SUPERVISION
    assert not report['second_model_resident'] and not report['teacher_cache_deployed']


def test_unknown_teacher_mask_rejected_during_collection_and_rehashed_reporting():
    report,rows,mask=counts();mask[next(i for i,r in enumerate(rows) if r['family']==m.core.UNKNOWN)]=True
    with pytest.raises(ValueError):m.FullSupplementCounts(FAMILIES).update(rows,mask)
    next(r for r in report['source_target_rows'] if r['target_family']==m.core.UNKNOWN)['eligible_rows']=1
    with pytest.raises(ValueError):m.validate_training_counts(report,FAMILIES,1)


def test_target_distribution_metadata_cannot_be_changed_without_rejection():
    report,_,_=counts();report=deepcopy(report);report['soft_unknown_supervision']['unknown_target_probability']=.8
    with pytest.raises(ValueError):m.validate_training_counts(report,FAMILIES,1)


def test_plain_image_cnn_all_four_groups_receive_soft_supervised_updates():
    torch=pytest.importorskip('torch');torch.set_num_threads(4);torch.manual_seed(9)
    from training.region_network import RegionFontClassifier
    parent=RegionFontClassifier(25);model=RegionFontClassifier(25)
    m.initialize(model,{'families':FAMILIES,'architecture':m.ARCHITECTURE,'state_dict':parent.state_dict()},FAMILIES)
    before=m.parameter_groups(model.state_dict());_,_,_,rows=samples()
    images=torch.rand(96,1,64,256);targets=torch.tensor([r['target'] for r in rows]);logits,ratios=model(images)
    teacher=logits.detach().clone();sizes=ratios.detach()+.3
    loss,*_=m.full_losses(logits,ratios,targets,sizes,teacher,rows,FAMILIES);loss.backward()
    for group in before:assert sum(float(p.grad.abs().sum()) for name,p in model.named_parameters() if name.startswith(group+'.') and p.grad is not None)>0
    torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=1e-4).step()
    after=m.parameter_groups(model.state_dict());assert all(before[k]!=after[k] for k in before)
    assert not any(k.startswith('residual') for k in model.state_dict())


def test_trial_has_fixed_new_seed_and_original_inference_acceptance():
    args=m.parser().parse_args([])
    assert args.seed==2026091409 and args.steps==3000 and m.EVAL_EVERY==500
    assert args.output.name=='run-soft-unknown-v1' and args.device=='mps'
    assert m.FIXED_RUNTIME=={'temperature':1.,'gates':{'min_score':.7,'min_margin':.01,'min_patch_agreement':2/3},'max_size_relative_spread':.2}
    assert m.OBJECTIVE['teacher_kl_weight']==2 and m.OBJECTIVE['unknown_ce_weight']==1
