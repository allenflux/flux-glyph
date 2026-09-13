"""Balanced true unknown sources and binary mass supervision preserve font competition."""
from copy import deepcopy
from collections import Counter
import numpy as np
import pytest
from training import train_unified_retention_source_balanced as m
from test_unified_retention import FAMILIES,sampler_data
from test_retention_supplement_sampler import supplemental_rows
from test_unified_retention_full_supplement import loss_batch,samples

OLD_SOURCES=['Times','Courier','Apple Chancery','Bradley Hand','Noteworthy','Lato','Liu Jian Mao Cao','Smiley Sans','Open Sans']


def balanced_sampler():
    original,pools=sampler_data();known=[r for r in original if r['family']!=m.core.UNKNOWN]
    template=[r for r in original if r['family']==m.core.UNKNOWN and r['source_font_family']=='Novel A']
    rows=deepcopy(known)
    for source in OLD_SOURCES:
        for example in template:
            row=deepcopy(example);row.update(source_font_family=source,font_face=source+'-'+row['font_face'].split('-')[-1],
                source_id='old-'+str(len(rows)),tile_start=len(rows),log_em_ratio=.25)
            rows.append(row)
    return m.SourceBalancedSampler(rows,FAMILIES,m.SEED,pools,supplemental_rows())


def test_binary_unknown_gradient_only_controls_total_named_mass():
    torch=pytest.importorskip('torch');torch.set_num_threads(4)
    named=torch.arange(24,dtype=torch.float64).reshape(1,24)/3
    logits=torch.cat((named,torch.logsumexp(named,dim=1,keepdim=True)),1).detach().requires_grad_(True)
    loss=m.unknown_binary_loss(logits,FAMILIES);loss.sum().backward()
    torch.testing.assert_close(loss,torch.full((1,),np.log(2),dtype=torch.float64))
    torch.testing.assert_close(logits.grad,torch.zeros_like(logits),atol=1e-14,rtol=0)
    # The named classes remain highly nonuniform at this minimum; there is no
    # uniform named-distribution target in the binary loss.
    assert float(logits.detach()[:,:24].softmax(1).max())>.25


def test_known_ce_binary_unknown_size_and_known_only_kl_exact_gradients():
    torch,rows,targets,logits,ratios,sizes,teacher=loss_batch()
    total,ce,size,kl,mask=m.full_losses(logits,ratios,targets,sizes,teacher,rows,FAMILIES)
    weights=torch.tensor(m.core.native_core_weights(rows,targets.tolist(),FAMILIES),dtype=torch.float64)
    unknown=targets==24;known=~unknown
    smooth=torch.nn.functional.one_hot(targets,25).double()*.97+.03/25
    per=(-smooth*logits.log_softmax(1)).sum(1)
    odds=logits[unknown,24]-torch.logsumexp(logits[unknown,:24],1)
    per[unknown]=torch.nn.functional.binary_cross_entropy_with_logits(odds,torch.full_like(odds,.5),reduction='none')
    torch.testing.assert_close(ce,(per*weights).sum()/96)
    torch.testing.assert_close(total,ce+.2*size+2*kl)
    assert not mask[unknown].any() and torch.all(weights[unknown]==1)
    total.backward();p=logits.detach().softmax(1);gradient=p-smooth
    gradient[unknown,24]=p[unknown,24]-.5
    gradient[unknown,:24]=(.5-p[unknown,24,None])*logits.detach()[unknown,:24].softmax(1)
    gradient*=weights[:,None]/96
    gradient[mask]+=2*(p[mask]-teacher[mask].softmax(1))/mask.sum()
    torch.testing.assert_close(logits.grad,gradient)
    torch.testing.assert_close(ratios.grad,.2*torch.clamp(ratios.detach()-sizes,-1,1)/96)
    assert teacher.grad is None and sizes.grad is None


def test_extreme_binary_logits_are_finite_and_shift_invariant():
    torch=pytest.importorskip('torch')
    logits=torch.full((2,25),-1000.,dtype=torch.float64,requires_grad=True)
    with torch.no_grad():logits[0,24]=1000.;logits[1,3]=1000.
    loss=m.unknown_binary_loss(logits,FAMILIES)
    torch.testing.assert_close(loss,m.unknown_binary_loss(logits+10000,FAMILIES))
    loss.sum().backward();assert torch.isfinite(loss).all() and torch.isfinite(logits.grad).all()


def test_correct_unknown_teacher_still_excluded_and_empty_kl_differentiable():
    torch,rows,targets,logits,ratios,sizes,teacher=loss_batch()
    teacher.fill_(-2.);teacher[torch.arange(96),(targets+1)%25]=2.;teacher[targets==24,24]=20
    total,ce,size,kl,mask=m.full_losses(logits,ratios,targets,sizes,teacher,rows,FAMILIES)
    assert not mask.any() and kl.item()==0 and kl.requires_grad
    torch.testing.assert_close(total,ce+.2*size);total.backward();assert logits.grad.abs().sum()>0


@pytest.mark.parametrize('fault',['teacher_grad','teacher_nan','size_grad','target_dtype','target_range','holdout','label'])
def test_invalid_supervision_rejected(fault):
    torch,rows,targets,logits,ratios,sizes,teacher=loss_batch()
    if fault=='teacher_grad':teacher.requires_grad_(True)
    elif fault=='teacher_nan':teacher[0,0]=float('nan')
    elif fault=='size_grad':sizes.requires_grad_(True)
    elif fault=='target_dtype':targets=targets.double()
    elif fault=='target_range':targets[0]=25
    elif fault=='holdout':rows[0]['split']='development_holdout'
    else:rows[0]['family']='fake'
    with pytest.raises(ValueError):m.full_losses(logits,ratios,targets,sizes,teacher,rows,FAMILIES)


def test_eleven_sources_use_real_dynamic_new_counts_and_exact_long_run_quota():
    sampler=balanced_sampler();counter=m.FullSupplementCounts(FAMILIES,sampler.unknown_source_order);new_counts=[]
    for _ in range(11):
        rows=sampler.batch();mask=[r['family'] not in (*m.core.CORE_FAMILIES,m.core.UNKNOWN) for r in rows]
        counter.update(rows,mask);new_counts.append(sum(r.get(m.SUPPLEMENT_MARKER,False) for r in rows))
    result=m.validate_training_counts(counter.report(),FAMILIES,11)
    assert result['unknown_source_rows']==dict.fromkeys(sampler.unknown_source_order,16)
    assert result['supplement_rows']==32 and set(new_counts)=={2,3,4}
    assert result['unknown_source_rows']==sampler.report()['unknown_source_rows']
    quota=m.expected_unknown_counts(sampler.unknown_source_order,3000*16)
    assert set(quota.values())=={4363,4364} and sum(quota.values())==48000
    assert [quota[source] for source in m.NEW_SOURCES]==[4363,4363]


def counter_fixture():
    sampler=balanced_sampler();rows=sampler.batch();counter=m.FullSupplementCounts(FAMILIES,sampler.unknown_source_order)
    mask=[r['family'] not in (*m.core.CORE_FAMILIES,m.core.UNKNOWN) for r in rows];counter.update(rows,mask)
    return sampler,rows,mask,counter.report()


@pytest.mark.parametrize('fault',['source_order','old_quota','new_quota','unknown_kl','binary_target','heldout'])
def test_report_cannot_restore_old_eight_new_quota_or_change_objective(fault):
    _,_,_,report=counter_fixture();report=deepcopy(report)
    if fault=='source_order':report['unknown_source_order'].reverse()
    elif fault=='old_quota':report['unknown_source_rows'][OLD_SOURCES[0]]+=1
    elif fault=='new_quota':report['supplement_source_rows']=dict.fromkeys(m.NEW_SOURCES,4)
    elif fault=='unknown_kl':next(r for r in report['source_target_rows'] if r['target_family']==m.core.UNKNOWN)['eligible_rows']=1
    elif fault=='binary_target':report['unknown_binary_supervision']['unknown_probability_target']=.8
    else:report['test_read']=True
    with pytest.raises(ValueError):m.validate_training_counts(report,FAMILIES,1)


def test_rows_cannot_lie_about_new_marker_or_replay_unknown_source_twice():
    sampler,rows,mask,_=counter_fixture()
    for fault in ('marker','source'):
        changed=deepcopy(rows);unknown=next(r for r in changed if r.get(m.SUPPLEMENT_MARKER))
        if fault=='marker':unknown[m.SUPPLEMENT_MARKER]=False
        else:unknown['source_font_family']=OLD_SOURCES[0]
        with pytest.raises(ValueError):m.FullSupplementCounts(FAMILIES,sampler.unknown_source_order).update(changed,mask)


def test_full_image_model_all_four_groups_still_update_with_binary_unknown_targets():
    torch=pytest.importorskip('torch');torch.set_num_threads(4);torch.manual_seed(10)
    from training.region_network import RegionFontClassifier
    parent=RegionFontClassifier(25);model=RegionFontClassifier(25)
    m.initialize(model,{'families':FAMILIES,'architecture':m.ARCHITECTURE,'state_dict':parent.state_dict()},FAMILIES)
    before=m.parameter_groups(model.state_dict());rows=balanced_sampler().batch()
    images=torch.rand(96,1,64,256);targets=torch.tensor([r['target'] for r in rows]);logits,ratios=model(images)
    teacher=logits.detach().clone();sizes=ratios.detach()+.3
    loss,*_=m.full_losses(logits,ratios,targets,sizes,teacher,rows,FAMILIES);loss.backward()
    for group in before:assert sum(float(p.grad.abs().sum()) for name,p in model.named_parameters() if name.startswith(group+'.') and p.grad is not None)>0
    torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=1e-4).step()
    after=m.parameter_groups(model.state_dict());assert all(before[k]!=after[k] for k in before)
    assert not any(k.startswith('residual') for k in model.state_dict())


def test_new_two_change_trial_keeps_runtime_and_optimizer_fixed():
    args=m.parser().parse_args([])
    assert args.seed==2026091410 and args.steps==3000 and m.EVAL_EVERY==500
    assert args.output.name=='run-source-balanced-v1' and len(m.DESIGN_CHANGES)==2
    assert m.UNKNOWN_BINARY_SUPERVISION['named_conditional_distribution_target'] is None
    assert m.FIXED_RUNTIME=={'temperature':1.,'gates':{'min_score':.7,'min_margin':.01,'min_patch_agreement':2/3},'max_size_relative_spread':.2}
