"""Full-CNN replay pairs native TRAIN images with frozen same-tile core targets."""
from copy import deepcopy
import numpy as np
import pytest
from training import train_unified_retention_full_supplement as m
from test_unified_retention import FAMILIES,sampler_data
from test_retention_supplement_sampler import supplemental_rows


def samples():
    old,pools=sampler_data();new=supplemental_rows()
    for rows in (old,new):
        for r in rows:r['log_em_ratio']=.25
    sampler=m.SupplementSampler(old,FAMILIES,m.SEED,pools,new)
    return old,new,sampler,sampler.batch()


def loss_batch():
    torch=pytest.importorskip('torch');torch.set_num_threads(4);torch.manual_seed(6)
    _,_,_,rows=samples();targets=torch.tensor([r['target'] for r in rows])
    logits=torch.randn(96,25,dtype=torch.float64,requires_grad=True)
    ratios=torch.linspace(-.5,.5,96,dtype=torch.float64,requires_grad=True)
    sizes=torch.full((96,),.25,dtype=torch.float64)
    teacher=torch.full((96,25),-2.,dtype=torch.float64);teacher[torch.arange(96),targets]=2.
    return torch,rows,targets,logits,ratios,sizes,teacher


def test_weighted_ce_unknown_weight_one_size_unweighted_and_kl_two_exact_gradients():
    torch,rows,targets,logits,ratios,sizes,teacher=loss_batch()
    total,ce,size,kl,mask=m.full_losses(logits,ratios,targets,sizes,teacher,rows,FAMILIES)
    weights=torch.tensor([2. if r['domain']=='ios' and r['view']=='native' and r['family'] in m.core.CORE_FAMILIES else 1. for r in rows],dtype=torch.float64)
    smooth=torch.nn.functional.one_hot(targets,25).double()*.97+.03/25
    torch.testing.assert_close(ce,(-(smooth*logits.log_softmax(1)).sum(1)*weights).sum()/96)
    torch.testing.assert_close(size,torch.nn.functional.smooth_l1_loss(ratios,sizes))
    assert torch.equal(mask,torch.tensor([r['family'] not in m.core.CORE_FAMILIES for r in rows]))
    assert mask[targets==24].all() and torch.all(weights[targets==24]==1)
    torch.testing.assert_close(total,ce+.2*size+2*kl)
    total.backward();gradient=(logits.detach().softmax(1)-smooth)*weights[:,None]/96
    gradient[mask]+=2*(logits.detach()[mask].softmax(1)-teacher[mask].softmax(1))/mask.sum()
    torch.testing.assert_close(logits.grad,gradient)
    torch.testing.assert_close(ratios.grad,.2*torch.clamp(ratios.detach()-sizes,-1,1)/96)
    assert teacher.grad is None and sizes.grad is None


def test_wrong_teacher_empty_mask_retains_supervised_image_and_size_gradient():
    torch,rows,targets,logits,ratios,sizes,teacher=loss_batch()
    teacher.fill_(-2.);teacher[torch.arange(96),(targets+1)%25]=2.
    total,ce,size,kl,mask=m.full_losses(logits,ratios,targets,sizes,teacher,rows,FAMILIES)
    assert not mask.any() and kl.item()==0 and kl.requires_grad
    total.backward();assert logits.grad.abs().sum()>0 and ratios.grad.abs().sum()>0
    torch.testing.assert_close(total,ce+.2*size)


@pytest.mark.parametrize('fault',['teacher_gradient','teacher_nan','size_gradient','size_dtype','holdout','wrong_label'])
def test_invalid_or_nontrain_supervision_fails(fault):
    torch,rows,targets,logits,ratios,sizes,teacher=loss_batch()
    if fault=='teacher_gradient':teacher.requires_grad_(True)
    elif fault=='teacher_nan':teacher[0,0]=float('nan')
    elif fault=='size_gradient':sizes.requires_grad_(True)
    elif fault=='size_dtype':sizes=sizes.float()
    elif fault=='holdout':rows[0]['split']='development_holdout'
    else:rows[0]['family']='Wrong'
    with pytest.raises(ValueError):m.full_losses(logits,ratios,targets,sizes,teacher,rows,FAMILIES)


def image_sources():
    old,new,sampler,rows=samples();sources=[]
    for number,items in enumerate((old,new)):
        tiles=np.stack([np.full((1,64,256),(i+1)/1000+number*.5,np.float32) for i in range(len(items))])
        teacher=np.repeat(np.arange(len(items),dtype=np.float32)[:,None]+number*1000,25,axis=1)
        sources.append(m.batch_source({'partition':{'split':'train'},'rows':items,'tiles':tiles},{'base_logits':teacher}))
    return sources,rows,sampler


def test_same_tile_and_correct_old_or_new_cache_are_paired():
    (old,new),rows,sampler=image_sources();images,teacher,sizes=m.pixel_batch(rows,old,new,sampler.rng)
    for i,row in enumerate(rows):
        extra=bool(row.get(m.SUPPLEMENT_MARKER));index=row['tile_start']
        np.testing.assert_array_equal(images[i],(new if extra else old)['tiles'][index])
        np.testing.assert_array_equal(teacher[i],(new if extra else old)['teacher_logits'][index])
        assert sizes[i]==.25
    assert images.shape==(96,1,64,256)


@pytest.mark.parametrize('fault',['missing_new_marker','old_marked_new','nonbool_marker','offset','label','size','split'])
def test_cache_marker_cannot_route_or_relabel_native_truth(fault):
    (old,new),rows,sampler=image_sources();rows=deepcopy(rows)
    extra=next(r for r in rows if r.get(m.SUPPLEMENT_MARKER));original=next(r for r in rows if not r.get(m.SUPPLEMENT_MARKER))
    if fault=='missing_new_marker':extra.pop(m.SUPPLEMENT_MARKER)
    elif fault=='old_marked_new':original[m.SUPPLEMENT_MARKER]=True
    elif fault=='nonbool_marker':extra[m.SUPPLEMENT_MARKER]=1
    elif fault=='offset':extra['tile_start']+=1
    elif fault=='label':extra['target']=0
    elif fault=='size':extra['log_em_ratio']=.5
    else:extra['split']='calibration'
    with pytest.raises(ValueError):m.pixel_batch(rows,old,new,sampler.rng)


def test_full_plain_model_inherits_all25_rows_and_updates_all_four_parameter_groups():
    torch=pytest.importorskip('torch');torch.set_num_threads(4);torch.manual_seed(19)
    from training.region_network import RegionFontClassifier
    parent=RegionFontClassifier(25);model=RegionFontClassifier(25)
    checkpoint={'families':FAMILIES,'architecture':m.ARCHITECTURE,'state_dict':parent.state_dict()}
    result=m.initialize(model,checkpoint,FAMILIES)
    assert result['all_parameters_inherited'] and result['new_random_output_rows']==0
    before=m.parameter_groups(model.state_dict());assert m.state_sha(model.state_dict())==m.state_sha(parent.state_dict())
    assert not any(k.startswith('residual') for k in model.state_dict())
    _,_,_,rows=samples();images=torch.rand(96,1,64,256);targets=torch.tensor([r['target'] for r in rows])
    model.train();logits,ratios=model(images);teacher=logits.detach().clone();sizes=ratios.detach()+.3
    total,*_=m.full_losses(logits,ratios,targets,sizes,teacher,rows,FAMILIES);total.backward()
    assert all(p.requires_grad for p in model.parameters())
    for group in before:assert sum(float(p.grad.abs().sum()) for name,p in model.named_parameters() if name.startswith(group+'.') and p.grad is not None)>0
    torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=1e-4).step()
    after=m.parameter_groups(model.state_dict());assert all(before[k]!=after[k] for k in before)


def counts_fixture():
    _,_,_,rows=samples();mask=[r['family'] not in m.core.CORE_FAMILIES for r in rows]
    counts=m.FullSupplementCounts(FAMILIES);counts.update(rows,mask)
    return counts.report()


def test_actual_supplement_teacher_and_native_core_counts_keep_honest_single_model_scope():
    report=m.validate_training_counts(counts_fixture(),FAMILIES,1)
    assert report['supplement_rows']==8 and report['original_rows']==88
    assert report['eligible_family_rows'][m.core.UNKNOWN]==16
    assert report['teacher_logits_used'] and not report['second_model_resident'] and not report['teacher_cache_deployed']


@pytest.mark.parametrize('fault',['supplement','source','core_teacher','native_weight','teacher_models','heldout'])
def test_bad_source_or_teacher_accounting_rejected(fault):
    report=counts_fixture()
    if fault=='supplement':report['supplement_rows']-=1
    elif fault=='source':report['source_target_rows'][0]['rows']+=1
    elif fault=='core_teacher':next(r for r in report['source_target_rows'] if r['target_family']=='PingFang')['eligible_rows']=1
    elif fault=='native_weight':report['native_core_weighted_rows']+=1
    elif fault=='teacher_models':report['second_model_resident']=True
    else:report['test_read']=True
    with pytest.raises(ValueError):m.validate_training_counts(report,FAMILIES,1)


def test_fixed_optimizer_image_input_and_unchanged_runtime_contract():
    args=m.parser().parse_args([])
    assert args.steps==3000 and args.seed==2026091408 and args.device=='mps'
    assert m.LEARNING_RATE==2e-5 and m.MINIMUM_LEARNING_RATE==2e-6 and m.EVAL_EVERY==500
    assert m.OBJECTIVE['training_inputs']==['image_tiles'] and m.OBJECTIVE['unknown_ce_weight']==1.
    assert m.FIXED_RUNTIME=={'temperature':1.,'gates':{'min_score':.7,'min_margin':.01,'min_patch_agreement':2/3},'max_size_relative_spread':.2}
