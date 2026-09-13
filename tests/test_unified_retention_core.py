"""The training-only teacher has no gradient or authority over PingFang targets."""
from copy import deepcopy
import pytest
from training import train_unified_retention_core as module
from test_unified_retention import FAMILIES,sampler_data

def identity_rows(targets):
    return [{'target':int(t),'family':FAMILIES[int(t)%len(FAMILIES)],'split':'train','native_font_verified':True,
             'domain':'android' if i%3==0 else 'ios','view':'half' if i%4==0 else 'native'}
            for i,t in enumerate(targets)]


def tensors():
    torch=pytest.importorskip('torch');torch.set_num_threads(4);torch.manual_seed(8)
    targets=torch.arange(96)%25
    student=torch.randn(96,25,dtype=torch.float64,requires_grad=True)
    teacher=torch.full((96,25),-2.,dtype=torch.float64)
    teacher[torch.arange(96),targets]=2.
    # Half the targets are deliberately predicted incorrectly by the teacher.
    teacher[::2,:]=-2.;teacher[torch.arange(0,96,2),(targets[::2]+1)%25]=2.
    sizes=torch.zeros(96,dtype=torch.float64)
    ratios=torch.linspace(-1.5,1.5,96,dtype=torch.float64,requires_grad=True)
    return torch,student,ratios,targets,sizes,teacher


def test_correct_non_pingfang_mask_kl_direction_and_exact_student_gradient():
    torch,student,ratios,targets,sizes,teacher=tensors()
    total,ce,size,kl,mask=module.distilled_losses(student,ratios,targets,sizes,teacher,FAMILIES,identity_rows(targets))
    expected=torch.tensor([FAMILIES[int(t)] not in module.CORE_FAMILIES for t in targets])&(teacher.argmax(1)==targets)
    assert torch.equal(mask,expected) and mask.any() and not mask[targets==4].any()
    probs=teacher[mask].softmax(1)
    explicit=(probs*(probs.log()-student[mask].log_softmax(1))).sum(1).mean()
    torch.testing.assert_close(kl,explicit)
    torch.testing.assert_close(total,ce+.2*size+explicit)
    kl.backward(retain_graph=True)
    expected_grad=torch.zeros_like(student)
    expected_grad[mask]=(student.detach()[mask].softmax(1)-probs)/mask.sum()
    torch.testing.assert_close(student.grad,expected_grad)
    assert not student.grad[~mask].any() and teacher.grad is None
    student.grad=None;total.backward()
    assert (student.grad.abs().sum(0)>0).all() and ratios.grad.abs().sum()>0


def test_empty_teacher_mask_is_differentiable_zero_and_supervised_loss_remains():
    torch,student,ratios,targets,sizes,teacher=tensors()
    targets.fill_(FAMILIES.index('PingFang'))
    total,ce,size,kl,mask=module.distilled_losses(student,ratios,targets,sizes,teacher,FAMILIES,identity_rows(targets))
    assert not mask.any() and kl.item()==0 and kl.requires_grad
    kl.backward(retain_graph=True);assert not student.grad.any()
    student.grad=None;total.backward()
    assert student.grad.abs().sum()>0 and ratios.grad.abs().sum()>0
    torch.testing.assert_close(total,ce+.2*size)


@pytest.mark.parametrize('fault',['trainable_teacher','nan_teacher','nan_student','wrong_shape','target_range','target_dtype'])
def test_invalid_distillation_inputs_are_rejected(fault):
    torch,student,ratios,targets,sizes,teacher=tensors()
    if fault=='trainable_teacher':teacher.requires_grad_(True)
    elif fault=='nan_teacher':teacher[0,0]=float('nan')
    elif fault=='nan_student':student=student.detach();student[0,0]=float('nan')
    elif fault=='wrong_shape':teacher=teacher[:95]
    elif fault=='target_range':targets[0]=25
    else:targets=targets.float()
    with pytest.raises(ValueError):module.distilled_losses(student,ratios,targets,sizes,teacher,FAMILIES,identity_rows(targets))


def test_teacher_forward_requires_eval_and_frozen_parameters_and_never_builds_graph():
    torch=pytest.importorskip('torch')
    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__();self.head=torch.nn.Linear(4,25)
        def forward(self,x):return self.head(x),x.sum(1)
    model=Teacher();images=torch.randn(3,4,requires_grad=True)
    with pytest.raises(ValueError):module.frozen_teacher_logits(model,images)
    model.requires_grad_(False);model.eval()
    before={k:v.clone() for k,v in model.state_dict().items()}
    result=module.frozen_teacher_logits(model,images)
    assert not result.requires_grad and images.grad is None
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    model.train()
    with pytest.raises(ValueError):module.frozen_teacher_logits(model,images)


def counter_report():
    rows,pools=sampler_data();sampler=module.RetentionSampler(rows,FAMILIES,2026091404,pools)
    counter=module.DistillationCounter(FAMILIES)
    for _ in range(2):
        batch=sampler.batch();counter.update(batch,[r['family'] not in module.CORE_FAMILIES and i%2==0 for i,r in enumerate(batch)])
    result=counter.report();result.update(teacher_state_before_sha256='a'*64,teacher_state_after_sha256='a'*64,
        teacher_optimizer_steps=0,training_model_count=2,deployed_model_count=1)
    return result


def test_count_report_reconciles_source_family_domain_and_step_without_pf_teacher():
    report=counter_report()
    assert module.validate_distillation_report(report,FAMILIES,2)==report
    assert report['rows']==192 and report['eligible_rows']>0 and all(report['eligible_family_rows'].get(f,0)==0 for f in module.CORE_FAMILIES)


@pytest.mark.parametrize('fault',['pf','source','steps','teacher_changed','teacher_optimizer','deployed_teacher'])
def test_distillation_accounting_cannot_claim_invalid_teacher_provenance(fault):
    report=counter_report()
    if fault=='pf':report['eligible_family_rows']['PingFang']=1
    elif fault=='source':report['source_target_rows'][0]['eligible_rows']+=1
    elif fault=='steps':report['eligible_rows_per_step'][0]+=1
    elif fault=='teacher_changed':report['teacher_state_after_sha256']='b'*64
    elif fault=='teacher_optimizer':report['teacher_optimizer_steps']=1
    else:report['deployed_model_count']=2
    with pytest.raises(ValueError):module.validate_distillation_report(report,FAMILIES,2)


def test_nontrain_rows_rejected_by_teacher_accounting_and_constants_frozen():
    rows,pools=sampler_data();batch=module.RetentionSampler(rows,FAMILIES,3,pools).batch()
    batch=deepcopy(batch);batch[0]['split']='development_holdout'
    with pytest.raises(ValueError):module.DistillationCounter(FAMILIES).update(batch,[False]*96)
    args=module.parser().parse_args(['--data','d','--plan','p','--output','o'])
    assert args.steps==1500 and args.seed==2026091404
    assert module.EVAL_EVERY==500 and module.BATCH_SIZE==96
    assert module.LEARNING_RATE==1e-5 and module.MINIMUM_LEARNING_RATE==1e-6
    assert module.OBJECTIVE['class_prior_weighting'] is False and module.OBJECTIVE['teacher_kl_weight']==1
    assert module.FIXED_RUNTIME['temperature']==1 and module.FIXED_RUNTIME['gates']['min_score']==.7


def test_native_core_ce_uses_fixed96_denominator_and_size_stays_unweighted():
    torch,student,ratios,targets,sizes,teacher=tensors();rows=identity_rows(targets)
    total,ce,size,kl,mask=module.distilled_losses(student,ratios,targets,sizes,teacher,FAMILIES,rows)
    expected_weights=torch.tensor([2. if r['domain']=='ios' and r['view']=='native' and r['family'] in module.CORE_FAMILIES
                                   else 1. for r in rows],dtype=torch.float64)
    assert expected_weights.sum()>96
    smoothed=torch.nn.functional.one_hot(targets,25).double()*.97+.03/25
    per_row=-(smoothed*student.log_softmax(1)).sum(1)
    torch.testing.assert_close(ce,(per_row*expected_weights).sum()/96)
    assert not torch.isclose(ce,(per_row*expected_weights).sum()/expected_weights.sum())
    torch.testing.assert_close(size,torch.nn.functional.smooth_l1_loss(ratios,sizes))
    total.backward()
    grad=(student.detach().softmax(1)-smoothed)*expected_weights[:,None]/96
    grad[mask]+=(student.detach()[mask].softmax(1)-teacher[mask].softmax(1))/mask.sum()
    torch.testing.assert_close(student.grad,grad)
    torch.testing.assert_close(ratios.grad,.2*(ratios.detach()-sizes).clamp(-1,1)/96)


@pytest.mark.parametrize('fault',['holdout','wrong_family','wrong_target','unverified','missing_view','bad_domain'])
def test_row_identity_cannot_change_loss_weight_or_admit_nontrain_inputs(fault):
    torch,student,ratios,targets,sizes,teacher=tensors();rows=identity_rows(targets)
    if fault=='holdout':rows[0]['split']='development_holdout'
    elif fault=='wrong_family':rows[0]['family']='PingFang'
    elif fault=='wrong_target':rows[0]['target']=4
    elif fault=='unverified':rows[0]['native_font_verified']=False
    elif fault=='missing_view':del rows[0]['view']
    else:rows[0]['domain']='windows'
    with pytest.raises(ValueError):module.distilled_losses(student,ratios,targets,sizes,teacher,FAMILIES,rows)


@pytest.mark.parametrize('fault',['wrong_total','wrong_source','wrong_rule','sf_teacher'])
def test_native_core_weight_evidence_is_bound_to_exact_rule_and_actual_counts(fault):
    report=counter_report()
    assert report['native_core_weighted_rows']==sum(report['native_core_weighted_family_rows'].values())
    assert all(27<=n<=33 for n in report['native_core_weighted_rows_per_step'])
    if fault=='wrong_total':report['native_core_weighted_rows']+=1
    elif fault=='wrong_source':report['native_core_source_rows'][0]['domain']='android'
    elif fault=='wrong_rule':report['native_core_weighting']={**report['native_core_weighting'],'denominator':128}
    else:report['eligible_family_rows']['SF Pro']=1
    with pytest.raises(ValueError):module.validate_distillation_report(report,FAMILIES,2)
