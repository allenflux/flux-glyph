"""Integration checks for asymmetric named-teacher relearning."""
from copy import deepcopy
import json

import pytest

from training import evaluate_unified_retention_margin as margin_evaluator
from training import evaluate_unified_retention_relearning as evaluator
from training import train_unified_retention_margin as margin_trial
from training import train_unified_retention_relearning as m
from training.prepare_unified_regions import FAMILIES
from training.retention_angular_margin_loss import angular_margin_loss
from training.retention_asymmetric_teacher_loss import asymmetric_teacher_loss
from training.retention_confidence_floor_loss import unknown_floor_loss
from test_retention_face_balanced_sampler import sampler


def production_rows():
    rows=deepcopy(sampler().batch())
    for row in rows:
        if row['family'] in FAMILIES:row['target']=FAMILIES.index(row['family'])
        else:row['family']=FAMILIES[row['target']]
    return rows


def test_named_preservation_is_floor_only_and_unknown_is_exact_full_kl():
    torch=pytest.importorskip('torch')
    targets=torch.tensor([2,3])
    teacher=torch.tensor([[0.,0.,3.,1.],[0.,0.,0.,3.]]).detach()
    better=torch.tensor([[0.,-4.,6.,-2.],[2.,-1.,0.,1.]],requires_grad=True)
    loss,mask=asymmetric_teacher_loss(better,teacher,targets,3)
    expected_unknown=torch.nn.functional.kl_div(
        better[1:].log_softmax(1),teacher[1:].softmax(1),reduction='none').sum()
    torch.testing.assert_close(loss,expected_unknown/2)
    assert mask.tolist()==[True,True]
    named_kl=torch.nn.functional.kl_div(
        better[:1].log_softmax(1),teacher[:1].softmax(1),reduction='none').sum()
    assert named_kl>0
    lower=better.detach().clone();lower[0]=torch.tensor([2.,2.,1.,2.])
    lower_loss,_=asymmetric_teacher_loss(lower,teacher,targets,3)
    assert lower_loss>loss


def test_full_loss_is_exact_ce_size_asymmetric_preservation_and_margin():
    torch=pytest.importorskip('torch')
    torch.manual_seed(1414)
    rows=production_rows();targets=torch.tensor([row['target'] for row in rows])
    logits=torch.randn(96,25,requires_grad=True);ratios=torch.randn(96,requires_grad=True)
    sizes=torch.randn(96);features=torch.randn(96,256,requires_grad=True)
    head=torch.randn(25,256,requires_grad=True)
    teacher=torch.full((96,25),-2.);teacher[torch.arange(96),targets]=2.
    total,ce,size,preservation,auxiliary,mask=m.full_losses(
        logits,ratios,features,head,targets,sizes,teacher,rows,FAMILIES)
    prior=margin_trial.full_losses(
        logits,ratios,features,head,targets,sizes,teacher,rows,FAMILIES)
    weights=torch.tensor(m.core.native_core_weights(rows,targets.tolist(),FAMILIES))
    known=targets!=24;unknown=~known
    expected_ce=((torch.nn.functional.cross_entropy(logits[known],targets[known],
        label_smoothing=.03,reduction='none')*weights[known]).sum()
        +(unknown_floor_loss(logits[unknown],24)*weights[unknown]).sum())/96
    expected_size=torch.nn.functional.smooth_l1_loss(ratios,sizes)
    expected_preservation,expected_mask=asymmetric_teacher_loss(logits,teacher,targets,24)
    expected_auxiliary=angular_margin_loss(features,head,targets,rows,FAMILIES)
    torch.testing.assert_close(ce,expected_ce);torch.testing.assert_close(size,expected_size)
    torch.testing.assert_close(preservation,expected_preservation)
    torch.testing.assert_close(auxiliary,expected_auxiliary)
    torch.testing.assert_close(ce,prior[1]);torch.testing.assert_close(size,prior[2])
    torch.testing.assert_close(auxiliary,prior[4]);assert torch.equal(mask,prior[5])
    torch.testing.assert_close(total,expected_ce+.2*expected_size
        +m.ASYMMETRIC_PRESERVATION['outer_weight']*expected_preservation+expected_auxiliary)
    assert torch.equal(mask,expected_mask) and bool(mask.all())
    total.backward()
    assert teacher.grad is None
    assert all(value.grad is not None and bool(torch.isfinite(value.grad).all())
        for value in (logits,ratios,features,head))


def test_feature_forward_is_bitwise_the_unchanged_public_model_forward():
    torch=pytest.importorskip('torch')
    from training.wide_region_network import WideRegionFontClassifier
    torch.manual_seed(19);torch.set_num_threads(4)
    model=WideRegionFontClassifier(25).eval()
    with torch.no_grad():
        model.size_head.weight.fill_(.003);model.size_head.bias.fill_(.2)
        tiles=torch.rand(3,1,64,256)
        expected=model(tiles);actual=m.forward_with_features(model,tiles)
    assert torch.equal(actual[0],expected[0]) and torch.equal(actual[1],expected[1])
    assert actual[2].shape==(3,256) and bool((actual[1]!=0).all())
    assert set(model.state_dict())==set(WideRegionFontClassifier(25).state_dict())


def objective_plan(cache_sha):
    return {'schema':'flux-glyph-wide-named-relearning-plan-v1',
        'architecture':m.ARCHITECTURE,'design_changes':m.DESIGN_CHANGES,
        'source_initializer_sha256':m.STUDENT_CHECKPOINT_SHA,
        'teacher_cache_sha256':m.TEACHER_CACHE_SHA,'objective':m.OBJECTIVE,
        'steps':m.STEPS,'eval_every':m.EVAL_EVERY,'seed':m.SEED,
        'weight_pairs_manifest_sha256':m.WEIGHT_PAIRS_MANIFEST_SHA,
        'weight_teacher_cache_manifest_sha256':cache_sha,
        'runtime':m.FIXED_RUNTIME,'acceptance_plan_sha256':'a'*64,
        'r21_inference_report_sha256':m.R21_REPORT_SHA,'teacher_policy':m.TEACHER_POLICY,
        'checkpoint_selection':'fixed final step 6000, no intermediate candidate selection',
        'calibration_used_in_prior_development':True,'blind_test':False,
        'new_training_started':False,'single_cause_claim':False}


def test_plan_and_objective_describe_asymmetry_without_all_row_kl(tmp_path):
    path=tmp_path/'PLAN.json';cache_sha=m.WEIGHT_TEACHER_CACHE_MANIFEST_SHA
    path.write_text(json.dumps(objective_plan(cache_sha)))
    assert m.read_objective_plan(path,'a'*64,cache_sha)==objective_plan(cache_sha)
    assert m.OBJECTIVE['asymmetric_teacher_preservation']==m.ASYMMETRIC_PRESERVATION
    assert m.OBJECTIVE['teacher_distribution_kl_all_eligible_rows'] is False
    assert m.OBJECTIVE['teacher_distribution_kl'] is False
    assert m.OBJECTIVE['named_teacher_distribution_kl'] is False
    assert m.OBJECTIVE['unknown_teacher_distribution_kl'] is True
    assert m.OBJECTIVE['named_teacher_confidence_floor'] is True
    assert 'teacher_kl_reduction' not in m.OBJECTIVE
    assert m.OBJECTIVE['angular_margin']==margin_trial.ANGULAR_MARGIN
    changed=json.loads(path.read_text());changed['objective']['named_teacher_confidence_floor']=False
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):m.read_objective_plan(path,'a'*64,cache_sha)


def test_replay_data_runtime_and_development_policy_are_unchanged():
    replay=sampler(17);counts=m.FullSupplementCounts(replay.base.families,replay.unknown_source_order)
    rows=replay.batch();predictions=[row['target'] for row in rows]
    counts.update(rows,[True]*96,predictions)
    report=m.validate_training_counts(counts.report(),replay.base.families,1)
    assert report['asymmetric_teacher_preservation']==m.ASYMMETRIC_PRESERVATION
    assert report['selected_teacher_true_target_rows']=={'named_b08':80,'unknown_r21':16}
    assert m.SAMPLING==margin_trial.SAMPLING and m.FIXED_RUNTIME==margin_trial.FIXED_RUNTIME
    args=m.parser().parse_args([])
    assert (m.STEPS,m.EVAL_EVERY,m.SEED,m.LEARNING_RATE,m.MINIMUM_LEARNING_RATE)==(
        6000,6000,2026091414,2e-5,2e-6)
    assert args.output.name=='run-wide-named-relearning-v1'
    assert args.objective_plan.parent.name=='named-relearning-plan-v1'
    assert args.weight_pairs==margin_trial.parser().parse_args([]).weight_pairs
    assert args.weight_teacher_cache==margin_trial.parser().parse_args([]).weight_teacher_cache
    assert evaluator.COMPARISON_POLICY==margin_evaluator.COMPARISON_POLICY
