"""Integration checks for the face-balanced angular-margin trial."""
from copy import deepcopy
import json

import pytest

from training import train_unified_retention_face_balanced as baseline
from training import train_unified_retention_margin as m
from training.prepare_unified_regions import FAMILIES
from training.retention_angular_margin_loss import angular_margin_loss
from test_retention_face_balanced_sampler import sampler


def production_rows():
    rows=deepcopy(sampler().batch())
    for row in rows:
        if row['family'] in FAMILIES:row['target']=FAMILIES.index(row['family'])
        else:row['family']=FAMILIES[row['target']]
    return rows


def test_feature_forward_is_exact_public_forward_with_learned_size_head():
    torch=pytest.importorskip('torch')
    from training.wide_region_network import WideRegionFontClassifier
    torch.manual_seed(14);torch.set_num_threads(4)
    model=WideRegionFontClassifier(25).eval()
    with torch.no_grad():
        model.size_head.weight.fill_(.002);model.size_head.bias.fill_(.1)
        tiles=torch.rand(3,1,64,256)
        expected_logits,expected_sizes=model(tiles)
        logits,sizes,features=m.forward_with_features(model,tiles)
    assert features.shape==(3,256) and bool((sizes!=0).all())
    assert torch.equal(logits,expected_logits)
    assert torch.equal(sizes,expected_sizes)


def test_full_loss_adds_only_coefficient_inclusive_known_margin():
    torch=pytest.importorskip('torch')
    torch.manual_seed(1414)
    rows=production_rows();targets=torch.tensor([row['target'] for row in rows])
    logits=torch.randn(96,25,requires_grad=True)
    ratios=torch.randn(96,requires_grad=True);sizes=torch.randn(96)
    features=torch.randn(96,256,requires_grad=True)
    head=torch.randn(25,256,requires_grad=True)
    teacher=torch.full((96,25),-2.);teacher[torch.arange(96),targets]=2.
    base=baseline.full_losses(logits,ratios,targets,sizes,teacher,rows,FAMILIES)
    total,ce,size,preservation,auxiliary,mask=m.full_losses(
        logits,ratios,features,head,targets,sizes,teacher,rows,FAMILIES)
    expected_auxiliary=angular_margin_loss(features,head,targets,rows,FAMILIES)
    torch.testing.assert_close(auxiliary,expected_auxiliary)
    torch.testing.assert_close(total,base[0]+expected_auxiliary)
    for actual,expected in zip((ce,size,preservation,mask),base[1:]):
        torch.testing.assert_close(actual,expected)
    total.backward()
    assert all(value.grad is not None and bool(torch.isfinite(value.grad).all())
        for value in (logits,ratios,features,head))
    unknown=targets==FAMILIES.index(m.core.UNKNOWN)
    assert bool((features.grad[unknown]==0).all())
    assert auxiliary.item()>0


def objective_plan(cache_sha):
    return {'schema':'flux-glyph-wide-angular-margin-plan-v1',
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


def test_plan_defaults_and_objective_pin_only_the_training_margin(tmp_path):
    path=tmp_path/'PLAN.json';cache_sha=m.WEIGHT_TEACHER_CACHE_MANIFEST_SHA
    path.write_text(json.dumps(objective_plan(cache_sha)))
    assert m.read_objective_plan(path,'a'*64,cache_sha)==objective_plan(cache_sha)
    changed=dict(path=json.loads(path.read_text()));changed['path']['objective']['angular_margin']['scale']=15.
    path.write_text(json.dumps(changed['path']))
    with pytest.raises(ValueError):m.read_objective_plan(path,'a'*64,cache_sha)
    assert {key:value for key,value in m.OBJECTIVE.items() if key!='angular_margin'} == baseline.OBJECTIVE
    assert m.OBJECTIVE['angular_margin']==m.ANGULAR_MARGIN
    assert m.ANGULAR_MARGIN['weight']==.10 and m.ANGULAR_MARGIN['scale']==16.
    assert m.ANGULAR_MARGIN['angle_radians']==.10 and m.ANGULAR_MARGIN['denominator']==96
    assert m.ANGULAR_MARGIN['inference_rule'] is False


def test_counts_and_defaults_preserve_candidate17_replay_and_runtime():
    replay=sampler(17);counts=m.FullSupplementCounts(replay.base.families,replay.unknown_source_order)
    rows=replay.batch();counts.update(rows,[True]*96,[row['target'] for row in rows])
    report=m.validate_training_counts(counts.report(),replay.base.families,1)
    assert report['angular_margin']==m.ANGULAR_MARGIN
    assert report['angular_margin_applies_at_inference'] is False
    args=m.parser().parse_args([])
    assert (m.STEPS,m.EVAL_EVERY,m.SEED,m.LEARNING_RATE,m.MINIMUM_LEARNING_RATE)==(
        6000,6000,2026091414,2e-5,2e-6)
    assert m.SAMPLING==baseline.SAMPLING and m.FIXED_RUNTIME==baseline.FIXED_RUNTIME
    assert args.output.name=='run-wide-angular-margin-v1'
    assert args.objective_plan.parent.name=='angular-margin-plan-v1'
    assert args.weight_pairs==baseline.parser().parse_args([]).weight_pairs
    assert args.weight_teacher_cache==baseline.parser().parse_args([]).weight_teacher_cache


def test_actual_model_margin_update_reaches_all_existing_parameter_groups():
    torch=pytest.importorskip('torch')
    from training.wide_region_network import WideRegionFontClassifier
    torch.manual_seed(1414);torch.set_num_threads(4)
    model=WideRegionFontClassifier(25);rows=production_rows()
    targets=torch.tensor([row['target'] for row in rows])
    teacher=torch.full((96,25),-2.);teacher[torch.arange(96),targets]=2.
    logits,ratios,features=m.forward_with_features(model,torch.rand(96,1,64,256))
    loss,*_=m.full_losses(logits,ratios,features,model.family_head.weight,
        targets,torch.full((96,),.2),teacher,rows,FAMILIES)
    loss.backward()
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters())

