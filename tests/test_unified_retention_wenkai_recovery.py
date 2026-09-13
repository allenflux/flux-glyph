"""Integration checks for the single-change WenKai recovery trial."""
from copy import deepcopy
import json

import pytest

from training import train_unified_retention_margin as margin
from training import train_unified_retention_wenkai_recovery as trial
from training.prepare_unified_regions import FAMILIES
from training.retention_wenkai_recovery_loss import (
    WENKAI_FAMILY, WENKAI_RECOVERY_WEIGHTING, wenkai_recovery_extra_ce)
from test_unified_retention_margin import production_rows


def inputs():
    torch = pytest.importorskip('torch'); torch.manual_seed(2026091414)
    rows = production_rows(); targets = torch.tensor([row['target'] for row in rows])
    logits = torch.randn(96, 25); ratios = torch.randn(96); sizes = torch.randn(96)
    features = torch.randn(96, 256); head = torch.randn(25, 256)
    teacher = torch.full((96, 25), -2.); teacher[torch.arange(96), targets] = 2.
    return rows, targets, logits, ratios, sizes, features, head, teacher


def test_full_loss_changes_only_wenkai_ce_and_corresponding_logits_gradient():
    torch = pytest.importorskip('torch')
    rows,targets,logits,ratios,sizes,features,head,teacher=inputs()
    old_values=[value.detach().clone().requires_grad_(True) for value in (logits,ratios,features,head)]
    new_values=[value.detach().clone().requires_grad_(True) for value in (logits,ratios,features,head)]
    old=margin.full_losses(old_values[0],old_values[1],old_values[2],old_values[3],
        targets,sizes,teacher,rows,FAMILIES)
    new=trial.full_losses(new_values[0],new_values[1],new_values[2],new_values[3],
        targets,sizes,teacher,rows,FAMILIES)
    extra=wenkai_recovery_extra_ce(new_values[0],targets,rows,FAMILIES)
    assert len(new)==len(old)==6
    torch.testing.assert_close(new[0],old[0]+extra)
    torch.testing.assert_close(new[1],old[1]+extra)
    for index in (2,3,4,5):torch.testing.assert_close(new[index],old[index])
    old_grad=torch.autograd.grad(old[0],old_values,retain_graph=True)
    new_grad=torch.autograd.grad(new[0],new_values,retain_graph=True)
    mask=targets==FAMILIES.index(WENKAI_FAMILY)
    assert bool(mask.any())
    torch.testing.assert_close(new_grad[0][~mask],old_grad[0][~mask])
    assert not torch.equal(new_grad[0][mask],old_grad[0][mask])
    for actual,expected in zip(new_grad[1:],old_grad[1:]):torch.testing.assert_close(actual,expected)
    unknown=targets==FAMILIES.index('__unknown__')
    torch.testing.assert_close(new_grad[0][unknown],old_grad[0][unknown])
    old_size,=torch.autograd.grad(old[2],old_values[1],retain_graph=True)
    new_size,=torch.autograd.grad(new[2],new_values[1],retain_graph=True)
    old_preservation,=torch.autograd.grad(old[3],old_values[0],retain_graph=True)
    new_preservation,=torch.autograd.grad(new[3],new_values[0],retain_graph=True)
    old_angular=torch.autograd.grad(old[4],old_values[2:4],retain_graph=True)
    new_angular=torch.autograd.grad(new[4],new_values[2:4],retain_graph=True)
    torch.testing.assert_close(new_size,old_size)
    torch.testing.assert_close(new_preservation,old_preservation)
    for actual,expected in zip(new_angular,old_angular):torch.testing.assert_close(actual,expected)


def objective_plan(cache_sha):
    return {'schema':'flux-glyph-wide-wenkai-recovery-plan-v1',
        'architecture':trial.ARCHITECTURE,'design_changes':trial.DESIGN_CHANGES,
        'source_initializer_sha256':trial.STUDENT_CHECKPOINT_SHA,
        'teacher_cache_sha256':trial.TEACHER_CACHE_SHA,'objective':deepcopy(trial.OBJECTIVE),
        'steps':trial.STEPS,'eval_every':trial.EVAL_EVERY,'seed':trial.SEED,
        'weight_pairs_manifest_sha256':trial.WEIGHT_PAIRS_MANIFEST_SHA,
        'weight_teacher_cache_manifest_sha256':cache_sha,'runtime':trial.FIXED_RUNTIME,
        'acceptance_plan_sha256':'a'*64,'r21_inference_report_sha256':trial.R21_REPORT_SHA,
        'teacher_policy':trial.TEACHER_POLICY,
        'checkpoint_selection':'fixed final step 6000, no intermediate candidate selection',
        'calibration_used_in_prior_development':True,'blind_test':False,
        'new_training_started':False,'single_cause_claim':False}


def test_plan_pins_weighting_and_rejects_drift(tmp_path):
    path=tmp_path/'PLAN.json';cache=trial.WEIGHT_TEACHER_CACHE_MANIFEST_SHA
    plan=objective_plan(cache);path.write_text(json.dumps(plan))
    assert trial.read_objective_plan(path,'a'*64,cache)==plan
    plan['objective']['wenkai_recovery_weighting']['resulting_known_ce_weight']=3.
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):trial.read_objective_plan(path,'a'*64,cache)
    expected={**margin.OBJECTIVE,'class_prior_weighting':True,
        'wenkai_recovery_weighting':WENKAI_RECOVERY_WEIGHTING,
        'inference_class_prior_changed':False}
    assert trial.OBJECTIVE==expected and trial.OBJECTIVE['teacher_distribution_kl'] is True


def test_counts_record_weighted_family_by_source_face_separate_from_native_core():
    rows=production_rows();counter=trial.FullSupplementCounts(FAMILIES,
        sorted({row['source_font_family'] for row in rows if row['family']=='__unknown__'}))
    counter.update(rows,[True]*96,[row['target'] for row in rows])
    report=trial.validate_training_counts(counter.report(),FAMILIES,1)
    assert report['wenkai_recovery_weighting']==WENKAI_RECOVERY_WEIGHTING
    assert report['wenkai_recovery_rows']==report['family_rows'][WENKAI_FAMILY]
    assert sum(row['rows'] for row in report['wenkai_recovery_source_face_rows'])==report['wenkai_recovery_rows']
    assert WENKAI_FAMILY not in report['native_core_weighted_family_rows']


def test_actual_single_cnn_font_and_size_update_and_public_forward_identity():
    torch=pytest.importorskip('torch')
    from training.wide_region_network import WideRegionFontClassifier
    torch.manual_seed(20);torch.set_num_threads(4)
    model=WideRegionFontClassifier(25);rows,targets,_,_,sizes,_,_,teacher=inputs()
    tiles=torch.rand(96,1,64,256)
    logits,ratios,features=trial.forward_with_features(model,tiles)
    expected_logits,expected_ratios=model(tiles)
    assert torch.equal(logits,expected_logits) and torch.equal(ratios,expected_ratios)
    loss,*_=trial.full_losses(logits,ratios,features,model.family_head.weight,
        targets,sizes,teacher,rows,FAMILIES)
    loss.backward()
    assert model.family_head.weight.grad is not None and model.size_head.weight.grad is not None
    assert bool((model.family_head.weight.grad!=0).any()) and bool((model.size_head.weight.grad!=0).any())


def test_fixed_replay_runtime_and_defaults_match_candidate18():
    args=trial.parser().parse_args([]);old=margin.parser().parse_args([])
    assert (trial.STEPS,trial.EVAL_EVERY,trial.SEED,trial.LEARNING_RATE,
        trial.MINIMUM_LEARNING_RATE)==(6000,6000,2026091414,2e-5,2e-6)
    assert trial.SAMPLING==margin.SAMPLING and trial.FIXED_RUNTIME==margin.FIXED_RUNTIME
    assert args.output.name=='run-wide-wenkai-recovery-v1'
    assert args.objective_plan.parent.name=='wenkai-recovery-plan-v1'
    assert args.weight_pairs==old.weight_pairs and args.weight_teacher_cache==old.weight_teacher_cache
