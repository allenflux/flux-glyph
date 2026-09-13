"""b08 preservation expands only the teacher mask and binds actual TRAIN eligibility."""
from copy import deepcopy
import numpy as np
import pytest
from training import train_unified_retention_teacher_preserve as m
from training import train_unified_retention_paired_known as old
from training.prepare_unified_regions import FAMILIES
from test_unified_retention_paired_known import sampler, routing_fixture


def tensors():
    torch=pytest.importorskip('torch');torch.set_num_threads(4)
    rows=sampler().batch();targets=torch.tensor([r['target'] for r in rows])
    logits=torch.linspace(-1.,1.,96*25,dtype=torch.float64).reshape(96,25).requires_grad_()
    teacher=torch.full((96,25),-2.,dtype=logits.dtype);teacher[torch.arange(96),targets]=2.
    ratios=torch.linspace(-.2,.4,96,dtype=logits.dtype).requires_grad_()
    sizes=torch.full((96,),.2,dtype=logits.dtype)
    return torch,rows,logits,ratios,targets,sizes,teacher


def test_all_correct_core_and_unknown_receive_kl_with_supervision_unchanged():
    torch,rows,z,ratios,y,sizes,teacher=tensors()
    total,ce,size,kl,mask=m.full_losses(z,ratios,y,sizes,teacher,rows,FAMILIES)
    _,old_ce,old_size,_,old_mask=old.full_losses(z,ratios,y,sizes,teacher,rows,FAMILIES)
    torch.testing.assert_close(ce,old_ce,rtol=0,atol=0)
    torch.testing.assert_close(size,old_size,rtol=0,atol=0)
    assert mask.all() and not old_mask[y==24].any()
    assert all(mask[y==FAMILIES.index(f)].all() for f in m.core.CORE_FAMILIES)
    expected=torch.nn.functional.kl_div(z.log_softmax(1),teacher.softmax(1),reduction='batchmean')
    torch.testing.assert_close(kl,expected)
    torch.testing.assert_close(total,ce+.2*size+2*expected)
    total.backward();assert torch.isfinite(z.grad).all() and torch.isfinite(ratios.grad).all()
    assert teacher.grad is None


def test_kl_gradient_equals_frozen_teacher_probability_difference_over_eligible_rows():
    torch,rows,z,ratios,y,sizes,teacher=tensors()
    wrong=torch.arange(0,96,3);teacher[wrong]=-2.;teacher[wrong,(y[wrong]+1)%25]=3.
    _,_,_,kl,mask=m.full_losses(z,ratios,y,sizes,teacher,rows,FAMILIES)
    assert (~mask[wrong]).all() and int(mask.sum())==64
    grad=torch.autograd.grad(kl,z)[0];expected=torch.zeros_like(z)
    expected[mask]=(z[mask].softmax(1)-teacher[mask].softmax(1))/64
    torch.testing.assert_close(grad,expected,rtol=1e-10,atol=1e-12)


def test_empty_truth_correct_mask_is_differentiable_zero():
    torch,rows,z,ratios,y,sizes,teacher=tensors()
    teacher[:]=-2.;teacher[torch.arange(96),(y+1)%25]=3.
    _,_,_,kl,mask=m.full_losses(z,ratios,y,sizes,teacher,rows,FAMILIES)
    assert not mask.any() and kl.item()==0
    torch.testing.assert_close(torch.autograd.grad(kl,z)[0],torch.zeros_like(z),atol=0,rtol=0)


@pytest.mark.parametrize('fault',['teacher_grad','teacher_nan','teacher_shape','teacher_dtype','native_size_grad','bad_target','nontrain','relabel','unverified'])
def test_invalid_training_evidence_or_teacher_is_rejected(fault):
    torch,rows,z,ratios,y,sizes,teacher=tensors()
    if fault=='teacher_grad':teacher.requires_grad_()
    elif fault=='teacher_nan':teacher[0,0]=float('nan')
    elif fault=='teacher_shape':teacher=teacher[:95]
    elif fault=='teacher_dtype':teacher=teacher.float()
    elif fault=='native_size_grad':sizes.requires_grad_()
    elif fault=='bad_target':y[0]=25
    elif fault=='nontrain':rows[0]['split']='calibration'
    elif fault=='relabel':rows[0]['family']='__unknown__'
    else:rows[0]['native_font_verified']=False
    with pytest.raises(ValueError):m.full_losses(z,ratios,y,sizes,teacher,rows,FAMILIES)


def test_expanded_counter_reconciles_correct_unknown_core_and_source_totals():
    replay=sampler();counts=m.FullSupplementCounts(FAMILIES,replay.unknown_source_order)
    for _ in range(11):
        rows=replay.batch();pred=[r['target'] for r in rows];counts.update(rows,[True]*96,pred)
    report=counts.report();m.validate_training_counts(report,FAMILIES,11)
    assert report['eligible_rows']==1056 and report['eligible_family_rows']['__unknown__']==176
    assert all(report['eligible_family_rows'][f]>0 for f in m.core.CORE_FAMILIES)
    assert all(r['rows']==r['eligible_rows'] for r in report['source_target_rows'])
    assert report['teacher_mask_verified_against_same_tile_argmax'] is True


@pytest.mark.parametrize('fault',['false_mask','wrong_pred','nonint_pred','source_count','eligibility_total','provenance_flag'])
def test_forged_teacher_eligibility_fails_closed(fault):
    replay=sampler();counts=m.FullSupplementCounts(FAMILIES,replay.unknown_source_order)
    rows=replay.batch();pred=[r['target'] for r in rows];mask=[True]*96
    if fault in ('false_mask','wrong_pred','nonint_pred'):
        if fault=='false_mask':mask[0]=False
        elif fault=='wrong_pred':pred[0]=(pred[0]+1)%25
        else:pred[0]=float(pred[0])
        with pytest.raises(ValueError):counts.update(rows,mask,pred)
        return
    counts.update(rows,mask,pred);report=counts.report()
    if fault=='source_count':report['source_target_rows'][0]['eligible_rows']-=1
    elif fault=='eligibility_total':report['eligible_rows']-=1
    else:report['teacher_mask_verified_against_same_tile_argmax']=False
    with pytest.raises(ValueError):m.validate_training_counts(report,FAMILIES,1)


def test_three_source_pixel_teacher_routing_remains_same_tile():
    rows,a,b,c=routing_fixture()
    pixels,teachers,sizes=m.pixel_batch(rows,a,b,c,np.random.default_rng(12))
    np.testing.assert_array_equal(pixels[:,0,0,0],teachers[:,0])
    assert pixels.shape==(96,1,64,256) and teachers.shape==(96,25)


def test_one_plain_cnn_all_parameter_groups_train_with_new_mask():
    torch=pytest.importorskip('torch');torch.set_num_threads(4)
    from training.region_network import RegionFontClassifier
    model=RegionFontClassifier(25);before=m.parameter_groups(model.state_dict());rows=sampler().batch()
    targets=torch.tensor([r['target'] for r in rows]);teacher=torch.full((96,25),-2.)
    teacher[torch.arange(96),targets]=2.
    logits,ratios=model(torch.rand(96,1,64,256));sizes=torch.full((96,),.2)
    loss,*_=m.full_losses(logits,ratios,targets,sizes,teacher,rows,FAMILIES)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5);optimizer.zero_grad();loss.backward();optimizer.step()
    after=m.parameter_groups(model.state_dict());assert all(after[k]!=v for k,v in before.items())
    assert not any('residual' in k or 'teacher' in k for k in model.state_dict())


def test_only_predeclared_teacher_mask_and_seed_change_supervised_math_stays_fixed():
    assert m.SAMPLING==old.SAMPLING and m.FIXED_RUNTIME==old.FIXED_RUNTIME
    assert {k:v for k,v in m.OBJECTIVE.items() if k not in ('teacher_mask','teacher_kl_reduction')}=={
        k:v for k,v in old.OBJECTIVE.items() if k not in ('teacher_mask','teacher_kl_reduction')}
    assert (m.STEPS,m.EVAL_EVERY,m.LEARNING_RATE,m.MINIMUM_LEARNING_RATE,m.SEED)==(3000,500,2e-5,2e-6,2026091412)
    args=m.parser().parse_args([])
    assert args.output.name=='run-teacher-preserve-v1' and args.teacher_cache.name=='teacher-preserve-cache-v1'
    assert m.STUDENT_CHECKPOINT_SHA==old.STUDENT_CHECKPOINT_SHA


def teacher_data_fixture(tmp_path):
    import json
    from training.cache_unified_student_teacher import PARTITIONS,ORDER
    teacher={'checkpoint':{'sha256':m.STUDENT_CHECKPOINT_SHA},'selection':{'sha256':m.STUDENT_SELECTION_SHA}}
    datasets={};parts={}
    for name in PARTITIONS:
        directory=tmp_path/name;directory.mkdir();path=directory/'tiles.raw'
        tiles=np.memmap(path,mode='w+',dtype=np.float32,shape=(2,1,64,256));tiles[:]=.25;tiles.flush()
        rows=[{'source_id':name,'region_id':str(i)} for i in range(2)]
        rows_path=directory/'rows.json';rows_path.write_text(json.dumps(rows))
        parts[name]={'split':'train','data_manifest':{'sha256':'a'*64},'partition_manifest':{'sha256':'b'*64},
            'rows':{'path':str(rows_path),'sha256':m.sha(rows_path)},'tiles':{'path':str(path),'sha256':m.sha(path)},
            'region_count':2,'tile_count':2,'shape':[2,1,64,256],'order':ORDER}
        datasets[name]={'tiles':tiles,'rows':rows,'families':FAMILIES,'manifest_sha256':'a'*64,'partition_sha256':'b'*64,
            'partition':{'split':'train','metadata':{'sha256':m.sha(rows_path)},'array':{'sha256':m.sha(path)}}}
    identity={'partitions':parts,'teacher':teacher,'architecture':m.ARCHITECTURE,'families':FAMILIES,
        'teacher_logits_recomputed':True,'historical_logits_reused':False,'optimizer_steps_executed':0,
        'teacher_optimizer_steps_executed':0,'test_read':False,'development_holdout_read':False,
        'calibration_images_read':False,'total_tiles':6}
    return identity,datasets,teacher


def test_teacher_identity_matches_all_three_actual_train_arrays(tmp_path):
    identity,datasets,teacher=teacher_data_fixture(tmp_path)
    m.validate_teacher_data(identity,datasets,teacher)


@pytest.mark.parametrize('fault',['teacher','historic_logits','split','order','rows','image_file','manifest','partition','rowhash','tilehash','families'])
def test_teacher_cannot_be_relabelled_or_routed_to_different_pixels(tmp_path,fault):
    identity,datasets,teacher=teacher_data_fixture(tmp_path)
    identity=deepcopy(identity)
    if fault=='teacher':identity['teacher']['checkpoint']['sha256']=m.BASE_CHECKPOINT_SHA
    elif fault=='historic_logits':identity['historical_logits_reused']=True
    elif fault=='split':identity['partitions']['known']['split']='calibration'
    elif fault=='order':identity['partitions']['known']['order']='shuffled'
    elif fault=='rows':datasets['known']['rows']=list(reversed(datasets['known']['rows']))
    elif fault=='image_file':identity['partitions']['known']['tiles']['path']=str(tmp_path/'different.raw')
    elif fault=='manifest':identity['partitions']['known']['data_manifest']['sha256']='c'*64
    elif fault=='partition':identity['partitions']['known']['partition_manifest']['sha256']='c'*64
    elif fault=='rowhash':identity['partitions']['known']['rows']['sha256']='c'*64
    elif fault=='tilehash':identity['partitions']['known']['tiles']['sha256']='c'*64
    else:identity['families']=list(reversed(FAMILIES))
    with pytest.raises(ValueError):m.validate_teacher_data(identity,datasets,teacher)
