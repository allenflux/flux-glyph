"""Sampling and provenance checks use synthetic metadata; no training or TEST input."""
from collections import Counter
from copy import deepcopy
import hashlib
import math

import numpy as np
import pytest

from training import train_android_balanced as balanced

FAMILIES=[f'Known {i}' for i in range(9)]+[balanced.UNKNOWN]


def synthetic_rows():
    rows=[]
    for target,family in enumerate(FAMILIES):
        sources=[family] if target<9 else ['Unknown A','Unknown B','Unknown C']
        for source in sources:
            faces=range(4) if source=='Unknown C' else range(1)
            for face in faces:
                for view in balanced.RECIPES:
                    # Deliberately skew face and family region/tile counts.
                    for repeat in range(8 if source=='Unknown C' and face==0 else 1):
                        rows.append({'split':'train','family':family,'target':target,'source_font_family':source,
                            'font_file_sha256':hashlib.sha256(f'{source}-{face}'.encode()).hexdigest(),'ttc_index':0,
                            'view':view,'tile_start':len(rows)*8,'tile_count':8 if repeat else 1})
    return rows


def test_exact_half_known_unknown_and_uniform_classes_families_faces():
    sampler=balanced.BalancedSampler(synthetic_rows(),FAMILIES,73)
    for _ in range(900):
        rows=sampler.batch()
        assert len(rows)==64 and sum(r['target']==9 for r in rows)==32
    report=sampler.report()
    assert report['sampled_rows']=={'known':28800,'unknown':28800}
    assert set(report['known_class_rows'].values())=={3200}
    assert set(report['unknown_source_family_rows'].values())=={9600}
    cfaces=[r['rows'] for r in report['unknown_source_face_rows'] if r['family']=='Unknown C']
    assert cfaces==[2400]*4
    for kind in ('known','unknown'):
        for view,probability in balanced.VIEW_PROBABILITIES.items():
            assert abs(report['view_rows'][kind][view]/28800-probability)<.012


def test_sampling_is_seeded_and_does_not_mutate_input_metadata():
    rows=synthetic_rows();before=deepcopy(rows)
    a=balanced.BalancedSampler(rows,FAMILIES,73);b=balanced.BalancedSampler(rows,FAMILIES,73)
    assert a.batch()==b.batch();assert a.batch()==b.batch();assert rows==before


@pytest.mark.parametrize('problem',['calibration','bad_target','missing_source','missing_face_view','missing_known'])
def test_sampling_rejects_wrong_split_or_incomplete_source_strata(problem):
    rows=synthetic_rows()
    if problem=='calibration':rows[0]['split']='calibration'
    elif problem=='bad_target':rows[0]['target']=9
    elif problem=='missing_source':next(r for r in rows if r['target']==9).pop('source_font_family')
    elif problem=='missing_face_view':rows=[r for r in rows if not(r['source_font_family']=='Unknown A' and r['view']=='half')]
    elif problem=='missing_known':rows=[r for r in rows if r['target']!=2]
    with pytest.raises(ValueError):balanced.BalancedSampler(rows,FAMILIES,1)


def initialization():
    state={'trunk.weight':np.ones((4,1,3,3)), 'size_head.weight':np.ones((1,128)),
        'size_head.bias':np.ones(1),'family_head.weight':np.ones((10,128)),'family_head.bias':np.ones(10)}
    checkpoint={'families':FAMILIES,'state_dict':deepcopy(state),'selection_sha256':'s'*64}
    selection={'schema':'flux-glyph-android-training-selection-v1','families':FAMILIES,'test_read':False,
        'data_manifest_sha256':'d'*64,'training_protocol_sha256':'p'*64,'state_after_sha256':'a'*64,
        'state_before_sha256':'b'*64,'optimizer_steps_executed':10000,'bindings':{},'calibration_passed':False}
    protocol={'schema':'flux-glyph-android-training-protocol-v1','families':FAMILIES,'test_read':False,
        'steps':10000,'bindings':{},'initial_state_sha256':'b'*64}
    return state,checkpoint,selection,protocol


def test_failed_calibration_initializer_is_allowed_but_complete_ten_class_state_is_required():
    state,checkpoint,selection,protocol=initialization()
    balanced.validate_initializer(state,checkpoint,FAMILIES,selection,protocol,'s'*64,'p'*64,'d'*64,'a'*64)


@pytest.mark.parametrize('problem',['nine_classes','reordered','missing_tensor','checkpoint_hash','protocol_hash',
    'state_hash','different_data','test_used','zero_steps','no_updates','protocol_initial_state'])
def test_initializer_identity_chain_cannot_be_silently_bypassed(problem):
    state,checkpoint,selection,protocol=initialization()
    if problem=='nine_classes':
        checkpoint['families']=FAMILIES[:9]
        checkpoint['state_dict']['family_head.weight']=np.ones((9,128));checkpoint['state_dict']['family_head.bias']=np.ones(9)
    elif problem=='reordered':checkpoint['families']=list(reversed(FAMILIES))
    elif problem=='missing_tensor':checkpoint['state_dict'].pop('size_head.weight')
    elif problem=='checkpoint_hash':checkpoint['selection_sha256']='x'*64
    elif problem=='protocol_hash':selection['training_protocol_sha256']='x'*64
    elif problem=='state_hash':selection['state_after_sha256']='x'*64
    elif problem=='different_data':selection['data_manifest_sha256']='x'*64
    elif problem=='test_used':selection['test_read']=True
    elif problem=='zero_steps':selection['optimizer_steps_executed']=protocol['steps']=0
    elif problem=='no_updates':selection['state_before_sha256']='a'*64
    elif problem=='protocol_initial_state':protocol['initial_state_sha256']='x'*64
    with pytest.raises(ValueError):balanced.validate_initializer(state,checkpoint,FAMILIES,selection,protocol,'s'*64,'p'*64,'d'*64,'a'*64)


def test_additional_balanced_provenance_is_accepted_and_verified_by_frozen_export(monkeypatch,tmp_path):
    from test_android_export import frozen_run
    import export_android_regions as export
    run,data,selection,protocol=frozen_run(monkeypatch,tmp_path)
    trainer=tmp_path/'new_trainer.py';trainer.write_text('balanced trainer fixture')
    initializer=tmp_path/'initializer.pth';initializer.write_text('ten-class initializer fixture')
    for path in (trainer,initializer):selection['bindings'][str(path)]=export.sha(path)
    protocol['bindings']=selection['bindings'];protocol['sampling']=balanced.SAMPLING;protocol['objective']=balanced.OBJECTIVE
    export.dump(run/'TRAINING_FREEZE.json',protocol);selection['training_protocol_sha256']=export.sha(run/'TRAINING_FREEZE.json')
    export.dump(run/'SELECTION.json',selection)
    assert export.validate(run,data)==selection
    trainer.write_text('changed new trainer')
    with pytest.raises(ValueError,match='frozen Android training source changed'):export.validate(run,data)
    assert not (data/'test').exists()


def test_uniform_ce_has_lower_loss_than_a_known_class_spike_on_unknown_inputs():
    torch=pytest.importorskip('torch')
    targets=torch.tensor([9,0]);uniform=torch.zeros((2,10),requires_grad=True)
    loss=balanced.unknown_uniform_ce(uniform,targets,9)
    assert loss.item()==pytest.approx(math.log(9),rel=1e-6)
    spiked=torch.zeros((2,10));spiked[0,2]=9.;spiked.requires_grad_(True)
    peaked=balanced.unknown_uniform_ce(spiked,targets,9)
    assert peaked.item()>loss.item()
    peaked.backward()
    assert spiked.grad[0,2]>0 and spiked.grad[0,0]<0
    assert spiked.grad[0,9]==0 and torch.count_nonzero(spiked.grad[1])==0
    assert abs(float(spiked.grad[0,:9].sum()))<1e-6


def test_uniform_ce_is_shift_invariant_and_handles_no_unknown_rows_with_zero_gradient():
    torch=pytest.importorskip('torch')
    logits=torch.arange(30,dtype=torch.float32).reshape(3,10).requires_grad_(True)
    targets=torch.tensor([0,1,9])
    a=balanced.unknown_uniform_ce(logits,targets,9)
    shifted=logits.detach().clone();shifted[2,:9]+=100
    assert balanced.unknown_uniform_ce(shifted,targets,9).item()==pytest.approx(a.item(),abs=1e-5)
    no_unknown=balanced.unknown_uniform_ce(logits,torch.tensor([0,1,2]),9)
    no_unknown.backward();assert no_unknown.item()==0 and torch.count_nonzero(logits.grad)==0


def test_defaults_record_fixed_recipe_and_reuse_original_policy():
    from training.train_android_regions import POLICY
    args=balanced.parser().parse_args(['--data','data','--output','new-run'])
    assert args.steps==6000 and args.learning_rate==.0001 and args.checkpoint.name=='model.pth'
    assert balanced.POLICY==POLICY and balanced.OBJECTIVE['unknown_known_head_uniform_ce_weight']==.5
    assert balanced.SAMPLING['known_fraction']==balanced.SAMPLING['unknown_fraction']==.5
