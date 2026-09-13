"""CAL export provenance and decision parity; no model training or TEST data."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
import export_android_regions as module
from train_android_regions import POLICY,region_outputs
from test_android_training import FAMILIES,GATES,row


def outputs(em=1.,score=8.):
    logits=np.zeros((1,10),dtype=np.float32);logits[0,0]=score
    return region_outputs(logits,np.full(1,np.log(em),dtype=np.float32),[row()],1.)


def test_size_availability_must_match_even_when_font_decision_is_identical():
    actual=outputs();expected=copy.deepcopy(actual)
    actual[0]['size_spread']=.19999;expected[0]['size_spread']=.20001
    with pytest.raises(ValueError,match='size availability'):
        module.compare_calibration_outputs(actual,expected,[row()],FAMILIES,GATES)


def test_size_values_are_compared_for_each_region_in_source_pixels():
    difference=module.compare_calibration_outputs(outputs(1.0001),outputs(),[row()],FAMILIES,GATES)
    assert difference==0  # Both runtime estimates round to 40.00 px.
    assert module.compare_calibration_outputs(outputs(1.0003),outputs(),[row()],FAMILIES,GATES)==pytest.approx(.01)
    with pytest.raises(AssertionError):
        module.compare_calibration_outputs(outputs(1.05),outputs(),[row()],FAMILIES,GATES)


def test_font_threshold_crossing_fails_even_for_numerically_close_scores():
    actual=outputs();expected=copy.deepcopy(actual)
    actual[0]['score']=.7000001;expected[0]['score']=.6999999
    with pytest.raises(ValueError,match='naming decision'):
        module.compare_calibration_outputs(actual,expected,[row()],FAMILIES,{**GATES,'min_score':.7})


def test_unknown_cannot_change_to_known_or_gain_size_during_export():
    actual=outputs();expected=copy.deepcopy(actual)
    expected[0]['predicted']=9
    with pytest.raises(ValueError,match='naming decision'):
        module.compare_calibration_outputs(actual,expected,[row()],FAMILIES,GATES)
    actual[0]['predicted']=9
    assert module.compare_calibration_outputs(actual,expected,[row()],FAMILIES,GATES)==0.


def frozen_run(monkeypatch,tmp_path,steps=1000):
    root=tmp_path/'repo';data=root/'data';run=root/'run';run.mkdir(parents=True)
    monkeypatch.setattr(module,'ROOT',root)
    names=['data/MANIFEST.json','data/train/MANIFEST.json','data/calibration/MANIFEST.json',
           'training/train_android_regions.py','training/prepare_android_regions.py','src/flux_glyph/android_font.py',
           'src/flux_glyph/region_font.py','training/region_network.py','training/network.py','training/train_regions.py']
    bindings={}
    for name in names:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
        bindings[str(path.resolve())]=module.sha(path)
    protocol={'policy':POLICY,'bindings':bindings,'families':FAMILIES,'steps':steps,'test_read':False,'initial_state_sha256':'a'*64}
    module.dump(run/'TRAINING_FREEZE.json',protocol)
    module.dump(run/'CALIBRATION_DECISIONS.json',{'records':[]})
    (run/'CALIBRATION_OUTPUTS.npz').write_bytes(b'hash fixture only; not opened by validation')
    measured={'passed':True,'checks':{'all':True},'correct_named':9,'wrong_named':0,'size':{'median_ape':.01}}
    history=[{'step':step,'grid_index':0,'temperature':.5,'gates':GATES,'calibration_nll':.01,'metrics':copy.deepcopy(measured)}
             for step in range(1000,steps+1,1000)]
    if steps%1000:history.append({**copy.deepcopy(history[0]),'step':steps})
    selection={'schema':'flux-glyph-android-training-selection-v1','passed':True,'calibration_passed':True,'test_read':False,
               'policy':POLICY,'bindings':bindings,'families':FAMILIES,'optimizer_steps_executed':steps,
               'state_before_sha256':'a'*64,'state_after_sha256':'b'*64,'history':history,'selected':copy.deepcopy(history[0]),
               'data_manifest_sha256':module.sha(data/'MANIFEST.json'),'training_protocol_sha256':module.sha(run/'TRAINING_FREEZE.json'),
               'calibration_decisions_sha256':module.sha(run/'CALIBRATION_DECISIONS.json'),
               'calibration_outputs_sha256':module.sha(run/'CALIBRATION_OUTPUTS.npz')}
    module.dump(run/'SELECTION.json',selection)
    return run,data,selection,protocol


def test_validate_binds_full_encoder_sources_and_reads_no_test(monkeypatch,tmp_path):
    run,data,selection,_=frozen_run(monkeypatch,tmp_path,1500)
    assert module.validate(run,data)==selection
    assert not (data/'test').exists()
    (module.ROOT/'training/network.py').write_text('changed encoder')
    with pytest.raises(ValueError,match='differs from training'):
        module.validate(run,data)


@pytest.mark.parametrize('field,change',[
    ('test_read',True),('calibration_passed',False),('state_before_sha256','b'*64),
    ('training_protocol_sha256','0'*64),('calibration_decisions_sha256','0'*64),
    ('calibration_outputs_sha256','0'*64),('data_manifest_sha256','0'*64),
])
def test_failed_or_changed_training_evidence_is_not_exportable(monkeypatch,tmp_path,field,change):
    run,data,selection,_=frozen_run(monkeypatch,tmp_path)
    selection[field]=change;module.dump(run/'SELECTION.json',selection)
    with pytest.raises(ValueError):module.validate(run,data)


def test_even_rehashed_protocol_cannot_claim_zero_optimizer_steps(monkeypatch,tmp_path):
    run,data,selection,protocol=frozen_run(monkeypatch,tmp_path)
    protocol['steps']=selection['optimizer_steps_executed']=0
    module.dump(run/'TRAINING_FREEZE.json',protocol)
    selection['training_protocol_sha256']=module.sha(run/'TRAINING_FREEZE.json')
    module.dump(run/'SELECTION.json',selection)
    with pytest.raises(ValueError,match='optimizer step count'):module.validate(run,data)


@pytest.mark.parametrize('problem',['incomplete_history','off_grid','wrong_grid_index','later_equal_checkpoint','nan_nll'])
def test_selection_must_be_frozen_grid_history_winner(monkeypatch,tmp_path,problem):
    run,data,selection,_=frozen_run(monkeypatch,tmp_path,2000)
    if problem=='incomplete_history':selection['history'].pop()
    elif problem=='off_grid':selection['history'][0]['gates']={**GATES,'min_score':.6}
    elif problem=='wrong_grid_index':selection['history'][0]['grid_index']=1
    elif problem=='later_equal_checkpoint':selection['selected']=copy.deepcopy(selection['history'][1])
    elif problem=='nan_nll':selection['history'][1]['calibration_nll']=float('nan')
    (run/'SELECTION.json').write_text(json.dumps(selection))
    with pytest.raises(ValueError):module.validate(run,data)
