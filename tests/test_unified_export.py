"""Unified export uses synthetic CAL provenance; no training or holdout pixels."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from training import export_unified_regions as module
from training.train_unified_regions import metrics

FAMILIES=[f'Font {i}' for i in range(24)]+[module.UNKNOWN]
GATES={'min_score':.5,'min_margin':.01,'min_patch_agreement':2/3}


def row(target=0):
    return {'tile_start':0,'tile_count':1,'target':target,'family':FAMILIES[target],
        'domain':'ios','view':'native','source_id':'cal-page','region_id':'cal-row',
        'source_font_family':FAMILIES[target],'ink_height_px':40,'font_size_px':40}


def outputs(target=0,em=1.):
    logits=np.zeros((1,25),np.float32);logits[0,target]=8
    return module.region_outputs(logits,np.full(1,np.log(em),np.float32),[row()],1.)


def test_dynamic_25_class_probabilities_scores_and_size_match():
    actual=outputs();assert len(actual[0]['probabilities'])==25
    assert module.compare_calibration_outputs(actual,copy.deepcopy(actual),[row()],FAMILIES,GATES)==0
    assert module.compare_calibration_outputs(outputs(em=1.0003),actual,[row()],FAMILIES,GATES)==pytest.approx(.01)
    with pytest.raises(AssertionError):module.compare_calibration_outputs(outputs(em=1.05),actual,[row()],FAMILIES,GATES)


@pytest.mark.parametrize('field', ['score','margin','agreement','probabilities'])
def test_numerical_score_drift_is_rejected_even_without_a_naming_change(field):
    actual=outputs();other=copy.deepcopy(actual)
    if field=='probabilities':other[0][field][0]-=.01
    else:other[0][field]-=.01
    with pytest.raises(AssertionError):module.compare_calibration_outputs(actual,other,[row()],FAMILIES,GATES)


@pytest.mark.parametrize('fault',['threshold','unknown','size_available'])
def test_near_threshold_font_unknown_and_size_decisions_must_match(fault):
    actual=outputs();other=copy.deepcopy(actual)
    if fault=='threshold':actual[0]['score']=.50000001;other[0]['score']=.49999999
    elif fault=='unknown':other[0]['predicted']=24
    else:actual[0]['size_spread']=.199999;other[0]['size_spread']=.200001
    with pytest.raises(ValueError):module.compare_calibration_outputs(actual,other,[row()],FAMILIES,GATES)


def measured_fixture():
    details=[]
    for i in range(100):
        correct=i<96
        details.append({'family':FAMILIES[0],'domain':'ios' if i%2 else 'android','view':'native',
            'source_font_family':FAMILIES[0],'named':True,'correct_named':correct,'wrong_named':not correct,
            'top1_correct':correct,'explicit_unknown':False,'size_ape':.01 if correct else None})
    for domain in ('ios','android'):
        details.append({'family':module.UNKNOWN,'domain':domain,'view':'native','source_font_family':'Unknown',
            'named':False,'correct_named':False,'wrong_named':False,'top1_correct':True,'explicit_unknown':True,'size_ape':None})
    return metrics(details,FAMILIES)


def frozen_run(monkeypatch,tmp_path):
    root=(tmp_path/'repo').resolve();run=root/'run';data=root/'data';run.mkdir(parents=True)
    monkeypatch.setattr(module,'ROOT',root)
    names=['data/MANIFEST.json','data/train/MANIFEST.json','data/calibration/MANIFEST.json',
        'training/train_unified_regions.py','training/prepare_unified_regions.py','training/train_android_regions.py',
        'src/flux_glyph/unified_font.py','src/flux_glyph/region_font.py','training/region_network.py',
        'training/network.py','training/train_regions.py','parent/model.pth','parent/SELECTION.json']
    bindings={}
    for name in names:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
        bindings[str(path)]=module.sha(path)
    parent=root/'parent/model.pth'
    inheritance={'checkpoint':{'path':str(parent),'sha256':module.sha(parent)},
        'selection_sha256':module.sha(parent.parent/'SELECTION.json'),'mapped_family_rows':[],
        **dict.fromkeys(('encoder_inherited_in_full','size_head_inherited_in_full','all_parameters_trainable','single_parent_encoder'),True)}
    groups=dict.fromkeys(('trunk','style','family_head','size_head'),'a'*64)
    final=dict.fromkeys(groups,'b'*64)
    protocol={'schema':'flux-glyph-unified-training-protocol-v1','architecture':module.ARCHITECTURE,
        'policy':module.POLICY,'bindings':bindings,'families':FAMILIES,'steps':module.STEPS,
        'test_read':False,'development_holdout_read':False,'model_count':1,'platform_routing':False,'score_merging':False,
        'initial_state_sha256':'a'*64,'initial_parameter_groups_sha256':groups,'initializer_evidence':inheritance,
        'training_inputs':['image_tiles'],'batch_size':module.BATCH_SIZE,'learning_rate':module.LEARNING_RATE,
        'sampling':module.SAMPLING,'objective':module.OBJECTIVE,'device':'mps'}
    module.dump(run/'TRAINING_FREEZE.json',protocol)
    module.dump(run/'CALIBRATION_DECISIONS.json',{'families':FAMILIES,'records':[]})
    (run/'CALIBRATION_OUTPUTS.npz').write_bytes(b'hash fixture; not opened by validate')
    module.dump(run/'SAMPLING.json',{'known_rows':module.STEPS*77})
    measured=measured_fixture();history=[]
    for step in range(1000,module.STEPS+1,1000):
        grid=[]
        for temperature in module.POLICY['temperatures']:
            for score in module.POLICY['score_gates']:
                for margin in module.POLICY['margin_gates']:
                    for agreement in module.POLICY['agreement_gates']:
                        grid.append({'step':step,'grid_index':len(grid),'temperature':temperature,
                            'gates':{'min_score':score,'min_margin':margin,'min_patch_agreement':agreement},
                            'calibration_nll':.1,'metrics':copy.deepcopy(measured)})
        module.dump(run/f'CAL_GRID_{step:05d}.json',grid);history.append(copy.deepcopy(grid[0]))
    selection={'schema':'flux-glyph-unified-training-selection-v1','architecture':module.ARCHITECTURE,
        'policy':module.POLICY,'bindings':bindings,'families':FAMILIES,'optimizer_steps_executed':module.STEPS,
        'passed':False,'calibration_passed':False,'test_read':False,'development_holdout_read':False,
        'model_count':1,'platform_routing':False,'score_merging':False,'training_device':'mps',
        'state_before_sha256':'a'*64,'state_after_sha256':'b'*64,'initial_parameter_groups_sha256':groups,
        'final_parameter_groups_sha256':final,'initializer_evidence':inheritance,
        'history':history,'selected':copy.deepcopy(history[0]),'data_manifest_sha256':module.sha(data/'MANIFEST.json'),
        'training_protocol_sha256':module.sha(run/'TRAINING_FREEZE.json'),
        'calibration_decisions_sha256':module.sha(run/'CALIBRATION_DECISIONS.json'),
        'calibration_outputs_sha256':module.sha(run/'CALIBRATION_OUTPUTS.npz'),'sampling_sha256':module.sha(run/'SAMPLING.json')}
    module.dump(run/'SELECTION.json',selection)
    return run,data,selection,protocol


def test_experimental_95_preference_does_not_require_or_claim_stable_98_pass(monkeypatch,tmp_path):
    run,data,selection,_=frozen_run(monkeypatch,tmp_path)
    validated=module.validate(run,data)
    assert validated==selection and validated['selected']['metrics']['named_precision']==.96
    assert validated['selected']['metrics']['experimental_precision_priority_met'] is True
    assert validated['passed'] is validated['calibration_passed'] is False
    assert not (data/'development_holdout').exists() and not (data/'test').exists()


@pytest.mark.parametrize('field,value',[
    ('test_read',True),('development_holdout_read',True),('model_count',2),('platform_routing',True),
    ('score_merging',True),('architecture','dual-font-model'),('passed',True),('calibration_passed',True),
    ('state_before_sha256','b'*64),('sampling_sha256','0'*64),('calibration_outputs_sha256','0'*64),
])
def test_changed_training_or_false_validation_claim_is_rejected(monkeypatch,tmp_path,field,value):
    run,data,selection,_=frozen_run(monkeypatch,tmp_path)
    selection[field]=value;module.dump(run/'SELECTION.json',selection)
    with pytest.raises(ValueError):module.validate(run,data)


@pytest.mark.parametrize('problem',['partial_steps','changed_group','missing_source','changed_initializer','missing_grid','nan_nll','off_grid','wrong_winner','wrong_priority'])
def test_full_provenance_and_every_fixed_cal_grid_are_checked(monkeypatch,tmp_path,problem):
    run,data,selection,protocol=frozen_run(monkeypatch,tmp_path)
    if problem=='partial_steps':protocol['steps']=selection['optimizer_steps_executed']=1000
    elif problem=='changed_group':selection['final_parameter_groups_sha256']['size_head']='a'*64
    elif problem=='missing_source':selection['bindings'].pop(str(module.ROOT/'training/train_android_regions.py'))
    elif problem=='changed_initializer':selection['initializer_evidence']['checkpoint']['sha256']='0'*64
    else:
        path=run/'CAL_GRID_01000.json';grid=json.loads(path.read_text())
        if problem=='missing_grid':grid.pop()
        elif problem=='nan_nll':grid[-1]['calibration_nll']=float('nan')
        elif problem=='off_grid':grid[-1]['gates']['min_score']=.999
        elif problem=='wrong_winner':grid[-1]['metrics']['correct_named']+=1
        elif problem=='wrong_priority':grid[-1]['metrics']['experimental_precision_priority_met']=False
        path.write_text(json.dumps(grid))
    module.dump(run/'TRAINING_FREEZE.json',protocol)
    selection['training_protocol_sha256']=module.sha(run/'TRAINING_FREEZE.json')
    module.dump(run/'SELECTION.json',selection)
    with pytest.raises(ValueError):module.validate(run,data)


def test_checkpoint_architecture_and_parameter_group_sha_are_required(monkeypatch):
    monkeypatch.setattr(module,'state_sha',lambda state:'state' if len(state)==4 else next(iter(state.values())))
    state={name+'.weight':name for name in ('trunk','style','family_head','size_head')}
    selected={'families':FAMILIES,'state_after_sha256':'state','final_parameter_groups_sha256':{name:name for name in ('trunk','style','family_head','size_head')}}
    checkpoint={'families':FAMILIES,'architecture':module.ARCHITECTURE,'state_dict':state,'selection_sha256':'selection'}
    module.validate_checkpoint(checkpoint,selected,'selection')
    checkpoint['architecture']='other'
    with pytest.raises(ValueError):module.validate_checkpoint(checkpoint,selected,'selection')
    checkpoint['architecture']=module.ARCHITECTURE;checkpoint['state_dict']['size_head.weight']='tampered'
    with pytest.raises(ValueError,match='parameter group'):module.validate_checkpoint(checkpoint,selected,'selection')
