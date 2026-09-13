"""Synthetic post-selection development checks, with sealed TEST paths absent."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from training import evaluate_unified_regions as module
from training.train_regions import dump

FAMILIES=[f'Font {i}' for i in range(24)]+[module.UNKNOWN]
GATES={'min_score':.5,'min_margin':.01,'min_patch_agreement':2/3}


@pytest.fixture
def fixture(monkeypatch,tmp_path):
    args=SimpleNamespace(**{key:tmp_path/key for key in ('run','data','region')},output=tmp_path/'DEVELOPMENT.json')
    for key in ('run','data','region'):getattr(args,key).mkdir()
    (args.run/'model.pth').write_bytes(b'mocked checkpoint')
    (args.region/'model.onnx').write_bytes(b'mocked single ONNX')
    dump(args.data/'MANIFEST.json',{'fixture':True})
    selection={'families':FAMILIES,'data_manifest_sha256':module.sha(args.data/'MANIFEST.json'),'passed':False,
        'selected':{'temperature':1.,'gates':GATES,'metrics':{'experimental_precision_priority_met':False}}}
    dump(args.run/'SELECTION.json',selection)
    meta={'temperature':1.,'gates':GATES,'max_size_relative_spread':.2,
        'training':{'selection_sha256':module.sha(args.run/'SELECTION.json'),
            'checkpoint_sha256':module.sha(args.run/'model.pth'),'data_manifest_sha256':selection['data_manifest_sha256']}}
    dump(args.region/'metadata.json',meta)
    evidence={str(args.run/'SELECTION.json'):module.sha(args.run/'SELECTION.json')}
    monkeypatch.setattr(module,'validate',lambda run,data:deepcopy(selection))
    monkeypatch.setattr(module,'calibration_evidence',lambda *args:{path:module.sha(path) for path in evidence})
    parity={'schema':'flux-glyph-unified-onnx-parity-v1','passed':True,'test_read':False,'development_holdout_read':False,
        'network_architecture':module.ARCHITECTURE,'font_model_count':1,'output_family_count':25,
        'selection_sha256':module.sha(args.run/'SELECTION.json'),'checkpoint_sha256':module.sha(args.run/'model.pth'),
        'model_sha256':module.sha(args.region/'model.onnx'),'metadata_sha256':module.sha(args.region/'metadata.json'),
        'source_bindings':module.export_sources(),'calibration_bindings':evidence,
        'calibration_font_decisions_identical':True,'original_torch_groupnorm_reference':True,
        'calibration_size_availability_identical':True,'calibration_size_values_close':True,'calibration_scores_close':True,
        'cached_training_outputs_reproduce_frozen_grid':True,'cached_mps_outputs_reproduce_selected_metrics':True,
        'size_value_atol_px':.02,'size_value_rtol':.0003,
        'batch_checks':[{'batch_size':n,'samples':40,'passed':True,'font_and_size_checked':True} for n in (1,7,32,128)]}
    dump(args.run/'PARITY.json',parity)
    rows=[]
    for domain in ('ios','android'):
        for target,family in enumerate(FAMILIES):
            rows.append({'family':family,'target':target,'domain':domain,'view':'native',
                'source_id':domain+'-page','region_id':str(target),'source_font_family':family,
                'split':'development_holdout','tile_start':len(rows),'tile_count':1,'ink_height_px':40,'font_size_px':40})
    tiles=np.zeros((len(rows),1,64,256),np.float32);tiles[:,0,0,0]=np.arange(len(rows))
    part_dir=args.data/'development_holdout';part_dir.mkdir()
    (part_dir/'tiles.raw').write_bytes(tiles.tobytes());dump(part_dir/'rows.json',rows)
    partition={'split':'development_holdout','rejected':[],
        'array':{'path':'tiles.raw','sha256':module.sha(part_dir/'tiles.raw')},
        'metadata':{'path':'rows.json','sha256':module.sha(part_dir/'rows.json')}}
    dump(part_dir/'MANIFEST.json',partition)
    data={'families':FAMILIES,'manifest_sha256':selection['data_manifest_sha256'],'partition':partition,
        'partition_sha256':module.sha(part_dir/'MANIFEST.json'),'rows':rows,'tiles':tiles,'manifest':{}}
    calls=[];models=[]
    def load(root,split,*,allow_holdout=False):
        assert root==args.data and split=='development_holdout' and allow_holdout is True
        assert (tmp_path/'DEVELOPMENT_FREEZE.json').is_file()
        calls.append(split);return data
    def infer(names,inputs):
        assert names==['logits','log_em_ratio']
        indices=inputs['tiles'][:,0,0,0].astype(int)
        logits=np.full((len(indices),25),-10,np.float32)
        for index,value in enumerate(indices):logits[index,rows[value]['target']]=10
        return [logits,np.zeros(len(indices),np.float32)]
    model=SimpleNamespace(output_families=FAMILIES,meta=meta,session=SimpleNamespace(run=infer))
    def constructor(directory):models.append(directory);return model
    monkeypatch.setattr(module,'load_split',load)
    monkeypatch.setattr('flux_glyph.unified_font.UnifiedFontClassifier',constructor)
    return SimpleNamespace(args=args,selection=selection,meta=meta,parity=parity,data=data,
        model=model,calls=calls,models=models,infer=infer,freeze=tmp_path/'DEVELOPMENT_FREEZE.json')


def test_fixed_dynamic_model_is_evaluated_only_as_development_not_stable_test(fixture):
    f=fixture;result=module.evaluate(f.args)
    assert result['evaluation_kind']=='development_regression' and result['development_policy_passed'] is True
    assert result['stable_validation_passed'] is result['test_passed'] is result['blind_test_performed'] is False
    assert result['test_read'] is False and result['used_to_adjust_model_or_gates'] is False
    assert result['metrics']['views']==50 and len(result['by_family'])==24 and set(result['by_domain'])=={'ios','android'}
    assert result['denominators']['native_regions']==50 and result['denominators']['pages']==2
    assert result['denominators']['unknown_outcomes']=={'wrongly_named':0,'explicit_unknown':2,'uncertain_without_name':0,'preprocessing_rejected':0}
    assert result['model_count']==1 and result['platform_routing'] is False
    assert all(view['passed'] is None and view['diagnostic_only'] for view in result['by_view'].values())
    assert f.calls==['development_holdout'] and len(f.models)==1 and not (f.args.data/'test').exists()
    with pytest.raises(ValueError,match='existing development'):module.evaluate(f.args)


@pytest.mark.parametrize('field,value',[
    ('passed',False),('test_read',True),('development_holdout_read',True),('source_bindings',{}),('calibration_bindings',{}),
    ('font_model_count',2),('output_family_count',10),
    ('calibration_scores_close',False),('calibration_size_availability_identical',False),('calibration_size_values_close',False),
    ('cached_training_outputs_reproduce_frozen_grid',False),('size_value_atol_px',1.),('batch_checks',[]),
])
def test_missing_or_changed_export_evidence_blocks_before_model_and_holdout(fixture,field,value):
    f=fixture;f.parity[field]=value;dump(f.args.run/'PARITY.json',f.parity)
    with pytest.raises(ValueError):module.evaluate(f.args)
    assert f.calls==f.models==[] and not f.freeze.exists()


@pytest.mark.parametrize('asset',['checkpoint','onnx','metadata'])
def test_changed_model_bytes_block_before_holdout(fixture,asset):
    f=fixture;path={'checkpoint':f.args.run/'model.pth','onnx':f.args.region/'model.onnx','metadata':f.args.region/'metadata.json'}[asset]
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError):module.evaluate(f.args)
    assert f.calls==[]


@pytest.mark.parametrize('change',[{'temperature':2.},{'gates':{**GATES,'min_score':.1}},{'training':{}}])
def test_runtime_gate_or_lineage_change_cannot_read_holdout(fixture,change):
    f=fixture;f.model.meta={**deepcopy(f.meta),**change}
    with pytest.raises(ValueError):module.evaluate(f.args)
    assert not f.calls and not f.freeze.exists()


@pytest.mark.parametrize('pathkey',['parity','checkpoint','selection','freeze','array','partition'])
def test_mutation_during_inference_never_publishes_a_report(fixture,pathkey):
    f=fixture;path={'parity':f.args.run/'PARITY.json','checkpoint':f.args.run/'model.pth',
        'selection':f.args.run/'SELECTION.json','freeze':f.freeze,'array':f.args.data/'development_holdout/tiles.raw',
        'partition':f.args.data/'development_holdout/MANIFEST.json'}[pathkey]
    def infer(names,inputs):
        out=f.infer(names,inputs);path.write_bytes(path.read_bytes()+b'\n');return out
    f.model.session.run=infer
    with pytest.raises(ValueError):module.evaluate(f.args)
    assert not f.args.output.exists() and f.freeze.exists()


def test_failure_preserves_freeze_and_prevents_silent_repeat(fixture):
    f=fixture
    def fail(*args):raise RuntimeError('simulated ORT failure')
    f.model.session.run=fail
    with pytest.raises(RuntimeError):module.evaluate(f.args)
    with pytest.raises(ValueError,match='already started'):module.evaluate(f.args)
    assert f.calls==['development_holdout'] and not f.args.output.exists()


def test_preprocessing_and_unknown_error_denominators_are_not_hidden(fixture):
    f=fixture;rows=[f.data['rows'][0],f.data['rows'][24]]
    logits=np.full((2,25),-10,np.float32);logits[:,0]=10
    adjusted=[{**r,'tile_start':i} for i,r in enumerate(rows)]
    outputs=module.region_outputs(logits,np.zeros(2,np.float32),adjusted,1.)
    details=module.decisions(outputs,adjusted,FAMILIES,GATES)
    rejected=[{**rows[0],'view':'half'},{**rows[1],'view':'half'}]
    measured,_,confusion,denom=module.summarize(details,FAMILIES,rejected)
    assert measured['known_correct_coverage']==.5 and measured['named_precision']==.5
    assert measured['unknown_not_named_rate']==.5 and measured['explicit_unknown_recall']==0
    assert denom['unknown_outcomes']=={'wrongly_named':1,'explicit_unknown':0,'uncertain_without_name':0,'preprocessing_rejected':1}
    assert denom['size_available_on_wrong_names']==1 and denom['size_correct_coverage_of_all_known_views']==.5
    assert sum(sum(v.values()) for v in confusion.values())==4
