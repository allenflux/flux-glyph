"""OVR provenance and parity boundaries, using synthetic evidence only."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
import export_android_ovr as exporter
import evaluate_android_ovr as evaluator
from test_android_export import frozen_run


@pytest.fixture
def run_fixture(monkeypatch,tmp_path):
    run,data,selection,protocol=frozen_run(monkeypatch,tmp_path,4000)
    root=run.parent
    monkeypatch.setattr(exporter,'ROOT',root)
    monkeypatch.setattr(evaluator,'ROOT',root)
    for name in ('training/train_android_ovr.py','training/android_ovr_network.py','parent/model.pth','cache/MANIFEST.json','run/INITIAL_HEADS.pth'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
        selection['bindings'][str(path)]=exporter.sha(path)
    extra={'architecture':exporter.ARCHITECTURE,
        'parent_checkpoint':{'path':str(root/'parent/model.pth'),'sha256':exporter.sha(root/'parent/model.pth')},
        'cache_manifest':{'path':str(root/'cache/MANIFEST.json'),'sha256':exporter.sha(root/'cache/MANIFEST.json')},
        'parent_frozen_state_sha256':'c'*64,'heads_before_sha256':'d'*64,
        'cached_output_execution':{'features_device':'mps','heads_device':'cpu','size_head_device':'mps'}}
    selection.update(copy.deepcopy(extra),heads_after_sha256='e'*64)
    protocol.update(copy.deepcopy(extra))
    selection['training_device']='cpu';protocol['device']='cpu'
    def save():
        exporter.dump(run/'TRAINING_FREEZE.json',protocol)
        selection['training_protocol_sha256']=exporter.sha(run/'TRAINING_FREEZE.json')
        exporter.dump(run/'SELECTION.json',selection)
    save()
    return SimpleNamespace(run=run,data=data,selection=selection,protocol=protocol,save=save)


def test_new_architecture_retains_frozen_policy_and_has_no_test_input(run_fixture):
    f=run_fixture
    assert exporter.validate(f.run,f.data)==f.selection
    assert not (f.data/'test').exists()


@pytest.mark.parametrize('change',['architecture','unknown_source','parent','cache','unchanged_heads','parent_state','heads_before','execution'])
def test_incomplete_or_changed_ovr_evidence_is_rejected(run_fixture,change):
    f=run_fixture
    if change=='architecture':f.selection['architecture']='plain linear head'
    elif change=='unknown_source':
        path=f.run.parent/'training/android_ovr_network.py';path.write_text('modified network')
    elif change in ('parent','cache'):
        key='parent_checkpoint' if change=='parent' else 'cache_manifest'
        Path(f.selection[key]['path']).write_text('changed bytes')
    elif change=='unchanged_heads':f.selection['heads_after_sha256']=f.selection['heads_before_sha256']
    elif change=='parent_state':f.protocol['parent_frozen_state_sha256']='f'*64
    elif change=='heads_before':f.protocol['heads_before_sha256']='f'*64
    elif change=='execution':f.selection['cached_output_execution']['features_device']='cpu'
    f.save()
    with pytest.raises(ValueError):exporter.validate(f.run,f.data)


@pytest.mark.parametrize('unknown_index',[0,4,9])
def test_unknown_reference_is_checked_at_metadata_position(unknown_index):
    logits=np.full((3,10),-4,dtype=np.float32);logits[:,unknown_index]=0
    exporter.require_zero_unknown(logits,unknown_index)
    logits[1,unknown_index]=1e-9
    with pytest.raises(ValueError):exporter.require_zero_unknown(logits,unknown_index)


@pytest.mark.parametrize('value',[np.zeros((1,9),dtype=np.float32),np.zeros((0,10),dtype=np.float32),
    np.zeros((1,10),dtype=np.float64),np.full((1,10),np.nan,dtype=np.float32)])
def test_invalid_ovr_output_fails_closed(value):
    with pytest.raises(ValueError):exporter.require_zero_unknown(value,9)


@pytest.fixture
def exported_fixture(run_fixture,monkeypatch):
    f=run_fixture;region=f.run.parent/'region';region.mkdir()
    (f.run/'model.pth').write_bytes(b'synthetic selected OVR checkpoint')
    (region/'model.onnx').write_bytes(b'synthetic OVR ONNX')
    (region/'metadata.json').write_text('{}')
    for path in exporter.export_sources():
        if not path.exists():path.write_text(str(path))
    # Use the real source-set function, pointed at the isolated fixture root.
    monkeypatch.setattr(evaluator,'export_sources',exporter.export_sources)
    parity={'schema':'flux-glyph-android-onnx-parity-v1','passed':True,'test_read':False,
        'architecture':exporter.ARCHITECTURE,'unknown_reference_logit_zero':True,'encoder_and_size_head_frozen':True,
        'selection_sha256':exporter.sha(f.run/'SELECTION.json'),'checkpoint_sha256':exporter.sha(f.run/'model.pth'),
        'model_sha256':exporter.sha(region/'model.onnx'),'metadata_sha256':exporter.sha(region/'metadata.json'),
        'source_bindings':{str(path):exporter.sha(path) for path in exporter.export_sources()},
        'calibration_font_decisions_identical':True,'original_torch_groupnorm_reference':True,
        'calibration_size_availability_identical':True,'calibration_size_values_close':True,
        'cached_mps_outputs_reproduce_selected_metrics':True,'size_value_atol_px':.02,'size_value_rtol':.0003,
        'cached_training_outputs_reproduce_selected_metrics':True,
        'cached_output_execution':f.selection['cached_output_execution'],
        'parent_checkpoint_sha256':f.selection['parent_checkpoint']['sha256'],
        'parent_frozen_state_sha256':f.selection['parent_frozen_state_sha256'],
        'batch_checks':[{'batch_size':count,'passed':True,'font_and_size_checked':True,'samples':40} for count in (1,7,32,128)]}
    exporter.dump(f.run/'PARITY.json',parity)
    args=SimpleNamespace(run=f.run,data=f.data,region=region,output=f.run/'TEST.json')
    return args,parity


def test_eval_binds_ovr_and_all_reused_parity_sources(exported_fixture):
    args,parity=exported_fixture
    selection,actual=evaluator.validate_export(args)
    assert actual==parity and selection['architecture']==exporter.ARCHITECTURE
    names={Path(path).name for path in actual['source_bindings']}
    assert {'export_android_ovr.py','android_ovr_network.py','export_android_regions.py',
            'train_android_regions.py','prepare_android_regions.py','export_region_stable.py'}<=names
    assert not (args.data/'test').exists()


@pytest.mark.parametrize('field',['architecture','unknown_reference_logit_zero','encoder_and_size_head_frozen',
                                'calibration_size_availability_identical','cached_mps_outputs_reproduce_selected_metrics'])
def test_incomplete_parity_cannot_open_test(exported_fixture,monkeypatch,field):
    args,parity=exported_fixture;parity[field]=False
    exporter.dump(args.run/'PARITY.json',parity)
    monkeypatch.setattr(evaluator,'load_split',lambda *_:pytest.fail('TEST must remain closed'))
    with pytest.raises(ValueError):evaluator.evaluate(args)
    assert not args.output.exists()


def test_missing_network_source_cannot_open_test(exported_fixture,monkeypatch):
    args,parity=exported_fixture
    parity['source_bindings'].pop(str(args.run.parent/'training/android_ovr_network.py'))
    exporter.dump(args.run/'PARITY.json',parity)
    monkeypatch.setattr(evaluator,'load_split',lambda *_:pytest.fail('TEST must remain closed'))
    with pytest.raises(ValueError,match='provenance'):evaluator.evaluate(args)


def test_complete_synthetic_eval_freezes_before_loading_and_refuses_rerun(exported_fixture,monkeypatch):
    args,parity=exported_fixture
    selection=json.loads((args.run/'SELECTION.json').read_text());families=selection['families']
    metadata={'network_architecture':exporter.ARCHITECTURE,'temperature':selection['selected']['temperature'],
        'gates':selection['selected']['gates'],'max_size_relative_spread':exporter.POLICY['max_size_relative_spread'],
        'training':{'architecture':exporter.ARCHITECTURE,'encoder_and_size_head_frozen':True,
            'cached_output_execution':selection['cached_output_execution'],
            'parent_checkpoint_sha256':parity['parent_checkpoint_sha256'],
            'parent_frozen_state_sha256':parity['parent_frozen_state_sha256'],
            'selection_sha256':parity['selection_sha256'],'checkpoint_sha256':parity['checkpoint_sha256'],
            'data_manifest_sha256':selection['data_manifest_sha256']}}
    exporter.dump(args.region/'metadata.json',metadata)
    parity['metadata_sha256']=exporter.sha(args.region/'metadata.json');exporter.dump(args.run/'PARITY.json',parity)
    for path in evaluator.evaluation_sources():
        if not path.exists():path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic source')
    rows=[]
    for target,family in enumerate(families):
        for view in evaluator.RECIPES:
            rows.append({'target':target,'family':family,'view':view,'source_id':'synthetic-page','region_id':str(target),
                'source_font_family':family,'font_file_sha256':'a'*64,'tile_start':len(rows),'tile_count':1,
                'ink_height_px':40,'font_size_px':40})
    tiles=np.zeros((len(rows),1,64,256),dtype=np.float32);tiles[:,0,0,0]=np.arange(len(rows))
    folder=args.data/'test';folder.mkdir();(folder/'tiles.raw').write_bytes(tiles.tobytes());exporter.dump(folder/'rows.json',rows)
    partition={'frozen_selection':{'sha256':parity['selection_sha256']},'rejected':[],
        'array':{'path':'tiles.raw','sha256':exporter.sha(folder/'tiles.raw')},
        'metadata':{'path':'rows.json','sha256':exporter.sha(folder/'rows.json')}}
    exporter.dump(folder/'MANIFEST.json',partition)
    data={'partition':partition,'partition_sha256':exporter.sha(folder/'MANIFEST.json'),
        'families':families,'manifest_sha256':selection['data_manifest_sha256'],'manifest':{'views':evaluator.RECIPES},
        'rows':rows,'tiles':tiles}
    calls=[]
    def load(root,split):
        assert split=='test'
        freeze=json.loads((args.run/'TEST_FREEZE.json').read_text())
        assert freeze['architecture']==exporter.ARCHITECTURE
        assert freeze['test_inference_started'] is False and freeze['source_bindings']
        calls.append(split);return data
    def infer(names,inputs):
        indices=inputs['tiles'][:,0,0,0].astype(int)
        logits=np.full((len(indices),10),-10,dtype=np.float32);logits[:,9]=0
        for index,row_index in enumerate(indices):
            target=rows[row_index]['target']
            if target!=9:logits[index,target]=10
        return logits,np.zeros(len(indices),dtype=np.float32)
    model=SimpleNamespace(output_families=families,meta=metadata,session=SimpleNamespace(run=infer))
    monkeypatch.setattr(evaluator,'load_split',load)
    monkeypatch.setattr(evaluator,'holdout_evidence',lambda _: {'synthetic_fixture':True})
    monkeypatch.setattr('flux_glyph.android_font.AndroidFontClassifier',lambda _:model)
    report=evaluator.evaluate(args)
    assert report['passed'] and report['metrics']['correct_named']==36
    assert report['denominators']['unknown_outcomes']['explicit_unknown']==4
    assert report['architecture']==exporter.ARCHITECTURE and calls==['test']
    with pytest.raises(ValueError,match='existing sealed'):evaluator.evaluate(args)
    assert calls==['test']
