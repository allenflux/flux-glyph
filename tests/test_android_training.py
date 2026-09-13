import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training'))
from train_android_regions import decisions,metrics,region_outputs,select_grid,POLICY,validate_initialization
FAMILIES=[f'Font {i}' for i in range(9)]+['__unknown__']
GATES={'min_score':.5,'min_margin':.01,'min_patch_agreement':2/3}


def row(target=0,view='native'):
    return {'family':FAMILIES[target],'target':target,'view':view,'source_id':'page','region_id':str(target),
            'source_font_family':FAMILIES[target],'font_file_sha256':'a'*64,'ink_height_px':40,'font_size_px':40,
            'tile_start':0,'tile_count':1}


def output(target=0,score=.9,margin=.8,agreement=1):
    return {'predicted':target,'score':score,'margin':margin,'agreement':agreement,'em_ratio':1,'size_spread':0}


def test_unknown_and_uncertainty_never_count_as_named_fonts():
    rows=[row(0),row(9),row(0)];pred=[output(9),output(0),output(0,score=.4)]
    details=decisions(pred,rows,FAMILIES,GATES);m=metrics(details,FAMILIES)
    assert m['named']==1 and m['wrong_named']==1 and m['correct_named']==0
    assert details[0]['explicit_unknown'] and not details[0]['named'] and details[0]['size_px'] is None
    assert m['unknown_wrongly_named']==1 and m['unknown_not_named_rate']==0


def test_preprocessor_abstentions_remain_in_coverage_denominators():
    details=decisions([output(0)],[row(0)],FAMILIES,GATES)
    m=metrics(details,FAMILIES,[{'family':FAMILIES[0],'view':'native'},{'family':'__unknown__','view':'half'}])
    assert m['known_views']==2 and m['known_correct_coverage']==.5
    assert m['native_known_top1_accuracy']==.5 and m['unknown_not_named_rate']==1
    assert m['explicit_unknown_recall']==0 and m['preprocessing_rejected']==2


@pytest.mark.parametrize('change',[{'margin':0},{'agreement':.5},{'score':.49}])
def test_runtime_gate_failures_clear_size(change):
    pred=output();pred.update(change)
    value=decisions([pred],[row()],FAMILIES,GATES)[0]
    assert not value['named'] and value['size_px'] is None


def test_size_output_uses_runtime_rounding_and_spread():
    pred=output();pred['em_ratio']=1.00123
    value=decisions([pred],[row()],FAMILIES,GATES)[0]
    assert value['size_px']==40.05
    pred['size_spread']=.201
    assert decisions([pred],[row()],FAMILIES,GATES)[0]['size_px'] is None


def test_full_softmax_preserves_unknown_runner_up():
    logits=np.full((1,10),-20,dtype=np.float32);logits[0,[0,9]]=4
    values=region_outputs(logits,np.zeros(1,dtype=np.float32),[row()],1)
    assert values[0]['margin']==0 and values[0]['score']<=.5
    assert not decisions(values,[row()],FAMILIES,GATES)[0]['named']


@pytest.mark.parametrize('value',[np.nan,np.inf,3.01])
def test_invalid_size_outputs_fail_closed(value):
    with pytest.raises(ValueError):region_outputs(np.zeros((1,10),dtype=np.float32),np.array([value],dtype=np.float32),[row()],1)


def test_calibration_grid_covers_all_fonts_and_unknown_without_test():
    rows=[row(i) for i in range(10)]
    for i,r in enumerate(rows):r['tile_start']=i
    logits=np.eye(10,dtype=np.float32)*15
    data={'rows':rows,'families':FAMILIES,'partition':{'rejected':[]}}
    selected,grid=select_grid(logits,np.zeros(10,dtype=np.float32),data,1000)
    assert len(grid)==60 and selected[1]['metrics']['passed']
    assert selected[1]['metrics']['named_precision']==1
    assert selected[1]['metrics']['unknown_wrongly_named']==0
    assert selected[1]['metrics']['known_correct_coverage']==1
    assert POLICY['test_used_for_selection'] is False


@pytest.mark.parametrize('problem',['nine_classes','float64','missing_tiles','unused_tiles','overlap','zero_tiles','bad_temperature'])
def test_offline_output_contract_matches_serving_contract(problem):
    logits=np.zeros((2,10),dtype=np.float32);ratios=np.zeros(2,dtype=np.float32)
    rows=[row(),row()];rows[1]['tile_start']=1;temperature=1.
    if problem=='nine_classes':logits=logits[:,:9]
    elif problem=='float64':logits=logits.astype(np.float64)
    elif problem=='missing_tiles':rows[1]['tile_count']=2
    elif problem=='unused_tiles':rows=rows[:1]
    elif problem=='overlap':rows[1]['tile_start']=0
    elif problem=='zero_tiles':rows[1]['tile_count']=0
    elif problem=='bad_temperature':temperature=np.nan
    with pytest.raises(ValueError):region_outputs(logits,ratios,rows,temperature)


def test_incomplete_or_misordered_checkpoint_cannot_silently_claim_warm_start():
    base={'trunk.weight':np.zeros((4,1,3,3)),'size_head.weight':np.zeros((1,128)),
          'size_head.bias':np.zeros(1),'family_head.weight':np.zeros((10,128)),'family_head.bias':np.zeros(10)}
    state={**base,'family_head.weight':np.zeros((9,128)),'family_head.bias':np.zeros(9)}
    checkpoint={'families':FAMILIES[:9],'state_dict':state}
    validate_initialization(base,checkpoint)
    for missing in ['trunk.weight','size_head.weight','family_head.bias']:
        with pytest.raises(ValueError,match='parameters'):
            validate_initialization(base,{**checkpoint,'state_dict':{k:v for k,v in state.items() if k!=missing}})
    with pytest.raises(ValueError,match='family registry'):
        validate_initialization(base,{**checkpoint,'families':['duplicate']*9})
    with pytest.raises(ValueError,match='shape differs'):
        validate_initialization(base,{**checkpoint,'families':FAMILIES})


def test_calibration_decisions_equal_real_android_predict_for_identical_tiles(monkeypatch):
    from test_android_font import fake_model,source
    import flux_glyph.android_font as runtime
    image=source();prepared=runtime.preprocess_region(image);n=len(prepared['tiles'])
    original=prepared.copy()
    monkeypatch.setattr(runtime,'preprocess_region',lambda image:original)
    patterns=[([8.]+[0.]*9,1.,GATES),([0.]*9+[8.],1.,GATES),
              ([5.]+[-10.]*8+[4.9],1.,GATES),([0.]*10,1.,GATES),
              ([8.]+[0.]*9,2.,{**GATES,'min_score':.99})]
    for values,temperature,gates in patterns:
        model=fake_model(values);model.meta['temperature']=temperature;model.meta['gates']=gates
        logits=np.tile(values,(n,1)).astype(np.float32);ratios=np.full(n,np.log(1.2),np.float32)
        r={**row(),'family':model.output_families[0],'tile_count':n,'ink_height_px':prepared['ink_height_px']}
        measured=decisions(region_outputs(logits,ratios,[r],temperature),[r],model.output_families,gates)[0]
        predicted=model.predict(image)
        assert measured['named']==(predicted['status']=='candidate')
        assert measured['explicit_unknown']==(predicted['status']=='out_of_scope')
        assert (measured['predicted_family'] if measured['named'] else None)==predicted['family']
        assert measured['size_px']==predicted['font_size_px_estimate']


def test_metrics_do_not_count_known_wrong_size_or_unknown_correct_as_coverage():
    details=decisions([output(1),output(9),output(0)],[row(0),row(9),row(0)],FAMILIES,GATES)
    value=metrics(details,FAMILIES)
    assert value['known_correct_coverage']==.5 and value['named_precision']==.5
    assert value['wrong_named']==1 and value['size']['correctly_named_with_size']==1
    assert value['unknown_not_named_rate']==1 and value['explicit_unknown_recall']==1
