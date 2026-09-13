"""Retention fixes weights with supervised TRAIN samples and frozen CAL checks."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from training import train_unified_retention as module

FAMILIES=module.IOS_ANCHOR_FAMILIES+[f'Joint {i}' for i in range(16)]+[module.UNKNOWN]


def sampler_data():
    rows=[]
    for target,family in enumerate(FAMILIES):
        sources=['Novel A','Novel B'] if family==module.UNKNOWN else [family]
        for source in sources:
            for domain in ('ios','android'):
                for face in (0,1):
                    for view in ('native','half'):
                        rows.append({'family':family,'target':target,'split':'train','domain':domain,'view':view,
                            'source_font_family':source,'font_face':f'{source}-{face}','font_file_sha256':str(face)*64,
                            'native_font_verified':True,'tile_start':len(rows),'tile_count':1,'source_id':str(len(rows)),
                            'region_id':'native','ink_height_px':20,'font_size_px':20})
    def indices(families,anchor=False):
        return [i for i,r in enumerate(rows) if r['domain']=='ios' and r['view']=='native' and r['family'] in families
                and (not anchor or r['font_face'].endswith('-0'))]
    pools=[{'id':'ios_native_pingfang','indices':indices(['PingFang'])},
        {'id':'ios_native_latin','indices':indices(['SF Pro','Helvetica'])},
        {'id':'ios_native_anchor_original8','indices':indices(module.IOS_ANCHOR_FAMILIES,True)}]
    return rows,pools


def test_exact96_batch_balance_and_old_anchor_source_restriction():
    rows,pools=sampler_data();sampler=module.RetentionSampler(rows,FAMILIES,71,pools)
    observed=[];original=sampler._focus_row
    def focus(family,pool):
        row=original(family,pool);observed.append((pool,row));return row
    sampler._focus_row=focus
    for _ in range(50):
        batch=sampler.batch();assert len(batch)==96
        assert sum(row['family']==module.UNKNOWN for row in batch)==16
    report=sampler.report()
    assert report['slots']=={'base_known':2400,'base_unknown':800,'ios_native_pingfang':800,
        'ios_native_sfpro_helvetica':400,'ios_native_original_eight':400}
    assert {r['rows'] for r in report['base']['class_rows'] if r['kind']=='known'}=={100}
    assert {r['rows'] for r in report['base']['source_family_rows'] if r['kind']=='unknown'}=={400}
    assert report['focus_family_rows']['PingFang']==850
    assert report['focus_family_rows']['SF Pro']==report['focus_family_rows']['Helvetica']==250
    assert all(r['domain']=='ios' and r['view']=='native' for _,r in observed)
    anchor=set(pools[2]['indices'])
    assert all(r['tile_start'] in anchor for pool,r in observed if pool=='ios_native_anchor_original8')
    pf_faces=Counter(r['font_face'] for pool,r in observed if pool=='ios_native_pingfang')
    assert set(pf_faces.values())=={400}


def test_sampling_is_reproducible_and_all24_known_classes_remain_competitors():
    rows,pools=sampler_data();a=module.RetentionSampler(rows,FAMILIES,71,pools);b=module.RetentionSampler(rows,FAMILIES,71,pools)
    for _ in range(5):assert a.batch()==b.batch()
    assert set(a.family_counts)==set(FAMILIES)


@pytest.mark.parametrize('fault',['holdout','cal','unverified','no_anchor','wrong_anchor','missing_pf','duplicate','unknown_index'])
def test_bad_partition_or_anchor_provenance_is_rejected(fault):
    rows,pools=sampler_data()
    if fault=='holdout':rows[0]['split']='development_holdout'
    elif fault=='cal':rows[0]['split']='calibration'
    elif fault=='unverified':rows[0]['native_font_verified']=False
    elif fault=='no_anchor':pools[2]['indices'].pop()
    elif fault=='wrong_anchor':pools[2]['indices'][0]=next(i for i,r in enumerate(rows) if r['domain']=='android')
    elif fault=='missing_pf':pools[0]['indices'].pop()
    elif fault=='duplicate':pools[2]['indices'].append(pools[2]['indices'][0])
    else:pools[2]['indices'][0]=len(rows)
    with pytest.raises(ValueError):module.RetentionSampler(rows,FAMILIES,71,pools)


def calibration():
    rows=[]
    for domain in ('ios','android'):
        for target in (4,5,24):
            rows.append({'family':FAMILIES[target],'target':target,'domain':domain,'view':'native',
                'source_id':domain,'region_id':str(target),'source_font_family':FAMILIES[target],
                'tile_start':len(rows),'tile_count':1,'ink_height_px':20,'font_size_px':20})
    data={'families':FAMILIES,'rows':rows,'partition':{'rejected':[]}}
    logits=np.full((len(rows),25),-10,np.float32)
    for i,r in enumerate(rows):logits[i,r['target']]=10
    plan={'masks':[{'id':'pf_anchor','indices':[0]}], 'constraints':[
        {'population':'all','metric':'named_precision','operator':'ge','value':1.},
        {'population':'pf_anchor','metric':'correct_named','operator':'ge','value':1},
        {'population':'ios','metric':'wrong_named','operator':'le','value':0},
        {'population':'android','metric':'unknown_not_named_rate','operator':'ge','value':1.},
        {'population':'family:PingFang','metric':'known_correct_coverage','operator':'ge','value':1.}]}
    return data,logits,np.zeros(len(rows),np.float32),plan


def test_retention_pass_is_separate_from_stable_policy_and_cannot_change_gates():
    data,logits,sizes,plan=calibration()
    result,outputs,detail=module.evaluate_outputs(logits,sizes,data,plan,500)
    assert result['promotion_allowed'] and result['retention_deficit']==0
    assert result['metrics']['passed'] is False  # The small fixture lacks 22 families.
    assert result['temperature']==1 and result['gates']=={'min_score':.7,'min_margin':.01,'min_patch_agreement':2/3}
    assert all(len(r['probabilities'])==25 for r in outputs)
    assert all(r['size_px'] is None for r in detail if r['explicit_unknown'])


def test_unknown_wrong_name_and_preprocess_rejection_reduce_true_denominators():
    data,logits,sizes,plan=calibration();logits[-1,:]=-10;logits[-1,4]=10
    data['partition']['rejected']=[{**data['rows'][0],'view':'half'}]
    result,_,_=module.evaluate_outputs(logits,sizes,data,plan,500)
    checks={(r['population'],r['metric']):r for r in result['retention_checks']}
    assert checks[('android','unknown_not_named_rate')]['actual']==0
    assert checks[('family:PingFang','known_correct_coverage')]['actual']==pytest.approx(2/3)
    assert checks[('all','named_precision')]['actual']==.8
    assert result['promotion_allowed'] is False


def test_fixed_normalized_deficit_then_correct_wrong_nll_and_earliest_rank():
    data,logits,sizes,plan=calibration()
    good,_,_=module.evaluate_outputs(logits,sizes,data,plan,500)
    later=deepcopy(good);later['step']=1000
    assert module.retention_rank(good)>module.retention_rank(later)
    bad=deepcopy(good);bad.update(promotion_allowed=False,retention_deficit=.1);bad['metrics']['correct_named']=10000
    assert module.retention_rank(good)>module.retention_rank(bad)
    worse=deepcopy(bad);worse['retention_deficit']=.2
    assert module.retention_rank(bad)>module.retention_rank(worse)
    logits[0,:]=-10;logits[0,5]=10
    failed,_,_=module.evaluate_outputs(logits,sizes,data,plan,500)
    zero=[r for r in failed['retention_checks'] if r['operator']=='le' and r['value']==0][0]
    assert zero['actual']==1 and zero['normalized_deficit']==1e6 and not failed['promotion_allowed']


def test_missing_population_metric_fails_instead_of_rewarding_no_predictions():
    data,logits,sizes,plan=calibration()
    plan['masks'].append({'id':'known_only','indices':[0]})
    plan['constraints'].append({'population':'known_only','metric':'unknown_not_named_rate','operator':'ge','value':.8})
    result,_,_=module.evaluate_outputs(logits,sizes,data,plan,500)
    assert result['retention_checks'][-1]['actual'] is None and result['retention_checks'][-1]['passed'] is False
    assert result['retention_deficit']==1e6 and result['promotion_allowed'] is False


def test_initializer_preserves_all25_rows_and_actual_optimization_changes_all_parameter_groups():
    torch=pytest.importorskip('torch')
    from region_network import RegionFontClassifier
    torch.set_num_threads(4);torch.manual_seed(1)
    parent=RegionFontClassifier(25);model=RegionFontClassifier(25)
    state={k:v.detach().clone() for k,v in parent.state_dict().items()}
    checkpoint={'state_dict':state,'families':FAMILIES,'architecture':module.ARCHITECTURE}
    evidence=module.initialize(model,checkpoint,FAMILIES)
    assert evidence['new_random_output_rows']==0 and evidence['all_parameters_trainable']
    assert module.state_sha(model.state_dict())==module.state_sha(state)
    logits,size=model(torch.randn(8,1,64,256))
    loss=torch.nn.functional.cross_entropy(logits,torch.arange(8),label_smoothing=.03)+.2*torch.nn.functional.smooth_l1_loss(size,torch.ones(8)*.2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=module.LEARNING_RATE);loss.backward();optimizer.step()
    for prefix in ('trunk.','style.','family_head.','size_head.'):
        assert any(not torch.equal(v,state[k]) for k,v in model.state_dict().items() if k.startswith(prefix))


@pytest.mark.parametrize('fault',['order','missing','dtype','nonfinite'])
def test_parent_checkpoint_cannot_reinitialize_or_reorder_output_classes(fault):
    torch=pytest.importorskip('torch')
    from region_network import RegionFontClassifier
    parent=RegionFontClassifier(25);state=parent.state_dict()
    checkpoint={'state_dict':state,'families':FAMILIES.copy(),'architecture':module.ARCHITECTURE}
    if fault=='order':checkpoint['families'][0],checkpoint['families'][1]=checkpoint['families'][1],checkpoint['families'][0]
    elif fault=='missing':state.pop('size_head.bias')
    elif fault=='dtype':state['size_head.bias']=state['size_head.bias'].double()
    else:state['family_head.bias'].fill_(float('nan'))
    with pytest.raises(ValueError):module.initialize(RegionFontClassifier(25),checkpoint,FAMILIES)


def test_cli_has_fixed_budget_and_no_threshold_search_options():
    args=module.parser().parse_args(['--data','d','--output','o','--plan','p'])
    assert args.steps==3000 and args.seed==2026091402
    assert module.EVAL_EVERY==500 and module.BATCH_SIZE==96 and module.LEARNING_RATE==5e-5 and module.MINIMUM_LEARNING_RATE==5e-6
    assert not hasattr(args,'gates') and module.OBJECTIVE['teacher_outputs'] is False
