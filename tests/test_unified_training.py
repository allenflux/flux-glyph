"""Synthetic unified-training fixtures; no real holdout/TEST data is opened."""
from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from training import train_unified_regions as unified

FAMILIES=['Known A','Known B','Known C',unified.UNKNOWN]


def sampled_rows():
    rows=[]
    for target,family in enumerate(FAMILIES):
        sources=['Unknown A','Unknown B'] if target==3 else [family]
        for source in sources:
            domains=['ios','android'] if source in ('Known A','Unknown A') else ['android']
            for domain in domains:
                for face in range(2 if domain=='android' else 1):
                    for view in (['native'] if domain=='ios' else unified.VIEW_WEIGHTS):
                        for _ in range(5 if face==1 else 1):
                            rows.append({'family':family,'target':target,'split':'train','domain':domain,
                                'view':view,'source_font_family':source,'font_face':f'{source}-{face}',
                                'font_file_sha256':str(face)*64,'tile_start':len(rows),'tile_count':1})
    return rows


def test_batch_has_77_known_19_unknown_and_balances_class_renderer_face():
    sampler=unified.UnifiedSampler(sampled_rows(),FAMILIES,91)
    for _ in range(300):
        rows=sampler.batch()
        assert len(rows)==96 and sum(row['target']==3 for row in rows)==19
    report=sampler.report()
    assert report['known_rows']==23100 and report['unknown_rows']==5700
    assert {row['rows'] for row in report['class_rows'] if row['kind']=='known'}=={7700}
    assert {row['rows'] for row in report['source_family_rows'] if row['kind']=='unknown'}=={2850}
    assert {row['rows'] for row in report['domain_rows'] if row['family']=='Known A'}=={3850}
    faces=[row['rows'] for row in report['face_rows'] if row['family']=='Known A' and row['domain']=='android']
    assert faces==[1925,1925]
    assert all(row['view']=='native' for row in report['view_rows'] if row['domain']=='ios')


@pytest.mark.parametrize('problem',['holdout','test','target','domain','source','face','tile','missing_known'])
def test_sampling_fails_closed_for_wrong_partitions_or_native_labels(problem):
    rows=sampled_rows()
    if problem=='holdout':rows[0]['split']='development_holdout'
    elif problem=='test':rows[0]['split']='test'
    elif problem=='target':rows[0]['target']=3
    elif problem=='domain':rows[0]['domain']='guessed_android'
    elif problem=='source':rows[0].pop('source_font_family')
    elif problem=='face':rows[0].pop('font_face')
    elif problem=='tile':rows[0]['tile_count']=0
    elif problem=='missing_known':rows=[row for row in rows if row['target']!=2]
    with pytest.raises(ValueError):unified.UnifiedSampler(rows,FAMILIES,1)


def metric_row(family='Known A',domain='ios',correct=True,named=True,view='native',source=None):
    return {'family':family,'domain':domain,'view':view,'source_font_family':source or family,
        'named':named,'correct_named':named and correct,'wrong_named':named and not correct,
        'top1_correct':correct,'explicit_unknown':family==unified.UNKNOWN and not named,
        'size_ape':.01 if named and correct else None}


def test_rejections_remain_in_family_domain_unknown_and_size_denominators():
    details=[metric_row(),metric_row(domain='android'),metric_row(unified.UNKNOWN,'ios',False,False),
             metric_row(unified.UNKNOWN,'android',False,True,source='Novel font')]
    rejected=[{'family':'Known A','domain':'ios','view':'half','source_font_family':'Known A'},
              {'family':unified.UNKNOWN,'domain':'android','view':'native','source_font_family':'Novel font'}]
    result=unified.metrics(details,['Known A',unified.UNKNOWN],rejected)
    assert result['views']==6 and result['known_views']==3 and result['unknown_views']==3
    assert result['named_precision']==pytest.approx(2/3)
    assert result['known_correct_coverage']==pytest.approx(2/3)
    assert result['unknown_not_named_rate']==pytest.approx(2/3)
    assert result['per_domain']['ios']['known_correct_coverage']==.5
    assert result['per_domain']['android']['unknown_sources']['Novel font']['not_named_rate']==.5
    assert result['size']['correctly_named_with_size']==2
    assert result['size']['coverage_of_correct_names']==1
    assert not result['passed']


def test_missing_native_or_domain_or_class_cannot_report_stable_pass():
    families=['Known A','Known B',unified.UNKNOWN]
    rows=[metric_row('Known A','ios',view='half'),metric_row(unified.UNKNOWN,'ios',False,False,view='half')]
    result=unified.metrics(rows,families)
    assert result['native_known_top1_accuracy'] is None
    assert result['per_family']['Known B']['correct_coverage'] is None
    assert result['per_domain']['android']['named_precision'] is None
    assert result['checks']['native_known_top1'] is False
    assert result['checks']['domains'] is False and result['checks']['per_family_coverage'] is False
    assert result['passed'] is False


def test_95_percent_preference_is_independent_from_stable_acceptance():
    rows=[metric_row(domain='ios' if i%2 else 'android',correct=i<96) for i in range(100)]
    rows+=[metric_row(unified.UNKNOWN,domain,False,False) for domain in ('ios','android')]
    measured=unified.metrics(rows,['Known A',unified.UNKNOWN])
    assert measured['named_precision']==.96 and measured['experimental_precision_priority_met']
    assert not measured['passed'] and not measured['checks']['named_precision']
    record={'metrics':measured,'calibration_nll':.1,'step':1000,'grid_index':0}
    low=deepcopy(record);low['metrics']['experimental_precision_priority_met']=False
    low['metrics']['correct_named']=1000
    assert unified.selection_rank(record)>unified.selection_rank(low)


def test_dynamic_25_outputs_keep_unknown_probability_and_do_not_force_a_font():
    families=[f'F{i}' for i in range(24)]+[unified.UNKNOWN]
    rows=[{'tile_start':0,'tile_count':1,'target':24,'family':unified.UNKNOWN,'domain':'android','view':'native',
        'source_id':'page','region_id':'row','source_font_family':'New font','ink_height_px':20,'font_size_px':24}]
    logits=np.zeros((1,25),np.float32);logits[0,24]=10;ratios=np.zeros(1,np.float32)
    outputs=unified.region_outputs(logits,ratios,rows,1.)
    result=unified.decisions(outputs,rows,families,{'min_score':0,'min_margin':0,'min_patch_agreement':0})[0]
    assert outputs[0]['predicted']==24 and len(outputs[0]['probabilities'])==25
    assert sum(outputs[0]['probabilities'])==pytest.approx(1.)
    assert result['explicit_unknown'] and not result['named'] and result['size_px'] is None


def test_equal_logits_never_name_a_font_even_with_zero_margin_gate():
    rows=[{'tile_start':0,'tile_count':1,'target':0,'family':'Known A','domain':'ios','view':'native',
        'source_id':'page','region_id':'row','source_font_family':'Known A','ink_height_px':20,'font_size_px':24}]
    outputs=unified.region_outputs(np.zeros((1,4),np.float32),np.zeros(1,np.float32),rows,1.)
    assert not unified.decisions(outputs,rows,FAMILIES,{'min_score':0,'min_margin':0,'min_patch_agreement':0})[0]['named']


def test_calibration_grid_uses_one_shared_policy_and_reproducible_rank():
    rows=[]
    for domain in ('ios','android'):
        for target,family in enumerate(FAMILIES):
            rows.append({'tile_start':len(rows),'tile_count':1,'target':target,'family':family,'domain':domain,
                'view':'native','source_id':domain,'region_id':str(target),'source_font_family':family,
                'ink_height_px':20,'font_size_px':20})
    logits=np.full((len(rows),len(FAMILIES)),-4.,np.float32)
    for index,row in enumerate(rows):logits[index,row['target']]=4.
    data={'families':FAMILIES,'rows':rows,'partition':{'rejected':[]}}
    best,grid=unified.select_grid(logits,np.zeros(len(rows),np.float32),data,1000)
    assert len(grid)==60 and [row['grid_index'] for row in grid]==list(range(60))
    assert best[1]==max(grid,key=unified.selection_rank)
    assert best[1]['metrics']['passed'] is True
    assert best[1]['metrics']['experimental_precision_priority_met'] is False  # fewer than 100 names
    for row in grid:
        assert set(row['gates'])=={'min_score','min_margin','min_patch_agreement'}
        assert set(row['metrics']['per_domain'])=={'ios','android'}


def test_warm_start_uses_one_encoder_and_maps_only_matching_font_rows():
    torch=pytest.importorskip('torch')
    from region_network import RegionFontClassifier
    torch.manual_seed(12);parent=RegionFontClassifier(3)
    checkpoint={'families':['Known C','Known A','Known B'],'state_dict':parent.state_dict()}
    model=RegionFontClassifier(4);model.unified_families=FAMILIES
    random_unknown=model.family_head.weight[3].detach().clone()
    evidence=unified.initialize(model,checkpoint)
    assert evidence['single_parent_encoder'] and evidence['all_parameters_trainable']
    assert evidence['new_family_rows']==[{'family':unified.UNKNOWN,'target_row':3}]
    assert torch.equal(model.family_head.weight[0],parent.family_head.weight[1])
    assert torch.equal(model.family_head.weight[3],random_unknown)
    for key,value in model.state_dict().items():
        if not key.startswith('family_head.'):assert torch.equal(value,checkpoint['state_dict'][key])
    before={key:value.detach().clone() for key,value in model.state_dict().items()}
    logits,size=model(torch.randn(8,1,64,256))
    loss=torch.nn.functional.cross_entropy(logits,torch.arange(8)%4)+.2*torch.nn.functional.smooth_l1_loss(size,torch.ones(8)*.3)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0002);loss.backward();optimizer.step()
    for prefix in ('trunk.','style.','family_head.','size_head.'):
        assert any(not torch.equal(value,before[key]) for key,value in model.state_dict().items() if key.startswith(prefix))


@pytest.mark.parametrize('problem',['missing_tensor','wrong_width','nonfinite','wrong_dtype','parent_unknown'])
def test_initializer_rejects_incomplete_or_incompatible_source(problem):
    torch=pytest.importorskip('torch')
    from region_network import RegionFontClassifier
    checkpoint={'families':['Known A','Known B','Known C'],'state_dict':RegionFontClassifier(3).state_dict()}
    model=RegionFontClassifier(4);model.unified_families=FAMILIES
    if problem=='missing_tensor':checkpoint['state_dict'].pop('size_head.bias')
    elif problem=='wrong_width':checkpoint['state_dict']['family_head.weight']=torch.zeros(3,127)
    elif problem=='nonfinite':checkpoint['state_dict']['size_head.bias'].fill_(float('nan'))
    elif problem=='wrong_dtype':checkpoint['state_dict']['size_head.bias']=checkpoint['state_dict']['size_head.bias'].double()
    elif problem=='parent_unknown':checkpoint['families'][0]=unified.UNKNOWN
    with pytest.raises(ValueError):unified.initialize(model,checkpoint)


def test_protocol_defaults_are_single_model_and_fixed_budget():
    args=unified.parser().parse_args(['--data','joint-data','--output','joint-run'])
    assert args.steps==6000 and args.checkpoint.parent.name=='run-v2'
    assert unified.BATCH_SIZE==96 and unified.KNOWN_PER_BATCH==77 and unified.LEARNING_RATE==.0002
    assert unified.POLICY['experimental_priority_precision']==.95 and unified.POLICY['minimum_named_precision']==.98
    assert unified.OBJECTIVE['teacher_outputs'] is False and unified.OBJECTIVE['second_encoder'] is False
