"""True paired known replay retains class/source quotas and same-tile teachers."""
from collections import Counter
from copy import deepcopy
import numpy as np
import pytest
from training import train_unified_retention_paired_known as m
from training.retention_paired_known_sampler import PairedKnownSampler,SAMPLING,KNOWN_FAMILIES,KNOWN_SUPPLEMENT_MARKER
from training.retention_source_balanced_sampler import expected_unknown_source_counts
from training.retention_supplement_sampler import SUPPLEMENT_MARKER,VIEW_WEIGHTS
from test_retention_source_balanced_sampler import source_balanced_data as generic_data
from test_retention_supplement_sampler import supplemental_rows
from training.prepare_unified_regions import FAMILIES


def source_balanced_data():
    rows,pools=generic_data()
    for row in rows:
        if row['family']!='__unknown__':
            previous=row['family'];row['family']=FAMILIES[row['target']]
            row['source_font_family']=row['family'];row['font_face']=row['font_face'].replace(previous,row['family'])
    return rows,pools


def known_rows():
    rows=[]
    for family in KNOWN_FAMILIES:
        for view in VIEW_WEIGHTS:
            index=len(rows)
            rows.append({'family':family,'target':FAMILIES.index(family),'source_font_family':family,
                'split':'train','domain':'android','source_dataset':'android_paired_known_supplement',
                'source_id':'paired-'+str(index),'region_id':'r','view':view,'native_font_verified':True,
                'font_face':family+'-Regular','font_file_sha256':'a'*64,'ttc_index':0,'tile_start':index,'tile_count':1,
                'log_em_ratio':.2,'pair_evidence':{'intentional_train_text_pair':True}})
    return rows


def sampler():
    old,pools=source_balanced_data()
    return PairedKnownSampler(old,FAMILIES,m.SEED,pools,supplemental_rows(),known_rows())


def test_actual_sampler_emits46old_known2paired_and_no_discarded_proposals():
    replay=sampler();counts=m.FullSupplementCounts(FAMILIES,replay.unknown_source_order)
    calls=[];draw=replay.base._row
    replay.base._row=lambda group:(calls.append(group),draw(group))[1]
    for step in range(1,100):
        before=len(calls);rows=replay.batch();positive=[r for r in rows if r.get(KNOWN_SUPPLEMENT_MARKER)]
        new_unknown=sum(bool(r.get(SUPPLEMENT_MARKER)) for r in rows)
        assert len(rows)==96 and Counter(r['family'] for r in positive)==dict.fromkeys(KNOWN_FAMILIES,1)
        assert len(calls)-before==46+16-new_unknown
        assert Counter(g[0] for g in calls[before:])['known']==46
        assert all(not (r.get(KNOWN_SUPPLEMENT_MARKER) and r.get(SUPPLEMENT_MARKER)) for r in rows)
        family_counts=Counter(r['family'] for r in rows)
        assert family_counts['PingFang']==19 and family_counts['SF Pro']==family_counts['Helvetica']==7
        assert all(family_counts[f]==(3 if f in FAMILIES[:8] else 2)
            for f in FAMILIES[:-1] if f not in ('PingFang','SF Pro','Helvetica'))
        counts.update(rows,[False]*96)
    actual=replay.report();report=counts.report();m.validate_training_counts(report,FAMILIES,99)
    assert actual['base']['known_rows']==99*46
    assert actual['known_supplement_rows']==99*2 and actual['known_supplement_source_rows']==dict.fromkeys(KNOWN_FAMILIES,99)
    assert actual['unknown_source_rows']==expected_unknown_source_counts(replay.unknown_source_order,99)
    assert sum(actual['slots'].values())==99*96 and actual['proposal_rows_discarded']==0
    assert report['original_rows']+report['supplement_rows']+report['known_supplement_rows']==99*96
    assert report['family_rows']==actual['family_rows'] and report['domain_rows']==actual['domain_rows']


@pytest.mark.parametrize('fault',['unknown','split','domain','pair','marker','both_markers','missing_view','font_source','dataset'])
def test_invalid_known_pool_never_becomes_a_sampling_label(fault):
    old,pools=source_balanced_data();positive=known_rows()
    if fault=='unknown':positive[0]['family']='__unknown__';positive[0]['target']=24
    elif fault=='split':positive[0]['split']='calibration'
    elif fault=='domain':positive[0]['domain']='ios'
    elif fault=='pair':positive[0]['pair_evidence']={}
    elif fault=='marker':positive[0][KNOWN_SUPPLEMENT_MARKER]=True
    elif fault=='both_markers':old[0][KNOWN_SUPPLEMENT_MARKER]=True
    elif fault=='missing_view':positive.pop()
    elif fault=='font_source':positive[0]['source_font_family']='Zhuque Fangsong'
    else:positive[0]['source_dataset']='old'
    with pytest.raises(ValueError):PairedKnownSampler(old,FAMILIES,m.SEED,pools,supplemental_rows(),positive)


def test_draws_do_not_mutate_native_rows_and_are_seed_reproducible():
    old,pools=source_balanced_data();negative=supplemental_rows();positive=known_rows();before=deepcopy((old,negative,positive))
    a=PairedKnownSampler(old,FAMILIES,m.SEED,pools,negative,positive);b=sampler()
    assert a.batch()==b.batch() and a.batch()==b.batch()
    assert (old,negative,positive)==before


def cache_source(tag,family,target):
    row={'source_id':tag,'region_id':'region','view':'native','split':'train','native_font_verified':True,
        'family':family,'target':target,'tile_start':0,'tile_count':2,'log_em_ratio':.3}
    value={'old':.1,'unknown':.2,'known':.3}[tag]
    tiles=np.stack([np.full((1,64,256),value+i*.01,np.float32) for i in range(2)])
    logits=np.stack([np.full(25,value+i*.01,np.float32) for i in range(2)])
    return row,m.batch_source({'partition':{'split':'train'},'tiles':tiles,'rows':[row]}, {'base_logits':logits})


def routing_fixture():
    a,old=cache_source('old',FAMILIES[0],0);b,unknown=cache_source('unknown','__unknown__',24);c,known=cache_source('known','LXGW WenKai',10)
    b=dict(b,**{SUPPLEMENT_MARKER:True});c=dict(c,**{KNOWN_SUPPLEMENT_MARKER:True})
    return [deepcopy(r) for _ in range(32) for r in (a,b,c)],old,unknown,known


def test_three_cache_routes_choose_the_same_random_tile_for_pixels_and_teacher():
    rows,old,unknown,known=routing_fixture()
    pixels,teachers,sizes=m.pixel_batch(rows,old,unknown,known,np.random.default_rng(11))
    np.testing.assert_array_equal(pixels[:,0,0,0],teachers[:,0])
    np.testing.assert_allclose(sizes,.3)
    assert all(.09<=float(pixels[i,0,0,0])<=.12 for i in range(0,96,3))
    assert all(.29<=float(pixels[i,0,0,0])<=.32 for i in range(2,96,3))


@pytest.mark.parametrize('fault',['both','missing','false_route','nonbool','relabel','tile','teacher_length'])
def test_marker_or_tile_tampering_fails_closed(fault):
    rows,old,unknown,known=routing_fixture()
    if fault=='both':rows[2][SUPPLEMENT_MARKER]=True
    elif fault=='missing':rows[2].pop(KNOWN_SUPPLEMENT_MARKER)
    elif fault=='false_route':rows[1].pop(SUPPLEMENT_MARKER);rows[1][KNOWN_SUPPLEMENT_MARKER]=True
    elif fault=='nonbool':rows[2][KNOWN_SUPPLEMENT_MARKER]=1
    elif fault=='relabel':rows[2]['family']='PingFang';rows[2]['target']=4
    elif fault=='tile':rows[2]['tile_count']=8
    else:known['teacher_logits']=known['teacher_logits'][:1]
    with pytest.raises(ValueError):m.pixel_batch(rows,old,unknown,known,np.random.default_rng(11))


@pytest.mark.parametrize('fault',['positive_rows','per_source','original','per_step','region_total','unknown_cycle'])
def test_training_counts_reject_changed_positive_or_negative_quotas(fault):
    replay=sampler();counts=m.FullSupplementCounts(FAMILIES,replay.unknown_source_order)
    for _ in range(11):counts.update(replay.batch(),[False]*96)
    report=counts.report();m.validate_training_counts(report,FAMILIES,11)
    if fault=='positive_rows':report['known_supplement_rows']+=1
    elif fault=='per_source':report['known_supplement_source_rows'][KNOWN_FAMILIES[0]]+=1
    elif fault=='original':report['original_rows']+=22
    elif fault=='per_step':report['known_supplement_rows_per_step'][0]=3
    elif fault=='region_total':report['known_supplement_region_rows'][0]['rows']+=1
    else:report['unknown_source_rows'][replay.unknown_source_order[0]]+=1
    with pytest.raises(ValueError):m.validate_training_counts(report,FAMILIES,11)


def test_previous_loss_is_reused_exactly_and_new_teacher_correct_positive_can_learn():
    assert m.full_losses is m.previous.full_losses and m.OBJECTIVE==m.previous.OBJECTIVE
    torch=pytest.importorskip('torch');torch.set_num_threads(4)
    from training.region_network import RegionFontClassifier
    model=RegionFontClassifier(25);before=m.parameter_groups(model.state_dict());rows=sampler().batch()
    targets=torch.tensor([r['target'] for r in rows]);teacher=torch.full((96,25),-2.)
    teacher[torch.arange(96),targets]=2.
    logits,size=model(torch.rand(96,1,64,256));sizes=torch.full((96,),.2)
    loss,_,_,_,mask=m.full_losses(logits,size,targets,sizes,teacher,rows,FAMILIES)
    assert not mask[targets==24].any()
    assert all(bool(mask[i]) for i,r in enumerate(rows) if r.get(KNOWN_SUPPLEMENT_MARKER))
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5);optimizer.zero_grad();loss.backward();optimizer.step()
    after=m.parameter_groups(model.state_dict())
    assert all(before[k]!=after[k] for k in before)


def test_fixed_sampling_initializer_and_teacher_are_separate():
    assert SAMPLING['known']==48 and SAMPLING['base_known']==46 and SAMPLING['known_supplement']==2
    assert SAMPLING['unknown']==16 and SAMPLING['known_class_prior_changed'] is False
    assert m.SEED==2026091411 and m.STEPS==3000 and m.EVAL_EVERY==500
    assert m.STUDENT_CHECKPOINT_SHA!=m.BASE_CHECKPOINT_SHA
    args=m.parser().parse_args([])
    assert args.initializer.parent.name=='run-source-balanced-v1' and args.checkpoint.parent.name=='run-core-v1'
    assert args.output.name=='run-paired-known-v1'
