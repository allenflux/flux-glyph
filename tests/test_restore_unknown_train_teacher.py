import numpy as np
import pytest

from training.prepare_unified_regions import FAMILIES
from training.restore_unknown_train_teacher import mix_partition


def arrays(rows):
    n = sum(r['tile_count'] for r in rows)
    a = np.zeros((n, 25), dtype=np.float32); b = np.ones((n, 25), dtype=np.float32)
    a.setflags(write=False); b.setflags(write=False)
    return a, b


def rows_for(targets):
    rows = []; cursor = 0
    for i, target in enumerate(targets):
        rows.append({'split': 'train', 'native_font_verified': True, 'target': target,
                     'family': FAMILIES[target], 'tile_start': cursor, 'tile_count': 2,
                     'source_id': str(i), 'region_id': str(i), 'view': 'native'})
        cursor += 2
    return rows


def test_true_unknown_rows_use_r21_and_known_rows_remain_r22():
    rows = rows_for([0, 24, 1]); a, b = arrays(rows)
    mixed, labels, mask = mix_partition(rows, a, b)
    assert labels.tolist() == [0, 0, 24, 24, 1, 1] and mask.tolist() == [False, False, True, True, False, False]
    np.testing.assert_array_equal(mixed[:2], 0.); np.testing.assert_array_equal(mixed[2:4], 1.); np.testing.assert_array_equal(mixed[4:], 0.)
    assert mixed.flags.writeable is False


def test_mixing_never_routes_by_teacher_prediction_or_platform():
    rows = rows_for([0, 24]); a, b = arrays(rows)
    a = np.array(a); b = np.array(b); a[:, 24] = 99.; b[:, 0] = 88.; a.setflags(write=False); b.setflags(write=False)
    mixed, _, _ = mix_partition(rows, a, b)
    assert mixed[0, 24] == 99. and mixed[2, 0] == 88.


def test_supplement_unknown_rows_can_be_explicitly_kept_r22():
    rows = rows_for([24, 24]); a, b = arrays(rows)
    mixed, _, mask = mix_partition(rows, a, a)
    np.testing.assert_array_equal(mixed, a)


@pytest.mark.parametrize('fault', ['shape', 'dtype', 'nonfinite', 'offset', 'split', 'label', 'writeable'])
def test_changed_teacher_or_train_identity_is_rejected(fault):
    rows = rows_for([0, 24]); a, b = arrays(rows)
    if fault == 'shape': b = np.zeros((3, 25), dtype=np.float32); b.setflags(write=False)
    elif fault == 'dtype': b = np.zeros_like(b, dtype=np.float64); b.setflags(write=False)
    elif fault == 'nonfinite': b = np.array(b); b[0, 0] = np.nan; b.setflags(write=False)
    elif fault == 'offset': rows[1]['tile_start'] = 3
    elif fault == 'split': rows[0]['split'] = 'calibration'
    elif fault == 'label': rows[0]['family'] = '__unknown__'
    else: a = np.array(a)
    with pytest.raises(ValueError): mix_partition(rows, a, b)


def test_family_order_and_tile_limit_are_enforced():
    rows=rows_for([0,24]);a,b=arrays(rows)
    changed=list(FAMILIES);changed[0],changed[1]=changed[1],changed[0]
    with pytest.raises(ValueError):mix_partition(rows,a,b,changed)
    rows[0]['tile_count']=9
    with pytest.raises(ValueError):mix_partition(rows,a,b)


def loader_fixture(monkeypatch,tmp_path):
    import hashlib,json
    from pathlib import Path
    from training import restore_unknown_train_teacher as module
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def put(p,obj):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj));return {'path':str(p),'sha256':sha(p)}
    orig_rows=rows_for([0,24,1]);supp_rows=rows_for([24]);known_rows=rows_for([10])
    data={};cache={};parts={};descs={};bindings={}
    for name,rows in [('original',orig_rows),('supplement',supp_rows),('known',known_rows)]:
        a,_=arrays(rows);n=len(a);ratios=np.zeros(n,np.float32);ratios.setflags(write=False)
        cache[name]={'logits':a,'log_em_ratio':ratios}
        root=tmp_path/'data'/name
        items={k:put(root/k,rows if k=='rows' else {'source':name,'item':k}) for k in ('data_manifest','partition_manifest','rows','tiles')}
        bindings.update({v['path']:v['sha256'] for v in items.values()})
        descs[name]={**items,'split':'train','order':module.ORDER,'tile_count':n}
        data[name]={'rows':rows,'tiles':range(n),'manifest_sha256':items['data_manifest']['sha256'],
                    'partition_sha256':items['partition_manifest']['sha256'],'families':FAMILIES}
        parts[name]={'files':{}}
    r21=tmp_path/'r21';r21.mkdir();np.save(r21/'original.npy',np.ones((6,25),np.float32))
    orig_output={'tiles':6,'logits':{'path':str(r21/'original.npy'),'sha256':sha(r21/'original.npy')}}
    bindings[orig_output['logits']['path']]=orig_output['logits']['sha256']
    ident={'checkpoint':put(r21/'model.pth',{'fake_weights':'R21'}),
           'selection':put(r21/'selection.json',{'state_after_sha256':module.STATE_SHA,'families':FAMILIES})}
    source=put(r21/'source.py',{'fixture':'not executed'})
    freeze={'schema':'flux-glyph-train-diagnostic-freeze-v1','device':'mps','optimizer_steps':0,
        'calibration_read':False,'development_read':False,'test_read':False,'families':FAMILIES,
        'checkpoints':{'r21':ident},'source':source,'partitions':{'original':descs['original']}}
    freeze_item=put(r21/'INFERENCE_FREEZE.json',freeze)
    report={'schema':'flux-glyph-train-inference-diagnostic-v1','freeze_sha256':freeze_item['sha256'],'optimizer_steps':0,
        'calibration_read':False,'development_read':False,'test_read':False,
        'models':{'r21':{'state_sha256':module.STATE_SHA,'partitions':{'original':orig_output,
        'supplement':{'logits':{'path':str(r21/'forbidden-supplement.npy')}},
        'known':{'logits':{'path':str(r21/'forbidden-known.npy')}}}}}}
    r=put(r21/'REPORT.json',report)
    labels=np.array([0,0,24,24,1,1],np.int64)
    base=tmp_path/'base';sel={'families':FAMILIES,'bindings':bindings,'teacher_mix':{'original':{
        'labels_sha256':hashlib.sha256(labels.tobytes()).hexdigest(),'unknown_teacher_tiles':2}}}
    sel_item=put(base/'SELECTION.json',sel)
    root=tmp_path/'r22';put(root/'CACHE_MANIFEST.json',{});put(root/'CACHE_FREEZE.json',{})
    for f in ('retention_r21_teacher_cache.py','cache_unified_student_teacher.py'):put(tmp_path/'training'/f,{'source':f})
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'BASE_RUN',base)
    monkeypatch.setattr(module,'BASE_SELECTION_SHA',sel_item['sha256']);monkeypatch.setattr(module,'R21_FOLDER',r21)
    for name,value in {'REPORT_SHA':r['sha256'],'FREEZE_SHA':freeze_item['sha256'],
            'CHECKPOINT_SHA':ident['checkpoint']['sha256'],'SELECTION_SHA':ident['selection']['sha256'],
            'EXPECTED_ORIGINAL_TILES':6,'EXPECTED_ORIGINAL_UNKNOWN_TILES':2}.items():monkeypatch.setattr(module,name,value)
    monkeypatch.setattr(module,'load_cache',lambda *args:(cache,{'bindings':bindings,'partitions':parts}))
    return module,root,data,cache,r21


def test_real_loader_uses_only_original_r21_and_preserves_other_partitions(monkeypatch,tmp_path):
    module,root,data,cache,r21=loader_fixture(monkeypatch,tmp_path)
    loaded=[];original=np.load
    def tracking(path,*args,**kwargs):
        loaded.append(str(path));return original(path,*args,**kwargs)
    monkeypatch.setattr(module.np,'load',tracking)
    mixed,proof=module.load_replay_teacher(root,data)
    assert loaded==[str(r21/'original.npy')]
    np.testing.assert_array_equal(mixed['original']['logits'][2:4],1.)
    np.testing.assert_array_equal(mixed['original']['logits'][[0,1,4,5]],0.)
    assert mixed['supplement'] is cache['supplement'] and mixed['known'] is cache['known']
    assert mixed['original']['log_em_ratio'] is cache['original']['log_em_ratio']
    assert [proof['partitions'][k]['restored_r21_tiles'] for k in ('original','supplement','known')]==[2,0,0]
    assert not mixed['original']['logits'].flags.writeable and proof['test_read'] is False


def test_loader_rejects_relabelled_in_memory_original_rows(monkeypatch,tmp_path):
    module,root,data,_,_=loader_fixture(monkeypatch,tmp_path)
    data['original']['rows'][0]['target']=1;data['original']['rows'][0]['family']=FAMILIES[1]
    with pytest.raises(ValueError,match='In-memory TRAIN rows'):
        module.load_replay_teacher(root,data)
