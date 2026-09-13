"""One fixed two-source student parameter average, never a new training search."""
from copy import deepcopy
import pytest
from training import average_unified_retention_source_balanced as module
from test_unified_retention import FAMILIES


def states(dtype=None):
    torch=pytest.importorskip('torch');dtype=dtype or torch.float32
    return torch,[{'weight':torch.tensor([3.4e38,4.],dtype=dtype),'count':torch.tensor(7,dtype=torch.int64)},
                  {'weight':torch.tensor([3.4e38,8.],dtype=dtype),'count':torch.tensor(7,dtype=torch.int64)}]


def test_float64_add_avoids_float32_overflow_and_does_not_mutate_sources():
    torch,source=states();before=deepcopy(source);result=module.average_states(source)
    assert result['weight'].dtype==torch.float32 and result['weight'].device.type=='cpu'
    torch.testing.assert_close(result['weight'],torch.tensor([3.4e38,6.]),rtol=0,atol=0)
    assert result['count'].dtype==torch.int64 and result['count'].item()==7
    assert all(torch.equal(s[k],old[k]) for s,old in zip(source,before) for k in s)
    assert all(result[k].data_ptr()!=s[k].data_ptr() for s in source for k in s)


@pytest.mark.parametrize('dtype',['float16','float32','float64'])
def test_all_floating_original_dtypes_preserved(dtype):
    torch=pytest.importorskip('torch');dtype=getattr(torch,dtype)
    a={'v':torch.tensor([1.,5.],dtype=dtype)};b={'v':torch.tensor([3.,9.],dtype=dtype)}
    result=module.average_states([a,b])
    assert result['v'].dtype==dtype
    torch.testing.assert_close(result['v'],torch.tensor([2.,7.],dtype=dtype),rtol=0,atol=0)


@pytest.mark.parametrize('fault',['one','three','keys','shape','dtype','nan','buffer','complex'])
def test_mixed_or_invalid_source_state_rejected(fault):
    torch,source=states()
    if fault=='one':source=source[:1]
    elif fault=='three':source.append(deepcopy(source[0]))
    elif fault=='keys':source[1].pop('weight')
    elif fault=='shape':source[1]['weight']=torch.ones(3)
    elif fault=='dtype':source[1]['weight']=source[1]['weight'].double()
    elif fault=='nan':source[1]['weight'][0]=float('nan')
    elif fault=='buffer':source[1]['count'].fill_(8)
    else:
        for row in source:row['weight']=row['weight'].to(torch.complex64)
    with pytest.raises(ValueError):module.average_states(source)


@pytest.mark.parametrize('fault',['family_order','architecture','teacher','missing_size'])
def test_only_exact_same_complete_25class_student_architecture_is_averaged(fault):
    torch,source=states();expected=deepcopy(source[0])
    cps=[{'architecture':module.ARCHITECTURE,'families':FAMILIES.copy(),'state_dict':s} for s in source]
    if fault=='family_order':cps[1]['families']=FAMILIES[::-1]
    elif fault=='architecture':cps[1]['architecture']='residual_or_two_model'
    elif fault=='teacher':cps[1]['state_dict']['teacher.weight']=torch.zeros(1)
    else:cps[1]['state_dict'].pop('count')
    with pytest.raises(ValueError):module.average_checkpoints(cps,FAMILIES,expected)


def documents():
    history=[]
    for step in (500,1000,1500):
        history.append({'step':step,'promotion_allowed':False,'retention_deficit':.1 if step==1000 else .2,
            'metrics':{'passed':False,'correct_named':5,'wrong_named':2},'calibration_nll':.5,
            'fixed_runtime':deepcopy(module.FIXED_RUNTIME),'retention_checks':[{'passed':False}]*46})
    selection={'promotion_allowed':False,'optimizer_steps_executed':1500,'fixed_runtime':module.FIXED_RUNTIME,
        'bindings':{},'test_read':False,'development_holdout_read':False,'history':history,
        'selected':deepcopy(history[1]),'passed':False,'calibration_passed':False}
    protocol={'steps':1500,'eval_every':500,'batch_size':96,'bindings':{},'fixed_runtime':module.FIXED_RUNTIME,
        'test_read':False,'development_holdout_read':False}
    failure={'status':'NO_PROMOTABLE_CHECKPOINT','promotion_allowed':False,'optimizer_steps_executed':1500,
        'selected_step':1000,'test_read':False,'development_holdout_read':False}
    return selection,protocol,failure


def test_source_uses_original_rank_not_the_last_or_reselected_step(monkeypatch):
    # The ranking itself remains the frozen implementation; isolate only detailed record shape validation.
    monkeypatch.setattr(module,'require_fixed_runtime',lambda r:None)
    s,p,f=documents();module.validate_source_documents(s,p,f,1500,1000)
    s['selected']=deepcopy(s['history'][-1])
    with pytest.raises(ValueError):module.validate_source_documents(s,p,f,1500,1000)


@pytest.mark.parametrize('fault',['incomplete','holdout','gates','history','promoted','objective'])
def test_incomplete_or_reinterpreted_failed_source_is_rejected(fault,monkeypatch):
    monkeypatch.setattr(module,'require_fixed_runtime',lambda r:None)
    s,p,f=documents()
    if fault=='incomplete':f['optimizer_steps_executed']=1000
    elif fault=='holdout':f['development_holdout_read']=True
    elif fault=='gates':s['fixed_runtime']={}
    elif fault=='history':s['history'].pop()
    elif fault=='promoted':s['promotion_allowed']=True
    else:s['objective']={'weight':1};p['objective']={'weight':2}
    with pytest.raises(ValueError):module.validate_source_documents(s,p,f,1500,1000)


def test_full_plain_cnn_all_four_groups_and_size_head_follow_formula():
    torch=pytest.importorskip('torch')
    from training.region_network import RegionFontClassifier
    a=RegionFontClassifier(25).state_dict();b={k:v.clone()+.002 for k,v in a.items()}
    cps=[{'state_dict':s,'architecture':module.ARCHITECTURE,'families':FAMILIES} for s in (a,b)]
    result=module.average_checkpoints(cps,FAMILIES,a)
    for key in result:
        torch.testing.assert_close(result[key],((a[key].double()+b[key].double())/2).to(a[key].dtype),rtol=0,atol=0)
    model=RegionFontClassifier(25);model.load_state_dict(result,strict=True)
    assert set(module.parameter_groups(result))=={'trunk','style','family_head','size_head'}
    assert not any('teacher' in key or 'residual' in key for key in result)
    assert all(module.parameter_groups(result)[k]!=module.parameter_groups(a)[k] for k in module.parameter_groups(a))


def test_no_inference_before_frozen_and_saved_average(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(module.core,'infer',lambda *args:calls.append(args) or ('logits','size'))
    freeze=tmp_path/'AVERAGING_FREEZE.json';checkpoint=tmp_path/'model.pth'
    with pytest.raises(ValueError):module.infer_after_freeze(None,{},'cpu',freeze,'a'*64,checkpoint,'b'*64)
    assert not calls
    freeze.write_text('{}');checkpoint.write_bytes(b'average')
    fsha,csha=module.sha(freeze),module.sha(checkpoint)
    assert module.infer_after_freeze(None,{},'cpu',freeze,fsha,checkpoint,csha)==('logits','size')
    freeze.write_text('{"changed":true}')
    with pytest.raises(ValueError):module.infer_after_freeze(None,{},'cpu',freeze,fsha,checkpoint,csha)
    assert len(calls)==1


def test_protocol_is_one_post_training_candidate_with_zero_new_optimizer_steps():
    assert module.METHOD['weights']==[.5,.5] and module.METHOD['candidate_count']==1
    assert module.METHOD['optimizer_steps_executed']==0 and module.METHOD['source_optimizer_steps_executed']==3000
    assert module.METHOD['source_selected_step']==1500 and module.METHOD['core_selected_step']==1000
    assert module.METHOD['declared_after_inspecting_source_calibration'] is True
    for key in ('predeclared_before_source_training','ratio_search','subset_search','teacher_in_model','output_averaging'):
        assert module.METHOD[key] is False
    args=module.parser().parse_args([])
    assert args.device=='mps' and args.output.name=='average-source-balanced-v1'
    assert not hasattr(args,'alpha') and not hasattr(args,'steps')
