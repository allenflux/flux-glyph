"""A single ordered parameter average, frozen before any calibration inference."""
from copy import deepcopy
import pytest
from training import average_unified_retention as module
from test_unified_retention import FAMILIES


def states():
    torch=pytest.importorskip('torch')
    return torch,[{'weight':torch.tensor([value,6.],dtype=torch.float32),'count':torch.tensor(4,dtype=torch.int64)}
                  for value in (1e8,1.,-1e8)]


def test_ordered_float64_mean_rounds_once_and_preserves_identical_buffers():
    torch,source=states();before=deepcopy(source)
    result=module.average_states(source)
    assert result['weight'].dtype==torch.float32 and result['weight'].device.type=='cpu'
    torch.testing.assert_close(result['weight'],torch.tensor([1/3,6.],dtype=torch.float32),rtol=0,atol=0)
    assert result['count'].dtype==torch.int64 and result['count'].item()==4
    assert all(torch.equal(row[key],original[key]) for row,original in zip(source,before) for key in row)
    assert all(result[key].data_ptr()!=row[key].data_ptr() for row in source for key in row)


@pytest.mark.parametrize('fault',['two','keys','shape','dtype','nan','buffer','half','teacher'])
def test_invalid_or_mixed_state_cannot_be_averaged(fault):
    torch,source=states()
    if fault=='two':source=source[:2]
    elif fault=='keys':source[1].pop('weight')
    elif fault=='shape':source[1]['weight']=torch.zeros(3)
    elif fault=='dtype':source[1]['weight']=source[1]['weight'].double()
    elif fault=='nan':source[1]['weight'][0]=float('nan')
    elif fault=='buffer':source[1]['count'].fill_(5)
    elif fault=='half':
        for row in source:row['weight']=row['weight'].half()
    else:source[1]['teacher.weight']=torch.zeros(2)
    with pytest.raises(ValueError):module.average_states(source)


@pytest.mark.parametrize('fault',['order','families','architecture','teacher','count'])
def test_checkpoint_contract_keeps_all25_classes_and_only_one_student(fault):
    torch,source=states()
    checkpoints=[{'state_dict':row,'step':step,'families':FAMILIES.copy(),'architecture':module.ARCHITECTURE}
                 for row,step in zip(source,module.SOURCE_STEPS)]
    expected=deepcopy(source[0])
    if fault=='order':checkpoints=checkpoints[::-1]
    elif fault=='families':checkpoints[1]['families']=FAMILIES[::-1]
    elif fault=='architecture':checkpoints[1]['architecture']='two_networks'
    elif fault=='teacher':checkpoints[1]['state_dict']['teacher.weight']=torch.ones(2)
    else:checkpoints=checkpoints[:2]
    with pytest.raises(ValueError):module.average_checkpoints(checkpoints,FAMILIES,expected)


def test_checkpoint_class_order_retained_in_complete_average():
    torch,source=states()
    checkpoints=[{'state_dict':row,'step':step,'families':FAMILIES.copy(),'architecture':module.ARCHITECTURE}
                 for row,step in zip(source,module.SOURCE_STEPS)]
    result=module.average_checkpoints(checkpoints,FAMILIES,source[0])
    assert result['weight'][0].item()==pytest.approx(1/3)
    assert all(c['families']==FAMILIES for c in checkpoints)


def test_inference_cannot_start_without_intact_freeze_and_saved_checkpoint(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(module.core,'infer',lambda *args:calls.append(args) or ('logits','sizes'))
    freeze=tmp_path/'AVERAGING_FREEZE.json';checkpoint=tmp_path/'model.pth'
    with pytest.raises(ValueError):module.infer_after_freeze(None,{},'cpu',freeze,'0'*64,checkpoint,'1'*64)
    assert calls==[]
    freeze.write_text('{}');checkpoint.write_bytes(b'checkpoint fixture')
    freeze_sha,checkpoint_sha=module.sha(freeze),module.sha(checkpoint)
    assert module.infer_after_freeze('model',{'split':'calibration'},'cpu',freeze,freeze_sha,checkpoint,checkpoint_sha)==('logits','sizes')
    assert len(calls)==1
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError):module.infer_after_freeze(None,{},'cpu',freeze,freeze_sha,checkpoint,checkpoint_sha)
    assert len(calls)==1


def test_method_is_single_postcal_candidate_with_no_new_optimizer_or_search():
    assert module.SOURCE_STEPS==(500,1000,1500)
    assert module.METHOD['weights']==[1/3]*3 and module.METHOD['candidate_count']==1
    assert module.METHOD['optimizer_steps_executed']==0 and module.METHOD['source_optimizer_steps_executed']==1500
    assert module.METHOD['declared_after_inspecting_source_calibration'] is True
    for key in ('predeclared_before_source_training','ratio_search','subset_search','teacher_in_model','output_averaging'):
        assert module.METHOD[key] is False
    args=module.parser().parse_args([])
    assert args.device=='mps' and args.output.name=='average-core-v1'
    assert not hasattr(args,'alpha') and not hasattr(args,'steps')
