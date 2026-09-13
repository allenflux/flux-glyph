#!/usr/bin/env python3
"""Export a frozen-encoder Android OVR model; verify complete CAL font/size parity."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'training')]
from train_regions import sha,dump,require,state_sha
from train_android_regions import POLICY,region_outputs,decisions,metrics
from prepare_android_regions import load_split,UNKNOWN
from export_android_regions import validate as validate_base,compare_calibration_outputs,SIZE_ATOL_PX,SIZE_RTOL

ARCHITECTURE='android_ovr_frozen_encoder_9x_mlp128_32_1_v1'


def export_sources():
    return [ROOT/'training'/name for name in ('export_android_ovr.py','android_ovr_network.py',
        'export_android_regions.py','export_region_stable.py','train_android_regions.py',
        'prepare_android_regions.py','region_network.py','network.py','train_regions.py')]


def validate(run,data):
    selection=validate_base(run,data)
    protocol=json.loads((Path(run)/'TRAINING_FREEZE.json').read_text())
    require(selection.get('architecture')==protocol.get('architecture')==ARCHITECTURE,
            'OVR architecture is not recorded consistently')
    for name in ('train_android_ovr.py','android_ovr_network.py'):
        path=ROOT/'training'/name
        require(selection['bindings'].get(str(path.resolve()))==sha(path),'OVR training source is not bound')
    for key in ('parent_checkpoint','cache_manifest'):
        evidence=selection.get(key)
        require(isinstance(evidence,dict) and evidence==protocol.get(key)
                and isinstance(evidence.get('path'),str) and Path(evidence['path']).is_absolute()
                and selection['bindings'].get(evidence['path'])==evidence.get('sha256')
                and Path(evidence['path']).is_file() and sha(evidence['path'])==evidence['sha256'],
                'OVR parent or cache evidence differs from its frozen binding')
    for key in ('parent_frozen_state_sha256','heads_before_sha256','heads_after_sha256'):
        digest=selection.get(key)
        require(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),
                'OVR parameter state evidence is missing')
    require(selection['parent_frozen_state_sha256']==protocol.get('parent_frozen_state_sha256')
            and selection['heads_before_sha256']==protocol.get('heads_before_sha256')
            and selection['heads_before_sha256']!=selection['heads_after_sha256'],
            'OVR heads were not trained or frozen parent identity changed')
    execution=selection.get('cached_output_execution')
    require(isinstance(execution,dict) and execution==protocol.get('cached_output_execution')
            and set(execution)=={'features_device','heads_device','size_head_device'}
            and all(value in ('cpu','mps') for value in execution.values())
            and execution['heads_device']==selection.get('training_device')==protocol.get('device')
            and execution['size_head_device']==execution['features_device'],
            'OVR cached output execution provenance is missing or inconsistent')
    initial=Path(run).resolve()/'INITIAL_HEADS.pth'
    require(initial.is_file() and selection['bindings'].get(str(initial))==sha(initial),'initial OVR heads are not bound')
    return selection


def require_zero_unknown(logits,unknown_index):
    require(isinstance(logits,np.ndarray) and logits.dtype==np.float32 and logits.ndim==2
            and logits.shape[1]==10 and len(logits)>0 and np.isfinite(logits).all()
            and type(unknown_index) is int and 0<=unknown_index<10
            and np.all(logits[:,unknown_index]==0),'OVR unknown reference logit or output contract changed')


def validate_checkpoint(checkpoint,selection,run):
    """Verify the nine trained heads and every unchanged parent encoder/size tensor."""
    import torch
    require(checkpoint.get('architecture')==ARCHITECTURE
            and checkpoint.get('unknown_index')==selection['families'].index(UNKNOWN),
            'OVR checkpoint architecture or unknown index differs')
    evidence=selection['parent_checkpoint']
    require(sha(evidence['path'])==evidence['sha256']==checkpoint.get('parent_checkpoint_sha256')
            and checkpoint.get('parent_frozen_state_sha256')==selection['parent_frozen_state_sha256'],
            'OVR checkpoint parent binding differs')
    parent=torch.load(evidence['path'],map_location='cpu',weights_only=True)
    require(parent.get('families')==selection['families'],'OVR parent family order differs')
    state=checkpoint.get('state_dict');parent_state=parent.get('state_dict')
    require(isinstance(state,dict) and isinstance(parent_state,dict),'OVR parameter mappings are missing')
    frozen={key:value for key,value in state.items() if not key.startswith('heads.')}
    heads={key:value for key,value in state.items() if key.startswith('heads.')}
    expected={key:value for key,value in parent_state.items() if not key.startswith('family_head.')}
    require(set(parent_state)==set(expected)|{'family_head.weight','family_head.bias'}
            and set(frozen)==set(expected) and bool(heads),'OVR inherited parameter inventory differs')
    require(all(isinstance(value,torch.Tensor) and value.dtype==torch.float32 and bool(torch.isfinite(value).all())
                for value in (*state.values(),*parent_state.values())),'OVR or parent tensors are invalid')
    require(all(value.shape==expected[key].shape and value.dtype==expected[key].dtype
                and torch.equal(value,expected[key]) for key,value in frozen.items()),
            'OVR encoder or size head changed from the frozen parent')
    require(state_sha(frozen)==state_sha(expected)==selection['parent_frozen_state_sha256']
            and state_sha(heads)==selection['heads_after_sha256'],'OVR frozen/base or trained-head SHA differs')
    initial_path=Path(run).resolve()/'INITIAL_HEADS.pth'
    require(initial_path.is_file() and selection['bindings'].get(str(initial_path))==sha(initial_path),
            'initial OVR head file changed')
    initial=torch.load(initial_path,map_location='cpu',weights_only=True)
    require(isinstance(initial,dict) and set(initial)==set(heads)
            and all(isinstance(value,torch.Tensor) and value.shape==heads[key].shape and value.dtype==heads[key].dtype
                    and bool(torch.isfinite(value).all()) for key,value in initial.items()),'invalid initial OVR head parameters')
    require(state_sha(initial)==selection['heads_before_sha256']
            and state_sha({**frozen,**initial})==selection['state_before_sha256'],
            'initial OVR parameter states do not reproduce the training freeze')


def export(args):
    import torch
    import onnx
    import onnxruntime as ort
    from android_ovr_network import AndroidOVRClassifier,ARCHITECTURE as NETWORK_ARCHITECTURE
    from export_region_stable import replace_groupnorm
    from flux_glyph.android_font import AndroidFontClassifier
    require(not args.output.exists() and not (args.run/'PARITY.json').exists(),'export output must be new')
    selection=validate(args.run,args.data);cal=load_split(args.data,'calibration')
    require(NETWORK_ARCHITECTURE==ARCHITECTURE,'OVR network architecture identifier changed')
    require(cal['families']==selection['families'],'export CAL family order changed')
    checkpoint=torch.load(args.run/'model.pth',map_location='cpu',weights_only=True)
    checkpoint_sha=sha(args.run/'model.pth')
    require(checkpoint['families']==selection['families'] and checkpoint['selection_sha256']==sha(args.run/'SELECTION.json')
            and state_sha(checkpoint['state_dict'])==selection['state_after_sha256'],'selected checkpoint binding differs')
    sources={str(path.resolve()):sha(path) for path in export_sources()}
    validate_checkpoint(checkpoint,selection,args.run)
    reference=AndroidOVRClassifier(len(selection['families']),selection['families'].index(UNKNOWN)).eval()
    reference.load_state_dict(checkpoint['state_dict'],strict=True)
    converted=copy.deepcopy(reference)
    require(replace_groupnorm(converted,high_precision=True)==4 and state_sha(converted.state_dict())==state_sha(reference.state_dict()),
            'GroupNorm lowering changed parameters')
    torch.set_num_threads(4);args.output.mkdir(parents=True);path=args.output/'model.onnx'
    torch.onnx.export(converted,torch.zeros(2,1,64,256),path,input_names=['tiles'],output_names=['logits','log_em_ratio'],
        dynamic_axes={'tiles':{0:'batch'},'logits':{0:'batch'},'log_em_ratio':{0:'batch'}},opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    actual_logits=[];actual_sizes=[];reference_logits=[];reference_sizes=[];max_logit=0.;max_size=0.
    with torch.inference_mode():
        for start in range(0,len(cal['tiles']),128):
            block=np.array(cal['tiles'][start:start+128],copy=True);logits,sizes=reference(torch.from_numpy(block))
            out,out_size=session.run(None,{'tiles':block})
            require_zero_unknown(out,selection['families'].index(UNKNOWN))
            require_zero_unknown(logits.numpy(),selection['families'].index(UNKNOWN))
            np.testing.assert_allclose(out,logits.numpy(),atol=2e-4,rtol=2e-4)
            np.testing.assert_allclose(out_size,sizes.numpy(),atol=2e-4,rtol=2e-4)
            max_logit=max(max_logit,float(np.max(np.abs(out-logits.numpy()))));max_size=max(max_size,float(np.max(np.abs(out_size-sizes.numpy()))))
            actual_logits.append(out);actual_sizes.append(out_size);reference_logits.append(logits.numpy());reference_sizes.append(sizes.numpy())
    actual_logits=np.concatenate(actual_logits);actual_sizes=np.concatenate(actual_sizes)
    reference_logits=np.concatenate(reference_logits);reference_sizes=np.concatenate(reference_sizes)
    selected=selection['selected'];temperature=selected['temperature'];gates=selected['gates']
    actual=region_outputs(actual_logits,actual_sizes,cal['rows'],temperature)
    expected=region_outputs(reference_logits,reference_sizes,cal['rows'],temperature)
    cached=json.loads((args.run/'CALIBRATION_DECISIONS.json').read_text())
    require(cached['families']==selection['families'],'cached family order differs')
    with np.load(args.run/'CALIBRATION_OUTPUTS.npz',allow_pickle=False) as stored:
        require(set(stored.files)=={'logits','log_em_ratio'},'cached CAL output schema differs')
        require_zero_unknown(stored['logits'],selection['families'].index(UNKNOWN))
        stored_outputs=region_outputs(stored['logits'],stored['log_em_ratio'],cal['rows'],temperature)
        np.testing.assert_allclose(reference_logits,stored['logits'],atol=3e-4,rtol=3e-4)
        np.testing.assert_allclose(reference_sizes,stored['log_em_ratio'],atol=3e-4,rtol=3e-4)
    require(cached['records']==stored_outputs,'cached CAL decisions differ from the cached training output arrays')
    cached_metrics=metrics(decisions(stored_outputs,cal['rows'],cal['families'],gates),cal['families'],cal['partition']['rejected'])
    require(cached_metrics==selected['metrics'],'cached training outputs do not reproduce every selected CAL metric')
    actual_decisions=decisions(actual,cal['rows'],cal['families'],gates)
    max_size_px=0.
    for reference_outputs in (expected,cached['records']):
        max_size_px=max(max_size_px,compare_calibration_outputs(actual,reference_outputs,cal['rows'],cal['families'],gates))
    measured=metrics(actual_decisions,cal['families'],cal['partition']['rejected'])
    require(measured['passed'] and measured['checks']==selected['metrics']['checks']
            and all(measured[key]==selected['metrics'][key] for key in ['named','correct_named','wrong_named','unknown_wrongly_named']),
            'ONNX changed CAL selection acceptance')
    indices=np.unique(np.linspace(0,len(cal['tiles'])-1,min(256,len(cal['tiles']))).astype(int));sample=np.array(cal['tiles'][indices],copy=True)
    batch_checks=[]
    for count in (1,7,32,128):
        blocks=[session.run(['logits','log_em_ratio'],{'tiles':sample[s:s+count]}) for s in range(0,len(sample),count)]
        out=np.concatenate([block[0] for block in blocks]);size=np.concatenate([block[1] for block in blocks])
        require_zero_unknown(out,selection['families'].index(UNKNOWN))
        np.testing.assert_allclose(out,reference_logits[indices],atol=2e-4,rtol=2e-4)
        np.testing.assert_allclose(size,reference_sizes[indices],atol=2e-4,rtol=2e-4)
        require(out.dtype==size.dtype==np.float32 and out.shape==(len(sample),10) and size.shape==(len(sample),)
                and np.isfinite(out).all() and np.isfinite(size).all(),'invalid dynamic-batch ONNX output')
        batch_checks.append({'batch_size':count,'samples':len(sample),'passed':True,'font_and_size_checked':True})
    metadata={'schema':'flux-glyph-android-region-font-v1','network_architecture':ARCHITECTURE,'algorithm':'android-region-cnn64x256-v1','font_mode':'android',
        'data_kind':'android_emulator_screenshot','model':{'path':'model.onnx','sha256':sha(path)},'families':selection['families'],
        'temperature':temperature,'gates':gates,'max_size_relative_spread':POLICY['max_size_relative_spread'],
        'font_label_groups':{'Noto Sans CJK SC':['Noto Sans CJK SC','Source Han Sans SC'],
                             'Noto Serif CJK SC':['Noto Serif CJK SC','Source Han Serif SC']},
        'font_sources':{family:['asset'] for family in selection['families'] if family!=UNKNOWN},
        'training':{'architecture':ARCHITECTURE,'parent_checkpoint_sha256':checkpoint['parent_checkpoint_sha256'],
                    'parent_frozen_state_sha256':checkpoint['parent_frozen_state_sha256'],'encoder_and_size_head_frozen':True,
                    'cached_output_execution':selection['cached_output_execution'],
                    'selection_sha256':sha(args.run/'SELECTION.json'),'checkpoint_sha256':checkpoint_sha,
                    'data_manifest_sha256':selection['data_manifest_sha256'],'optimizer_steps_executed':selection['optimizer_steps_executed'],
                    'selected_step':selected['step'],'unknown_fonts_held_out_by_family':True,'ocr_text_used':False,'training_device':selection['training_device']}}
    dump(args.output/'metadata.json',metadata);AndroidFontClassifier(args.output)
    require(validate(args.run,args.data)==selection and sha(args.run/'model.pth')==checkpoint_sha
            and all(sha(p)==s for p,s in sources.items()),'frozen source or model changed during export')
    report={'schema':'flux-glyph-android-onnx-parity-v1','architecture':ARCHITECTURE,'passed':True,'test_read':False,
        'unknown_reference_logit_zero':True,'encoder_and_size_head_frozen':True,
        'parent_checkpoint_sha256':checkpoint['parent_checkpoint_sha256'],
        'parent_frozen_state_sha256':checkpoint['parent_frozen_state_sha256'],
        'cached_output_execution':selection['cached_output_execution'],
        'selection_sha256':sha(args.run/'SELECTION.json'),'checkpoint_sha256':checkpoint_sha,'model_sha256':sha(path),
        'metadata_sha256':sha(args.output/'metadata.json'),'source_bindings':sources,
        'calibration_regions':len(cal['rows']),'calibration_tiles':len(cal['tiles']),
        'max_logit_absolute_error':max_logit,'max_size_absolute_error':max_size,
        'calibration_font_decisions_identical':True,'original_torch_groupnorm_reference':True,
        'calibration_size_availability_identical':True,'calibration_size_values_close':True,
        'size_value_atol_px':SIZE_ATOL_PX,'size_value_rtol':SIZE_RTOL,'max_size_pixel_error':max_size_px,
        'cached_mps_outputs_reproduce_selected_metrics':True,
        'cached_training_outputs_reproduce_selected_metrics':True,
        'runtime_calibration_metrics':measured,'batch_checks':batch_checks}
    dump(args.run/'PARITY.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','data','output'):p.add_argument('--'+name,type=Path,required=True)
    export(p.parse_args())
