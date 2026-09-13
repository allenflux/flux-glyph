#!/usr/bin/env python3
"""Export the frozen Android CNN and verify CAL decisions against original PyTorch."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'training')]
from train_regions import sha,dump,require,state_sha
from train_android_regions import POLICY,region_outputs,decisions,metrics
from prepare_android_regions import load_split,UNKNOWN

SIZE_ATOL_PX=.02
SIZE_RTOL=3e-4


def selection_rank(record):
    measured=record['metrics'];size=measured['size']['median_ape'];nll=record['calibration_nll']
    require(type(nll) in (int,float) and math.isfinite(nll) and nll>=0,'invalid calibration NLL')
    require(size is None or type(size) in (int,float) and math.isfinite(size) and size>=0,'invalid calibration size error')
    return (int(measured['passed']),measured['correct_named'],-measured['wrong_named'],
            -(size if size is not None else 1e6),-nll,-record['step'],-record['grid_index'])


def compare_calibration_outputs(actual,other,rows,families,gates):
    """Compare individual serving decisions, including size withholding."""
    actual_decisions=decisions(actual,rows,families,gates)
    other_decisions=decisions(other,rows,families,gates)
    require([(r['predicted_family'],r['named']) for r in actual_decisions]
            ==[(r['predicted_family'],r['named']) for r in other_decisions],
            'ONNX or CPU changed a frozen CAL naming decision')
    require([r['size_px'] is not None for r in actual_decisions]==[r['size_px'] is not None for r in other_decisions],
            'ONNX or CPU changed a frozen CAL size availability decision')
    np.testing.assert_allclose([r['probabilities'] for r in actual],[r['probabilities'] for r in other],atol=2e-5,rtol=2e-4)
    actual_size=np.array([r['size_px'] for r in actual_decisions if r['size_px'] is not None])
    other_size=np.array([r['size_px'] for r in other_decisions if r['size_px'] is not None])
    np.testing.assert_allclose(actual_size,other_size,atol=SIZE_ATOL_PX,rtol=SIZE_RTOL)
    return float(np.max(np.abs(actual_size-other_size))) if len(actual_size) else 0.


def validate(run,data):
    run,data=Path(run).resolve(),Path(data).resolve()
    selection=json.loads((run/'SELECTION.json').read_text())
    require(selection.get('schema')=='flux-glyph-android-training-selection-v1' and selection.get('passed') is True
            and selection.get('calibration_passed') is True and selection.get('test_read') is False
            and selection.get('policy')==POLICY,'Android export requires the original successful CAL selection')
    bindings=selection['bindings']
    for path in [data/'MANIFEST.json',data/'train/MANIFEST.json',data/'calibration/MANIFEST.json',
                 ROOT/'training/train_android_regions.py',ROOT/'training/prepare_android_regions.py',ROOT/'src/flux_glyph/android_font.py',
                 ROOT/'src/flux_glyph/region_font.py',ROOT/'training/region_network.py',
                 ROOT/'training/network.py',ROOT/'training/train_regions.py']:
        require(str(path.resolve()) in bindings and bindings[str(path.resolve())]==sha(path),'Android export input differs from training')
    require(all(Path(path).is_file() and sha(path)==digest for path,digest in bindings.items()),'frozen Android training source changed')
    require(selection['data_manifest_sha256']==sha(data/'MANIFEST.json')
            and sha(run/'TRAINING_FREEZE.json')==selection['training_protocol_sha256']
            and sha(run/'CALIBRATION_DECISIONS.json')==selection['calibration_decisions_sha256']
            and sha(run/'CALIBRATION_OUTPUTS.npz')==selection['calibration_outputs_sha256'],'saved Android training evidence changed')
    protocol=json.loads((run/'TRAINING_FREEZE.json').read_text())
    require(protocol['policy']==POLICY and protocol['bindings']==bindings and protocol['families']==selection['families']
            and protocol['steps']==selection['optimizer_steps_executed'] and protocol['test_read'] is False
            and protocol['initial_state_sha256']==selection['state_before_sha256']
            and selection['state_before_sha256']!=selection['state_after_sha256'],
            'training protocol differs from final selection')
    steps=protocol['steps'];history=selection['history']
    require(type(steps) is int and steps>=1000 and steps%500==0,'invalid completed optimizer step count')
    expected_steps=list(range(1000,steps+1,1000))
    if steps%1000:expected_steps.append(steps)
    require(isinstance(history,list) and [r['step'] for r in history]==expected_steps,
            'CAL history does not cover the declared completed training')
    grid=[(temperature,score,margin,agreement) for temperature in POLICY['temperatures']
          for score in POLICY['score_gates'] for margin in POLICY['margin_gates'] for agreement in POLICY['agreement_gates']]
    for record in history:
        index=record['grid_index'];gates=record['gates']
        require(type(index) is int and 0<=index<len(grid) and grid[index]==
                (record['temperature'],gates['min_score'],gates['min_margin'],gates['min_patch_agreement']),
                'CAL record differs from its frozen grid index')
    selected=selection['selected']
    require(selected in history and selected==max(history,key=selection_rank) and selected['temperature'] in POLICY['temperatures']
            and selected['gates']['min_score'] in POLICY['score_gates']
            and selected['gates']['min_margin'] in POLICY['margin_gates']
            and selected['gates']['min_patch_agreement'] in POLICY['agreement_gates'],'selected gates were not in the frozen search grid')
    require(selected['metrics']['passed'] and all(selected['metrics']['checks'].values()),'selected candidate did not pass CAL')
    return selection


def export(args):
    import torch
    import onnx
    import onnxruntime as ort
    from region_network import RegionFontClassifier
    from export_region_stable import replace_groupnorm
    from flux_glyph.android_font import AndroidFontClassifier
    require(not args.output.exists() and not (args.run/'PARITY.json').exists(),'export output must be new')
    selection=validate(args.run,args.data);cal=load_split(args.data,'calibration')
    require(cal['families']==selection['families'],'export CAL family order changed')
    checkpoint=torch.load(args.run/'model.pth',map_location='cpu',weights_only=True)
    checkpoint_sha=sha(args.run/'model.pth')
    require(checkpoint['families']==selection['families'] and checkpoint['selection_sha256']==sha(args.run/'SELECTION.json')
            and state_sha(checkpoint['state_dict'])==selection['state_after_sha256'],'selected checkpoint binding differs')
    sources={str(path.resolve()):sha(path) for path in [Path(__file__),ROOT/'training/export_region_stable.py',ROOT/'training/train_regions.py']}
    reference=RegionFontClassifier(len(selection['families'])).eval();reference.load_state_dict(checkpoint['state_dict'],strict=True)
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
        stored_outputs=region_outputs(stored['logits'],stored['log_em_ratio'],cal['rows'],temperature)
        np.testing.assert_allclose(reference_logits,stored['logits'],atol=3e-4,rtol=3e-4)
        np.testing.assert_allclose(reference_sizes,stored['log_em_ratio'],atol=3e-4,rtol=3e-4)
    require(cached['records']==stored_outputs,'cached CAL decisions differ from the frozen MPS output arrays')
    cached_metrics=metrics(decisions(stored_outputs,cal['rows'],cal['families'],gates),cal['families'],cal['partition']['rejected'])
    require(cached_metrics==selected['metrics'],'frozen MPS outputs do not reproduce every selected CAL metric')
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
        np.testing.assert_allclose(out,reference_logits[indices],atol=2e-4,rtol=2e-4)
        np.testing.assert_allclose(size,reference_sizes[indices],atol=2e-4,rtol=2e-4)
        require(out.dtype==size.dtype==np.float32 and out.shape==(len(sample),10) and size.shape==(len(sample),)
                and np.isfinite(out).all() and np.isfinite(size).all(),'invalid dynamic-batch ONNX output')
        batch_checks.append({'batch_size':count,'samples':len(sample),'passed':True,'font_and_size_checked':True})
    metadata={'schema':'flux-glyph-android-region-font-v1','algorithm':'android-region-cnn64x256-v1','font_mode':'android',
        'data_kind':'android_emulator_screenshot','model':{'path':'model.onnx','sha256':sha(path)},'families':selection['families'],
        'temperature':temperature,'gates':gates,'max_size_relative_spread':POLICY['max_size_relative_spread'],
        'font_label_groups':{'Noto Sans CJK SC':['Noto Sans CJK SC','Source Han Sans SC'],
                             'Noto Serif CJK SC':['Noto Serif CJK SC','Source Han Serif SC']},
        'font_sources':{family:['asset'] for family in selection['families'] if family!=UNKNOWN},
        'training':{'selection_sha256':sha(args.run/'SELECTION.json'),'checkpoint_sha256':checkpoint_sha,
                    'data_manifest_sha256':selection['data_manifest_sha256'],'optimizer_steps_executed':selection['optimizer_steps_executed'],
                    'selected_step':selected['step'],'unknown_fonts_held_out_by_family':True,'ocr_text_used':False,'training_device':selection['training_device']}}
    dump(args.output/'metadata.json',metadata);AndroidFontClassifier(args.output)
    require(validate(args.run,args.data)==selection and sha(args.run/'model.pth')==checkpoint_sha
            and all(sha(p)==s for p,s in sources.items()),'frozen source or model changed during export')
    report={'schema':'flux-glyph-android-onnx-parity-v1','passed':True,'test_read':False,
        'selection_sha256':sha(args.run/'SELECTION.json'),'checkpoint_sha256':checkpoint_sha,'model_sha256':sha(path),
        'metadata_sha256':sha(args.output/'metadata.json'),'source_bindings':sources,
        'calibration_regions':len(cal['rows']),'calibration_tiles':len(cal['tiles']),
        'max_logit_absolute_error':max_logit,'max_size_absolute_error':max_size,
        'calibration_font_decisions_identical':True,'original_torch_groupnorm_reference':True,
        'calibration_size_availability_identical':True,'calibration_size_values_close':True,
        'size_value_atol_px':SIZE_ATOL_PX,'size_value_rtol':SIZE_RTOL,'max_size_pixel_error':max_size_px,
        'cached_mps_outputs_reproduce_selected_metrics':True,
        'runtime_calibration_metrics':measured,'batch_checks':batch_checks}
    dump(args.run/'PARITY.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','data','output'):p.add_argument('--'+name,type=Path,required=True)
    export(p.parse_args())
