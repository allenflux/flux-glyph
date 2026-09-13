#!/usr/bin/env python3
"""Evaluate one sealed Android OVR TEST, after CAL selection and full ONNX parity."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'training')]
from train_regions import sha,dump,require
from prepare_android_regions import load_split,UNKNOWN,RECIPES
from train_android_regions import POLICY,region_outputs,decisions
from export_android_ovr import validate,export_sources,ARCHITECTURE
from evaluate_android_regions import summarize,holdout_evidence


def evaluation_sources():
    return export_sources()+[ROOT/'training/evaluate_android_ovr.py',ROOT/'training/evaluate_android_regions.py',
        ROOT/'src/flux_glyph/android_font.py',ROOT/'src/flux_glyph/region_font.py']


def validate_export(args):
    """Check all sealed export evidence before constructing a model or reading TEST."""
    selection=validate(args.run,args.data)
    parity=json.loads((args.run/'PARITY.json').read_text())
    require(parity.get('schema')=='flux-glyph-android-onnx-parity-v1' and parity.get('passed') is True
            and parity.get('test_read') is False and parity.get('selection_sha256')==sha(args.run/'SELECTION.json')
            and parity.get('checkpoint_sha256')==sha(args.run/'model.pth'),
            'export must be frozen and verified before TEST')
    expected_sources={str(p.resolve()) for p in export_sources()}
    require(isinstance(parity.get('source_bindings'),dict) and set(parity['source_bindings'])==expected_sources,
            'incomplete export provenance')
    require(all(Path(p).is_file() and sha(p)==s for p,s in parity['source_bindings'].items()),
            'export/lowering source changed before TEST')
    require(all(parity.get(field) is True for field in ('calibration_font_decisions_identical',
            'original_torch_groupnorm_reference','calibration_size_availability_identical','calibration_size_values_close',
            'cached_mps_outputs_reproduce_selected_metrics'))
            and parity.get('size_value_atol_px')==.02 and parity.get('size_value_rtol')==.0003,
            'export lacks verified font/size parity evidence')
    batches=parity.get('batch_checks')
    require(isinstance(batches,list) and len(batches)==4 and {r.get('batch_size') for r in batches}=={1,7,32,128}
            and all(r.get('passed') is True and r.get('font_and_size_checked') is True
                    and type(r.get('samples')) is int and r['samples']>0 for r in batches),
            'export batch parity evidence is incomplete')
    require(sha(args.region/'metadata.json')==parity['metadata_sha256'] and sha(args.region/'model.onnx')==parity['model_sha256'],
            'TEST model differs from frozen export')
    require(parity.get('architecture')==ARCHITECTURE and parity.get('unknown_reference_logit_zero') is True
            and parity.get('encoder_and_size_head_frozen') is True
            and parity.get('cached_training_outputs_reproduce_selected_metrics') is True
            and parity.get('cached_output_execution')==selection['cached_output_execution']
            and parity.get('parent_checkpoint_sha256')==selection['parent_checkpoint']['sha256']
            and parity.get('parent_frozen_state_sha256')==selection['parent_frozen_state_sha256'],
            'OVR export evidence is incomplete')
    return selection,parity


def evaluate(args):
    from flux_glyph.android_font import AndroidFontClassifier
    require(not args.output.exists(),'retain the existing sealed test report')
    selection,parity=validate_export(args)
    model=AndroidFontClassifier(args.region);selected=selection['selected']
    require(model.meta.get('network_architecture')==ARCHITECTURE,'TEST runtime architecture differs from frozen OVR')
    require(model.output_families==selection['families'] and model.meta['temperature']==selected['temperature']
            and model.meta['gates']==selected['gates'] and model.meta['max_size_relative_spread']==POLICY['max_size_relative_spread'],
            'TEST family order or runtime gates differ from selected CAL')
    training=model.meta.get('training',{})
    require(training.get('architecture')==ARCHITECTURE and training.get('encoder_and_size_head_frozen') is True
            and training.get('cached_output_execution')==parity['cached_output_execution']
            and training.get('parent_checkpoint_sha256')==parity.get('parent_checkpoint_sha256')
            and training.get('parent_frozen_state_sha256')==parity.get('parent_frozen_state_sha256'),'OVR parent identity differs')
    require(training.get('selection_sha256')==sha(args.run/'SELECTION.json')
            and training.get('checkpoint_sha256')==parity['checkpoint_sha256']
            and training.get('data_manifest_sha256')==selection['data_manifest_sha256'],
            'TEST runtime metadata differs from frozen training provenance')
    require(validate_export(args)==(selection,parity),'export changed before TEST freeze')
    # This evaluator and its source digest are recorded before opening a TEST array.
    source_sha=sha(__file__)
    source_bindings={str(path.resolve()):sha(path) for path in evaluation_sources()}
    freeze={'architecture':ARCHITECTURE,'source_bindings':source_bindings,'schema':'flux-glyph-android-test-freeze-v1','selection_sha256':sha(args.run/'SELECTION.json'),
            'parity_sha256':sha(args.run/'PARITY.json'),'evaluation_source_sha256':source_sha,
            'model_sha256':parity['model_sha256'],'metadata_sha256':parity['metadata_sha256'],
            'checkpoint_sha256':parity['checkpoint_sha256'],
            'policy':POLICY,'selected_runtime':{'temperature':selected['temperature'],'gates':selected['gates']},
            'test_inference_started':False,'test_used_to_adjust_model_or_gates':False}
    freeze_path=args.output.with_name(args.output.stem+'_FREEZE.json')
    require(not freeze_path.exists(),'this TEST evaluation was already started; investigate rather than silently repeat it')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with freeze_path.open('x') as stream:json.dump(freeze,stream,ensure_ascii=False,indent=2);stream.write('\n')
    freeze_sha=sha(freeze_path)
    data=load_split(args.data,'test')
    require(data['partition']['frozen_selection']['sha256']==sha(args.run/'SELECTION.json')
            and data['families']==selection['families'] and data['manifest_sha256']==selection['data_manifest_sha256'],
            'TEST was not derived for this frozen selection')
    require(data['manifest']['views']==RECIPES and set(r['view'] for r in data['rows'])==set(RECIPES),
            'TEST view contract differs')
    holdouts=holdout_evidence(data)
    logits=[];ratios=[]
    for start in range(0,len(data['tiles']),128):
        block=np.array(data['tiles'][start:start+128],copy=True)
        family,size=model.session.run(['logits','log_em_ratio'],{'tiles':block})
        require(family.dtype==size.dtype==np.float32 and family.shape==(len(block),10) and size.shape==(len(block),)
                and np.all(family[:,selection['families'].index(UNKNOWN)]==0)
                and np.isfinite(family).all() and np.isfinite(size).all() and np.all(np.abs(size)<=3),'invalid TEST model output')
        logits.append(family);ratios.append(size)
    outputs=region_outputs(np.concatenate(logits),np.concatenate(ratios),data['rows'],selected['temperature'])
    details=decisions(outputs,data['rows'],data['families'],selected['gates'])
    measured,by_view,confusion,denominators=summarize(details,data['families'],data['partition']['rejected'],RECIPES)
    require(sha(args.run/'SELECTION.json')==freeze['selection_sha256'] and sha(args.run/'PARITY.json')==freeze['parity_sha256']
            and sha(args.run/'model.pth')==freeze['checkpoint_sha256'] and sha(freeze_path)==freeze_sha
            and all(sha(path)==digest for path,digest in source_bindings.items())
            and sha(__file__)==source_sha and validate_export(args)==(selection,parity),'source/model changed during TEST')
    require(sha(args.data/'test/MANIFEST.json')==data['partition_sha256']
            and all(sha(args.data/'test'/data['partition'][key]['path'])==data['partition'][key]['sha256'] for key in ('array','metadata')),
            'TEST data changed during evaluation')
    report={'architecture':ARCHITECTURE,'source_bindings':source_bindings,'schema':'flux-glyph-android-test-v1','passed':measured['passed'],'metrics':measured,'by_view':by_view,'confusion':confusion,
        'rows':details,'preprocessing_rejected':data['partition']['rejected'],
        'denominators':denominators,'holdout_evidence':holdouts,
        'selection_sha256':sha(args.run/'SELECTION.json'),'parity_sha256':sha(args.run/'PARITY.json'),
        'test_freeze_sha256':freeze_sha,'test_partition_sha256':data['partition_sha256'],
        'model_sha256':parity['model_sha256'],'evaluation_source_sha256':source_sha,
        'test_used_for_training_or_selection':False,'test_inference_completed':True,
        'scope':'Controlled native Android emulator font regions, four correlated views per source; user-selected scope is not proof of device OS or font provenance.'}
    with args.output.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'passed':report['passed'],'metrics':measured},ensure_ascii=False,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','data','region','output'):p.add_argument('--'+name,type=Path,required=True)
    evaluate(p.parse_args())
