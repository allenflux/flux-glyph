#!/usr/bin/env python3
"""Post-selection development regression for one unified font CNN; never a blind TEST."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'training')]
from train_regions import sha,require
from prepare_unified_regions import load_split,UNKNOWN,RECIPES
from train_unified_regions import POLICY,ARCHITECTURE,region_outputs,decisions,metrics
from export_unified_regions import validate,export_sources,calibration_evidence,SIZE_ATOL_PX,SIZE_RTOL


def validate_export(args):
    """Verify the fixed classifier and complete CAL parity before opening holdout pixels."""
    selection=validate(args.run,args.data)
    parity=json.loads((args.run/'PARITY.json').read_text())
    require(parity.get('schema')=='flux-glyph-unified-onnx-parity-v1' and parity.get('passed') is True
            and parity.get('test_read') is False and parity.get('development_holdout_read') is False
            and parity.get('network_architecture')==ARCHITECTURE and parity.get('font_model_count')==1
            and parity.get('output_family_count')==len(selection['families'])
            and parity.get('selection_sha256')==sha(args.run/'SELECTION.json')
            and parity.get('checkpoint_sha256')==sha(args.run/'model.pth'),
            'export must be frozen and verified before development regression')
    require(parity.get('source_bindings')==export_sources()
            and parity.get('calibration_bindings')==calibration_evidence(args.run,args.data,selection),
            'export source or CAL evidence changed')
    require(all(parity.get(field) is True for field in ('calibration_font_decisions_identical',
            'original_torch_groupnorm_reference','calibration_size_availability_identical','calibration_size_values_close',
            'calibration_scores_close','cached_training_outputs_reproduce_frozen_grid','cached_mps_outputs_reproduce_selected_metrics'))
            and parity.get('size_value_atol_px')==SIZE_ATOL_PX and parity.get('size_value_rtol')==SIZE_RTOL,
            'export lacks verified font, score or size parity evidence')
    batches=parity.get('batch_checks')
    require(isinstance(batches,list) and len(batches)==4 and {r.get('batch_size') for r in batches}=={1,7,32,128}
            and all(r.get('passed') is True and r.get('font_and_size_checked') is True
                    and type(r.get('samples')) is int and r['samples']>0 for r in batches),
            'export batch parity evidence is incomplete')
    require(sha(args.region/'metadata.json')==parity['metadata_sha256'] and sha(args.region/'model.onnx')==parity['model_sha256'],
            'development model differs from frozen export')
    return selection,parity


def summarize(details,families,rejected,views=RECIPES):
    measured=metrics(details,families,rejected)
    by_view={}
    for view in views:
        value=metrics([r for r in details if r['view']==view],families,[r for r in rejected if r['view']==view])
        # Missing native/domain/class populations in slices are not acceptance results.
        value.update(passed=None,diagnostic_only=True,policy_applied_to='whole development regression only')
        by_view[view]=value
    confusion={}
    for row in details:
        predicted=row['predicted_family'] if row['named'] else '(explicit unknown)' if row['explicit_unknown'] else '(uncertain)'
        bucket=confusion.setdefault(row['family'],{});bucket[predicted]=bucket.get(predicted,0)+1
    for row in rejected:
        bucket=confusion.setdefault(row['family'],{});bucket['(preprocessing rejected)']=bucket.get('(preprocessing rejected)',0)+1
    unknown=[r for r in details if r['family']==UNKNOWN]
    outcomes={'wrongly_named':sum(r['named'] for r in unknown),'explicit_unknown':sum(r['explicit_unknown'] for r in unknown),
        'uncertain_without_name':sum(not r['named'] and not r['explicit_unknown'] for r in unknown),
        'preprocessing_rejected':sum(r['family']==UNKNOWN for r in rejected)}
    require(sum(outcomes.values())==measured['unknown_views']
            and sum(sum(bucket.values()) for bucket in confusion.values())==measured['views'],
            'development diagnostic denominators differ')
    denominators={'known_views_including_rejected':measured['known_views'],
        'unknown_views_including_rejected':measured['unknown_views'],'unknown_outcomes':outcomes,
        'named_precision_population':'All named outputs, including incorrect names on unknown fonts.',
        'unknown_not_named_definition':'Explicit unknown, uncertainty and preprocessing rejection withhold a name; explicit_unknown_recall counts only the unknown class winner.',
        'size_error_population':'Correct names with available size only; missing sizes reduce size coverage.',
        'size_available_on_wrong_names':sum(r['wrong_named'] and r['size_px'] is not None for r in details),
        'size_correct_coverage_of_all_known_views':measured['size']['correctly_named_with_size']/measured['known_views'] if measured['known_views'] else None,
        'native_regions':len({(r['domain'],r['source_id'],r['region_id']) for r in [*details,*rejected]}),
        'pages':len({(r['domain'],r['source_id']) for r in [*details,*rejected]}),
        'derived_views_are_correlated':True,'color_accuracy_evaluated':False}
    return measured,by_view,confusion,denominators


def evaluate(args):
    from flux_glyph.unified_font import UnifiedFontClassifier
    require(not args.output.exists(),'retain the existing development regression report')
    selection,parity=validate_export(args)
    model=UnifiedFontClassifier(args.region);selected=selection['selected']
    require(model.output_families==selection['families'] and model.meta['temperature']==selected['temperature']
            and model.meta['gates']==selected['gates'] and model.meta['max_size_relative_spread']==POLICY['max_size_relative_spread'],
            'development family order or runtime gates differ from CAL selection')
    training=model.meta.get('training',{})
    require(training.get('selection_sha256')==sha(args.run/'SELECTION.json')
            and training.get('checkpoint_sha256')==parity['checkpoint_sha256']
            and training.get('data_manifest_sha256')==selection['data_manifest_sha256'],
            'runtime metadata differs from frozen training provenance')
    require(validate_export(args)==(selection,parity),'export changed before evaluation freeze')
    source_sha=sha(__file__);split='development_holdout'
    freeze={'schema':'flux-glyph-unified-development-freeze-v1','evaluation_kind':'development_regression',
        'selection_sha256':sha(args.run/'SELECTION.json'),'parity_sha256':sha(args.run/'PARITY.json'),
        'evaluation_source_sha256':source_sha,'model_sha256':parity['model_sha256'],
        'metadata_sha256':parity['metadata_sha256'],'checkpoint_sha256':parity['checkpoint_sha256'],
        'development_partition_sha256':sha(args.data/split/'MANIFEST.json'),
        'policy':POLICY,'selected_runtime':{'temperature':selected['temperature'],'gates':selected['gates']},
        'split':split,'test_read':False,'blind_test_performed':False,'development_inference_started':False,
        'used_to_adjust_model_or_gates':False}
    freeze_path=args.output.with_name(args.output.stem+'_FREEZE.json')
    require(not freeze_path.exists(),'development evaluation already started; do not silently repeat')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with freeze_path.open('x') as stream:json.dump(freeze,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    freeze_sha=sha(freeze_path)
    data=load_split(args.data,split,allow_holdout=True)
    require(data['families']==selection['families'] and data['manifest_sha256']==selection['data_manifest_sha256']
            and data['partition_sha256']==freeze['development_partition_sha256'], 'development partition differs from frozen union')
    require(data['rows'] and all(row['split']==split and row['view'] in RECIPES for row in data['rows']),
            'development view contract differs')
    logits=[];ratios=[]
    for start in range(0,len(data['tiles']),128):
        block=np.array(data['tiles'][start:start+128],copy=True)
        family,size=model.session.run(['logits','log_em_ratio'],{'tiles':block})
        require(family.dtype==size.dtype==np.float32 and family.shape==(len(block),len(data['families'])) and size.shape==(len(block),)
                and np.isfinite(family).all() and np.isfinite(size).all() and np.all(np.abs(size)<=3),
                'invalid development model output')
        logits.append(family);ratios.append(size)
    outputs=region_outputs(np.concatenate(logits),np.concatenate(ratios),data['rows'],selected['temperature'])
    details=decisions(outputs,data['rows'],data['families'],selected['gates'])
    measured,by_view,confusion,denominators=summarize(details,data['families'],data['partition']['rejected'])
    require(sha(args.run/'SELECTION.json')==freeze['selection_sha256'] and sha(args.run/'PARITY.json')==freeze['parity_sha256']
            and sha(args.run/'model.pth')==freeze['checkpoint_sha256'] and sha(freeze_path)==freeze_sha
            and sha(__file__)==source_sha and validate_export(args)==(selection,parity),
            'source/model changed during development regression')
    require(sha(args.data/split/'MANIFEST.json')==data['partition_sha256']
            and all(sha(args.data/split/data['partition'][key]['path'])==data['partition'][key]['sha256'] for key in ('array','metadata')),
            'development data changed during evaluation')
    report={'schema':'flux-glyph-unified-development-regression-v1','evaluation_kind':'development_regression',
        'development_policy_passed':measured['passed'],'stable_validation_passed':False,'test_passed':False,
        'metrics':measured,'by_domain':measured['per_domain'],'by_family':measured['per_family'],
        'unknown_sources':measured['unknown_sources'],'by_view':by_view,'confusion':confusion,
        'rows':details,'preprocessing_rejected':data['partition']['rejected'],'denominators':denominators,
        'selection_sha256':sha(args.run/'SELECTION.json'),'parity_sha256':sha(args.run/'PARITY.json'),
        'development_freeze_sha256':freeze_sha,'development_partition_sha256':data['partition_sha256'],
        'model_sha256':parity['model_sha256'],'evaluation_source_sha256':source_sha,
        'calibration_passed':selection['passed'],'experimental_precision_priority_met':selected['metrics']['experimental_precision_priority_met'],
        'test_read':False,'blind_test_performed':False,'used_to_adjust_model_or_gates':False,
        'development_inference_completed':True,'model_count':1,'platform_routing':False,
        'scope':'Development regression on controlled native mobile regions and correlated views. Historical initialization may have seen some source images. This is not an independent blind TEST, device identification or real-world accuracy estimate.'}
    with args.output.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'development_policy_passed':measured['passed'],'metrics':measured},ensure_ascii=False,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','data','region','output'):p.add_argument('--'+name,type=Path,required=True)
    evaluate(p.parse_args())
