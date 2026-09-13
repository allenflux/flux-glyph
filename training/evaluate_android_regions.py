#!/usr/bin/env python3
"""One sealed Android TEST evaluation of a CAL-selected, parity-verified ONNX."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'training')]
from train_regions import sha,dump,require
from prepare_android_regions import load_split,UNKNOWN,RECIPES,normalized_text
from train_android_regions import POLICY,region_outputs,decisions,metrics
from export_android_regions import validate


def validate_export(args):
    """Check all sealed export evidence before constructing a model or reading TEST."""
    selection=validate(args.run,args.data)
    parity=json.loads((args.run/'PARITY.json').read_text())
    require(parity.get('schema')=='flux-glyph-android-onnx-parity-v1' and parity.get('passed') is True
            and parity.get('test_read') is False and parity.get('selection_sha256')==sha(args.run/'SELECTION.json')
            and parity.get('checkpoint_sha256')==sha(args.run/'model.pth'),
            'export must be frozen and verified before TEST')
    expected_sources={str(p.resolve()) for p in [ROOT/'training/export_android_regions.py',
                     ROOT/'training/export_region_stable.py',ROOT/'training/train_regions.py']}
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
    return selection,parity


def summarize(details,families,rejected,views):
    """The whole sealed TEST has one policy result; view slices are diagnostic."""
    measured=metrics(details,families,rejected)
    by_view={}
    for view in views:
        visible=[r for r in details if r['view']==view]
        rejected_view=[r for r in rejected if r['view']==view]
        value=metrics(visible,families,rejected_view)
        value.update(passed=None,diagnostic_only=True,policy_applied_to='whole TEST only',not_applicable_checks=[])
        if view!='native':
            value['native_known_top1_accuracy']=None
            value['checks']['native_known_top1']=None
            value['not_applicable_checks']=['native_known_top1']
        if not value['unknown_views']:
            value['unknown_not_named_rate']=None;value['explicit_unknown_recall']=None
            value['checks']['unseen_unknown_not_named']=None
            value['not_applicable_checks'].append('unseen_unknown_not_named')
        if not value['named']:value['named_precision']=None
        by_view[view]=value
    confusion={}
    for row in details:
        key=row['family'];pred=row['predicted_family'] if row['named'] else '(abstain)'
        confusion.setdefault(key,{})
        confusion[key][pred]=confusion[key].get(pred,0)+1
    for row in rejected:
        confusion.setdefault(row['family'],{})
        key='(preprocessing rejected)'
        confusion[row['family']][key]=confusion[row['family']].get(key,0)+1
    unknown=[r for r in details if r['family']==UNKNOWN]
    known=[r for r in details if r['family']!=UNKNOWN]
    outcome_counts={'wrongly_named':sum(r['named'] for r in unknown),
        'explicit_unknown':sum(r['explicit_unknown'] for r in unknown),
        'uncertain_without_name':sum(not r['named'] and not r['explicit_unknown'] for r in unknown),
        'preprocessing_rejected':sum(r['family']==UNKNOWN for r in rejected)}
    require(sum(outcome_counts.values())==measured['unknown_views']
            and sum(sum(c.values()) for c in confusion.values())==measured['views'],'TEST diagnostic denominators differ')
    denominators={'known_views_including_rejected':measured['known_views'],
        'unknown_views_including_rejected':measured['unknown_views'],
        'native_known_regions_including_rejected':sum(r['view']=='native' for r in known)
            +sum(r['view']=='native' and r['family']!=UNKNOWN for r in rejected),
        'named_precision_population':'All named outputs, including wrong names on unknown fonts.',
        'unknown_not_named_definition':'Explicit unknown, uncertainty and preprocessing rejection all withhold a name; only explicit unknown counts toward explicit_unknown_recall.',
        'unknown_outcomes':outcome_counts,
        'size_error_population':'Correctly named regions with an available size only; wrong names and missing sizes are excluded from error quantiles and reflected in coverage.',
        'size_available_among_correct_names':measured['size']['correctly_named_with_size'],
        'correctly_named_views':measured['correct_named'],
        'size_available_on_wrong_names':sum(r['wrong_named'] and r['size_px'] is not None for r in details),
        'size_correct_coverage_of_all_known_views':measured['size']['correctly_named_with_size']/max(1,measured['known_views']),
        'color_accuracy_evaluated':False}
    return measured,by_view,confusion,denominators


def holdout_evidence(data):
    """Read only already bound native metadata to describe unknown-source splits."""
    from training.capture.capture_android import load_capture
    source=load_capture(Path(data['manifest']['capture']))
    require(source['manifest_sha256']==data['manifest']['capture_manifest_sha256'],'TEST capture provenance changed')
    groups={split:{} for split in ('train','calibration','test')};owners={}
    for row in source['rows']:
        if row['training_family']!=UNKNOWN:continue
        split=row['split'];family=row['font_family'];digest=row['font_file_sha256']
        require(split in groups,'unknown font split is invalid')
        for key in (('family',normalized_text(family)),('file',digest)):
            require(owners.setdefault(key,split)==split,'unknown source holdout crosses partitions')
        group=groups[split].setdefault(family,{'native_regions':0,'font_file_sha256':set()})
        group['native_regions']+=1;group['font_file_sha256'].add(digest)
    require(all(groups.values()),'unknown source holdout evidence is incomplete')
    for group in groups.values():
        for entry in group.values():entry['font_file_sha256']=sorted(entry['font_file_sha256'])
    expected=Counter((r['source_id'],r['region_id'],r['font_family'],r['font_file_sha256'])
                     for r in source['rows'] if r['split']=='test' and r['training_family']==UNKNOWN)
    actual=Counter((r['source_id'],r['region_id'],r['source_font_family'],r['font_file_sha256'])
                   for r in data['rows'] if r['family']==UNKNOWN and r['view']=='native')
    index={(r['source_id'],r['region_id']):r for r in source['rows'] if r['split']=='test'}
    for row in data['partition']['rejected']:
        if row['family']==UNKNOWN and row['view']=='native':
            native=index.get((row['source_id'],row['region_id']))
            require(native is not None and native['training_family']==UNKNOWN,'rejected unknown has no native source')
            actual[(native['source_id'],native['region_id'],native['font_family'],native['font_file_sha256'])]+=1
    require(actual==expected,'TEST unknown population differs from native holdout')
    return {'source_manifest_sha256':source['manifest_sha256'],'unknown_by_split':groups,
        'whole_unknown_families_and_files_disjoint':True,
        'independence_scope':'This frozen Android capture corpus; absence from all initialization pretraining is not asserted.',
        'user_images_used_as_font_truth':False,'native_renderer':'Android emulator',
        'statistical_unit':'Native region/page; four views of one region are correlated, not four independent captures.'}


def evaluate(args):
    from flux_glyph.android_font import AndroidFontClassifier
    require(not args.output.exists(),'retain the existing sealed test report')
    selection,parity=validate_export(args)
    model=AndroidFontClassifier(args.region);selected=selection['selected']
    require(model.output_families==selection['families'] and model.meta['temperature']==selected['temperature']
            and model.meta['gates']==selected['gates'] and model.meta['max_size_relative_spread']==POLICY['max_size_relative_spread'],
            'TEST family order or runtime gates differ from selected CAL')
    training=model.meta.get('training',{})
    require(training.get('selection_sha256')==sha(args.run/'SELECTION.json')
            and training.get('checkpoint_sha256')==parity['checkpoint_sha256']
            and training.get('data_manifest_sha256')==selection['data_manifest_sha256'],
            'TEST runtime metadata differs from frozen training provenance')
    require(validate_export(args)==(selection,parity),'export changed before TEST freeze')
    # This evaluator and its source digest are recorded before opening a TEST array.
    source_sha=sha(__file__)
    freeze={'schema':'flux-glyph-android-test-freeze-v1','selection_sha256':sha(args.run/'SELECTION.json'),
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
                and np.isfinite(family).all() and np.isfinite(size).all() and np.all(np.abs(size)<=3),'invalid TEST model output')
        logits.append(family);ratios.append(size)
    outputs=region_outputs(np.concatenate(logits),np.concatenate(ratios),data['rows'],selected['temperature'])
    details=decisions(outputs,data['rows'],data['families'],selected['gates'])
    measured,by_view,confusion,denominators=summarize(details,data['families'],data['partition']['rejected'],RECIPES)
    require(sha(args.run/'SELECTION.json')==freeze['selection_sha256'] and sha(args.run/'PARITY.json')==freeze['parity_sha256']
            and sha(args.run/'model.pth')==freeze['checkpoint_sha256'] and sha(freeze_path)==freeze_sha
            and sha(__file__)==source_sha and validate_export(args)==(selection,parity),'source/model changed during TEST')
    require(sha(args.data/'test/MANIFEST.json')==data['partition_sha256']
            and all(sha(args.data/'test'/data['partition'][key]['path'])==data['partition'][key]['sha256'] for key in ('array','metadata')),
            'TEST data changed during evaluation')
    report={'schema':'flux-glyph-android-test-v1','passed':measured['passed'],'metrics':measured,'by_view':by_view,'confusion':confusion,
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
