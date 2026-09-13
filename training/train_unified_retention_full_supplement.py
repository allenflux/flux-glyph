#!/usr/bin/env python3
"""Train the full image CNN on verified native negatives with cached core targets."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'training')]
import train_unified_retention_core as core
from train_unified_retention import RetentionSampler,evaluate_outputs,retention_rank,FIXED_RUNTIME,SAMPLING
from train_unified_retention_adapter import (BASE_CHECKPOINT_SHA,BASE_SELECTION_SHA,
    read,cache_identity,load_cache)
from train_unified_retention import ARCHITECTURE,BATCH_SIZE,parameter_groups,initialize,infer
from train_regions import require,sha,dump,state_sha
from retention_supplement_sampler import SupplementSampler, SAMPLING, SUPPLEMENT_MARKER, NEW_SOURCES
from prepare_unified_unknown_supplement import load_supplement
from cache_unified_unknown_supplement import load_cache as load_supplement_cache

CACHE_MANIFEST_SHA='c3d3e57cf4d8377174e4b7981bbc1d30ee0e72af547f41f2f96a9a674998974a'
STEPS=3000
EVAL_EVERY=500
LEARNING_RATE=2e-5
MINIMUM_LEARNING_RATE=2e-6
SEED=2026091408
OBJECTIVE_VARIANT='full_cnn_native_unknown_supplement_cached_core_kl'
OBJECTIVE={'family_cross_entropy_label_smoothing':.03,'native_core_weighting':core.NATIVE_CORE_WEIGHTING,
    'unknown_ce_weight':1.,'ce_denominator':96,'size_smooth_l1_weight':.2,'size_loss_weighted':False,
    'teacher_kl_weight':2.,'teacher_temperature':1.,'teacher_logits_used':True,'second_model_resident':False,
    'teacher_mask':'TRAIN true target outside PingFang/SF Pro/Helvetica and same-tile frozen core argmax equals truth, including unknown',
    'teacher_kl_reduction':'Mean eligible KL(core || student); differentiable zero if empty',
    'teacher_cache_deployed':False,'all_parameters_trained':True,'inference_rules_changed':False,
    'training_inputs':['image_tiles'],'class_prior_weighting':False}

SUPPLEMENT_MANIFEST_SHA='43aac66dfc3ce3a09fd0acd088dcda82450a7203997befd1e35aebe016713b05'
SUPPLEMENT_CACHE_SHA='4c9114a8f49b7bbdb54d5c915a67643c6e4bef581ca195ecfe6264bf251e434b'


def full_losses(logits,ratios,targets,sizes,teacher_logits,rows,families):
    import torch
    require(logits.dtype in (torch.float32,torch.float64) and all(value.dtype==logits.dtype
            and value.device==logits.device for value in (ratios,sizes,teacher_logits))
            and targets.device==logits.device and not sizes.requires_grad,
            'Full-CNN loss tensors must share finite floating dtype/device and frozen native targets')
    total,ce,size_loss,kl,mask=core.distilled_losses(logits,ratios,targets,sizes,teacher_logits,families,rows)
    result=total+kl
    require(bool(torch.isfinite(result)),'Nonfinite full-CNN loss')
    return result,ce,size_loss,kl,mask


def batch_source(data,cache):
    require(data['partition']['split']=='train' and len(data['tiles'])==len(cache['base_logits'])
            and cache['base_logits'].shape==(len(data['tiles']),25),'TRAIN image/teacher tile order differs')
    index={}
    for row in data['rows']:
        key=(row['source_id'],row['region_id'],row['view'])
        require(key not in index and row['split']=='train' and SUPPLEMENT_MARKER not in row,'Invalid or duplicate TRAIN source region')
        index[key]=row
    return {'tiles':data['tiles'],'teacher_logits':cache['base_logits'],'index':index}


def pixel_batch(rows,old,new,rng):
    require(len(rows)==BATCH_SIZE,'Full-CNN sampling requires 96 TRAIN regions')
    images=[];teachers=[];sizes=[]
    for row in rows:
        marker=row.get(SUPPLEMENT_MARKER,False)
        require(type(marker) is bool,'Invalid supplement cache marker')
        source=new if marker else old
        raw={k:v for k,v in row.items() if k!=SUPPLEMENT_MARKER}
        key=(row.get('source_id'),row.get('region_id'),row.get('view'))
        require(source['index'].get(key)==raw and raw.get('split')=='train'
                and raw.get('native_font_verified') is True,'TRAIN row was relabeled or routed to the wrong image/teacher cache')
        start,count=row['tile_start'],row['tile_count']
        require(type(start) is int and type(count) is int and 1<=count<=8
                and 0<=start<start+count<=len(source['tiles'])
                and len(source['tiles'])==len(source['teacher_logits']),'TRAIN tile index escapes its paired cache')
        index=start+int(rng.integers(count))
        images.append(source['tiles'][index]);teachers.append(source['teacher_logits'][index]);sizes.append(row['log_em_ratio'])
    images=np.asarray(images,dtype=np.float32);teachers=np.asarray(teachers,dtype=np.float32);sizes=np.asarray(sizes,dtype=np.float32)
    require(images.shape==(96,1,64,256) and teachers.shape==(96,25) and sizes.shape==(96,)
            and all(np.isfinite(x).all() for x in (images,teachers,sizes))
            and ((images>=0)&(images<=1)).all() and (np.abs(sizes)<=3).all(), 'Invalid TRAIN pixels or cached teacher/native size targets')
    return images,teachers,sizes


class FullSupplementCounts(core.DistillationCounter):
    def __init__(self,families):
        super().__init__(families);self.supplement_rows=0;self.supplement_sources=Counter()
    def update(self,rows,mask):
        extra=[r for r in rows if r.get(SUPPLEMENT_MARKER)]
        require(len(extra)==8 and Counter(r['source_font_family'] for r in extra)==dict.fromkeys(NEW_SOURCES,4),
                'Every batch requires four rows from each new TRAIN unknown source')
        super().update(rows,mask);self.supplement_rows+=len(extra)
        self.supplement_sources.update(r['source_font_family'] for r in extra)
    def report(self):
        return {**super().report(),'schema':'flux-glyph-retention-full-supplement-counts-v1',
            'objective_variant':OBJECTIVE_VARIANT,'mask_rule':OBJECTIVE['teacher_mask'],'objective':OBJECTIVE,
            'supplement_rows':self.supplement_rows,'supplement_source_rows':dict(self.supplement_sources),
            'original_rows':self.steps*96-self.supplement_rows,'teacher_logits_used':True,
            'second_model_resident':False,'teacher_optimizer_steps':0,'teacher_cache_deployed':False}


def validate_training_counts(report,families,steps):
    require(report.get('schema')=='flux-glyph-retention-full-supplement-counts-v1'
            and report.get('objective_variant')==OBJECTIVE_VARIANT and report.get('objective')==OBJECTIVE
            and report.get('mask_rule')==OBJECTIVE['teacher_mask'] and report.get('steps')==steps
            and report.get('rows')==steps*96 and report.get('supplement_rows')==steps*8
            and report.get('original_rows')==steps*88
            and report.get('supplement_source_rows')==dict.fromkeys(NEW_SOURCES,steps*4)
            and report.get('teacher_logits_used') is True and report.get('second_model_resident') is False
            and report.get('teacher_optimizer_steps')==0 and report.get('teacher_cache_deployed') is False,
            'Full-CNN cached teacher or new TRAIN sample accounting differs')
    require(report.get('training_split')=='train' and report.get('test_read') is False
            and report.get('development_holdout_read') is False
            and report.get('native_core_weighting')==core.NATIVE_CORE_WEIGHTING,
            'Teacher evidence has invalid TRAIN scope or native weighting')
    eligible=report['eligible_rows_per_step'];weighted=report['native_core_weighted_rows_per_step']
    require(len(eligible)==len(weighted)==steps
            and all(type(n) is int and 0<=n<=96 for n in eligible)
            and all(type(n) is int and 27<=n<=33 for n in weighted)
            and sum(eligible)==report['eligible_rows'] and sum(weighted)==report['native_core_weighted_rows'],
            'Per-step mask or native core totals differ')
    totals=Counter();domain=Counter();selected=Counter();selected_domain=Counter();source_lookup={}
    for row in report['source_target_rows']:
        key=(row['domain'],row['source_font_family'],row['target_family']);n=row['rows'];e=row['eligible_rows']
        require(key not in source_lookup and key[0] in ('ios','android') and key[2] in families
                and isinstance(key[1],str) and key[1] and type(n) is int and type(e) is int and 0<=e<=n
                and (e==0 or key[2] not in core.CORE_FAMILIES),'Invalid teacher source population')
        source_lookup[key]=n;totals[key[2]]+=n;domain[key[0]]+=n;selected[key[2]]+=e;selected_domain[key[0]]+=e
    require(dict(totals)==report['family_rows'] and dict(domain)==report['domain_rows'] and sum(totals.values())==steps*96
            and sum(selected.values())==report['eligible_rows']
            and set(report['eligible_family_rows'])<=set(families) and set(report['eligible_domain_rows'])<={'ios','android'}
            and all(selected[f]==report['eligible_family_rows'].get(f,0) for f in families)
            and all(selected_domain[d]==report['eligible_domain_rows'].get(d,0) for d in ('ios','android')),
            'Teacher source and family/domain totals differ')
    core_counts=Counter();seen=set()
    for row in report['native_core_source_rows']:
        key=(row['domain'],row['source_font_family'],row['target_family']);n=row['rows']
        require(key not in seen and key in source_lookup and key[0]=='ios' and key[2] in core.CORE_FAMILIES
                and type(n) is int and 0<n<=source_lookup[key],'Invalid native core weighted source count')
        seen.add(key);core_counts[key[2]]+=n
    require(dict(core_counts)==report['native_core_weighted_family_rows']
            and sum(core_counts.values())==report['native_core_weighted_rows'],'Native core totals differ')
    sources={r['source_font_family']:r for r in report['source_target_rows'] if r['source_font_family'] in NEW_SOURCES}
    require(set(sources)==set(NEW_SOURCES) and all(r['rows']==steps*4 and r['target_family']==core.UNKNOWN
            and r['domain']=='android' for r in sources.values()),'New unknown source count/target differs')
    return report


def train(args):
    import torch
    from region_network import RegionFontClassifier
    args.data=args.data.resolve();args.output=args.output.resolve();args.checkpoint=args.checkpoint.resolve();args.plan=args.plan.resolve()
    args.cache=args.cache.resolve()
    require(sha(args.cache/'CACHE_MANIFEST.json')==CACHE_MANIFEST_SHA,'Use only the already frozen core feature cache')
    args.feature_device=read(args.cache/'CACHE_MANIFEST.json')['identity']['feature_device']
    require(not args.output.exists() and args.steps==STEPS and args.seed==SEED,'Full-CNN run needs a new output and fixed 3000 steps/seed')
    require(sha(args.checkpoint)==BASE_CHECKPOINT_SHA and sha(args.checkpoint.parent/'SELECTION.json')==BASE_SELECTION_SHA,
            'Frozen core step-1000 base checkpoint differs')
    source=read(args.checkpoint.parent/'SELECTION.json');source_freeze=read(args.checkpoint.parent/'TRAINING_FREEZE.json')
    require(source['selected']['step']==1000 and source['optimizer_steps_executed']==1500 and source.get('promotion_allowed') is False
            and source['objective_variant']==core.OBJECTIVE_VARIANT and source['families']==source_freeze['families']
            and sha(args.checkpoint.parent/'TRAINING_FREEZE.json')==source['training_protocol_sha256']
            and source['test_read'] is False and source['development_holdout_read'] is False,'Invalid frozen core initializer lineage')
    core.verify_bindings(source['bindings'])
    plan=core.read_plan(args.plan,args.data,(ROOT/source['parent_checkpoint']['path']).resolve())
    datasets={split:core.load_split(args.data,split) for split in ('train','calibration')};training=datasets['train'];cal=datasets['calibration']
    families=training['families'];require(families==source['families']==plan['families'],'Full-CNN family order differs')
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    require(checkpoint.get('selection_sha256')==BASE_SELECTION_SHA and checkpoint.get('families')==families
            and checkpoint.get('architecture')==core.ARCHITECTURE and state_sha(checkpoint['state_dict'])==source['state_after_sha256'],
            'Base checkpoint state or architecture differs')
    model=RegionFontClassifier(25)
    inheritance=initialize(model,checkpoint,families)
    base_before=state_sha(model.state_dict());initial_groups=parameter_groups(model.state_dict())
    full_before=base_before
    require(base_before==source['state_after_sha256'] and all(p.requires_grad for p in model.parameters())
            and not any(k.startswith('residual') for k in model.state_dict()),
            'Full-CNN did not inherit/train exactly the plain 25-class core model')
    base_identity={'path':str(args.checkpoint),'sha256':BASE_CHECKPOINT_SHA};base_selection={'path':str(args.checkpoint.parent/'SELECTION.json'),'sha256':BASE_SELECTION_SHA}
    bindings=dict(source['bindings'])
    paths=[Path(__file__),ROOT/'training/retention_adapter_network.py',ROOT/'training/train_unified_retention_core.py',
        ROOT/'training/train_unified_retention.py',ROOT/'training/train_unified_regions.py',ROOT/'training/prepare_unified_regions.py',
        ROOT/'training/region_network.py',ROOT/'training/network.py',ROOT/'training/train_regions.py',
        ROOT/'src/flux_glyph/unified_font.py',ROOT/'src/flux_glyph/region_font.py',ROOT/'training/evaluate_unified_retention_full_supplement.py',ROOT/'training/retention_supplement_sampler.py',
        ROOT/'training/prepare_unified_unknown_supplement.py',
        ROOT/'training/train_unified_retention_adapter.py',
        ROOT/'artifacts/unified-font-v1/run-v1/DEVELOPMENT_REGRESSION.json',args.checkpoint,args.plan,
        args.checkpoint.parent/'SELECTION.json',args.checkpoint.parent/'TRAINING_FREEZE.json',
        args.checkpoint.parent/'CALIBRATION_OUTPUTS.npz',args.checkpoint.parent/'CALIBRATION_DECISIONS.json',args.data/'MANIFEST.json']
    for split in ('train','calibration'):
        part=read(args.data/split/'MANIFEST.json');paths.append(args.data/split/'MANIFEST.json')
        for key in ('array','metadata'):
            path=(args.data/split/part[key]['path']).resolve()
            require(path.parent==args.data/split and sha(path)==part[key]['sha256'],'Full-CNN data changed')
            paths.append(path)
    for path in paths:
        path=path.resolve();digest=sha(path)
        require(str(path) not in bindings or bindings[str(path)]==digest,'Full-CNN source binding conflict');bindings[str(path)]=digest
    cached_identity=read(args.cache/'CACHE_MANIFEST.json')['identity']
    identity=cache_identity(datasets,base_identity,base_before,cached_identity['bindings'],args.feature_device)
    require(identity==cached_identity,'Reused cache does not match actual TRAIN/CAL tiles, row order or base model')
    cache,cache_manifest=load_cache(args.cache,identity)
    for path,digest in identity['bindings'].items():
        require(path not in bindings or bindings[path]==digest,'Original cache and new training source conflict')
        bindings[path]=digest
    args.supplement=args.supplement.resolve();args.supplement_cache=args.supplement_cache.resolve()
    supplement_manifest_path=args.supplement/'MANIFEST.json'
    supplement_cache_path=args.supplement_cache/'CACHE_MANIFEST.json'
    require(sha(supplement_manifest_path)==SUPPLEMENT_MANIFEST_SHA
            and sha(supplement_cache_path)==SUPPLEMENT_CACHE_SHA,'Frozen new native TRAIN supplement differs')
    supplement=load_supplement(args.supplement)
    supplement_cache,supplement_cache_manifest=load_supplement_cache(args.supplement_cache)
    extra_identity=supplement_cache_manifest['identity']
    require(supplement['families']==families==extra_identity['families']
            and extra_identity['data_manifest']['sha256']==supplement['manifest_sha256']==SUPPLEMENT_MANIFEST_SHA
            and extra_identity['base_checkpoint']['sha256']==BASE_CHECKPOINT_SHA
            and extra_identity['base_state_sha256']==base_before
            and extra_identity['partition']['split']=='train'
            and len(supplement_cache['features'])==len(supplement['tiles']),
            'Supplement features do not come from the same frozen base and verified TRAIN partition')
    for path,digest in extra_identity['bindings'].items():
        require(path not in bindings or bindings[path]==digest,'Supplement source conflicts with frozen original evidence')
        require(sha(path)==digest,'Supplement source binding changed');bindings[path]=digest
    extra_paths=[Path(__file__),ROOT/'training/retention_supplement_sampler.py',
        ROOT/'training/prepare_unified_unknown_supplement.py',ROOT/'training/cache_unified_unknown_supplement.py',
        supplement_manifest_path,args.supplement/'PREPARATION_FREEZE.json',args.supplement/'train/MANIFEST.json',
        supplement_cache_path,args.supplement_cache/'CACHE_FREEZE.json']
    extra_paths.extend(args.supplement/'train'/supplement['partition'][key]['path'] for key in ('metadata','array'))
    extra_paths.extend(args.supplement_cache/'train'/(name+'.npy') for name in ('features','base_logits','log_em_ratio'))
    for path in extra_paths:
        path=path.resolve();digest=sha(path)
        require(str(path) not in bindings or bindings[str(path)]==digest,'Supplement file changed');bindings[str(path)]=digest
    supplement_identity={'path':str(supplement_manifest_path),'sha256':SUPPLEMENT_MANIFEST_SHA}
    supplement_cache_identity={'path':str(supplement_cache_path),'sha256':SUPPLEMENT_CACHE_SHA}
    original_regions={(r['source_id'],r['region_id']) for r in training['rows']}
    new_regions={(r['source_id'],r['region_id']) for r in supplement['rows']}
    require(not original_regions&new_regions,'New TRAIN supplement overlaps an old source region')
    training_data_counts={'original_views':len(training['rows']),'supplement_views':len(supplement['rows']),
        'combined_views':len(training['rows'])+len(supplement['rows']),
        'original_native_regions':len(original_regions),'supplement_native_regions':len(new_regions),
        'combined_native_regions':len(original_regions)+len(new_regions),
        'original_tiles':len(training['tiles']),'supplement_tiles':len(supplement['tiles']),
        'combined_tiles':len(training['tiles'])+len(supplement['tiles']),
        'calibration_unchanged':True,'derived_views_are_correlated':True}
    model.cpu();require(state_sha(model.state_dict())==full_before,'Cache validation changed initial student parameters')
    # MPS features/logits/size are frozen training artifacts, not deployable lookup data.
    cache_path=args.cache/'CACHE_MANIFEST.json';cache_binding={'path':str(cache_path),'sha256':sha(cache_path)}
    for path in (args.cache/'CACHE_FREEZE.json',cache_path):bindings[str(path)]=sha(path)
    for part in cache_manifest['partitions'].values():
        for entry in part.values():bindings[str((args.cache/entry['path']).resolve())]=entry['sha256']
    require(sha(args.checkpoint.parent/'CALIBRATION_OUTPUTS.npz')==source['calibration_outputs_sha256'], 'Source base CAL changed')
    with np.load(args.checkpoint.parent/'CALIBRATION_OUTPUTS.npz',allow_pickle=False) as original:
        np.testing.assert_allclose(cache['calibration']['base_logits'],original['logits'],atol=3e-4,rtol=3e-4)
        np.testing.assert_allclose(cache['calibration']['log_em_ratio'],original['log_em_ratio'],atol=3e-4,rtol=3e-4)
    baseline,base_outputs,_=evaluate_outputs(cache['calibration']['base_logits'],cache['calibration']['log_em_ratio'],cal,plan,0)
    args.output.mkdir(parents=True,exist_ok=True)
    dump(args.output/'BASELINE.json',baseline);dump(args.output/'BASELINE_CALIBRATION_DECISIONS.json',{'families':families,'records':base_outputs})
    baseline_bindings={name:sha(args.output/name) for name in ('BASELINE.json','BASELINE_CALIBRATION_DECISIONS.json')}
    protocol={'schema':'flux-glyph-unified-retention-full-supplement-protocol-v1','architecture':ARCHITECTURE,'families':families,
        'objective_variant':OBJECTIVE_VARIANT,'objective':OBJECTIVE,'sampling':SAMPLING,'fixed_runtime':FIXED_RUNTIME,
        'feature_cache_reused':False,'teacher_cache_reused':True,'cached_feature_extraction_performed':False,
        'supplement_data':supplement_identity,'supplement_cache':supplement_cache_identity,
        'training_data_counts':training_data_counts,
        'steps':STEPS,'eval_every':EVAL_EVERY,'batch_size':BATCH_SIZE,'learning_rate':LEARNING_RATE,'minimum_learning_rate':MINIMUM_LEARNING_RATE,
        'weight_decay':1e-4,'gradient_clip_norm':5.,'seed':args.seed,'training_device':args.device,'feature_device':args.feature_device,
        'base_checkpoint':base_identity,'base_selection':base_selection,'base_state_sha256':base_before,
        'base_selected_step':1000,'source_optimizer_steps_executed':1500,
        'optimizer_steps_executed':STEPS,'initial_state_sha256':full_before,'state_before_sha256':full_before,
        'initial_parameter_groups_sha256':initial_groups,'inheritance':inheritance,
        'cached_teacher_state_sha256':base_before,'cache_manifest':cache_binding,'bindings':bindings,
        'data_manifest_sha256':training['manifest_sha256'],'retention_plan':{'path':str(args.plan),'sha256':sha(args.plan)},
        'parent_metadata':source['parent_metadata'],'parent_checkpoint':source['parent_checkpoint'],
        'parent_selection_sha256':source['parent_selection_sha256'],
        'baseline_bindings':baseline_bindings,
        'frozen_groups':[],'trainable_groups':['trunk','style','family_head','size_head'],
        'all_parameters_trained':True,'base_frozen':False,'size_head_frozen':False,'residual_head_trained':False,
        'model_count':1,'encoder_count':1,'platform_routing':False,'score_merging':False,
        'teacher_logits_used':True,'second_model_resident':False,'teacher_optimizer_steps':0,'teacher_cache_deployed':False,
        'test_read':False,'development_holdout_read':False,'runtime_gates_searched':False,'training_inputs':['image_tiles'],
        'calibration_execution':'Current complete CNN forward over all original CAL image tiles; current learned size head.'}
    core.verify_bindings(bindings);dump(args.output/'TRAINING_FREEZE.json',protocol);protocol_sha=sha(args.output/'TRAINING_FREEZE.json')
    sampler=SupplementSampler(training['rows'],families,args.seed,plan['train_pools'],supplement['rows']);counts=FullSupplementCounts(families)
    old_batch=batch_source(training,cache['train']);new_batch=batch_source(supplement,supplement_cache)
    model.to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=MINIMUM_LEARNING_RATE)
    best=None;history=[];started=time.monotonic()
    for step in range(1,STEPS+1):
        rows=sampler.batch()
        image_values,base_values,size_values=pixel_batch(rows,old_batch,new_batch,sampler.rng)
        images=torch.from_numpy(image_values).to(args.device);base=torch.from_numpy(base_values).to(args.device)
        targets=torch.tensor([r['target'] for r in rows],device=args.device);sizes=torch.from_numpy(size_values).to(args.device)
        model.train();logits,ratios=model(images)
        loss,ce,size_loss,kl,mask=full_losses(logits,ratios,targets,sizes,base,rows,families)
        counts.update(rows,mask.detach().cpu().tolist())
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step();scheduler.step()
        if step%100==0:print(json.dumps({'step':step,'loss':float(loss.detach()),'ce':float(ce.detach()),'base_kl':float(kl.detach()),'size_loss':float(size_loss.detach()),'seconds':round(time.monotonic()-started,1)}),flush=True)
        if step%EVAL_EVERY==0:
            logits,ratios=infer(model,cal,args.device);record,outputs,_=evaluate_outputs(logits,ratios,cal,plan,step)
            directory=args.output/'checkpoints'/f'step{step:05d}';directory.mkdir(parents=True)
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({'state_dict':state,'families':families,'architecture':ARCHITECTURE,'step':step,'training_protocol_sha256':protocol_sha},directory/'model.pth')
            np.savez(directory/'CALIBRATION_OUTPUTS.npz',logits=logits,log_em_ratio=ratios)
            dump(directory/'CALIBRATION_DECISIONS.json',{'families':families,'records':outputs});dump(directory/'METRICS.json',record)
            record['artifacts']={key:{'path':str(path.relative_to(args.output)),'sha256':sha(path)} for key,path in [('checkpoint',directory/'model.pth'),('outputs',directory/'CALIBRATION_OUTPUTS.npz'),('decisions',directory/'CALIBRATION_DECISIONS.json'),('metrics',directory/'METRICS.json')]}
            history.append(record)
            if best is None or retention_rank(record)>retention_rank(best[0]):best=(copy.deepcopy(record),state)
            dump(args.output/'CAL_PROGRESS.json',history);dump(args.output/'TRAINING_COUNTS_PROGRESS.json',counts.report())
            print(json.dumps({'calibration_step':step,'promotion_allowed':record['promotion_allowed'],'checks_passed':sum(r['passed'] for r in record['retention_checks']),'deficit':record['retention_deficit']}),flush=True)
    selected_groups=parameter_groups(best[1]);final_groups=parameter_groups(model.state_dict())
    require(all(selected_groups[key]!=initial_groups[key] and final_groups[key]!=initial_groups[key] for key in initial_groups)
            and all(p.requires_grad for p in model.parameters()),'A full-CNN parameter group did not train')
    core.verify_bindings(bindings)
    require(sha(args.output/'TRAINING_FREEZE.json')==protocol_sha and all(sha(args.output/name)==digest for name,digest in baseline_bindings.items()),'Full-CNN protocol or baseline changed')
    for record in history:require(all(sha(args.output/item['path'])==item['sha256'] for item in record['artifacts'].values()),'Saved full-CNN checkpoint evidence changed')
    selected=best[0];dump(args.output/'SAMPLING.json',sampler.report())
    training_counts=counts.report();validate_training_counts(training_counts,families,STEPS)
    dump(args.output/'TRAINING_COUNTS.json',training_counts)
    for key,name in [('outputs','CALIBRATION_OUTPUTS.npz'),('decisions','CALIBRATION_DECISIONS.json')]:
        (args.output/name).write_bytes((args.output/selected['artifacts'][key]['path']).read_bytes())
    selection={**protocol,'schema':'flux-glyph-unified-retention-full-supplement-selection-v1','selected':selected,'history':history,
        'promotion_allowed':selected['promotion_allowed'],'passed':selected['metrics']['passed'],'calibration_passed':selected['metrics']['passed'],
        'training_protocol_sha256':protocol_sha,'state_after_sha256':state_sha(best[1]),
        'selected_parameter_groups_sha256':selected_groups,'final_parameter_groups_sha256':final_groups,
        'sampling_sha256':sha(args.output/'SAMPLING.json'),
        'training_counts_sha256':sha(args.output/'TRAINING_COUNTS.json'),
        'calibration_outputs_sha256':sha(args.output/'CALIBRATION_OUTPUTS.npz'),'calibration_decisions_sha256':sha(args.output/'CALIBRATION_DECISIONS.json')}
    dump(args.output/'SELECTION.json',selection)
    torch.save({'state_dict':best[1],'families':families,'architecture':ARCHITECTURE,'selection_sha256':sha(args.output/'SELECTION.json')},args.output/'model.pth')
    report={'status':'PROMOTABLE_CHECKPOINT' if selected['promotion_allowed'] else 'NO_PROMOTABLE_CHECKPOINT',
        'promotion_allowed':selected['promotion_allowed'],'calibration_passed':selected['metrics']['passed'],
        'optimizer_steps_executed':STEPS,'source_optimizer_steps_executed':1500,
        'supplement_training_rows_executed':training_counts['supplement_rows'],'training_data_counts':training_data_counts,
        'base_selected_step':1000,'selected_step':selected['step'],'all_parameter_groups_changed':True,
        'all_parameters_trained':True,'teacher_logits_used':True,'second_model_resident':False,'teacher_optimizer_steps':0,
        'teacher_cache_deployed':False,'model_count':1,'runtime_gates_changed':False,
        'last_checkpoint_sha256':history[-1]['artifacts']['checkpoint']['sha256'],'selected_checkpoint_sha256':sha(args.output/'model.pth'),
        'test_read':False,'development_holdout_read':False}
    dump(args.output/'report.json',report)
    if not selected['promotion_allowed']:dump(args.output/'NO_PROMOTABLE_CHECKPOINT.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'artifacts/unified-font-v1/data-v1')
    p.add_argument('--plan',type=Path,default=ROOT/'artifacts/unified-font-v2/RETENTION_PLAN.json')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/unified-font-v2/run-core-v1/model.pth')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/unified-font-v2/run-full-supplement-v1')
    p.add_argument('--cache',type=Path,default=ROOT/'artifacts/unified-font-v2/run-adapter-v1/cache')
    p.add_argument('--supplement',type=Path,default=ROOT/'artifacts/unified-font-v2/new-unknown-capture-v1/data')
    p.add_argument('--supplement-cache',type=Path,default=ROOT/'artifacts/unified-font-v2/new-unknown-capture-v1/cache')
    p.add_argument('--steps',type=int,default=STEPS);p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--device',choices=('cpu','mps'),default='mps')
    return p


if __name__=='__main__':train(parser().parse_args())
