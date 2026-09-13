#!/usr/bin/env python3
"""Residual-head repair with independently captured native TRAIN negatives."""
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
from train_unified_retention_adapter import (ARCHITECTURE,BASE_CHECKPOINT_SHA,BASE_SELECTION_SHA,STEPS,EVAL_EVERY,
    BATCH_SIZE,LEARNING_RATE,MINIMUM_LEARNING_RATE,read,residual_state,frozen_state,cache_identity,load_cache,
    infer_cached,TrainingCounts)
from train_regions import require,sha,dump,state_sha
from train_unified_retention_adapter import adapter_loss, OBJECTIVE
from retention_supplement_sampler import SupplementSampler, SAMPLING, SUPPLEMENT_MARKER, NEW_SOURCES
from prepare_unified_unknown_supplement import load_supplement
from cache_unified_unknown_supplement import load_cache as load_supplement_cache

CACHE_MANIFEST_SHA='c3d3e57cf4d8377174e4b7981bbc1d30ee0e72af547f41f2f96a9a674998974a'
OBJECTIVE_VARIANT='native_font_unknown_supplement_replay'
SUPPLEMENT_MANIFEST_SHA='43aac66dfc3ce3a09fd0acd088dcda82450a7203997befd1e35aebe016713b05'
SUPPLEMENT_CACHE_SHA='4c9114a8f49b7bbdb54d5c915a67643c6e4bef581ca195ecfe6264bf251e434b'


class SupplementCounts(TrainingCounts):
    def __init__(self):
        super().__init__();self.supplement_rows=0;self.supplement_sources=Counter()
    def update(self,rows,mask,weights):
        extra=[r for r in rows if r.get(SUPPLEMENT_MARKER)]
        require(len(extra)==8 and Counter(r['source_font_family'] for r in extra)==dict.fromkeys(NEW_SOURCES,4),
                'Each training batch must actually include all eight declared new negatives')
        super().update(rows,mask,weights)
        self.supplement_rows+=len(extra);self.supplement_sources.update(r['source_font_family'] for r in extra)
    def report(self):
        return {**super().report(),'schema':'flux-glyph-retention-supplement-counts-v1',
            'objective_variant':OBJECTIVE_VARIANT,'supplement_rows':self.supplement_rows,
            'supplement_source_rows':dict(self.supplement_sources),'original_rows':self.rows-self.supplement_rows}


def feature_batch(rows,old,new,rng):
    features=[];logits=[]
    for row in rows:
        cache=new if row.get(SUPPLEMENT_MARKER) else old
        index=row['tile_start']+int(rng.integers(row['tile_count']))
        require(0<=index<len(cache['features']),'TRAIN row points outside its own feature cache')
        features.append(cache['features'][index]);logits.append(cache['base_logits'][index])
    return np.asarray(features,dtype=np.float32),np.asarray(logits,dtype=np.float32)


def train(args):
    import torch
    from retention_adapter_network import RetentionAdapterClassifier,ARCHITECTURE as NETWORK_ARCH
    require(NETWORK_ARCH==ARCHITECTURE,'Residual network architecture differs')
    args.data=args.data.resolve();args.output=args.output.resolve();args.checkpoint=args.checkpoint.resolve();args.plan=args.plan.resolve()
    args.cache=args.cache.resolve()
    require(sha(args.cache/'CACHE_MANIFEST.json')==CACHE_MANIFEST_SHA,'Use only the already frozen core feature cache')
    args.feature_device=read(args.cache/'CACHE_MANIFEST.json')['identity']['feature_device']
    require(not args.output.exists() and args.steps==STEPS,'Adapter run needs a new output and exactly 2000 steps')
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
    families=training['families'];require(families==source['families']==plan['families'],'Adapter family order differs')
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    require(checkpoint.get('selection_sha256')==BASE_SELECTION_SHA and checkpoint.get('families')==families
            and checkpoint.get('architecture')==core.ARCHITECTURE and state_sha(checkpoint['state_dict'])==source['state_after_sha256'],
            'Base checkpoint state or architecture differs')
    model=RetentionAdapterClassifier(25).from_parent(checkpoint['state_dict'])
    base_before=state_sha(frozen_state(model.state_dict()));head_before=state_sha(residual_state(model.state_dict()));full_before=state_sha(model.state_dict())
    require(base_before==source['state_after_sha256'] and all(p.requires_grad==name.startswith('residual_family_head.') for name,p in model.named_parameters()),
            'Adapter did not preserve/freeze the whole base student')
    base_identity={'path':str(args.checkpoint),'sha256':BASE_CHECKPOINT_SHA};base_selection={'path':str(args.checkpoint.parent/'SELECTION.json'),'sha256':BASE_SELECTION_SHA}
    bindings=dict(source['bindings'])
    paths=[Path(__file__),ROOT/'training/retention_adapter_network.py',ROOT/'training/train_unified_retention_core.py',
        ROOT/'training/train_unified_retention.py',ROOT/'training/train_unified_regions.py',ROOT/'training/prepare_unified_regions.py',
        ROOT/'training/region_network.py',ROOT/'training/network.py',ROOT/'training/train_regions.py',
        ROOT/'src/flux_glyph/unified_font.py',ROOT/'src/flux_glyph/region_font.py',ROOT/'training/evaluate_unified_retention_supplement.py',ROOT/'training/retention_supplement_sampler.py',
        ROOT/'training/prepare_unified_unknown_supplement.py',
        ROOT/'training/train_unified_retention_adapter.py',
        ROOT/'artifacts/unified-font-v1/run-v1/DEVELOPMENT_REGRESSION.json',args.checkpoint,args.plan,
        args.checkpoint.parent/'SELECTION.json',args.checkpoint.parent/'TRAINING_FREEZE.json',
        args.checkpoint.parent/'CALIBRATION_OUTPUTS.npz',args.checkpoint.parent/'CALIBRATION_DECISIONS.json',args.data/'MANIFEST.json']
    for split in ('train','calibration'):
        part=read(args.data/split/'MANIFEST.json');paths.append(args.data/split/'MANIFEST.json')
        for key in ('array','metadata'):
            path=(args.data/split/part[key]['path']).resolve()
            require(path.parent==args.data/split and sha(path)==part[key]['sha256'],'Adapter data changed')
            paths.append(path)
    for path in paths:
        path=path.resolve();digest=sha(path)
        require(str(path) not in bindings or bindings[str(path)]==digest,'Adapter source binding conflict');bindings[str(path)]=digest
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
    model.cpu();require(state_sha(model.state_dict())==full_before,'Feature extraction changed initial student parameters')
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
    initial_path=args.output/'INITIAL_RESIDUAL.pth';torch.save(residual_state(model.state_dict()),initial_path);bindings[str(initial_path)]=sha(initial_path)
    protocol={'schema':'flux-glyph-unified-retention-supplement-protocol-v1','architecture':ARCHITECTURE,'families':families,
        'objective_variant':OBJECTIVE_VARIANT,'objective':OBJECTIVE,'sampling':SAMPLING,'fixed_runtime':FIXED_RUNTIME,
        'feature_cache_reused':True,'cached_feature_extraction_performed':False,
        'supplement_data':supplement_identity,'supplement_cache':supplement_cache_identity,
        'training_data_counts':training_data_counts,
        'steps':STEPS,'eval_every':EVAL_EVERY,'batch_size':BATCH_SIZE,'learning_rate':LEARNING_RATE,'minimum_learning_rate':MINIMUM_LEARNING_RATE,
        'weight_decay':1e-4,'gradient_clip_norm':5.,'seed':args.seed,'training_device':'cpu','feature_device':args.feature_device,
        'base_checkpoint':base_identity,'base_selection':base_selection,'base_state_sha256':base_before,
        'base_selected_step':1000,'source_optimizer_steps_executed':1500,'base_optimizer_steps':0,
        'residual_optimizer_steps':STEPS,'optimizer_steps_executed':STEPS,'initial_state_sha256':full_before,
        'residual_state_before_sha256':head_before,'cache_manifest':cache_binding,'bindings':bindings,
        'data_manifest_sha256':training['manifest_sha256'],'retention_plan':{'path':str(args.plan),'sha256':sha(args.plan)},
        'parent_metadata':source['parent_metadata'],'parent_checkpoint':source['parent_checkpoint'],
        'parent_selection_sha256':source['parent_selection_sha256'],
        'baseline_bindings':baseline_bindings,'initial_residual':{'path':str(initial_path),'sha256':sha(initial_path)},
        'frozen_groups':['trunk','pool','style','family_head','size_head'],'trainable_groups':['residual_family_head'],
        'all_parameters_trained':False,'base_frozen':True,'size_head_frozen':True,'residual_head_trained':True,
        'model_count':1,'encoder_count':1,'platform_routing':False,'score_merging':False,'external_teacher':False,
        'test_read':False,'development_holdout_read':False,'runtime_gates_searched':False,'training_inputs':['image_features_only'],
        'calibration_execution':'Frozen MPS/CPU base logits + CPU residual(features); copied frozen size logits.'}
    core.verify_bindings(bindings);dump(args.output/'TRAINING_FREEZE.json',protocol);protocol_sha=sha(args.output/'TRAINING_FREEZE.json')
    sampler=SupplementSampler(training['rows'],families,args.seed,plan['train_pools'],supplement['rows']);counts=SupplementCounts()
    optimizer=torch.optim.AdamW(model.residual_family_head.parameters(),lr=LEARNING_RATE,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=MINIMUM_LEARNING_RATE)
    best=None;history=[];started=time.monotonic()
    for step in range(1,STEPS+1):
        rows=sampler.batch()
        feature_values,base_values=feature_batch(rows,cache['train'],supplement_cache,sampler.rng)
        features=torch.from_numpy(feature_values);base=torch.from_numpy(base_values)
        targets=torch.tensor([r['target'] for r in rows]);model.train();logits=base+model.residual_family_head(features)
        loss,ce,kl,mask,weights=adapter_loss(logits,base,targets,rows,families)
        counts.update(rows,mask.detach().tolist(),weights)
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.residual_family_head.parameters(),5.)
        optimizer.step();scheduler.step()
        if step%100==0:print(json.dumps({'step':step,'loss':float(loss.detach()),'ce':float(ce.detach()),'base_kl':float(kl.detach()),'seconds':round(time.monotonic()-started,1)}),flush=True)
        if step%EVAL_EVERY==0:
            logits,ratios=infer_cached(model,cache['calibration']);record,outputs,_=evaluate_outputs(logits,ratios,cal,plan,step)
            directory=args.output/'checkpoints'/f'step{step:05d}';directory.mkdir(parents=True)
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            require(state_sha(frozen_state(state))==base_before,'Adapter altered frozen encoder, base font head or size head')
            torch.save({'state_dict':state,'families':families,'architecture':ARCHITECTURE,'step':step,'training_protocol_sha256':protocol_sha},directory/'model.pth')
            np.savez(directory/'CALIBRATION_OUTPUTS.npz',logits=logits,log_em_ratio=ratios)
            dump(directory/'CALIBRATION_DECISIONS.json',{'families':families,'records':outputs});dump(directory/'METRICS.json',record)
            record['artifacts']={key:{'path':str(path.relative_to(args.output)),'sha256':sha(path)} for key,path in [('checkpoint',directory/'model.pth'),('outputs',directory/'CALIBRATION_OUTPUTS.npz'),('decisions',directory/'CALIBRATION_DECISIONS.json'),('metrics',directory/'METRICS.json')]}
            history.append(record)
            if best is None or retention_rank(record)>retention_rank(best[0]):best=(copy.deepcopy(record),state)
            dump(args.output/'CAL_PROGRESS.json',history);dump(args.output/'TRAINING_COUNTS_PROGRESS.json',counts.report())
            print(json.dumps({'calibration_step':step,'promotion_allowed':record['promotion_allowed'],'checks_passed':sum(r['passed'] for r in record['retention_checks']),'deficit':record['retention_deficit']}),flush=True)
    require(state_sha(frozen_state(model.state_dict()))==state_sha(frozen_state(best[1]))==base_before
            and all(not p.requires_grad and p.grad is None for name,p in model.named_parameters() if not name.startswith('residual_family_head.')),
            'Frozen base changed or received gradients')
    head_after=state_sha(residual_state(best[1]));require(head_after!=head_before,'The residual head did not train')
    core.verify_bindings(bindings)
    require(sha(args.output/'TRAINING_FREEZE.json')==protocol_sha and all(sha(args.output/name)==digest for name,digest in baseline_bindings.items()),'Adapter protocol or baseline changed')
    for record in history:require(all(sha(args.output/item['path'])==item['sha256'] for item in record['artifacts'].values()),'Saved adapter checkpoint evidence changed')
    selected=best[0];dump(args.output/'SAMPLING.json',sampler.report())
    training_counts=counts.report();require(training_counts['supplement_rows']==STEPS*8 and training_counts['original_rows']==STEPS*88,'Supplement execution counts differ')
    dump(args.output/'TRAINING_COUNTS.json',training_counts)
    for key,name in [('outputs','CALIBRATION_OUTPUTS.npz'),('decisions','CALIBRATION_DECISIONS.json')]:
        (args.output/name).write_bytes((args.output/selected['artifacts'][key]['path']).read_bytes())
    selection={**protocol,'schema':'flux-glyph-unified-retention-supplement-selection-v1','selected':selected,'history':history,
        'promotion_allowed':selected['promotion_allowed'],'passed':selected['metrics']['passed'],'calibration_passed':selected['metrics']['passed'],
        'training_protocol_sha256':protocol_sha,'state_after_sha256':state_sha(best[1]),'base_state_after_sha256':base_before,
        'residual_state_after_sha256':head_after,'sampling_sha256':sha(args.output/'SAMPLING.json'),
        'training_counts_sha256':sha(args.output/'TRAINING_COUNTS.json'),
        'calibration_outputs_sha256':sha(args.output/'CALIBRATION_OUTPUTS.npz'),'calibration_decisions_sha256':sha(args.output/'CALIBRATION_DECISIONS.json')}
    dump(args.output/'SELECTION.json',selection)
    torch.save({'state_dict':best[1],'families':families,'architecture':ARCHITECTURE,'selection_sha256':sha(args.output/'SELECTION.json')},args.output/'model.pth')
    report={'status':'PROMOTABLE_CHECKPOINT' if selected['promotion_allowed'] else 'NO_PROMOTABLE_CHECKPOINT',
        'promotion_allowed':selected['promotion_allowed'],'calibration_passed':selected['metrics']['passed'],
        'base_optimizer_steps':0,'residual_optimizer_steps':STEPS,'optimizer_steps_executed':STEPS,'source_optimizer_steps_executed':1500,
        'supplement_training_rows_executed':training_counts['supplement_rows'],'training_data_counts':training_data_counts,
        'base_selected_step':1000,'selected_step':selected['step'],'base_parameters_unchanged':True,'residual_parameters_changed':True,
        'all_parameters_trained':False,'teacher_model_used':False,'model_count':1,'runtime_gates_changed':False,
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
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/unified-font-v2/run-supplement-v1')
    p.add_argument('--cache',type=Path,default=ROOT/'artifacts/unified-font-v2/run-adapter-v1/cache')
    p.add_argument('--supplement',type=Path,default=ROOT/'artifacts/unified-font-v2/new-unknown-capture-v1/data')
    p.add_argument('--supplement-cache',type=Path,default=ROOT/'artifacts/unified-font-v2/new-unknown-capture-v1/cache')
    p.add_argument('--steps',type=int,default=STEPS);p.add_argument('--seed',type=int,default=2026091407)
    return p


if __name__=='__main__':train(parser().parse_args())
