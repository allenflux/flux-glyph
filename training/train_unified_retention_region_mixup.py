#!/usr/bin/env python3
"""Residual-head training on complete regions plus supervised manifold mixup."""
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
from retention_region_mixup_loss import region_mixup_loss

CACHE_MANIFEST_SHA='c3d3e57cf4d8377174e4b7981bbc1d30ee0e72af547f41f2f96a9a674998974a'
OBJECTIVE_VARIANT='region_mean_softmax_supervision_and_manifold_mixup'
MIXUP={'space':'Frozen 128-dimensional image features; training only','alpha':.4,'loss_weight':.5,
    'pairing':'One uniformly selected tile per each of 96 regions, then one random permutation.',
    'lambda':'One Beta(.4,.4) scalar per batch; no thresholding or fixed-point exclusion.',
    'targets':'lambda * onehot(true class) + (1-lambda) * onehot(paired true class), including unknown',
    'base_logits':'Frozen base family_head applied to interpolated features; not interpolated output probabilities.',
    'synthetic_unknown_labels':False,'deployed':False}
OBJECTIVE={'family_cross_entropy_label_smoothing':.03,'native_ios_core_weight':2.,'unknown_weight':2.,
    'other_weight':1.,'ce_denominator':96,'region_probabilities':'Mean of all tile softmax probabilities at T=1.',
    'size_loss_weight':0.,'base_kl_weight':1.,'base_temperature':1.,
    'base_kl_mask':'TRAIN true target is known, outside PingFang/SF Pro/Helvetica, and frozen base region argmax equals target',
    'base_kl_reduction':'Mean eligible KL(base region || adapted region); differentiable zero for empty mask',
    'native_core_families':list(core.CORE_FAMILIES),'native_weight_split':'train','native_weight_domain':'ios',
    'native_weight_view':'native','label_weights_are_inference_rules':False,'base_frozen':True,
    'size_head_frozen':True,'trainable_parameter_prefix':'residual_family_head.','external_teacher':False,'mixup':MIXUP}


class MixupCounts(TrainingCounts):
    def __init__(self):
        super().__init__();self.tile_counts=Counter();self.total_tiles=0;self.lambda_sum=0.;self.lambda_square_sum=0.
        self.lambda_min=1.;self.lambda_max=0.;self.pairs=Counter();self.different=0;self.self_pairs=0
    def update(self,rows,mask,weights,tile_counts,lam,permutation):
        require(len(tile_counts)==96 and all(type(n) is int and 1<=n<=8 for n in tile_counts)
                and sorted(permutation)==list(range(96)) and math.isfinite(lam) and 0<=lam<=1
                and all(not eligible or row['family']!=core.UNKNOWN for row,eligible in zip(rows,mask)),
                'Invalid full-region, known-only KL or mixup accounting')
        super().update(rows,mask,weights)
        self.tile_counts.update(tile_counts);self.total_tiles+=sum(tile_counts)
        self.lambda_sum+=lam;self.lambda_square_sum+=lam*lam;self.lambda_min=min(self.lambda_min,lam);self.lambda_max=max(self.lambda_max,lam)
        for i,j in enumerate(permutation):
            unknowns=int(rows[i]['family']==core.UNKNOWN)+int(rows[j]['family']==core.UNKNOWN)
            self.pairs[str(unknowns)]+=1;self.different+=int(rows[i]['target']!=rows[j]['target']);self.self_pairs+=int(i==j)
    def report(self):
        result=super().report();result.update(schema='flux-glyph-retention-region-mixup-counts-v1',objective=OBJECTIVE,
            objective_variant=OBJECTIVE_VARIANT,full_region_tile_histogram={str(k):v for k,v in sorted(self.tile_counts.items())},
            full_region_tiles=self.total_tiles,mixup_batches=self.steps,mixup_examples=self.steps*96,
            mixup_lambda_sum=self.lambda_sum,mixup_lambda_square_sum=self.lambda_square_sum,
            mixup_lambda_min=self.lambda_min,mixup_lambda_max=self.lambda_max,
            mixup_pairs_by_unknown_count=dict(self.pairs),mixup_different_target_pairs=self.different,
            mixup_self_pairs=self.self_pairs,synthetic_unknown_labels=False,mixup_deployed=False)
        return result


def validate_mixup_counts(report,steps):
    require(report.get('schema')=='flux-glyph-retention-region-mixup-counts-v1'
            and report.get('objective_variant')==OBJECTIVE_VARIANT and report.get('objective')==OBJECTIVE
            and report.get('steps')==report.get('mixup_batches')==steps
            and report.get('rows')==report.get('mixup_examples')==steps*96
            and report.get('test_read') is False and report.get('development_holdout_read') is False
            and report.get('synthetic_unknown_labels') is False and report.get('mixup_deployed') is False,
            'Region mixup execution scope differs')
    hist=report['full_region_tile_histogram'];require(set(hist)<=set(map(str,range(1,9)))
        and all(type(v) is int and v>=0 for v in hist.values()) and sum(hist.values())==steps*96
        and sum(int(k)*v for k,v in hist.items())==report['full_region_tiles'],'Full region tile accounting differs')
    pairs=report['mixup_pairs_by_unknown_count'];require(set(pairs)<={'0','1','2'}
        and all(type(v) is int and v>=0 for v in pairs.values()) and sum(pairs.values())==steps*96,
        'Mixup pairing totals differ')
    for key in ('mixup_lambda_sum','mixup_lambda_square_sum','mixup_lambda_min','mixup_lambda_max'):
        require(type(report[key]) in (int,float) and math.isfinite(report[key]),'Invalid mixup lambda statistics')
    require(0<=report['mixup_lambda_min']<=report['mixup_lambda_max']<=1
            and 0<=report['mixup_lambda_square_sum']<=report['mixup_lambda_sum']<=steps,
            'Mixup lambda values differ from a convex combination')
    for key in ('mixup_self_pairs','mixup_different_target_pairs'):
        require(type(report[key]) is int and 0<=report[key]<=steps*96,'Invalid mixup pair count')
    sources=report['source_target_rows']
    for row in sources:
        require(all(type(row[k]) is int and 0<=row[k]<=row['rows'] for k in ('weight_two_rows','base_kl_rows'))
                and (row['base_kl_rows']==0 or row['target_family'] not in (*core.CORE_FAMILIES,core.UNKNOWN)),
                'Base KL included unknown/core targets or invalid population counts')
    require(sum(r['rows'] for r in sources)==steps*96
            and sum(r['weight_two_rows'] for r in sources)==report['weight_two_rows']
            and sum(r['base_kl_rows'] for r in sources)==report['base_kl_rows'], 'Region mixup source totals differ')
    return report


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
        ROOT/'src/flux_glyph/unified_font.py',ROOT/'src/flux_glyph/region_font.py',ROOT/'training/evaluate_unified_retention_region_mixup.py',ROOT/'training/retention_region_mixup_loss.py',
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
    protocol={'schema':'flux-glyph-unified-retention-region-mixup-protocol-v1','architecture':ARCHITECTURE,'families':families,
        'objective_variant':OBJECTIVE_VARIANT,'objective':OBJECTIVE,'sampling':SAMPLING,'fixed_runtime':FIXED_RUNTIME,
        'region_tile_training':'All tiles of each sampled region; mean-softmax supervised CE and known-only base KL.',
        'manifold_mixup':MIXUP,'feature_cache_reused':True,'cached_feature_extraction_performed':False,
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
    sampler=RetentionSampler(training['rows'],families,args.seed,plan['train_pools']);counts=MixupCounts()
    optimizer=torch.optim.AdamW(model.residual_family_head.parameters(),lr=LEARNING_RATE,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=MINIMUM_LEARNING_RATE)
    best=None;history=[];started=time.monotonic()
    for step in range(1,STEPS+1):
        rows=sampler.batch();tile_counts=[r['tile_count'] for r in rows]
        indices=[index for row in rows for index in range(row['tile_start'],row['tile_start']+row['tile_count'])]
        features=torch.from_numpy(np.array(cache['train']['features'][indices],copy=True));base=torch.from_numpy(np.array(cache['train']['base_logits'][indices],copy=True))
        starts=np.cumsum([0,*tile_counts[:-1]]);offsets=[int(sampler.rng.integers(count)) for count in tile_counts]
        chosen=[int(start)+offset for start,offset in zip(starts,offsets)]
        permutation=sampler.rng.permutation(BATCH_SIZE).tolist();lam=float(sampler.rng.beta(.4,.4))
        targets=torch.tensor([r['target'] for r in rows]);model.train()
        losses=region_mixup_loss(model,features,base,tile_counts,targets,rows,families,chosen,permutation,lam)
        loss,ce,kl,mix=losses['loss'],losses['region_ce'],losses['base_kl'],losses['mixup_ce']
        counts.update(rows,losses['kl_mask'].detach().tolist(),losses['ce_weights'],tile_counts,lam,permutation)
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.residual_family_head.parameters(),5.)
        optimizer.step();scheduler.step()
        if step%100==0:print(json.dumps({'step':step,'loss':float(loss.detach()),'ce':float(ce.detach()),'base_kl':float(kl.detach()),'mixup_ce':float(mix.detach()),'mixup_lambda':lam,'region_tiles':len(indices),'seconds':round(time.monotonic()-started,1)}),flush=True)
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
    training_counts=counts.report();validate_mixup_counts(training_counts,STEPS)
    dump(args.output/'TRAINING_COUNTS.json',training_counts)
    for key,name in [('outputs','CALIBRATION_OUTPUTS.npz'),('decisions','CALIBRATION_DECISIONS.json')]:
        (args.output/name).write_bytes((args.output/selected['artifacts'][key]['path']).read_bytes())
    selection={**protocol,'schema':'flux-glyph-unified-retention-region-mixup-selection-v1','selected':selected,'history':history,
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
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/unified-font-v2/run-region-mixup-v1')
    p.add_argument('--cache',type=Path,default=ROOT/'artifacts/unified-font-v2/run-adapter-v1/cache')
    p.add_argument('--steps',type=int,default=STEPS);p.add_argument('--seed',type=int,default=2026091406)
    return p


if __name__=='__main__':train(parser().parse_args())
