#!/usr/bin/env python3
"""Train only a residual font head on frozen native TRAIN/CAL CNN features."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'training')]
import train_unified_retention_core as core
from train_unified_retention import RetentionSampler,evaluate_outputs,retention_rank,FIXED_RUNTIME,SAMPLING
from train_regions import require,sha,dump,state_sha

ARCHITECTURE='region-cnn64x256-residual-head-v1'
BASE_CHECKPOINT_SHA='f2f7f7c6d44aad5b3e22a2ba7d5c78b069fbb0a04423ee9bf186bfa60e52533e'
BASE_SELECTION_SHA='18b5cbf95e45fee83283256581f32c94dae80f653e2fdfe1e24ddf25bfb414da'
STEPS=2000;EVAL_EVERY=200;BATCH_SIZE=96;LEARNING_RATE=5e-4;MINIMUM_LEARNING_RATE=5e-5
OBJECTIVE_VARIANT='native_core_unknown_weighted_residual_head'
OBJECTIVE={'family_cross_entropy_label_smoothing':.03,'native_ios_core_weight':2.,'unknown_weight':2.,
    'other_weight':1.,'ce_denominator':96,'size_loss_weight':0.,'base_kl_weight':1.,'base_temperature':1.,
    'base_kl_mask':'TRAIN true target outside PingFang/SF Pro/Helvetica and frozen base argmax equals target',
    'base_kl_reduction':'Mean eligible KL(base || adapted); differentiable zero for empty mask',
    'native_core_families':list(core.CORE_FAMILIES),'native_weight_split':'train','native_weight_domain':'ios',
    'native_weight_view':'native','label_weights_are_inference_rules':False,'base_frozen':True,
    'size_head_frozen':True,'trainable_parameter_prefix':'residual_family_head.','external_teacher':False}


def read(path):return json.loads(Path(path).read_text())
def residual_state(state):return {k:v for k,v in state.items() if k.startswith('residual_family_head.')}
def frozen_state(state):return {k:v for k,v in state.items() if not k.startswith('residual_family_head.')}


def adapter_loss(logits,base_logits,targets,rows,families):
    import torch
    require(len(families)==25 and logits.shape==base_logits.shape==(BATCH_SIZE,25)
            and targets.shape==(BATCH_SIZE,) and targets.dtype==torch.int64 and not base_logits.requires_grad
            and all(bool(torch.isfinite(v).all()) for v in (logits,base_logits)), 'Invalid residual-head loss inputs')
    values=core.native_core_weights(rows,targets.detach().cpu().tolist(),families)
    values=[2. if r['family']==core.UNKNOWN else value for r,value in zip(rows,values)]
    weight=torch.tensor(values,dtype=logits.dtype,device=logits.device)
    is_core=torch.zeros_like(targets,dtype=torch.bool)
    for family in core.CORE_FAMILIES:is_core|=targets==families.index(family)
    mask=(~is_core)&(base_logits.argmax(1)==targets)
    ce=(torch.nn.functional.cross_entropy(logits,targets,label_smoothing=.03,reduction='none')*weight).sum()/96
    kl=(torch.nn.functional.kl_div(logits[mask].log_softmax(1),base_logits[mask].softmax(1),reduction='none').sum(1).mean()
        if bool(mask.any()) else logits.sum()*0.)
    require(bool(torch.isfinite(ce+kl)),'Nonfinite residual-head loss')
    return ce+kl,ce,kl,mask,values


def cache_identity(datasets,base_checkpoint,base_state_sha,bindings,feature_device='mps'):
    require(set(datasets)=={'train','calibration'},'Cache may contain only TRAIN and CAL')
    partitions={}
    for split,data in datasets.items():
        require(data['partition']['split']==split and all(r['split']==split for r in data['rows']), 'Cache source split differs')
        partitions[split]={'partition_sha256':data['partition_sha256'],'tile_count':len(data['tiles']),
            'rows':data['partition']['metadata'],'tiles':data['partition']['array'],
            'order':'Exact prepared tile array order; rows retain tile_start and tile_count.'}
    require(datasets['train']['families']==datasets['calibration']['families']
            and datasets['train']['manifest_sha256']==datasets['calibration']['manifest_sha256'], 'Cache class or root identity differs')
    require(feature_device in ('cpu','mps'),'Invalid cache extraction device')
    return {'architecture':ARCHITECTURE,'base_checkpoint':base_checkpoint,'base_state_sha256':base_state_sha,'feature_device':feature_device,
        'families':datasets['train']['families'],'data_manifest_sha256':datasets['train']['manifest_sha256'],
        'partitions':partitions,'bindings':dict(bindings),'test_read':False,'development_holdout_read':False}


def load_cache(folder,identity):
    manifest=read(folder/'CACHE_MANIFEST.json');freeze=read(folder/'CACHE_FREEZE.json')
    require(manifest.get('schema')=='flux-glyph-retention-adapter-cache-v1' and manifest.get('identity')==identity
            and manifest.get('cache_freeze_sha256')==sha(folder/'CACHE_FREEZE.json')
            and freeze.get('schema')=='flux-glyph-retention-adapter-cache-freeze-v1'
            and freeze.get('identity')==identity and freeze.get('feature_device')==identity['feature_device']
            and freeze.get('batch_size')==128 and freeze.get('optimizer_steps_executed')==0
            and set(manifest['partitions'])=={'train','calibration'},'Frozen cache identity differs')
    core.verify_bindings(identity['bindings']);result={}
    for split in ('train','calibration'):
        count=identity['partitions'][split]['tile_count'];arrays={};part=manifest['partitions'][split]
        require(set(part)=={'features','base_logits','log_em_ratio'},'Cache outputs differ')
        for name,shape in (('features',(count,128)),('base_logits',(count,25)),('log_em_ratio',(count,))):
            item=part[name];relative=f'{split}/{name}.npy';path=folder/relative
            require(item.get('path')==relative and item.get('shape')==list(shape) and item.get('dtype')=='float32'
                    and sha(path)==item.get('sha256'),'Cache array bytes or shape binding differs')
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            require(array.dtype==np.float32 and array.shape==shape and bool(np.isfinite(array).all()),'Invalid cached array')
            if name=='log_em_ratio':require(bool((np.abs(array)<=3).all()),'Invalid frozen size output')
            arrays[name]=array
        result[split]=arrays
    return result,manifest


def extract_cache(folder,model,datasets,identity,device):
    import torch
    if folder.exists():return load_cache(folder,identity)
    core.verify_bindings(identity['bindings']);folder.mkdir(parents=True)
    dump(folder/'CACHE_FREEZE.json',{'schema':'flux-glyph-retention-adapter-cache-freeze-v1','identity':identity,
        'feature_device':device,'torch_version':str(torch.__version__),'batch_size':128,'optimizer_steps_executed':0})
    freeze_sha=sha(folder/'CACHE_FREEZE.json');base_before=state_sha(frozen_state(model.state_dict()))
    model.eval().to(device);partitions={}
    with torch.inference_mode():
        for split in ('train','calibration'):
            data=datasets[split];count=len(data['tiles']);directory=folder/split;directory.mkdir()
            arrays={name:np.lib.format.open_memmap(directory/(name+'.npy'),mode='w+',dtype=np.float32,shape=shape)
                for name,shape in (('features',(count,128)),('base_logits',(count,25)),('log_em_ratio',(count,)))}
            for start in range(0,count,128):
                tiles=torch.from_numpy(np.array(data['tiles'][start:start+128],copy=True)).to(device)
                features=model.features(tiles);base=model.family_head(features);size=model.size_head(features).squeeze(-1)
                arrays['features'][start:start+len(tiles)]=features.cpu().numpy()
                arrays['base_logits'][start:start+len(tiles)]=base.cpu().numpy()
                arrays['log_em_ratio'][start:start+len(tiles)]=size.cpu().numpy()
            partitions[split]={}
            for name,array in arrays.items():
                array.flush();path=directory/(name+'.npy')
                partitions[split][name]={'path':f'{split}/{name}.npy','shape':list(array.shape),'dtype':'float32','sha256':sha(path)}
            print(json.dumps({'cache_split':split,'tiles':count,'feature_device':device,'test_read':False}),flush=True)
    require(sha(folder/'CACHE_FREEZE.json')==freeze_sha and state_sha(frozen_state(model.state_dict()))==base_before,
            'Cache extraction altered the freeze or base parameters')
    core.verify_bindings(identity['bindings'])
    dump(folder/'CACHE_MANIFEST.json',{'schema':'flux-glyph-retention-adapter-cache-v1','identity':identity,
        'cache_freeze_sha256':freeze_sha,'partitions':partitions})
    return load_cache(folder,identity)


def infer_cached(model,cache):
    import torch
    model.eval();logits=[]
    with torch.inference_mode():
        for start in range(0,len(cache['features']),1024):
            features=torch.from_numpy(np.array(cache['features'][start:start+1024],copy=True))
            base=torch.from_numpy(np.array(cache['base_logits'][start:start+1024],copy=True))
            logits.append((base+model.residual_family_head(features)).numpy())
    return np.concatenate(logits).astype(np.float32,copy=False),np.array(cache['log_em_ratio'],copy=True)


class TrainingCounts:
    def __init__(self):self.rows=0;self.steps=0;self.groups={};self.eligible=0;self.weighted=0
    def update(self,rows,mask,weights):
        require(len(rows)==len(mask)==len(weights)==96,'Incomplete adapter training batch accounting')
        for row,eligible,weight in zip(rows,mask,weights):
            key=(row['domain'],row['source_font_family'],row['family']);value=self.groups.setdefault(key,Counter())
            require(type(eligible) is bool and weight in (1.,2.) and (not eligible or row['family'] not in core.CORE_FAMILIES),
                    'Invalid base-correct KL mask or supervised weight')
            value['rows']+=1;value['weight_two_rows']+=int(weight==2.);value['base_kl_rows']+=int(eligible)
            self.rows+=1;self.eligible+=int(eligible);self.weighted+=int(weight==2.)
        self.steps+=1
    def report(self):
        return {'schema':'flux-glyph-retention-adapter-training-counts-v1','steps':self.steps,'rows':self.rows,
            'weight_two_rows':self.weighted,'base_kl_rows':self.eligible,'objective':OBJECTIVE,
            'source_target_rows':[{'domain':d,'source_font_family':s,'target_family':f,**dict(value)}
                for (d,s,f),value in sorted(self.groups.items())],'test_read':False,'development_holdout_read':False}


def train(args):
    import torch
    from retention_adapter_network import RetentionAdapterClassifier,ARCHITECTURE as NETWORK_ARCH
    require(NETWORK_ARCH==ARCHITECTURE,'Residual network architecture differs')
    args.data=args.data.resolve();args.output=args.output.resolve();args.checkpoint=args.checkpoint.resolve();args.plan=args.plan.resolve()
    args.cache=args.cache.resolve() if args.cache else args.output/'cache'
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
        ROOT/'src/flux_glyph/unified_font.py',ROOT/'src/flux_glyph/region_font.py',ROOT/'training/evaluate_unified_retention_adapter.py',
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
    identity=cache_identity(datasets,base_identity,base_before,bindings,args.feature_device)
    cache,cache_manifest=extract_cache(args.cache,model,datasets,identity,args.feature_device)
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
    protocol={'schema':'flux-glyph-unified-retention-adapter-protocol-v1','architecture':ARCHITECTURE,'families':families,
        'objective_variant':OBJECTIVE_VARIANT,'objective':OBJECTIVE,'sampling':SAMPLING,'fixed_runtime':FIXED_RUNTIME,
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
    sampler=RetentionSampler(training['rows'],families,args.seed,plan['train_pools']);counts=TrainingCounts()
    optimizer=torch.optim.AdamW(model.residual_family_head.parameters(),lr=LEARNING_RATE,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=MINIMUM_LEARNING_RATE)
    best=None;history=[];started=time.monotonic()
    for step in range(1,STEPS+1):
        rows=sampler.batch();indices=[r['tile_start']+int(sampler.rng.integers(r['tile_count'])) for r in rows]
        features=torch.from_numpy(np.array(cache['train']['features'][indices],copy=True));base=torch.from_numpy(np.array(cache['train']['base_logits'][indices],copy=True))
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
    selected=best[0];dump(args.output/'SAMPLING.json',sampler.report());dump(args.output/'TRAINING_COUNTS.json',counts.report())
    for key,name in [('outputs','CALIBRATION_OUTPUTS.npz'),('decisions','CALIBRATION_DECISIONS.json')]:
        (args.output/name).write_bytes((args.output/selected['artifacts'][key]['path']).read_bytes())
    selection={**protocol,'schema':'flux-glyph-unified-retention-adapter-selection-v1','selected':selected,'history':history,
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
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/unified-font-v2/run-adapter-v1')
    p.add_argument('--cache',type=Path)
    p.add_argument('--feature-device',choices=('mps','cpu'),default='mps')
    p.add_argument('--steps',type=int,default=STEPS);p.add_argument('--seed',type=int,default=2026091405)
    return p


if __name__=='__main__':train(parser().parse_args())
