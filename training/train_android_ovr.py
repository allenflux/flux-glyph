#!/usr/bin/env python3
"""Train independent Android font heads on bound TRAIN/CAL image embeddings."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'training')]
from train_regions import sha,dump,require,state_sha
from prepare_android_regions import load_split,UNKNOWN
from train_android_regions import POLICY,select_grid
from train_android_balanced import BalancedSampler,VIEW_PROBABILITIES,validate_initializer,bind_inputs

ARCHITECTURE='android_ovr_frozen_encoder_9x_mlp128_32_1_v1'
BATCH_SIZE=128
STEPS=4000
LEARNING_RATE=.001
OBJECTIVE={'positive_known_bce_weight':.5,'other_known_negative_bce_weight':.25,
    'unknown_all_known_negative_bce_weight':.25,'size_loss_weight':0.,
    'formula':'.5 * mean(BCE(known row, true head, 1)) + .25 * mean(BCE(known row, other eight heads, 0)) + .25 * mean(BCE(unknown row, all nine heads, 0))',
    'unknown_logit':'Fixed zero baseline; not trainable; no loss term on this output.'}
SAMPLING={'known_fraction':.5,'unknown_fraction':.5,'batch_size':BATCH_SIZE,
    'implementation':'Concatenate two independent consecutive BalancedSampler batches of 64 rows.',
    'known':'Equal cycling over nine classes; then view, region and tile.',
    'unknown':'Equal cycling over source families, then source file/TTC faces; then view, region and tile.',
    'view_probabilities':VIEW_PROBABILITIES,'tile':'Uniformly sample one cached image embedding per region/view.'}


def balanced_batch(sampler):
    rows=sampler.batch()+sampler.batch()
    require(len(rows)==BATCH_SIZE and sum(row['target']==sampler.unknown for row in rows)==BATCH_SIZE//2,
            'OVR batch must contain exactly 64 known and 64 unknown regions')
    return rows


def ovr_loss(logits,targets,unknown_index):
    import torch
    require(logits.ndim==2 and logits.shape[1]==10 and targets.shape==(len(logits),)
            and type(unknown_index) is int and 0<=unknown_index<10
            and targets.dtype==torch.long and bool(((targets>=0)&(targets<10)).all()),'invalid OVR loss inputs')
    known_mask=targets!=unknown_index
    require(bool(known_mask.any()) and bool((~known_mask).any()),'OVR loss requires known and unknown rows')
    known_indices=[i for i in range(10) if i!=unknown_index]
    known_logits=logits[known_mask][:,known_indices]
    positive_mask=targets[known_mask,None]==torch.tensor(known_indices,device=targets.device)[None,:]
    positive=torch.nn.functional.softplus(-known_logits[positive_mask]).mean()
    other=torch.nn.functional.softplus(known_logits[~positive_mask]).mean()
    unknown=torch.nn.functional.softplus(logits[~known_mask][:,known_indices]).mean()
    return .5*positive+.25*other+.25*unknown,{'positive_known_bce':positive,
        'other_known_negative_bce':other,'unknown_all_known_negative_bce':unknown}


def verify_bindings(bindings):
    require(isinstance(bindings,dict) and bindings and all(Path(path).is_file() and sha(path)==digest
            for path,digest in bindings.items()),'OVR bound input changed')


def cache_identity(datasets,parent_checkpoint,parent_frozen_sha,bindings):
    require(set(datasets)=={'train','calibration'},'embedding cache only accepts TRAIN and CAL')
    partitions={}
    for split,data in datasets.items():
        require(data['partition']['split']==split and all(row.get('split')==split for row in data['rows']),
                'embedding split metadata differs')
        partitions[split]={'manifest_sha256':data['partition_sha256'],'tiles':len(data['tiles']),
            'tile_array':data['partition']['array'],'row_metadata':data['partition']['metadata']}
    require(datasets['train']['manifest_sha256']==datasets['calibration']['manifest_sha256']
            and datasets['train']['families']==datasets['calibration']['families'],'embedding data sources differ')
    return {'architecture':ARCHITECTURE,'parent_checkpoint':parent_checkpoint,
        'parent_frozen_state_sha256':parent_frozen_sha,'data_manifest_sha256':datasets['train']['manifest_sha256'],
        'families':datasets['train']['families'],'partitions':partitions,'bindings':bindings,'test_read':False}


def load_cache(root,expected):
    """Every reuse verifies provenance and actual array bytes before opening memmaps."""
    root=Path(root).resolve();manifest_path=root/'MANIFEST.json'
    require(manifest_path.is_file(),'incomplete embedding cache; use a new cache directory')
    manifest=json.loads(manifest_path.read_text())
    require(manifest.get('schema')=='flux-glyph-android-ovr-embedding-cache-v1'
            and manifest.get('identity')==expected and set(manifest.get('partitions',{}))=={'train','calibration'},
            'embedding cache identity/partition binding differs')
    plan_path=root/'PLAN.json'
    require(plan_path.is_file() and sha(plan_path)==manifest.get('extraction_plan_sha256'),
            'embedding extraction plan changed')
    plan=json.loads(plan_path.read_text())
    require(plan.get('identity')==expected and plan.get('device') in ('mps','cpu'),
            'embedding extraction plan identity differs')
    verify_bindings(expected['bindings'])
    result={}
    for split in ('train','calibration'):
        partition=manifest['partitions'][split];count=expected['partitions'][split]['tiles'];arrays={}
        require(set(partition)=={'features','log_em_ratio'},'unexpected embedding output')
        for name,shape in (('features',(count,128)),('log_em_ratio',(count,))):
            descriptor=partition[name];relative=f'{split}/{name}.npy';path=root/relative
            require(descriptor.get('path')==relative and descriptor.get('shape')==list(shape)
                    and descriptor.get('dtype')=='float32' and path.is_file() and sha(path)==descriptor.get('sha256'),
                    'embedding cache array bytes/shape binding differs')
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            require(array.dtype==np.float32 and array.shape==shape and bool(np.isfinite(array).all()),
                    'invalid embedding array shape/dtype/values')
            if name=='log_em_ratio':require(bool((np.abs(array)<=3).all()),'invalid cached size output')
            arrays[name]=array
        result[split]=arrays
    return result,manifest


def extract_cache(root,model,datasets,identity,device):
    import torch
    root=Path(root).resolve()
    require(set(datasets)=={'train','calibration'} and identity.get('test_read') is False,
            'embedding extraction only accepts TRAIN and CAL')
    if root.exists():return load_cache(root,identity)
    verify_bindings(identity['bindings']);root.mkdir(parents=True)
    dump(root/'PLAN.json',{'schema':'flux-glyph-android-ovr-embedding-plan-v1','identity':identity,
        'device':device,'torch_version':str(torch.__version__),'batch_size':256})
    model.eval();partitions={}
    with torch.inference_mode():
        for split in ('train','calibration'):
            data=datasets[split];directory=root/split;directory.mkdir();count=len(data['tiles'])
            arrays={name:np.lib.format.open_memmap(directory/(name+'.npy'),mode='w+',dtype=np.float32,shape=shape)
                    for name,shape in (('features',(count,128)),('log_em_ratio',(count,)))}
            for start in range(0,count,256):
                tiles=torch.from_numpy(np.array(data['tiles'][start:start+256],copy=True)).to(device)
                features=model.features(tiles);ratios=model.size_head(features).squeeze(-1)
                arrays['features'][start:start+len(tiles)]=features.cpu().numpy()
                arrays['log_em_ratio'][start:start+len(tiles)]=ratios.cpu().numpy()
            partitions[split]={}
            for name,array in arrays.items():
                array.flush();path=directory/(name+'.npy')
                partitions[split][name]={'path':f'{split}/{name}.npy','sha256':sha(path),
                    'shape':list(array.shape),'dtype':'float32'}
            print(json.dumps({'embedding_split':split,'tiles':count,'test_read':False}),flush=True)
    verify_bindings(identity['bindings'])
    dump(root/'MANIFEST.json',{'schema':'flux-glyph-android-ovr-embedding-cache-v1','identity':identity,
        'partitions':partitions,'extraction_plan_sha256':sha(root/'PLAN.json')})
    return load_cache(root,identity)


def infer_cached(model,cache,device):
    import torch
    model.eval();chunks=[]
    with torch.inference_mode():
        for start in range(0,len(cache['features']),1024):
            features=torch.from_numpy(np.array(cache['features'][start:start+1024],copy=True)).to(device)
            chunks.append(model.classify(features).cpu().numpy())
    logits=np.concatenate(chunks).astype(np.float32,copy=False)
    return logits,np.array(cache['log_em_ratio'],dtype=np.float32,copy=True)


def frozen_state(state):return {key:value for key,value in state.items() if not key.startswith('heads.')}
def head_state(state):return {key:value for key,value in state.items() if key.startswith('heads.')}


def verify_inheritance(model,parent):
    import torch
    expected={key:value for key,value in parent.items() if not key.startswith('family_head.')}
    actual=frozen_state(model.state_dict())
    require(actual.keys()==expected.keys() and all(actual[key].dtype==expected[key].dtype
            and actual[key].shape==expected[key].shape and torch.equal(actual[key].detach().cpu(),expected[key].detach().cpu())
            for key in expected),'OVR encoder or size parameters differ from the parent')
    require(all(parameter.requires_grad==key.startswith('heads.') for key,parameter in model.named_parameters()),
            'OVR must train only its independent heads')
    return state_sha(actual)


def train(args):
    import torch
    from region_network import RegionFontClassifier
    from android_ovr_network import AndroidOVRClassifier,ARCHITECTURE as NETWORK_ARCHITECTURE
    require(NETWORK_ARCHITECTURE==ARCHITECTURE,'OVR architecture contract differs')
    require(not args.output.exists(),'OVR training output must be new')
    require(args.steps==STEPS,'OVR protocol fixes exactly 4000 optimization steps')
    args.data=Path(args.data).resolve();args.checkpoint=Path(args.checkpoint).resolve();args.output=args.output.resolve()
    training=load_split(args.data,'train');cal=load_split(args.data,'calibration')
    require(training['families']==cal['families'] and training['manifest_sha256']==cal['manifest_sha256'],
            'OVR TRAIN/CAL sources differ')
    families=training['families'];sampler=BalancedSampler(training['rows'],families,args.seed)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    parent_ref={'path':str(args.checkpoint),'sha256':sha(args.checkpoint)}
    parent=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    initializer_selection=args.checkpoint.parent/'SELECTION.json';initializer_protocol=args.checkpoint.parent/'TRAINING_FREEZE.json'
    prior=json.loads(initializer_selection.read_text());prior_protocol=json.loads(initializer_protocol.read_text())
    validate_initializer(RegionFontClassifier(10).state_dict(),parent,families,prior,prior_protocol,
        sha(initializer_selection),sha(initializer_protocol),training['manifest_sha256'],state_sha(parent['state_dict']))
    verify_bindings(prior['bindings'])
    source_bindings=dict(prior['bindings'])
    for path,digest in training['manifest']['bindings'].items():
        require(path not in source_bindings or source_bindings[path]==digest,'OVR parent/data source binding conflict')
        source_bindings[path]=digest
    bindings=bind_inputs(args.data,args.checkpoint,initializer_selection,initializer_protocol,source_bindings)
    for path in (Path(__file__).resolve(),ROOT/'training/android_ovr_network.py'):
        bindings[str(path)]=sha(path)
    require(bindings[str(args.checkpoint)]==parent_ref['sha256'],'OVR parent changed while loading')
    model=AndroidOVRClassifier(10,sampler.unknown).from_parent(parent['state_dict'])
    frozen_sha=verify_inheritance(model,parent['state_dict']);before=state_sha(model.state_dict())
    heads_before=state_sha(head_state(model.state_dict()))
    args.output.mkdir(parents=True)
    initial_heads=args.output/'INITIAL_HEADS.pth';torch.save(head_state(model.state_dict()),initial_heads)
    model.to(args.feature_device)
    datasets={'train':training,'calibration':cal};identity=cache_identity(datasets,parent_ref,frozen_sha,dict(bindings))
    cache_root=args.output/'embedding-cache'
    cache,cache_document=extract_cache(cache_root,model,datasets,identity,args.feature_device)
    require(verify_inheritance(model,parent['state_dict'])==frozen_sha,'feature extraction changed frozen parameters')
    cache_ref={'path':str(cache_root/'MANIFEST.json'),'sha256':sha(cache_root/'MANIFEST.json')}
    bindings[cache_ref['path']]=cache_ref['sha256'];bindings[str(initial_heads)]=sha(initial_heads)
    bindings[str(cache_root/'PLAN.json')]=sha(cache_root/'PLAN.json')
    for partition in cache_document['partitions'].values():
        for descriptor in partition.values():bindings[str(cache_root/descriptor['path'])]=descriptor['sha256']
    common={'architecture':ARCHITECTURE,'parent_checkpoint':parent_ref,'parent_frozen_state_sha256':frozen_sha,
        'cache_manifest':cache_ref,'unknown_index':sampler.unknown,
        'cached_output_execution':{'features_device':args.feature_device,'heads_device':args.device,
                                   'size_head_device':args.feature_device}}
    inheritance={'checkpoint':str(args.checkpoint),'checkpoint_sha256':parent_ref['sha256'],
        'selection_sha256':sha(initializer_selection),'protocol_sha256':sha(initializer_protocol),
        'source_state_sha256':state_sha(parent['state_dict']),'families':families,
        'source_optimizer_steps_executed':prior['optimizer_steps_executed'],'source_selected_step':prior['selected']['step'],
        'source_calibration_passed':prior.get('calibration_passed') is True,
        'encoder_and_size_head_inherited_in_full':True,'encoder_and_size_head_frozen':True,
        'discarded_parent_parameters':['family_head.weight','family_head.bias'],
        'newly_initialized_parameters':sorted(head_state(model.state_dict())),
        'additional_optimizer_steps_planned':STEPS,'all_parameters_trainable':False}
    protocol={'schema':'flux-glyph-android-training-protocol-v1','policy':POLICY,'bindings':bindings,'families':families,
        'steps':STEPS,'batch_size':BATCH_SIZE,'seed':args.seed,'learning_rate':LEARNING_RATE,'device':args.device,
        'initial_state_sha256':before,'heads_before_sha256':heads_before,'test_read':False,
        'training_inputs':['frozen_image_embeddings'],'initialization':'Nine new independent binary MLP heads; complete parent encoder and size head inherited and frozen.',
        'initializer_evidence':inheritance,'sampling':SAMPLING,'objective':OBJECTIVE,
        'optimizer':'Only heads.parameters(); AdamW weight_decay=.0002; cosine decay from .001 to .00001 in 4000 steps; gradient norm clip 5.',**common}
    verify_bindings(bindings);dump(args.output/'TRAINING_FREEZE.json',protocol)
    model.to(args.device)
    optimizer=torch.optim.AdamW(model.heads.parameters(),lr=LEARNING_RATE,weight_decay=.0002)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=1e-5)
    best=None;history=[];started=time.monotonic()
    for step in range(1,STEPS+1):
        rows=balanced_batch(sampler)
        indices=[row['tile_start']+int(sampler.rng.integers(row['tile_count'])) for row in rows]
        features=torch.from_numpy(np.array(cache['train']['features'][indices],copy=True)).to(args.device)
        targets=torch.tensor([row['target'] for row in rows],dtype=torch.long,device=args.device)
        model.train();logits=model.classify(features);loss,parts=ovr_loss(logits,targets,sampler.unknown)
        require(bool(torch.isfinite(loss)),'nonfinite OVR training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.heads.parameters(),5.)
        optimizer.step();scheduler.step()
        if step%100==0:
            print(json.dumps({'step':step,'loss':float(loss.detach().cpu()),
                **{key:float(value.detach().cpu()) for key,value in parts.items()},'seconds':round(time.monotonic()-started,1)}),flush=True)
        if step%1000==0:
            logits,ratios=infer_cached(model,cache['calibration'],args.device)
            selected,grid=select_grid(logits,ratios,cal,step)
            history.append(selected[1]);dump(args.output/f'CAL_GRID_{step:05d}.json',grid)
            if best is None or selected[0]>best[0]:
                best=(selected[0],copy.deepcopy(selected[1]),{key:value.detach().cpu().clone() for key,value in model.state_dict().items()},
                      copy.deepcopy(selected[2]),logits.copy(),ratios.copy())
            dump(args.output/'CAL_PROGRESS.json',history);dump(args.output/'SAMPLING_PROGRESS.json',sampler.report())
            print(json.dumps({'calibration_step':step,'selection':selected[1]},ensure_ascii=False),flush=True)
    model.cpu().load_state_dict(best[2]);after=state_sha(model.state_dict());heads_after=state_sha(head_state(model.state_dict()))
    require(before!=after and heads_before!=heads_after,'OVR heads did not change')
    require(verify_inheritance(model,parent['state_dict'])==frozen_sha,'OVR training changed frozen parameters')
    require(all(value.grad is None for key,value in model.named_parameters() if not key.startswith('heads.')),
            'a frozen OVR parameter received gradients')
    verify_bindings(bindings)
    dump(args.output/'CALIBRATION_DECISIONS.json',{'records':best[3],'families':families})
    np.savez(args.output/'CALIBRATION_OUTPUTS.npz',logits=best[4],log_em_ratio=best[5])
    dump(args.output/'SAMPLING.json',sampler.report())
    selection={'schema':'flux-glyph-android-training-selection-v1','policy':POLICY,'selected':best[1],'families':families,
        'passed':best[1]['metrics']['passed'],'calibration_passed':best[1]['metrics']['passed'],'test_read':False,
        'data_manifest_sha256':training['manifest_sha256'],'bindings':bindings,
        'training_protocol_sha256':sha(args.output/'TRAINING_FREEZE.json'),'optimizer_steps_executed':STEPS,
        'training_device':args.device,'state_before_sha256':before,'state_after_sha256':after,
        'heads_before_sha256':heads_before,'heads_after_sha256':heads_after,
        'calibration_decisions_sha256':sha(args.output/'CALIBRATION_DECISIONS.json'),
        'calibration_outputs_sha256':sha(args.output/'CALIBRATION_OUTPUTS.npz'),'history':history,
        'initializer_evidence':inheritance,'sampling_sha256':sha(args.output/'SAMPLING.json'),**common}
    dump(args.output/'SELECTION.json',selection)
    torch.save({'state_dict':model.state_dict(),'families':families,'selection_sha256':sha(args.output/'SELECTION.json'),
        'architecture':ARCHITECTURE,'unknown_index':sampler.unknown,'parent_checkpoint_sha256':parent_ref['sha256'],
        'parent_frozen_state_sha256':frozen_sha},args.output/'model.pth')
    print(json.dumps({'passed':selection['passed'],'selected_step':best[1]['step'],'output':str(args.output)}),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/android-font-v1/run-v2/model.pth')
    p.add_argument('--steps',type=int,default=STEPS);p.add_argument('--seed',type=int,default=2026091304)
    p.add_argument('--device',choices=('mps','cpu'),default='cpu')
    p.add_argument('--feature-device',choices=('mps','cpu'),default='mps')
    return p


if __name__=='__main__':train(parser().parse_args())
