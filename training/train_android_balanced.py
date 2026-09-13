#!/usr/bin/env python3
"""Balanced Android fine-tuning of a complete ten-class checkpoint; CAL only."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import copy
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'training')]
from train_regions import sha,dump,require,state_sha
from prepare_android_regions import load_split,UNKNOWN,RECIPES
from train_android_regions import POLICY,select_grid,infer,validate_initialization

BATCH_SIZE=64
VIEW_PROBABILITIES={'native':.4,'half':.2,'three_quarters_jpeg75':.2,'jpeg75':.2}
SAMPLING={'known_fraction':.5,'unknown_fraction':.5,'batch_size':BATCH_SIZE,
    'known':'Cycle equally over the nine known classes, then sample a view and a native region uniformly.',
    'unknown':'Cycle equally over real source font families, then equally over source file/TTC faces, then sample a view and native region uniformly.',
    'view_probabilities':VIEW_PROBABILITIES,'tile':'Uniformly choose one image tile within the sampled native region/view.',
    'order':'Shuffle the complete batch after selecting exactly 32 known and 32 unknown rows.'}
OBJECTIVE={'ten_class_cross_entropy_label_smoothing':.03,'size_smooth_l1_weight':.2,
    'unknown_known_head_uniform_ce_weight':.5,
    'unknown_known_head_uniform_ce':'mean_over_unknown_rows(logsumexp(nine_known_logits) - mean(nine_known_logits))',
    'reference':'https://arxiv.org/abs/1812.04606',
    'scope':'Inspired by Outlier Exposure; not a reproduction of the paper. Explicit ten-class unknown supervision is retained.'}


class BalancedSampler:
    """Balance source families independently of their face/region/tile counts."""
    def __init__(self,rows,families,seed):
        require(len(families)==10 and len(set(families))==10 and families.count(UNKNOWN)==1,
                'balanced training requires nine known classes and one unknown')
        require(set(RECIPES)==set(VIEW_PROBABILITIES),'view sampling contract differs')
        self.families=list(families);self.unknown=families.index(UNKNOWN);self.rng=np.random.default_rng(seed)
        self.known_targets=[i for i in range(10) if i!=self.unknown]
        self.known=defaultdict(list);self.unknown_pools=defaultdict(list)
        unknown_faces=defaultdict(set)
        for row in rows:
            target=row.get('target');view=row.get('view')
            require(row.get('split')=='train' and type(target) is int and 0<=target<10
                    and row.get('family')==families[target] and view in RECIPES,
                    'sampler only accepts correctly labelled TRAIN regions')
            require(type(row.get('tile_count')) is int and 1<=row['tile_count']<=8
                    and type(row.get('tile_start')) is int and row['tile_start']>=0,'invalid native tile mapping')
            if target==self.unknown:
                family=row.get('source_font_family');digest=row.get('font_file_sha256');ttc=row.get('ttc_index')
                require(isinstance(family,str) and family and isinstance(digest,str)
                        and re.fullmatch('[0-9a-f]{64}',digest) and type(ttc) is int and ttc>=0,
                        'unknown source family/face identity is missing')
                face=(digest,ttc);unknown_faces[family].add(face)
                self.unknown_pools[(family,face,view)].append(row)
            else:self.known[(target,view)].append(row)
        self.unknown_families=sorted(unknown_faces)
        require(self.unknown_families,'unknown source families are missing')
        self.unknown_faces={family:sorted(faces) for family,faces in unknown_faces.items()}
        require(all(self.known[(target,view)] for target in self.known_targets for view in RECIPES),
                'a known class/view has no TRAIN regions')
        require(all(self.unknown_pools[(family,face,view)] for family,faces in self.unknown_faces.items()
                    for face in faces for view in RECIPES),'an unknown source family/face/view has no TRAIN regions')
        self.position=0;self.face_positions=Counter();self.known_counts=Counter();self.unknown_counts=Counter()
        self.face_counts=Counter();self.view_counts=Counter();self.total_counts=Counter()

    def batch(self):
        rows=[];views=list(VIEW_PROBABILITIES);probabilities=list(VIEW_PROBABILITIES.values())
        for index in range(BATCH_SIZE//2):
            position=self.position+index
            target=self.known_targets[position%len(self.known_targets)]
            view=str(self.rng.choice(views,p=probabilities));pool=self.known[(target,view)]
            rows.append(pool[int(self.rng.integers(len(pool)))])
            self.known_counts[self.families[target]]+=1;self.view_counts[('known',view)]+=1
            family=self.unknown_families[position%len(self.unknown_families)]
            faces=self.unknown_faces[family];face=faces[self.face_positions[family]%len(faces)]
            self.face_positions[family]+=1
            view=str(self.rng.choice(views,p=probabilities));pool=self.unknown_pools[(family,face,view)]
            rows.append(pool[int(self.rng.integers(len(pool)))])
            self.unknown_counts[family]+=1;self.face_counts[(family,*face)]+=1;self.view_counts[('unknown',view)]+=1
        self.position+=BATCH_SIZE//2;self.total_counts.update(known=32,unknown=32)
        self.rng.shuffle(rows)
        return rows

    def report(self):
        return {'sampled_rows':dict(self.total_counts),'known_class_rows':dict(self.known_counts),
            'unknown_source_family_rows':dict(self.unknown_counts),
            'unknown_source_face_rows':[{'family':f,'font_file_sha256':digest,'ttc_index':ttc,'rows':count}
                                        for (f,digest,ttc),count in sorted(self.face_counts.items())],
            'view_rows':{kind:{view:self.view_counts[(kind,view)] for view in RECIPES} for kind in ('known','unknown')}}


def unknown_uniform_ce(logits,targets,unknown_index):
    """Encourage flat known-class logits on unknown rows; keep unknown CE separate."""
    import torch
    require(logits.ndim==2 and logits.shape[1]==10 and targets.shape==(len(logits),)
            and type(unknown_index) is int and 0<=unknown_index<10,'invalid uniform-CE input shape')
    mask=targets==unknown_index
    if not bool(mask.any()):return logits.sum()*0
    known=logits[mask][:,[i for i in range(10) if i!=unknown_index]]
    return (torch.logsumexp(known,dim=1)-known.mean(dim=1)).mean()


def validate_initializer(base,checkpoint,families,selection,protocol,selection_sha,protocol_sha,data_sha,actual_state_sha):
    """Require all ten rows and every encoder/size tensor from the declared run."""
    validate_initialization(base,checkpoint)
    require(checkpoint['families']==families and len(families)==10 and families.count(UNKNOWN)==1,
            'initializer must contain the same ordered ten-class family registry')
    require(selection.get('schema')=='flux-glyph-android-training-selection-v1' and selection.get('families')==families
            and selection.get('test_read') is False and selection.get('data_manifest_sha256')==data_sha,
            'initializer selection does not describe this Android training dataset')
    require(checkpoint.get('selection_sha256')==selection_sha and selection.get('training_protocol_sha256')==protocol_sha
            and selection.get('state_after_sha256')==actual_state_sha,'initializer checkpoint/selection/state binding differs')
    steps=selection.get('optimizer_steps_executed')
    require(type(steps) is int and steps>=1000 and steps%500==0
            and selection.get('state_before_sha256')!=actual_state_sha,'initializer has no recorded optimizer changes')
    require(protocol.get('schema')=='flux-glyph-android-training-protocol-v1' and protocol.get('families')==families
            and protocol.get('steps')==steps and protocol.get('test_read') is False
            and protocol.get('bindings')==selection.get('bindings')
            and protocol.get('initial_state_sha256')==selection.get('state_before_sha256'),
            'initializer training protocol differs')


def bind_inputs(data,checkpoint,initializer_selection,initializer_protocol,source_bindings):
    bindings={str(Path(path).resolve()):digest for path,digest in source_bindings.items()}
    paths=[Path(__file__),ROOT/'training/train_android_regions.py',ROOT/'training/prepare_android_regions.py',
        ROOT/'training/region_network.py',ROOT/'training/network.py',ROOT/'training/train_regions.py',
        ROOT/'src/flux_glyph/android_font.py',ROOT/'src/flux_glyph/region_font.py',checkpoint,initializer_selection,initializer_protocol,
        data/'MANIFEST.json',data/'train/MANIFEST.json',data/'calibration/MANIFEST.json']
    for path in paths:
        path=Path(path).resolve();digest=sha(path)
        require(str(path) not in bindings or bindings[str(path)]==digest,'initializer/current source binding conflict')
        bindings[str(path)]=digest
    return bindings


def train(args):
    import torch
    from region_network import RegionFontClassifier
    require(not args.output.exists(),'balanced training output must be new')
    require(type(args.steps) is int and args.steps>=1000 and args.steps%500==0,
            'training must execute at least 1000 steps in multiples of 500')
    require(type(args.learning_rate) in (int,float) and math.isfinite(args.learning_rate)
            and 1e-5<=args.learning_rate<=.001,'invalid fine-tuning learning rate')
    args.data=Path(args.data).resolve();args.checkpoint=Path(args.checkpoint).resolve()
    training=load_split(args.data,'train');cal=load_split(args.data,'calibration')
    require(training['families']==cal['families'] and training['manifest_sha256']==cal['manifest_sha256'],
            'train/CAL sources differ')
    families=training['families'];sampler=BalancedSampler(training['rows'],families,args.seed)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    model=RegionFontClassifier(len(families));checkpoint_sha=sha(args.checkpoint)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    initializer_selection=args.checkpoint.parent/'SELECTION.json';initializer_protocol=args.checkpoint.parent/'TRAINING_FREEZE.json'
    prior=json.loads(initializer_selection.read_text());prior_protocol=json.loads(initializer_protocol.read_text())
    initial_state=state_sha(checkpoint['state_dict'])
    validate_initializer(model.state_dict(),checkpoint,families,prior,prior_protocol,
                         sha(initializer_selection),sha(initializer_protocol),training['manifest_sha256'],initial_state)
    require(all(bool(torch.isfinite(value).all()) for value in checkpoint['state_dict'].values()),
            'initializer contains nonfinite parameters')
    require(all(Path(path).is_file() and sha(path)==digest for path,digest in prior['bindings'].items()),
            'initializer bound source changed')
    model.load_state_dict(checkpoint['state_dict'],strict=True);model.requires_grad_(True)
    before=state_sha(model.state_dict());require(before==initial_state,'full ten-class initialization changed')
    source_bindings=dict(prior['bindings'])
    for path,digest in training['manifest']['bindings'].items():
        require(path not in source_bindings or source_bindings[path]==digest,'initializer data source binding conflict')
        source_bindings[path]=digest
    bindings=bind_inputs(args.data,args.checkpoint,initializer_selection,initializer_protocol,source_bindings)
    require(bindings[str(args.checkpoint)]==checkpoint_sha,'initializer changed while loading')
    model.to(args.device);args.output.mkdir(parents=True)
    inheritance={'checkpoint':str(args.checkpoint),'checkpoint_sha256':checkpoint_sha,
        'selection_sha256':sha(initializer_selection),'protocol_sha256':sha(initializer_protocol),
        'source_state_sha256':initial_state,'families':families,'inherited_family_head_rows':list(range(10)),
        'encoder_and_size_head_inherited_in_full':True,'all_parameters_trainable':True,
        'source_optimizer_steps_executed':prior['optimizer_steps_executed'],
        'source_selected_step':prior['selected']['step'],'additional_optimizer_steps_planned':args.steps,
        'source_calibration_passed':prior.get('calibration_passed') is True,
        'newly_initialized_parameters':[],'reset_or_remapped_family_rows':False}
    protocol={'schema':'flux-glyph-android-training-protocol-v1','policy':POLICY,'bindings':bindings,'families':families,
        'steps':args.steps,'batch_size':BATCH_SIZE,'seed':args.seed,'learning_rate':args.learning_rate,'device':args.device,
        'initial_state_sha256':before,'test_read':False,'training_inputs':['image_tiles'],
        'initialization':'Complete ten-class Android checkpoint: exact encoder, size head and all family-head rows inherited; every parameter fine-tuned.',
        'initializer_evidence':inheritance,'sampling':SAMPLING,'objective':OBJECTIVE,
        'optimizer':'AdamW weight_decay=.0002; cosine decay to 1e-5; gradient norm clip 5; no teacher outputs.'}
    dump(args.output/'TRAINING_FREEZE.json',protocol)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.0002)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=1e-5)
    best=None;history=[];started=time.monotonic()
    for step in range(1,args.steps+1):
        rows=sampler.batch();images=[];targets=[];sizes=[]
        for row in rows:
            tile=row['tile_start']+int(sampler.rng.integers(row['tile_count']))
            images.append(training['tiles'][tile]);targets.append(row['target']);sizes.append(row['log_em_ratio'])
        model.train();logits,ratio=model(torch.from_numpy(np.stack(images)).to(args.device))
        target_tensor=torch.tensor(targets,device=args.device);size_tensor=torch.tensor(sizes,device=args.device,dtype=torch.float32)
        ce=torch.nn.functional.cross_entropy(logits,target_tensor,label_smoothing=.03)
        size_loss=torch.nn.functional.smooth_l1_loss(ratio,size_tensor)
        uniform=unknown_uniform_ce(logits,target_tensor,sampler.unknown)
        loss=ce+.2*size_loss+.5*uniform
        require(bool(torch.isfinite(loss)),'nonfinite balanced training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step();scheduler.step()
        if step%100==0:
            print(json.dumps({'step':step,'loss':float(loss.detach().cpu()),'cross_entropy':float(ce.detach().cpu()),
                'size_loss':float(size_loss.detach().cpu()),'unknown_uniform_ce':float(uniform.detach().cpu()),
                'seconds':round(time.monotonic()-started,1)}),flush=True)
        if step%1000==0 or step==args.steps:
            logits,ratios=infer(model,cal,args.device);selected,grid=select_grid(logits,ratios,cal,step)
            history.append(selected[1]);dump(args.output/f'CAL_GRID_{step:05d}.json',grid)
            if best is None or selected[0]>best[0]:
                best=(selected[0],copy.deepcopy(selected[1]),{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                      copy.deepcopy(selected[2]),logits.copy(),ratios.copy())
            dump(args.output/'CAL_PROGRESS.json',history);dump(args.output/'SAMPLING_PROGRESS.json',sampler.report())
            print(json.dumps({'calibration_step':step,'selection':selected[1]},ensure_ascii=False),flush=True)
    model.cpu().load_state_dict(best[2]);after=state_sha(model.state_dict());require(before!=after,'no optimizer changes')
    require(all(Path(path).is_file() and sha(path)==digest for path,digest in bindings.items()),
            'balanced training input changed before selection')
    dump(args.output/'CALIBRATION_DECISIONS.json',{'records':best[3],'families':families})
    np.savez(args.output/'CALIBRATION_OUTPUTS.npz',logits=best[4],log_em_ratio=best[5])
    dump(args.output/'SAMPLING.json',sampler.report())
    selection={'schema':'flux-glyph-android-training-selection-v1','policy':POLICY,'selected':best[1],'families':families,
        'passed':best[1]['metrics']['passed'],'calibration_passed':best[1]['metrics']['passed'],'test_read':False,
        'data_manifest_sha256':training['manifest_sha256'],'bindings':bindings,
        'training_protocol_sha256':sha(args.output/'TRAINING_FREEZE.json'),'optimizer_steps_executed':args.steps,
        'training_device':args.device,'state_before_sha256':before,'state_after_sha256':after,
        'calibration_decisions_sha256':sha(args.output/'CALIBRATION_DECISIONS.json'),
        'calibration_outputs_sha256':sha(args.output/'CALIBRATION_OUTPUTS.npz'),'history':history,
        'initializer_evidence':inheritance,'sampling_sha256':sha(args.output/'SAMPLING.json')}
    dump(args.output/'SELECTION.json',selection)
    torch.save({'state_dict':model.state_dict(),'families':families,'selection_sha256':sha(args.output/'SELECTION.json')},
               args.output/'model.pth')
    print(json.dumps({'passed':selection['passed'],'selected_step':best[1]['step'],'output':str(args.output)}),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/android-font-v1/run-v2/model.pth')
    p.add_argument('--steps',type=int,default=6000);p.add_argument('--seed',type=int,default=2026091303)
    p.add_argument('--learning-rate',type=float,default=.0001);p.add_argument('--device',choices=('mps','cpu'),default='mps')
    return p


if __name__=='__main__':train(parser().parse_args())
