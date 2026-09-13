#!/usr/bin/env python3
"""Locally train a separate Android font/size CNN; select using CAL only."""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import itertools
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'training')]
from train_regions import sha,dump,require,state_sha
from prepare_android_regions import load_split,UNKNOWN,RECIPES
from flux_glyph.region_font import aggregate_predictions

POLICY={
    'schema':'flux-glyph-android-selection-policy-v1',
    'temperatures':[.5,1.0,2.0],
    'score_gates':[.5,.7,.85,.95,.99],
    'margin_gates':[.01,.1],
    'agreement_gates':[2/3,1.0],
    'max_size_relative_spread':.2,
    'minimum_named_precision':.98,
    'minimum_known_correct_coverage':.70,
    'minimum_per_family_correct_coverage':.50,
    'minimum_native_known_top1_accuracy':.90,
    'minimum_unseen_unknown_not_named':.80,
    'maximum_size_median_ape':.10,
    'maximum_size_p90_ape':.25,
    'minimum_size_coverage_of_correct_names':.70,
    'unknown_goal':'Avoid naming unseen font families as covered fonts; uncertainty counts as withholding a name, not as verified unknown identity.',
    'preprocessing_rejections':'Count as abstentions in coverage denominators.',
    'rank':['passed','correct_known_named','negative_wrong_named','negative_size_median_ape','negative_calibration_nll','earliest_step','earliest_grid_index'],
    'test_used_for_selection':False,
}


def region_outputs(logits,ratios,rows,temperature):
    require(isinstance(logits,np.ndarray) and isinstance(ratios,np.ndarray)
            and logits.dtype==ratios.dtype==np.float32 and logits.ndim==2 and logits.shape[1]==10
            and len(logits)>0 and ratios.shape==(len(logits),) and np.isfinite(logits).all()
            and np.isfinite(ratios).all() and np.all(np.abs(ratios)<=3),'invalid neural outputs')
    require(type(temperature) in (int,float) and math.isfinite(temperature) and .01<=temperature<=100,
            'invalid inference temperature')
    result=[];offset=0
    for row in rows:
        start,count=row['tile_start'],row['tile_count']
        require(type(start) is int and start==offset and type(count) is int and 1<=count<=8
                and start+count<=len(logits),'unaligned region tile outputs')
        value=aggregate_predictions(logits[start:start+count],ratios[start:start+count],temperature=temperature)
        result.append({'predicted':int(value['order'][0]),'score':value['score'],'margin':value['margin'],
            'agreement':value['patch_agreement'],'em_ratio':value['em_ratio'],'size_spread':value['size_relative_spread'],
            'probabilities':value['probabilities'].tolist()})
        offset+=count
    require(offset==len(logits),'unused neural tile outputs')
    return result


def validate_initialization(base,checkpoint):
    """A warm start must include the entire encoder and size head it declares."""
    require(isinstance(checkpoint,dict),'invalid initialization checkpoint')
    names=checkpoint.get('families');state=checkpoint.get('state_dict')
    require(isinstance(names,list) and len(names)>1 and all(isinstance(name,str) and name for name in names)
            and len(set(names))==len(names),'invalid initialization family registry')
    require(isinstance(state,dict) and set(state)==set(base),'incomplete or unexpected initialization parameters')
    for key,value in state.items():
        shape=(len(names),base[key].shape[1]) if key=='family_head.weight' else (len(names),) if key=='family_head.bias' else base[key].shape
        require(tuple(value.shape)==tuple(shape),'initialization parameter shape differs: '+key)


def decisions(outputs,rows,families,gates):
    unknown=families.index(UNKNOWN);result=[]
    require(len(outputs)==len(rows),'unaligned region output rows')
    for row,pred in zip(rows,outputs):
        named=(pred['predicted']!=unknown and pred['score']>=gates['min_score'] and pred['margin']>1e-8
               and pred['margin']>=gates['min_margin'] and pred['agreement']>=gates['min_patch_agreement'])
        correct=named and pred['predicted']==row['target']
        size=round(row['ink_height_px']*pred['em_ratio'],2) if named and pred['size_spread']<=POLICY['max_size_relative_spread'] else None
        result.append({'family':row['family'],'view':row['view'],'source_id':row['source_id'],'region_id':row['region_id'],
            'source_font_family':row['source_font_family'],'font_file_sha256':row['font_file_sha256'],
            'predicted_family':families[pred['predicted']],'named':bool(named),'correct_named':bool(correct),
            'wrong_named':bool(named and not correct),'top1_correct':bool(pred['predicted']==row['target']),
            'explicit_unknown':bool(pred['predicted']==unknown),'size_px':size,
            'size_ape':abs(size/row['font_size_px']-1) if correct and size is not None else None})
    return result


def metrics(details,families,rejected=()):
    # Preprocessing losses are genuine abstentions, not discarded evaluation cases.
    evaluated=list(details)+[{'family':r['family'],'view':r['view'],'named':False,'correct_named':False,'wrong_named':False,
                             'top1_correct':False,'explicit_unknown':False,'size_ape':None,'preprocessing_rejected':True} for r in rejected]
    known=[r for r in evaluated if r['family']!=UNKNOWN];unknown=[r for r in evaluated if r['family']==UNKNOWN]
    named=sum(r['named'] for r in evaluated);correct=sum(r['correct_named'] for r in known);wrong=sum(r['wrong_named'] for r in evaluated)
    native=[r for r in known if r['view']=='native'];size_errors=[r['size_ape'] for r in known if r['size_ape'] is not None]
    per_family={f:{'views':sum(r['family']==f for r in known),'correct_named':sum(r['family']==f and r['correct_named'] for r in known),
                   'wrong_named':sum(r['family']==f and r['wrong_named'] for r in known)} for f in families if f!=UNKNOWN}
    for count in per_family.values():count['correct_coverage']=count['correct_named']/max(1,count['views'])
    precision=correct/max(1,named);coverage=correct/max(1,len(known));unknown_wrong=sum(r['named'] for r in unknown)
    native_accuracy=sum(r['top1_correct'] for r in native)/max(1,len(native))
    median=float(np.median(size_errors)) if size_errors else None;p90=float(np.quantile(size_errors,.9)) if size_errors else None
    size_coverage=len(size_errors)/max(1,correct)
    checks={
        'named_precision':named>0 and precision>=POLICY['minimum_named_precision'],
        'known_correct_coverage':bool(known) and coverage>=POLICY['minimum_known_correct_coverage'],
        'per_family_coverage':all(v['views']>0 and v['correct_coverage']>=POLICY['minimum_per_family_correct_coverage'] for v in per_family.values()),
        'native_known_top1':bool(native) and native_accuracy>=POLICY['minimum_native_known_top1_accuracy'],
        'unseen_unknown_not_named':bool(unknown) and 1-unknown_wrong/len(unknown)>=POLICY['minimum_unseen_unknown_not_named'],
        'size_median':median is not None and median<=POLICY['maximum_size_median_ape'],
        'size_p90':p90 is not None and p90<=POLICY['maximum_size_p90_ape'],
        'size_coverage':size_coverage>=POLICY['minimum_size_coverage_of_correct_names'],
    }
    return {'passed':all(checks.values()),'checks':checks,'views':len(evaluated),'preprocessing_rejected':len(rejected),
        'known_views':len(known),'unknown_views':len(unknown),'named':named,'correct_named':correct,'wrong_named':wrong,
        'named_precision':precision,'known_correct_coverage':coverage,'native_known_top1_accuracy':native_accuracy,
        'unknown_wrongly_named':unknown_wrong,'unknown_not_named_rate':1-unknown_wrong/max(1,len(unknown)),
        'explicit_unknown_recall':sum(r['explicit_unknown'] for r in unknown)/max(1,len(unknown)),
        'size':{'correctly_named_with_size':len(size_errors),'coverage_of_correct_names':size_coverage,'median_ape':median,'p90_ape':p90},
        'per_family':per_family}


def select_grid(logits,ratios,data,step):
    best=None;history=[]
    combinations=list(itertools.product(POLICY['score_gates'],POLICY['margin_gates'],POLICY['agreement_gates']))
    for temperature in POLICY['temperatures']:
        outputs=region_outputs(logits,ratios,data['rows'],temperature)
        nll=float(np.mean([-math.log(max(1e-12,o['probabilities'][r['target']])) for r,o in zip(data['rows'],outputs)]))
        for score,margin,agreement in combinations:
            gates={'min_score':score,'min_margin':margin,'min_patch_agreement':agreement}
            detail=decisions(outputs,data['rows'],data['families'],gates)
            measured=metrics(detail,data['families'],data['partition']['rejected'])
            record={'step':step,'grid_index':len(history),'temperature':temperature,'gates':gates,'calibration_nll':nll,'metrics':measured}
            rank=(int(measured['passed']),measured['correct_named'],-measured['wrong_named'],
                  -(measured['size']['median_ape'] if measured['size']['median_ape'] is not None else 1e6),-nll,-step,-len(history))
            history.append(record)
            if best is None or rank>best[0]:best=(rank,record,outputs)
    return best,history


def infer(model,data,device):
    import torch
    model.eval();logits=[];ratios=[]
    with torch.inference_mode():
        for start in range(0,len(data['tiles']),256):
            x=torch.from_numpy(np.array(data['tiles'][start:start+256],copy=True)).to(device)
            family,size=model(x);logits.append(family.cpu().numpy());ratios.append(size.cpu().numpy())
    return np.concatenate(logits),np.concatenate(ratios)


def train(args):
    import torch
    from region_network import RegionFontClassifier
    require(not args.output.exists(),'training output must be new')
    require(args.steps>=1000 and args.steps%500==0,'training must execute at least 1000 steps in multiples of 500')
    train=load_split(args.data,'train');cal=load_split(args.data,'calibration')
    require(train['families']==cal['families'] and train['manifest_sha256']==cal['manifest_sha256'],'train/CAL sources differ')
    families=train['families'];torch.set_num_threads(4);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    model=RegionFontClassifier(len(families));checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    base=model.state_dict()
    validate_initialization(base,checkpoint)
    require(all(bool(torch.isfinite(value).all()) for value in checkpoint['state_dict'].values()),
            'initialization parameters are not finite')
    for name,value in checkpoint['state_dict'].items():
        if not name.startswith('family_head.'):
            require(name in base and base[name].shape==value.shape,'warm encoder or size shape mismatch');base[name]=value.clone()
    for index,family in enumerate(families):
        if family in checkpoint['families']:
            old=checkpoint['families'].index(family)
            base['family_head.weight'][index]=checkpoint['state_dict']['family_head.weight'][old]
            base['family_head.bias'][index]=checkpoint['state_dict']['family_head.bias'][old]
    model.load_state_dict(base,strict=True);before=state_sha(model.state_dict());model.to(args.device)
    bindings={str(Path(path).resolve()):digest for path,digest in train['manifest']['bindings'].items()}
    for path in [Path(__file__),ROOT/'training/prepare_android_regions.py',ROOT/'training/region_network.py',ROOT/'training/network.py',
                 ROOT/'training/train_regions.py',ROOT/'src/flux_glyph/android_font.py',ROOT/'src/flux_glyph/region_font.py',args.checkpoint,
                 args.data/'MANIFEST.json',args.data/'train/MANIFEST.json',args.data/'calibration/MANIFEST.json']:
        bindings[str(path.resolve())]=sha(path)
    args.output.mkdir(parents=True)
    protocol={'schema':'flux-glyph-android-training-protocol-v1','policy':POLICY,'bindings':bindings,'families':families,
        'steps':args.steps,'batch_size':64,'seed':args.seed,'learning_rate':args.learning_rate,'device':args.device,
        'initial_state_sha256':before,'test_read':False,'training_inputs':['image_tiles'],
        'initialization':'Prior locally trained image encoder/size head; shared Noto Sans and Roboto rows warm started, other rows initialized anew.',
        'optimizer':'AdamW cosine decay; class-balanced native 40%, remaining views 20% each; no teacher or iOS score used as label.'}
    dump(args.output/'TRAINING_FREEZE.json',protocol)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.0002)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=1e-5)
    pools=defaultdict(list)
    for row in train['rows']:pools[(row['target'],row['view'])].append(row)
    views=list(RECIPES);best=None;history=[];started=time.monotonic()
    for step in range(1,args.steps+1):
        images=[];targets=[];sizes=[]
        for i in range(64):
            target=(step*64+i)%len(families);view=str(rng.choice(views,p=[.4,.2,.2,.2]));pool=pools[(target,view)]
            require(pool,'missing class/view in training');row=pool[int(rng.integers(len(pool)))]
            images.append(train['tiles'][row['tile_start']+int(rng.integers(row['tile_count']))]);targets.append(target);sizes.append(row['log_em_ratio'])
        model.train();logits,ratio=model(torch.from_numpy(np.stack(images)).to(args.device))
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor(targets,device=args.device),label_smoothing=.03)
        loss+=.2*torch.nn.functional.smooth_l1_loss(ratio,torch.tensor(sizes,device=args.device,dtype=torch.float32))
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();scheduler.step()
        if step%100==0:print(json.dumps({'step':step,'loss':float(loss.detach().cpu()),'seconds':round(time.monotonic()-started,1)}),flush=True)
        if step%1000==0 or step==args.steps:
            logits,ratios=infer(model,cal,args.device);selected,grid=select_grid(logits,ratios,cal,step)
            history.append(selected[1]);dump(args.output/f'CAL_GRID_{step:05d}.json',grid)
            if best is None or selected[0]>best[0]:
                best=(selected[0],copy.deepcopy(selected[1]),{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},copy.deepcopy(selected[2]),logits.copy(),ratios.copy())
            dump(args.output/'CAL_PROGRESS.json',history)
            print(json.dumps({'calibration_step':step,'selection':selected[1]},ensure_ascii=False),flush=True)
    model.cpu().load_state_dict(best[2]);after=state_sha(model.state_dict());require(before!=after,'no optimizer changes')
    require(all(sha(path)==digest for path,digest in bindings.items()),'training source changed before selection')
    dump(args.output/'CALIBRATION_DECISIONS.json',{'records':best[3],'families':families})
    np.savez(args.output/'CALIBRATION_OUTPUTS.npz',logits=best[4],log_em_ratio=best[5])
    selection={'schema':'flux-glyph-android-training-selection-v1','policy':POLICY,'selected':best[1],'families':families,
        'passed':best[1]['metrics']['passed'],'calibration_passed':best[1]['metrics']['passed'],'test_read':False,
        'data_manifest_sha256':train['manifest_sha256'],'bindings':bindings,'training_protocol_sha256':sha(args.output/'TRAINING_FREEZE.json'),
        'optimizer_steps_executed':args.steps,'training_device':args.device,'state_before_sha256':before,'state_after_sha256':after,
        'calibration_decisions_sha256':sha(args.output/'CALIBRATION_DECISIONS.json'),'calibration_outputs_sha256':sha(args.output/'CALIBRATION_OUTPUTS.npz'),
        'history':history}
    dump(args.output/'SELECTION.json',selection)
    torch.save({'state_dict':model.state_dict(),'families':families,'selection_sha256':sha(args.output/'SELECTION.json')},args.output/'model.pth')
    print(json.dumps({'passed':selection['passed'],'selected_step':best[1]['step'],'output':str(args.output)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=6000);p.add_argument('--seed',type=int,default=2026091302)
    p.add_argument('--learning-rate',type=float,default=.0003);p.add_argument('--device',choices=('mps','cpu'),default='mps')
    train(p.parse_args())
