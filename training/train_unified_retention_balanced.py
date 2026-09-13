#!/usr/bin/env python3
"""Retain native focus while restoring R21 class priors in the supervised loss."""
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
from train_unified_retention import (STEPS,EVAL_EVERY,BATCH_SIZE,LEARNING_RATE,MINIMUM_LEARNING_RATE,
    FIXED_RUNTIME,IOS_ANCHOR_FAMILIES,SAMPLING,OBJECTIVE,ARCHITECTURE,POLICY,UNKNOWN,
    RetentionSampler,initialize,parameter_groups,read_plan,evaluate_outputs,retention_rank,
    registry,sha,dump,require,state_sha,verify_bindings,load_split,infer)
from train_unified_regions import KNOWN_PER_BATCH as R21_KNOWN_PER_BATCH

OBJECTIVE_VARIANT='r21_class_prior_weighted'


def class_prior_weighting(families):
    registry(families)
    known=[family for family in families if family!=UNKNOWN]
    require(len(families)==25 and all(f in known for f in IOS_ANCHOR_FAMILIES),
            'balanced retention requires the complete R21 class registry')
    require(SAMPLING['base_known']%len(known)==0 and SAMPLING['ios_native_sfpro_helvetica']%2==0
            and SAMPLING['ios_native_original_eight']%len(IOS_ANCHOR_FAMILIES)==0,
            'sample prior is not an exact fixed batch')
    counts={family:SAMPLING['base_known']//len(known) for family in known}
    counts[UNKNOWN]=SAMPLING['base_unknown']
    counts['PingFang']+=SAMPLING['ios_native_pingfang']
    for family in ('SF Pro','Helvetica'):counts[family]+=SAMPLING['ios_native_sfpro_helvetica']//2
    for family in IOS_ANCHOR_FAMILIES:counts[family]+=SAMPLING['ios_native_original_eight']//len(IOS_ANCHOR_FAMILIES)
    targets={family:R21_KNOWN_PER_BATCH/len(known) for family in known}
    targets[UNKNOWN]=BATCH_SIZE-R21_KNOWN_PER_BATCH
    weights={family:targets[family]/counts[family] for family in families}
    require(sum(counts.values())==BATCH_SIZE and all(math.isfinite(v) and v>0 for v in weights.values())
            and math.isclose(sum(counts[f]*weights[f] for f in families),BATCH_SIZE,abs_tol=1e-10),
            'invalid class-prior loss weights')
    return {'target_expected_counts':targets,'sampled_counts':counts,'class_weights':weights,
        'normalization':'Sum of per-sample weights, exactly 96 for the frozen batch composition.',
        'target_distribution':'Original R21 sampling: 77 known slots equally among 24 classes, 19 unknown slots.',
        'source_of_weights':'TRAIN sampler constants and family registry only; no CAL or holdout prediction fitting.'}


def weighted_objective(families):
    return {**OBJECTIVE,'variant':OBJECTIVE_VARIANT,'class_prior_weighting':class_prior_weighting(families),
        'family_loss':'Per-sample smoothed CE multiplied by target-prior/sample-prior weight; divide by sum of weights.',
        'size_loss':'Per-sample SmoothL1 multiplied by the same class weight; divide by sum of weights.'}


def weighted_losses(logits,ratios,targets,sizes,families,weighting):
    import torch
    require(logits.shape==(BATCH_SIZE,len(families)) and ratios.shape==targets.shape==sizes.shape==(BATCH_SIZE,)
            and targets.dtype==torch.int64,'weighted training tensor contract differs')
    indices=targets.detach().cpu().tolist()
    require(all(type(i) is int and 0<=i<len(families) for i in indices),'invalid supervised class targets')
    counts=Counter(families[i] for i in indices)
    require(dict(counts)==weighting['sampled_counts'],'actual batch differs from the fixed 96-sample class prior')
    values=[weighting['class_weights'][families[i]] for i in indices]
    require(all(math.isfinite(v) and v>0 for v in values) and math.isclose(sum(values),BATCH_SIZE,abs_tol=1e-8),
            'actual batch loss weights are nonfinite or do not sum to 96')
    weight=torch.tensor(values,dtype=logits.dtype,device=logits.device)
    denominator=weight.sum()
    require(bool(torch.isfinite(denominator)) and bool(torch.abs(denominator-BATCH_SIZE)<=1e-4),
            'device loss-weight sum differs')
    family=torch.nn.functional.cross_entropy(logits,targets,label_smoothing=.03,reduction='none')
    size=torch.nn.functional.smooth_l1_loss(ratios,sizes,reduction='none')
    family_loss=(family*weight).sum()/denominator;size_loss=(size*weight).sum()/denominator
    total=family_loss+.2*size_loss
    require(bool(torch.isfinite(total)),'nonfinite class-prior weighted loss')
    return total,family_loss,size_loss


def train(args):
    import torch
    from region_network import RegionFontClassifier
    from flux_glyph.unified_font import unified_metadata
    args.data=args.data.resolve();args.checkpoint=args.checkpoint.resolve();args.plan=args.plan.resolve();args.output=args.output.resolve()
    require(not args.output.exists() and args.steps==STEPS,'retention run needs a new output and exactly 3000 steps')
    plan=read_plan(args.plan,args.data,args.checkpoint)
    training=load_split(args.data,'train');cal=load_split(args.data,'calibration')
    require(training['families']==cal['families'] and training['manifest_sha256']==cal['manifest_sha256'],
            'retention TRAIN/CAL sources differ')
    families=training['families'];registry(families)
    weighting=class_prior_weighting(families);objective=weighted_objective(families)
    prior_dir=args.checkpoint.parent;prior=json.loads((prior_dir/'SELECTION.json').read_text())
    parent_meta_path=(ROOT/plan['parent_metadata']['path']).resolve();parent_meta=json.loads(parent_meta_path.read_text())
    validated=unified_metadata(parent_meta);parent_onnx=parent_meta_path.parent/validated['model']['path']
    parity=json.loads((prior_dir/'PARITY.json').read_text())
    require(prior.get('schema')=='flux-glyph-unified-training-selection-v1' and prior['families']==families
            and prior.get('test_read') is False and prior.get('development_holdout_read') is False
            and prior['data_manifest_sha256']==sha(args.data/'MANIFEST.json')
            and prior['selected']['temperature']==FIXED_RUNTIME['temperature'] and prior['selected']['gates']==FIXED_RUNTIME['gates']
            and validated['families']==families and validated['temperature']==FIXED_RUNTIME['temperature']
            and validated['gates']==FIXED_RUNTIME['gates'] and validated['max_size_relative_spread']==FIXED_RUNTIME['max_size_relative_spread']
            and sha(parent_onnx)==validated['model']['sha256']==parity['model_sha256']
            and parity.get('passed') is True and parity['checkpoint_sha256']==sha(args.checkpoint)
            and parity['selection_sha256']==sha(prior_dir/'SELECTION.json'),
            'retention parent is not the frozen unified R21 classifier and fixed runtime')
    require(parent_meta.get('training',{}).get('selection_sha256')==sha(prior_dir/'SELECTION.json')
            and parent_meta['training'].get('checkpoint_sha256')==sha(args.checkpoint), 'parent model provenance differs')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    require(checkpoint.get('selection_sha256')==sha(prior_dir/'SELECTION.json')
            and state_sha(checkpoint['state_dict'])==prior['state_after_sha256'],'retention parent checkpoint identity differs')
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    model=RegionFontClassifier(len(families));inheritance=initialize(model,checkpoint,families)
    before=state_sha(model.state_dict());initial_groups=parameter_groups(model.state_dict())
    sampler=RetentionSampler(training['rows'],families,args.seed,plan['train_pools'])
    bindings=dict(training['manifest']['bindings'])
    for entry in [*plan['bindings'].values(),*plan.get('indices',{}).values()]:
        path=str((ROOT/entry['path']).resolve());digest=entry['sha256']
        require(path not in bindings or bindings[path]==digest,'retention plan/data source conflicts');bindings[path]=digest
    for mapping in (prior['bindings'],parity['source_bindings'],parity['calibration_bindings']):
        for path,digest in mapping.items():
            require(path not in bindings or bindings[path]==digest,'retention parent/data source conflicts');bindings[path]=digest
    paths=[Path(__file__),ROOT/'training/train_unified_retention.py',ROOT/'training/evaluate_unified_retention.py',
        prior_dir/'DEVELOPMENT_REGRESSION.json',ROOT/'training/train_unified_regions.py',ROOT/'training/prepare_unified_regions.py',
        ROOT/'training/train_android_regions.py',ROOT/'training/region_network.py',ROOT/'training/network.py',ROOT/'training/train_regions.py',
        ROOT/'src/flux_glyph/unified_font.py',ROOT/'src/flux_glyph/region_font.py',args.checkpoint,args.plan,parent_meta_path,parent_onnx,
        prior_dir/'SELECTION.json',prior_dir/'PARITY.json',prior_dir/'CALIBRATION_OUTPUTS.npz',prior_dir/'CALIBRATION_DECISIONS.json']
    for split in ('train','calibration'):
        folder=args.data/split;part=json.loads((folder/'MANIFEST.json').read_text());paths.append(folder/'MANIFEST.json')
        for key in ('array','metadata'):
            path=(folder/part[key]['path']).resolve();digest=part[key]['sha256']
            require(str(path) not in bindings or bindings[str(path)]==digest,'retention dataset source conflicts');bindings[str(path)]=digest
    paths.append(args.data/'MANIFEST.json')
    for path in paths:
        path=path.resolve();digest=sha(path)
        require(str(path) not in bindings or bindings[str(path)]==digest,'retention source changed before freeze');bindings[str(path)]=digest
    verify_bindings(bindings)
    require(sha(prior_dir/'CALIBRATION_OUTPUTS.npz')==prior['calibration_outputs_sha256']
            and sha(prior_dir/'CALIBRATION_DECISIONS.json')==prior['calibration_decisions_sha256'], 'parent CAL cache changed')
    with np.load(prior_dir/'CALIBRATION_OUTPUTS.npz',allow_pickle=False) as cache:
        require(set(cache.files)=={'logits','log_em_ratio'},'parent CAL cache schema differs')
        baseline,baseline_outputs,_=evaluate_outputs(cache['logits'],cache['log_em_ratio'],cal,plan,0)
    cached=json.loads((prior_dir/'CALIBRATION_DECISIONS.json').read_text())
    require(cached=={'families':families,'records':baseline_outputs} and baseline['metrics']==prior['selected']['metrics'],
            'parent cached CAL does not reproduce the frozen baseline')
    args.output.mkdir(parents=True)
    dump(args.output/'BASELINE.json',baseline)
    (args.output/'BASELINE_CALIBRATION_OUTPUTS.npz').write_bytes((prior_dir/'CALIBRATION_OUTPUTS.npz').read_bytes())
    dump(args.output/'BASELINE_CALIBRATION_DECISIONS.json',cached)
    baseline_bindings={name:sha(args.output/name) for name in ('BASELINE.json','BASELINE_CALIBRATION_OUTPUTS.npz','BASELINE_CALIBRATION_DECISIONS.json')}
    inheritance.update(checkpoint={'path':str(args.checkpoint),'sha256':sha(args.checkpoint)},
        selection_sha256=sha(prior_dir/'SELECTION.json'),source_selected_step=prior['selected']['step'],
        source_optimizer_steps_executed=prior['optimizer_steps_executed'])
    protocol={'schema':'flux-glyph-unified-retention-training-protocol-v1','architecture':ARCHITECTURE,
        'policy':POLICY,'fixed_runtime':FIXED_RUNTIME,'families':families,'bindings':bindings,
        'parent_checkpoint':plan['parent_checkpoint'],'parent_selection_sha256':plan['parent_selection_sha256'],
        'parent_metadata':plan['parent_metadata'],'retention_plan':{'path':str(args.plan),'sha256':sha(args.plan)},
        'steps':STEPS,'eval_every':EVAL_EVERY,'batch_size':BATCH_SIZE,'learning_rate':LEARNING_RATE,
        'minimum_learning_rate':MINIMUM_LEARNING_RATE,'seed':args.seed,'device':args.device,'sampling':SAMPLING,'objective':objective,
        'objective_variant':OBJECTIVE_VARIANT,'class_prior_weighting':weighting,
        'optimizer':'All parameters; AdamW weight_decay=.0002; cosine LR; gradient norm clip 5.',
        'initial_state_sha256':before,'initial_parameter_groups_sha256':initial_groups,'initializer_evidence':inheritance,
        'baseline_bindings':baseline_bindings,'training_inputs':['image_tiles'],'test_read':False,'development_holdout_read':False,
        'model_count':1,'platform_routing':False,'score_merging':False,'runtime_gates_searched':False,
        'rank':['all_constraints_passed','negative_normalized_deficit','correct_named','negative_wrong_named','negative_calibration_nll','earliest_step']}
    dump(args.output/'TRAINING_FREEZE.json',protocol);protocol_sha=sha(args.output/'TRAINING_FREEZE.json')
    model.to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=.0002)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=MINIMUM_LEARNING_RATE)
    best=None;history=[];started=time.monotonic()
    for step in range(1,STEPS+1):
        rows=sampler.batch();indices=[r['tile_start']+int(sampler.rng.integers(r['tile_count'])) for r in rows]
        images=torch.from_numpy(np.array(training['tiles'][indices],copy=True)).to(args.device)
        targets=torch.tensor([r['target'] for r in rows],device=args.device)
        sizes=torch.tensor([r['log_em_ratio'] for r in rows],dtype=torch.float32,device=args.device)
        model.train();logits,ratio=model(images)
        loss,ce,size_loss=weighted_losses(logits,ratio,targets,sizes,families,weighting)
        require(bool(torch.isfinite(loss)),'nonfinite retention training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step();scheduler.step()
        if step%100==0:print(json.dumps({'step':step,'loss':float(loss.detach().cpu()),'family_loss':float(ce.detach().cpu()),
            'size_loss':float(size_loss.detach().cpu()),'seconds':round(time.monotonic()-started,1)}),flush=True)
        if step%EVAL_EVERY==0:
            logits,ratios=infer(model,cal,args.device);record,outputs,_=evaluate_outputs(logits,ratios,cal,plan,step)
            directory=args.output/'checkpoints'/f'step{step:05d}';directory.mkdir(parents=True)
            state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            torch.save({'state_dict':state,'families':families,'architecture':ARCHITECTURE,'step':step,
                'training_protocol_sha256':protocol_sha},directory/'model.pth')
            np.savez(directory/'CALIBRATION_OUTPUTS.npz',logits=logits,log_em_ratio=ratios)
            dump(directory/'CALIBRATION_DECISIONS.json',{'records':outputs,'families':families});dump(directory/'METRICS.json',record)
            record['artifacts']={key:{'path':str(path.relative_to(args.output)),'sha256':sha(path)} for key,path in
                [('checkpoint',directory/'model.pth'),('outputs',directory/'CALIBRATION_OUTPUTS.npz'),
                 ('decisions',directory/'CALIBRATION_DECISIONS.json'),('metrics',directory/'METRICS.json')]}
            history.append(record)
            if best is None or retention_rank(record)>retention_rank(best[0]):best=(copy.deepcopy(record),state)
            dump(args.output/'CAL_PROGRESS.json',history);dump(args.output/'SAMPLING_PROGRESS.json',sampler.report())
            print(json.dumps({'calibration_step':step,'promotion_allowed':record['promotion_allowed'],
                'deficit':record['retention_deficit'],'correct_named':record['metrics']['correct_named'],
                'wrong_named':record['metrics']['wrong_named']},ensure_ascii=False),flush=True)
    after=state_sha(best[1]);final_groups=parameter_groups(best[1])
    require(before!=after and all(final_groups[key]!=value for key,value in initial_groups.items()),'retention parameter group did not train')
    verify_bindings(bindings)
    require(sha(args.output/'TRAINING_FREEZE.json')==protocol_sha
            and all(sha(args.output/name)==digest for name,digest in baseline_bindings.items()),'retention freeze changed during training')
    for record in history:
        require(all(sha(args.output/entry['path'])==entry['sha256'] for entry in record['artifacts'].values()),'saved checkpoint evidence changed')
    selected=best[0];dump(args.output/'SAMPLING.json',sampler.report())
    for key,name in [('outputs','CALIBRATION_OUTPUTS.npz'),('decisions','CALIBRATION_DECISIONS.json')]:
        (args.output/name).write_bytes((args.output/selected['artifacts'][key]['path']).read_bytes())
    selection={'schema':'flux-glyph-unified-retention-selection-v1','architecture':ARCHITECTURE,'policy':POLICY,
        'fixed_runtime':FIXED_RUNTIME,'families':families,'selected':selected,'history':history,'bindings':bindings,
        'objective_variant':OBJECTIVE_VARIANT,'objective':objective,'class_prior_weighting':weighting,
        'passed':selected['metrics']['passed'],'calibration_passed':selected['metrics']['passed'],
        'promotion_allowed':selected['promotion_allowed'],'retention_plan':protocol['retention_plan'],
        'parent_checkpoint':protocol['parent_checkpoint'],'parent_selection_sha256':protocol['parent_selection_sha256'],
        'parent_metadata':protocol['parent_metadata'],'initializer_evidence':inheritance,'baseline_bindings':baseline_bindings,
        'data_manifest_sha256':training['manifest_sha256'],'training_protocol_sha256':protocol_sha,
        'optimizer_steps_executed':STEPS,'training_device':args.device,'state_before_sha256':before,'state_after_sha256':after,
        'initial_parameter_groups_sha256':initial_groups,'final_parameter_groups_sha256':final_groups,
        'calibration_outputs_sha256':sha(args.output/'CALIBRATION_OUTPUTS.npz'),
        'calibration_decisions_sha256':sha(args.output/'CALIBRATION_DECISIONS.json'),'sampling_sha256':sha(args.output/'SAMPLING.json'),
        'test_read':False,'development_holdout_read':False,'model_count':1,'platform_routing':False,'score_merging':False,
        'runtime_gates_searched':False}
    dump(args.output/'SELECTION.json',selection)
    torch.save({'state_dict':best[1],'families':families,'architecture':ARCHITECTURE,
        'selection_sha256':sha(args.output/'SELECTION.json')},args.output/'model.pth')
    report={'status':'PROMOTABLE_CHECKPOINT' if selected['promotion_allowed'] else 'NO_PROMOTABLE_CHECKPOINT',
        'promotion_allowed':selected['promotion_allowed'],'calibration_passed':selection['calibration_passed'],
        'objective_variant':OBJECTIVE_VARIANT,
        'optimizer_steps_executed':STEPS,'selected_step':selected['step'],'parameters_changed':before!=after,
        'last_checkpoint_sha256':history[-1]['artifacts']['checkpoint']['sha256'],'selected_checkpoint_sha256':sha(args.output/'model.pth'),
        'test_read':False,'development_holdout_read':False,'runtime_gates_changed':False}
    dump(args.output/'report.json',report)
    if not selected['promotion_allowed']:dump(args.output/'NO_PROMOTABLE_CHECKPOINT.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)



def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','output','plan'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/unified-font-v1/run-v1/model.pth')
    p.add_argument('--steps',type=int,default=STEPS);p.add_argument('--seed',type=int,default=2026091403)
    p.add_argument('--device',choices=('mps','cpu'),default='mps');return p


if __name__=='__main__':train(parser().parse_args())
