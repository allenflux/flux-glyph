#!/usr/bin/env python3
"""Native core-font supervised weight repair with a training-only R21 teacher."""
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
from train_unified_retention import (BATCH_SIZE,EVAL_EVERY,FIXED_RUNTIME,SAMPLING,OBJECTIVE as BASE_OBJECTIVE,
    ARCHITECTURE,POLICY,UNKNOWN,RetentionSampler,initialize,parameter_groups,read_plan,evaluate_outputs,
    retention_rank,registry,sha,dump,require,state_sha,verify_bindings,load_split,infer)

STEPS=1500
LEARNING_RATE=1e-5
MINIMUM_LEARNING_RATE=1e-6
STUDENT_CHECKPOINT_SHA='ab8fa2476732cbbfdaba3fba4ff424db5c9c42a1eb64f4221696fe9862594821'
STUDENT_SELECTION_SHA='87523193db1b87b3306d073b4ec119ecb1c0fe34ef80b663c110b96bc04fa1cf'
TEACHER_CHECKPOINT_SHA='960b210091f7c8b13a7283e6bd7f4a7e48e1bd7a1bf3a8d091d0fe11b4bf367e'
CORE_FAMILIES=('PingFang','SF Pro','Helvetica')
NATIVE_CORE_WEIGHTING={'families':list(CORE_FAMILIES),'split':'train','domain':'ios','view':'native',
    'weight':2.0,'other_weight':1.0,'denominator':BATCH_SIZE,'size_loss_weighted':False,'inference_rule':False}
OBJECTIVE_VARIANT='r21_native_core_weighted_teacher_kl'
OBJECTIVE={**BASE_OBJECTIVE,'teacher_outputs':True,'distillation':True,'second_encoder':True,
    'training_only_teacher':True,'deployed_model_count':1,'teacher_kl_weight':1.,'teacher_temperature':1.,
    'teacher_mask':'TRAIN true target is outside PingFang/SF Pro/Helvetica and teacher argmax equals true target',
    'teacher_kl_reduction':'Mean KL(teacher || student) over eligible samples; differentiable zero for an empty mask',
    'class_prior_weighting':False,'native_core_weighting':NATIVE_CORE_WEIGHTING,
    'family_loss':'Sum of per-row smoothed CE times native-core weight, divided by 96; not by sum of weights.'}


def read_json(path):
    return json.loads(path.read_text())


def frozen_teacher_logits(teacher,images):
    import torch
    require(not teacher.training and all(not p.requires_grad and p.grad is None for p in teacher.parameters()),
            'Teacher must remain frozen in eval mode')
    with torch.no_grad():
        logits,_=teacher(images)
    return logits


def native_core_weights(rows,targets,families):
    require(len(rows)==len(targets)==BATCH_SIZE,'Native core weighting requires the complete 96-row TRAIN batch')
    weights=[]
    for row,target in zip(rows,targets):
        require(type(target) is int and 0<=target<len(families) and type(row.get('target')) is int
                and row['target']==target and row.get('family')==families[target]
                and row.get('split')=='train' and row.get('native_font_verified') is True
                and row.get('domain') in ('ios','android')
                and row.get('view') in ('native','half','three_quarters_jpeg75','jpeg75'),
                'Native core weighting row identity, provenance or TRAIN partition differs')
        weights.append(2. if row['domain']=='ios' and row['view']=='native' and row['family'] in CORE_FAMILIES else 1.)
    return weights


def distilled_losses(logits,ratios,targets,sizes,teacher_logits,families,rows):
    import torch
    registry(families)
    require(len(families)==25 and all(f in families for f in CORE_FAMILIES) and logits.shape==(BATCH_SIZE,len(families))
            and teacher_logits.shape==logits.shape and ratios.shape==targets.shape==sizes.shape==(BATCH_SIZE,)
            and targets.dtype==torch.int64 and not teacher_logits.requires_grad,
            'Distillation tensor or frozen teacher contract differs')
    require(all(bool(torch.isfinite(value).all()) for value in (logits,ratios,sizes,teacher_logits))
            and bool(((targets>=0)&(targets<len(families))).all()),'Invalid or nonfinite distillation inputs')
    values=native_core_weights(rows,targets.detach().cpu().tolist(),families)
    weight=torch.tensor(values,dtype=logits.dtype,device=logits.device)
    core=torch.zeros_like(targets,dtype=torch.bool)
    for family in CORE_FAMILIES:core|=targets==families.index(family)
    mask=(~core) & (teacher_logits.argmax(1)==targets)
    ce=(torch.nn.functional.cross_entropy(logits,targets,label_smoothing=.03,reduction='none')*weight).sum()/BATCH_SIZE
    size_loss=torch.nn.functional.smooth_l1_loss(ratios,sizes)
    if bool(mask.any()):
        target_prob=teacher_logits[mask].softmax(1)
        kl=torch.nn.functional.kl_div(logits[mask].log_softmax(1),target_prob,reduction='none').sum(1).mean()
    else:
        kl=logits.sum()*0.
    total=ce+.2*size_loss+kl
    require(bool(torch.isfinite(total)),'Nonfinite supervised plus teacher loss')
    return total,ce,size_loss,kl,mask


class DistillationCounter:
    def __init__(self,families):
        self.families=list(families);self.steps=0;self.family=Counter();self.eligible_family=Counter()
        self.domain=Counter();self.eligible_domain=Counter();self.sources=Counter();self.eligible_sources=Counter()
        self.eligible_per_step=[];self.core_per_step=[];self.core_family=Counter();self.core_source=Counter()

    def update(self,rows,mask):
        require(len(rows)==len(mask)==BATCH_SIZE and all(type(v) is bool for v in mask), 'Invalid teacher mask accounting')
        weights=native_core_weights(rows,[r['target'] for r in rows],self.families)
        for row,eligible,weight in zip(rows,mask,weights):
            family=row['family'];domain=row['domain'];source=row.get('source_font_family') or family
            require(row.get('split')=='train' and row.get('native_font_verified') is True
                    and self.families[row['target']]==family and domain in ('ios','android')
                    and (not eligible or family not in CORE_FAMILIES),'Teacher mask contains invalid or non-TRAIN evidence')
            key=(domain,source,family);self.family[family]+=1;self.domain[domain]+=1;self.sources[key]+=1
            if eligible:self.eligible_family[family]+=1;self.eligible_domain[domain]+=1;self.eligible_sources[key]+=1
            if weight==2.:self.core_family[family]+=1;self.core_source[key]+=1
        self.steps+=1;self.eligible_per_step.append(sum(mask));self.core_per_step.append(sum(w==2. for w in weights))

    def report(self):
        return {'schema':'flux-glyph-retention-teacher-mask-counts-v1','steps':self.steps,'rows':sum(self.family.values()),
            'eligible_rows':sum(self.eligible_family.values()),'eligible_rows_per_step':self.eligible_per_step.copy(),
            'family_rows':dict(self.family),'eligible_family_rows':dict(self.eligible_family),
            'domain_rows':dict(self.domain),'eligible_domain_rows':dict(self.eligible_domain),
            'source_target_rows':[{'domain':d,'source_font_family':s,'target_family':f,'rows':n,
                'eligible_rows':self.eligible_sources[(d,s,f)]} for (d,s,f),n in sorted(self.sources.items())],
            'objective_variant':OBJECTIVE_VARIANT,'native_core_weighting':NATIVE_CORE_WEIGHTING,
            'native_core_weighted_rows':sum(self.core_family.values()),'native_core_weighted_rows_per_step':self.core_per_step.copy(),
            'native_core_weighted_family_rows':dict(self.core_family),
            'native_core_source_rows':[{'domain':d,'source_font_family':s,'target_family':f,'rows':n}
                for (d,s,f),n in sorted(self.core_source.items())],
            'mask_rule':OBJECTIVE['teacher_mask'],'training_split':'train','test_read':False,'development_holdout_read':False}


def validate_distillation_report(report,families,steps):
    require(report.get('schema')=='flux-glyph-retention-teacher-mask-counts-v1' and report['steps']==steps
            and report['rows']==steps*BATCH_SIZE and report.get('training_split')=='train'
            and report.get('mask_rule')==OBJECTIVE['teacher_mask']
            and report.get('test_read') is False and report.get('development_holdout_read') is False,
            'Teacher mask totals or split differ')
    counts=report['eligible_rows_per_step']
    require(len(counts)==steps and all(type(v) is int and 0<=v<=BATCH_SIZE for v in counts)
            and sum(counts)==report['eligible_rows'],'Invalid teacher per-step accounting')
    for all_key,eligible_key,allowed in (('family_rows','eligible_family_rows',set(families)),
        ('domain_rows','eligible_domain_rows',{'ios','android'})):
        total=report[all_key];eligible=report[eligible_key]
        require(set(total)<=allowed and set(eligible)<=set(total)
                and all(type(v) is int and v>=0 for v in [*total.values(),*eligible.values()])
                and sum(total.values())==report['rows'] and sum(eligible.values())==report['eligible_rows']
                and all(eligible.get(k,0)<=v for k,v in total.items()),'Teacher population totals differ')
    require(all(report['eligible_family_rows'].get(f,0)==0 for f in CORE_FAMILIES),'The three core families must not receive teacher targets')
    source_rows=report['source_target_rows'];seen=set();totals=Counter();eligible_totals=Counter();domains=Counter();eligible_domains=Counter()
    for row in source_rows:
        key=(row['domain'],row['source_font_family'],row['target_family'])
        require(key not in seen and key[0] in ('ios','android') and key[2] in families
                and isinstance(key[1],str) and key[1] and type(row['rows']) is int and type(row['eligible_rows']) is int
                and 0<=row['eligible_rows']<=row['rows'],'Invalid teacher source/target accounting')
        seen.add(key);totals[key[2]]+=row['rows'];eligible_totals[key[2]]+=row['eligible_rows']
        domains[key[0]]+=row['rows'];eligible_domains[key[0]]+=row['eligible_rows']
    require(dict(totals)==report['family_rows'] and dict(domains)==report['domain_rows']
            and all(eligible_totals[f]==report['eligible_family_rows'].get(f,0) for f in families)
            and all(eligible_domains[d]==report['eligible_domain_rows'].get(d,0) for d in ('ios','android')),
            'Teacher source accounting does not reproduce family/domain totals')
    require(report.get('teacher_state_before_sha256')==report.get('teacher_state_after_sha256')
            and isinstance(report.get('teacher_state_before_sha256'),str) and len(report['teacher_state_before_sha256'])==64
            and report.get('teacher_optimizer_steps')==0 and report.get('training_model_count')==2
            and report.get('deployed_model_count')==1,'Teacher changed or was included in the deployed model')
    require(report.get('objective_variant')==OBJECTIVE_VARIANT and report.get('native_core_weighting')==NATIVE_CORE_WEIGHTING,
            'Native core training-weight rule differs')
    weighted=report['native_core_weighted_rows_per_step'];family_counts=report['native_core_weighted_family_rows']
    require(len(weighted)==steps and all(type(v) is int and 27<=v<=33 for v in weighted)
            and type(report['native_core_weighted_rows']) is int and sum(weighted)==report['native_core_weighted_rows']
            and set(family_counts)<=set(CORE_FAMILIES) and sum(family_counts.values())==sum(weighted)
            and all(type(n) is int and 0<=n<=report['family_rows'].get(f,0) for f,n in family_counts.items()),
            'Native core weighted sample counts differ from the frozen sampler')
    weighted_sources=Counter();seen=set()
    original={(r['domain'],r['source_font_family'],r['target_family']):r['rows'] for r in source_rows}
    for row in report['native_core_source_rows']:
        key=(row['domain'],row['source_font_family'],row['target_family'])
        require(key not in seen and key[0]=='ios' and key[2] in CORE_FAMILIES and key in original
                and type(row['rows']) is int and 0<row['rows']<=original[key], 'Invalid native core source accounting')
        seen.add(key);weighted_sources[key[2]]+=row['rows']
    require(dict(weighted_sources)==family_counts,'Native core weighted source/family totals differ')
    return report


def train(args):
    import torch
    from region_network import RegionFontClassifier
    from flux_glyph.unified_font import unified_metadata
    args.data=args.data.resolve();args.teacher=args.teacher.resolve();args.checkpoint=args.checkpoint.resolve();args.plan=args.plan.resolve();args.output=args.output.resolve()
    require(not args.output.exists() and args.steps==STEPS,'retention run needs a new output and exactly 1500 steps')
    require(sha(args.teacher)==TEACHER_CHECKPOINT_SHA and sha(args.checkpoint)==STUDENT_CHECKPOINT_SHA, 'Distillation initializer or teacher is not the predeclared checkpoint')
    plan=read_plan(args.plan,args.data,args.teacher)
    training=load_split(args.data,'train');cal=load_split(args.data,'calibration')
    require(training['families']==cal['families'] and training['manifest_sha256']==cal['manifest_sha256'],
            'retention TRAIN/CAL sources differ')
    families=training['families'];registry(families)
    prior_dir=args.teacher.parent;prior=json.loads((prior_dir/'SELECTION.json').read_text())
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
            and parity.get('passed') is True and parity['checkpoint_sha256']==sha(args.teacher)
            and parity['selection_sha256']==sha(prior_dir/'SELECTION.json'),
            'retention parent is not the frozen unified R21 classifier and fixed runtime')
    require(parent_meta.get('training',{}).get('selection_sha256')==sha(prior_dir/'SELECTION.json')
            and parent_meta['training'].get('checkpoint_sha256')==sha(args.teacher), 'parent model provenance differs')
    checkpoint=torch.load(args.teacher,map_location='cpu',weights_only=True)
    require(checkpoint.get('selection_sha256')==sha(prior_dir/'SELECTION.json')
            and state_sha(checkpoint['state_dict'])==prior['state_after_sha256'],'retention parent checkpoint identity differs')
    source_dir=args.checkpoint.parent;source_selection=read_json(source_dir/'SELECTION.json')
    require(sha(source_dir/'SELECTION.json')==STUDENT_SELECTION_SHA
            and source_selection.get('schema')=='flux-glyph-unified-retention-selection-v1'
            and source_selection.get('promotion_allowed') is False
            and source_selection.get('test_read') is False and source_selection.get('development_holdout_read') is False
            and source_selection['families']==families and source_selection['retention_plan']['sha256']==sha(args.plan)
            and source_selection['data_manifest_sha256']==training['manifest_sha256']
            and sha(source_dir/'TRAINING_FREEZE.json')==source_selection['training_protocol_sha256'],
            'Student initializer is not the frozen first focus run')
    verify_bindings(source_selection['bindings'])
    student_checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    require(student_checkpoint.get('selection_sha256')==sha(source_dir/'SELECTION.json')
            and state_sha(student_checkpoint['state_dict'])==source_selection['state_after_sha256'],
            'First-focus student checkpoint identity differs')
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    model=RegionFontClassifier(len(families));inheritance=initialize(model,student_checkpoint,families)
    teacher=RegionFontClassifier(len(families));initialize(teacher,checkpoint,families)
    teacher.requires_grad_(False);teacher.eval();teacher_state_before=state_sha(teacher.state_dict())
    student_initializer={'checkpoint':{'path':str(args.checkpoint),'sha256':sha(args.checkpoint)},
        'selection':{'path':str(source_dir/'SELECTION.json'),'sha256':sha(source_dir/'SELECTION.json')},
        'state_sha256':source_selection['state_after_sha256'],'source_selected_step':source_selection['selected']['step'],
        'source_optimizer_steps_executed':source_selection['optimizer_steps_executed']}
    teacher_identity={'checkpoint':{'path':str(args.teacher),'sha256':sha(args.teacher)},
        'selection':{'path':str(prior_dir/'SELECTION.json'),'sha256':sha(prior_dir/'SELECTION.json')},
        'state_sha256':teacher_state_before,'temperature':1.0,'frozen':True,'optimizer_steps':0}
    distillation=DistillationCounter(families)
    before=state_sha(model.state_dict());initial_groups=parameter_groups(model.state_dict())
    sampler=RetentionSampler(training['rows'],families,args.seed,plan['train_pools'])
    bindings=dict(training['manifest']['bindings'])
    for entry in [*plan['bindings'].values(),*plan.get('indices',{}).values()]:
        path=str((ROOT/entry['path']).resolve());digest=entry['sha256']
        require(path not in bindings or bindings[path]==digest,'retention plan/data source conflicts');bindings[path]=digest
    for mapping in (prior['bindings'],source_selection['bindings'],parity['source_bindings'],parity['calibration_bindings']):
        for path,digest in mapping.items():
            require(path not in bindings or bindings[path]==digest,'retention parent/data source conflicts');bindings[path]=digest
    paths=[Path(__file__),ROOT/'training/train_unified_retention.py',ROOT/'training/evaluate_unified_retention_core.py',
        args.checkpoint,source_dir/'SELECTION.json',source_dir/'TRAINING_FREEZE.json',source_dir/'report.json',
        prior_dir/'DEVELOPMENT_REGRESSION.json',ROOT/'training/train_unified_regions.py',ROOT/'training/prepare_unified_regions.py',
        ROOT/'training/train_android_regions.py',ROOT/'training/region_network.py',ROOT/'training/network.py',ROOT/'training/train_regions.py',
        ROOT/'src/flux_glyph/unified_font.py',ROOT/'src/flux_glyph/region_font.py',args.teacher,args.plan,parent_meta_path,parent_onnx,
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
    inheritance.update(checkpoint=student_initializer['checkpoint'],
        selection_sha256=student_initializer['selection']['sha256'],source_selected_step=student_initializer['source_selected_step'],
        source_optimizer_steps_executed=student_initializer['source_optimizer_steps_executed'])
    protocol={'schema':'flux-glyph-unified-retention-training-protocol-v1','architecture':ARCHITECTURE,
        'policy':POLICY,'fixed_runtime':FIXED_RUNTIME,'families':families,'bindings':bindings,
        'parent_checkpoint':plan['parent_checkpoint'],'parent_selection_sha256':plan['parent_selection_sha256'],
        'parent_metadata':plan['parent_metadata'],'retention_plan':{'path':str(args.plan),'sha256':sha(args.plan)},
        'steps':STEPS,'eval_every':EVAL_EVERY,'batch_size':BATCH_SIZE,'learning_rate':LEARNING_RATE,
        'minimum_learning_rate':MINIMUM_LEARNING_RATE,'seed':args.seed,'device':args.device,'sampling':SAMPLING,'objective':OBJECTIVE,
        'objective_variant':OBJECTIVE_VARIANT,'native_core_weighting':NATIVE_CORE_WEIGHTING,'student_initializer':student_initializer,'teacher':teacher_identity,
        'training_model_count':2,'teacher_in_deployed_model':False,
        'optimizer':'All parameters; AdamW weight_decay=.0002; cosine LR; gradient norm clip 5.',
        'initial_state_sha256':before,'initial_parameter_groups_sha256':initial_groups,'initializer_evidence':inheritance,
        'baseline_bindings':baseline_bindings,'training_inputs':['image_tiles'],'test_read':False,'development_holdout_read':False,
        'model_count':1,'platform_routing':False,'score_merging':False,'runtime_gates_searched':False,
        'rank':['all_constraints_passed','negative_normalized_deficit','correct_named','negative_wrong_named','negative_calibration_nll','earliest_step']}
    dump(args.output/'TRAINING_FREEZE.json',protocol);protocol_sha=sha(args.output/'TRAINING_FREEZE.json')
    model.to(args.device);teacher.to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=.0002)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=MINIMUM_LEARNING_RATE)
    best=None;history=[];started=time.monotonic()
    for step in range(1,STEPS+1):
        rows=sampler.batch();indices=[r['tile_start']+int(sampler.rng.integers(r['tile_count'])) for r in rows]
        images=torch.from_numpy(np.array(training['tiles'][indices],copy=True)).to(args.device)
        targets=torch.tensor([r['target'] for r in rows],device=args.device)
        sizes=torch.tensor([r['log_em_ratio'] for r in rows],dtype=torch.float32,device=args.device)
        model.train();logits,ratio=model(images)
        teacher_logits=frozen_teacher_logits(teacher,images)
        loss,ce,size_loss,kl,mask=distilled_losses(logits,ratio,targets,sizes,teacher_logits,families,rows)
        distillation.update(rows,mask.detach().cpu().tolist())
        require(bool(torch.isfinite(loss)),'nonfinite retention training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step();scheduler.step()
        if step%100==0:print(json.dumps({'step':step,'loss':float(loss.detach().cpu()),'family_loss':float(ce.detach().cpu()),
            'size_loss':float(size_loss.detach().cpu()),'teacher_kl':float(kl.detach().cpu()),'teacher_eligible':int(mask.sum().detach().cpu()),'seconds':round(time.monotonic()-started,1)}),flush=True)
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
            dump(args.output/'DISTILLATION_PROGRESS.json',distillation.report())
            print(json.dumps({'calibration_step':step,'promotion_allowed':record['promotion_allowed'],
                'deficit':record['retention_deficit'],'correct_named':record['metrics']['correct_named'],
                'wrong_named':record['metrics']['wrong_named']},ensure_ascii=False),flush=True)
    teacher_state_after=state_sha(teacher.state_dict())
    require(teacher_state_after==teacher_state_before and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
            and not teacher.training,'Frozen teacher parameters changed or received gradients')
    distillation_report=distillation.report()
    distillation_report.update(teacher_state_before_sha256=teacher_state_before,teacher_state_after_sha256=teacher_state_after,
        teacher_optimizer_steps=0,training_model_count=2,deployed_model_count=1)
    validate_distillation_report(distillation_report,families,STEPS)
    dump(args.output/'DISTILLATION.json',distillation_report)
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
        'objective_variant':OBJECTIVE_VARIANT,'objective':OBJECTIVE,'native_core_weighting':NATIVE_CORE_WEIGHTING,'student_initializer':student_initializer,'teacher':teacher_identity,
        'training_model_count':2,'teacher_in_deployed_model':False,'distillation_sha256':sha(args.output/'DISTILLATION.json'),
        'core_weighting_sha256':sha(args.output/'DISTILLATION.json'),'core_weighting_evidence_file':'DISTILLATION.json',
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
        'optimizer_steps_executed':STEPS,'selected_step':selected['step'],'parameters_changed':before!=after,
        'objective_variant':OBJECTIVE_VARIANT,'teacher_optimizer_steps':0,'teacher_parameters_unchanged':teacher_state_after==teacher_state_before,
        'source_optimizer_steps_executed':source_selection['optimizer_steps_executed'],
        'student_lineage_optimizer_steps_executed':source_selection['optimizer_steps_executed']+STEPS,
        'last_checkpoint_sha256':history[-1]['artifacts']['checkpoint']['sha256'],'selected_checkpoint_sha256':sha(args.output/'model.pth'),
        'test_read':False,'development_holdout_read':False,'runtime_gates_changed':False}
    dump(args.output/'report.json',report)
    if not selected['promotion_allowed']:dump(args.output/'NO_PROMOTABLE_CHECKPOINT.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','output','plan'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/unified-font-v2/run-v1/model.pth')
    p.add_argument('--teacher',type=Path,default=ROOT/'artifacts/unified-font-v1/run-v1/model.pth')
    p.add_argument('--steps',type=int,default=STEPS);p.add_argument('--seed',type=int,default=2026091404)
    p.add_argument('--device',choices=('mps','cpu'),default='mps');return p


if __name__=='__main__':train(parser().parse_args())
