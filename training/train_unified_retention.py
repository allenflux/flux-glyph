#!/usr/bin/env python3
"""Supervised retention fine-tuning of one unified CNN, with fixed runtime gates."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import copy
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'training')]
from train_regions import sha,dump,require,state_sha
from train_unified_regions import (ARCHITECTURE,POLICY,UNKNOWN,UnifiedSampler,registry,
    region_outputs,decisions,metrics,verify_bindings)
from train_android_regions import infer
from prepare_unified_regions import load_split

STEPS=3000
EVAL_EVERY=500
BATCH_SIZE=96
LEARNING_RATE=5e-5
MINIMUM_LEARNING_RATE=5e-6
FIXED_RUNTIME={'temperature':1.,'gates':{'min_score':.7,'min_margin':.01,'min_patch_agreement':2/3},
    'max_size_relative_spread':.2}
IOS_ANCHOR_FAMILIES=['HarmonyOS Sans SC','MiSans','Noto Sans CJK SC','OPPO Sans',
    'PingFang','SF Pro','Helvetica','Alipay Number']
SAMPLING={'batch_size':96,'base_known':48,'base_unknown':16,'ios_native_pingfang':16,
    'ios_native_sfpro_helvetica':8,'ios_native_original_eight':8,
    'base_pool':'TRAIN only: known class or unknown source-family cycling; renderer and face cycling; inherited view weights.',
    'focus_pool':'TRAIN iOS native only; each requested family and source face cycle; uniform regions within each face.',
    'original_eight_families':IOS_ANCHOR_FAMILIES,'platform_labels_are_network_inputs':False}
OBJECTIVE={'family_cross_entropy_label_smoothing':.03,'size_smooth_l1_weight':.2,
    'teacher_outputs':False,'distillation':False,'second_encoder':False,'all_parameters_trainable':True}


class RetentionSampler:
    def __init__(self,rows,families,seed,train_pools):
        require(len(families)==25 and all(f in families for f in IOS_ANCHOR_FAMILIES),
                'retention experiment requires the complete 25-class joint registry')
        require(all(r.get('native_font_verified') is True for r in rows),'unverified retention TRAIN source')
        self.base=UnifiedSampler(rows,families,seed);self.rng=self.base.rng
        self.focus=defaultdict(list);self.focus_faces=defaultdict(set)
        pools={entry['id']:entry['indices'] for entry in train_pools}
        expected={'ios_native_pingfang':['PingFang'],'ios_native_latin':['SF Pro','Helvetica'],
            'ios_native_anchor_original8':IOS_ANCHOR_FAMILIES}
        require(len(pools)==len(train_pools) and set(pools)==set(expected),'retention TRAIN pools differ')
        for name,indices in pools.items():
            validate_indices(indices,len(rows))
            if name!='ios_native_anchor_original8':
                require(set(indices)=={i for i,r in enumerate(rows) if r['domain']=='ios' and r['view']=='native'
                        and r['family'] in expected[name]},'full iOS native focus population differs')
            for index in indices:
                row=rows[index]
                require(row['domain']=='ios' and row['view']=='native' and row['family'] in expected[name],
                        'retention focus pool must contain only its declared iOS native TRAIN families')
                face=(row['font_face'],str(row.get('font_file_sha256') or ''),row.get('ttc_index',0))
                self.focus[(name,row['family'],face)].append(row);self.focus_faces[(name,row['family'])].add(face)
            require(all(self.focus_faces[(name,f)] for f in expected[name]),'missing required iOS native TRAIN anchor family')
        self.focus_faces={key:sorted(value) for key,value in self.focus_faces.items()}
        self.face_positions=Counter();self.extra_positions=Counter();self.slot_counts=Counter();self.focus_counts=Counter()
        self.family_counts=Counter();self.domain_counts=Counter();self.view_counts=Counter();self.face_counts=Counter()

    def _focus_row(self,family,pool_name):
        key=(pool_name,family);faces=self.focus_faces[key]
        face=faces[self.face_positions[key]%len(faces)];self.face_positions[key]+=1
        pool=self.focus[(pool_name,family,face)];row=pool[int(self.rng.integers(len(pool)))];self.focus_counts[family]+=1
        return row

    def batch(self):
        rows=[]
        for _ in range(48):
            target=self.base.known_targets[self.base.known_position%len(self.base.known_targets)]
            self.base.known_position+=1;rows.append(self.base._row(('known',target)))
        for _ in range(16):
            family=self.base.unknown_families[self.base.unknown_position%len(self.base.unknown_families)]
            self.base.unknown_position+=1;rows.append(self.base._row(('unknown',family)))
        rows.extend(self._focus_row('PingFang','ios_native_pingfang') for _ in range(16))
        for _ in range(8):
            family=['SF Pro','Helvetica'][self.extra_positions['system']%2];self.extra_positions['system']+=1
            rows.append(self._focus_row(family,'ios_native_latin'))
        for _ in range(8):
            family=IOS_ANCHOR_FAMILIES[self.extra_positions['original_eight']%8];self.extra_positions['original_eight']+=1
            rows.append(self._focus_row(family,'ios_native_anchor_original8'))
        self.slot_counts.update(base_known=48,base_unknown=16,ios_native_pingfang=16,ios_native_sfpro_helvetica=8,ios_native_original_eight=8)
        for row in rows:
            self.family_counts[row['family']]+=1;self.domain_counts[row['domain']]+=1;self.view_counts[row['view']]+=1
            self.face_counts[(row['family'],row['domain'],row['font_face'],str(row.get('font_file_sha256') or ''),row.get('ttc_index',0))]+=1
        self.rng.shuffle(rows);return rows

    def report(self):
        return {'slots':dict(self.slot_counts),'base':self.base.report(),'focus_family_rows':dict(self.focus_counts),
            'family_rows':dict(self.family_counts),'domain_rows':dict(self.domain_counts),'view_rows':dict(self.view_counts),
            'face_rows':[{'family':f,'domain':d,'font_face':face,'font_file_sha256':digest,'ttc_index':ttc,'rows':count}
                for (f,d,face,digest,ttc),count in sorted(self.face_counts.items())]}


def initialize(model,checkpoint,families):
    """Inherit every parameter and all 25 output rows exactly, then train all groups."""
    import torch
    expected=model.state_dict();state=checkpoint.get('state_dict',{})
    require(checkpoint.get('families')==families and checkpoint.get('architecture')==ARCHITECTURE
            and len(families)==25 and set(state)==set(expected),'retention initializer architecture or complete class order differs')
    for name,value in state.items():
        require(isinstance(value,torch.Tensor) and value.shape==expected[name].shape and value.dtype==expected[name].dtype
                and bool(torch.isfinite(value).all()),'invalid retention initializer parameters')
    model.load_state_dict(state,strict=True);model.requires_grad_(True)
    require(state_sha(model.state_dict())==state_sha(state),'retention initializer was not inherited exactly')
    return {'all_family_rows_inherited':True,'all_parameters_inherited':True,'all_parameters_trainable':True,
        'source_state_sha256':state_sha(state),'family_count':len(families),'new_random_output_rows':0}


def parameter_groups(state):
    return {name:state_sha({key:value for key,value in state.items() if key.startswith(name+'.')})
            for name in ('trunk','style','family_head','size_head')}


def validate_indices(indices,count):
    require(isinstance(indices,list) and indices and all(type(i) is int and 0<=i<count for i in indices)
            and len(set(indices))==len(indices),'invalid or duplicate retention row indices')


def read_plan(path,data,checkpoint):
    """Read frozen row identities and policy only; never open development/TEST arrays."""
    path,data,checkpoint=Path(path).resolve(),Path(data).resolve(),Path(checkpoint).resolve()
    plan=json.loads(path.read_text())
    require(plan.get('schema')=='flux-glyph-unified-retention-plan-v1' and plan.get('fixed_runtime')==FIXED_RUNTIME,
            'retention plan schema or fixed runtime differs')
    evidence=[*plan.get('bindings',{}).values(),*plan.get('indices',{}).values()]
    require(evidence and all(isinstance(entry,dict) and isinstance(entry.get('path'),str)
            and sha(ROOT/entry['path'])==entry.get('sha256') for entry in evidence),'retention plan source evidence changed')
    parent=plan.get('parent_checkpoint',{})
    require(plan.get('data_manifest_sha256')==sha(data/'MANIFEST.json')
            and plan.get('train_partition_sha256')==sha(data/'train/MANIFEST.json')
            and plan.get('calibration_partition_sha256')==sha(data/'calibration/MANIFEST.json')
            and (ROOT/parent.get('path','')).resolve()==checkpoint and parent.get('sha256')==sha(checkpoint)
            and plan.get('parent_selection_sha256')==sha(checkpoint.parent/'SELECTION.json'),
            'retention plan data or parent binding differs')
    manifests={split:json.loads((data/split/'MANIFEST.json').read_text()) for split in ('train','calibration')}
    rows={}
    for split,part in manifests.items():
        meta=(data/split/part['metadata']['path']).resolve()
        require(meta.parent==data/split and sha(meta)==part['metadata']['sha256'],'retention row metadata changed')
        expected=plan.get('train_rows_sha256' if split=='train' else 'calibration_rows_sha256')
        require(expected==sha(meta),'retention plan row order changed')
        rows[split]=json.loads(meta.read_text())
        require(all(r.get('split')==split for r in rows[split]),'retention plan contains another partition')
    families=json.loads((data/'MANIFEST.json').read_text())['families'];registry(families)
    require(plan.get('families')==families and plan.get('test_read') is False
            and plan.get('development_holdout_read') is False,'retention plan class order or partition scope differs')
    masks=plan.get('masks');pools=plan.get('train_pools');constraints=plan.get('constraints')
    require(isinstance(masks,list) and isinstance(pools,list) and isinstance(constraints,list) and constraints,
            'retention masks, TRAIN pools or constraints missing')
    ids=[]
    for entry in masks:
        require(isinstance(entry,dict) and isinstance(entry.get('id'),str) and entry['id'] not in ('all','ios','android'),
                'invalid retention mask ID')
        ids.append(entry['id']);validate_indices(entry.get('indices'),len(rows['calibration']))
        if entry['id'].startswith('family:'):
            family=entry['id'][7:]
            require(family in families and set(entry['indices'])=={i for i,r in enumerate(rows['calibration']) if r['family']==family},
                    'retention family mask does not cover its whole CAL population')
    require(len(set(ids))==len(ids),'duplicate retention mask ID')
    # Construction checks that named focus populations are complete and native.
    RetentionSampler(rows['train'],families,0,pools)
    allowed_metrics={'correct_named','wrong_named','named_precision','known_correct_coverage','unknown_not_named_rate'}
    populations={'all','ios','android',*ids,*('family:'+family for family in families)}
    for constraint in constraints:
        require(isinstance(constraint,dict) and constraint.get('population') in populations
                and constraint.get('metric') in allowed_metrics and constraint.get('operator') in ('ge','le')
                and type(constraint.get('value')) in (int,float) and math.isfinite(constraint['value']) and constraint['value']>=0,
                'invalid retention acceptance constraint')
    metadata=plan.get('parent_metadata')
    require(isinstance(metadata,dict) and set(metadata)=={'path','sha256'} and sha(ROOT/metadata['path'])==metadata['sha256'],
            'retention parent metadata binding differs')
    return plan


def retention_rank(record):
    return (int(record['promotion_allowed']),-record['retention_deficit'],record['metrics']['correct_named'],
        -record['metrics']['wrong_named'],-record['calibration_nll'],-record['step'])


def assess_retention(details,data,plan):
    """Evaluate predeclared CAL populations without changing a prediction or gate."""
    families=data['families'];rejected=data['partition'].get('rejected',[])
    masks={entry['id']:entry['indices'] for entry in plan['masks']};populations={}
    for name in dict.fromkeys(c['population'] for c in plan['constraints']):
        if name=='all':selected=details;excluded=rejected
        elif name in ('ios','android'):
            selected=[r for r in details if r['domain']==name];excluded=[r for r in rejected if r['domain']==name]
        elif name.startswith('family:'):
            selected=[r for r in details if r['family']==name[7:]];excluded=[r for r in rejected if r['family']==name[7:]]
        else:selected=[details[index] for index in masks[name]];excluded=[]
        populations[name]=metrics(selected,families,excluded)
    checks=[];deficits=[]
    for constraint in plan['constraints']:
        value=populations[constraint['population']][constraint['metric']];bound=constraint['value']
        valid=type(value) in (int,float) and math.isfinite(value)
        passed=valid and (value>=bound if constraint['operator']=='ge' else value<=bound)
        deficit=(max(0.,bound-value if constraint['operator']=='ge' else value-bound)/max(abs(bound),1e-6)) if valid else 1e6
        checks.append({**constraint,'actual':value,'passed':bool(passed),'normalized_deficit':float(deficit)})
        deficits.append(deficit)
    return {'retention_checks':checks,'promotion_allowed':all(c['passed'] for c in checks),
        'retention_deficit':float(sum(deficits)),'retention_populations':populations}


def evaluate_outputs(logits,ratios,data,plan,step):
    outputs=region_outputs(logits,ratios,data['rows'],FIXED_RUNTIME['temperature'])
    detail=decisions(outputs,data['rows'],data['families'],FIXED_RUNTIME['gates'])
    nll=float(np.mean([-math.log(max(1e-12,p['probabilities'][r['target']])) for r,p in zip(data['rows'],outputs)]))
    require(math.isfinite(nll),'nonfinite retention CAL NLL')
    record={'step':step,'temperature':FIXED_RUNTIME['temperature'],'gates':FIXED_RUNTIME['gates'],
        'calibration_nll':nll,'metrics':metrics(detail,data['families'],data['partition'].get('rejected',[])),
        **assess_retention(detail,data,plan)}
    return record,outputs,detail


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
    paths=[Path(__file__),ROOT/'training/evaluate_unified_retention.py',
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
        'minimum_learning_rate':MINIMUM_LEARNING_RATE,'seed':args.seed,'device':args.device,'sampling':SAMPLING,'objective':OBJECTIVE,
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
        ce=torch.nn.functional.cross_entropy(logits,targets,label_smoothing=.03)
        size_loss=torch.nn.functional.smooth_l1_loss(ratio,sizes);loss=ce+.2*size_loss
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
        'last_checkpoint_sha256':history[-1]['artifacts']['checkpoint']['sha256'],'selected_checkpoint_sha256':sha(args.output/'model.pth'),
        'test_read':False,'development_holdout_read':False,'runtime_gates_changed':False}
    dump(args.output/'report.json',report)
    if not selected['promotion_allowed']:dump(args.output/'NO_PROMOTABLE_CHECKPOINT.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','output','plan'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/unified-font-v1/run-v1/model.pth')
    p.add_argument('--steps',type=int,default=STEPS);p.add_argument('--seed',type=int,default=2026091402)
    p.add_argument('--device',choices=('mps','cpu'),default='mps');return p


if __name__=='__main__':train(parser().parse_args())
