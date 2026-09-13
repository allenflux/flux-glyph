#!/usr/bin/env python3
"""Jointly train one image-only font/size CNN from native iOS and Android regions."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
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
from train_android_regions import infer,validate_initialization
from flux_glyph.region_font import aggregate_predictions

UNKNOWN='__unknown__'
ARCHITECTURE='region-cnn64x256-unified-v1'
BATCH_SIZE=96
KNOWN_PER_BATCH=77
STEPS=6000
LEARNING_RATE=.0002
VIEW_WEIGHTS={'native':.4,'half':.2,'three_quarters_jpeg75':.2,'jpeg75':.2}
POLICY={'schema':'flux-glyph-unified-selection-policy-v1',
    'temperatures':[.5,1.,2.],'score_gates':[.5,.7,.85,.95,.99],
    'margin_gates':[.01,.1],'agreement_gates':[2/3,1.],
    'experimental_priority_precision':.95,'experimental_priority_minimum_named':100,
    'minimum_named_precision':.98,'minimum_known_correct_coverage':.70,
    'minimum_per_family_correct_coverage':.50,'minimum_native_known_top1_accuracy':.90,
    'minimum_unknown_not_named':.80,'required_domains':['ios','android'],
    'maximum_size_median_ape':.10,'maximum_size_p90_ape':.25,
    'minimum_size_coverage_of_correct_names':.70,'max_size_relative_spread':.2,
    'domain_requirements':'Each domain separately meets named precision, known coverage and unknown withholding requirements.',
    'rank':['precision_at_least_95pct_and_100_names','correct_known_named','negative_wrong_named',
            'negative_calibration_nll','earliest_step','earliest_grid_index'],
    'preprocessing_rejections':'Count as abstentions in all applicable coverage denominators.',
    'calibration_scope':'Development calibration, including predeclared source-group splits of former TRAIN images that an initializer may have seen.',
    'test_used_for_selection':False,'development_holdout_used_for_selection':False}
SAMPLING={'batch_size':BATCH_SIZE,'known_per_batch':KNOWN_PER_BATCH,'unknown_per_batch':BATCH_SIZE-KNOWN_PER_BATCH,
    'known':'Equal class cycling, then equal available renderer domains, source font faces, weighted available views, uniform regions and tiles.',
    'unknown':'Equal real source-family cycling, then equal available renderer domains, source font faces, weighted available views, uniform regions and tiles.',
    'view_weights':VIEW_WEIGHTS,'missing_views':'Renormalize weights over existing views for each face; never invent another view.',
    'platform_labels_are_network_inputs':False}
OBJECTIVE={'family_cross_entropy_label_smoothing':.03,'size_smooth_l1_weight':.2,
    'teacher_outputs':False,'distillation':False,'second_encoder':False}


def registry(families):
    require(isinstance(families,list) and 2<=len(families)<=128 and len(set(families))==len(families)
            and all(isinstance(f,str) and f for f in families) and families.count(UNKNOWN)==1,
            'invalid unified family registry')
    return families.index(UNKNOWN)


class UnifiedSampler:
    """Use renderer/source metadata only to balance labelled TRAIN examples."""
    def __init__(self,rows,families,seed):
        self.unknown=registry(families);self.families=families;self.rng=np.random.default_rng(seed)
        self.pools=defaultdict(list);domains=defaultdict(set);faces=defaultdict(set);views=defaultdict(set)
        self.known_targets=[i for i in range(len(families)) if i!=self.unknown];unknown_families=set()
        for row in rows:
            target=row.get('target');family=row.get('family');domain=row.get('domain');view=row.get('view')
            require(type(target) is int and 0<=target<len(families) and family==families[target]
                    and row.get('split')=='train' and domain in POLICY['required_domains'] and view in VIEW_WEIGHTS,
                    'sampler only accepts verified TRAIN labels/domains/views')
            require(type(row.get('tile_start')) is int and row['tile_start']>=0
                    and type(row.get('tile_count')) is int and 1<=row['tile_count']<=8,
                    'invalid unified tile mapping')
            source=row.get('source_font_family');face_name=row.get('font_face')
            require(isinstance(source,str) and source and isinstance(face_name,str) and face_name,
                    'native source family or face is missing')
            group=('unknown',source) if target==self.unknown else ('known',target)
            if target==self.unknown:unknown_families.add(source)
            face=(face_name,str(row.get('font_file_sha256') or ''),row.get('ttc_index',0))
            domains[group].add(domain);faces[(group,domain)].add(face);views[(group,domain,face)].add(view)
            self.pools[(group,domain,face,view)].append(row)
        require(unknown_families and all(('known',target) in domains for target in self.known_targets),
                'unified TRAIN lacks an unknown source or known class')
        self.unknown_families=sorted(unknown_families);self.domains={k:sorted(v) for k,v in domains.items()}
        self.faces={k:sorted(v) for k,v in faces.items()};self.views={k:sorted(v) for k,v in views.items()}
        self.known_position=0;self.unknown_position=0;self.domain_positions=Counter();self.face_positions=Counter()
        self.counts=Counter();self.source_counts=Counter();self.domain_counts=Counter();self.face_counts=Counter();self.view_counts=Counter()

    def _row(self,group):
        domains=self.domains[group];domain=domains[self.domain_positions[group]%len(domains)];self.domain_positions[group]+=1
        faces=self.faces[(group,domain)];face=faces[self.face_positions[(group,domain)]%len(faces)];self.face_positions[(group,domain)]+=1
        views=self.views[(group,domain,face)];weights=np.asarray([VIEW_WEIGHTS[v] for v in views]);weights/=weights.sum()
        view=str(self.rng.choice(views,p=weights));pool=self.pools[(group,domain,face,view)]
        row=pool[int(self.rng.integers(len(pool)))];kind=group[0]
        self.counts[(kind,row['family'])]+=1;self.domain_counts[(kind,row['family'],domain)]+=1
        self.source_counts[(kind,row['source_font_family'])]+=1;self.face_counts[(kind,row['source_font_family'],domain,*face)]+=1
        self.view_counts[(kind,domain,view)]+=1
        return row

    def batch(self):
        result=[]
        for _ in range(KNOWN_PER_BATCH):
            target=self.known_targets[self.known_position%len(self.known_targets)];self.known_position+=1
            result.append(self._row(('known',target)))
        for _ in range(BATCH_SIZE-KNOWN_PER_BATCH):
            family=self.unknown_families[self.unknown_position%len(self.unknown_families)];self.unknown_position+=1
            result.append(self._row(('unknown',family)))
        self.rng.shuffle(result);return result

    def report(self):
        return {'known_rows':self.known_position,'unknown_rows':self.unknown_position,
            'class_rows':[{ 'kind':k,'family':f,'rows':n} for (k,f),n in sorted(self.counts.items())],
            'source_family_rows':[{'kind':k,'family':f,'rows':n} for (k,f),n in sorted(self.source_counts.items())],
            'domain_rows':[{'kind':k,'family':f,'domain':d,'rows':n} for (k,f,d),n in sorted(self.domain_counts.items())],
            'face_rows':[{'kind':k,'family':f,'domain':d,'font_face':face,'font_file_sha256':digest,'ttc_index':ttc,'rows':n}
                         for (k,f,d,face,digest,ttc),n in sorted(self.face_counts.items())],
            'view_rows':[{'kind':k,'domain':d,'view':v,'rows':n} for (k,d,v),n in sorted(self.view_counts.items())]}


def region_outputs(logits,ratios,rows,temperature):
    require(isinstance(logits,np.ndarray) and isinstance(ratios,np.ndarray) and logits.dtype==ratios.dtype==np.float32
            and logits.ndim==2 and 2<=logits.shape[1]<=128 and len(logits)>0 and ratios.shape==(len(logits),)
            and bool(np.isfinite(logits).all()) and bool(np.isfinite(ratios).all()) and bool((np.abs(ratios)<=3).all()),
            'invalid unified neural outputs')
    require(type(temperature) in (int,float) and math.isfinite(temperature) and .01<=temperature<=100,'invalid temperature')
    result=[];offset=0
    for row in rows:
        start,count=row['tile_start'],row['tile_count']
        require(type(start) is int and start==offset and type(count) is int and 1<=count<=8
                and start+count<=len(logits),'unaligned unified tile outputs')
        value=aggregate_predictions(logits[start:start+count],ratios[start:start+count],temperature=temperature)
        result.append({'predicted':int(value['order'][0]),'score':value['score'],'margin':value['margin'],
            'agreement':value['patch_agreement'],'em_ratio':value['em_ratio'],'size_spread':value['size_relative_spread'],
            'probabilities':value['probabilities'].tolist()});offset+=count
    require(offset==len(logits),'unused unified outputs');return result


def decisions(outputs,rows,families,gates):
    unknown=registry(families);result=[];require(len(outputs)==len(rows),'unaligned unified rows')
    for row,pred in zip(rows,outputs):
        require(len(pred['probabilities'])==len(families) and type(row['target']) is int and 0<=row['target']<len(families)
                and families[row['target']]==row['family'],'unified output family width/target differs')
        named=(pred['predicted']!=unknown and pred['score']>=gates['min_score'] and pred['margin']>1e-8
               and pred['margin']>=gates['min_margin'] and pred['agreement']>=gates['min_patch_agreement'])
        correct=named and pred['predicted']==row['target']
        size=round(row['ink_height_px']*pred['em_ratio'],2) if named and pred['size_spread']<=POLICY['max_size_relative_spread'] else None
        result.append({'family':row['family'],'domain':row['domain'],'view':row['view'],
            'source_id':row['source_id'],'region_id':row['region_id'],'source_font_family':row['source_font_family'],
            'font_file_sha256':row.get('font_file_sha256'),'predicted_family':families[pred['predicted']],
            'named':bool(named),'correct_named':bool(correct),'wrong_named':bool(named and not correct),
            'top1_correct':bool(pred['predicted']==row['target']),'explicit_unknown':bool(pred['predicted']==unknown),
            'size_px':size,'size_ape':abs(size/row['font_size_px']-1) if correct and size is not None else None})
    return result


def _summary(rows,families):
    known=[r for r in rows if r['family']!=UNKNOWN];unknown=[r for r in rows if r['family']==UNKNOWN]
    named=sum(r['named'] for r in rows);correct=sum(r['correct_named'] for r in known);wrong=sum(r['wrong_named'] for r in rows)
    native=[r for r in known if r['view']=='native'];errors=[r['size_ape'] for r in known if r['size_ape'] is not None]
    per_family={f:{'views':0,'correct_named':0,'wrong_named':0} for f in families if f!=UNKNOWN}
    for row in known:
        entry=per_family[row['family']];entry['views']+=1;entry['correct_named']+=int(row['correct_named']);entry['wrong_named']+=int(row['wrong_named'])
    for entry in per_family.values():entry['correct_coverage']=entry['correct_named']/entry['views'] if entry['views'] else None
    unknown_sources=defaultdict(lambda:{'views':0,'wrongly_named':0,'explicit_unknown':0,'preprocessing_rejected':0})
    for row in unknown:
        entry=unknown_sources[row['source_font_family']];entry['views']+=1;entry['wrongly_named']+=int(row['named'])
        entry['explicit_unknown']+=int(row['explicit_unknown']);entry['preprocessing_rejected']+=int(row.get('preprocessing_rejected',False))
    for entry in unknown_sources.values():entry['not_named_rate']=1-entry['wrongly_named']/entry['views']
    return {'views':len(rows),'preprocessing_rejected':sum(r.get('preprocessing_rejected',False) for r in rows),
        'known_views':len(known),'unknown_views':len(unknown),'named':named,'correct_named':correct,'wrong_named':wrong,
        'named_precision':correct/named if named else None,'known_correct_coverage':correct/len(known) if known else None,
        'native_known_views':len(native),'native_known_top1_accuracy':sum(r['top1_correct'] for r in native)/len(native) if native else None,
        'unknown_wrongly_named':sum(r['named'] for r in unknown),
        'unknown_not_named_rate':1-sum(r['named'] for r in unknown)/len(unknown) if unknown else None,
        'explicit_unknown_recall':sum(r['explicit_unknown'] for r in unknown)/len(unknown) if unknown else None,
        'size':{'correctly_named_with_size':len(errors),'coverage_of_correct_names':len(errors)/correct if correct else None,
                'median_ape':float(np.median(errors)) if errors else None,'p90_ape':float(np.quantile(errors,.9)) if errors else None},
        'per_family':per_family,'unknown_sources':dict(unknown_sources)}


def metrics(details,families,rejected=()):
    registry(families)
    abstentions=[]
    for row in rejected:
        require(row.get('family') in families and row.get('domain') in POLICY['required_domains']
                and isinstance(row.get('source_font_family'),str),'rejected region lacks domain/family provenance')
        abstentions.append({**row,'named':False,'correct_named':False,'wrong_named':False,'top1_correct':False,
            'explicit_unknown':False,'size_ape':None,'preprocessing_rejected':True})
    evaluated=list(details)+abstentions
    require(all(row.get('domain') in POLICY['required_domains'] and row['family'] in families for row in evaluated),
            'invalid metric domain/family')
    measured=_summary(evaluated,families)
    domains={domain:_summary([row for row in evaluated if row['domain']==domain],families) for domain in POLICY['required_domains']}
    ge=lambda value,minimum:value is not None and value>=minimum
    le=lambda value,maximum:value is not None and value<=maximum
    checks={'named_precision':ge(measured['named_precision'],POLICY['minimum_named_precision']),
        'known_correct_coverage':ge(measured['known_correct_coverage'],POLICY['minimum_known_correct_coverage']),
        'per_family_coverage':all(ge(entry['correct_coverage'],POLICY['minimum_per_family_correct_coverage']) for entry in measured['per_family'].values()),
        'native_known_top1':ge(measured['native_known_top1_accuracy'],POLICY['minimum_native_known_top1_accuracy']),
        'unknown_not_named':ge(measured['unknown_not_named_rate'],POLICY['minimum_unknown_not_named']),
        'domains':all(ge(d['named_precision'],POLICY['minimum_named_precision']) and ge(d['known_correct_coverage'],POLICY['minimum_known_correct_coverage'])
                      and ge(d['unknown_not_named_rate'],POLICY['minimum_unknown_not_named']) for d in domains.values()),
        'size_median':le(measured['size']['median_ape'],POLICY['maximum_size_median_ape']),
        'size_p90':le(measured['size']['p90_ape'],POLICY['maximum_size_p90_ape']),
        'size_coverage':ge(measured['size']['coverage_of_correct_names'],POLICY['minimum_size_coverage_of_correct_names'])}
    return {**measured,'per_domain':domains,'checks':checks,'passed':all(checks.values()),
        'experimental_precision_priority_met':ge(measured['named_precision'],POLICY['experimental_priority_precision'])
            and measured['named']>=POLICY['experimental_priority_minimum_named']}


def selection_rank(record):
    m=record['metrics']
    return (int(m['experimental_precision_priority_met']),m['correct_named'],-m['wrong_named'],
            -record['calibration_nll'],-record['step'],-record['grid_index'])


def select_grid(logits,ratios,data,step):
    require(logits.shape[1]==len(data['families']),'unified CAL output width differs')
    best=None;history=[]
    combinations=list(itertools.product(POLICY['score_gates'],POLICY['margin_gates'],POLICY['agreement_gates']))
    for temperature in POLICY['temperatures']:
        outputs=region_outputs(logits,ratios,data['rows'],temperature)
        nll=float(np.mean([-math.log(max(1e-12,p['probabilities'][r['target']])) for r,p in zip(data['rows'],outputs)]))
        for score,margin,agreement in combinations:
            gates={'min_score':score,'min_margin':margin,'min_patch_agreement':agreement}
            measured=metrics(decisions(outputs,data['rows'],data['families'],gates),data['families'],data['partition'].get('rejected',[]))
            record={'step':step,'grid_index':len(history),'temperature':temperature,'gates':gates,'calibration_nll':nll,'metrics':measured}
            rank=selection_rank(record);history.append(record)
            if best is None or rank>best[0]:best=(rank,record,outputs)
    return best,history


def initialize(model,checkpoint):
    import torch
    base=model.state_dict();validate_initialization(base,checkpoint)
    require(all(value.dtype==base[key].dtype and bool(torch.isfinite(value).all()) for key,value in checkpoint['state_dict'].items()),
            'invalid initializer parameter dtype/values')
    require(UNKNOWN not in checkpoint['families'],'unified warm start expects a known-font parent')
    parent=checkpoint['state_dict'];inherited=[];new=[]
    for key,value in parent.items():
        if not key.startswith('family_head.'):base[key]=value.detach().clone()
    for index,family in enumerate(model.unified_families):
        if family in checkpoint['families']:
            old=checkpoint['families'].index(family);base['family_head.weight'][index]=parent['family_head.weight'][old]
            base['family_head.bias'][index]=parent['family_head.bias'][old];inherited.append({'family':family,'source_row':old,'target_row':index})
        else:new.append({'family':family,'target_row':index})
    require(len(inherited)==len(checkpoint['families']),'initializer families are absent from the joint registry')
    model.load_state_dict(base,strict=True);model.requires_grad_(True)
    return {'mapped_family_rows':inherited,'new_family_rows':new,'encoder_inherited_in_full':True,
        'size_head_inherited_in_full':True,'all_parameters_trainable':True,'single_parent_encoder':True}


def verify_bindings(bindings):
    require(bindings and all(Path(path).is_file() and sha(path)==digest for path,digest in bindings.items()),'unified frozen source changed')


def train(args):
    import torch
    from region_network import RegionFontClassifier
    from prepare_unified_regions import load_split
    require(args.steps==STEPS and not args.output.exists(),'unified training needs a new output and the fixed 6000-step protocol')
    args.data=args.data.resolve();args.checkpoint=args.checkpoint.resolve();args.output=args.output.resolve()
    training=load_split(args.data,'train');cal=load_split(args.data,'calibration')
    require(training['families']==cal['families'] and training['manifest_sha256']==cal['manifest_sha256'],'unified TRAIN/CAL sources differ')
    families=training['families'];registry(families);sampler=UnifiedSampler(training['rows'],families,args.seed)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    checkpoint_sha=sha(args.checkpoint);checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    prior_path=args.checkpoint.parent/'SELECTION.json';prior=json.loads(prior_path.read_text());prior_sha=sha(prior_path)
    require(checkpoint.get('selection_sha256')==prior_sha and prior.get('families')==checkpoint.get('families')
            and prior.get('test_read') is False and prior.get('state_after_sha256')==state_sha(checkpoint['state_dict']),
            'unified parent checkpoint/selection identity differs')
    verify_bindings(prior['bindings'])
    model=RegionFontClassifier(len(families));model.unified_families=families;inheritance=initialize(model,checkpoint)
    before=state_sha(model.state_dict());initial_groups={name:state_sha({k:v for k,v in model.state_dict().items() if k.startswith(prefix)})
        for name,prefix in [('trunk','trunk.'),('style','style.'),('family_head','family_head.'),('size_head','size_head.')]}
    bindings=dict(training['manifest']['bindings'])
    for path,digest in prior['bindings'].items():
        require(path not in bindings or bindings[path]==digest,'joint data/initializer source conflict');bindings[path]=digest
    for path in [Path(__file__),ROOT/'training/prepare_unified_regions.py',ROOT/'training/train_android_regions.py',
        ROOT/'training/region_network.py',ROOT/'training/network.py',ROOT/'training/train_regions.py',
        ROOT/'src/flux_glyph/region_font.py',ROOT/'src/flux_glyph/unified_font.py',args.checkpoint,prior_path,
        args.data/'MANIFEST.json',args.data/'train/MANIFEST.json',args.data/'calibration/MANIFEST.json']:
        path=path.resolve();digest=sha(path);require(str(path) not in bindings or bindings[str(path)]==digest,'unified source changed')
        bindings[str(path)]=digest
    require(bindings[str(args.checkpoint)]==checkpoint_sha,'unified parent changed while loading')
    inheritance.update(checkpoint={'path':str(args.checkpoint),'sha256':checkpoint_sha},selection_sha256=prior_sha,
        source_state_sha256=state_sha(checkpoint['state_dict']),source_selected_step=prior['selected']['step'],
        source_optimizer_steps_executed=prior['optimizer_steps_executed'],source_calibration_passed=prior.get('calibration_passed') is True)
    args.output.mkdir(parents=True);model.to(args.device)
    protocol={'schema':'flux-glyph-unified-training-protocol-v1','architecture':ARCHITECTURE,'policy':POLICY,'families':families,
        'bindings':bindings,'steps':STEPS,'batch_size':BATCH_SIZE,'learning_rate':LEARNING_RATE,'seed':args.seed,'device':args.device,
        'test_read':False,'development_holdout_read':False,'training_inputs':['image_tiles'],
        'initial_state_sha256':before,'initial_parameter_groups_sha256':initial_groups,'initializer_evidence':inheritance,
        'sampling':SAMPLING,'objective':OBJECTIVE,'optimizer':'All parameters; AdamW weight_decay=.0002; cosine .0002 to .00001; gradient norm clip 5.',
        'model_count':1,'platform_routing':False,'score_merging':False}
    verify_bindings(bindings);dump(args.output/'TRAINING_FREEZE.json',protocol)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=.0002)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=1e-5)
    best=None;history=[];started=time.monotonic()
    for step in range(1,STEPS+1):
        rows=sampler.batch();indices=[r['tile_start']+int(sampler.rng.integers(r['tile_count'])) for r in rows]
        images=torch.from_numpy(np.array(training['tiles'][indices],copy=True)).to(args.device)
        targets=torch.tensor([r['target'] for r in rows],device=args.device)
        sizes=torch.tensor([r['log_em_ratio'] for r in rows],dtype=torch.float32,device=args.device)
        model.train();logits,ratio=model(images)
        ce=torch.nn.functional.cross_entropy(logits,targets,label_smoothing=.03)
        size_loss=torch.nn.functional.smooth_l1_loss(ratio,sizes);loss=ce+.2*size_loss
        require(bool(torch.isfinite(loss)),'nonfinite unified training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step();scheduler.step()
        if step%100==0:print(json.dumps({'step':step,'loss':float(loss.detach().cpu()),'family_loss':float(ce.detach().cpu()),
            'size_loss':float(size_loss.detach().cpu()),'seconds':round(time.monotonic()-started,1)}),flush=True)
        if step%1000==0:
            logits,ratios=infer(model,cal,args.device);selected,grid=select_grid(logits,ratios,cal,step)
            history.append(selected[1]);dump(args.output/f'CAL_GRID_{step:05d}.json',grid)
            if best is None or selected[0]>best[0]:
                best=(selected[0],copy.deepcopy(selected[1]),{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    copy.deepcopy(selected[2]),logits.copy(),ratios.copy())
            dump(args.output/'CAL_PROGRESS.json',history);dump(args.output/'SAMPLING_PROGRESS.json',sampler.report())
            print(json.dumps({'calibration_step':step,'selection':selected[1]},ensure_ascii=False),flush=True)
    model.cpu().load_state_dict(best[2]);after=state_sha(model.state_dict())
    final_groups={name:state_sha({k:v for k,v in model.state_dict().items() if k.startswith(prefix)})
        for name,prefix in [('trunk','trunk.'),('style','style.'),('family_head','family_head.'),('size_head','size_head.')]}
    require(before!=after and all(final_groups[key]!=value for key,value in initial_groups.items()),'a joint parameter group did not train')
    verify_bindings(bindings);dump(args.output/'SAMPLING.json',sampler.report())
    dump(args.output/'CALIBRATION_DECISIONS.json',{'records':best[3],'families':families})
    np.savez(args.output/'CALIBRATION_OUTPUTS.npz',logits=best[4],log_em_ratio=best[5])
    selection={'schema':'flux-glyph-unified-training-selection-v1','architecture':ARCHITECTURE,'policy':POLICY,'families':families,
        'selected':best[1],'passed':best[1]['metrics']['passed'],'calibration_passed':best[1]['metrics']['passed'],
        'test_read':False,'development_holdout_read':False,'history':history,'bindings':bindings,
        'data_manifest_sha256':training['manifest_sha256'],'training_protocol_sha256':sha(args.output/'TRAINING_FREEZE.json'),
        'optimizer_steps_executed':STEPS,'training_device':args.device,'state_before_sha256':before,'state_after_sha256':after,
        'initial_parameter_groups_sha256':initial_groups,'final_parameter_groups_sha256':final_groups,
        'initializer_evidence':inheritance,'calibration_decisions_sha256':sha(args.output/'CALIBRATION_DECISIONS.json'),
        'calibration_outputs_sha256':sha(args.output/'CALIBRATION_OUTPUTS.npz'),'sampling_sha256':sha(args.output/'SAMPLING.json'),
        'model_count':1,'platform_routing':False,'score_merging':False}
    dump(args.output/'SELECTION.json',selection)
    torch.save({'state_dict':model.state_dict(),'families':families,'architecture':ARCHITECTURE,
        'selection_sha256':sha(args.output/'SELECTION.json')},args.output/'model.pth')
    print(json.dumps({'passed':selection['passed'],'selected_step':best[1]['step'],'output':str(args.output)}),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/font-sans-v1/run-v2/verifier.pth')
    p.add_argument('--steps',type=int,default=STEPS);p.add_argument('--seed',type=int,default=2026091401)
    p.add_argument('--device',choices=('mps','cpu'),default='mps');return p


if __name__=='__main__':train(parser().parse_args())
