#!/usr/bin/env python3
"""Train/export a font classifier. Only calibration chooses weights and gates."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from data import ROOT, build_latin, dump, npz_dataset, old_dataset, sha
from network import FAMILIES, SCRIPTS, FontClassifier
from real_data import PreparedScreenshotData

R8 = 'alipay-font-generalization-v8-20260911'
R10 = 'alipay-font-smalltext-v10-20260911'
SEED = 2026091201


def state_sha(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def script_indices(script):
    return [FAMILIES.index(name) for name in SCRIPTS[script]]


def annotated_source_summary(bundle):
    if bundle is None:
        return None
    by_kind={}
    for row in bundle.manifest['sources']:
        by_kind.setdefault(row['source_kind'],set()).add(row['source_sha256'])
    return {'annotated_source_count':bundle.manifest['labelled_screenshot_count'],
            'declared_source_kind_unique_counts':{kind:len(values) for kind,values in by_kind.items()},
            'font_truth_independently_certified':False,
            'note':'Source kinds are declared annotations; synthetic fixtures are not counted as verified real screenshots.'}


def augment(x):
    # Mild, class-independent screenshot resampling and OCR box perturbation.
    side = int(torch.randint(42, 65, ()).item())
    x = F.interpolate(F.interpolate(x, size=(side, side), mode='bilinear', align_corners=False), size=(64,64), mode='bilinear', align_corners=False)
    if torch.rand(()).item() < .3:
        x = .9*x + .1*F.avg_pool2d(x, 3, 1, 1)
    if torch.rand(()).item() < .3:
        shift = [int(v) for v in torch.randint(-1, 2, (2,)).tolist()]
        x = torch.roll(x, shifts=shift, dims=(2,3))
        if shift[0] > 0: x[:, :, :shift[0], :] = 0
        if shift[0] < 0: x[:, :, shift[0]:, :] = 0
        if shift[1] > 0: x[:, :, :, :shift[1]] = 0
        if shift[1] < 0: x[:, :, :, shift[1]:] = 0
    return (x + torch.randn_like(x)*.004).clamp(0,1)


def predict(net, dataset, device, batch=256):
    net.eval(); pieces=[]
    with torch.inference_mode():
        for start in range(0, len(dataset['x']), batch):
            x=torch.from_numpy(np.array(dataset['x'][start:start+batch,None], copy=True)).to(device)
            pieces.append(net(x).cpu().numpy())
    return np.concatenate(pieces)


def metrics(logits, data, temperature=1., gates=None):
    indices=script_indices(data['script']); scores=logits[:,indices]/temperature
    scores=scores-scores.max(1,keepdims=True); probs=np.exp(scores); probs/=probs.sum(1,keepdims=True)
    ranking=np.argsort(-probs,axis=1); confidence=probs[np.arange(len(probs)),ranking[:,0]]
    margin=confidence-probs[np.arange(len(probs)),ranking[:,1]]
    pred=np.asarray(indices)[ranking[:,0]]; correct=pred==data['y'];valid=data['valid']
    per_family={FAMILIES[int(c)]: {'rows': int((data['y']==c).sum()), 'correct': int((correct & (data['y']==c)).sum()),
                    'recall': float(correct[data['y']==c].mean())} for c in sorted(set(data['y'].tolist()))}
    report={'rows':len(pred), 'valid':int(valid.sum()), 'accuracy':float((correct & valid).mean()),
            'macro_accuracy':float(np.mean([v['recall'] for v in per_family.values()])), 'families':per_family}
    if gates:
        accepted=(confidence>=gates['min_score']) & (margin>=gates['min_margin']) & valid
        report['accepted']={'rows':int(accepted.sum()),'correct':int((accepted & correct).sum()),
            'wrong':int((accepted & ~correct).sum()), 'coverage':float(accepted.mean()),
            'precision':float(correct[accepted].mean()) if accepted.any() else None}
    return report, confidence, margin, correct


def pool_calibration(datasets, predictions, script):
    chosen=[(d,p) for d,p in zip(datasets,predictions) if d['script']==script]
    return {'y':np.concatenate([d['y'] for d,_ in chosen]),'valid':np.concatenate([d['valid'] for d,_ in chosen]),'script':script}, np.concatenate([p for _,p in chosen])


def region_observations(logits, data, temperature, count):
    """Runtime parity: repeat-character probabilities average before region mean."""
    if data.get('real_data'):
        # Prepared inputs contain glyphs from different real regions. Never
        # manufacture a region by mixing characters from different screenshots.
        return np.empty(0),np.empty(0),np.empty(0,dtype=bool)
    indices=script_indices(data['script']);scores=logits[:,indices]/temperature
    scores-=scores.max(1,keepdims=True);probs=np.exp(scores);probs/=probs.sum(1,keepdims=True)
    groups={}
    for i,(face,size,char) in enumerate(zip(data['faces'],data['sizes'],data['chars'])):
        if data['valid'][i]:groups.setdefault((str(face),int(size),int(data['y'][i])),{}).setdefault(str(char),[]).append(i)
    confidences=[];margins=[];correct=[]
    for (_,_,target),characters in groups.items():
        keys=sorted(characters)
        for start in range(0,len(keys)-count+1,count):
            mean=np.mean([probs[characters[c]].mean(0) for c in keys[start:start+count]],axis=0)
            ranked=np.argsort(-mean);confidences.append(float(mean[ranked[0]]));margins.append(float(mean[ranked[0]]-mean[ranked[1]]));correct.append(indices[int(ranked[0])]==target)
    return np.array(confidences),np.array(margins),np.array(correct,dtype=bool)


def region_report(logits,data,temperature,gates):
    if data.get('real_data'):
        return {'skipped':True,'reason':'Prepared screenshot glyphs are evaluated individually; no cross-image synthetic region aggregation.'}
    result={}
    for count in (2,4):
        conf,margin,correct=region_observations(logits,data,temperature,count)
        accepted=(conf>=gates['min_score'])&(margin>=gates['min_margin'])
        result[str(count)]={'rows':len(conf),'accuracy':float(correct.mean()) if len(conf) else None,
            'accepted':int(accepted.sum()),'correct_accepted':int((correct&accepted).sum()),
            'wrong_accepted':int((~correct&accepted).sum())}
    return result


def calibrate(datasets, predictions):
    temperatures={};gates={};report={}
    for script in SCRIPTS:
        data,logits=pool_calibration(datasets,predictions,script); inds=script_indices(script)
        target=torch.tensor([inds.index(int(y)) for y in data['y']]); raw=torch.from_numpy(logits[:,inds])
        temperature=min([.75,1.,1.25,1.5,2.,3.], key=lambda t: float(F.cross_entropy(raw/t,target)))
        temperatures[script]=temperature
        base,conf,margin,correct=metrics(logits,data,temperature)
        region=[region_observations(p,d,temperature,count) for d,p in zip(datasets,predictions) if d['script']==script for count in (2,4)]
        conf=np.concatenate([conf]+[v[0] for v in region]);margin=np.concatenate([margin]+[v[1] for v in region]);correct=np.concatenate([correct]+[v[2] for v in region])
        options=[]
        for score in (.6,.7,.75,.8,.85,.9,.925,.95,.975,.99):
            for gap in (.05,.1,.15,.2,.3,.4):
                accept=(conf>=score)&(margin>=gap);n=int(accept.sum());right=int((accept&correct).sum())
                if n>=20 and right/n>=.97:
                    options.append((n,right/n,-score,-gap,score,gap))
        # Candidate-only; high-confidence known-font accuracy does not prove
        # unknown-font rejection. If no gate reaches target, reject cautiously.
        selected=max(options) if options else None
        gates[script]={'min_score':selected[-2] if selected else .99,'min_margin':selected[-1] if selected else .4}
        report[script]={'temperature':temperature,'gate_found':bool(selected),
                        'metrics':metrics(logits,data,temperature,gates[script])[0],
                        'scope':'Calibration labels come from source fonts and, if supplied, explicit screenshot annotations; unknown-font rejection is unvalidated.'}
    return temperatures,gates,report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old-root',type=Path,required=True);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/neural-font-v1')
    ap.add_argument('--device',choices=['cpu','mps','auto'],default='cpu');ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--steps-per-epoch',type=int,default=100)
    ap.add_argument('--real-data',type=Path,help='Optional source-labeled screenshot datasets from prepare_screenshots.py')
    ap.add_argument('--build-data-only',action='store_true');args=ap.parse_args();out=args.output.resolve()
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED);rng=np.random.default_rng(SEED)
    device='mps' if args.device=='auto' and torch.backends.mps.is_available() else ('cpu' if args.device=='auto' else args.device)
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS requested but unavailable')
    if not (out/'data/latin/manifest.json').exists(): build_latin(args.old_root,out/'data/latin')
    if args.build_data_only:return
    if (out/'TRAINING_FREEZE.json').exists():raise FileExistsError('Training output already frozen; use a new --output')
    real_bundle=PreparedScreenshotData(args.real_data,families=FAMILIES,scripts=SCRIPTS,
                                      allow_empty_evaluation=True) if args.real_data else None
    train=[old_dataset(args.old_root,R8,'train'),old_dataset(args.old_root,R10,'train'),npz_dataset(out/'data/latin/train.npz','latin')]
    cal=[old_dataset(args.old_root,R8,'calibration'),old_dataset(args.old_root,R10,'calibration'),npz_dataset(out/'data/latin/calibration.npz','latin')]
    has_ft=(out/'data/han-freetype/train.npz').exists()
    if has_ft:
        train.append(npz_dataset(out/'data/han-freetype/train.npz','han'));cal.append(npz_dataset(out/'data/han-freetype/calibration.npz','han'))
    if real_bundle:
        train.extend(real_bundle.datasets('train'));cal.extend(real_bundle.datasets('calibration'))
    train_chars={s:set(c for d in train if d['script']==s and not d.get('real_data') for c in d['chars']) for s in SCRIPTS}
    cal_chars={s:set(c for d in cal if d['script']==s and not d.get('real_data') for c in d['chars']) for s in SCRIPTS}
    for script in SCRIPTS:
        if train_chars[script]&cal_chars[script]:raise ValueError('Training/calibration character leakage')
    parent=args.old_root/'runs'/R8/'font-experiment-package/glyph-font-runtime.pth'
    net=FontClassifier();net.warm_start(parent);initial=copy.deepcopy(net.state_dict());teacher=copy.deepcopy(net).to(device).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    net=net.to(device);before=state_sha(net.state_dict());start=time.perf_counter()
    initial_predictions=[predict(net,d,device) for d in cal]
    baseline=[metrics(p,d)[0] for p,d in zip(initial_predictions,cal)]
    freeze={'schema':'flux-glyph-neural-training-v1','seed':SEED,'device':device,'torch':torch.__version__,
            'families':FAMILIES,'scripts':SCRIPTS,'epochs':args.epochs,'steps_per_epoch':args.steps_per_epoch,
            'batch_size':96,'optimizer':{'name':'AdamW','lr':.0001,'weight_decay':.0001},
            'loss':'script-masked family cross entropy + 0.15 R8 teacher KL on Chinese replay',
            'selection':'max mean Han/Latin calibration macro accuracy; Han R8 macro drop <=0.03 preferred; earliest tie',
            'parent_sha256':sha(parent),'initial_state_sha256':before,'train_sources':[d['source'] for d in train],
            'calibration_sources':[d['source'] for d in cal], 'synthetic_train_cal_characters_disjoint':True,
            'preprocessing':'extract_glyphs(image, 1); float32 foreground [1,64,64]',
            'code':{p.name:sha(p) for p in (Path(__file__),ROOT/'training/network.py',ROOT/'training/data.py',ROOT/'training/real_data.py')},
            'test_glyph_pixels_loaded':False,'test_metadata_inspected_for_contract':bool(real_bundle),
            'real_data_manifest':sha(args.real_data/'manifest.json') if args.real_data else None,
            'annotated_sources':annotated_source_summary(real_bundle),
            'prepared_screenshot_audit':copy.deepcopy(real_bundle.audit) if real_bundle else None}
    dump(out/'TRAINING_FREEZE.json',freeze)
    frozen=out/'frozen-training';frozen.mkdir(exist_ok=False)
    for name in freeze['code']:
        (frozen/name).write_bytes((ROOT/'training'/name).read_bytes())
    pools=[{int(c):np.where((d['y']==c)&d['valid'])[0] for c in np.unique(d['y'])} for d in train]
    sampler_classes=[np.array(list(p)) for p in pools]
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4)
    history=[];selections=[];best=None
    for epoch in range(1,args.epochs+1):
        net.train();losses=[];ces=[];kls=[]
        for step in range(args.steps_per_epoch):
            # Latin receives a fixed third of each batch; Chinese datasets share
            # the rest. Uniform family sampling prevents dominant PingFang counts.
            latin_indices=[i for i,d in enumerate(train) if d['script']=='latin'];han_indices=[i for i,d in enumerate(train) if d['script']=='han']
            chosen=[(i,32//len(latin_indices)+(j<32%len(latin_indices))) for j,i in enumerate(latin_indices)]+[(i,64//len(han_indices)+(j<64%len(han_indices))) for j,i in enumerate(han_indices)]
            xs=[];ys=[];scripts=[];replay=[]
            for di,n in chosen:
                selected=rng.choice(sampler_classes[di],n);idx=np.array([int(rng.choice(pools[di][int(c)])) for c in selected])
                xs.append(np.array(train[di]['x'][idx,None],copy=True));ys.extend(selected.tolist());scripts.extend([train[di]['script']]*n);replay.extend([di==0]*n)
            x=augment(torch.from_numpy(np.concatenate(xs)).to(device));y=torch.tensor(ys,dtype=torch.long,device=device)
            optimizer.zero_grad(set_to_none=True);logits=net(x);ce=0.
            for script in SCRIPTS:
                rows=torch.tensor([s==script for s in scripts],device=device);inds=script_indices(script)
                local=torch.tensor([inds.index(ys[i]) for i,s in enumerate(scripts) if s==script],dtype=torch.long,device=device)
                ce=ce+F.cross_entropy(logits[rows][:,inds],local)/2.
            replay_mask=torch.tensor(replay,device=device)
            with torch.no_grad():expected=teacher(x[replay_mask])[:,:11]
            kl=F.kl_div(F.log_softmax(logits[replay_mask,:11]/2,dim=1),F.softmax(expected/2,dim=1),reduction='batchmean')*4
            loss=ce+.15*kl
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            progress=((epoch-1)*args.steps_per_epoch+step+1)/(args.epochs*args.steps_per_epoch)
            optimizer.param_groups[0]['lr']=2e-5+.5*8e-5*(1+math.cos(math.pi*progress))
            losses.append(float(loss.detach()));ces.append(float(ce.detach()));kls.append(float(kl.detach()))
        record={'epoch':epoch,'optimizer_steps':epoch*args.steps_per_epoch,'loss':float(np.mean(losses)),
                'cross_entropy':float(np.mean(ces)),'teacher_kl':float(np.mean(kls)),'seconds':time.perf_counter()-start}
        if epoch%2==0 or epoch==args.epochs:
            predictions=[predict(net,d,device) for d in cal];reports=[metrics(p,d)[0] for p,d in zip(predictions,cal)]
            script_macro={s:float(np.mean([r['macro_accuracy'] for d,r in zip(cal,reports) if d['script']==s])) for s in SCRIPTS}
            eligible=reports[0]['macro_accuracy']>=baseline[0]['macro_accuracy']-.03
            score=float(np.mean(list(script_macro.values())));key=(eligible,score,-epoch)
            checkpoint=out/f'checkpoint-epoch-{epoch:02}.pth';state={k:v.detach().cpu() for k,v in net.state_dict().items()}
            torch.save({'state_dict':state,'families':FAMILIES,'scripts':SCRIPTS,'epoch':epoch,'optimizer_steps':record['optimizer_steps']},checkpoint)
            choice={'epoch':epoch,'eligible':eligible,'score':score,'script_macro':script_macro,'checkpoint':str(checkpoint),'sha256':sha(checkpoint),'reports':reports}
            selections.append(choice);record['calibration']={k:choice[k] for k in ('eligible','score','script_macro')}
            if best is None or key>best[0]:best=(key,choice)
        history.append(record);dump(out/'training-history.json',history);print(json.dumps(record),flush=True)
    selected=best[1];checkpoint=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True);net.load_state_dict(checkpoint['state_dict'])
    after=state_sha(net.state_dict());assert after!=before
    parameter_change={k:float((checkpoint['state_dict'][k]-initial[k]).norm()) for k in ['trunk.0.weight','style.0.weight','family_head.weight']}
    cal_predictions=[predict(net,d,device) for d in cal];temperatures,gates,cal_report=calibrate(cal,cal_predictions)
    selection={'schema':'neural-selection-before-test-v1','selected':selected,'all_checkpoints':selections,
                'state_before_sha256':before,'state_after_sha256':after,'parameter_l2_change':parameter_change,
                'optimizer_steps_executed':args.epochs*args.steps_per_epoch,'elapsed_training_seconds':time.perf_counter()-start,
                'temperatures':temperatures,'gates':gates,'calibration':cal_report,
                'test_glyph_pixels_loaded':False,'test_metadata_inspected_for_contract':bool(real_bundle)}
    dump(out/'SELECTION_FREEZE.json',selection)
    # The first and only access to held-out test arrays happens after selection.
    test=[old_dataset(args.old_root,R10,'test'),npz_dataset(out/'data/latin/test.npz','latin')]
    if has_ft:test.append(npz_dataset(out/'data/han-freetype/test.npz','han'))
    if real_bundle:test.extend(real_bundle.datasets('test'))
    for d in test:
        if not d.get('real_data') and set(d['chars'])&(train_chars[d['script']]|cal_chars[d['script']]):raise ValueError('Test character leakage')
    selected_steps=selected['epoch']*args.steps_per_epoch
    report={'schema':'neural-font-test-report-v1','selected_epoch':selected['epoch'],'optimizer_steps':args.epochs*args.steps_per_epoch,
            'optimizer_steps_executed':args.epochs*args.steps_per_epoch,'selected_optimizer_steps':selected_steps,
            'selected_previous_chinese_protection_passed':selected['eligible'],'calibration':cal_report,'release_status':'experimental',
            'annotated_sources':annotated_source_summary(real_bundle),
            'prepared_screenshot_audit':copy.deepcopy(real_bundle.audit) if real_bundle else None,
            'training_seconds':selection['elapsed_training_seconds'],'weights_before_sha256':before,'weights_after_sha256':after,
            'parameter_l2_change':parameter_change,'datasets':[],'limitations':[
            ('Explicit screenshot annotations were supplied; their font labels and source kinds are not independently certified by the importer.' if real_bundle else 'Synthetic source-font labels only; zero verified real-device screenshots.'),
            'R10 test is a previously reported benchmark, held out from this training and the R8 parent, not a newly untouched research test.',
            'Unknown-font rejection is not validated; predictions remain candidates.']}
    for d in test:
        p=predict(net,d,device);base_net=FontClassifier().to(device);base_net.load_state_dict(initial)
        p0=predict(base_net,d,device)
        m=metrics(p,d,temperatures[d['script']],gates[d['script']])[0]
        entry={'name':d['name'],'script':d['script'],'source':d['source'],'before':metrics(p0,d)[0], 'after':m,
            'region_after':region_report(p,d,temperatures[d['script']],gates[d['script']])}
        report['datasets'].append(entry);print(json.dumps({'test':d['name'],'before':entry['before']['accuracy'],'after':m['accuracy']}),flush=True)
    neural=out/'neural';neural.mkdir(parents=True,exist_ok=True);net=net.cpu().eval()
    model_path=neural/'model.onnx';sample=torch.zeros(2,1,64,64,dtype=torch.float32)
    torch.onnx.export(net,sample,model_path,input_names=['glyphs'],output_names=['logits'],dynamic_axes={'glyphs':{0:'batch'},'logits':{0:'batch'}},opset_version=17,dynamo=False)
    torch.save(checkpoint,neural/'model.pth')
    meta={'schema':'flux-glyph-neural-font-v1','algorithm':'glyph-cnn64-v1','model':{'path':'model.onnx','sha256':sha(model_path)},
            'families':FAMILIES,'scripts':SCRIPTS,'gates':gates,'temperature':temperatures,
            'preprocessing':{'algorithm':'extract_glyphs-v1','shape':[1,64,64],'dtype':'float32'},
            'training':{'selected_epoch':selected['epoch'],'optimizer_steps_executed':args.epochs*args.steps_per_epoch,
                        'selected_optimizer_steps':selected_steps,'previous_chinese_calibration_protection_passed':selected['eligible'],
                        'selection_sha256':sha(out/'SELECTION_FREEZE.json'),'parent_sha256':sha(parent),
                        'state_before_sha256':before,'state_after_sha256':after,
                        'data':'Source-verified font renders plus explicit screenshot annotations' if real_bundle else 'source-verified synthetic font renders'},
            'release_status':'experimental','calibration':{'acceptance_precision_target':.97,'minimum_accepted_rows':20,
                'gate_found':{key:value['gate_found'] for key,value in cal_report.items()},
                'fallback_gate_note':'If gate_found=false, fallback thresholds do not demonstrate target precision.'},
            'scope':'Font-family candidates; real screenshot accuracy and unknown-font rejection not established.'}
    dump(neural/'metadata.json',meta);report['onnx_sha256']=sha(model_path);report['onnx_bytes']=model_path.stat().st_size
    dump(out/'report.json',report);dump(ROOT/'docs/neural-training-report.json',report)
    print(json.dumps({'finished':True,'output':str(out),'selected_epoch':selected['epoch'],'onnx':str(model_path)}),flush=True)

if __name__=='__main__':main()
