#!/usr/bin/env python3
"""One post-CAL-declared equal-weight parameter average, with no new training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'training')]
import train_unified_retention_core as core
from train_unified_retention import evaluate_outputs,parameter_groups,initialize,state_sha,sha,dump,require,verify_bindings
from train_unified_regions import ARCHITECTURE

SOURCE_RUN=ROOT/'artifacts/unified-font-v2/run-core-v1'
DATA=ROOT/'artifacts/unified-font-v1/data-v1'
PLAN=ROOT/'artifacts/unified-font-v2/RETENTION_PLAN.json'
SOURCE_STEPS=(500,1000,1500)
SOURCE_TRAINER_SHA='cda5026d88e0938b66f805693f57332bc91060bb0402c56e94e69925eb5b0649'
PLAN_SHA='b82283b54b5e47f4639b0df62231544c0419f6299d10820d86dca96ef24094ff'
METHOD={'kind':'equal_weight_student_parameter_average','checkpoint_steps':list(SOURCE_STEPS),
    'weights':[1/3,1/3,1/3],'accumulation':'CPU float64, checkpoint order 500 then 1000 then 1500, divide by 3, cast to float32',
    'nonfloating_buffers':'All three must be exactly equal; preserve the original dtype.',
    'candidate_count':1,'ratio_search':False,'subset_search':False,'teacher_in_model':False,
    'output_averaging':False,'optimizer_steps_executed':0,'source_optimizer_steps_executed':1500,
    'declared_after_inspecting_source_calibration':True,'predeclared_before_source_training':False,
    'test_read':False,'development_holdout_read':False}


def read(path):
    return json.loads(Path(path).read_text())


def average_states(states):
    """Fixed ordered arithmetic; model architecture is checked by the caller."""
    import torch
    require(len(states)==3 and all(isinstance(state,dict) and state for state in states),
            'Exactly three complete states are required')
    require(all(set(state)==set(states[0]) for state in states),'Checkpoint state keys differ')
    averaged={}
    for name,original in states[0].items():
        require(isinstance(original,torch.Tensor),'Checkpoint state must contain tensors only')
        values=[state[name] for state in states]
        require(all(isinstance(value,torch.Tensor) and value.shape==original.shape and value.dtype==original.dtype
                    and not value.is_complex() and bool(torch.isfinite(value).all()) for value in values),
                'Checkpoint shape, dtype or finite values differ: '+name)
        if original.is_floating_point():
            require(original.dtype==torch.float32,'Student floating parameters must be float32')
            total=torch.zeros(original.shape,dtype=torch.float64,device='cpu')
            for value in values:total.add_(value.detach().to(device='cpu',dtype=torch.float64))
            averaged[name]=(total/3.).to(dtype=torch.float32)
            require(bool(torch.isfinite(averaged[name]).all()),'Nonfinite averaged parameters')
        else:
            require(all(torch.equal(original.cpu(),value.cpu()) for value in values),'Nonfloating buffers differ: '+name)
            averaged[name]=original.detach().cpu().clone()
    return averaged


def average_checkpoints(checkpoints,families,expected):
    require(len(families)==25 and len(set(families))==25 and families.count(core.UNKNOWN)==1,
            'The complete 25-class unified registry is required')
    require(len(checkpoints)==3,'Exactly checkpoints 500, 1000 and 1500 are required')
    for checkpoint,step in zip(checkpoints,SOURCE_STEPS):
        require(checkpoint.get('families')==families and checkpoint.get('architecture')==ARCHITECTURE
                and checkpoint.get('step')==step and set(checkpoint.get('state_dict',{}))==set(expected),
                'Checkpoint order, class order, student-only architecture or state differs')
        state=checkpoint['state_dict']
        require(all(state[name].shape==value.shape and state[name].dtype==value.dtype for name,value in expected.items()),
                'Checkpoint parameters do not match the one student CNN')
    return average_states([checkpoint['state_dict'] for checkpoint in checkpoints])


def infer_after_freeze(model,data,device,freeze_path,freeze_sha,checkpoint_path,checkpoint_sha):
    require(Path(freeze_path).is_file() and sha(freeze_path)==freeze_sha
            and Path(checkpoint_path).is_file() and sha(checkpoint_path)==checkpoint_sha,
            'Average freeze and saved student checkpoint must be intact before CAL inference')
    return core.infer(model,data,device)


def run(args):
    import torch
    from region_network import RegionFontClassifier
    output=args.output.resolve()
    require(not output.exists(),'Preserve prior candidate evidence; output must be new')
    selection=read(SOURCE_RUN/'SELECTION.json');protocol=read(SOURCE_RUN/'TRAINING_FREEZE.json')
    failure=read(SOURCE_RUN/'report.json')
    require(selection.get('schema')=='flux-glyph-unified-retention-selection-v1'
            and selection.get('objective_variant')==core.OBJECTIVE_VARIANT
            and selection.get('objective')==protocol.get('objective')==core.OBJECTIVE
            and selection.get('promotion_allowed') is False
            and failure.get('status')=='NO_PROMOTABLE_CHECKPOINT'
            and failure.get('promotion_allowed') is False
            and failure.get('optimizer_steps_executed')==selection.get('optimizer_steps_executed')==protocol.get('steps')==1500
            and protocol.get('eval_every')==500 and protocol.get('batch_size')==96
            and selection.get('fixed_runtime')==protocol.get('fixed_runtime')==core.FIXED_RUNTIME
            and all(document.get(key) is False for document in (selection,protocol,failure)
                    for key in ('test_read','development_holdout_read')),
            'The complete failed core run is required; no holdout may have been read')
    require(sha(SOURCE_RUN/'TRAINING_FREEZE.json')==selection['training_protocol_sha256']
            and sha(ROOT/'training/train_unified_retention_core.py')==SOURCE_TRAINER_SHA
            and sha(PLAN)==PLAN_SHA==selection['retention_plan']['sha256'],
            'Source training freeze, trainer or 46-check retention plan changed')
    parent=(ROOT/selection['parent_checkpoint']['path']).resolve()
    plan=core.read_plan(PLAN,DATA,parent);families=plan['families']
    require(families==selection['families'] and selection['data_manifest_sha256']==sha(DATA/'MANIFEST.json'),
            'Source class order or prepared data differs')
    require(selection['bindings']==protocol['bindings'],'Source training bindings differ')
    bindings=dict(selection['bindings']);verify_bindings(bindings)
    history=selection['history']
    require([record['step'] for record in history]==list(SOURCE_STEPS)
            and all(record.get('promotion_allowed') is False for record in history),
            'Expected all three completed nonpromotable checkpoints, in chronological order')
    torch.set_num_threads(4)
    model=RegionFontClassifier(25)
    checkpoints=[];source_records=[]
    for record,step in zip(history,SOURCE_STEPS):
        relative=f'checkpoints/step{step:05d}/model.pth';entry=record['artifacts']['checkpoint']
        path=SOURCE_RUN/relative
        require(entry['path']==relative and sha(path)==entry['sha256'],'Saved source checkpoint changed')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        require(checkpoint.get('training_protocol_sha256')==selection['training_protocol_sha256'],
                'Source checkpoint does not belong to the completed core training run')
        # Validates the exact student parameter set and all 25 output rows; never loads a teacher model.
        initialize(model,checkpoint,families)
        checkpoints.append(checkpoint)
        source_records.append({'step':step,'path':str(path),'sha256':sha(path),
            'state_sha256':state_sha(checkpoint['state_dict']),'parameter_groups_sha256':parameter_groups(checkpoint['state_dict'])})
        bindings[str(path)]=sha(path)
    require(failure['last_checkpoint_sha256']==source_records[-1]['sha256'], 'Incomplete source optimizer execution evidence')
    paths=[Path(__file__),ROOT/'training/train_unified_retention_core.py',ROOT/'training/train_unified_retention.py',
        ROOT/'training/train_unified_regions.py',ROOT/'training/prepare_unified_regions.py',ROOT/'training/region_network.py',
        ROOT/'training/network.py',ROOT/'training/train_regions.py',ROOT/'src/flux_glyph/unified_font.py',
        ROOT/'src/flux_glyph/region_font.py',ROOT/'training/evaluate_unified_retention_average.py',
        ROOT/'artifacts/unified-font-v1/run-v1/DEVELOPMENT_REGRESSION.json',
        SOURCE_RUN/'TRAINING_FREEZE.json',SOURCE_RUN/'SELECTION.json',SOURCE_RUN/'report.json',
        SOURCE_RUN/'model.pth',SOURCE_RUN/'DISTILLATION.json',SOURCE_RUN/'SAMPLING.json',
        PLAN,DATA/'MANIFEST.json',DATA/'train/MANIFEST.json',DATA/'calibration/MANIFEST.json']
    for split in ('train','calibration'):
        part=read(DATA/split/'MANIFEST.json')
        for key in ('metadata','array'):
            path=(DATA/split/part[key]['path']).resolve()
            require(path.parent==DATA/split and sha(path)==part[key]['sha256'],'Prepared source asset changed')
            paths.append(path)
    for path in paths:
        path=path.resolve();digest=sha(path)
        require(str(path) not in bindings or bindings[str(path)]==digest,'Source binding conflict')
        bindings[str(path)]=digest
    averaged=average_checkpoints(checkpoints,families,model.state_dict())
    model.load_state_dict(averaged,strict=True)
    average_sha=state_sha(averaged);average_groups=parameter_groups(averaged)
    require(all(average_sha!=record['state_sha256'] for record in source_records),
            'Averaged state must be a new candidate rather than an unchanged source checkpoint')
    freeze={'schema':'flux-glyph-unified-retention-averaging-freeze-v1','architecture':ARCHITECTURE,
        'method':METHOD,'source_run':str(SOURCE_RUN),'source_training_protocol_sha256':sha(SOURCE_RUN/'TRAINING_FREEZE.json'),
        'source_selection_sha256':sha(SOURCE_RUN/'SELECTION.json'),'source_checkpoints':source_records,
        'source_training_objective':selection['objective'],'source_teacher_used_only_during_training':True,
        'families':families,'model_count':1,'platform_routing':False,'score_merging':False,
        'data_manifest_sha256':sha(DATA/'MANIFEST.json'),'retention_plan':{'path':str(PLAN),'sha256':PLAN_SHA},
        'fixed_runtime':core.FIXED_RUNTIME,'runtime_gates_changed':False,'bindings':bindings,
        'state_sha256':average_sha,'parameter_groups_sha256':average_groups,
        'calibration_device':args.device,'candidate_step':0,'selection_search_performed':False,
        'test_read':False,'development_holdout_read':False,
        'scope':'One new candidate declared after observing failed source CAL. It is not a previously registered training selection or independent test.'}
    verify_bindings(bindings)
    output.mkdir(parents=True);dump(output/'AVERAGING_FREEZE.json',freeze);freeze_sha=sha(output/'AVERAGING_FREEZE.json')
    checkpoint_path=output/'model.pth'
    torch.save({'state_dict':averaged,'families':families,'architecture':ARCHITECTURE,'step':0,
        'averaging_freeze_sha256':freeze_sha,'optimizer_steps_executed':0,'source_optimizer_steps_executed':1500},checkpoint_path)
    checkpoint_sha=sha(checkpoint_path)
    # No source CAL outputs are averaged: run the one averaged student's complete CAL exactly once.
    cal=core.load_split(DATA,'calibration')
    require(cal['families']==families and cal['manifest_sha256']==freeze['data_manifest_sha256'],'CAL source changed')
    model.eval().to(args.device)
    logits,ratios=infer_after_freeze(model,cal,args.device,output/'AVERAGING_FREEZE.json',freeze_sha,checkpoint_path,checkpoint_sha)
    record,outputs,_=evaluate_outputs(logits,ratios,cal,plan,0)
    require(len(record['retention_checks'])==46,'The frozen 46 retention checks changed')
    verify_bindings(bindings)
    require(sha(output/'AVERAGING_FREEZE.json')==freeze_sha and sha(checkpoint_path)==checkpoint_sha
            and state_sha(model.state_dict())==average_sha,'Candidate or freeze changed during inference')
    np.savez(output/'CALIBRATION_OUTPUTS.npz',logits=logits,log_em_ratio=ratios)
    dump(output/'CALIBRATION_DECISIONS.json',{'families':families,'records':outputs})
    dump(output/'RESULT.json',record)
    report={'schema':'flux-glyph-unified-retention-averaging-result-v1',
        'status':'PROMOTABLE_CANDIDATE' if record['promotion_allowed'] else 'NO_PROMOTABLE_CHECKPOINT',
        'promotion_allowed':record['promotion_allowed'],'calibration_passed':record['metrics']['passed'],
        'stable_validation_passed':False,'test_passed':False,'method':METHOD,'optimizer_steps_executed':0,
        'source_optimizer_steps_executed':1500,'state_sha256':average_sha,'parameter_groups_sha256':average_groups,
        'averaging_freeze_sha256':freeze_sha,'checkpoint_sha256':checkpoint_sha,
        'calibration_outputs_sha256':sha(output/'CALIBRATION_OUTPUTS.npz'),
        'calibration_decisions_sha256':sha(output/'CALIBRATION_DECISIONS.json'),'result_sha256':sha(output/'RESULT.json'),
        'calibration_inference_completed':True,'calibration_inference_passes':1,'runtime_gates_changed':False,
        'test_read':False,'development_holdout_read':False,'selected_step':0,'fixed_runtime':core.FIXED_RUNTIME,
        'retention_checks_passed':sum(row['passed'] for row in record['retention_checks']),'retention_checks_total':46}
    dump(output/'report.json',report)
    if not record['promotion_allowed']:dump(output/'NO_PROMOTABLE_CHECKPOINT.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/unified-font-v2/average-core-v1')
    p.add_argument('--device',choices=('mps','cpu'),default='mps')
    return p


if __name__=='__main__':run(parser().parse_args())
