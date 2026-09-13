#!/usr/bin/env python3
"""One fixed post-training 50/50 parameter average; no optimization or gate search."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'training')]
import train_unified_retention_core as core
import train_unified_retention_source_balanced as source
from train_unified_retention import (evaluate_outputs,parameter_groups,initialize,state_sha,
    retention_rank,sha,dump,require,verify_bindings,FIXED_RUNTIME)
from train_unified_regions import ARCHITECTURE
from export_unified_retention_core import checked_file,strip_artifacts,require_fixed_runtime

DATA=ROOT/'artifacts/unified-font-v1/data-v1'
PLAN=ROOT/'artifacts/unified-font-v2/RETENTION_PLAN.json'
PLAN_SHA='b82283b54b5e47f4639b0df62231544c0419f6299d10820d86dca96ef24094ff'
SOURCE_RUNS=(ROOT/'artifacts/unified-font-v2/run-core-v1',ROOT/'artifacts/unified-font-v2/run-source-balanced-v1')
CHECKPOINT_SHAS=('f2f7f7c6d44aad5b3e22a2ba7d5c78b069fbb0a04423ee9bf186bfa60e52533e',
    'b08c0e20e98835e55094debedcad495d579d478b8f793aa733601383bec87763')
SELECTION_SHAS=('18b5cbf95e45fee83283256581f32c94dae80f653e2fdfe1e24ddf25bfb414da',
    '14825ee2c994aeb1a387f590f95ff4354efe6323ab92891230a47c69837002d8')
TRAINER_SHAS=('cda5026d88e0938b66f805693f57332bc91060bb0402c56e94e69925eb5b0649',
    'b846578cf4ebb46dc6413d7729795afbed3251be7630070d7db348fde54bdc10')
EVALUATOR=ROOT/'training/evaluate_unified_retention_source_average.py'
EVALUATOR_SHA='10bd2860f2d2a0dd738a00d29600eb431cf052017cdd4f91afe7364b9ee5c850'
OBJECTIVE_VARIANT='fixed_half_core_and_source_balanced_parameter_average'
METHOD={'kind':'equal_weight_complete_cnn_parameter_average','source_order':['core1000','source_balanced_final_selected'],
    'weights':[.5,.5],'accumulation':'CPU float64: core plus final selected source-balanced state, divide by 2, cast to original dtype',
    'nonfloating_buffers':'Require identical values and preserve original dtype.',
    'candidate_count':1,'ratio_search':False,'subset_search':False,'output_averaging':False,
    'optimizer_steps_executed':0,'source_optimizer_steps_executed':3000,'source_selected_step':1500,
    'core_optimizer_steps_executed':1500,'core_selected_step':1000,
    'declared_after_inspecting_source_calibration':True,'predeclared_before_source_training':False,
    'teacher_in_model':False,'test_read':False,'development_holdout_read':False}


def read(path):
    return json.loads(Path(path).read_text())


def validate_source_documents(selection,protocol,failure,steps,selected_step):
    """Check actual completion and the unchanged original ranking, including failures."""
    require(selection.get('promotion_allowed') is False and failure.get('status')=='NO_PROMOTABLE_CHECKPOINT'
        and failure.get('promotion_allowed') is False and selection.get('optimizer_steps_executed')==steps
        and failure.get('optimizer_steps_executed')==protocol.get('steps')==steps
        and failure.get('selected_step')==selected_step and protocol.get('eval_every')==500
        and protocol.get('batch_size')==96 and selection.get('fixed_runtime')==protocol.get('fixed_runtime')==FIXED_RUNTIME
        and selection.get('bindings')==protocol.get('bindings')
        and all(document.get(key) is False for document in (selection,protocol,failure)
            for key in ('test_read','development_holdout_read')),'Complete failed source training is required')
    for key in set(selection)&set(protocol)-{'schema'}:
        require(selection[key]==protocol[key],'Source selection differs from its training protocol: '+key)
    history=selection['history']
    require([record['step'] for record in history]==list(range(500,steps+1,500))
        and all(record.get('promotion_allowed') is False for record in history)
        and selection['selected']==max(history,key=retention_rank)
        and selection['selected']['step']==selected_step,'Source must use its original rank-selected checkpoint')
    require(selection['passed'] is selection['calibration_passed'] is selection['selected']['metrics']['passed'],
        'Source stable calibration status differs')
    for record in history:
        require_fixed_runtime(record)
        require(len(record['retention_checks'])==46,'Source retention conditions differ')


def validate_sources(data=DATA):
    """Read only JSON and file hashes; never load images, Torch, or held-out arrays."""
    data=Path(data).resolve();require(data==DATA.resolve(),'Use the unchanged original TRAIN/CAL data')
    selections=[];identities=[];bindings={}
    for index,(run,steps,selected_step,trainer) in enumerate(zip(SOURCE_RUNS,(1500,3000),(1000,1500),(core,source))):
        require(sha(run/'model.pth')==CHECKPOINT_SHAS[index] and sha(run/'SELECTION.json')==SELECTION_SHAS[index]
            and sha(trainer.__file__)==TRAINER_SHAS[index],'Fixed source weights, selection, or trainer changed')
        selection=read(run/'SELECTION.json');protocol=read(run/'TRAINING_FREEZE.json');failure=read(run/'report.json')
        validate_source_documents(selection,protocol,failure,steps,selected_step)
        require(sha(run/'TRAINING_FREEZE.json')==selection['training_protocol_sha256']
            and read(run/'NO_PROMOTABLE_CHECKPOINT.json')==failure
            and selection['objective_variant']==trainer.OBJECTIVE_VARIANT
            and selection['objective']==trainer.OBJECTIVE and selection['architecture']==ARCHITECTURE,
            'Source failure, objective, or freeze identity changed')
        for path,digest in selection['bindings'].items():
            require(path not in bindings or bindings[path]==digest,'Conflicting source identity')
            bindings[path]=digest
        for record in selection['history']:
            directory=f"checkpoints/step{record['step']:05d}"
            require(set(record['artifacts'])=={'checkpoint','outputs','decisions','metrics'},'Incomplete source step evidence')
            for key,name in [('checkpoint','model.pth'),('outputs','CALIBRATION_OUTPUTS.npz'),
                ('decisions','CALIBRATION_DECISIONS.json'),('metrics','METRICS.json')]:
                path=checked_file(run,record['artifacts'][key],directory+'/'+name)
                bindings[str(path)]=record['artifacts'][key]['sha256']
                if key=='metrics':require(read(path)==strip_artifacts(record),'Saved source CAL metrics changed')
        require(failure['selected_checkpoint_sha256']==CHECKPOINT_SHAS[index]
            and failure['last_checkpoint_sha256']==selection['history'][-1]['artifacts']['checkpoint']['sha256'],
            'Source execution evidence does not reach the planned final step')
        for key,name,field in [('outputs','CALIBRATION_OUTPUTS.npz','calibration_outputs_sha256'),
            ('decisions','CALIBRATION_DECISIONS.json','calibration_decisions_sha256')]:
            require(sha(run/name)==selection[field]==selection['selected']['artifacts'][key]['sha256'],
                'Source selected CAL bytes differ from its selected step')
        groups=selection.get('selected_parameter_groups_sha256',selection['final_parameter_groups_sha256'])
        require(set(groups)==set(selection['initial_parameter_groups_sha256'])=={'trunk','style','family_head','size_head'}
            and all(groups[k]!=selection['initial_parameter_groups_sha256'][k] for k in groups),
            'Each source selected CNN parameter group must have trained')
        names=['model.pth','SELECTION.json','TRAINING_FREEZE.json','report.json','NO_PROMOTABLE_CHECKPOINT.json',
            'CALIBRATION_OUTPUTS.npz','CALIBRATION_DECISIONS.json','SAMPLING.json']
        names += ['DISTILLATION.json'] if index==0 else ['TRAINING_COUNTS.json']
        for name in names:bindings[str((run/name).resolve())]=sha(run/name)
        identities.append({'role':METHOD['source_order'][index],'run':str(run),
            'checkpoint':{'path':str(run/'model.pth'),'sha256':CHECKPOINT_SHAS[index]},
            'selection':{'path':str(run/'SELECTION.json'),'sha256':SELECTION_SHAS[index]},
            'training_freeze':{'path':str(run/'TRAINING_FREEZE.json'),'sha256':selection['training_protocol_sha256']},
            'selected_step':selected_step,'optimizer_steps_executed':steps,
            'state_sha256':selection['state_after_sha256'],'parameter_groups_sha256':groups})
        selections.append(selection)
    a,b=selections
    require(a['families']==b['families'] and len(a['families'])==len(set(a['families']))==25
        and a['families'].count(core.UNKNOWN)==1 and b['base_checkpoint']['sha256']==CHECKPOINT_SHAS[0]
        and b['base_selection']['sha256']==SELECTION_SHAS[0]
        and b['state_before_sha256']==a['state_after_sha256']
        and b['initial_parameter_groups_sha256']==identities[0]['parameter_groups_sha256']
        and a['retention_plan']==b['retention_plan'] and sha(PLAN)==PLAN_SHA==a['retention_plan']['sha256']
        and a['data_manifest_sha256']==b['data_manifest_sha256']==sha(data/'MANIFEST.json')
        and a['parent_metadata']==b['parent_metadata'] and a['parent_checkpoint']==b['parent_checkpoint'],
        'The adapted model must inherit this exact core and the same original 46-condition plan')
    require(b['unknown_binary_supervision']==source.UNKNOWN_BINARY_SUPERVISION
        and b['sampling']==source.SAMPLING and b['design_changes']==source.DESIGN_CHANGES,
        'Source-balanced training contract changed')
    # These existing pure validators accept completed failed runs and verify true TRAIN sampling evidence.
    from export_unified_retention_source_balanced import validate_counts,validate_supplement
    validate_counts(SOURCE_RUNS[1],b,data);validate_supplement(b,data)
    paths=[Path(__file__),EVALUATOR,ROOT/'training/train_unified_retention.py',ROOT/'training/train_unified_regions.py',
        ROOT/'training/export_unified_retention_core.py',ROOT/'training/export_unified_retention_source_balanced.py',
        ROOT/'training/prepare_unified_regions.py',ROOT/'training/region_network.py',ROOT/'training/network.py',
        ROOT/'training/train_regions.py',ROOT/'src/flux_glyph/region_font.py',ROOT/'src/flux_glyph/unified_font.py',
        ROOT/'artifacts/unified-font-v1/run-v1/DEVELOPMENT_REGRESSION.json',PLAN,data/'MANIFEST.json']
    for split in ('train','calibration'):
        path=data/split/'MANIFEST.json';part=read(path);paths.append(path)
        for key in ('metadata','array'):
            item=(path.parent/part[key]['path']).resolve()
            require(item.parent==path.parent and sha(item)==part[key]['sha256'],'Original prepared partition changed')
            paths.append(item)
    require(sha(EVALUATOR)==EVALUATOR_SHA,'Fixed development evaluation source changed')
    for path in paths:
        path=path.resolve();digest=sha(path)
        require(str(path) not in bindings or bindings[str(path)]==digest,'New averaging dependency changed source identity')
        bindings[str(path)]=digest
    verify_bindings(bindings)
    plan=core.read_plan(PLAN,data,(ROOT/a['parent_checkpoint']['path']).resolve())
    return {'families':a['families'],'plan':plan,'source_selections':selections,'source_checkpoints':identities,
        'bindings':bindings,'data_manifest_sha256':sha(data/'MANIFEST.json'),
        'parent_metadata':a['parent_metadata'],'parent_checkpoint':a['parent_checkpoint']}


def average_states(states):
    import torch
    require(len(states)==2 and all(isinstance(s,dict) and s for s in states),'Exactly two complete states are required')
    require(set(states[0])==set(states[1]),'Source parameter keys differ')
    result={}
    for name,first in states[0].items():
        values=[s[name] for s in states]
        require(isinstance(first,torch.Tensor) and all(isinstance(v,torch.Tensor)
            and v.shape==first.shape and v.dtype==first.dtype and not v.is_complex()
            and bool(torch.isfinite(v).all()) for v in values),'Invalid source shape, dtype, or values: '+name)
        if first.is_floating_point():
            total=values[0].detach().to(device='cpu',dtype=torch.float64).clone()
            total.add_(values[1].detach().to(device='cpu',dtype=torch.float64))
            result[name]=(total/2.).to(dtype=first.dtype)
            require(bool(torch.isfinite(result[name]).all()),'Nonfinite averaged parameter')
        else:
            require(torch.equal(values[0].cpu(),values[1].cpu()),'Nonfloating source buffers differ: '+name)
            result[name]=first.detach().cpu().clone()
    return result


def average_checkpoints(checkpoints,families,expected):
    require(len(checkpoints)==2 and len(families)==len(set(families))==25 and families.count(core.UNKNOWN)==1,
        'Exactly two complete 25-class CNNs are required')
    for checkpoint in checkpoints:
        require(checkpoint.get('architecture')==ARCHITECTURE and checkpoint.get('families')==families
            and set(checkpoint.get('state_dict',{}))==set(expected),'Single CNN architecture or class order differs')
        require(all(checkpoint['state_dict'][key].shape==value.shape and checkpoint['state_dict'][key].dtype==value.dtype
            for key,value in expected.items()),'Checkpoint is not the complete original CNN')
    return average_states([c['state_dict'] for c in checkpoints])


def load_source_states(evidence):
    import torch
    from region_network import RegionFontClassifier
    model=RegionFontClassifier(25);checkpoints=[]
    for identity,selection in zip(evidence['source_checkpoints'],evidence['source_selections']):
        checkpoint=torch.load(identity['checkpoint']['path'],map_location='cpu',weights_only=True)
        require(checkpoint.get('selection_sha256')==identity['selection']['sha256']
            and state_sha(checkpoint['state_dict'])==identity['state_sha256']
            and parameter_groups(checkpoint['state_dict'])==identity['parameter_groups_sha256'],
            'Source state or selected parameter-group identity differs')
        initialize(model,checkpoint,evidence['families'])
        saved=Path(identity['run'])/selection['selected']['artifacts']['checkpoint']['path']
        step=torch.load(saved,map_location='cpu',weights_only=True)
        require(step.get('step')==identity['selected_step'] and step.get('training_protocol_sha256')==selection['training_protocol_sha256']
            and step.get('families')==evidence['families'] and step.get('architecture')==ARCHITECTURE
            and state_sha(step['state_dict'])==identity['state_sha256'],'Source final file differs from original selected step')
        if identity['role']=='source_balanced_final_selected':
            last=torch.load(Path(identity['run'])/selection['history'][-1]['artifacts']['checkpoint']['path'],map_location='cpu',weights_only=True)
            require(last['step']==3000 and last['training_protocol_sha256']==selection['training_protocol_sha256']
                and parameter_groups(last['state_dict'])==selection['final_parameter_groups_sha256'],
                'Source training final-step group evidence differs')
        checkpoints.append(checkpoint)
    return model,checkpoints


def infer_after_freeze(model,cal,device,freeze_path,freeze_sha,checkpoint_path,checkpoint_sha):
    require(Path(freeze_path).is_file() and sha(freeze_path)==freeze_sha and Path(checkpoint_path).is_file()
        and sha(checkpoint_path)==checkpoint_sha,'Saved average and freeze must precede CAL inference')
    return core.infer(model,cal,device)


def run(args):
    import torch
    output=args.output.resolve();require(not output.exists(),'Preserve previous averages; output must be new')
    evidence=validate_sources();torch.set_num_threads(4)
    model,checkpoints=load_source_states(evidence)
    averaged=average_checkpoints(checkpoints,evidence['families'],model.state_dict());model.load_state_dict(averaged,strict=True)
    state=state_sha(averaged);groups=parameter_groups(averaged)
    require(all(state!=r['state_sha256'] for r in evidence['source_checkpoints']),'Average must be a distinct new state')
    freeze={key:copy.deepcopy(evidence[key]) for key in ('families','source_checkpoints','bindings','data_manifest_sha256','parent_metadata','parent_checkpoint')}
    freeze.update(schema='flux-glyph-unified-retention-source-averaging-freeze-v1',architecture=ARCHITECTURE,
        objective_variant=OBJECTIVE_VARIANT,method=METHOD,retention_plan={'path':str(PLAN),'sha256':PLAN_SHA},
        source_training_objectives=[s['objective'] for s in evidence['source_selections']],
        fixed_runtime=FIXED_RUNTIME,runtime_gates_changed=False,state_sha256=state,parameter_groups_sha256=groups,
        optimizer_steps_executed=0,source_optimizer_steps_executed=3000,source_selected_step=1500,
        core_optimizer_steps_executed=1500,core_selected_step=1000,model_count=1,encoder_count=1,
        teacher_in_deployed_model=False,platform_routing=False,score_merging=False,
        calibration_device=args.device,candidate_step=0,selection_search_performed=False,
        test_read=False,development_holdout_read=False,
        scope='One new fixed parameter-average candidate declared after inspecting failed source CAL; no optimizer, ratio search, or independent test.')
    verify_bindings(freeze['bindings']);output.mkdir(parents=True);dump(output/'AVERAGING_FREEZE.json',freeze)
    freeze_sha=sha(output/'AVERAGING_FREEZE.json');checkpoint_path=output/'model.pth'
    torch.save({'state_dict':averaged,'families':evidence['families'],'architecture':ARCHITECTURE,'step':0,
        'averaging_freeze_sha256':freeze_sha,'optimizer_steps_executed':0,'source_optimizer_steps_executed':3000},checkpoint_path)
    checkpoint_sha=sha(checkpoint_path)
    cal=core.load_split(DATA,'calibration')
    require(cal['families']==evidence['families'] and cal['manifest_sha256']==freeze['data_manifest_sha256'],'CAL source changed')
    model.eval().to(args.device)
    logits,ratios=infer_after_freeze(model,cal,args.device,output/'AVERAGING_FREEZE.json',freeze_sha,checkpoint_path,checkpoint_sha)
    record,outputs,_=evaluate_outputs(logits,ratios,cal,evidence['plan'],0)
    require(len(record['retention_checks'])==46,'Original 46 conditions changed')
    verify_bindings(freeze['bindings'])
    require(sha(output/'AVERAGING_FREEZE.json')==freeze_sha and sha(checkpoint_path)==checkpoint_sha
        and state_sha(model.state_dict())==state,'Average weights or evidence changed during CAL inference')
    np.savez(output/'CALIBRATION_OUTPUTS.npz',logits=logits,log_em_ratio=ratios)
    dump(output/'CALIBRATION_DECISIONS.json',{'families':evidence['families'],'records':outputs});dump(output/'RESULT.json',record)
    selection={**freeze,'schema':'flux-glyph-unified-retention-source-average-selection-v1','selected':record,'history':[record],
        'promotion_allowed':record['promotion_allowed'],'passed':record['metrics']['passed'],'calibration_passed':record['metrics']['passed'],
        'averaging_freeze_sha256':freeze_sha,'checkpoint_sha256':checkpoint_sha,
        'calibration_outputs_sha256':sha(output/'CALIBRATION_OUTPUTS.npz'),
        'calibration_decisions_sha256':sha(output/'CALIBRATION_DECISIONS.json'),'result_sha256':sha(output/'RESULT.json')}
    dump(output/'SELECTION.json',selection)
    report={key:copy.deepcopy(selection[key]) for key in ('objective_variant','method','optimizer_steps_executed',
        'source_optimizer_steps_executed','source_selected_step','core_optimizer_steps_executed','core_selected_step',
        'state_sha256','parameter_groups_sha256','promotion_allowed','calibration_passed','averaging_freeze_sha256',
        'checkpoint_sha256','calibration_outputs_sha256','calibration_decisions_sha256','result_sha256','fixed_runtime')}
    report.update(schema='flux-glyph-unified-retention-source-average-result-v1',
        status='PROMOTABLE_CANDIDATE' if record['promotion_allowed'] else 'NO_PROMOTABLE_CHECKPOINT',
        selection_sha256=sha(output/'SELECTION.json'),stable_validation_passed=False,test_passed=False,
        calibration_inference_completed=True,calibration_inference_passes=1,runtime_gates_changed=False,
        test_read=False,development_holdout_read=False,selected_step=0,
        retention_checks_passed=sum(r['passed'] for r in record['retention_checks']),retention_checks_total=46)
    dump(output/'report.json',report)
    if not record['promotion_allowed']:dump(output/'NO_PROMOTABLE_CHECKPOINT.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True);return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/unified-font-v2/average-source-balanced-v1')
    p.add_argument('--device',choices=('mps','cpu'),default='mps')
    return p


if __name__=='__main__':run(parser().parse_args())
