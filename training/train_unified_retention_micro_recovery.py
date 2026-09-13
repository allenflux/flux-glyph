#!/usr/bin/env python3
"""Train the candidate20 CNN with extra true-Micro TRAIN cross entropy."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
import math
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'training')]
import train_unified_retention_core as core
from train_unified_retention import RetentionSampler,evaluate_outputs,retention_rank,FIXED_RUNTIME,SAMPLING
from train_unified_retention_adapter import (BASE_CHECKPOINT_SHA,BASE_SELECTION_SHA,
    read,cache_identity,load_cache)
from train_unified_retention import ARCHITECTURE,BATCH_SIZE,parameter_groups,initialize,infer
from train_regions import require,sha,dump,state_sha
from retention_paired_known_sampler import KNOWN_FAMILIES,KNOWN_SUPPLEMENT_MARKER
from retention_face_balanced_sampler import FaceBalancedSampler,SAMPLING
from retention_supplement_sampler import SUPPLEMENT_MARKER,NEW_SOURCES
from prepare_unified_unknown_supplement import load_supplement
from cache_unified_unknown_supplement import load_cache as load_supplement_cache
from prepare_unified_known_supplement import load_supplement as load_known
from cache_unified_known_supplement import load_cache as load_known_cache
import train_unified_retention_source_balanced as previous
import train_unified_retention_confidence_floor as narrow
import train_unified_retention_wide as oldwide
import train_unified_retention_wenkai_recovery as wenkai_trial
from retention_dual_teacher_loss import select_teacher, correct_teacher_kl
from retention_r21_teacher_cache import load_r21_train_logits, TEACHER_POLICY, REPORT_SHA as R21_REPORT_SHA
from merge_unified_weight_pairs import load_merged
from cache_unified_weight_teacher import load_cache as load_weight_teacher_cache
from retention_angular_margin_loss import ANGULAR_MARGIN
from retention_wenkai_recovery_loss import (wenkai_recovery_extra_ce,
    WENKAI_RECOVERY_WEIGHTING,WENKAI_FAMILY)
from retention_micro_recovery_loss import (micro_recovery_extra_ce,
    MICRO_RECOVERY_WEIGHTING,MICRO_FAMILY)

CACHE_MANIFEST_SHA='c3d3e57cf4d8377174e4b7981bbc1d30ee0e72af547f41f2f96a9a674998974a'
ARCHITECTURE="region-cnn64x256-unified-wide-v1"
STEPS=6000
EVAL_EVERY=6000
LEARNING_RATE=2e-5
MINIMUM_LEARNING_RATE=2e-6
SEED=2026091414
from retention_confidence_floor_loss import unknown_floor_loss
UNKNOWN_FLOOR_SUPERVISION={'minimum_unknown_probability':.5,
    'loss':'relu(softplus(-unknown_log_odds) - log(2))',
    'log_odds':'z_unknown - logsumexp(z_named)',
    'applies_to':'TRAIN rows whose true class is __unknown__',
    'named_conditional_distribution_target':None,'known_label_smoothing':.03,
    'labels_unchanged':True,'inference_rule':False,'penalizes_confidence_above_floor':False}
OBJECTIVE_VARIANT='wide_cnn_face_balanced_angular_margin_wenkai_micro_ce_recovery'
DESIGN_CHANGES=[
    'Add exactly 0.25 extra label-smoothed family CE for true WenQuanYi Micro Hei TRAIN rows, raising their family CE weight from 1 to 1.25.',
    'Keep candidate 20 WenKai CE weight 2, full eligible-teacher KL, angular auxiliary, native-core weights, unknown floor, size loss, b08-to-wide initializer, face-balanced replay, data, schedule, architecture and runtime fixed.']
OBJECTIVE={**wenkai_trial.OBJECTIVE,'micro_recovery_weighting':MICRO_RECOVERY_WEIGHTING}
TEACHER_CACHE_SHA='04e9f19a525f9b681f2f86d46bfcb2c2cad30e7e178a3858c68b4575f2b1d8ea'
STUDENT_CHECKPOINT_SHA='b08c0e20e98835e55094debedcad495d579d478b8f793aa733601383bec87763'
STUDENT_SELECTION_SHA='14825ee2c994aeb1a387f590f95ff4354efe6323ab92891230a47c69837002d8'
KNOWN_MANIFEST_SHA='4caff363a62c4df7f58dd3f63fe271dff7c0e6a91952cd5bff7958e78d60e357'
KNOWN_CACHE_SHA='bef47534a7088bb1cb8af2d13dfb540b1d02e406279b39ea519f0000fd3d3e5e'
WEIGHT_PAIRS_MANIFEST_SHA='ec9b616538444741742f91414c7e66ae51bdf036c3956fdca9d6bdbbb7873b83'
WEIGHT_TEACHER_CACHE_MANIFEST_SHA='17732d05f0102c6e384a4d234afdfc363adb94987ec300085485882e7593ef41'
EXPECTED_TRAINING_DATA_COUNTS={'original_views':70128,'supplement_views':1918,
    'known_supplement_views':2880,'combined_views':74926,
    'original_native_regions':19803,'supplement_native_regions':480,
    'known_supplement_native_regions':720,'combined_native_regions':21003,
    'original_tiles':130854,'supplement_tiles':3156,'known_supplement_tiles':4820,
    'combined_tiles':138830,'calibration_unchanged':True,'derived_views_are_correlated':True}
SUPPLEMENT_MANIFEST_SHA='43aac66dfc3ce3a09fd0acd088dcda82450a7203997befd1e35aebe016713b05'
SUPPLEMENT_CACHE_SHA='4c9114a8f49b7bbdb54d5c915a67643c6e4bef581ca195ecfe6264bf251e434b'


def read_objective_plan(path, acceptance_plan_sha256, weight_teacher_cache_sha256):
    plan=read(path)
    require(weight_teacher_cache_sha256==WEIGHT_TEACHER_CACHE_MANIFEST_SHA,
        'Weight teacher cache manifest SHA must be concrete before plan validation')
    expected={'schema':'flux-glyph-wide-micro-recovery-plan-v1',
        'architecture':ARCHITECTURE,'design_changes':DESIGN_CHANGES,
        'source_initializer_sha256':STUDENT_CHECKPOINT_SHA,'teacher_cache_sha256':TEACHER_CACHE_SHA,
        'objective':OBJECTIVE,'steps':STEPS,'eval_every':EVAL_EVERY,'seed':SEED,
        'weight_pairs_manifest_sha256':WEIGHT_PAIRS_MANIFEST_SHA,
        'weight_teacher_cache_manifest_sha256':weight_teacher_cache_sha256,
        'runtime':FIXED_RUNTIME,'acceptance_plan_sha256':acceptance_plan_sha256,
        'r21_inference_report_sha256':R21_REPORT_SHA,'teacher_policy':TEACHER_POLICY,
        'checkpoint_selection':'fixed final step 6000, no intermediate candidate selection',
        'calibration_used_in_prior_development':True,'blind_test':False,
        'new_training_started':False,'single_cause_claim':False}
    require(all(plan.get(key)==value for key,value in expected.items()),
        'Pretraining objective plan differs from the frozen loss or original acceptance rules')
    return plan


def forward_with_features(model,tiles):
    """Run the deployable encoder once and expose its existing style features."""
    features=model.style(model.pool(model.trunk(tiles)))
    return model.family_head(features),model.size_head(features).squeeze(-1),features


def full_losses(logits,ratios,features,family_head_weight,targets,sizes,teacher_logits,rows,families):
    import torch
    base_total,base_ce,size_loss,preservation,auxiliary,mask=wenkai_trial.full_losses(
        logits,ratios,features,family_head_weight,targets,sizes,teacher_logits,rows,families)
    recovery=micro_recovery_extra_ce(logits,targets,rows,families)
    ce=base_ce+recovery;total=base_total+recovery
    require(bool(torch.isfinite(total)),'Nonfinite wide-teacher training loss')
    return total,ce,size_loss,preservation,auxiliary,mask


def batch_source(data,cache):
    require(data['partition']['split']=='train' and len(data['tiles'])==len(cache['base_logits'])
            and cache['base_logits'].shape==(len(data['tiles']),25),'TRAIN image/teacher tile order differs')
    index={}
    for row in data['rows']:
        key=(row['source_id'],row['region_id'],row['view'])
        require(key not in index and row['split']=='train' and SUPPLEMENT_MARKER not in row and KNOWN_SUPPLEMENT_MARKER not in row,'Invalid or duplicate TRAIN source region')
        index[key]=row
    return {'tiles':data['tiles'],'teacher_logits':cache['base_logits'],'index':index}


def pixel_batch(rows,old,new,known,rng,return_indices=False):
    require(len(rows)==BATCH_SIZE,'Full-CNN sampling requires 96 TRAIN regions')
    images=[];teachers=[];sizes=[];sampled_indices=[]
    for row in rows:
        marker=row.get(SUPPLEMENT_MARKER,False);positive=row.get(KNOWN_SUPPLEMENT_MARKER,False)
        require(type(marker) is bool and type(positive) is bool and not (marker and positive),'Invalid or conflicting supplement markers')
        source_name='known' if positive else 'supplement' if marker else 'original'
        source=known if positive else new if marker else old
        raw={k:v for k,v in row.items() if k not in (SUPPLEMENT_MARKER,KNOWN_SUPPLEMENT_MARKER)}
        key=(row.get('source_id'),row.get('region_id'),row.get('view'))
        require(source['index'].get(key)==raw and raw.get('split')=='train'
                and raw.get('native_font_verified') is True,'TRAIN row was relabeled or routed to the wrong image/teacher cache')
        start,count=row['tile_start'],row['tile_count']
        require(type(start) is int and type(count) is int and 1<=count<=8
                and 0<=start<start+count<=len(source['tiles'])
                and len(source['tiles'])==len(source['teacher_logits']),'TRAIN tile index escapes its paired cache')
        index=start+int(rng.integers(count))
        images.append(source['tiles'][index]);teachers.append(source['teacher_logits'][index]);sizes.append(row['log_em_ratio'])
        sampled_indices.append((source_name,index))
    images=np.asarray(images,dtype=np.float32);teachers=np.asarray(teachers,dtype=np.float32);sizes=np.asarray(sizes,dtype=np.float32)
    require(images.shape==(96,1,64,256) and teachers.shape==(96,25) and sizes.shape==(96,)
            and all(np.isfinite(x).all() for x in (images,teachers,sizes))
            and ((images>=0)&(images<=1)).all() and (np.abs(sizes)<=3).all(), 'Invalid TRAIN pixels or cached teacher/native size targets')
    return (images,teachers,sizes,sampled_indices) if return_indices else (images,teachers,sizes)


def expected_unknown_counts(source_order,slots):
    require(isinstance(source_order,list) and len(source_order)==len(set(source_order))==11
            and all(isinstance(name,str) and name for name in source_order)
            and source_order==sorted(source_order) and set(NEW_SOURCES)<=set(source_order)
            and type(slots) is int and slots>=0,'Invalid 11-source unknown cycle')
    quotient,remainder=divmod(slots,len(source_order))
    return {source:quotient+int(i<remainder) for i,source in enumerate(source_order)}


class TruthCorrectCounter(core.DistillationCounter):
    def update(self,rows,mask):
        require(len(rows)==len(mask)==96 and all(type(v) is bool for v in mask), 'Invalid teacher mask accounting')
        weights=core.native_core_weights(rows,[r['target'] for r in rows],self.families)
        for row,eligible,weight in zip(rows,mask,weights):
            family=row['family'];domain=row['domain'];source=row.get('source_font_family') or family
            key=(domain,source,family);self.family[family]+=1;self.domain[domain]+=1;self.sources[key]+=1
            if eligible:self.eligible_family[family]+=1;self.eligible_domain[domain]+=1;self.eligible_sources[key]+=1
            if weight==2.:self.core_family[family]+=1;self.core_source[key]+=1
        self.steps+=1;self.eligible_per_step.append(sum(mask));self.core_per_step.append(sum(w==2. for w in weights))


class FullSupplementCounts(TruthCorrectCounter):
    def __init__(self,families,source_order):
        super().__init__(families);expected_unknown_counts(source_order,0)
        self.unknown_source_order=list(source_order);self.unknown_sources=Counter()
        self.supplement_rows=0;self.supplement_sources=Counter()
        self.known_rows=0;self.known_sources=Counter();self.known_per_step=[];self.known_regions=Counter()
        self.known_faces=Counter();self.eligible_known_faces=Counter();self.known_tile_indices=Counter()
        self.named_teacher_rows=0;self.unknown_teacher_rows=0
        self.wenkai_recovery_rows=0;self.wenkai_recovery_per_step=[]
        self.wenkai_recovery_source_faces=Counter()
        self.micro_recovery_rows=0;self.micro_recovery_per_step=[]
        self.micro_recovery_source_faces=Counter()
    def update(self,rows,mask,teacher_predictions,sampled_indices=None):
        require(len(rows)==len(mask)==len(teacher_predictions)==96
            and all(type(v) is bool for v in mask)
            and all(type(p) is int and 0<=p<len(self.families) for p in teacher_predictions)
            and mask==[p==r['target'] for p,r in zip(teacher_predictions,rows)],
            'Teacher eligibility must equal the actual same-tile argmax versus TRAIN truth')
        unknown=[r for r in rows if r['family']==core.UNKNOWN]
        before=expected_unknown_counts(self.unknown_source_order,self.steps*16)
        after=expected_unknown_counts(self.unknown_source_order,(self.steps+1)*16)
        expected={source:after[source]-before[source] for source in self.unknown_source_order}
        require(len(unknown)==16 and Counter(r['source_font_family'] for r in unknown)==expected,
                'Unknown sources do not follow the fixed continuous 11-source cycle')
        for row in rows:
            marker=row.get(SUPPLEMENT_MARKER,False)
            positive=row.get(KNOWN_SUPPLEMENT_MARKER,False)
            require(type(marker) is bool and type(positive) is bool and not (marker and positive)
                    and marker==(row['family']==core.UNKNOWN and row['source_font_family'] in NEW_SOURCES),
                    'New/old TRAIN marker differs from the declared source family')
        known=[r for r in rows if r.get(KNOWN_SUPPLEMENT_MARKER)]
        require(len(known)==2 and Counter(r['family'] for r in known)==dict.fromkeys(KNOWN_FAMILIES,1)
            and all(r['domain']=='android' and r['split']=='train' and r['source_font_family']==r['family']
                and r.get('source_dataset')=='android_paired_known_supplement'
                and r.get('pair_evidence',{}).get('intentional_train_text_pair') is True for r in known),
            'Known supplement must provide one true paired TRAIN row per declared class')
        extra=[r for r in rows if r.get(SUPPLEMENT_MARKER)]
        require(2<=len(extra)<=4,'New-source cycle differs')
        super().update(rows,mask);self.supplement_rows+=len(extra)
        recovery=[row for row in rows if row['family']==WENKAI_FAMILY]
        self.wenkai_recovery_rows+=len(recovery);self.wenkai_recovery_per_step.append(len(recovery))
        for row in recovery:
            key=(row['domain'],row.get('source_font_family') or row['family'],row.get('font_face'))
            require(all(isinstance(value,str) and value for value in key),
                'Weighted WenKai TRAIN source/face identity is incomplete')
            self.wenkai_recovery_source_faces[key]+=1
        micro=[row for row in rows if row['family']==MICRO_FAMILY]
        self.micro_recovery_rows+=len(micro);self.micro_recovery_per_step.append(len(micro))
        for row in micro:
            key=(row['domain'],row.get('source_font_family') or row['family'],row.get('font_face'))
            require(all(isinstance(value,str) and value for value in key),
                'Weighted Micro TRAIN source/face identity is incomplete')
            self.micro_recovery_source_faces[key]+=1
        self.supplement_sources.update(r['source_font_family'] for r in extra)
        self.unknown_sources.update(r['source_font_family'] for r in unknown)
        self.known_rows+=len(known);self.known_sources.update(r['family'] for r in known);self.known_per_step.append(len(known))
        self.known_regions.update((r['source_id'],r['region_id'],r['view'],r['family']) for r in known)
        self.named_teacher_rows+=sum(r['target']!=24 for r in rows)
        self.unknown_teacher_rows+=sum(r['target']==24 for r in rows)
        for row,eligible in zip(rows,mask):
            if row.get(KNOWN_SUPPLEMENT_MARKER):
                key=(row['family'],row['font_face']);self.known_faces[key]+=1
                if eligible:self.eligible_known_faces[key]+=1
        if sampled_indices is not None:
            require(len(sampled_indices)==96,
                'Sampled cache indices must identify each exact TRAIN teacher tile')
            for row,(name,index) in zip(rows,sampled_indices):
                expected='known' if row.get(KNOWN_SUPPLEMENT_MARKER,False) else (
                    'supplement' if row.get(SUPPLEMENT_MARKER,False) else 'original')
                require(name==expected and type(index) is int
                    and row['tile_start']<=index<row['tile_start']+row['tile_count'],
                    'Sampled cache index drifted from its exact routed TRAIN row')
            self.known_tile_indices.update(index for name,index in sampled_indices if name=='known')
    def report(self):
        return {**super().report(),'schema':'flux-glyph-retention-micro-recovery-counts-v1',
            'objective_variant':OBJECTIVE_VARIANT,'mask_rule':OBJECTIVE['teacher_mask'],'objective':OBJECTIVE,
            'angular_margin':ANGULAR_MARGIN,'angular_margin_applies_at_inference':False,
            'wenkai_recovery_weighting':WENKAI_RECOVERY_WEIGHTING,
            'wenkai_recovery_rows':self.wenkai_recovery_rows,
            'wenkai_recovery_rows_per_step':self.wenkai_recovery_per_step,
            'wenkai_recovery_source_face_rows':[{'domain':domain,'source_font_family':source,
                'font_face':face,'rows':count} for (domain,source,face),count in
                sorted(self.wenkai_recovery_source_faces.items())],
            'micro_recovery_weighting':MICRO_RECOVERY_WEIGHTING,
            'micro_recovery_rows':self.micro_recovery_rows,
            'micro_recovery_rows_per_step':self.micro_recovery_per_step,
            'micro_recovery_source_face_rows':[{'domain':domain,'source_font_family':source,
                'font_face':face,'rows':count} for (domain,source,face),count in
                sorted(self.micro_recovery_source_faces.items())],
            'unknown_floor_supervision':UNKNOWN_FLOOR_SUPERVISION,
            'unknown_source_order':self.unknown_source_order,'unknown_rows':self.steps*16,
            'unknown_source_rows':dict(self.unknown_sources),
            'supplement_rows':self.supplement_rows,'supplement_source_rows':dict(self.supplement_sources),
            'known_supplement_rows':self.known_rows,'known_supplement_source_rows':dict(self.known_sources),
            'known_supplement_rows_per_step':self.known_per_step,
            'known_supplement_region_rows':[{'source_id':s,'region_id':r,'view':v,'family':f,'rows':n}
                for (s,r,v,f),n in sorted(self.known_regions.items())],
            'known_supplement_face_rows':[{'family':f,'font_face':face,'rows':n,
                'eligible_rows':self.eligible_known_faces[(f,face)]}
                for (f,face),n in sorted(self.known_faces.items())],
            'known_teacher_tile_index_counts':[{'tile_index':index,'rows':count}
                for index,count in sorted(self.known_tile_indices.items())],
            'selected_teacher_true_target_rows':{'named_b08':self.named_teacher_rows,
                'unknown_r21':self.unknown_teacher_rows},
            'original_rows':self.steps*96-self.supplement_rows-self.known_rows,'teacher_logits_used':True,
            'teacher_mask_verified_against_same_tile_argmax':True,
            'second_model_resident':False,'teacher_optimizer_steps':0,'teacher_cache_deployed':False}


def validate_training_counts(report,families,steps):
    expected=expected_unknown_counts(report.get('unknown_source_order'),steps*16)
    extra={source:expected[source] for source in NEW_SOURCES}
    require(report.get('schema')=='flux-glyph-retention-micro-recovery-counts-v1'
            and report.get('objective_variant')==OBJECTIVE_VARIANT and report.get('objective')==OBJECTIVE
            and report.get('angular_margin')==ANGULAR_MARGIN
            and report.get('angular_margin_applies_at_inference') is False
            and report.get('wenkai_recovery_weighting')==WENKAI_RECOVERY_WEIGHTING
            and report.get('micro_recovery_weighting')==MICRO_RECOVERY_WEIGHTING
            and report.get('unknown_floor_supervision')==UNKNOWN_FLOOR_SUPERVISION
            and report.get('mask_rule')==OBJECTIVE['teacher_mask'] and report.get('steps')==steps
            and report.get('rows')==steps*96 and report.get('supplement_rows')==sum(extra.values())
            and report.get('original_rows')==steps*94-sum(extra.values())
            and report.get('supplement_source_rows')==extra and report.get('unknown_rows')==steps*16
            and report.get('unknown_source_rows')==expected
            and report.get('teacher_logits_used') is True and report.get('second_model_resident') is False
            and report.get('teacher_optimizer_steps')==0 and report.get('teacher_cache_deployed') is False
            and report.get('teacher_mask_verified_against_same_tile_argmax') is True,
            'Full-CNN cached teacher or new TRAIN sample accounting differs')
    require(report.get('training_split')=='train' and report.get('test_read') is False
            and report.get('development_holdout_read') is False
            and report.get('native_core_weighting')==core.NATIVE_CORE_WEIGHTING,
            'Teacher evidence has invalid TRAIN scope or native weighting')
    eligible=report['eligible_rows_per_step'];weighted=report['native_core_weighted_rows_per_step']
    require(len(eligible)==len(weighted)==steps
            and all(type(n) is int and 0<=n<=96 for n in eligible)
            and all(type(n) is int and 27<=n<=33 for n in weighted)
            and sum(eligible)==report['eligible_rows'] and sum(weighted)==report['native_core_weighted_rows'],
            'Per-step mask or native core totals differ')
    totals=Counter();domain=Counter();selected=Counter();selected_domain=Counter();source_lookup={}
    for row in report['source_target_rows']:
        key=(row['domain'],row['source_font_family'],row['target_family']);n=row['rows'];e=row['eligible_rows']
        require(key not in source_lookup and key[0] in ('ios','android') and key[2] in families
                and isinstance(key[1],str) and key[1] and type(n) is int and type(e) is int and 0<=e<=n,'Invalid teacher source population')
        source_lookup[key]=n;totals[key[2]]+=n;domain[key[0]]+=n;selected[key[2]]+=e;selected_domain[key[0]]+=e
    require(dict(totals)==report['family_rows'] and dict(domain)==report['domain_rows'] and sum(totals.values())==steps*96
            and sum(selected.values())==report['eligible_rows']
            and set(report['eligible_family_rows'])<=set(families) and set(report['eligible_domain_rows'])<={'ios','android'}
            and all(selected[f]==report['eligible_family_rows'].get(f,0) for f in families)
            and all(selected_domain[d]==report['eligible_domain_rows'].get(d,0) for d in ('ios','android')),
            'Teacher source and family/domain totals differ')
    core_counts=Counter();seen=set()
    for row in report['native_core_source_rows']:
        key=(row['domain'],row['source_font_family'],row['target_family']);n=row['rows']
        require(key not in seen and key in source_lookup and key[0]=='ios' and key[2] in core.CORE_FAMILIES
                and type(n) is int and 0<n<=source_lookup[key],'Invalid native core weighted source count')
        seen.add(key);core_counts[key[2]]+=n
    require(dict(core_counts)==report['native_core_weighted_family_rows']
            and sum(core_counts.values())==report['native_core_weighted_rows'],'Native core totals differ')
    recovery_per_step=report.get('wenkai_recovery_rows_per_step')
    recovery_sources={};recovery_faces=Counter()
    for row in report.get('wenkai_recovery_source_face_rows',[]):
        key=(row.get('domain'),row.get('source_font_family'),row.get('font_face'));n=row.get('rows')
        require(key not in recovery_sources and key[0] in ('ios','android')
                and all(isinstance(value,str) and value for value in key[1:])
                and type(n) is int and n>0,'Invalid weighted WenKai source/face accounting')
        recovery_sources[key]=n;recovery_faces[key[2]]+=n
    recovery_total=report.get('wenkai_recovery_rows')
    require(isinstance(recovery_per_step,list) and len(recovery_per_step)==steps
            and all(type(n) is int and 0<=n<=96 for n in recovery_per_step)
            and type(recovery_total) is int and recovery_total==sum(recovery_per_step)
            and recovery_total==sum(recovery_sources.values())==report['family_rows'][WENKAI_FAMILY]
            and WENKAI_FAMILY not in core.CORE_FAMILIES,
            'Weighted WenKai counts differ from true TRAIN family/source rows')
    if steps==STEPS:
        require(recovery_total==12000 and dict(recovery_faces)=={
            'LXGWWenKai-Light':4000,'LXGWWenKai-Medium':4000,'LXGWWenKai-Regular':4000},
            'Fixed replay must weight exactly 4000 true WenKai rows per face')
    micro_per_step=report.get('micro_recovery_rows_per_step');micro_sources={};micro_faces=Counter()
    for row in report.get('micro_recovery_source_face_rows',[]):
        key=(row.get('domain'),row.get('source_font_family'),row.get('font_face'));n=row.get('rows')
        require(key not in micro_sources and key[0] in ('ios','android')
                and all(isinstance(value,str) and value for value in key[1:])
                and type(n) is int and n>0,'Invalid weighted Micro source/face accounting')
        micro_sources[key]=n;micro_faces[key[2]]+=n
    micro_total=report.get('micro_recovery_rows')
    require(isinstance(micro_per_step,list) and len(micro_per_step)==steps
            and all(type(n) is int and n>=0 for n in micro_per_step)
            and type(micro_total) is int and micro_total==sum(micro_per_step)
            and micro_total==sum(micro_sources.values())==report['family_rows'][MICRO_FAMILY]
            and MICRO_FAMILY not in core.CORE_FAMILIES,
            'Weighted Micro counts differ from true TRAIN family/source rows')
    if steps==STEPS:
        require(micro_per_step==[2]*STEPS and micro_total==12000 and micro_sources=={
            ('android','WenQuanYi Micro Hei','WenQuanYiMicroHei'):12000},
            'Fixed replay must weight exactly two true Micro rows per step from the verified face')
    unknown_totals=Counter()
    for row in report['source_target_rows']:
        if row['target_family']==core.UNKNOWN:unknown_totals[row['source_font_family']]+=row['rows']
    require(dict(unknown_totals)==expected,'Reported unknown source totals differ from the exact cycle')
    sources={r['source_font_family']:r for r in report['source_target_rows'] if r['source_font_family'] in NEW_SOURCES}
    require(set(sources)==set(NEW_SOURCES) and all(r['rows']==extra[source] and r['target_family']==core.UNKNOWN
            and r['domain']=='android' for source,r in sources.items()),'New unknown source count/target differs')
    require(report.get('known_supplement_rows')==steps*2
        and report.get('known_supplement_source_rows')==dict.fromkeys(KNOWN_FAMILIES,steps)
        and report.get('known_supplement_rows_per_step')==[2]*steps,'Paired known row allocation differs')
    known_counts=Counter();seen=set()
    for row in report.get('known_supplement_region_rows',[]):
        key=(row['source_id'],row['region_id'],row['view'],row['family']);n=row['rows']
        require(key not in seen and all(isinstance(k,str) and k for k in key) and key[3] in KNOWN_FAMILIES
            and key[2] in SAMPLING['supplement_view_weights'] and type(n) is int and n>0,
            'Invalid paired known region evidence')
        seen.add(key);known_counts[key[3]]+=n
    require(dict(known_counts)==dict.fromkeys(KNOWN_FAMILIES,steps),'Paired known per-region counts differ')
    face_counts=Counter();eligible_face_counts=Counter()
    for row in report.get('known_supplement_face_rows',[]):
        key=(row.get('family'),row.get('font_face'));n=row.get('rows');eligible=row.get('eligible_rows')
        require(key not in face_counts and key[0] in KNOWN_FAMILIES and isinstance(key[1],str)
            and type(n) is int and type(eligible) is int and 0<=eligible<=n,
            'Invalid paired known face/mask accounting')
        face_counts[key]=n;eligible_face_counts[key]=eligible
    require(sum(face_counts.values())==steps*2 and report.get('selected_teacher_true_target_rows')
        =={'named_b08':steps*80,'unknown_r21':steps*16},'True-target teacher source counts differ')
    if steps==STEPS:
        require(dict(face_counts)=={('LXGW WenKai','LXGWWenKai-Light'):2000,
            ('LXGW WenKai','LXGWWenKai-Medium'):2000,
            ('LXGW WenKai','LXGWWenKai-Regular'):2000,
            ('WenQuanYi Micro Hei','WenQuanYiMicroHei'):6000},
            'Sampled paired teacher masks do not cover the exact face cycle')
    tile_counts=report.get('known_teacher_tile_index_counts')
    tile_total=sum(row.get('rows',0) for row in tile_counts) if isinstance(tile_counts,list) else -1
    require(isinstance(tile_counts,list) and tile_total in (0,steps*2)
        and (steps!=STEPS or tile_total==steps*2)
        and len({row.get('tile_index') for row in tile_counts})==len(tile_counts)
        and all(type(row.get('tile_index')) is int and row['tile_index']>=0
            and (steps!=STEPS or row['tile_index']<4820)
            and type(row.get('rows')) is int and row['rows']>0 for row in tile_counts),
        'Merged known teacher cache tile-index counts differ')
    return report


def validate_sampling_report(report,steps):
    require(steps==STEPS and report.get('schema')=='flux-glyph-retention-face-balanced-sampling-v1'
        and report.get('known_supplement_tile_count')==4820
        and sum(report.get('family_rows',{}).values())==steps*96,
        'Face-balanced sampler report scope differs')
    paired={(row.get('family'),row.get('font_face')):row.get('rows')
        for row in report.get('known_supplement_face_rows',[])}
    require(paired=={('LXGW WenKai','LXGWWenKai-Light'):2000,
        ('LXGW WenKai','LXGWWenKai-Medium'):2000,
        ('LXGW WenKai','LXGWWenKai-Regular'):2000,
        ('WenQuanYi Micro Hei','WenQuanYiMicroHei'):6000},
        'Paired known 6000-step face cycle differs')
    totals=Counter()
    for row in report.get('face_rows',[]):
        if row.get('family')=='LXGW WenKai':totals[row.get('font_face')]+=row.get('rows',0)
    require({face:totals[face] for face in ('LXGWWenKai-Light','LXGWWenKai-Medium','LXGWWenKai-Regular')}
        ==dict.fromkeys(('LXGWWenKai-Light','LXGWWenKai-Medium','LXGWWenKai-Regular'),4000),
        'Total WenKai face exposure differs from 4000 rows per face')
    return report


load_student_initializer = narrow.load_student_initializer
load_training_teacher = narrow.load_training_teacher


def mixed_teacher_cache(datasets, named_cache, unknown_cache):
    import torch
    require(set(datasets)==set(named_cache)==set(unknown_cache)=={'original','supplement','known'},
            'Mixed teachers require the complete three-source TRAIN union')
    arrays={};proof={}
    for name,data in datasets.items():
        core.registry(data['families'])
        offset=0
        for row in data['rows']:
            target=row.get('target');count=row.get('tile_count')
            require(row.get('split')==row.get('original_split')=='train'
                    and row.get('native_font_verified') is True
                    and type(target) is int and 0<=target<25 and row.get('family')==data['families'][target]
                    and type(row.get('tile_start')) is int and row['tile_start']==offset
                    and type(count) is int and 1<=count<=8 and offset+count<=len(data['tiles']),
                    'Mixed teacher labels must cover exact contiguous verified TRAIN tile order')
            offset+=count
        require(offset==len(data['tiles']),'Mixed teacher metadata leaves TRAIN tiles unclaimed')
        targets=np.concatenate([np.full(row['tile_count'],row['target'],dtype=np.int64) for row in data['rows']])
        named=torch.from_numpy(np.array(named_cache[name]['base_logits'],copy=True))
        unknown=torch.from_numpy(np.array(unknown_cache[name],copy=True))
        selected,mask=select_teacher(named,unknown,torch.from_numpy(targets),24)
        values=selected.numpy()
        require(values.shape==(len(data['tiles']),25) and np.isfinite(values).all(),'Invalid same-tile mixed teacher cache')
        arrays[name]={'base_logits':values}
        proof[name]={'tiles':len(values),'named_teacher_tiles':int((~mask).sum()),
                     'unknown_teacher_tiles':int(mask.sum()),'logits_sha256':hashlib.sha256(values.tobytes()).hexdigest(),
                     'labels_sha256':hashlib.sha256(targets.tobytes()).hexdigest()}
    return arrays,proof


def merged_known_teacher_cache(data, cache):
    """Bind recomputed b08 logits to every exact merged known TRAIN tile."""
    core.registry(data['families'])
    require(set(cache)=={'base_logits'},'Merged known cache must contain only b08 logits')
    offset=0;labels=[]
    for row in data['rows']:
        target=row.get('target');count=row.get('tile_count')
        require(row.get('split')==row.get('original_split')=='train'
                and row.get('native_font_verified') is True and target in (10,11)
                and row.get('family')==data['families'][target]
                and row.get('tile_start')==offset and type(count) is int and 1<=count<=8,
                'Merged known labels must cover exact verified TRAIN tile order')
        labels.extend([target]*count);offset+=count
    values=cache['base_logits']
    require(offset==len(data['tiles']) and isinstance(values,np.ndarray)
            and values.dtype==np.float32 and values.shape==(offset,25)
            and bool(np.isfinite(values).all()),'Merged b08 cache differs from exact known TRAIN tiles')
    labels=np.asarray(labels,dtype=np.int64)
    return {'base_logits':values},{'tiles':offset,'named_teacher_tiles':offset,
        'unknown_teacher_tiles':0,'logits_sha256':hashlib.sha256(values.tobytes()).hexdigest(),
        'labels_sha256':hashlib.sha256(labels.tobytes()).hexdigest()}


def training_data_identity(training,supplement,known,historical_known,expected=None):
    """Separate actual optimizer inputs from the overlapping historical known witness."""
    regions=lambda data:{(row['source_id'],row['region_id']) for row in data['rows']}
    original_regions,new_regions,known_regions,historical_regions=map(
        regions,(training,supplement,known,historical_known))
    require(not original_regions&new_regions and not known_regions&(original_regions|new_regions)
        and historical_regions<known_regions,
        'Actual TRAIN sources must be distinct and merged known must contain the historical paired source')
    counts={'original_views':len(training['rows']),'supplement_views':len(supplement['rows']),
        'known_supplement_views':len(known['rows']),
        'combined_views':len(training['rows'])+len(supplement['rows'])+len(known['rows']),
        'original_native_regions':len(original_regions),'supplement_native_regions':len(new_regions),
        'known_supplement_native_regions':len(known_regions),
        'combined_native_regions':len(original_regions)+len(new_regions)+len(known_regions),
        'original_tiles':len(training['tiles']),'supplement_tiles':len(supplement['tiles']),
        'known_supplement_tiles':len(known['tiles']),
        'combined_tiles':len(training['tiles'])+len(supplement['tiles'])+len(known['tiles']),
        'calibration_unchanged':True,'derived_views_are_correlated':True}
    if expected is not None:require(counts==expected,'Actual face-balanced TRAIN data counts differ')
    return counts


def verify_widening(source, model, datasets, device):
    import torch
    before_source=state_sha(source.state_dict());before_wide=state_sha(model.state_dict())
    rng=np.random.default_rng(SEED);indices={};max_logits=0.;max_size=0.;tile_hash=hashlib.sha256()
    source.eval().to(device);model.eval().to(device)
    with torch.inference_mode():
        for name,data in datasets.items():
            chosen=rng.choice(len(data['tiles']),128,replace=False);indices[name]=chosen.tolist()
            for start in range(0,128,32):
                block=np.array(data['tiles'][chosen[start:start+32]],copy=True)
                tile_hash.update(block.tobytes());x=torch.from_numpy(block).to(device)
                a,b=source(x);c,d=model(x)
                torch.testing.assert_close(c,a,atol=3e-4,rtol=3e-4)
                torch.testing.assert_close(d,b,atol=3e-4,rtol=3e-4)
                max_logits=max(max_logits,float((a-c).abs().max().cpu()))
                max_size=max(max_size,float((b-d).abs().max().cpu()))
    require(state_sha(source.cpu().state_dict())==before_source and state_sha(model.cpu().state_dict())==before_wide,
            'Widening validation changed source or widened parameters')
    return {'passed':True,'device':device,'train_tiles':384,'indices':indices,'tiles_sha256':tile_hash.hexdigest(),
            'logits_max_abs_error':max_logits,'log_em_ratio_max_abs_error':max_size,
            'atol':3e-4,'rtol':3e-4,'source_state_sha256':before_source,'wide_state_sha256':before_wide,
            'source_parameters_unchanged':True,'wide_parameters_unchanged':True,
            'optimizer_steps':0,'calibration_read':False,'development_read':False,'test_read':False}


def train(args):
    import torch
    from region_network import RegionFontClassifier
    from wide_region_network import widen_region_model
    args.data=args.data.resolve();args.output=args.output.resolve();args.checkpoint=args.checkpoint.resolve();args.plan=args.plan.resolve()
    args.objective_plan=args.objective_plan.resolve()
    args.weight_pairs=args.weight_pairs.resolve();args.weight_teacher_cache=args.weight_teacher_cache.resolve()
    weight_teacher_cache_manifest_sha=sha(args.weight_teacher_cache/'CACHE_MANIFEST.json')
    require(weight_teacher_cache_manifest_sha==WEIGHT_TEACHER_CACHE_MANIFEST_SHA,
        'Use only the frozen recomputed merged b08 teacher cache')
    read_objective_plan(args.objective_plan,sha(args.plan),weight_teacher_cache_manifest_sha)
    args.cache=args.cache.resolve()
    require(sha(args.cache/'CACHE_MANIFEST.json')==CACHE_MANIFEST_SHA,'Use only the already frozen core feature cache')
    args.feature_device=read(args.cache/'CACHE_MANIFEST.json')['identity']['feature_device']
    require(not args.output.exists() and args.steps==STEPS and args.seed==SEED,'Full-CNN run needs a new output and fixed 6000 steps/seed')
    require(sha(args.checkpoint)==BASE_CHECKPOINT_SHA and sha(args.checkpoint.parent/'SELECTION.json')==BASE_SELECTION_SHA,
            'Frozen core step-1000 base checkpoint differs')
    source=read(args.checkpoint.parent/'SELECTION.json');source_freeze=read(args.checkpoint.parent/'TRAINING_FREEZE.json')
    require(source['selected']['step']==1000 and source['optimizer_steps_executed']==1500 and source.get('promotion_allowed') is False
            and source['objective_variant']==core.OBJECTIVE_VARIANT and source['families']==source_freeze['families']
            and sha(args.checkpoint.parent/'TRAINING_FREEZE.json')==source['training_protocol_sha256']
            and source['test_read'] is False and source['development_holdout_read'] is False,'Invalid frozen core initializer lineage')
    core.verify_bindings(source['bindings'])
    plan=core.read_plan(args.plan,args.data,(ROOT/source['parent_checkpoint']['path']).resolve())
    datasets={split:core.load_split(args.data,split) for split in ('train','calibration')};training=datasets['train'];cal=datasets['calibration']
    families=training['families'];require(families==source['families']==plan['families'],'Full-CNN family order differs')
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    require(checkpoint.get('selection_sha256')==BASE_SELECTION_SHA and checkpoint.get('families')==families
            and checkpoint.get('architecture')==core.ARCHITECTURE and state_sha(checkpoint['state_dict'])==source['state_after_sha256'],
            'Base checkpoint state or architecture differs')
    model=RegionFontClassifier(25)
    inheritance=initialize(model,checkpoint,families)
    base_before=state_sha(model.state_dict());initial_groups=parameter_groups(model.state_dict())
    full_before=base_before
    require(base_before==source['state_after_sha256'] and all(p.requires_grad for p in model.parameters())
            and not any(k.startswith('residual') for k in model.state_dict()),
            'Full-CNN did not inherit/train exactly the plain 25-class core model')
    base_identity={'path':str(args.checkpoint),'sha256':BASE_CHECKPOINT_SHA};base_selection={'path':str(args.checkpoint.parent/'SELECTION.json'),'sha256':BASE_SELECTION_SHA}
    bindings=dict(source['bindings'])
    paths=[Path(__file__),ROOT/'training/retention_adapter_network.py',ROOT/'training/train_unified_retention_core.py',
        ROOT/'training/train_unified_retention.py',ROOT/'training/train_unified_regions.py',ROOT/'training/prepare_unified_regions.py',
        ROOT/'training/region_network.py',ROOT/'training/network.py',ROOT/'training/train_regions.py',
        ROOT/'src/flux_glyph/unified_font.py',ROOT/'src/flux_glyph/region_font.py',ROOT/'training/evaluate_unified_retention_micro_recovery.py',ROOT/'training/retention_supplement_sampler.py',ROOT/'training/retention_source_balanced_sampler.py',ROOT/'training/retention_paired_known_sampler.py',ROOT/'training/train_unified_retention_source_balanced.py',
        ROOT/'training/prepare_unified_unknown_supplement.py',
        ROOT/'training/train_unified_retention_adapter.py',ROOT/'training/train_unified_retention_paired_known.py',
        ROOT/'training/cache_unified_student_teacher.py',ROOT/'training/retention_confidence_floor_loss.py',
        ROOT/'training/train_unified_retention_confidence_floor.py',ROOT/'training/retention_r21_teacher_cache.py',
        ROOT/'training/retention_dual_teacher_loss.py',ROOT/'training/wide_region_network.py',
        ROOT/'training/train_unified_retention_wide.py',
        ROOT/'training/retention_face_balanced_sampler.py',ROOT/'training/train_unified_retention_face_balanced.py',
        ROOT/'training/retention_micro_recovery_loss.py',ROOT/'training/train_unified_retention_wenkai_recovery.py',
        ROOT/'training/retention_wenkai_recovery_loss.py',
        ROOT/'training/train_unified_retention_margin.py',
        ROOT/'training/retention_angular_margin_loss.py',ROOT/'training/merge_unified_weight_pairs.py',
        ROOT/'training/cache_unified_weight_teacher.py',
        ROOT/'artifacts/unified-font-v1/run-v1/DEVELOPMENT_REGRESSION.json',args.checkpoint,args.plan,args.objective_plan,
        args.checkpoint.parent/'SELECTION.json',args.checkpoint.parent/'TRAINING_FREEZE.json',
        args.checkpoint.parent/'CALIBRATION_OUTPUTS.npz',args.checkpoint.parent/'CALIBRATION_DECISIONS.json',args.data/'MANIFEST.json']
    for split in ('train','calibration'):
        part=read(args.data/split/'MANIFEST.json');paths.append(args.data/split/'MANIFEST.json')
        for key in ('array','metadata'):
            path=(args.data/split/part[key]['path']).resolve()
            require(path.parent==args.data/split and sha(path)==part[key]['sha256'],'Full-CNN data changed')
            paths.append(path)
    for path in paths:
        path=path.resolve();digest=sha(path)
        require(str(path) not in bindings or bindings[str(path)]==digest,'Full-CNN source binding conflict');bindings[str(path)]=digest
    cached_identity=read(args.cache/'CACHE_MANIFEST.json')['identity']
    identity=cache_identity(datasets,base_identity,base_before,cached_identity['bindings'],args.feature_device)
    require(identity==cached_identity,'Reused cache does not match actual TRAIN/CAL tiles, row order or base model')
    cache,cache_manifest=load_cache(args.cache,identity)
    for path,digest in identity['bindings'].items():
        require(path not in bindings or bindings[path]==digest,'Original cache and new training source conflict')
        bindings[path]=digest
    args.supplement=args.supplement.resolve();args.supplement_cache=args.supplement_cache.resolve()
    supplement_manifest_path=args.supplement/'MANIFEST.json'
    supplement_cache_path=args.supplement_cache/'CACHE_MANIFEST.json'
    require(sha(supplement_manifest_path)==SUPPLEMENT_MANIFEST_SHA
            and sha(supplement_cache_path)==SUPPLEMENT_CACHE_SHA,'Frozen new native TRAIN supplement differs')
    supplement=load_supplement(args.supplement)
    supplement_cache,supplement_cache_manifest=load_supplement_cache(args.supplement_cache)
    extra_identity=supplement_cache_manifest['identity']
    require(supplement['families']==families==extra_identity['families']
            and extra_identity['data_manifest']['sha256']==supplement['manifest_sha256']==SUPPLEMENT_MANIFEST_SHA
            and extra_identity['base_checkpoint']['sha256']==BASE_CHECKPOINT_SHA
            and extra_identity['base_state_sha256']==base_before
            and extra_identity['partition']['split']=='train'
            and len(supplement_cache['features'])==len(supplement['tiles']),
            'Supplement features do not come from the same frozen base and verified TRAIN partition')
    for path,digest in extra_identity['bindings'].items():
        require(path not in bindings or bindings[path]==digest,'Supplement source conflicts with frozen original evidence')
        require(sha(path)==digest,'Supplement source binding changed');bindings[path]=digest
    extra_paths=[Path(__file__),ROOT/'training/retention_supplement_sampler.py',ROOT/'training/retention_source_balanced_sampler.py',ROOT/'training/retention_paired_known_sampler.py',ROOT/'training/train_unified_retention_source_balanced.py',
        ROOT/'training/prepare_unified_unknown_supplement.py',ROOT/'training/cache_unified_unknown_supplement.py',
        supplement_manifest_path,args.supplement/'PREPARATION_FREEZE.json',args.supplement/'train/MANIFEST.json',
        supplement_cache_path,args.supplement_cache/'CACHE_FREEZE.json']
    extra_paths.extend(args.supplement/'train'/supplement['partition'][key]['path'] for key in ('metadata','array'))
    extra_paths.extend(args.supplement_cache/'train'/(name+'.npy') for name in ('features','base_logits','log_em_ratio'))
    for path in extra_paths:
        path=path.resolve();digest=sha(path)
        require(str(path) not in bindings or bindings[str(path)]==digest,'Supplement file changed');bindings[str(path)]=digest
    supplement_identity={'path':str(supplement_manifest_path),'sha256':SUPPLEMENT_MANIFEST_SHA}
    supplement_cache_identity={'path':str(supplement_cache_path),'sha256':SUPPLEMENT_CACHE_SHA}
    args.known=args.known.resolve();args.known_cache=args.known_cache.resolve()
    known_manifest_path=args.known/'MANIFEST.json';known_cache_path=args.known_cache/'CACHE_MANIFEST.json'
    require(sha(known_manifest_path)==KNOWN_MANIFEST_SHA and sha(known_cache_path)==KNOWN_CACHE_SHA,
        'Use only the frozen paired known capture and historical core cache')
    historical_known=load_known(args.known)
    historical_known_cache,historical_known_cache_manifest=load_known_cache(args.known_cache)
    known_cache_source=historical_known_cache_manifest['identity']
    require(historical_known['families']==families==known_cache_source['families']
        and known_cache_source['data_manifest']['sha256']==historical_known['manifest_sha256']==KNOWN_MANIFEST_SHA
        and known_cache_source['base_checkpoint']['sha256']==BASE_CHECKPOINT_SHA
        and known_cache_source['base_state_sha256']==base_before and known_cache_source['partition']['split']=='train'
        and len(historical_known_cache['base_logits'])==len(historical_known['tiles']),
        'Historical known pixels and core cache identity differ')
    for path,digest in known_cache_source['bindings'].items():
        require(path not in bindings or bindings[path]==digest,'Known proof conflicts with original source identity')
        require(sha(path)==digest,'Known pair evidence changed');bindings[path]=digest
    known_paths=[ROOT/'training/prepare_unified_known_supplement.py',ROOT/'training/cache_unified_known_supplement.py',
        known_manifest_path,args.known/'PREPARATION_FREEZE.json',args.known/'train/MANIFEST.json',
        known_cache_path,args.known_cache/'CACHE_FREEZE.json']
    known_paths.extend(args.known/'train'/historical_known['partition'][key]['path'] for key in ('metadata','array'))
    known_paths.extend(args.known_cache/'train'/(name+'.npy') for name in ('features','base_logits','log_em_ratio'))
    for path in known_paths:
        path=path.resolve();digest=sha(path)
        require(str(path) not in bindings or bindings[str(path)]==digest,'Known cache or data changed');bindings[str(path)]=digest
    historical_known_identity={'path':str(known_manifest_path),'sha256':KNOWN_MANIFEST_SHA}
    historical_known_cache_identity={'path':str(known_cache_path),'sha256':KNOWN_CACHE_SHA}
    known=load_merged(args.weight_pairs)
    require(known['families']==families and known['manifest_sha256']==WEIGHT_PAIRS_MANIFEST_SHA
        and known['partition']['native_regions']==720 and len(known['rows'])==2880
        and len(known['tiles'])==4820,'Actual merged known TRAIN root differs')
    for path in (args.weight_pairs/'MANIFEST.json',args.weight_pairs/'PREPARATION_FREEZE.json',
            args.weight_pairs/'train/MANIFEST.json',args.weight_pairs/'train/rows.json',
            args.weight_pairs/'train/tiles.raw',args.weight_pairs/'train/SOURCE_MAPPING.json'):
        path=path.resolve();digest=sha(path)
        require(str(path) not in bindings or bindings[str(path)]==digest,'Merged known source conflicts')
        bindings[str(path)]=digest
    for path,digest in known['manifest']['bindings'].items():
        require(path not in bindings or bindings[path]==digest,'Merged native proof closure conflicts')
        require(sha(path)==digest,'Merged native proof closure changed');bindings[path]=digest
    known_identity={'path':str(args.weight_pairs/'MANIFEST.json'),'sha256':WEIGHT_PAIRS_MANIFEST_SHA}
    training_data_counts=training_data_identity(
        training,supplement,known,historical_known,EXPECTED_TRAINING_DATA_COUNTS)
    model.cpu();require(state_sha(model.state_dict())==full_before,'Cache validation changed initial student parameters')
    student_identity,inheritance,student_bindings,student_selection=load_student_initializer(args.initializer,model,families)
    for path,digest in student_bindings.items():
        require(path not in bindings or bindings[path]==digest,'Prior student and historical core source closures conflict')
        bindings[path]=digest
    require(state_sha(model.state_dict())==student_identity['state_sha256']!=base_before,
            'Narrow student initializer and historical core identities were conflated')
    source_model=model
    model,widening=widen_region_model(source_model,seed=SEED)
    training_sources={'original':training,'supplement':supplement,'known':known}
    historical_training_sources={'original':training,'supplement':supplement,'known':historical_known}
    widening_parity=verify_widening(source_model,model,training_sources,args.device)
    del source_model
    full_before=state_sha(model.state_dict());initial_groups=parameter_groups(model.state_dict())
    inheritance={'kind':'function_preserving_width_doubling','source_inheritance':inheritance,
                 'source_state_sha256':student_identity['state_sha256'],'initial_state_sha256':full_before}
    named_cache,teacher_cache_manifest,teacher_cache_identity=load_training_teacher(
        args.teacher_cache,historical_training_sources,student_identity,bindings)
    unknown_cache,unknown_teacher_identity=load_r21_train_logits(historical_training_sources,bindings)
    historical_teacher_cache,historical_teacher_mix=oldwide.mixed_teacher_cache(
        historical_training_sources,named_cache,unknown_cache)
    weight_cache,_,weight_cache_identity=load_weight_teacher_cache(
        args.weight_teacher_cache,known,student_identity,bindings)
    known_teacher,known_teacher_mix=merged_known_teacher_cache(known,weight_cache)
    teacher_cache={'original':historical_teacher_cache['original'],
        'supplement':historical_teacher_cache['supplement'],'known':known_teacher}
    teacher_mix={'original':historical_teacher_mix['original'],
        'supplement':historical_teacher_mix['supplement'],'known':known_teacher_mix}
    # MPS features/logits/size are frozen training artifacts, not deployable lookup data.
    cache_path=args.cache/'CACHE_MANIFEST.json';cache_binding={'path':str(cache_path),'sha256':sha(cache_path)}
    for path in (args.cache/'CACHE_FREEZE.json',cache_path):bindings[str(path)]=sha(path)
    for part in cache_manifest['partitions'].values():
        for entry in part.values():bindings[str((args.cache/entry['path']).resolve())]=entry['sha256']
    require(sha(args.checkpoint.parent/'CALIBRATION_OUTPUTS.npz')==source['calibration_outputs_sha256'], 'Source base CAL changed')
    with np.load(args.checkpoint.parent/'CALIBRATION_OUTPUTS.npz',allow_pickle=False) as original:
        np.testing.assert_allclose(cache['calibration']['base_logits'],original['logits'],atol=3e-4,rtol=3e-4)
        np.testing.assert_allclose(cache['calibration']['log_em_ratio'],original['log_em_ratio'],atol=3e-4,rtol=3e-4)
    baseline,base_outputs,_=evaluate_outputs(cache['calibration']['base_logits'],cache['calibration']['log_em_ratio'],cal,plan,0)
    args.output.mkdir(parents=True,exist_ok=True)
    dump(args.output/'BASELINE.json',baseline);dump(args.output/'BASELINE_CALIBRATION_DECISIONS.json',{'families':families,'records':base_outputs})
    student_run=Path(student_identity['checkpoint']['path']).parent
    require(sha(student_run/'CALIBRATION_OUTPUTS.npz')==student_selection['calibration_outputs_sha256'],
        'Student cached CAL baseline changed')
    with np.load(student_run/'CALIBRATION_OUTPUTS.npz',allow_pickle=False) as initial:
        student_baseline,student_outputs,_=evaluate_outputs(initial['logits'],initial['log_em_ratio'],cal,plan,0)
    dump(args.output/'STUDENT_BASELINE.json',student_baseline)
    dump(args.output/'STUDENT_BASELINE_CALIBRATION_DECISIONS.json',{'families':families,'records':student_outputs})
    baseline_bindings={name:sha(args.output/name) for name in ('BASELINE.json','BASELINE_CALIBRATION_DECISIONS.json',
        'STUDENT_BASELINE.json','STUDENT_BASELINE_CALIBRATION_DECISIONS.json')}
    sampler=FaceBalancedSampler(training['rows'],families,args.seed,plan['train_pools'],supplement['rows'],known['rows'])
    protocol={'schema':'flux-glyph-unified-retention-wide-micro-recovery-protocol-v1','architecture':ARCHITECTURE,'families':families,
        'objective_variant':OBJECTIVE_VARIANT,'objective':OBJECTIVE,'unknown_floor_supervision':UNKNOWN_FLOOR_SUPERVISION,
        'angular_margin':ANGULAR_MARGIN,'angular_margin_applies_at_inference':False,
        'angular_margin_adds_deployment_parameters':False,
        'wenkai_recovery_weighting':WENKAI_RECOVERY_WEIGHTING,
        'wenkai_recovery_applies_at_inference':False,
        'wenkai_recovery_adds_deployment_parameters':False,
        'micro_recovery_weighting':MICRO_RECOVERY_WEIGHTING,
        'micro_recovery_applies_at_inference':False,
        'micro_recovery_adds_deployment_parameters':False,
        'sampling':SAMPLING,'fixed_runtime':FIXED_RUNTIME,'unknown_source_order':sampler.unknown_source_order,
        'design_changes':DESIGN_CHANGES,'single_change_causal_attribution':False,
        'feature_cache_reused':False,'teacher_cache_reused':True,'cached_feature_extraction_performed':False,
        'supplement_data':supplement_identity,'supplement_cache':supplement_cache_identity,
        'historical_known_data':historical_known_identity,
        'historical_known_cache':historical_known_cache_identity,
        'historical_known_data_used_for_sampling':False,
        'historical_known_cache_used_for_optimizer':False,
        'known_data':known_identity,'known_cache':weight_cache_identity,
        'known_cache_kind':'merged_known_b08_logits',
        'training_data_counts':training_data_counts,
        'steps':STEPS,'eval_every':EVAL_EVERY,'batch_size':BATCH_SIZE,'learning_rate':LEARNING_RATE,'minimum_learning_rate':MINIMUM_LEARNING_RATE,
        'weight_decay':1e-4,'gradient_clip_norm':5.,'seed':args.seed,'training_device':args.device,'feature_device':args.feature_device,
        'base_checkpoint':base_identity,'base_selection':base_selection,'base_state_sha256':base_before,
        'base_selected_step':1000,'core_optimizer_steps_executed':1500,'source_optimizer_steps_executed':3000,'source_selected_step':1500,'student_initializer':student_identity,
        'optimizer_steps_executed':STEPS,'initial_state_sha256':full_before,'state_before_sha256':full_before,
        'initial_parameter_groups_sha256':initial_groups,'inheritance':inheritance,
        'named_teacher_state_sha256':student_identity['state_sha256'],
        'named_teacher_identity':student_identity,'named_teacher_cache':teacher_cache_identity,
        'named_teacher_cache_used_partitions':['original','supplement'],
        'unknown_teacher_identity':unknown_teacher_identity,
        'unknown_teacher_used_partitions':['original','supplement'],
        'merged_known_teacher_cache':weight_cache_identity,
        'teacher_policy':TEACHER_POLICY,'teacher_mix':teacher_mix,
        'historical_teacher_mix':historical_teacher_mix,
        'widening':widening,'widening_parity':widening_parity,'offline_teacher_count':2,
        'teacher_models_resident_during_optimizer':0,'initial_calibration_baseline':'Frozen narrow b08 outputs; widened initializer checked only on TRAIN',
        'historical_core_caches_used_for_training':False,'cache_manifest':cache_binding,'bindings':bindings,
        'data_manifest_sha256':training['manifest_sha256'],'retention_plan':{'path':str(args.plan),'sha256':sha(args.plan)},
        'objective_plan':{'path':str(args.objective_plan),'sha256':sha(args.objective_plan)},
        'parent_metadata':source['parent_metadata'],'parent_checkpoint':source['parent_checkpoint'],
        'parent_selection_sha256':source['parent_selection_sha256'],
        'baseline_bindings':baseline_bindings,
        'frozen_groups':[],'trainable_groups':['trunk','style','family_head','size_head'],
        'all_parameters_trained':True,'base_frozen':False,'size_head_frozen':False,'residual_head_trained':False,
        'model_count':1,'encoder_count':1,'platform_routing':False,'score_merging':False,
        'teacher_logits_used':True,'second_model_resident':False,'teacher_optimizer_steps':0,'teacher_cache_deployed':False,
        'test_read':False,'development_holdout_read':False,'runtime_gates_searched':False,'training_inputs':['image_tiles'],
        'calibration_execution':'Current complete CNN forward over all original CAL image tiles; current learned size head.',
        'checkpoint_selection':'Fixed final step 6000; no intermediate CAL inference or checkpoint search',
        'calibration_reused_for_prior_development':True}
    core.verify_bindings(bindings);dump(args.output/'TRAINING_FREEZE.json',protocol);protocol_sha=sha(args.output/'TRAINING_FREEZE.json')
    counts=FullSupplementCounts(families,sampler.unknown_source_order)
    old_batch=batch_source(training,teacher_cache['original']);new_batch=batch_source(supplement,teacher_cache['supplement']);known_batch=batch_source(known,teacher_cache['known'])
    model.to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=MINIMUM_LEARNING_RATE)
    best=None;history=[];started=time.monotonic()
    for step in range(1,STEPS+1):
        rows=sampler.batch()
        image_values,base_values,size_values,sampled_indices=pixel_batch(
            rows,old_batch,new_batch,known_batch,sampler.rng,return_indices=True)
        images=torch.from_numpy(image_values).to(args.device);base=torch.from_numpy(base_values).to(args.device)
        targets=torch.tensor([r['target'] for r in rows],device=args.device);sizes=torch.from_numpy(size_values).to(args.device)
        model.train();logits,ratios,features=forward_with_features(model,images)
        loss,ce,size_loss,preservation,auxiliary,mask=full_losses(
            logits,ratios,features,model.family_head.weight,targets,sizes,base,rows,families)
        wenkai_recovery=wenkai_recovery_extra_ce(logits,targets,rows,families).detach()
        micro_recovery=micro_recovery_extra_ce(logits,targets,rows,families).detach()
        counts.update(rows,mask.detach().cpu().tolist(),base.argmax(1).detach().cpu().tolist(),sampled_indices)
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step();scheduler.step()
        if step%100==0:print(json.dumps({'step':step,'loss':float(loss.detach()),'ce':float(ce.detach()),'teacher_kl':float(preservation.detach()),'size_loss':float(size_loss.detach()),'angular_margin_weighted_loss':float(auxiliary.detach()),'wenkai_recovery_extra_ce':float(wenkai_recovery.detach()),'micro_recovery_extra_ce':float(micro_recovery.detach()),'seconds':round(time.monotonic()-started,1)}),flush=True)
        if step%EVAL_EVERY==0:
            logits,ratios=infer(model,cal,args.device);record,outputs,_=evaluate_outputs(logits,ratios,cal,plan,step)
            directory=args.output/'checkpoints'/f'step{step:05d}';directory.mkdir(parents=True)
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({'state_dict':state,'families':families,'architecture':ARCHITECTURE,'step':step,'training_protocol_sha256':protocol_sha},directory/'model.pth')
            np.savez(directory/'CALIBRATION_OUTPUTS.npz',logits=logits,log_em_ratio=ratios)
            dump(directory/'CALIBRATION_DECISIONS.json',{'families':families,'records':outputs});dump(directory/'METRICS.json',record)
            record['artifacts']={key:{'path':str(path.relative_to(args.output)),'sha256':sha(path)} for key,path in [('checkpoint',directory/'model.pth'),('outputs',directory/'CALIBRATION_OUTPUTS.npz'),('decisions',directory/'CALIBRATION_DECISIONS.json'),('metrics',directory/'METRICS.json')]}
            history.append(record)
            if best is None or retention_rank(record)>retention_rank(best[0]):best=(copy.deepcopy(record),state)
            dump(args.output/'CAL_PROGRESS.json',history);dump(args.output/'TRAINING_COUNTS_PROGRESS.json',counts.report())
            print(json.dumps({'calibration_step':step,'promotion_allowed':record['promotion_allowed'],'checks_passed':sum(r['passed'] for r in record['retention_checks']),'deficit':record['retention_deficit']}),flush=True)
    selected_groups=parameter_groups(best[1]);final_groups=parameter_groups(model.state_dict())
    require(all(selected_groups[key]!=initial_groups[key] and final_groups[key]!=initial_groups[key] for key in initial_groups)
            and all(p.requires_grad for p in model.parameters()),'A full-CNN parameter group did not train')
    core.verify_bindings(bindings)
    require(sha(args.output/'TRAINING_FREEZE.json')==protocol_sha and all(sha(args.output/name)==digest for name,digest in baseline_bindings.items()),'Full-CNN protocol or baseline changed')
    for record in history:require(all(sha(args.output/item['path'])==item['sha256'] for item in record['artifacts'].values()),'Saved full-CNN checkpoint evidence changed')
    selected=best[0];sampling_report=sampler.report();validate_sampling_report(sampling_report,STEPS)
    dump(args.output/'SAMPLING.json',sampling_report)
    training_counts=counts.report();validate_training_counts(training_counts,families,STEPS)
    dump(args.output/'TRAINING_COUNTS.json',training_counts)
    for key,name in [('outputs','CALIBRATION_OUTPUTS.npz'),('decisions','CALIBRATION_DECISIONS.json')]:
        (args.output/name).write_bytes((args.output/selected['artifacts'][key]['path']).read_bytes())
    selection={**protocol,'schema':'flux-glyph-unified-retention-wide-micro-recovery-selection-v1','selected':selected,'history':history,
        'promotion_allowed':selected['promotion_allowed'],'passed':selected['metrics']['passed'],'calibration_passed':selected['metrics']['passed'],
        'training_protocol_sha256':protocol_sha,'state_after_sha256':state_sha(best[1]),
        'selected_parameter_groups_sha256':selected_groups,'final_parameter_groups_sha256':final_groups,
        'sampling_sha256':sha(args.output/'SAMPLING.json'),
        'training_counts_sha256':sha(args.output/'TRAINING_COUNTS.json'),
        'calibration_outputs_sha256':sha(args.output/'CALIBRATION_OUTPUTS.npz'),'calibration_decisions_sha256':sha(args.output/'CALIBRATION_DECISIONS.json')}
    dump(args.output/'SELECTION.json',selection)
    torch.save({'state_dict':best[1],'families':families,'architecture':ARCHITECTURE,'selection_sha256':sha(args.output/'SELECTION.json')},args.output/'model.pth')
    report={'status':'PROMOTABLE_CHECKPOINT' if selected['promotion_allowed'] else 'NO_PROMOTABLE_CHECKPOINT',
        'promotion_allowed':selected['promotion_allowed'],'calibration_passed':selected['metrics']['passed'],
        'optimizer_steps_executed':STEPS,'source_optimizer_steps_executed':3000,'source_selected_step':1500,'core_optimizer_steps_executed':1500,
        'supplement_training_rows_executed':training_counts['supplement_rows'],'known_supplement_training_rows_executed':training_counts['known_supplement_rows'],'training_data_counts':training_data_counts,
        'base_selected_step':1000,'selected_step':selected['step'],'all_parameter_groups_changed':True,
        'all_parameters_trained':True,'teacher_logits_used':True,'second_model_resident':False,'teacher_optimizer_steps':0,
        'teacher_cache_deployed':False,'model_count':1,'runtime_gates_changed':False,
        'last_checkpoint_sha256':history[-1]['artifacts']['checkpoint']['sha256'],'selected_checkpoint_sha256':sha(args.output/'model.pth'),
        'test_read':False,'development_holdout_read':False}
    dump(args.output/'report.json',report)
    if not selected['promotion_allowed']:dump(args.output/'NO_PROMOTABLE_CHECKPOINT.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'artifacts/unified-font-v1/data-v1')
    p.add_argument('--plan',type=Path,default=ROOT/'artifacts/unified-font-v2/RETENTION_PLAN.json')
    p.add_argument('--objective-plan',type=Path,default=ROOT/'artifacts/unified-font-v3/micro-recovery-plan-v1/PLAN.json')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/unified-font-v2/run-core-v1/model.pth')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/unified-font-v3/run-wide-micro-recovery-v1')
    p.add_argument('--cache',type=Path,default=ROOT/'artifacts/unified-font-v2/run-adapter-v1/cache')
    p.add_argument('--supplement',type=Path,default=ROOT/'artifacts/unified-font-v2/new-unknown-capture-v1/data')
    p.add_argument('--supplement-cache',type=Path,default=ROOT/'artifacts/unified-font-v2/new-unknown-capture-v1/cache')
    p.add_argument('--initializer',type=Path,default=ROOT/'artifacts/unified-font-v2/run-source-balanced-v1/model.pth')
    p.add_argument('--known',type=Path,default=ROOT/'artifacts/unified-font-v2/paired-known-capture-v1/data')
    p.add_argument('--known-cache',type=Path,default=ROOT/'artifacts/unified-font-v2/paired-known-capture-v1/cache')
    p.add_argument('--weight-pairs',type=Path,default=ROOT/'artifacts/unified-font-v3/paired-wenkai-weight-capture-v1/merged-data')
    p.add_argument('--weight-teacher-cache',type=Path,default=ROOT/'artifacts/unified-font-v3/paired-wenkai-weight-capture-v1/teacher-cache')
    p.add_argument('--teacher-cache',type=Path,default=ROOT/'artifacts/unified-font-v2/teacher-preserve-cache-v1')
    p.add_argument('--steps',type=int,default=STEPS);p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--device',choices=('cpu','mps'),default='mps')
    return p


if __name__=='__main__':train(parser().parse_args())
