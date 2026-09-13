"""Paired-known release lineage, actual region counts and teacher-cache contracts."""
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from training import export_unified_retention_paired_known as module
import train_unified_retention_paired_known as trainer
import test_unified_retention_source_balanced_export as previous
import test_cache_unified_known_supplement as known_cache_tests
from prepare_unified_regions import FAMILIES


def fixture_student(monkeypatch, tmp_path):
    run, _, source, protocol, *_ = previous.fixture_run(monkeypatch, tmp_path)
    monkeypatch.setattr(module, 'ROOT', run.parent)
    for record in source['history']:
        record['promotion_allowed'] = False
        record['retention_deficit'] = .1 if record['step'] == 1500 else .2
        metrics = run/record['artifacts']['metrics']['path']
        module.dump(metrics, module.strip_artifacts(record))
        record['artifacts']['metrics']['sha256'] = module.sha(metrics)
    source.update(selected=deepcopy(source['history'][2]), promotion_allowed=False, state_after_sha256='d'*64)
    (run/'model.pth').write_bytes(b'The fixed prior rank-selected student')
    for key, name, field in [('outputs','CALIBRATION_OUTPUTS.npz','calibration_outputs_sha256'),
                             ('decisions','CALIBRATION_DECISIONS.json','calibration_decisions_sha256')]:
        (run/name).write_bytes((run/source['selected']['artifacts'][key]['path']).read_bytes())
        source[field] = module.sha(run/name)
    module.dump(run/'SELECTION.json', source)
    report = {'status':'NO_PROMOTABLE_CHECKPOINT','optimizer_steps_executed':3000,'selected_step':1500,
        'selected_checkpoint_sha256':module.sha(run/'model.pth'),
        'last_checkpoint_sha256':source['history'][-1]['artifacts']['checkpoint']['sha256'],
        'test_read':False,'development_holdout_read':False}
    for name in ('report.json','NO_PROMOTABLE_CHECKPOINT.json'):module.dump(run/name,report)
    monkeypatch.setattr(trainer,'STUDENT_CHECKPOINT_SHA',module.sha(run/'model.pth'))
    monkeypatch.setattr(trainer,'STUDENT_SELECTION_SHA',module.sha(run/'SELECTION.json'))
    identity = {'checkpoint':{'path':str(run/'model.pth'),'sha256':trainer.STUDENT_CHECKPOINT_SHA},
        'selection':{'path':str(run/'SELECTION.json'),'sha256':trainer.STUDENT_SELECTION_SHA},
        'training_freeze':{'path':str(run/'TRAINING_FREEZE.json'),'sha256':module.sha(run/'TRAINING_FREEZE.json')},
        'state_sha256':source['state_after_sha256'],'parameter_groups_sha256':deepcopy(source['selected_parameter_groups_sha256']),
        'selected_step':1500,'optimizer_steps_executed':3000}
    current = deepcopy(source)
    current.update(student_initializer=identity,initial_state_sha256=identity['state_sha256'],state_before_sha256=identity['state_sha256'],
        initial_parameter_groups_sha256=deepcopy(identity['parameter_groups_sha256']),state_after_sha256='f'*64,
        source_optimizer_steps_executed=3000,source_selected_step=1500,core_optimizer_steps_executed=1500,
        known_data={'path':'paired/data/MANIFEST.json','sha256':'7'*64},
        known_cache={'path':'paired/cache/CACHE_MANIFEST.json','sha256':'8'*64})
    for record in source['history']:
        for item in record['artifacts'].values():current['bindings'][str(run/item['path'])] = item['sha256']
    for name in ('model.pth','SELECTION.json','TRAINING_FREEZE.json','report.json','NO_PROMOTABLE_CHECKPOINT.json',
                 'SAMPLING.json','TRAINING_COUNTS.json','CALIBRATION_OUTPUTS.npz','CALIBRATION_DECISIONS.json'):
        current['bindings'][str(run/name)] = module.sha(run/name)
    return current, source


def test_prior_student_lineage_is_distinct_from_the_frozen_core_teacher(monkeypatch,tmp_path):
    selection, source = fixture_student(monkeypatch,tmp_path)
    assert module.validate_student_initializer(selection) == source
    assert selection['initial_state_sha256'] != selection['cached_teacher_state_sha256']
    assert selection['initial_parameter_groups_sha256'] == source['selected_parameter_groups_sha256']


@pytest.mark.parametrize('fault',['teacher_as_student','teacher_groups','wrong_checkpoint','wrong_source_step','wrong_source_steps',
    'unbound_selected_step','changed_source_bytes','missing_student_group','teacher_cache_changed'])
def test_prior_student_source_or_teacher_cannot_be_substituted(monkeypatch,tmp_path,fault):
    selection, source = fixture_student(monkeypatch,tmp_path)
    identity = selection['student_initializer']
    if fault == 'teacher_as_student':selection['initial_state_sha256'] = selection['base_state_sha256']
    elif fault == 'teacher_groups':selection['initial_parameter_groups_sha256'] = source['initial_parameter_groups_sha256']
    elif fault == 'wrong_checkpoint':monkeypatch.setattr(trainer,'STUDENT_CHECKPOINT_SHA','0'*64)
    elif fault == 'wrong_source_step':identity['selected_step'] = 3000
    elif fault == 'wrong_source_steps':identity['optimizer_steps_executed'] = 1500
    elif fault == 'unbound_selected_step':
        path = Path(identity['checkpoint']['path']).parent/source['selected']['artifacts']['checkpoint']['path']
        selection['bindings'].pop(str(path))
    elif fault == 'changed_source_bytes':Path(identity['checkpoint']['path']).write_bytes(b'Another model')
    elif fault == 'missing_student_group':identity['parameter_groups_sha256'].pop('size_head')
    else:selection['cache_manifest']['sha256'] = '0'*64
    with pytest.raises(ValueError):module.validate_student_initializer(selection)


def test_metadata_preserves_student_known_data_and_teacher_separately(monkeypatch,tmp_path):
    selection,_ = fixture_student(monkeypatch,tmp_path)
    meta = module.training_metadata(selection)
    for key in ('student_initializer','known_data','known_cache','base_checkpoint','base_state_sha256',
                'cached_teacher_state_sha256','initial_parameter_groups_sha256','source_optimizer_steps_executed',
                'source_selected_step','core_optimizer_steps_executed'):
        assert meta[key] == selection[key]
    assert meta['student_initializer'] is not selection['student_initializer']
    assert meta['initial_state_sha256'] == meta['student_initializer']['state_sha256'] != meta['base_state_sha256']
    assert meta['model_count'] == meta['encoder_count'] == 1 and meta['teacher_in_deployed_model'] is False


def fixture_known(monkeypatch,tmp_path):
    data,identity,folder = known_cache_tests.fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(trainer,'load_known_cache',known_cache_tests.module.load_cache)
    cache = known_cache_tests.synthetic_cache(folder,identity)
    bindings = deepcopy(identity['bindings'])
    for path in folder.rglob('*'):
        if path.is_file():bindings[str(path)] = module.sha(path)
    selection = {key:deepcopy(identity[key]) for key in ('families','base_checkpoint','base_selection','base_state_sha256')}
    selection.update(bindings=bindings,cache_manifest=identity['old_cache_manifest'],known_data=identity['data_manifest'],
        known_cache={'path':str(folder/'CACHE_MANIFEST.json'),'sha256':module.sha(folder/'CACHE_MANIFEST.json')},
        cached_teacher_state_sha256=identity['base_state_sha256'])
    monkeypatch.setattr(trainer,'KNOWN_MANIFEST_SHA',selection['known_data']['sha256'])
    monkeypatch.setattr(trainer,'KNOWN_CACHE_SHA',selection['known_cache']['sha256'])
    unknown = [{'source_id':pair['source_id'],'region_id':pair['region_id'],'family':'__unknown__',
        'source_font_family':pair['font_family'],'normalized_text_sha256':pair['normalized_text_sha256'],
        'source_sha256':pair['source_image_sha256'],'proof':pair['source_proof'],'proof_sha256':pair['source_proof_sha256']}
        for pair in (row['pair_evidence'] for row in data['rows'])]
    return selection,data,unknown


def test_known_data_uses_core_logits_and_pairs_to_the_actual_unknown_train_regions(monkeypatch,tmp_path):
    selection,data,unknown = fixture_known(monkeypatch,tmp_path)
    rows,part = module.validate_known_supplement(selection,[],unknown)
    assert rows == data['rows'] and part == data['partition']
    assert [row['target'] for row in rows] == [10,11,11]


@pytest.mark.parametrize('fault',['student_teacher','missing_counterpart','changed_text','changed_proof','overlap','wrong_cache'])
def test_known_teacher_cache_or_pair_counterpart_mismatch_is_rejected(monkeypatch,tmp_path,fault):
    selection,data,unknown = fixture_known(monkeypatch,tmp_path)
    old = []
    if fault == 'student_teacher':selection['cached_teacher_state_sha256'] = '0'*64
    elif fault == 'missing_counterpart':unknown.pop(0)
    elif fault == 'changed_text':unknown[0]['normalized_text_sha256'] = '0'*64
    elif fault == 'changed_proof':unknown[0]['proof_sha256'] = '0'*64
    elif fault == 'overlap':old = [data['rows'][0]]
    else:monkeypatch.setattr(trainer,'KNOWN_CACHE_SHA','0'*64)
    with pytest.raises(ValueError):module.validate_known_supplement(selection,old,unknown)


@pytest.fixture(scope='module')
def actual_counts():
    order = sorted([*(f'Original Unknown {i}' for i in range(9)),*trainer.NEW_SOURCES])
    quantities = dict.fromkeys(FAMILIES,2);quantities['__unknown__'] = 0;quantities['PingFang'] += 16
    quantities['SF Pro'] += 4;quantities['Helvetica'] += 4
    for family in trainer.SAMPLING['original_eight_families']:quantities[family] += 1
    known = [];new_known = []
    for family,number in quantities.items():
        for index in range(number):
            row = {'family':family,'target':FAMILIES.index(family),'domain':'ios','source_font_family':family,
                'view':'native','split':'train','native_font_verified':True}
            if family in trainer.KNOWN_FAMILIES and index == 1:
                row.update(domain='android',source_dataset='android_paired_known_supplement',
                    source_id='paired-'+family,region_id='paired-region-'+family,
                    pair_evidence={'intentional_train_text_pair':True})
                new_known.append(deepcopy(row));row[trainer.KNOWN_SUPPLEMENT_MARKER] = True
            known.append(row)
    counter = trainer.FullSupplementCounts(FAMILIES,order)
    for step in range(3000):
        rows = list(known)
        for offset in range(16):
            source = order[(step*16+offset)%11]
            row = {'family':'__unknown__','target':24,'domain':'android','source_font_family':source,
                'split':'train','view':'native','native_font_verified':True}
            if source in trainer.NEW_SOURCES:row[trainer.SUPPLEMENT_MARKER] = True
            rows.append(row)
        counter.update(rows,[row['family'] not in (*trainer.core.CORE_FAMILIES,'__unknown__') for row in rows])
    return counter.report(),new_known


def fixture_counts(monkeypatch,tmp_path,actual_counts):
    counts,known_rows = deepcopy(actual_counts)
    run = tmp_path/'run';run.mkdir();data = tmp_path/'data';(data/'train').mkdir(parents=True)
    old_rows = [{'family':'__unknown__','target':24,'source_font_family':source,'split':'train','domain':'android'}
        for source in counts['unknown_source_order'] if source not in trainer.NEW_SOURCES]
    module.dump(data/'train/rows.json',old_rows)
    module.dump(data/'train/MANIFEST.json',{'split':'train','families':FAMILIES,
        'metadata':{'path':'rows.json','sha256':module.sha(data/'train/rows.json')}})
    known = tmp_path/'known';(known/'train').mkdir(parents=True)
    module.dump(known/'MANIFEST.json',{'families':FAMILIES,'split':'train'})
    module.dump(known/'train/rows.json',known_rows)
    module.dump(known/'train/MANIFEST.json',{'split':'train','families':FAMILIES,
        'metadata':{'path':'rows.json','sha256':module.sha(known/'train/rows.json')}})
    bindings = {str(path):module.sha(path) for root in (data,known) for path in root.rglob('*.json')}
    total_new = counts['supplement_rows']
    sampling = {'schema':'flux-glyph-retention-paired-known-sampling-v1','source_balanced':True,
        'unknown_source_order':counts['unknown_source_order'],'unknown_source_rows':counts['unknown_source_rows'],'unknown_rows':48000,
        'supplement_source_rows':counts['supplement_source_rows'],'supplement_view_rows':{'native':total_new},
        'known_supplement_rows':6000,'known_supplement_source_rows':dict.fromkeys(trainer.KNOWN_FAMILIES,3000),
        'known_supplement_view_rows':{'native':6000},'known_class_prior_changed':False,
        'slots':{'base_known':138000,'known_supplement':6000,'base_unknown':48000-total_new,'supplement_unknown':total_new,
            'ios_native_pingfang':48000,'ios_native_sfpro_helvetica':24000,'ios_native_original_eight':24000},
        'base':{'known_rows':138000,'unknown_rows':48000-total_new},'proposal_rows_discarded':0,
        'test_read':False,'development_holdout_read':False,'family_rows':counts['family_rows'],'domain_rows':counts['domain_rows'],
        'view_rows':{'native':288000},'source_rows':[{key:row[key] for key in ('domain','source_font_family','target_family','rows')}
            for row in counts['source_target_rows']]}
    module.dump(run/'SAMPLING.json',sampling);module.dump(run/'TRAINING_COUNTS.json',counts)
    selection = {'families':FAMILIES,'bindings':bindings,'unknown_source_order':counts['unknown_source_order'],
        'known_data':{'path':str(known/'MANIFEST.json'),'sha256':module.sha(known/'MANIFEST.json')},
        'sampling_sha256':module.sha(run/'SAMPLING.json'),'training_counts_sha256':module.sha(run/'TRAINING_COUNTS.json')}
    return run,data,selection,sampling,counts


def test_actual_3000_batches_keep_class_priors_and_separate_all_three_sources(monkeypatch,tmp_path,actual_counts):
    run,data,selection,sampling,counts = fixture_counts(monkeypatch,tmp_path,actual_counts)
    module.validate_counts(run,selection,data)
    assert counts['rows'] == 288000 and counts['known_supplement_rows'] == 6000
    assert counts['supplement_rows'] == 8726 and counts['original_rows'] == 273274
    assert counts['family_rows']['LXGW WenKai'] == counts['family_rows']['WenQuanYi Micro Hei'] == 6000
    assert sampling['base']['known_rows'] == 138000 and counts['known_supplement_rows_per_step'] == [2]*3000


@pytest.mark.parametrize('fault',['old_known_proposals','double_counted_total','wrong_known_quota','unknown_kl','unseen_region','changed_known_view'])
def test_actual_paired_execution_counts_reject_proposals_relabeling_and_unseen_regions(monkeypatch,tmp_path,actual_counts,fault):
    run,data,selection,sampling,counts = fixture_counts(monkeypatch,tmp_path,actual_counts)
    if fault == 'old_known_proposals':sampling['base']['known_rows'] = 144000
    elif fault == 'double_counted_total':counts['original_rows'] += 6000
    elif fault == 'wrong_known_quota':sampling['known_supplement_source_rows'][trainer.KNOWN_FAMILIES[0]] -= 1
    elif fault == 'unknown_kl':counts['eligible_family_rows']['__unknown__'] = 1
    elif fault == 'unseen_region':counts['known_supplement_region_rows'][0]['region_id'] = 'never-captured'
    else:sampling['known_supplement_view_rows'] = {'half':6000}
    module.dump(run/'SAMPLING.json',sampling);module.dump(run/'TRAINING_COUNTS.json',counts)
    selection['sampling_sha256'] = module.sha(run/'SAMPLING.json');selection['training_counts_sha256'] = module.sha(run/'TRAINING_COUNTS.json')
    with pytest.raises(ValueError):module.validate_counts(run,selection,data)
