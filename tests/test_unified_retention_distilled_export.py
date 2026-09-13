"""Distilled export provenance and release-contract tests; no network inference."""
import copy
import json
from pathlib import Path

import pytest

import test_unified_retention_export as previous_tests
from training import export_unified_retention_distilled as module
from flux_glyph.model_download import release_metadata


def fixture_run(monkeypatch, tmp_path):
    families = list(previous_tests.module.retention.IOS_ANCHOR_FAMILIES)+[f'Extra {i}' for i in range(16)]+[module.UNKNOWN]
    monkeypatch.setattr(previous_tests, 'FAMILIES', families)
    init_run, data, initial, old_protocol, cal, plan, parent_metadata = previous_tests.fixture_run(monkeypatch, tmp_path)
    root = init_run.parent
    monkeypatch.setattr(module, 'ROOT', root)
    monkeypatch.setattr(module.retention, 'ROOT', root)
    monkeypatch.setattr(module.retention, 'read_plan', previous_tests.module.retention.read_plan)
    # A failed first focus run is allowed as a TRAIN initializer, never a release.
    initial['selected'] = copy.deepcopy(initial['history'][-1])
    initial['promotion_allowed'] = initial['selected']['promotion_allowed'] = False
    module.dump(init_run/'SELECTION.json', initial)
    (init_run/'model.pth').write_bytes(b'frozen failed first-focus checkpoint fixture')
    module.dump(init_run/'report.json', {'promotion_allowed': False})
    parent_path = root/'parent/model.pth'
    monkeypatch.setattr(module.retention, 'STUDENT_CHECKPOINT_SHA', module.sha(init_run/'model.pth'))
    monkeypatch.setattr(module.retention, 'STUDENT_SELECTION_SHA', module.sha(init_run/'SELECTION.json'))
    monkeypatch.setattr(module.retention, 'TEACHER_CHECKPOINT_SHA', module.sha(parent_path))
    run = root/'distilled'; run.mkdir()
    source_names = ['training/train_unified_retention_distilled.py', 'training/evaluate_unified_retention_distilled.py']
    for name in source_names:
        (root/name).write_text('bound source fixture: '+name)
    bindings = dict(initial['bindings'])
    paths = [root/name for name in source_names]+[init_run/name for name in
        ('model.pth', 'SELECTION.json', 'TRAINING_FREEZE.json', 'report.json')]
    bindings.update({str(path): module.sha(path) for path in paths})
    initializer = {'checkpoint': {'path': str(init_run/'model.pth'), 'sha256': module.sha(init_run/'model.pth')},
        'selection': {'path': str(init_run/'SELECTION.json'), 'sha256': module.sha(init_run/'SELECTION.json')},
        'state_sha256': initial['state_after_sha256'], 'source_selected_step': 3000, 'source_optimizer_steps_executed': 3000}
    teacher = {'checkpoint': {'path': str(parent_path), 'sha256': module.sha(parent_path)},
        'selection': {'path': str(parent_path.parent/'SELECTION.json'), 'sha256': module.sha(parent_path.parent/'SELECTION.json')},
        'state_sha256': 'a'*64, 'temperature': 1., 'frozen': True, 'optimizer_steps': 0}
    inheritance = copy.deepcopy(initial['initializer_evidence'])
    inheritance.update(checkpoint=initializer['checkpoint'], selection_sha256=initializer['selection']['sha256'],
        source_state_sha256='b'*64, source_selected_step=3000, source_optimizer_steps_executed=3000)
    protocol = copy.deepcopy(old_protocol)
    protocol.update(bindings=bindings, initializer_evidence=inheritance,
        steps=module.retention.STEPS, learning_rate=module.retention.LEARNING_RATE,
        minimum_learning_rate=module.retention.MINIMUM_LEARNING_RATE, objective=module.retention.OBJECTIVE,
        objective_variant=module.retention.OBJECTIVE_VARIANT, student_initializer=initializer, teacher=teacher,
        training_model_count=2, teacher_in_deployed_model=False,
        initial_state_sha256='b'*64, initial_parameter_groups_sha256=initial['final_parameter_groups_sha256'])
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    history = copy.deepcopy(initial['history'][:3])
    for record in history:
        for entry in record['artifacts'].values():
            path = run/entry['path']; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((init_run/entry['path']).read_bytes())
    for name in ('BASELINE.json', 'BASELINE_CALIBRATION_OUTPUTS.npz', 'BASELINE_CALIBRATION_DECISIONS.json'):
        (run/name).write_bytes((init_run/name).read_bytes())
    selected = history[0]
    for key, name in [('outputs', 'CALIBRATION_OUTPUTS.npz'), ('decisions', 'CALIBRATION_DECISIONS.json')]:
        (run/name).write_bytes((run/selected['artifacts'][key]['path']).read_bytes())
    counts = {family: 2 for family in families}
    counts[module.UNKNOWN] = 16; counts['PingFang'] += 16
    for family in ('SF Pro', 'Helvetica'):counts[family] += 4
    for family in previous_tests.module.retention.IOS_ANCHOR_FAMILIES:counts[family] += 1
    rows = [{'family': family, 'target': families.index(family), 'split': 'train', 'native_font_verified': True,
        'domain': 'android' if family == module.UNKNOWN else 'ios', 'source_font_family': family}
        for family, count in counts.items() for _ in range(count)]
    counter = module.retention.DistillationCounter(families)
    counter.update(rows, [row['family'] != 'PingFang' for row in rows])
    distillation = counter.report(); steps = module.retention.STEPS
    distillation.update(steps=steps, rows=96*steps, eligible_rows=77*steps, eligible_rows_per_step=[77]*steps,
        teacher_state_before_sha256='a'*64, teacher_state_after_sha256='a'*64,
        teacher_optimizer_steps=0, training_model_count=2, deployed_model_count=1)
    for name in ('family_rows', 'eligible_family_rows', 'domain_rows', 'eligible_domain_rows'):
        distillation[name] = {key: value*steps for key, value in distillation[name].items()}
    for row in distillation['source_target_rows']:
        row['rows'] *= steps; row['eligible_rows'] *= steps
    sampled = {'slots': {key: module.retention.SAMPLING[key]*steps for key in
        ('base_known', 'base_unknown', 'ios_native_pingfang', 'ios_native_sfpro_helvetica', 'ios_native_original_eight')},
        'base': {'known_rows': 48*steps, 'unknown_rows': 16*steps},
        'family_rows': distillation['family_rows'], 'domain_rows': distillation['domain_rows'],
        'view_rows': {'native': 96*steps}}
    module.dump(run/'SAMPLING.json', sampled); module.dump(run/'DISTILLATION.json', distillation)
    selection = copy.deepcopy(initial)
    selection.update(bindings=bindings, objective_variant=module.retention.OBJECTIVE_VARIANT,
        objective=module.retention.OBJECTIVE, student_initializer=initializer, teacher=teacher,
        training_model_count=2, teacher_in_deployed_model=False, initializer_evidence=inheritance,
        selected=selected, history=history, promotion_allowed=True,
        optimizer_steps_executed=steps, state_before_sha256='b'*64, state_after_sha256='c'*64,
        initial_parameter_groups_sha256=initial['final_parameter_groups_sha256'],
        final_parameter_groups_sha256={name: 'c'*64 for name in module.GROUPS},
        training_protocol_sha256=module.sha(run/'TRAINING_FREEZE.json'),
        sampling_sha256=module.sha(run/'SAMPLING.json'), distillation_sha256=module.sha(run/'DISTILLATION.json'))
    module.dump(run/'SELECTION.json', selection)
    return run, data, selection, protocol, cal, parent_metadata, distillation, sampled


def test_failed_initializer_and_frozen_teacher_can_produce_a_separately_promotable_single_student(monkeypatch, tmp_path):
    run, data, selection, _, _, _, _, _ = fixture_run(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    assert module.validate(run, data) == selection
    old = json.loads(Path(selection['student_initializer']['selection']['path']).read_text())
    assert old['promotion_allowed'] is False
    assert selection['promotion_allowed'] is True
    assert selection['passed'] is selection['calibration_passed'] is False
    assert not (data/'test').exists() and not (data/'development_holdout').exists()


@pytest.mark.parametrize('fault', ['variant', 'objective', 'teacher_weight', 'mask_rule', 'class_prior',
    'teacher_state', 'teacher_temperature', 'teacher_trainable', 'teacher_optimizer', 'deployed_teacher',
    'training_network_count', 'init_state', 'init_steps', 'init_checkpoint', 'init_selection', 'init_closure',
    'init_freeze', 'teacher_checkpoint', 'source_binding', 'steps', 'learning_rate', 'gate', 'promotable',
    'cached_array', 'history', 'sampling', 'distillation_hash'])
def test_distilled_export_rejects_contract_and_provenance_mutations(monkeypatch, tmp_path, fault):
    run, data, selection, protocol, *_ = fixture_run(monkeypatch, tmp_path)
    if fault == 'variant':selection['objective_variant'] = protocol['objective_variant'] = 'extra_teacher'
    elif fault in ('objective', 'teacher_weight', 'mask_rule', 'class_prior'):
        key, value = {'objective': ('teacher_outputs', False), 'teacher_weight': ('teacher_kl_weight', .5),
            'mask_rule': ('teacher_mask', 'All rows including PingFang'), 'class_prior': ('class_prior_weighting', True)}[fault]
        for doc in (selection, protocol):doc['objective'] = {**doc['objective'], key: value}
    elif fault in ('teacher_state', 'teacher_temperature', 'teacher_trainable', 'teacher_optimizer'):
        key, value = {'teacher_state': ('state_sha256', 'b'*64), 'teacher_temperature': ('temperature', 2.),
            'teacher_trainable': ('frozen', False), 'teacher_optimizer': ('optimizer_steps', 1)}[fault]
        selection['teacher'][key] = protocol['teacher'][key] = value
    elif fault == 'deployed_teacher':selection['teacher_in_deployed_model'] = protocol['teacher_in_deployed_model'] = True
    elif fault == 'training_network_count':selection['training_model_count'] = protocol['training_model_count'] = 1
    elif fault in ('init_state', 'init_steps'):
        key, value = ('state_sha256', 'a'*64) if fault == 'init_state' else ('source_selected_step', 2500)
        selection['student_initializer'][key] = protocol['student_initializer'][key] = value
    elif fault == 'init_checkpoint':Path(selection['student_initializer']['checkpoint']['path']).write_bytes(b'changed')
    elif fault == 'init_selection':Path(selection['student_initializer']['selection']['path']).write_text('{}')
    elif fault == 'init_freeze':Path(selection['student_initializer']['checkpoint']['path']).with_name('TRAINING_FREEZE.json').write_text('{}')
    elif fault in ('init_closure', 'source_binding'):
        key = str(module.ROOT/('training/evaluate_unified_retention.py' if fault == 'init_closure'
                              else 'training/train_unified_retention_distilled.py'))
        selection['bindings'].pop(key); protocol['bindings'].pop(key, None)
    elif fault == 'teacher_checkpoint':Path(selection['teacher']['checkpoint']['path']).write_bytes(b'changed')
    elif fault == 'steps':selection['optimizer_steps_executed'] = protocol['steps'] = 3000
    elif fault == 'learning_rate':protocol['learning_rate'] = 5e-5
    elif fault == 'gate':selection['fixed_runtime'] = {**selection['fixed_runtime'], 'temperature': .5}
    elif fault == 'promotable':selection['promotion_allowed'] = False
    elif fault == 'cached_array':(run/'checkpoints/step01500/CALIBRATION_OUTPUTS.npz').write_bytes(b'changed')
    elif fault == 'history':selection['history'].pop()
    elif fault == 'sampling':selection['sampling_sha256'] = '0'*64
    elif fault == 'distillation_hash':selection['distillation_sha256'] = '0'*64
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection['training_protocol_sha256'] = module.sha(run/'TRAINING_FREEZE.json')
    module.dump(run/'SELECTION.json', selection)
    with pytest.raises(ValueError):module.validate(run, data)


@pytest.mark.parametrize('fault', ['pf_eligible', 'step_count', 'mask_count', 'family_total', 'source_total',
    'teacher_changed', 'teacher_identity', 'teacher_optimizer', 'teacher_deployed', 'holdout', 'sampling_mismatch'])
def test_teacher_statistics_must_match_supervised_train_and_frozen_identity(monkeypatch, tmp_path, fault):
    run, data, selection, _, _, _, report, sampled = fixture_run(monkeypatch, tmp_path)
    if fault == 'pf_eligible':report['eligible_family_rows']['PingFang'] = 1
    elif fault == 'step_count':report['eligible_rows_per_step'].pop()
    elif fault == 'mask_count':report['eligible_rows_per_step'][0] = 96
    elif fault == 'family_total':report['eligible_family_rows']['SF Pro'] -= 1
    elif fault == 'source_total':report['source_target_rows'][0]['eligible_rows'] -= 1
    elif fault == 'teacher_changed':report['teacher_state_after_sha256'] = 'b'*64
    elif fault == 'teacher_identity':report['teacher_state_before_sha256'] = report['teacher_state_after_sha256'] = 'b'*64
    elif fault == 'teacher_optimizer':report['teacher_optimizer_steps'] = 1
    elif fault == 'teacher_deployed':report['deployed_model_count'] = 2
    elif fault == 'holdout':report['training_split'] = 'calibration'
    else:
        sampled = copy.deepcopy(sampled); sampled['family_rows']['PingFang'] -= 1; sampled['family_rows']['SF Pro'] += 1
        module.dump(run/'SAMPLING.json', sampled); selection['sampling_sha256'] = module.sha(run/'SAMPLING.json')
    module.dump(run/'DISTILLATION.json', report); selection['distillation_sha256'] = module.sha(run/'DISTILLATION.json')
    module.dump(run/'SELECTION.json', selection)
    with pytest.raises(ValueError):module.validate(run, data)


def test_metadata_discloses_teacher_only_during_training_and_keeps_original_stable_failure(monkeypatch, tmp_path):
    _, _, selection, _, cal, parent_meta, report, _ = fixture_run(monkeypatch, tmp_path)
    metadata = module.metadata_for_export(selection, cal, parent_meta, '0'*64, '1'*64, '2'*64,
        module.strip_artifacts(selection['selected']), report)
    assert metadata['training']['student_initializer'] == selection['student_initializer']
    assert metadata['training']['teacher'] == selection['teacher']
    assert metadata['training']['distillation_sha256'] == selection['distillation_sha256']
    assert metadata['training']['model_count'] == 1 and metadata['training']['training_model_count'] == 2
    assert metadata['training']['teacher_outputs_used'] is True
    assert metadata['training']['teacher_in_deployed_model'] is False
    assert metadata['validation']['teacher_eligible_training_rows'] == 77*1500
    assert metadata['stable_validation_passed'] is metadata['test_passed'] is False
    assert metadata['gates'] == module.retention.FIXED_RUNTIME['gates']
    assert metadata['font_sources'] == parent_meta['font_sources']
    assert 'rejection' not in metadata and 'verifier' not in metadata
    assert release_metadata(metadata)['validation'] == metadata['validation']


def test_teacher_and_initializer_pth_need_the_full_bound_state_not_only_a_family_list(monkeypatch):
    monkeypatch.setattr(module, 'state_sha', lambda state: state['digest'])
    identity = {'selection': {'sha256': 'selection'}, 'state_sha256': 'source'}
    checkpoint = {'families': ['A', 'B'], 'architecture': module.ARCHITECTURE,
        'selection_sha256': 'selection', 'state_dict': {'digest': 'source'}}
    module.validate_source_checkpoint(checkpoint, identity, ['A', 'B'])
    checkpoint['state_dict']['digest'] = 'another teacher'
    with pytest.raises(ValueError):module.validate_source_checkpoint(checkpoint, identity, ['A', 'B'])


def test_distilled_runtime_parity_still_checks_abstention_reasons_and_pixel_sizes():
    logits, ratios = previous_tests.sample_outputs(); rows = previous_tests.sample_rows()
    actual = module.region_outputs(logits, ratios, rows, 1.)
    expected = copy.deepcopy(actual)
    assert module.compare_runtime_outputs(actual, expected, rows, previous_tests.FAMILIES) == 0
    expected[0]['size_spread'] = .2000001
    with pytest.raises(ValueError):module.compare_runtime_outputs(actual, expected, rows, previous_tests.FAMILIES)
