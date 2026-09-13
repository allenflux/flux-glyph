"""Native-core export additions, reusing the frozen distilled provenance fixture."""
import copy
import json
from pathlib import Path

import pytest

import test_unified_retention_distilled_export as distilled_tests
from training import export_unified_retention_core as module
from flux_glyph.model_download import release_metadata


def fixture_run(monkeypatch, tmp_path):
    run, data, selection, protocol, cal, parent_meta, _, sampled = distilled_tests.fixture_run(monkeypatch, tmp_path)
    root = run.parent; families = selection['families']
    monkeypatch.setattr(module, 'ROOT', root)
    monkeypatch.setattr(module.retention, 'ROOT', root)
    monkeypatch.setattr(module.retention, 'read_plan', distilled_tests.module.retention.read_plan)
    for key in ('STUDENT_CHECKPOINT_SHA', 'STUDENT_SELECTION_SHA', 'TEACHER_CHECKPOINT_SHA'):
        monkeypatch.setattr(module.retention, key, getattr(distilled_tests.module.retention, key))
    for name in ('training/train_unified_retention_core.py', 'training/evaluate_unified_retention_core.py'):
        path = root/name; path.write_text('Synthetic frozen source: '+name)
        selection['bindings'][str(path)] = protocol['bindings'][str(path)] = module.sha(path)
    for document in (selection, protocol):
        document.update(objective_variant=module.retention.OBJECTIVE_VARIANT,
            objective=copy.deepcopy(module.retention.OBJECTIVE),
            native_core_weighting=copy.deepcopy(module.retention.NATIVE_CORE_WEIGHTING))
    counts = {family: count//1500 for family, count in sampled['family_rows'].items()}
    rows = [{'family': family, 'target': families.index(family), 'split': 'train', 'native_font_verified': True,
        'view': 'native', 'domain': 'android' if family == module.UNKNOWN else 'ios', 'source_font_family': family}
        for family, count in counts.items() for _ in range(count)]
    counter = module.retention.DistillationCounter(families)
    counter.update(rows, [row['family'] not in module.retention.CORE_FAMILIES for row in rows])
    report = copy.deepcopy(counter.report()); steps = module.retention.STEPS
    report.update(steps=steps, rows=96*steps, eligible_rows=63*steps, eligible_rows_per_step=[63]*steps,
        native_core_weighted_rows=33*steps, native_core_weighted_rows_per_step=[33]*steps,
        teacher_state_before_sha256='a'*64, teacher_state_after_sha256='a'*64,
        teacher_optimizer_steps=0, training_model_count=2, deployed_model_count=1)
    for name in ('family_rows', 'eligible_family_rows', 'domain_rows', 'eligible_domain_rows', 'native_core_weighted_family_rows'):
        report[name] = {key: value*steps for key, value in report[name].items()}
    for row in report['source_target_rows']:
        row['rows'] *= steps; row['eligible_rows'] *= steps
    for row in report['native_core_source_rows']:row['rows'] *= steps
    module.dump(run/'DISTILLATION.json', report)
    selection.update(distillation_sha256=module.sha(run/'DISTILLATION.json'),
        core_weighting_sha256=module.sha(run/'DISTILLATION.json'), core_weighting_evidence_file='DISTILLATION.json')
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection['training_protocol_sha256'] = module.sha(run/'TRAINING_FREEZE.json')
    module.dump(run/'SELECTION.json', selection)
    return run, data, selection, protocol, cal, parent_meta, report


def test_core_weights_and_three_family_teacher_exclusion_keep_one_promotable_student(monkeypatch, tmp_path):
    run, data, selection, _, _, _, report = fixture_run(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    assert module.validate(run, data) == selection
    assert all(report['eligible_family_rows'].get(f, 0) == 0 for f in module.retention.CORE_FAMILIES)
    assert report['native_core_weighted_rows'] == 33*1500
    assert selection['core_weighting_sha256'] == selection['distillation_sha256']
    assert selection['passed'] is False and selection['promotion_allowed'] is True
    assert not (data/'test').exists() and not (data/'development_holdout').exists()


@pytest.mark.parametrize('fault', ['weight', 'denominator', 'size_weighted', 'domain', 'view', 'core_families',
    'teacher_mask', 'teacher_kl', 'evidence_sha', 'evidence_file', 'report_rule', 'weighted_step',
    'weighted_total', 'weighted_family', 'weighted_source', 'sf_teacher', 'helvetica_teacher',
    'source_binding', 'runtime_gate', 'failed_promotion'])
def test_core_objective_and_actual_weighting_cannot_be_relaxed_or_forged(monkeypatch, tmp_path, fault):
    run, data, selection, protocol, _, _, report = fixture_run(monkeypatch, tmp_path)
    if fault in ('weight', 'denominator', 'size_weighted', 'domain', 'view', 'core_families'):
        key, value = {'weight': ('weight', 3.), 'denominator': ('denominator', 'sum_of_weights'),
            'size_weighted': ('size_loss_weighted', True), 'domain': ('domain', 'android'),
            'view': ('view', 'half'), 'core_families': ('families', ['PingFang'])}[fault]
        for doc in (selection, protocol):
            doc['native_core_weighting'][key] = value
            doc['objective']['native_core_weighting'][key] = value
        report['native_core_weighting'][key] = value
    elif fault in ('teacher_mask', 'teacher_kl'):
        key, value = ('teacher_mask', 'Only exclude PingFang') if fault == 'teacher_mask' else ('teacher_kl_weight', 2.)
        for doc in (selection, protocol):doc['objective'][key] = value
    elif fault == 'evidence_file':selection['core_weighting_evidence_file'] = 'Unbound.json'
    elif fault == 'report_rule':report['native_core_weighting']['weight'] = 3.
    elif fault == 'weighted_step':report['native_core_weighted_rows_per_step'][0] = 26
    elif fault == 'weighted_total':report['native_core_weighted_rows'] -= 1
    elif fault == 'weighted_family':report['native_core_weighted_family_rows']['PingFang'] -= 1
    elif fault == 'weighted_source':report['native_core_source_rows'][0]['domain'] = 'android'
    elif fault in ('sf_teacher', 'helvetica_teacher'):
        report['eligible_family_rows']['SF Pro' if fault == 'sf_teacher' else 'Helvetica'] = 1
    elif fault == 'source_binding':
        key = str(module.ROOT/'training/train_unified_retention_core.py')
        selection['bindings'].pop(key); protocol['bindings'].pop(key, None)
    elif fault == 'runtime_gate':selection['fixed_runtime'] = {**selection['fixed_runtime'], 'temperature': .5}
    elif fault == 'failed_promotion':selection['promotion_allowed'] = False
    module.dump(run/'DISTILLATION.json', report)
    selection['distillation_sha256'] = selection['core_weighting_sha256'] = module.sha(run/'DISTILLATION.json')
    if fault == 'evidence_sha':selection['core_weighting_sha256'] = '0'*64
    module.dump(run/'TRAINING_FREEZE.json', protocol)
    selection['training_protocol_sha256'] = module.sha(run/'TRAINING_FREEZE.json')
    module.dump(run/'SELECTION.json', selection)
    with pytest.raises(ValueError):module.validate(run, data)


def test_public_core_metadata_carries_exact_selection_rules_and_actual_evidence_hash(monkeypatch, tmp_path):
    _, _, selection, _, cal, parent_meta, report = fixture_run(monkeypatch, tmp_path)
    metadata = module.metadata_for_export(selection, cal, parent_meta, '0'*64, '1'*64, '2'*64,
        module.strip_artifacts(selection['selected']), report)
    for key in ('native_core_weighting', 'core_weighting_sha256', 'core_weighting_evidence_file',
                'teacher', 'student_initializer', 'distillation_sha256'):
        assert metadata['training'][key] == selection[key]
    assert metadata['validation']['native_core_weighted_training_rows'] == 33*1500
    assert metadata['training']['training_model_count'] == 2 and metadata['training']['model_count'] == 1
    assert metadata['training']['teacher_in_deployed_model'] is False
    assert metadata['stable_validation_passed'] is metadata['test_passed'] is False
    assert metadata['gates'] == module.retention.FIXED_RUNTIME['gates']
    assert release_metadata(metadata)['validation'] == metadata['validation']
