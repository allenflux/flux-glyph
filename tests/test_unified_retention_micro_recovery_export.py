"""Portable export-boundary checks for candidate 21's TRAIN-only Micro quarter-CE weight."""
from copy import deepcopy

import pytest

from training import export_unified_retention_micro_recovery as module
from training import train_unified_retention_micro_recovery as trainer
from training.retention_angular_margin_loss import ANGULAR_MARGIN
from training.retention_wenkai_recovery_loss import WENKAI_RECOVERY_WEIGHTING
from training.retention_micro_recovery_loss import MICRO_RECOVERY_WEIGHTING
from test_unified_retention_face_balanced_export import selection_fixture as face_balanced_selection


def selection_fixture():
    selection = deepcopy(face_balanced_selection())
    selection.update(schema='flux-glyph-unified-retention-wide-micro-recovery-selection-v1',
        architecture=trainer.ARCHITECTURE, objective_variant=trainer.OBJECTIVE_VARIANT,
        objective=deepcopy(trainer.OBJECTIVE), angular_margin=deepcopy(ANGULAR_MARGIN),
        angular_margin_applies_at_inference=False, angular_margin_adds_deployment_parameters=False,
        wenkai_recovery_weighting=deepcopy(WENKAI_RECOVERY_WEIGHTING),
        wenkai_recovery_applies_at_inference=False,
        wenkai_recovery_adds_deployment_parameters=False,
        micro_recovery_weighting=deepcopy(MICRO_RECOVERY_WEIGHTING),
        micro_recovery_applies_at_inference=False,
        micro_recovery_adds_deployment_parameters=False,
        design_changes=deepcopy(trainer.DESIGN_CHANGES), steps=trainer.STEPS,
        eval_every=trainer.EVAL_EVERY, model_count=1, encoder_count=1,
        platform_routing=False, score_merging=False, training_inputs=['image_tiles'],
        offline_teacher_count=2, teacher_models_resident_during_optimizer=0,
        teacher_logits_used=True, second_model_resident=False, teacher_optimizer_steps=0,
        teacher_cache_deployed=False,
        passed=False, calibration_passed=False, promotion_allowed=True)
    selection['selected'].update(step=trainer.STEPS, promotion_allowed=True, metrics={'passed': False},
        retention_checks=[{'population': 'synthetic', 'metric': f'gate-{index}', 'passed': True}
                          for index in range(46)])
    selection['history'] = [deepcopy(selection['selected'])]
    return selection


def runtime_record():
    return {**deepcopy(module.FIXED_RUNTIME), 'promotion_allowed': True,
        'retention_checks': [], 'retention_populations': {},
        'metrics': {'passed': False, 'named_precision': 1., 'known_correct_coverage': 1.,
            'unknown_not_named_rate': 1., 'checks': {}}}


def build_report(selection):
    return module.build_parity_report(selection, 'a'*64, 'b'*64, 'c'*64, 'd'*64,
        {'exporter': 'e'*64}, {'selection': 'f'*64}, {'logits': 0., 'size': 0.}, 0.,
        46, 137, [{'batch_size': 1, 'passed': True}], {'metrics': {'passed': False}})


def counts_fixture():
    return {'schema': 'flux-glyph-retention-micro-recovery-counts-v1',
        'wenkai_recovery_weighting': deepcopy(WENKAI_RECOVERY_WEIGHTING),
        'wenkai_recovery_rows': 12000, 'wenkai_recovery_rows_per_step': [2] * 6000,
        'wenkai_recovery_source_face_rows': [
            {'domain': 'android', 'source_font_family': 'LXGW WenKai', 'font_face': face, 'rows': 4000}
            for face in ('LXGWWenKai-Light', 'LXGWWenKai-Medium', 'LXGWWenKai-Regular')],
        'micro_recovery_weighting': deepcopy(MICRO_RECOVERY_WEIGHTING),
        'micro_recovery_rows': 12000, 'micro_recovery_rows_per_step': [2] * 6000,
        'micro_recovery_source_face_rows': [{'domain': 'android',
            'source_font_family': 'WenQuanYi Micro Hei', 'font_face': 'WenQuanYiMicroHei', 'rows': 12000}],
        'native_core_weighted_family_rows': {'PingFang': 1, 'SF Pro': 1, 'Helvetica': 1}}


def test_builders_preserve_weighting_full_kl_and_raw_inference_contract():
    selection = selection_fixture(); metadata = module.training_metadata(selection); report = build_report(selection)
    assert module.validate_fixed_final_selection(selection) == selection['history']
    exported = module.metadata_for_export(selection, {'manifest': {'font_label_groups': {}}},
        {'font_sources': {}}, 'c'*64, 'b'*64, 'a'*64, runtime_record())
    for value in (metadata, report, exported['training']):
        assert value['wenkai_recovery_weighting'] == WENKAI_RECOVERY_WEIGHTING
        assert value['objective']['wenkai_recovery_weighting'] == WENKAI_RECOVERY_WEIGHTING
        assert value['objective']['class_prior_weighting'] is True
        assert value['objective']['inference_class_prior_changed'] is False
        assert value['wenkai_recovery_applies_at_inference'] is False
        assert value['wenkai_recovery_adds_deployment_parameters'] is False
        assert value['micro_recovery_weighting'] == MICRO_RECOVERY_WEIGHTING
        assert value['objective']['micro_recovery_weighting'] == MICRO_RECOVERY_WEIGHTING
        assert value['micro_recovery_applies_at_inference'] is False
        assert value['micro_recovery_adds_deployment_parameters'] is False
        assert value['teacher_distribution_kl'] is True
        assert value['angular_margin'] == ANGULAR_MARGIN
        assert value['known_cache'] == selection['known_cache']
        assert value['teacher_mix']['known']['tiles'] == 4820
    assert exported['validation']['kind'] == 'full_cnn_training_micro_recovery_calibration_only_at_export'
    assert report['font_model_count'] == report['encoder_count'] == 1
    assert selection['passed'] is selection['calibration_passed'] is False
    assert selection['selected']['metrics']['passed'] is False
    assert report['runtime_calibration_metrics']['passed'] is False
    assert exported['validation']['calibration_passed'] is False


@pytest.mark.parametrize('fault', ('weighting', 'objective', 'inference', 'deployment',
    'micro_weighting', 'micro_objective', 'micro_inference', 'micro_deployment',
    'prior', 'kl', 'retention', 'promotion'))
def test_builders_reject_weighting_inference_or_failed_selection_drift(fault):
    selection = selection_fixture()
    if fault == 'weighting': selection['wenkai_recovery_weighting']['extra_known_ce_weight'] = .5
    elif fault == 'objective': selection['objective']['wenkai_recovery_weighting']['denominator'] = 95
    elif fault == 'inference': selection['wenkai_recovery_applies_at_inference'] = True
    elif fault == 'deployment': selection['wenkai_recovery_adds_deployment_parameters'] = True
    elif fault == 'micro_weighting': selection['micro_recovery_weighting']['extra_known_ce_weight'] = .5
    elif fault == 'micro_objective': selection['objective']['micro_recovery_weighting']['denominator'] = 95
    elif fault == 'micro_inference': selection['micro_recovery_applies_at_inference'] = True
    elif fault == 'micro_deployment': selection['micro_recovery_adds_deployment_parameters'] = True
    elif fault == 'prior': selection['objective']['inference_class_prior_changed'] = True
    elif fault == 'kl': selection['objective']['teacher_distribution_kl'] = False
    elif fault == 'retention': selection['selected']['retention_checks'][17]['passed'] = False
    else:
        selection['promotion_allowed'] = selection['selected']['promotion_allowed'] = False
    with pytest.raises(ValueError): module.training_metadata(selection)
    with pytest.raises(ValueError): build_report(selection)


def test_fixed_replay_validates_recovery_faces_separately_from_native_core():
    counts = counts_fixture(); selection = selection_fixture()
    wenkai = module.validate_wenkai_recovery_counts(counts, selection)
    micro = module.validate_micro_recovery_counts(counts, selection)
    assert wenkai['rows'] == micro['rows'] == 12000
    assert {row['font_face']: row['rows'] for row in wenkai['source_face_rows']} == {
        'LXGWWenKai-Light': 4000, 'LXGWWenKai-Medium': 4000, 'LXGWWenKai-Regular': 4000}
    assert micro['source_face_rows'] == [{'domain': 'android',
        'source_font_family': 'WenQuanYi Micro Hei', 'font_face': 'WenQuanYiMicroHei', 'rows': 12000}]


@pytest.mark.parametrize('fault', ('total', 'step', 'face', 'source', 'canonical', 'native'))
def test_fixed_replay_rejects_recovery_count_or_scope_drift(fault):
    counts = counts_fixture(); selection = selection_fixture()
    if fault == 'total': counts['wenkai_recovery_rows'] -= 1
    elif fault == 'step': counts['wenkai_recovery_rows_per_step'][0] = 1
    elif fault == 'face': counts['wenkai_recovery_source_face_rows'][0]['rows'] -= 1
    elif fault == 'source': counts['wenkai_recovery_source_face_rows'][0]['source_font_family'] = 'Other'
    elif fault == 'canonical': counts['wenkai_recovery_weighting']['resulting_known_ce_weight'] = 3.
    else: counts['native_core_weighted_family_rows']['LXGW WenKai'] = 12000
    with pytest.raises(ValueError): module.validate_wenkai_recovery_counts(counts, selection)


@pytest.mark.parametrize('fault', ('total', 'step', 'face', 'source', 'canonical', 'native', 'wenkai'))
def test_fixed_replay_rejects_micro_count_or_scope_drift(fault):
    counts = counts_fixture(); selection = selection_fixture()
    if fault == 'total': counts['micro_recovery_rows'] -= 1
    elif fault == 'step': counts['micro_recovery_rows_per_step'][0] = 1
    elif fault == 'face': counts['micro_recovery_source_face_rows'][0]['rows'] -= 1
    elif fault == 'source': counts['micro_recovery_source_face_rows'][0]['font_face'] = 'Other'
    elif fault == 'canonical': counts['micro_recovery_weighting']['extra_known_ce_weight'] = .5
    elif fault == 'native': counts['native_core_weighted_family_rows']['WenQuanYi Micro Hei'] = 12000
    else: counts['wenkai_recovery_rows'] -= 1
    with pytest.raises(ValueError): module.validate_micro_recovery_counts(counts, selection)


def test_fixed_family_quota_and_final_step_contract_remain_unchanged():
    selection = selection_fixture()
    assert module.validate_angular_margin_provenance(selection) == ANGULAR_MARGIN
    assert module.validate_wenkai_recovery_provenance(selection) == WENKAI_RECOVERY_WEIGHTING
    assert module.validate_micro_recovery_provenance(selection) == MICRO_RECOVERY_WEIGHTING
    expected = module.expected_family_counts(selection['families'], trainer.SAMPLING, trainer.STEPS)
    assert sum(expected.values()) == trainer.BATCH_SIZE * trainer.STEPS == 576000
    assert expected['LXGW WenKai'] == 12000 and expected['PingFang'] == 114000
    assert selection['selected']['step'] == selection['steps'] == selection['eval_every'] == 6000
