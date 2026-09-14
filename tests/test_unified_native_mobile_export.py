"""Validate the bounded native-mobile export boundary without running inference."""
from copy import deepcopy

import pytest

from training import export_unified_native_mobile as export
from training.train_unified_native_mobile import ANDROID_SYSTEM_GUARD, FAMILIES, OBJECTIVE, STEPS
from training.native_short_focus import FOCUS_KNOWN_FAMILIES, FOCUS_UNKNOWN_SOURCES, VIEWS


def counts_fixture():
    sources = sorted([*export.SHORT_UNKNOWN_SOURCES, 'WenQuanYi Zen Hei', 'Zhuque Fangsong'])
    quotient, remainder = divmod(STEPS * 16, len(sources))
    unknown = {name: quotient + int(i < remainder) for i, name in enumerate(sources)}
    families = {name: STEPS * 2 for name in FAMILIES[:-1]}
    families['__unknown__'] = STEPS * 16
    families['PingFang'] += STEPS * 16
    families['SF Pro'] += STEPS * 4
    families['Helvetica'] += STEPS * 4
    for name in export.REPLAY_SAMPLING['original_eight_families']:
        families[name] += STEPS
    replay = {'schema': 'flux-glyph-retention-face-balanced-sampling-v1',
        'unknown_source_order': sources, 'unknown_source_rows': unknown,
        'unknown_rows': STEPS * 16, 'family_rows': families,
        'known_supplement_rows': STEPS * 2,
        'known_supplement_source_rows': {'LXGW WenKai': STEPS, 'WenQuanYi Micro Hei': STEPS},
        'known_supplement_face_rows': [
            {'family': 'LXGW WenKai', 'font_face': 'LXGWWenKai-Light', 'rows': STEPS // 3},
            {'family': 'LXGW WenKai', 'font_face': 'LXGWWenKai-Medium', 'rows': STEPS // 3},
            {'family': 'LXGW WenKai', 'font_face': 'LXGWWenKai-Regular', 'rows': STEPS // 3},
            {'family': 'WenQuanYi Micro Hei', 'font_face': 'WenQuanYiMicroHei', 'rows': STEPS}]}
    focus_family = {f: STEPS * 2 for f in FOCUS_KNOWN_FAMILIES} | {'__unknown__': STEPS * 4}
    lengths = {'3': STEPS * 16, '4': STEPS * 16}
    focus = {'schema': 'flux-glyph-native-short-focus-sampling-v1', 'steps': STEPS,
        'rows': STEPS * 32, 'sampling': deepcopy(export.FOCUS_SAMPLING),
        'by_family': deepcopy(focus_family), 'by_glyph_count': deepcopy(lengths),
        'by_unknown_source': {s: STEPS * 4 // len(FOCUS_UNKNOWN_SOURCES) for s in FOCUS_UNKNOWN_SOURCES},
        'by_view': {v: STEPS * 8 for v in VIEWS},
        'by_face': [{'family': f, 'font_face': f + '-face', 'rows': n}
                    for f, n in focus_family.items()]}
    mobile = export.read(export.MOBILE_RECONSTRUCTION)['full']
    mobile_family = {name: STEPS for name in export.MOBILE_KNOWN} | {'__unknown__': STEPS * 3}
    mobile_source = {name: STEPS for name in export.MOBILE_KNOWN} | {
        name: STEPS * 3 // 4 for name in export.MOBILE_UNKNOWN_SOURCES}
    return {'optimizer_steps': STEPS, 'replay_rows': STEPS * 96,
        'unknown_oe_rows': STEPS * 16, 'unknown_oe_policy': deepcopy(export.UNKNOWN_OE_POLICY),
        'unknown_oe_multiplier': .25, 'effective_unknown_oe_coefficient': .125,
        'focus_rows': STEPS * 32, 'focus_supervised_rows': STEPS * 32,
        'focus_by_family': focus_family, 'focus_by_glyph_count': lengths,
        'focus_teacher_eligible_rows': 68712, 'focus_sampling': focus,
        'replay_sampling': replay, 'replay_unknown_by_source': unknown,
        'replay_by_family': families, 'mobile_rows': 48000, 'mobile_unknown_rows': 7200,
        'mobile_by_family': mobile_family, 'mobile_by_source': mobile_source,
        'mobile_by_platform': {'ios': 7200, 'android': 40800}, 'mobile_sampling': mobile}


def test_mobile_objective_and_complete_accounting_are_exact():
    export.validate_v1_objective(OBJECTIVE)
    assert export.validate_counts(counts_fixture(), FAMILIES)


@pytest.mark.parametrize(('field', 'value'), [
    ('mobile_batch', 24), ('mobile_weight', .25), ('mobile_loss', 'classification only'),
    ('unknown_oe_multiplier', .5), ('effective_unknown_oe_coefficient', .25)])
def test_changed_objective_is_rejected(field, value):
    changed = deepcopy(OBJECTIVE); changed[field] = value
    with pytest.raises(ValueError):
        export.validate_v1_objective(changed)


@pytest.mark.parametrize(('field', 'value'), [
    ('mobile_rows', 47999), ('mobile_unknown_rows', 7199),
    ('mobile_by_platform', {'ios': 40800, 'android': 7200}),
    ('unknown_oe_rows', 48000)])
def test_changed_mobile_or_oe_budget_is_rejected(field, value):
    counts = counts_fixture(); counts[field] = value
    with pytest.raises(ValueError):
        export.validate_counts(counts, FAMILIES)


def test_mobile_reconstruction_and_both_prepared_manifests_are_required():
    proof = export.read(export.MOBILE_RECONSTRUCTION)['source_proof']
    assert export.validate_mobile_proof(proof, deepcopy(proof), proof['bindings']) is proof
    changed = deepcopy(proof)
    changed['bindings'][str((export.ANDROID_DATA_ROOT / 'MANIFEST.json').resolve())] = '0' * 64
    with pytest.raises(ValueError):
        export.validate_mobile_proof(changed, changed, changed['bindings'])
    with pytest.raises(ValueError):
        export.validate_mobile_proof(proof, proof, {})


def test_preflight_trace_executes_half_weight_mobile_loss_and_replay_only_oe():
    parts = {'unknown_oe_rows': 16, 'unknown_oe': .2, 'unknown_oe_weighted': .05,
             'mobile_ce': 4., 'mobile_size': .5, 'mobile_loss': 2.05}
    assert export.validate_preflight_loss_trace(parts) is parts
    for field, value in [('unknown_oe_weighted', .1), ('mobile_loss', 1.025),
                         ('mobile_size', float('nan'))]:
        changed = deepcopy(parts); changed[field] = value
        with pytest.raises(ValueError):
            export.validate_preflight_loss_trace(changed)


def guard(actual=26, passed=True):
    return {**ANDROID_SYSTEM_GUARD, 'actual': actual, 'passed': passed}


def test_android_guard_policy_and_measurement_are_separate_and_fail_closed():
    assert export.validate_android_system_guard(ANDROID_SYSTEM_GUARD, guard(), guard(27)) == guard()
    for policy, measured, baseline in [
            ({**ANDROID_SYSTEM_GUARD, 'maximum': 28}, guard(), guard(27)),
            (ANDROID_SYSTEM_GUARD, guard(28, False), guard(27)),
            (ANDROID_SYSTEM_GUARD, guard(), guard(26))]:
        with pytest.raises(ValueError):
            export.validate_android_system_guard(policy, measured, baseline)


def selection_fixture():
    proof = export.read(export.MOBILE_RECONSTRUCTION)['source_proof']
    return {'calibration_promotion_allowed': True, 'promotion_allowed': False,
        'families': list(FAMILIES), 'training_teacher_proof': {'policy': {}},
        'native_focus_proof': {'identity_count': 4752}, 'mobile_short_proof': proof,
        'training_protocol_sha256': 'a' * 64, 'training_counts_sha256': 'b' * 64,
        'initializer_state_sha256': 'c' * 64, 'state_after_sha256': 'd' * 64,
        'initial_parameter_groups_sha256': {}, 'selected_parameter_groups_sha256': {},
        'objective': deepcopy(OBJECTIVE), 'focus_sampling': deepcopy(export.FOCUS_SAMPLING),
        'android_system_guard_policy': deepcopy(ANDROID_SYSTEM_GUARD),
        'android_system_guard': guard(), 'android_system_guard_baseline': guard(27),
        'selected': {}}


def test_parity_is_one_cnn_cal53_plus_android_guard():
    selection = selection_fixture()
    measured = {'retention_checks': [{'passed': True} for _ in range(46)]}
    added = {'passed': True, 'checks': [{'passed': True} for _ in range(7)]}
    parity = export.build_parity_report(selection, '1' * 64, '2' * 64, '3' * 64,
        '4' * 64, {'source': 'sha'}, {'cal': 'sha'}, {'logits': 1e-5, 'size': 1e-5},
        .01, [], measured, added, guard())
    assert parity['schema'] == 'flux-glyph-unified-native-mobile-onnx-parity-v1'
    assert parity['calibration_checks_passed'] == 53
    assert parity['android_system_guard_policy'] == ANDROID_SYSTEM_GUARD
    assert parity['android_system_guard']['actual'] == 26
    assert parity['training']['cnn_input_rows_per_step'] == 148
    assert parity['training']['mobile_rows'] == 48000
    assert parity['training']['mobile_by_platform'] == {'ios': 7200, 'android': 40800}
    assert parity['font_model_count'] == 1 and parity['platform_routing'] is False


def test_frozen_paths_and_android_manifest_are_pinned():
    assert export.NATIVE_RUN.as_posix().endswith('/artifacts/unified-font-v4/run-native-mobile-v1')
    assert export.NATIVE_PLAN.as_posix().endswith('/artifacts/unified-font-v4/native-mobile-plan-v1/PLAN.json')
    assert export.ANDROID_MANIFEST_SHA == '09584bd629c966148ed27a2c0a82698b3e09bc0f7b5e8c974f0bc0b20ceae872'
