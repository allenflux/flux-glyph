"""Verify exact replay/focus/iOS accounting and the raw native-iOS ONNX boundary."""
from copy import deepcopy

import pytest

from training import export_unified_native_ios as export
from training.train_unified_native_ios import OBJECTIVE, STEPS, FAMILIES
from training.native_short_focus import FOCUS_KNOWN_FAMILIES, FOCUS_UNKNOWN_SOURCES, VIEWS
from training.ios_short_training import IOS_FAMILIES, MANIFEST_SHA


def counts_fixture():
    sources = sorted([*export.SHORT_UNKNOWN_SOURCES, 'WenQuanYi Zen Hei', 'Zhuque Fangsong'])
    quotient, remainder = divmod(STEPS * 16, len(sources))
    unknown = {name: quotient + int(index < remainder) for index, name in enumerate(sources)}
    families = {name: STEPS * 2 for name in FAMILIES[:-1]}; families['__unknown__'] = STEPS * 16
    families['PingFang'] += STEPS * 16; families['SF Pro'] += STEPS * 4; families['Helvetica'] += STEPS * 4
    for name in export.REPLAY_SAMPLING['original_eight_families']: families[name] += STEPS
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
    focus = {'schema': 'flux-glyph-native-short-focus-sampling-v1', 'steps': STEPS, 'rows': STEPS * 32,
        'sampling': deepcopy(export.FOCUS_SAMPLING), 'by_family': deepcopy(focus_family),
        'by_glyph_count': deepcopy(lengths), 'by_unknown_source': {s: STEPS * 4 // 6 for s in FOCUS_UNKNOWN_SOURCES},
        'by_view': {v: STEPS * 8 for v in VIEWS},
        'by_face': [{'family': f, 'font_face': f + '-face', 'rows': n} for f, n in focus_family.items()]}
    ios = export.read(export.IOS_RECONSTRUCTION)['full']
    return {'optimizer_steps': STEPS, 'replay_rows': STEPS * 96, 'focus_rows': STEPS * 32,
        'unknown_oe_rows': STEPS * 16, 'unknown_oe_policy': deepcopy(export.UNKNOWN_OE_POLICY),
        'unknown_oe_multiplier': .25, 'effective_unknown_oe_coefficient': .125,
        'ios_rows': STEPS * 24, 'ios_by_family': deepcopy(ios['by_family']),
        'ios_by_glyph_count': deepcopy(ios['by_glyph_count']), 'ios_sampling': deepcopy(ios),
        'focus_supervised_rows': STEPS * 32, 'focus_by_family': focus_family,
        'focus_by_glyph_count': lengths, 'focus_teacher_eligible_rows': 68712,
        'replay_sampling': replay, 'replay_unknown_by_source': deepcopy(unknown),
        'replay_by_family': deepcopy(families), 'focus_sampling': focus}


def test_native_ios_objective_is_exact():
    export.validate_v1_objective(OBJECTIVE)
    changed = deepcopy(OBJECTIVE); changed['derived_short_crops_used'] = True
    with pytest.raises(ValueError): export.validate_v1_objective(changed)


@pytest.mark.parametrize('field,value', [
    ('focus_weight', .4), ('ios_batch', 23), ('ios_weight', .5),
    ('unknown_oe_multiplier', .5), ('effective_unknown_oe_coefficient', .25),
    ('ios_loss', 'classification only')])
def test_export_rejects_changed_three_group_objective(field, value):
    changed = deepcopy(OBJECTIVE); changed[field] = value
    with pytest.raises(ValueError): export.validate_v1_objective(changed)


def test_complete_replay_and_focus_counts_pass():
    counts = counts_fixture()
    assert export.validate_counts(counts, FAMILIES) is counts


@pytest.mark.parametrize('field,value', [('coefficient', 1.0), ('denominator', 32),
    ('focus_rows_regularized', True), ('unknown_probability_target', .5),
    ('original_full_teacher_kl_preserved', False), ('inference_rule', True)])
def test_export_rejects_changed_oe_objective(field, value):
    changed = deepcopy(OBJECTIVE); changed['unknown_oe'][field] = value
    with pytest.raises(ValueError): export.validate_v1_objective(changed)


@pytest.mark.parametrize('field,value', [('unknown_oe_rows', 0), ('unknown_oe_rows', 48000),
                                       ('unknown_oe_policy', {})])
def test_export_rejects_missing_or_focus_inclusive_oe_budget(field, value):
    counts = counts_fixture(); counts[field] = value
    with pytest.raises(ValueError): export.validate_counts(counts, FAMILIES)


@pytest.mark.parametrize('fault', ['supervision', 'unknown_source', 'unknown_distribution',
    'replay_family', 'replay_face', 'focus_source', 'focus_view', 'focus_face', 'focus_length',
    'teacher_eligible', 'oe_multiplier', 'oe_effective', 'ios_rows', 'ios_family', 'ios_length',
    'ios_view', 'ios_face', 'ios_identity'])
def test_budget_mutations_with_preserved_totals_are_rejected(fault):
    counts = counts_fixture()
    if fault == 'supervision': counts['focus_supervised_rows'] = 0
    elif fault == 'unknown_source':
        counts['replay_unknown_by_source'] = {'not-a-real-source': STEPS * 16}
        counts['replay_sampling']['unknown_source_rows'] = counts['replay_unknown_by_source']
    elif fault == 'unknown_distribution':
        sources = counts['replay_unknown_by_source']; names = list(sources)
        sources[names[0]] += 1; sources[names[1]] -= 1
        counts['replay_sampling']['unknown_source_rows'] = deepcopy(sources)
    elif fault == 'replay_family':
        counts['replay_by_family']['PingFang'] -= 1; counts['replay_by_family']['SF Pro'] += 1
        counts['replay_sampling']['family_rows'] = deepcopy(counts['replay_by_family'])
    elif fault == 'replay_face':
        faces = counts['replay_sampling']['known_supplement_face_rows']
        faces[0]['rows'] += 1; faces[1]['rows'] -= 1
    elif fault == 'focus_source':
        counts['focus_sampling']['by_unknown_source']['Yusei Magic'] = counts['focus_sampling']['by_unknown_source'].pop('Lato')
    elif fault == 'focus_view':
        views = counts['focus_sampling']['by_view']; views['not-a-view'] = views.pop('native')
    elif fault == 'focus_face':
        faces = counts['focus_sampling']['by_face']; faces[0]['font_face'] = ''
    elif fault == 'focus_length':
        lengths = counts['focus_by_glyph_count']; lengths['3'] -= 1; lengths['4'] += 1
        counts['focus_sampling']['by_glyph_count'] = deepcopy(lengths)
    elif fault == 'teacher_eligible': counts['focus_teacher_eligible_rows'] = STEPS * 32 + 1
    elif fault == 'oe_multiplier': counts['unknown_oe_multiplier'] = .5
    elif fault == 'oe_effective': counts['effective_unknown_oe_coefficient'] = .25
    elif fault == 'ios_rows': counts['ios_rows'] -= 1
    elif fault == 'ios_family':
        counts['ios_by_family']['PingFang'] -= 1; counts['ios_by_family']['SF Pro'] += 1
        counts['ios_sampling']['by_family'] = deepcopy(counts['ios_by_family'])
    elif fault == 'ios_length':
        counts['ios_by_glyph_count']['1'] -= 1; counts['ios_by_glyph_count']['2'] += 1
        counts['ios_sampling']['by_glyph_count'] = deepcopy(counts['ios_by_glyph_count'])
    elif fault == 'ios_view':
        views = counts['ios_sampling']['by_view']; views['native'] -= 1; views['half'] += 1
    elif fault == 'ios_face':
        key = next(iter(counts['ios_sampling']['by_face'])); counts['ios_sampling']['by_face'][key] -= 1
    else:
        key = next(iter(counts['ios_sampling']['by_identity'])); counts['ios_sampling']['by_identity'][key] -= 1
    with pytest.raises(ValueError): export.validate_counts(counts, FAMILIES)


def focus_proof_fixture():
    return {'schema': 'flux-glyph-native-short-focus-proof-v1',
        'sampling': deepcopy(export.FOCUS_SAMPLING), 'focus_known_families': list(FOCUS_KNOWN_FAMILIES),
        'model_inference': False, 'calibration_read': False, 'development_read': False, 'test_read': False,
        'identity_count': 4752, 'bindings': {'/proof': 'a' * 64}}


def test_native_focus_must_equal_actual_reconstruction_and_bind_all_sources():
    proof = focus_proof_fixture()
    assert export.validate_native_focus_proof(proof, deepcopy(proof), proof['bindings']) is proof
    changed = deepcopy(proof); changed['identity_count'] -= 1
    with pytest.raises(ValueError): export.validate_native_focus_proof(changed, proof, proof['bindings'])
    with pytest.raises(ValueError): export.validate_native_focus_proof(proof, proof, {})
    changed = deepcopy(proof); changed['test_read'] = True
    with pytest.raises(ValueError): export.validate_native_focus_proof(changed, changed, proof['bindings'])


def ios_proof_fixture():
    manifest = str((export.ROOT / 'artifacts/ios-native-short-train-v1/prepared/MANIFEST.json').resolve())
    return {'schema': 'flux-glyph-ios-native-short-proof-v1',
        'sampling': deepcopy(export.IOS_SAMPLING), 'families': list(IOS_FAMILIES),
        'identity_count': 595, 'views': 2380,
        'by_family': {'PingFang': 241, 'SF Pro': 252, 'Helvetica': 102},
        'model_inference': False, 'calibration_read': False, 'development_read': False,
        'test_read': False, 'bindings': {manifest: MANIFEST_SHA}}


def test_ios_proof_must_equal_reconstruction_and_bind_manifest():
    proof = ios_proof_fixture()
    assert export.validate_ios_proof(proof, deepcopy(proof), proof['bindings']) is proof
    changed = deepcopy(proof); changed['identity_count'] -= 1
    with pytest.raises(ValueError): export.validate_ios_proof(changed, changed, changed['bindings'])
    changed = deepcopy(proof); changed['bindings'][next(iter(changed['bindings']))] = '0' * 64
    with pytest.raises(ValueError): export.validate_ios_proof(changed, changed, changed['bindings'])


def test_preflight_trace_executes_effective_oe_and_ios_coefficients():
    parts = {'unknown_oe_rows': 16, 'unknown_oe': .2, 'unknown_oe_weighted': .05,
             'ios_ce': 4., 'ios_size': .5, 'ios_loss': 1.025}
    assert export.validate_preflight_loss_trace(parts) is parts
    for field, value in [('unknown_oe_weighted', .1), ('ios_loss', 1.), ('ios_size', float('nan'))]:
        changed = deepcopy(parts); changed[field] = value
        with pytest.raises(ValueError): export.validate_preflight_loss_trace(changed)


def confusion_fixtures(candidate_count=26, baseline_count=27):
    rows = [{'domain': 'android', 'family': '__unknown__', 'source_id': f's{i}',
             'region_id': f'r{i}'} for i in range(30)]
    def details(count):
        return [{'domain': row['domain'], 'family': row['family'],
                 'source_id': row['source_id'], 'region_id': row['region_id'],
                 'wrong_named': i < count, 'predicted_family': 'PingFang' if i < count else 'Roboto'}
                for i, row in enumerate(rows)]
    return details(candidate_count), rows, details(baseline_count)


def test_android_system_confusion_is_an_extra_fixed_r22_guard():
    current, rows, baseline = confusion_fixtures()
    result = export.validate_android_system_confusion(current, rows, baseline)
    assert result == {'baseline_count': 27, 'maximum_count': 27, 'actual_count': 26,
        'passed': True, 'predicted_families': ['PingFang', 'SF Pro', 'Helvetica'],
        'scope': 'all Android CAL regions, including known truth and true unknowns; wrong named decisions only',
        'additional_to_original_53_cal_checks': True}
    with pytest.raises(ValueError):
        export.validate_android_system_confusion(*confusion_fixtures(candidate_count=28))
    current, rows, baseline = confusion_fixtures(baseline_count=26)
    with pytest.raises(ValueError): export.validate_android_system_confusion(current, rows, baseline)


def test_parity_schema_is_single_cnn_and_53_cal_checks():
    measured = {'retention_checks': [{'passed': True} for _ in range(46)]}
    added = {'passed': True, 'checks': [{'passed': True} for _ in range(7)]}
    selection = {'calibration_promotion_allowed': True, 'promotion_allowed': False,
        'selected': {}, 'training_teacher_proof': {'policy': {}}, 'native_focus_proof': focus_proof_fixture(),
        'ios_short_proof': ios_proof_fixture(),
        'training_protocol_sha256': 'a' * 64, 'training_counts_sha256': 'b' * 64,
        'initializer_state_sha256': 'c' * 64, 'state_after_sha256': 'd' * 64,
        'initial_parameter_groups_sha256': {}, 'selected_parameter_groups_sha256': {},
        'objective': {}, 'focus_sampling': {}}
    parity = export.build_parity_report(selection, '1' * 64, '2' * 64, '3' * 64, '4' * 64,
        {'source': 'sha'}, {'cal': 'sha'}, {'logits': 1e-5, 'size': 1e-5}, .01,
        [], measured, added, {'baseline_count': 27, 'maximum_count': 27, 'actual_count': 26,
                              'passed': True, 'predicted_families': list(export.IOS_SYSTEM_FAMILIES)})
    assert parity['schema'] == 'flux-glyph-unified-native-ios-onnx-parity-v1'
    assert parity['calibration_checks_passed'] == 53 and parity['font_model_count'] == 1
    assert parity['training']['native_focus_identities'] == 4752
    assert parity['training']['cnn_input_rows_per_step'] == 152
    assert parity['training']['ios_native_identities'] == 595
    assert parity['training']['ios_rows'] == 57600
    assert parity['training']['effective_unknown_oe_coefficient'] == .125
    assert parity['android_system_font_confusion']['actual_count'] == 26
    assert parity['calibration_checks_passed'] == 53
    assert parity['training']['derived_crops_used'] is False
    assert parity['promotion_allowed'] is False


def test_baseline_paths_are_the_actual_r22_release_evidence():
    from training import evaluate_unified_native_ios as evaluator
    assert export.R22_RUN.as_posix().endswith('/artifacts/unified-font-v3/run-wide-micro-recovery-v1')
    assert export.R22_METADATA.as_posix().endswith('/artifacts/unified-font-v3/region-wide-micro-recovery-v1/metadata.json')
    assert evaluator.R22_BASELINE.parent == export.R22_RUN
    assert evaluator.R21_BASELINE.as_posix().endswith('/artifacts/unified-font-v1/run-v1/DEVELOPMENT_REGRESSION.json')
