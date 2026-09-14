"""Verify exact replay/focus accounting and the raw native ONNX boundary."""
from copy import deepcopy

import pytest

from training import export_unified_native_short as export
from training.train_unified_native_short import OBJECTIVE, STEPS, FAMILIES
from training.native_short_focus import FOCUS_KNOWN_FAMILIES, FOCUS_UNKNOWN_SOURCES, VIEWS


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
    return {'optimizer_steps': STEPS, 'replay_rows': STEPS * 96, 'focus_rows': STEPS * 32,
        'focus_supervised_rows': STEPS * 32, 'focus_by_family': focus_family,
        'focus_by_glyph_count': lengths, 'focus_teacher_eligible_rows': 50000,
        'replay_sampling': replay, 'replay_unknown_by_source': deepcopy(unknown),
        'replay_by_family': deepcopy(families), 'focus_sampling': focus}


def test_native_objective_is_exact_and_focus_only():
    export.validate_v1_objective(OBJECTIVE)
    changed = deepcopy(OBJECTIVE); changed['derived_short_crops_used'] = True
    with pytest.raises(ValueError): export.validate_v1_objective(changed)


def test_complete_replay_and_focus_counts_pass():
    counts = counts_fixture()
    assert export.validate_counts(counts, FAMILIES) is counts


@pytest.mark.parametrize('fault', ['supervision', 'unknown_source', 'unknown_distribution',
    'replay_family', 'replay_face', 'focus_source', 'focus_view', 'focus_face', 'focus_length', 'teacher_eligible'])
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
    else: counts['focus_teacher_eligible_rows'] = STEPS * 32 + 1
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


def test_parity_schema_is_single_cnn_and_53_cal_checks():
    measured = {'retention_checks': [{'passed': True} for _ in range(46)]}
    added = {'passed': True, 'checks': [{'passed': True} for _ in range(7)]}
    selection = {'calibration_promotion_allowed': True, 'promotion_allowed': False,
        'selected': {}, 'training_teacher_proof': {'policy': {}}, 'native_focus_proof': focus_proof_fixture(),
        'training_protocol_sha256': 'a' * 64, 'training_counts_sha256': 'b' * 64,
        'initializer_state_sha256': 'c' * 64, 'state_after_sha256': 'd' * 64,
        'initial_parameter_groups_sha256': {}, 'selected_parameter_groups_sha256': {},
        'objective': {}, 'focus_sampling': {}}
    parity = export.build_parity_report(selection, '1' * 64, '2' * 64, '3' * 64, '4' * 64,
        {'source': 'sha'}, {'cal': 'sha'}, {'logits': 1e-5, 'size': 1e-5}, .01,
        [], measured, added)
    assert parity['schema'] == 'flux-glyph-unified-native-short-onnx-parity-v1'
    assert parity['calibration_checks_passed'] == 53 and parity['font_model_count'] == 1
    assert parity['training']['native_focus_identities'] == 4752
    assert parity['training']['cnn_input_rows_per_step'] == 128
    assert parity['training']['derived_crops_used'] is False
    assert parity['promotion_allowed'] is False


def test_baseline_paths_are_the_actual_r22_release_evidence():
    from training import evaluate_unified_native_short as evaluator
    assert export.R22_RUN.as_posix().endswith('/artifacts/unified-font-v3/run-wide-micro-recovery-v1')
    assert export.R22_METADATA.as_posix().endswith('/artifacts/unified-font-v3/region-wide-micro-recovery-v1/metadata.json')
    assert evaluator.R22_BASELINE.parent == export.R22_RUN
    assert evaluator.R21_BASELINE.as_posix().endswith('/artifacts/unified-font-v1/run-v1/DEVELOPMENT_REGRESSION.json')
