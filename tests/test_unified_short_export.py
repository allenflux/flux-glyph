"""Portable report, count, checkpoint and CAL gates for the short-region exporter."""
from copy import deepcopy

import pytest

from training import export_unified_short_regions as export
from training import short_region_objective as short


def counts_fixture():
    families = list(short.FAMILIES)
    short_family = {name: export.STEPS for name in families[:-1]}
    short_family['__unknown__'] = export.STEPS * 8
    replay_sources = sorted([*short.SHORT_UNKNOWN_SOURCES, 'WenQuanYi Zen Hei', 'Zhuque Fangsong'])
    quotient, remainder = divmod(export.STEPS * 16, len(replay_sources))
    replay_unknown = {name: quotient + int(index < remainder)
                      for index, name in enumerate(replay_sources)}
    details = [{'family': name, 'source_font_family': name, 'glyph_count': 1,
                'domain': 'android', 'font_face': name + '-face', 'view': 'native',
                'rows': export.STEPS} for name in families[:-1]]
    quotient, remainder = divmod(export.STEPS * 8, len(short.SHORT_UNKNOWN_SOURCES))
    details += [{'family': '__unknown__', 'source_font_family': name, 'glyph_count': 1,
                 'domain': 'ios', 'font_face': name + '-face', 'view': 'native',
                 'rows': quotient + int(index < remainder)}
                for index, name in enumerate(short.SHORT_UNKNOWN_SOURCES)]
    replay_families = {name: export.STEPS * 2 for name in families[:-1]}
    replay_families['__unknown__'] = export.STEPS * 16
    replay_families['PingFang'] += export.STEPS * 16
    replay_families['SF Pro'] += export.STEPS * 4
    replay_families['Helvetica'] += export.STEPS * 4
    for name in export.REPLAY_SAMPLING['original_eight_families']:
        replay_families[name] += export.STEPS
    replay_sampling = {'schema': 'flux-glyph-retention-face-balanced-sampling-v1',
        'unknown_source_order': replay_sources, 'unknown_source_rows': replay_unknown,
        'unknown_rows': export.STEPS * 16, 'family_rows': replay_families,
        'known_supplement_rows': export.STEPS * 2,
        'known_supplement_source_rows': {'LXGW WenKai': export.STEPS,
                                         'WenQuanYi Micro Hei': export.STEPS},
        'known_supplement_face_rows': [
            {'family': 'LXGW WenKai', 'font_face': 'LXGWWenKai-Light', 'rows': export.STEPS // 3},
            {'family': 'LXGW WenKai', 'font_face': 'LXGWWenKai-Medium', 'rows': export.STEPS // 3},
            {'family': 'LXGW WenKai', 'font_face': 'LXGWWenKai-Regular', 'rows': export.STEPS // 3},
            {'family': 'WenQuanYi Micro Hei', 'font_face': 'WenQuanYiMicroHei', 'rows': export.STEPS}]}
    return {'optimizer_steps': export.STEPS, 'replay_rows': export.STEPS * 96,
            'short_rows': export.STEPS * 32, 'short_by_family': short_family,
            'replay_unknown_by_source': replay_unknown, 'replay_by_family': replay_families,
            'replay_sampling': replay_sampling,
            'short_sampling': {'steps': export.STEPS, 'rows': export.STEPS * 32,
                'unknown_source_order': list(short.SHORT_UNKNOWN_SOURCES), 'counts': details}}, families


def metrics():
    return {'passed': False, 'known_correct_coverage': .66, 'named_precision': .95,
            'unknown_not_named_rate': .86,
            'per_domain': {'ios': {'known_correct_coverage': .75},
                           'android': {'known_correct_coverage': .56}}}


def calibration_fixture():
    old = metrics(); current = deepcopy(old)
    old_short = {'known_views': 100, 'correct_named': 50, 'known_correct_coverage': .5,
                 'unknown_views': 50, 'unknown_falsely_named': 10}
    now_short = {**old_short, 'correct_named': 52, 'known_correct_coverage': .52}
    measured = {'step': export.STEPS, 'temperature': 1., 'gates': deepcopy(export.FIXED_RUNTIME['gates']),
                'metrics': current, 'retention_checks': [{'name': str(i), 'passed': True} for i in range(46)],
                'promotion_allowed': True, 'retention_populations': {}}
    added = short.compare_r22(current, old, now_short, old_short)
    selection = {'selected': deepcopy(measured), 'short_metrics': now_short,
                 'short_baseline': old_short, 'baseline_metrics': old,
                 'r22_comparison': added, 'calibration_promotion_allowed': True,
                 'promotion_allowed': False, 'development_evaluated': False}
    return selection, measured, now_short, old, old_short, added


def selection_fixture():
    selection, measured, _, _, _, added = calibration_fixture()
    selection.update({'families': list(short.FAMILIES), 'training_protocol_sha256': 'a' * 64,
        'training_counts_sha256': 'b' * 64, 'initializer_state_sha256': 'c' * 64,
        'state_after_sha256': 'd' * 64,
        'initial_parameter_groups_sha256': {name: name + '-old' for name in export.GROUPS},
        'selected_parameter_groups_sha256': {name: name + '-new' for name in export.GROUPS},
        'objective': {'short_teacher': None}, 'short_sampling': deepcopy(short.SHORT_SAMPLING)})
    return selection, measured, added


def test_exact_short_and_replay_counts_are_accepted():
    counts, families = counts_fixture()
    assert export.validate_counts(counts, families) is counts


@pytest.mark.parametrize('fault', ['steps', 'replay_total', 'short_total', 'known_family',
                                   'unknown_family', 'replay_source', 'replay_family',
                                   'replay_face', 'heldout', 'detail_total'])
def test_count_and_source_isolation_mutations_fail(fault):
    counts, families = counts_fixture()
    if fault == 'steps': counts['optimizer_steps'] -= 1
    elif fault == 'replay_total': counts['replay_rows'] -= 1
    elif fault == 'short_total': counts['short_rows'] -= 1
    elif fault == 'known_family': counts['short_by_family'][families[0]] -= 1
    elif fault == 'unknown_family': counts['short_by_family']['__unknown__'] -= 1
    elif fault == 'replay_source': counts['replay_unknown_by_source'].pop(next(iter(counts['replay_unknown_by_source'])))
    elif fault == 'replay_family': counts['replay_sampling']['family_rows']['PingFang'] -= 1
    elif fault == 'replay_face': counts['replay_sampling']['known_supplement_face_rows'][0]['rows'] -= 1
    elif fault == 'heldout': counts['short_sampling']['counts'][-1]['source_font_family'] = 'Yusei Magic'
    else: counts['short_sampling']['counts'][0]['rows'] -= 1
    with pytest.raises(ValueError):
        export.validate_counts(counts, families)


def test_calibration_contract_requires_all_46_plus_7_and_pre_dev_state():
    selection, measured, now_short, old, old_short, added = calibration_fixture()
    assert export.calibration_contract(selection, measured, now_short, old, old_short) == added
    for mutation in ('retention', 'short', 'top_promotion', 'development'):
        changed_selection, changed, changed_short, base, base_short, _ = calibration_fixture()
        if mutation == 'retention': changed['retention_checks'][0]['passed'] = False
        elif mutation == 'short': changed_short['unknown_falsely_named'] += 1
        elif mutation == 'top_promotion': changed_selection['promotion_allowed'] = True
        else: changed_selection['development_evaluated'] = True
        with pytest.raises(ValueError):
            export.calibration_contract(changed_selection, changed, changed_short, base, base_short)


def test_runtime_calibration_accepts_small_float_report_drift_without_selection_equality():
    _, measured, now_short, old, old_short, _ = calibration_fixture()
    measured['metrics']['named_precision'] += 1e-7
    measured['metrics']['per_domain']['ios']['known_correct_coverage'] += 1e-7
    added = export.runtime_calibration_contract(measured, now_short, old, old_short)
    assert added['passed'] is True and len(added['checks']) == 7


def test_raw_metadata_and_parity_are_cal_only_single_cnn_without_teacher_lineage():
    selection, measured, added = selection_fixture()
    cal = {'manifest': {'font_label_groups': {'PingFang': ['PingFang SC']}}}
    parent = {'font_sources': {'PingFang': {'kind': 'ios_system'}}}
    metadata = export.metadata_for_export(selection, cal, parent, '1' * 64, '2' * 64,
                                          '3' * 64, measured, added)
    assert metadata['release_tier'] == 'experimental'
    assert metadata['stable_validation_passed'] is metadata['test_passed'] is False
    assert metadata['validation']['calibration_checks'] == 53
    assert metadata['validation']['promotion_allowed'] is False
    assert metadata['training']['model_count'] == 1
    assert metadata['training']['r22_same_tile_teacher_cache_training_only'] is True
    forbidden = {'offline_teacher_count', 'named_teacher_identity', 'unknown_teacher_identity',
                 'teacher_mix', 'historical_teacher_mix'}
    assert not forbidden & set(metadata['training'])
    parity = export.build_parity_report(selection, '3' * 64, '2' * 64, '1' * 64,
        '4' * 64, {'source': 'sha'}, {'evidence': 'sha'}, {'logits': 1e-5, 'size': 2e-5},
        .01, [{'batch_size': n, 'passed': True} for n in (1, 7, 32, 128)], measured, added)
    assert parity['passed'] is True and parity['calibration_promotion_allowed'] is True
    assert parity['promotion_allowed'] is parity['development_evaluated'] is False
    assert parity['font_model_count'] == parity['encoder_count'] == 1
    assert parity['calibration_checks_passed'] == 53 and parity['test_read'] is False


def test_checkpoint_requires_exact_final_state_and_all_four_group_hashes():
    torch = pytest.importorskip('torch')
    from wide_region_network import WideRegionFontClassifier
    model = WideRegionFontClassifier(25)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    selection_sha = 'e' * 64
    groups = {name: export.state_sha({key: value for key, value in state.items()
                                     if key.startswith(name + '.')}) for name in export.GROUPS}
    selection = {'families': list(short.FAMILIES), 'state_after_sha256': export.state_sha(state),
                 'training_protocol_sha256': 'f' * 64,
                 'selected_parameter_groups_sha256': groups}
    checkpoint = {'architecture': export.ARCHITECTURE, 'families': list(short.FAMILIES),
                  'selection_sha256': selection_sha, 'state_dict': state}
    final = {'architecture': export.ARCHITECTURE, 'families': list(short.FAMILIES),
             'step': export.STEPS, 'training_protocol_sha256': 'f' * 64, 'state_dict': state}
    export.validate_checkpoint(checkpoint, final, selection, selection_sha)
    checkpoint['selection_sha256'] = 'wrong'
    with pytest.raises(ValueError):
        export.validate_checkpoint(checkpoint, final, selection, selection_sha)


def test_initializer_groups_must_come_from_exact_pinned_checkpoint():
    torch = pytest.importorskip('torch')
    from wide_region_network import WideRegionFontClassifier
    model = WideRegionFontClassifier(25)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    digest = export.state_sha(state)
    groups = {name: export.state_sha({key: value for key, value in state.items()
                                     if key.startswith(name + '.')}) for name in export.GROUPS}
    selection_sha = 'a' * 64
    selection = {'families': list(short.FAMILIES), 'initializer_state_sha256': digest,
                 'initializer_selection_sha256': selection_sha,
                 'initial_parameter_groups_sha256': groups}
    checkpoint = {'architecture': export.ARCHITECTURE, 'families': list(short.FAMILIES),
                  'selection_sha256': selection_sha, 'state_dict': state}
    export.validate_initializer_checkpoint(checkpoint, selection, digest, selection_sha)
    selection['initial_parameter_groups_sha256']['style'] = 'wrong'
    with pytest.raises(ValueError):
        export.validate_initializer_checkpoint(checkpoint, selection, digest, selection_sha)
