"""Wide export identity, deterministic reconstruction and dual-teacher metadata."""
from collections import defaultdict
from copy import deepcopy
import ast
import inspect

import pytest

torch = pytest.importorskip('torch')

from training import export_unified_retention_wide as module
import train_unified_retention_wide as trainer
from wide_region_network import widen_region_model
from region_network import RegionFontClassifier

ACCEPTANCE_PLAN_SHA = 'b82283b54b5e47f4639b0df62231544c0419f6299d10820d86dca96ef24094ff'
WIDE_PLAN_FIXTURE = module.ROOT/'tests/fixtures/unified_retention_wide_objective_plan.json'


def final_selection():
    record = {'step': 6000, 'promotion_allowed': True, 'metrics': {'passed': True}}
    return {'history': [record], 'selected': deepcopy(record), 'passed': True,
        'checkpoint_selection': 'Fixed final step 6000; no intermediate CAL inference or checkpoint search'}


def widened_fixture():
    torch.manual_seed(1414)
    narrow = RegionFontClassifier(25).eval()
    with torch.no_grad():
        narrow.size_head.weight.normal_(0., .03);narrow.size_head.bias.fill_(.1)
    wide, report = widen_region_model(narrow, trainer.SEED)
    groups = {name: module.state_sha({key: value for key, value in wide.state_dict().items()
                                     if key.startswith(name+'.')}) for name in module.GROUPS}
    identity = {'selection': {'sha256': 'a'*64}, 'state_sha256': module.state_sha(narrow.state_dict())}
    selection = {'families': [str(i) for i in range(25)], 'student_initializer': identity,
        'named_teacher_identity': identity, 'named_teacher_state_sha256': identity['state_sha256'],
        'initial_state_sha256': module.state_sha(wide.state_dict()),
        'initial_parameter_groups_sha256': groups, 'widening': report}
    checkpoint = {'families': selection['families'], 'architecture': trainer.narrow.ARCHITECTURE,
        'selection_sha256': 'a'*64, 'state_dict': deepcopy(narrow.state_dict())}
    return narrow, wide, checkpoint, selection


def test_only_fixed_final_6000_step_is_exportable():
    selection = final_selection()
    assert module.validate_fixed_final_selection(selection) == selection['history']
    for fault in ('extra', 'searched', 'different', 'failed'):
        broken = deepcopy(selection)
        if fault == 'extra': broken['history'].insert(0, {'step': 3000})
        elif fault == 'searched': broken['checkpoint_selection'] = 'best CAL checkpoint'
        elif fault == 'different': broken['selected']['step'] = 5999
        else: broken['selected']['promotion_allowed'] = False
        with pytest.raises(ValueError): module.validate_fixed_final_selection(broken)


def test_wide_objective_plan_is_bound_separately(monkeypatch, tmp_path):
    retention = tmp_path/'RETENTION_PLAN.json';retention.write_text('{}')
    objective = tmp_path/'PLAN.json';objective.write_text('{}')
    selection = {'objective_plan': {'path': str(objective), 'sha256': module.sha(objective)},
        'bindings': {str(objective): module.sha(objective)}}
    seen = {}
    monkeypatch.setattr(trainer, 'read_objective_plan',
        lambda path, digest: seen.update(path=path, digest=digest) or {'passed': True})
    assert module.validate_objective_plan(selection, retention) == {'passed': True}
    assert seen == {'path': objective, 'digest': module.sha(retention)}


def test_actual_wide_plan_matches_6000_step_dual_teacher_objective():
    plan = trainer.read_objective_plan(WIDE_PLAN_FIXTURE, ACCEPTANCE_PLAN_SHA)
    assert plan['steps'] == plan['eval_every'] == 6000 and plan['seed'] == 2026091414
    assert plan['teacher_policy'] == trainer.TEACHER_POLICY
    assert plan['objective']['teacher_distribution_kl'] is True


def test_narrow_b08_reconstructs_exact_recorded_wide_state_without_source_mutation():
    narrow, wide, checkpoint, selection = widened_fixture()
    before = {name: value.clone() for name, value in checkpoint['state_dict'].items()}
    reconstructed = module.reconstruct_widened_initializer(checkpoint, selection)
    assert module.state_sha(reconstructed.state_dict()) == module.state_sha(wide.state_dict())
    assert all(torch.equal(value, checkpoint['state_dict'][name]) for name, value in before.items())


@pytest.mark.parametrize('fault', ['source_state', 'seed_report', 'wide_state', 'group'])
def test_widened_reconstruction_fails_closed_on_identity_drift(fault):
    _, _, checkpoint, selection = widened_fixture()
    if fault == 'source_state': selection['named_teacher_state_sha256'] = '0'*64
    elif fault == 'seed_report': selection['widening']['seed'] += 1
    elif fault == 'wide_state': selection['initial_state_sha256'] = '0'*64
    else: selection['initial_parameter_groups_sha256']['style'] = '0'*64
    with pytest.raises(ValueError): module.reconstruct_widened_initializer(checkpoint, selection)


def test_train_only_widening_parity_record_binds_narrow_and_wide_states():
    _, _, _, selection = widened_fixture()
    selection['widening_parity'] = {'passed': True, 'train_tiles': 384,
        'source_state_sha256': selection['named_teacher_state_sha256'],
        'wide_state_sha256': selection['initial_state_sha256'],
        'source_parameters_unchanged': True, 'wide_parameters_unchanged': True,
        'optimizer_steps': 0, 'calibration_read': False, 'development_read': False, 'test_read': False}
    module.validate_widening_record(selection)
    selection['widening_parity']['calibration_read'] = True
    with pytest.raises(ValueError): module.validate_widening_record(selection)


def test_selected_checkpoint_requires_changed_wide_groups():
    _, wide, _, selection = widened_fixture()
    final = deepcopy(wide.state_dict())
    for name in module.GROUPS:
        key = next(key for key in final if key.startswith(name+'.'))
        final[key] = final[key] + .01
    selection.update(state_after_sha256=module.state_sha(final),
        selected_parameter_groups_sha256={name: module.state_sha({
            key: value for key, value in final.items() if key.startswith(name+'.')}) for name in module.GROUPS})
    checkpoint = {'state_dict': final, 'families': selection['families'],
        'architecture': module.ARCHITECTURE, 'selection_sha256': 's'}
    module.validate_checkpoint(checkpoint, selection, 's')
    selection['selected_parameter_groups_sha256']['style'] = selection['initial_parameter_groups_sha256']['style']
    with pytest.raises(ValueError, match='group was not trained'):
        module.validate_checkpoint(checkpoint, selection, 's')


def test_training_metadata_keeps_two_offline_teachers_and_one_deployed_encoder():
    selection = defaultdict(lambda: None);selection.update(final_selection())
    selection.update(objective_variant=trainer.OBJECTIVE_VARIANT, objective=trainer.OBJECTIVE,
        unknown_floor_supervision=trainer.UNKNOWN_FLOOR_SUPERVISION,
        named_teacher_identity={'state_sha256': 'a'}, named_teacher_cache={'sha256': 'b'},
        named_teacher_state_sha256='a', unknown_teacher_identity={'state_sha256': 'c'},
        teacher_policy=trainer.TEACHER_POLICY, teacher_mix={}, widening={}, widening_parity={},
        offline_teacher_count=2, teacher_models_resident_during_optimizer=0,
        calibration_reused_for_prior_development=True)
    result = module.training_metadata(selection)
    assert result['teacher_distribution_kl'] is True
    assert result['offline_teacher_count'] == 2 and result['teacher_models_resident_during_optimizer'] == 0
    assert result['model_count'] == result['encoder_count'] == 1
    assert result['teacher_in_deployed_model'] is False and result['score_merging'] is False


def test_actual_parity_builder_preserves_and_validates_mps_training_device():
    _, _, _, selection = widened_fixture()
    student = selection['student_initializer']
    selection.update(
        supplement_data={'path': 'supplement', 'sha256': '1'*64},
        supplement_cache={'path': 'supplement-cache', 'sha256': '2'*64},
        known_data={'path': 'known', 'sha256': '3'*64},
        known_cache={'path': 'known-cache', 'sha256': '4'*64},
        named_teacher_cache={'path': 'named-cache', 'sha256': '0'*64},
        unknown_teacher_identity={'state_sha256': '5'*64},
        teacher_policy={'selection_input': 'true target'}, teacher_mix={'original': {'tiles': 1}},
        widening_parity={'passed': True}, historical_core_caches_used_for_training=False,
        source_optimizer_steps_executed=3000, source_selected_step=1500,
        core_optimizer_steps_executed=1500, base_selected_step=1000,
        training_data_counts={'combined_tiles': 3}, retention_plan={'sha256': '6'*64},
        objective_plan={'path': 'objective-plan', 'sha256': '7'*64},
        base_checkpoint={'path': 'base', 'sha256': '8'*64},
        base_selection={'path': 'base-selection', 'sha256': '9'*64},
        base_state_sha256='a'*64, cache_manifest={'path': 'cache', 'sha256': 'b'*64},
        state_before_sha256=selection['initial_state_sha256'], state_after_sha256='c'*64,
        objective_variant=trainer.OBJECTIVE_VARIANT,
        objective=trainer.OBJECTIVE, training_counts_sha256='d'*64,
        unknown_floor_supervision=trainer.UNKNOWN_FLOOR_SUPERVISION,
        sampling=trainer.SAMPLING, unknown_source_order=['source'], design_changes=trainer.DESIGN_CHANGES,
        single_change_causal_attribution=False,
        selected_parameter_groups_sha256=dict.fromkeys(module.GROUPS, 'e'*64),
        final_parameter_groups_sha256=dict.fromkeys(module.GROUPS, 'f'*64),
        checkpoint_selection='Fixed final step 6000; no intermediate CAL inference or checkpoint search',
        calibration_reused_for_prior_development=True, training_device='mps', selected={'step': 6000})
    assert selection['named_teacher_identity'] == student
    measured = {'metrics': {}}
    report = module.build_parity_report(selection, '1'*64, '2'*64, '3'*64, '8'*64,
        {'source': '9'*64}, {'evidence': 'a'*64}, {'logits': 0., 'size': 0.}, 0.,
        46, 137, [{'batch_size': 1}], measured)
    assert report['training_device'] == selection['training_device'] == 'mps'
    assert report['student_initializer'] == report['named_teacher_identity'] == student
    assert report['unknown_teacher_identity'] == selection['unknown_teacher_identity']
    assert report['source_bindings'] == {'source': '9'*64}
    assert report['calibration_bindings'] == {'evidence': 'a'*64}
    for invalid in (None, 'cuda'):
        selection['training_device'] = invalid
        with pytest.raises(ValueError, match='training device provenance'):
            module.build_parity_report(selection, '1', '2', '3', '4', {}, {}, {}, 0., 0, 0, [], measured)


def test_teacher_mix_allows_target_pure_partitions_but_requires_both_teachers_in_union():
    digest = 'a'*64
    mix = {
        'original': {'tiles': 7, 'named_teacher_tiles': 4, 'unknown_teacher_tiles': 3,
                     'logits_sha256': digest, 'labels_sha256': digest},
        'supplement': {'tiles': 5, 'named_teacher_tiles': 0, 'unknown_teacher_tiles': 5,
                       'logits_sha256': digest, 'labels_sha256': digest},
        'known': {'tiles': 6, 'named_teacher_tiles': 6, 'unknown_teacher_tiles': 0,
                  'logits_sha256': digest, 'labels_sha256': digest},
    }
    module.validate_teacher_mix(mix, ('original', 'supplement', 'known'))
    broken = deepcopy(mix)
    for item in broken.values():
        item['named_teacher_tiles'] = 0
        item['unknown_teacher_tiles'] = item['tiles']
    with pytest.raises(ValueError, match='mixture evidence'):
        module.validate_teacher_mix(broken, ('original', 'supplement', 'known'))


def test_validate_path_has_no_torch_import():
    tree = ast.parse(inspect.getsource(module.validate))
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom))
                   and any(alias.name == 'torch' for alias in node.names) for node in ast.walk(tree))
