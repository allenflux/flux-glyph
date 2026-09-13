"""Confidence-floor export identity, fixed selection and TRAIN accounting."""
from collections import defaultdict
from copy import deepcopy

import pytest

from training import export_unified_retention_confidence_floor as module
import train_unified_retention_confidence_floor as trainer
import test_unified_retention_paired_known_export as paired

ACCEPTANCE_PLAN_SHA = 'b82283b54b5e47f4639b0df62231544c0419f6299d10820d86dca96ef24094ff'
OBJECTIVE_PLAN_FIXTURE = module.ROOT/'tests/fixtures/unified_retention_confidence_floor_objective_plan.json'


def final_selection():
    record = {'step': 3000, 'promotion_allowed': True, 'metrics': {'passed': True}}
    return {'history': [record], 'selected': deepcopy(record), 'passed': True,
        'checkpoint_selection': 'Fixed final step 3000; no intermediate CAL inference or checkpoint search'}


def test_only_final_step_is_eligible_for_export():
    selection = final_selection()
    assert module.validate_fixed_final_selection(selection) == selection['history']


def test_objective_plan_is_bound_separately_to_acceptance_plan(monkeypatch, tmp_path):
    retention = tmp_path/'RETENTION_PLAN.json'; retention.write_text('{"constraints": 46}')
    objective = tmp_path/'PLAN.json'; objective.write_text('{"objective": "confidence-floor"}')
    selection = {'objective_plan': {'path': str(objective), 'sha256': module.sha(objective)},
        'bindings': {str(objective): module.sha(objective)}}
    observed = {}
    def read_objective_plan(path, acceptance_sha):
        observed.update(path=path, acceptance_sha=acceptance_sha)
        return {'validated': True}
    monkeypatch.setattr(trainer, 'read_objective_plan', read_objective_plan, raising=False)
    assert module.validate_objective_plan(selection, retention) == {'validated': True}
    assert observed == {'path': objective, 'acceptance_sha': module.sha(retention)}


def test_actual_pretraining_objective_plan_matches_frozen_acceptance():
    plan = trainer.read_objective_plan(OBJECTIVE_PLAN_FIXTURE, ACCEPTANCE_PLAN_SHA)
    assert plan['objective'] == trainer.OBJECTIVE
    assert plan['runtime'] == trainer.FIXED_RUNTIME
    assert plan['checkpoint_selection'] == 'fixed final step 3000, no intermediate candidate selection'


@pytest.mark.parametrize('fault', ['objective', 'final_step'])
def test_tampered_objective_or_final_step_plan_is_rejected(tmp_path, fault):
    plan = trainer.read(OBJECTIVE_PLAN_FIXTURE)
    if fault == 'objective':
        plan['objective']['unknown_floor_supervision']['minimum_unknown_probability'] = .49
    else:
        plan['checkpoint_selection'] = 'best calibration checkpoint'
    path = tmp_path/'PLAN.json'; module.dump(path, plan)
    with pytest.raises(ValueError):
        trainer.read_objective_plan(path, ACCEPTANCE_PLAN_SHA)


@pytest.mark.parametrize('fault', ['intermediate', 'searched', 'different_selected', 'failed'])
def test_cal_cannot_choose_a_confidence_floor_checkpoint(fault):
    selection = final_selection()
    if fault == 'intermediate':
        selection['history'].insert(0, {'step': 1500, 'promotion_allowed': True, 'metrics': {'passed': True}})
    elif fault == 'searched':
        selection['checkpoint_selection'] = 'Best CAL candidate'
    elif fault == 'different_selected':
        selection['selected']['step'] = 2999
    else:
        selection['selected']['promotion_allowed'] = False
    with pytest.raises(ValueError):
        module.validate_fixed_final_selection(selection)


def test_training_metadata_names_the_monotone_floor_objective():
    selection = defaultdict(lambda: None)
    selection.update(final_selection())
    selection.update(objective_variant=trainer.OBJECTIVE_VARIANT, objective=deepcopy(trainer.OBJECTIVE),
        unknown_floor_supervision=deepcopy(trainer.UNKNOWN_FLOOR_SUPERVISION),
        calibration_reused_for_prior_development=True)
    result = module.training_metadata(selection)
    assert result['unknown_floor_supervision'] == trainer.UNKNOWN_FLOOR_SUPERVISION
    assert result['unknown_floor_supervision']['minimum_unknown_probability'] == .5
    assert result['unknown_floor_supervision']['penalizes_confidence_above_floor'] is False
    assert result['objective']['teacher_confidence_floor_weight'] == 2.
    assert result['objective']['teacher_distribution_kl'] is False
    assert 'unknown_binary_supervision' not in result
    assert result['teacher_distribution_kl'] is False


@pytest.fixture(scope='module')
def actual_counts():
    counts, known = deepcopy(paired.actual_counts.__wrapped__())
    counts.pop('unknown_binary_supervision', None)
    counts.update(schema='flux-glyph-retention-confidence-floor-counts-v1',
        objective_variant=trainer.OBJECTIVE_VARIANT, objective=trainer.OBJECTIVE,
        unknown_floor_supervision=trainer.UNKNOWN_FLOOR_SUPERVISION,
        mask_rule=trainer.OBJECTIVE['teacher_mask'], teacher_mask_verified_against_same_tile_argmax=True,
        eligible_rows=288000, eligible_rows_per_step=[96]*3000,
        eligible_family_rows=deepcopy(counts['family_rows']), eligible_domain_rows=deepcopy(counts['domain_rows']))
    for row in counts['source_target_rows']:
        row['eligible_rows'] = row['rows']
    return counts, known


def test_same_truth_correct_train_mask_and_counts_are_preserved(monkeypatch, tmp_path, actual_counts):
    run, data, selection, _, counts = paired.fixture_counts(monkeypatch, tmp_path, actual_counts)
    module.validate_counts(run, selection, data)
    assert counts['eligible_family_rows']['__unknown__'] == 48000
    assert counts['eligible_rows'] == counts['rows'] == 288000


@pytest.mark.parametrize('fault', ['old_supervision', 'wrong_objective', 'unchecked_mask'])
def test_wrong_floor_or_unverified_teacher_mask_is_rejected(monkeypatch, tmp_path, actual_counts, fault):
    run, data, selection, sampling, counts = paired.fixture_counts(monkeypatch, tmp_path, actual_counts)
    if fault == 'old_supervision':
        counts['unknown_binary_supervision'] = {'obsolete': True}
    elif fault == 'wrong_objective':
        counts['objective'] = paired.trainer.OBJECTIVE
    else:
        counts['teacher_mask_verified_against_same_tile_argmax'] = False
    module.dump(run/'TRAINING_COUNTS.json', counts)
    module.dump(run/'SAMPLING.json', sampling)
    selection['training_counts_sha256'] = module.sha(run/'TRAINING_COUNTS.json')
    selection['sampling_sha256'] = module.sha(run/'SAMPLING.json')
    with pytest.raises(ValueError):
        module.validate_counts(run, selection, data)
