#!/usr/bin/env python3
"""Separate product-objective calibration after preserving the old CAL failure.

Roboto is an extra competing class used to prevent wrong font names. This stage
does not certify independently identifying Roboto or adding it to named output.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'training')]
from train_regions import dump, require, sha, state_sha
from train_sans_verifier import POLICY, summarize
import calibrate_sans_verifier as original

SCHEMA = 'flux-glyph-sans-blocking-calibration-v1'
OBJECTIVE = 'wrong-name-blocking'
PROTOCOL = {
    'schema': 'flux-glyph-sans-wrong-name-blocking-protocol-v1', 'objective': OBJECTIVE,
    'agreement_grid': original.AGREEMENT_GRID,
    'fixed_original_policy': POLICY,
    'replaced_release_check': 'roboto_veto', 'replacement_release_check': 'roboto_wrong_name_blocking',
    'minimum_roboto_wrong_name_blocking_rate': .8,
    'denominator': 'Roboto views in the evaluated partition (CAL for selection, sealed TEST for evaluation) accepted and wrongly named by the original R19 primary plus rejection model',
    'numerator': 'Those baseline-wrong Roboto views no longer accepted with a wrong name by the unchanged-weight consensus model',
    'zero_denominator': 'fail', 'independent_roboto_recall_remains_a_reported_diagnostic': True,
    'formal_roboto_name_coverage': False, 'primary_and_rejection_thresholds_unchanged': True,
    'change_disclosure': 'Release objective revised after inspecting CAL failures, before opening held-out TEST; original failures remain failures.',
    'test_policy': 'Freeze one selected gate and this protocol before one held-out TEST. A failed TEST cannot be used to revise this release protocol.',
}
GRID_SPEC = {**original.GRID_SPEC, 'objective_protocol': PROTOCOL}


def effective_policy(selection):
    if selection.get('schema') != SCHEMA:
        return original.effective_policy(selection)
    value = selection.get('policy')
    agreement = value.get('gates', {}).get('min_patch_agreement') if isinstance(value, dict) else None
    expected = original.policy_for(agreement)
    require(value == expected and selection.get('objective') == OBJECTIVE and selection.get('objective_protocol') == PROTOCOL
            and selection.get('grid_spec') == GRID_SPEC, 'blocking calibration may change only its declared objective and two-choice voting gate')
    require(type(selection.get('optimizer_steps_executed')) is int and selection['optimizer_steps_executed'] == 0
            and selection.get('weights_unchanged') is True and selection.get('test_read') is False,
            'blocking calibration must preserve weights and exclude TEST')
    return expected


def blocking_metrics(rows, baseline, verifier, families, primary_families):
    metrics, details = summarize(rows, baseline, verifier, families, primary_families)
    roboto = [row for row in details if row['family'] == 'Roboto']
    before_wrong = [row for row in roboto if row['before_wrong']]
    prevented = sum(not row['after_wrong'] for row in before_wrong)
    rate = prevented / len(before_wrong) if before_wrong else None
    original_checks = dict(metrics['checks'])
    checks = {key: value for key, value in original_checks.items() if key != 'roboto_veto'}
    checks['roboto_wrong_name_blocking'] = bool(before_wrong) and rate >= PROTOCOL['minimum_roboto_wrong_name_blocking_rate']
    metrics.update(original_release_criteria_passed=metrics['passed'], original_release_checks=original_checks,
                   checks=checks, passed=all(checks.values()), release_objective=OBJECTIVE,
                   roboto_wrong_name_blocking={'baseline_wrong_names': len(before_wrong), 'prevented_wrong_names': prevented,
                       'rate': rate, 'minimum_rate': PROTOCOL['minimum_roboto_wrong_name_blocking_rate'],
                       'baseline_already_not_wrongly_named': len(roboto) - len(before_wrong)},
                   independent_roboto_diagnostic={'correct_and_above_gate': metrics['roboto']['roboto_veto'],
                       'views': metrics['roboto']['regions'], 'rate': metrics['roboto_veto_recall'],
                       'original_minimum_rate': POLICY['minimum_roboto_veto_recall'],
                       'original_check_passed': original_checks['roboto_veto'], 'formal_name_coverage_claim': False})
    return metrics, details


def summarize_selection(selection, rows, baseline, verifier, families, primary_families):
    effective_policy(selection)
    function = blocking_metrics if selection['schema'] == SCHEMA else summarize
    return function(rows, baseline, verifier, families, primary_families)


def source_evidence(source_run, previous_calibration, data, primary):
    parent, rows, before, cached, bindings = original.validate_parent(source_run, data, primary)
    previous_calibration = Path(previous_calibration).resolve()
    previous = json.loads((previous_calibration/'SELECTION.json').read_text())
    require(previous.get('schema') == original.SCHEMA and previous.get('calibration_passed') is False
            and previous.get('passed') is False, 'the original two-choice CAL failure must be retained first')
    original.validate_calibrated_selection(previous, previous_calibration, data, primary)
    require(previous['parent']['run'] == str(Path(source_run).resolve())
            and previous['parent']['selection_sha256'] == sha(Path(source_run)/'SELECTION.json')
            and previous['parent']['checkpoint_sha256'] == sha(Path(source_run)/'verifier.pth'),
            'product-objective calibration must use the identical original trained checkpoint')
    grid = json.loads((previous_calibration/'GRID.json').read_text())
    require(len(grid['records']) == 2 and all(row['metrics']['passed'] is False for row in grid['records']),
            'original CAL grid was not a two-choice failure')
    failed = json.loads((previous_calibration/'FAILED.json').read_text())
    require(failed.get('export_allowed') is False and failed.get('test_read') is False
            and failed.get('grid_sha256') == previous['grid_sha256']
            and not (previous_calibration/'verifier.pth').exists(), 'original failed calibration was changed or exported')
    lineage = {'run': str(previous_calibration), 'selection_sha256': sha(previous_calibration/'SELECTION.json'),
               'grid_sha256': sha(previous_calibration/'GRID.json'), 'failed_report_sha256': sha(previous_calibration/'FAILED.json'),
               'original_protocol_passed': False, 'both_original_vote_choices_failed': True}
    bindings = dict(bindings)
    for filename in ('SELECTION.json', 'POLICY.json', 'BASELINE.json', 'CALIBRATION_DECISIONS.json', 'GRID.json', 'FAILED.json'):
        path = previous_calibration/filename
        bindings[str(path)] = sha(path)
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    return parent, rows, before, cached, bindings, previous['parent'], lineage


def compute_grid(rows, baseline, records, families, primary_families):
    nll = float(np.mean([-math.log(max(1e-12, record['probabilities'][row['target']])) for row, record in zip(rows, records)]))
    grid, candidates = [], []
    for index, agreement in enumerate(original.AGREEMENT_GRID):
        changed = original.recompute_passed(records, rows, original.policy_for(agreement), len(families))
        metrics, _ = blocking_metrics(rows, baseline, changed, families, primary_families)
        record = {'grid_index': index, 'min_patch_agreement': agreement, 'calibration_nll': nll, 'metrics': metrics,
                  'changed_passed_decisions': sum(a['passed'] != b['passed'] for a, b in zip(records, changed))}
        grid.append(record)
        rank = (int(metrics['passed']), metrics['total']['before_wrong'] - metrics['total']['after_wrong'],
                metrics['total']['after_correct'], -nll, -index)
        candidates.append((rank, index, changed))
    _, index, changed = max(candidates, key=lambda row: row[0])
    return grid, index, changed


def validate_blocking_selection(selection, run, data, primary):
    policy = effective_policy(selection)
    require(selection['schema'] == SCHEMA, 'expected explicit blocking-objective calibration')
    parent_ref, failure_ref = selection.get('parent'), selection.get('original_calibration')
    require(isinstance(parent_ref, dict) and isinstance(failure_ref, dict)
            and isinstance(parent_ref.get('run'), str) and isinstance(failure_ref.get('run'), str),
            'missing preserved parent and original failed calibration lineage')
    parent, rows, before, cached, bindings, expected_parent, expected_failure = source_evidence(
        parent_ref['run'], failure_ref['run'], data, primary)
    require(parent_ref == expected_parent and failure_ref == expected_failure and selection.get('bindings') == bindings,
            'blocking-objective calibration changed its immutable failed lineage')
    require(selection.get('state_before_sha256') == selection.get('state_after_sha256') == parent['state_after_sha256']
            and selection.get('families') == parent['families'], 'blocking-objective calibration changed weights or classes')
    policy_doc = {'schema': SCHEMA, 'objective': OBJECTIVE, 'objective_protocol': PROTOCOL, 'grid_spec': GRID_SPEC,
                  'bindings': bindings, 'parent': expected_parent, 'original_calibration': expected_failure,
                  'test_read': False, 'optimizer_steps_executed': 0, 'weights_unchanged': True}
    run = Path(run)
    require(sha(run/'POLICY.json') == selection.get('policy_manifest_sha256')
            and json.loads((run/'POLICY.json').read_text()) == policy_doc, 'blocking protocol was not frozen as declared')
    grid, index, records = compute_grid(rows, before['records'], cached['records'], parent['families'], before['families'])
    require(sha(run/'GRID.json') == selection.get('grid_sha256')
            and json.loads((run/'GRID.json').read_text()) == {'grid_spec': GRID_SPEC, 'records': grid, 'test_read': False,
                'parent_selection_sha256': expected_parent['selection_sha256'], 'original_calibration': expected_failure},
            'blocking calibration grid differs from its predeclared objective')
    require(selection.get('selected') == grid[index] and selection.get('policy') == original.policy_for(original.AGREEMENT_GRID[index])
            and selection.get('calibration_passed') is grid[index]['metrics']['passed']
            and selection.get('passed') is selection['calibration_passed'], 'blocking selection does not match the declared CAL ranking')
    require(sha(run/'CALIBRATION_DECISIONS.json') == selection.get('calibration_decisions_sha256')
            and json.loads((run/'CALIBRATION_DECISIONS.json').read_text()) == {'records': records, 'families': parent['families'], 'policy': policy}
            and sha(run/'BASELINE.json') == selection.get('baseline_sha256') == parent['baseline_sha256'],
            'blocking calibration changed cached probabilities or baseline')
    return policy


def main(args):
    require(getattr(args, 'objective', None) == OBJECTIVE, 'explicit --objective wrong-name-blocking is required')
    require(not args.output.exists(), 'blocking calibration output must be new')
    parent, rows, before, cached, bindings, parent_ref, failed_ref = source_evidence(
        args.source_run, args.previous_calibration, args.data, args.primary)
    import torch
    checkpoint = torch.load(args.source_run/'verifier.pth', map_location='cpu', weights_only=True)
    require(checkpoint.get('selection_sha256') == parent_ref['selection_sha256']
            and checkpoint.get('families') == parent['families']
            and state_sha(checkpoint['state_dict']) == parent['state_after_sha256'], 'blocking source checkpoint does not match the frozen parent')
    args.output.mkdir(parents=True)
    dump(args.output/'POLICY.json', {'schema': SCHEMA, 'objective': OBJECTIVE, 'objective_protocol': PROTOCOL, 'grid_spec': GRID_SPEC,
        'bindings': bindings, 'parent': parent_ref, 'original_calibration': failed_ref,
        'test_read': False, 'optimizer_steps_executed': 0, 'weights_unchanged': True})
    grid, index, records = compute_grid(rows, before['records'], cached['records'], parent['families'], before['families'])
    selected, policy = grid[index], original.policy_for(original.AGREEMENT_GRID[index])
    dump(args.output/'GRID.json', {'grid_spec': GRID_SPEC, 'records': grid, 'test_read': False,
        'parent_selection_sha256': parent_ref['selection_sha256'], 'original_calibration': failed_ref})
    shutil.copyfile(args.source_run/'BASELINE.json', args.output/'BASELINE.json')
    dump(args.output/'CALIBRATION_DECISIONS.json', {'records': records, 'families': parent['families'], 'policy': policy})
    selection = {'schema': SCHEMA, 'objective': OBJECTIVE, 'objective_protocol': PROTOCOL, 'grid_spec': GRID_SPEC, 'policy': policy,
        'selected': selected, 'passed': selected['metrics']['passed'], 'calibration_passed': selected['metrics']['passed'],
        'test_read': False, 'optimizer_steps_executed': 0, 'weights_unchanged': True, 'families': parent['families'],
        'parent': parent_ref, 'original_calibration': failed_ref, 'bindings': bindings,
        'state_before_sha256': parent['state_after_sha256'], 'state_after_sha256': parent['state_after_sha256'],
        'policy_manifest_sha256': sha(args.output/'POLICY.json'), 'grid_sha256': sha(args.output/'GRID.json'),
        'calibration_decisions_sha256': sha(args.output/'CALIBRATION_DECISIONS.json'), 'baseline_sha256': sha(args.output/'BASELINE.json')}
    dump(args.output/'SELECTION.json', selection)
    validate_blocking_selection(selection, args.output, args.data, args.primary)
    if selection['passed']:
        torch.save({'state_dict': checkpoint['state_dict'], 'families': parent['families'],
                    'selection_sha256': sha(args.output/'SELECTION.json')}, args.output/'verifier.pth')
        require(state_sha(torch.load(args.output/'verifier.pth', map_location='cpu', weights_only=True)['state_dict']) == parent['state_after_sha256'],
                'blocking checkpoint rewrap changed weights')
    else:
        dump(args.output/'FAILED.json', {'reason': 'Both vote thresholds failed the separately declared product-objective CAL protocol.',
            'export_allowed': False, 'test_read': False, 'grid_sha256': selection['grid_sha256']})
    require(all(sha(path) == digest for path, digest in bindings.items()), 'blocking calibration input changed during evaluation')
    print(json.dumps({'output': str(args.output), 'objective': OBJECTIVE, 'passed': selection['passed'],
        'selected_vote_threshold': selected['min_patch_agreement'], 'optimizer_steps_executed': 0,
        'weights_unchanged': True, 'metrics': selected['metrics']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'previous-calibration', 'data', 'primary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--objective', choices=[OBJECTIVE], required=True)
    main(parser.parse_args())
