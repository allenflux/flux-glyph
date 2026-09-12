#!/usr/bin/env python3
"""A separate, two-choice CAL-only voting calibration of unchanged CNN weights.

This never changes the old training policy or converts its failed report into a
passed training run. Only a new calibration selection may pass for export.
"""
from __future__ import annotations

import argparse
import copy
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

TRAINING_SCHEMA = 'flux-glyph-sans-verifier-training-v1'
SCHEMA = 'flux-glyph-sans-verifier-calibration-v1'
AGREEMENT_GRID = [.7, 2 / 3]
GRID_SPEC = {'min_patch_agreement': AGREEMENT_GRID,
             'fixed_policy': POLICY, 'temperature_recalibrated': False,
             'weights_optimized': False, 'source_partition': 'calibration',
             'ranking': ['passed', 'wrong_names_removed', 'retained_correct', 'negative_calibration_nll', 'earliest_grid_index']}


def policy_for(agreement):
    require(type(agreement) in (int, float) and agreement in AGREEMENT_GRID, 'vote threshold outside the predeclared two-choice grid')
    policy = copy.deepcopy(POLICY)
    policy['gates']['min_patch_agreement'] = agreement
    return policy


def effective_policy(selection):
    """Permit the original policy or precisely the declared new voting stage."""
    if selection.get('schema') == TRAINING_SCHEMA:
        require(selection.get('policy') == POLICY, 'training selection changed its original CAL policy')
        return copy.deepcopy(POLICY)
    require(selection.get('schema') == SCHEMA, 'unknown font-verifier selection schema')
    policy = selection.get('policy')
    agreement = policy.get('gates', {}).get('min_patch_agreement') if isinstance(policy, dict) else None
    expected = policy_for(agreement)
    require(policy == expected and selection.get('grid_spec') == GRID_SPEC,
            'calibration may change only the predeclared verifier vote threshold')
    require(type(selection.get('optimizer_steps_executed')) is int and selection['optimizer_steps_executed'] == 0
            and selection.get('weights_unchanged') is True and selection.get('test_read') is False,
            'vote calibration must keep weights unchanged and use no TEST')
    return expected


def recompute_passed(records, rows, policy, family_count):
    """Use cached predictions exactly; never alter a probability or winner."""
    require(policy in [policy_for(value) for value in AGREEMENT_GRID], 'invalid calibration policy')
    require(len(records) == len(rows) > 0, 'unaligned cached calibration rows')
    gates, result = policy['gates'], []
    for record, row in zip(records, rows):
        require(row.get('split') == 'calibration' and type(row.get('tile_count')) is int and row['tile_count'] > 0,
                'only CAL observations may be recalibrated')
        probabilities = np.asarray(record.get('probabilities'), dtype=np.float64)
        require(probabilities.shape == (family_count,) and np.isfinite(probabilities).all()
                and np.all((probabilities >= 0) & (probabilities <= 1))
                and math.isclose(float(probabilities.sum()), 1., abs_tol=1e-9), 'invalid cached probabilities')
        order = np.argsort(-probabilities, kind='stable')
        require(type(record.get('predicted')) is int and record['predicted'] == int(order[0]), 'cached winner differs from probabilities')
        score, margin, agreement = record.get('score'), record.get('margin'), record.get('agreement')
        require(all(type(value) in (float, int) and math.isfinite(value) for value in (score, margin, agreement))
                and 0 <= agreement <= 1 and math.isclose(score, float(probabilities[order[0]]), abs_tol=1e-12)
                and math.isclose(margin, float(probabilities[order[0]] - probabilities[order[1]]), abs_tol=1e-12)
                and math.isclose(agreement * row['tile_count'], round(agreement * row['tile_count']), abs_tol=1e-9),
                'cached scores or patch votes are inconsistent')
        passed = score >= gates['min_score'] and margin >= gates['min_margin'] and margin > 1e-8 and agreement >= gates['min_patch_agreement']
        result.append({**record, 'passed': bool(passed)})
    return result


def validate_parent(source_run, data, primary):
    """Read completed training provenance, including failed CAL lineage."""
    source_run, data, primary = map(lambda path: Path(path).resolve(), (source_run, data, primary))
    source = json.loads((source_run/'SELECTION.json').read_text())
    require(source.get('schema') == TRAINING_SCHEMA and source.get('policy') == POLICY and source.get('test_read') is False,
            'calibration requires a completed original-policy training selection')
    require(type(source.get('optimizer_steps_executed')) is int and source['optimizer_steps_executed'] > 0
            and type(source.get('selected', {}).get('step')) is int
            and 0 < source['selected']['step'] <= source['optimizer_steps_executed']
            and source['selected'] in source.get('history', [])
            and source.get('calibration_passed') is source['selected']['metrics']['passed']
            and source.get('passed') is source['calibration_passed'], 'parent training completion/history/pass state differs')
    bindings = source.get('bindings')
    require(isinstance(bindings, dict) and bindings and all(isinstance(path, str) and str(Path(path).resolve()) == path
            and Path(path).is_file() and sha(path) == digest for path, digest in bindings.items()), 'parent source binding changed')
    required = [data/'MANIFEST.json', data/'train/MANIFEST.json', data/'calibration/MANIFEST.json',
                primary/'metadata.json', primary/'model.onnx', primary/'rejection.onnx',
                ROOT/'training/train_sans_verifier.py', ROOT/'training/prepare_sans_views.py',
                ROOT/'training/region_network.py', ROOT/'training/network.py']
    require(all(str(path) in bindings and path.is_file() and bindings[str(path)] == sha(path) for path in required),
            'calibration data/primary exact paths differ from training')
    source_policy = json.loads((source_run/'POLICY.json').read_text())
    require(source_policy.get('policy') == POLICY and source_policy.get('test_read') is False
            and source_policy.get('steps') == source['optimizer_steps_executed']
            and source_policy.get('bindings') == bindings, 'parent POLICY differs from completed selection')
    require(sha(source_run/'BASELINE.json') == source['baseline_sha256']
            and sha(source_run/'CALIBRATION_DECISIONS.json') == source['calibration_decisions_sha256'], 'parent cached CAL files changed')
    root = json.loads((data/'MANIFEST.json').read_text())
    partition = json.loads((data/'calibration/MANIFEST.json').read_text())
    require(partition.get('split') == 'calibration' and partition.get('test_pixels_opened') is False
            and partition.get('root_manifest_sha256') == sha(data/'MANIFEST.json'), 'CAL partition provenance differs')
    paths = {}
    for key, filename in (('metadata', 'metadata.json'), ('array', 'tiles.npy')):
        item = partition[key]
        require(item['path'] == filename, 'CAL asset path differs')
        path = data/'calibration'/filename
        require(sha(path) == item['sha256'], 'CAL asset changed')
        paths[key] = path
    cache = json.loads(paths['metadata'].read_text())
    require(cache.get('split') == 'calibration', 'cached rows are not CAL')
    rows = cache['rows']
    before = json.loads((source_run/'BASELINE.json').read_text())
    decisions = json.loads((source_run/'CALIBRATION_DECISIONS.json').read_text())
    primary_meta = json.loads((primary/'metadata.json').read_text())
    families = source.get('families')
    require(isinstance(families, list) and len(families) == 9 and len(set(families)) == 9 and families[-1] == 'Roboto'
            and root.get('families') == partition.get('families') == decisions.get('families') == families
            and before.get('families') == primary_meta.get('families') == families[:-1]
            and decisions.get('policy') == POLICY, 'parent CAL family order or policy differs')
    require(len(rows) == len(before.get('records', [])) == len(decisions.get('records', [])) == partition.get('rows'),
            'parent CAL cache count differs')
    require(primary_meta.get('algorithm') == 'region-cnn64x256-rejection-v2'
            and primary_meta.get('model', {}).get('sha256') == sha(primary/'model.onnx')
            and primary_meta.get('rejection', {}).get('model', {}).get('sha256') == sha(primary/'rejection.onnx'),
            'original R19 model/rejection bytes differ')
    original_decisions = recompute_passed(decisions['records'], rows, POLICY, len(families))
    require(original_decisions == decisions['records'], 'parent cached passed decisions differ from its original policy')
    metrics, _ = summarize(rows, before['records'], decisions['records'], families, before['families'])
    require(metrics == source['selected']['metrics'], 'parent original failed/passed CAL metrics do not reproduce')
    extended = dict(bindings)
    for path in (source_run/'SELECTION.json', source_run/'POLICY.json', source_run/'BASELINE.json',
                 source_run/'CALIBRATION_DECISIONS.json', source_run/'verifier.pth', paths['metadata'],
                 Path(__file__).resolve(), ROOT/'training/train_regions.py'):
        extended[str(path)] = sha(path)
    return source, rows, before, decisions, extended


def compute_grid(rows, baseline, records, families, primary_families):
    nll = float(np.mean([-math.log(max(1e-12, record['probabilities'][row['target']])) for row, record in zip(rows, records)]))
    grid, choices = [], []
    for index, agreement in enumerate(AGREEMENT_GRID):
        policy = policy_for(agreement)
        changed = recompute_passed(records, rows, policy, len(families))
        metrics, _ = summarize(rows, baseline, changed, families, primary_families)
        record = {'grid_index': index, 'min_patch_agreement': agreement, 'calibration_nll': nll, 'metrics': metrics,
                  'changed_passed_decisions': sum(a['passed'] != b['passed'] for a, b in zip(records, changed))}
        grid.append(record)
        rank = (int(metrics['passed']), metrics['total']['before_wrong'] - metrics['total']['after_wrong'],
                metrics['total']['after_correct'], -nll, -index)
        choices.append((rank, index, changed))
    _, selected_index, changed = max(choices, key=lambda choice: choice[0])
    return grid, selected_index, changed


def validate_calibrated_selection(selection, run, data, primary):
    policy = effective_policy(selection)
    require(selection['schema'] == SCHEMA, 'expected separate voting calibration selection')
    parent_ref = selection.get('parent')
    require(isinstance(parent_ref, dict) and isinstance(parent_ref.get('run'), str), 'missing original training lineage')
    source_path = Path(parent_ref['run']).resolve()
    require(str(source_path) == parent_ref['run'] and source_path != Path(run).resolve(), 'invalid original training lineage path')
    parent, rows, before, cached, expected_bindings = validate_parent(source_path, data, primary)
    expected_parent = {'run': str(source_path), 'selection_sha256': sha(source_path/'SELECTION.json'),
                      'checkpoint_sha256': sha(source_path/'verifier.pth'), 'schema': parent['schema'],
                      'calibration_passed': parent['calibration_passed'], 'selected': parent['selected'],
                      'state_after_sha256': parent['state_after_sha256'],
                      'optimizer_steps_executed': parent['optimizer_steps_executed']}
    require(parent_ref == expected_parent and selection.get('bindings') == expected_bindings,
            'calibration lost or altered parent failed lineage/bindings')
    require(selection.get('state_before_sha256') == selection.get('state_after_sha256') == parent['state_after_sha256']
            and selection.get('families') == parent['families'], 'calibration changed the trained parameters or classes')
    grid, selected_index, decisions = compute_grid(rows, before['records'], cached['records'], parent['families'], before['families'])
    grid_path = Path(run)/'GRID.json'
    policy_path = Path(run)/'POLICY.json'
    require(sha(policy_path) == selection.get('policy_manifest_sha256')
            and json.loads(policy_path.read_text()) == {'schema': SCHEMA, 'grid_spec': GRID_SPEC, 'bindings': expected_bindings,
                                                        'test_read': False, 'optimizer_steps_executed': 0, 'weights_unchanged': True},
            'separate calibration preregistration differs')
    require(sha(grid_path) == selection.get('grid_sha256')
            and json.loads(grid_path.read_text()) == {'grid_spec': GRID_SPEC, 'records': grid, 'test_read': False,
                                                   'parent_selection_sha256': expected_parent['selection_sha256']},
            'calibration two-choice grid differs')
    require(selection.get('selected') == grid[selected_index] and selection.get('policy') == policy_for(AGREEMENT_GRID[selected_index])
            and selection.get('calibration_passed') is grid[selected_index]['metrics']['passed']
            and selection.get('passed') is selection['calibration_passed'], 'calibration selection is not the declared grid winner')
    actual = json.loads((Path(run)/'CALIBRATION_DECISIONS.json').read_text())
    require(actual == {'records': decisions, 'families': parent['families'], 'policy': policy}
            and sha(Path(run)/'CALIBRATION_DECISIONS.json') == selection.get('calibration_decisions_sha256')
            and sha(Path(run)/'BASELINE.json') == selection.get('baseline_sha256') == parent['baseline_sha256'],
            'calibrated observations changed beyond their passed flag')
    return policy


def main(args):
    require(not args.output.exists(), 'calibration output must be new; preserve old failed reports')
    parent, rows, before, cached, bindings = validate_parent(args.source_run, args.data, args.primary)
    import torch
    checkpoint = torch.load(args.source_run/'verifier.pth', map_location='cpu', weights_only=True)
    require(checkpoint.get('selection_sha256') == sha(args.source_run/'SELECTION.json')
            and checkpoint.get('families') == parent['families']
            and state_sha(checkpoint['state_dict']) == parent['state_after_sha256'], 'source checkpoint/state differs from original selection')
    args.output.mkdir(parents=True)
    dump(args.output/'POLICY.json', {'schema': SCHEMA, 'grid_spec': GRID_SPEC, 'bindings': bindings,
                                   'test_read': False, 'optimizer_steps_executed': 0, 'weights_unchanged': True})
    grid, selected_index, decisions = compute_grid(rows, before['records'], cached['records'], parent['families'], before['families'])
    selected = grid[selected_index]
    policy = policy_for(selected['min_patch_agreement'])
    parent_ref = {'run': str(args.source_run.resolve()), 'selection_sha256': sha(args.source_run/'SELECTION.json'),
                  'checkpoint_sha256': sha(args.source_run/'verifier.pth'), 'schema': parent['schema'],
                  'calibration_passed': parent['calibration_passed'], 'selected': parent['selected'],
                  'state_after_sha256': parent['state_after_sha256'],
                  'optimizer_steps_executed': parent['optimizer_steps_executed']}
    dump(args.output/'GRID.json', {'grid_spec': GRID_SPEC, 'records': grid, 'test_read': False,
                                  'parent_selection_sha256': parent_ref['selection_sha256']})
    shutil.copyfile(args.source_run/'BASELINE.json', args.output/'BASELINE.json')
    dump(args.output/'CALIBRATION_DECISIONS.json', {'records': decisions, 'families': parent['families'], 'policy': policy})
    selection = {'schema': SCHEMA, 'policy': policy, 'grid_spec': GRID_SPEC, 'selected': selected,
                 'calibration_passed': selected['metrics']['passed'], 'passed': selected['metrics']['passed'],
                 'test_read': False, 'families': parent['families'], 'bindings': bindings, 'parent': parent_ref,
                 'optimizer_steps_executed': 0, 'weights_unchanged': True,
                 'state_before_sha256': parent['state_after_sha256'], 'state_after_sha256': parent['state_after_sha256'],
                 'policy_manifest_sha256': sha(args.output/'POLICY.json'),
                 'grid_sha256': sha(args.output/'GRID.json'), 'calibration_decisions_sha256': sha(args.output/'CALIBRATION_DECISIONS.json'),
                 'baseline_sha256': sha(args.output/'BASELINE.json')}
    dump(args.output/'SELECTION.json', selection)
    validate_calibrated_selection(selection, args.output, args.data, args.primary)
    if selection['passed']:
        torch.save({'state_dict': checkpoint['state_dict'], 'families': parent['families'],
                    'selection_sha256': sha(args.output/'SELECTION.json')}, args.output/'verifier.pth')
        repackaged = torch.load(args.output/'verifier.pth', map_location='cpu', weights_only=True)
        require(state_sha(repackaged['state_dict']) == parent['state_after_sha256'], 'checkpoint rewrapping changed parameters')
    else:
        dump(args.output/'FAILED.json', {'reason': 'Neither predeclared CAL voting threshold met the unchanged release criteria.',
                                         'export_allowed': False, 'test_read': False, 'grid_sha256': selection['grid_sha256']})
    require(all(sha(path) == digest for path, digest in bindings.items()), 'calibration source changed during processing')
    print(json.dumps({'output': str(args.output), 'passed': selection['passed'], 'selected_vote_threshold': selected['min_patch_agreement'],
                      'optimizer_steps_executed': 0, 'weights_unchanged': True, 'metrics': selected['metrics']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'data', 'primary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
