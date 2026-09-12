"""Rejection export contracts using synthetic arrays and temporary provenance."""
import builtins
import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def exporter():
    return importlib.import_module('training.export_rejection')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')


def calibration_fixture():
    rows = [{'domain': domain, 'source_id': source, 'region_id': 'line-1'}
            for domain in ('new_native', 'r17_native') for source in ('page-1', 'page-2')]
    scores = [.25, .5, .75, 1.]
    decisions = {'schema': 'flux-glyph-rejection-calibration-decisions-v1', 'min_known_score': .5,
                 'records': [{**row, 'known_score': score, 'passed': score >= .5} for row, score in zip(rows, scores)]}
    return rows, decisions


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    module = exporter()
    args = SimpleNamespace(run=tmp_path / 'run', data=tmp_path / 'gate', anchor=tmp_path / 'anchor',
                           primary=tmp_path / 'primary', output=tmp_path / 'not-created', snapshot=tmp_path / 'snapshot.py')
    root = tmp_path / 'repo'
    monkeypatch.setattr(module, 'ROOT', root)
    for directory in (args.run, args.data, args.anchor, args.primary, root / 'training'):
        directory.mkdir(parents=True)
    for name in ('train_rejection.py', 'rejection_network.py'):
        (root / 'training' / name).write_text('# Temporary frozen source: ' + name + '\n')
    dump(args.data / 'MANIFEST.json', {'fixture': 'gate data manifest; no arrays exist'})
    dump(args.anchor / 'MANIFEST.json', {'fixture': 'R17 anchor manifest; no arrays exist'})
    (args.primary / 'model.onnx').write_bytes(b'Not an ONNX model; metadata validation must not execute it')
    (args.run / 'rejection.pth').write_bytes(b'Not a checkpoint; the helper receives an already loaded mapping')
    args.snapshot.write_text('# Temporary frozen preprocessing snapshot\n')
    base_onnx_sha = sha(args.primary / 'model.onnx')
    base_data_sha = sha(args.anchor / 'MANIFEST.json')
    monkeypatch.setattr(module, 'BASE_ONNX_SHA', base_onnx_sha)
    monkeypatch.setattr(module, 'BASE_CHECKPOINT_SHA', 'a' * 64)
    monkeypatch.setattr(module, 'SNAPSHOT_SHA', sha(args.snapshot))
    if hasattr(module, 'BASE_DATA_SHA'):
        monkeypatch.setattr(module, 'BASE_DATA_SHA', base_data_sha)
    families = ['HarmonyOS Sans SC', 'MiSans', 'Noto Sans CJK SC', 'OPPO Sans',
                'PingFang', 'SF Pro', 'Helvetica', 'Alipay Number']
    primary = {'schema': 'flux-glyph-region-font-v1', 'algorithm': 'region-cnn64x256-v1', 'families': families,
               'model': {'path': 'model.onnx', 'sha256': base_onnx_sha}}
    rows, decisions = calibration_fixture()
    selection = {'schema': 'flux-glyph-region-rejection-training-v1', 'calibration_passed': True, 'test_read': False,
                 'policy': copy.deepcopy(module.POLICY),
                 'selected': {'step': 250, 'min_known_score': .5}, 'optimizer_steps_executed': 2000,
                 'data_manifest_sha256': sha(args.data / 'MANIFEST.json'), 'anchor_manifest_sha256': base_data_sha,
                 'parent_onnx_sha256': base_onnx_sha, 'parent_checkpoint_sha256': 'a' * 64,
                 'source_sha256': sha(root / 'training/train_rejection.py'),
                 'network_source_sha256': sha(root / 'training/rejection_network.py')}
    checkpoint = {'families': families, 'labels': ['unknown', 'known'], 'temperature': 1.,
                  'min_known_score': .5, 'selected_step': 250, 'state_dict': {}}

    def rebind():
        dump(args.run / 'CALIBRATION_DECISIONS.json', decisions)
        selection['calibration_decisions_sha256'] = sha(args.run / 'CALIBRATION_DECISIONS.json')
        dump(args.run / 'SELECTION.json', selection)
        checkpoint['selection_sha256'] = sha(args.run / 'SELECTION.json')
        dump(args.primary / 'metadata.json', primary)

    rebind()
    return SimpleNamespace(module=module, args=args, root=root, checkpoint=checkpoint, selection=selection,
                           primary=primary, rows=rows, decisions=decisions, rebind=rebind)


def validate(case):
    return case.module.validate_frozen_inputs(case.args, case.checkpoint, case.selection, case.primary)


def test_frozen_validation_binds_both_domains_without_loading_models_or_writing_output(frozen, monkeypatch):
    before = {str(path): path.read_bytes() for path in frozen.args.run.parent.rglob('*') if path.is_file()}
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        assert name.split('.')[0] not in ('torch', 'onnx', 'onnxruntime'), 'Metadata validation imported model execution code'
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guard)
    bindings, decisions = validate(frozen)
    assert isinstance(bindings, dict) and bindings
    assert all(entry['sha256'] == sha(entry['path']) for entry in bindings.values())
    assert decisions == frozen.decisions
    assert not frozen.args.output.exists()
    assert before == {str(path): path.read_bytes() for path in frozen.args.run.parent.rglob('*') if path.is_file()}


def test_export_binds_the_actual_approved_end_to_end_trainer(frozen):
    trainer=frozen.root/'training/train_rejection_e2e.py'
    trainer.write_text('# Distinct approved end-to-end trainer\n')
    frozen.selection['source_sha256']=sha(trainer)
    frozen.rebind()
    bindings,_=validate(frozen)
    assert bindings['training_source']=={'path':str(trainer),'sha256':sha(trainer)}


def test_export_cannot_substitute_an_arbitrary_trainer_path(frozen):
    trainer=frozen.root/'training/arbitrary_trainer.py'
    trainer.write_text('# Outside the explicit approved trainer list\n')
    frozen.selection['source_sha256']=sha(trainer)
    frozen.selection['source_path']=str(trainer)
    frozen.rebind()
    with pytest.raises(ValueError,match='approved trainer'):validate(frozen)


def test_export_requires_a_unique_trainer_identity(frozen):
    (frozen.root/'training/train_rejection_e2e.py').write_bytes((frozen.root/'training/train_rejection.py').read_bytes())
    with pytest.raises(ValueError,match='approved trainer'):validate(frozen)


@pytest.mark.parametrize('field,value', [('labels', ['known', 'unknown']), ('temperature', .5),
                                       ('min_known_score', .49), ('selected_step', 500),
                                       ('selection_sha256', '0' * 64)])
def test_checkpoint_header_cannot_drift_from_frozen_selection(frozen, field, value):
    frozen.checkpoint[field] = value
    with pytest.raises(ValueError):
        validate(frozen)


@pytest.mark.parametrize('failure', ['failed_calibration', 'test_read', 'policy_temperature', 'policy_labels', 'decisions_schema'])
def test_rehashed_invalid_training_policy_is_rejected(frozen, failure):
    if failure == 'failed_calibration':
        frozen.selection['calibration_passed'] = False
    elif failure == 'test_read':
        frozen.selection['test_read'] = True
    elif failure == 'policy_temperature':
        frozen.selection['policy']['temperature'] = 2.
        frozen.checkpoint['temperature'] = 2.
    elif failure == 'policy_labels':
        frozen.selection['policy']['labels'] = ['known', 'unknown']
    else:
        frozen.decisions['schema'] = 'unverified-records'
    frozen.rebind()
    with pytest.raises(ValueError):
        validate(frozen)


@pytest.mark.parametrize('artifact', ['gate_manifest', 'anchor_manifest', 'train_source', 'network_source',
                                    'decisions', 'selection', 'primary_model', 'preprocessing_snapshot'])
def test_changed_provenance_is_rejected(frozen, artifact):
    paths = {'gate_manifest': frozen.args.data / 'MANIFEST.json', 'anchor_manifest': frozen.args.anchor / 'MANIFEST.json',
             'train_source': frozen.root / 'training/train_rejection.py', 'network_source': frozen.root / 'training/rejection_network.py',
             'decisions': frozen.args.run / 'CALIBRATION_DECISIONS.json', 'selection': frozen.args.run / 'SELECTION.json',
             'primary_model': frozen.args.primary / 'model.onnx', 'preprocessing_snapshot': frozen.args.snapshot}
    path = paths[artifact]
    path.write_bytes(path.read_bytes() + b'\nchanged')
    with pytest.raises(ValueError):
        validate(frozen)


def test_calibration_record_order_and_equal_threshold_are_preserved():
    rows, decisions = calibration_fixture()
    scores = exporter().calibration_records(decisions, rows, .5)
    np.testing.assert_array_equal(scores, [.25, .5, .75, 1.])
    assert decisions['records'][1]['passed'] is True


@pytest.mark.parametrize('failure', ['reordered', 'missing_anchor_domain', 'duplicate_identity', 'source_changed',
                                    'region_changed', 'domain_changed', 'passed_wrong', 'passed_not_bool',
                                    'nan_score', 'infinite_score', 'negative_score', 'excessive_score', 'threshold_changed'])
def test_calibration_records_reject_identity_or_decision_drift(failure):
    rows, decisions = calibration_fixture()
    if failure == 'reordered':
        decisions['records'].reverse()
    elif failure == 'missing_anchor_domain':
        decisions['records'] = decisions['records'][:2]
    elif failure == 'duplicate_identity':
        rows[1] = copy.deepcopy(rows[0])
        decisions['records'][1] = copy.deepcopy(decisions['records'][0])
    elif failure == 'source_changed':
        decisions['records'][0]['source_id'] = 'other-page'
    elif failure == 'region_changed':
        decisions['records'][0]['region_id'] = 'other-region'
    elif failure == 'domain_changed':
        decisions['records'][2]['domain'] = 'new_native'
    elif failure == 'passed_wrong':
        decisions['records'][1]['passed'] = False
    elif failure == 'passed_not_bool':
        decisions['records'][1]['passed'] = 1
    elif failure == 'threshold_changed':
        decisions['min_known_score'] = .49
    else:
        decisions['records'][0]['known_score'] = {'nan_score': float('nan'), 'infinite_score': float('inf'),
                                                 'negative_score': -.01, 'excessive_score': 1.01}[failure]
    with pytest.raises(ValueError):
        exporter().calibration_records(decisions, rows, .5)


def test_logits_allow_small_finite_roundoff_without_changing_binary_winner():
    expected = np.array([[.5, -.5], [-1., 1.]], dtype=np.float32)
    actual = expected + np.array([[1e-6, -1e-6], [0., 1e-6]], dtype=np.float32)
    exporter().compare_logits(actual, expected)


@pytest.mark.parametrize('failure', ['both_nan', 'both_infinite', 'wrong_columns', 'broadcast_shape',
                                    'logits_drift', 'probability_drift', 'winner_flip'])
def test_logits_reject_nonfinite_shape_and_probability_parity_failures(failure):
    expected = np.array([[0., .2]], dtype=np.float32)
    actual = expected.copy()
    if failure in ('both_nan', 'both_infinite'):
        expected[0, 0] = actual[0, 0] = np.nan if failure == 'both_nan' else np.inf
    elif failure == 'wrong_columns':
        actual = np.ones((1, 3), dtype=np.float32)
    elif failure == 'broadcast_shape':
        actual = np.repeat(actual, 2, axis=0)
    elif failure == 'logits_drift':
        actual += .001  # Same probabilities and winner; raw logits still differ.
    elif failure == 'probability_drift':
        actual += np.array([[.00019, -.00019]], dtype=np.float32)
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
    else:
        expected = np.array([[0., 1e-8]], dtype=np.float32)
        actual = np.array([[1e-8, 0.]], dtype=np.float32)
    with pytest.raises((ValueError, AssertionError)):
        exporter().compare_logits(actual, expected)


def test_three_way_calibration_parity_accepts_roundoff_and_equal_gate():
    frozen = np.array([.1, .5, .75, 1.])
    expected = np.array([.100001, .5, .750001, .999999])
    actual = np.array([.100002, .500001, .749999, 1.])
    exporter().compare_calibration_scores(actual, expected, frozen, .5)


@pytest.mark.parametrize('failure', ['all_nan', 'actual_inf', 'expected_nan', 'frozen_nan', 'negative', 'excessive',
                                    'actual_crosses_gate', 'both_cross_frozen_gate', 'both_drift_from_frozen', 'shape'])
def test_calibration_parity_rejects_invalid_scores_and_cross_gate_roundoff(failure):
    frozen = np.array([.1, .5, .75])
    expected, actual = frozen.copy(), frozen.copy()
    if failure == 'all_nan':
        actual[0] = expected[0] = frozen[0] = np.nan
        assert np.array_equal(actual >= .5, expected >= .5)  # The former Boolean-only check missed this.
    elif failure == 'actual_inf':
        actual[0] = np.inf
    elif failure == 'expected_nan':
        expected[0] = np.nan
    elif failure == 'frozen_nan':
        frozen[0] = np.nan
    elif failure == 'negative':
        actual[0] = -1e-8
    elif failure == 'excessive':
        expected[2] = 1 + 1e-8
    elif failure == 'actual_crosses_gate':
        actual[1] -= 1e-6
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-4)
    elif failure == 'both_cross_frozen_gate':
        expected[1] = actual[1] = .5 - 1e-6
        np.testing.assert_allclose(actual, frozen, atol=2e-5, rtol=1e-4)
    elif failure == 'both_drift_from_frozen':
        actual[0] = expected[0] = .101
    else:
        actual = actual[:1]
    with pytest.raises((ValueError, AssertionError)):
        exporter().compare_calibration_scores(actual, expected, frozen, .5)
