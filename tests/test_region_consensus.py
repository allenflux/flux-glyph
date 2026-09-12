"""Three-network contracts, independent rankings and fail-closed decisions."""
import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

import flux_glyph.region_font as module
from test_region_font import fake_model, attach_rejection, image, rejection_bundle


def attach_verifier(model, logits=(0., 6., 0.), temperature=1.):
    # Deliberately reordered: compare family names, never output indices.
    model.verifier_meta = {'families': ['Two', 'One', 'Roboto'], 'temperature': temperature,
                           'gates': {'min_score': .7, 'min_margin': .15, 'min_patch_agreement': .7}}
    def run(names, inputs):
        assert names == ['logits', 'log_em_ratio'] and set(inputs) == {'tiles'}
        n = len(inputs['tiles'])
        values = logits(n) if callable(logits) else np.tile(logits, (n, 1)).astype(np.float32)
        return [values, np.full(n, 20., dtype=np.float32)]
    model.verifier_session = SimpleNamespace(run=run)
    return model


def test_agreement_preserves_primary_scores_and_primary_size_not_verifier_size():
    old = attach_rejection(fake_model()).predict(image())
    new = attach_verifier(attach_rejection(fake_model())).predict(image())
    verified = new.pop('verifier')
    assert new == old
    assert verified['status'] == 'passed' and verified['family'] == 'One'
    assert verified['candidates'][0]['family'] == 'One'


def test_disagreement_keeps_both_relative_rankings_without_confirming_font_or_size():
    result = attach_verifier(attach_rejection(fake_model()), (6., 0., 0.)).predict(image())
    assert result['status'] == 'uncertain' and result['reason_code'] == 'neural_model_disagreement'
    assert result['family'] is result['font_size_px_estimate'] is None
    assert result['candidates'][0]['family'] == 'One' and result['score'] > .99
    assert result['verifier']['status'] == 'disagreed'
    assert result['verifier']['candidates'][0]['family'] == 'Two'


def test_external_verifier_winner_only_rejects_when_its_gates_pass():
    result = attach_verifier(attach_rejection(fake_model()), (0., 0., 6.)).predict(image())
    assert result['status'] == 'out_of_scope' and result['reason_code'] == 'verifier_font_out_of_scope'
    assert result['family'] is result['score'] is result['font_size_px_estimate'] is None
    assert result['candidates'] == result['verifier']['candidates'] == []
    weak = attach_verifier(attach_rejection(fake_model()), (0., 0., .01)).predict(image())
    assert weak['status'] == 'uncertain' and weak['reason_code'] == 'neural_model_disagreement'


def test_rejection_still_precedes_both_classifiers():
    model = attach_verifier(attach_rejection(fake_model(), (8., -8.)))
    model.session.run = model.verifier_session.run = lambda *a: pytest.fail('Rejected pixels cannot be named')
    assert model.predict(image())['reason_code'] == 'unknown_font_rejected'


@pytest.mark.parametrize('gate,value,reason', [('min_score', .99999, 'below_score_gate'),
    ('min_margin', .99999, 'ambiguous_neural_families')])
def test_agreement_does_not_override_either_network_gate(gate, value, reason):
    for target in ('primary', 'verifier'):
        model = attach_verifier(attach_rejection(fake_model()))
        (model.meta if target == 'primary' else model.verifier_meta)['gates'][gate] = value
        result = model.predict(image())
        assert result['status'] == 'uncertain' and result['family'] is result['font_size_px_estimate'] is None
        assert result['reason_code'] == ('verifier_' if target == 'verifier' else '') + reason


def test_verifier_uses_its_temperature_and_mean_patch_probabilities():
    model = attach_verifier(attach_rejection(fake_model()), (0., 2., 0.), temperature=2.)
    result = model.predict(image())
    assert result['verifier']['score'] == pytest.approx(np.exp(1) / (2 + np.exp(1)))
    assert result['reason_code'] == 'verifier_below_score_gate'
    def patches(n):
        values = np.tile([0., 8., 0.], (n, 1)).astype(np.float32)
        values[0] = [8., 0., 0.]
        return values
    model = attach_verifier(attach_rejection(fake_model()), patches)
    model.verifier_meta['gates']['min_patch_agreement'] = .99
    result = model.predict(image('苹方未'*4))
    assert result['reason_code'] == 'verifier_mixed_or_ambiguous_region'
    assert result['verifier']['patch_agreement'] < .99


@pytest.mark.parametrize('fault', ['nan', 'inf', 'wrong_classes', 'wrong_batch', 'dtype', 'list',
                                  'missing_size', 'nan_size', 'size_shape', 'exception'])
def test_invalid_verifier_never_falls_back_to_confident_primary(fault):
    model = attach_verifier(attach_rejection(fake_model()))
    original = model.verifier_session.run
    def run(names, inputs):
        outputs = original(names, inputs)
        if fault == 'exception': raise RuntimeError('broken verifier')
        if fault == 'nan': outputs[0][0, 0] = np.nan
        elif fault == 'inf': outputs[0][0, 0] = np.inf
        elif fault == 'wrong_classes': outputs[0] = outputs[0][:, :2]
        elif fault == 'wrong_batch': outputs[0] = np.concatenate([outputs[0], outputs[0]])
        elif fault == 'dtype': outputs[0] = outputs[0].astype(np.float64)
        elif fault == 'list': outputs[0] = outputs[0].tolist()
        elif fault == 'missing_size': outputs.pop()
        elif fault == 'nan_size': outputs[1][0] = np.nan
        elif fault == 'size_shape': outputs[1] = outputs[1][:, None]
        return outputs
    model.verifier_session.run = run
    result = model.predict(image())
    assert result['reason_code'] == 'invalid_verifier_output'
    assert result['family'] is result['score'] is result['font_size_px_estimate'] is None
    assert result['candidates'] == [] and result['verifier']['status'] == 'unavailable'


def consensus_bundle(tmp_path, monkeypatch):
    meta, save, reject = rejection_bundle(tmp_path, monkeypatch)
    payload = b'font verifier ONNX'
    (tmp_path / 'verifier.onnx').write_bytes(payload)
    meta['algorithm'] = module.CONSENSUS_ALGORITHM
    meta['verifier'] = {'schema': 'flux-glyph-region-verifier-v1', 'algorithm': 'region-font-verifier-cnn64x256-v1',
        'families': ['SF Pro', 'Roboto', 'PingFang'], 'temperature': .75, 'gates': copy.deepcopy(meta['gates']),
        'base_model_sha256': meta['model']['sha256'],
        'model': {'path': 'verifier.onnx', 'sha256': hashlib.sha256(payload).hexdigest()}}
    node = lambda name, shape: SimpleNamespace(name=name, shape=shape, type='tensor(float)')
    session = SimpleNamespace(get_inputs=lambda: [node('tiles', ['batch', 1, 64, 256])],
        get_outputs=lambda: [node('logits', ['batch', 3]), node('log_em_ratio', ['batch'])])
    original = module.ort.InferenceSession
    monkeypatch.setattr(module.ort, 'InferenceSession', lambda data, **kwargs:
        session if data == payload else original(data, **kwargs))
    save()
    return meta, save, session


def test_v3_loads_complete_three_model_contract(tmp_path, monkeypatch):
    meta, _, session = consensus_bundle(tmp_path, monkeypatch)
    model = module.RegionFontClassifier(tmp_path)
    assert model.verifier_session is session and model.verifier_meta == meta['verifier']


@pytest.mark.parametrize('fault', ['v1', 'v2', 'missing_verifier', 'missing_rejection', 'schema', 'algorithm',
    'families', 'duplicate', 'base', 'temperature', 'nan', 'bool', 'gates', 'missing_gate',
    'path', 'primary_collision', 'rejection_collision', 'sha'])
def test_invalid_v3_metadata_cannot_be_silently_ignored(tmp_path, monkeypatch, fault):
    meta, save, _ = consensus_bundle(tmp_path, monkeypatch)
    value = meta['verifier']
    if fault == 'v1': meta['algorithm'] = module.ALGORITHM; meta.pop('rejection')
    elif fault == 'v2': meta['algorithm'] = module.REJECTION_ALGORITHM
    elif fault == 'missing_verifier': meta.pop('verifier')
    elif fault == 'missing_rejection': meta.pop('rejection')
    elif fault in ('schema', 'algorithm'): value[fault] = 'other'
    elif fault == 'families': value['families'] = ['SF Pro', 'Roboto']
    elif fault == 'duplicate': value['families'].append('Roboto')
    elif fault == 'base': value['base_model_sha256'] = '0' * 64
    elif fault == 'temperature': value['temperature'] = 0
    elif fault == 'nan': value['gates']['min_score'] = float('nan')
    elif fault == 'bool': value['gates']['min_margin'] = True
    elif fault == 'gates': value['gates'] = []
    elif fault == 'missing_gate': value['gates'].pop('min_patch_agreement')
    elif fault == 'path': value['model']['path'] = '../verifier.onnx'
    elif fault == 'primary_collision': value['model']['path'] = 'model.onnx'
    elif fault == 'rejection_collision': value['model']['path'] = 'rejection.onnx'
    elif fault == 'sha': value['model']['sha256'] = 'oops'
    save()
    with pytest.raises(ValueError): module.RegionFontClassifier(tmp_path)


@pytest.mark.parametrize('fault', ['missing', 'sha', 'input_name', 'input_shape', 'input_type', 'output_name',
    'output_shape', 'output_type', 'fixed_batch', 'size_output_shape'])
def test_v3_asset_and_onnx_contract_is_strict(tmp_path, monkeypatch, fault):
    _, _, session = consensus_bundle(tmp_path, monkeypatch)
    inputs, outputs = session.get_inputs(), session.get_outputs()
    session.get_inputs = lambda: inputs
    session.get_outputs = lambda: outputs
    if fault == 'missing': (tmp_path / 'verifier.onnx').unlink()
    elif fault == 'sha': (tmp_path / 'verifier.onnx').write_bytes(b'changed verifier')
    elif fault == 'input_name': inputs[0].name = 'glyphs'
    elif fault == 'input_shape': inputs[0].shape[-1] = 64
    elif fault == 'input_type': inputs[0].type = 'tensor(double)'
    elif fault == 'output_name': outputs[0].name = 'known_logits'
    elif fault == 'output_shape': outputs[0].shape[-1] = 2
    elif fault == 'output_type': outputs[0].type = 'tensor(double)'
    elif fault == 'fixed_batch': outputs[0].shape[0] = 1
    elif fault == 'size_output_shape': outputs[1].shape = ['batch', 1]
    with pytest.raises(ValueError): module.RegionFontClassifier(tmp_path)
