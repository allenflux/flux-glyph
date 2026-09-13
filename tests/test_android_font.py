import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

import flux_glyph.android_font as module
from flux_glyph.android_font import AndroidFontClassifier, android_metadata


def metadata():
    return {'schema': module.SCHEMA, 'algorithm': module.ALGORITHM, 'font_mode': 'android',
            'data_kind': 'android_emulator_screenshot',
            'families': ['Noto Sans CJK SC', 'Noto Serif CJK SC', 'LXGW WenKai', 'WenQuanYi Micro Hei',
                         'ZCOOL KuaiLe', 'ZCOOL XiaoWei', 'ZCOOL QingKe HuangYou', 'Ma Shan Zheng',
                         'Roboto', '__unknown__'],
            'temperature': 1., 'gates': {'min_score': .7, 'min_margin': .1, 'min_patch_agreement': .7},
            'max_size_relative_spread': .2,
            'model': {'path': 'model.onnx', 'sha256': hashlib.sha256(b'android-weights').hexdigest()},
            'font_label_groups': {'Noto Sans CJK SC': ['Noto Sans CJK SC', 'Source Han Sans']}}


def fake_model(logits=None):
    result = AndroidFontClassifier.__new__(AndroidFontClassifier)
    result.meta = metadata()
    result.output_families = result.meta['families']
    result.families = result.output_families[:-1]
    result.font_label_groups = result.meta['font_label_groups']
    result.font_mode = 'android'
    def run(names, data):
        assert names == ['logits', 'log_em_ratio'] and set(data) == {'tiles'}
        n = len(data['tiles'])
        assert data['tiles'].dtype == np.float32 and data['tiles'].shape[1:] == (1, 64, 256)
        values = np.zeros((n, 10), np.float32)
        values[:, 0] = 8
        if logits is not None:
            values = logits(n) if callable(logits) else np.tile(logits, (n, 1)).astype(np.float32)
        return [values, np.full(n, np.log(1.2), np.float32)]
    result.session = SimpleNamespace(run=run)
    return result


def source():
    result = Image.new('RGB', (180, 50), 'white')
    draw = ImageDraw.Draw(result)
    for x in range(8, 170, 12):
        draw.rectangle((x, 10, x+5, 38), fill='#123456')
    return result


def test_independent_ten_class_prediction_preserves_original_preprocessor():
    from flux_glyph.region_font import preprocess_region, aggregate_predictions
    assert module.preprocess_region is preprocess_region
    assert module.aggregate_predictions is aggregate_predictions
    model = fake_model()
    value = model.predict(source())
    assert value['status'] == 'candidate' and value['family'] == 'Noto Sans CJK SC'
    assert value['font_family_variants'] == ['Noto Sans CJK SC', 'Source Han Sans']
    assert value['font_mode'] == 'android' and value['device_inference_performed'] is False
    assert value['font_size_px_estimate'] == pytest.approx(preprocess_region(source())['ink_height_px']*1.2, abs=.01)
    assert module.RegionFontClassifier is AndroidFontClassifier


@pytest.mark.parametrize('index', [0, 4, 9])
def test_unknown_uses_metadata_order_and_clears_every_named_result(index):
    model = fake_model([8. if i == index else 0. for i in range(10)])
    families = model.output_families.copy()
    families[index], families[9] = families[9], families[index]
    model.output_families = families
    result = model.predict(source())
    assert result['status'] == 'out_of_scope' and result['reason_code'] == 'unknown_font_rejected'
    assert result['family'] is result['score'] is result['margin'] is result['font_size_px_estimate'] is None
    assert result['candidates'] == []


def test_unknown_runner_up_is_not_removed_from_score_or_margin():
    model = fake_model([5.] + [-10.]*8 + [4.9])
    result = model.predict(source())
    assert result['status'] == 'uncertain' and result['reason_code'] == 'below_score_gate'
    assert .52 < result['score'] < .53 and .04 < result['margin'] < .06
    assert result['candidates'][0]['score'] == result['score']
    assert '__unknown__' not in [row['family'] for row in result['candidates']]


@pytest.mark.parametrize('key,reason', [('min_score', 'below_score_gate'),
                                      ('min_margin', 'ambiguous_neural_families'),
                                      ('min_patch_agreement', 'mixed_or_ambiguous_region')])
def test_all_gates_withhold_family_and_size(key, reason):
    model = fake_model()
    if key == 'min_patch_agreement':
        def disagreement(n):
            assert n > 1
            logits = np.zeros((n, 10), np.float32)
            logits[:, 0] = 8
            logits[0] = 0
            logits[0, 1] = 8
            return logits
        model = fake_model(disagreement)
    model.meta['gates'][key] = 1.
    result = model.predict(source())
    assert result['reason_code'] == reason and result['status'] == 'uncertain'
    assert result['family'] is result['font_size_px_estimate'] is None


@pytest.mark.parametrize('invalid', [
    lambda n: [np.zeros((n, 9), np.float32), np.zeros(n, np.float32)],
    lambda n: [np.full((n, 10), np.nan, np.float32), np.zeros(n, np.float32)],
    lambda n: [np.zeros((n, 10), np.float64), np.zeros(n, np.float32)],
    lambda n: [np.zeros((n, 10), np.float32), np.full(n, 4., np.float32)],
])
def test_invalid_outputs_fail_closed(invalid):
    model = fake_model()
    model.session.run = lambda names, data: invalid(len(data['tiles']))
    result = model.predict(source())
    assert result['reason_code'] == 'invalid_neural_output'
    assert result['family'] is result['score'] is result['font_size_px_estimate'] is None
    assert result['candidates'] == []


@pytest.mark.parametrize('field,value', [
    ('algorithm', 'region-cnn64x256-v1'), ('font_mode', 'ios'), ('data_kind', 'desktop_render'),
    ('families', ['a']*10), ('families', [str(i) for i in range(10)]),
    ('temperature', float('nan')), ('max_size_relative_spread', True),
    ('gates', {'min_score': .5, 'min_margin': .1}),
    ('model', {'path': '../model.onnx', 'sha256': 'a'*64}),
    ('font_label_groups', {'__unknown__': ['Secret']}), ('rejection', None), ('verifier', None),
])
def test_malformed_metadata_rejected(field, value):
    meta = metadata()
    meta[field] = value
    with pytest.raises(ValueError):
        android_metadata(meta)


def test_constructor_sha_and_onnx_contracts(monkeypatch, tmp_path):
    meta = metadata()
    (tmp_path/'metadata.json').write_text(json.dumps(meta))
    (tmp_path/'model.onnx').write_bytes(b'android-weights')
    input_node = SimpleNamespace(name='tiles', type='tensor(float)', shape=['N', 1, 64, 256])
    output_nodes = [SimpleNamespace(name='logits', type='tensor(float)', shape=['N', 10]),
                    SimpleNamespace(name='log_em_ratio', type='tensor(float)', shape=['N'])]
    def session(data, sess_options, providers):
        assert data == b'android-weights' and providers == ['CPUExecutionProvider']
        assert sess_options.intra_op_num_threads == sess_options.inter_op_num_threads == 1
        return SimpleNamespace(get_inputs=lambda: [input_node], get_outputs=lambda: output_nodes)
    monkeypatch.setattr(module.ort, 'InferenceSession', session)
    assert len(AndroidFontClassifier(tmp_path).families) == 9
    output_nodes[0].shape = ['N', 9]
    with pytest.raises(ValueError, match='contract'):
        AndroidFontClassifier(tmp_path)
    (tmp_path/'model.onnx').write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA'):
        AndroidFontClassifier(tmp_path)
