"""One dynamic-class CNN and unchanged pixel math, independent of any OS input."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import flux_glyph.unified_font as module
from test_android_font import source


def metadata(count=23):
    names=['PingFang','SF Pro','Helvetica','Alipay Number','Noto Sans CJK SC','Roboto','LXGW WenKai']
    names=(names+[f'Covered family {index}' for index in range(128)])[:count-1]+['__unknown__']
    return {'schema':module.SCHEMA,'algorithm':module.ALGORITHM,'font_mode':'unified',
        'data_kind':'native_mobile_screenshots','families':names,'temperature':1.,
        'gates':{'min_score':.7,'min_margin':.1,'min_patch_agreement':.7},'max_size_relative_spread':.2,
        'model':{'path':'model.onnx','sha256':hashlib.sha256(b'unified-weights').hexdigest()},
        'font_label_groups':{'PingFang':['PingFang SC','PingFang TC']}}


def fake_model(logits,meta=None):
    result=module.UnifiedFontClassifier.__new__(module.UnifiedFontClassifier)
    result.meta=meta or metadata();result.output_families=result.meta['families']
    result.families=[name for name in result.output_families if name!=module.UNKNOWN]
    result.font_label_groups=result.meta['font_label_groups'];result.font_mode='unified'
    def run(names,inputs):
        assert names==['logits','log_em_ratio'] and set(inputs)=={'tiles'}
        assert inputs['tiles'].dtype==np.float32 and inputs['tiles'].shape[1:]==(1,64,256)
        count=len(inputs['tiles'])
        values=logits(count) if callable(logits) else np.tile(logits,(count,1)).astype(np.float32)
        return [values,np.full(count,np.log(1.2),dtype=np.float32)]
    result.session=SimpleNamespace(run=run)
    return result


@pytest.mark.parametrize('winner',[0,1,2,3,4,5,6,21])
def test_all_jointly_covered_families_compete_in_one_model(winner):
    meta=metadata();logits=np.full(23,-4,dtype=np.float32);logits[winner]=8
    model=fake_model(logits,meta);result=model.predict(source())
    assert result['family']==meta['families'][winner] and result['status']=='candidate'
    assert result['font_mode']=='unified' and result['device_inference_performed'] is False
    assert result['font_size_px_estimate']==pytest.approx(module.preprocess_region(source())['ink_height_px']*1.2,abs=.01)
    assert 'rejection' not in result and 'verifier' not in result


def test_old_preprocessing_and_aggregation_are_imported_unchanged():
    from flux_glyph.region_font import preprocess_region,aggregate_predictions
    assert module.preprocess_region is preprocess_region and module.aggregate_predictions is aggregate_predictions
    assert module.RegionFontClassifier is module.UnifiedFontClassifier


@pytest.mark.parametrize('classes',[2,23,128])
def test_full_softmax_keeps_unknown_competition_at_every_class_count(classes):
    meta=metadata(classes);logits=np.full(classes,-10,dtype=np.float32);logits[0]=5;logits[-1]=4.9
    result=fake_model(logits,meta).predict(source())
    assert result['status']=='uncertain' and .52<result['score']<.53 and .04<result['margin']<.06
    assert result['family'] is result['font_size_px_estimate'] is None
    assert result['candidates'][0]['score']==result['score']
    assert all(row['family']!=module.UNKNOWN for row in result['candidates'])


@pytest.mark.parametrize('index',[0,11,22])
def test_unknown_position_is_metadata_defined_and_clears_names_and_size(index):
    meta=metadata();meta['families'][index],meta['families'][-1]=meta['families'][-1],meta['families'][index]
    logits=np.zeros(23,dtype=np.float32);logits[index]=8
    result=fake_model(logits,meta).predict(source())
    assert result['status']=='out_of_scope' and result['reason_code']=='unknown_font_rejected'
    assert result['family'] is result['score'] is result['margin'] is result['font_size_px_estimate'] is None
    assert result['candidates']==[]


@pytest.mark.parametrize('field,value',[
    ('families',['__unknown__']),('families',[f'F{i}' for i in range(128)]+['__unknown__']),
    ('families',['Font','Font','__unknown__']),('families',['Font',' ']),('families',['A','B']),
    ('rejection',None),('verifier',None),('font_mode','android'),('data_kind','desktop_render'),
    ('algorithm','android-region-cnn64x256-v1'),('temperature',float('nan')),
    ('gates',{'min_score':.5,'min_margin':0,'min_patch_agreement':float('inf')}),
    ('model',{'path':'../model.onnx','sha256':'a'*64}),
])
def test_malformed_or_platform_specific_metadata_is_rejected(field,value):
    meta=metadata();meta[field]=value
    with pytest.raises(ValueError):module.unified_metadata(meta)


@pytest.mark.parametrize('value',[
    lambda n:[np.zeros((n,10),np.float32),np.zeros(n,np.float32)],
    lambda n:[np.full((n,23),np.nan,np.float32),np.zeros(n,np.float32)],
    lambda n:[np.zeros((n,23),np.float64),np.zeros(n,np.float32)],
    lambda n:[np.zeros((n,23),np.float32),np.full(n,4.,np.float32)],
])
def test_invalid_outputs_fail_closed(value):
    model=fake_model(np.zeros(23));model.session.run=lambda names,inputs:value(len(inputs['tiles']))
    result=model.predict(source())
    assert result['reason_code']=='invalid_neural_output' and result['candidates']==[]
    assert result['family'] is result['font_size_px_estimate'] is result['score'] is None


@pytest.mark.parametrize('classes',[2,23,128])
def test_constructor_binds_dynamic_onnx_shape_and_sha(monkeypatch,tmp_path,classes):
    meta=metadata(classes);(tmp_path/'metadata.json').write_text(json.dumps(meta));(tmp_path/'model.onnx').write_bytes(b'unified-weights')
    input_node=SimpleNamespace(name='tiles',type='tensor(float)',shape=['N',1,64,256])
    outputs=[SimpleNamespace(name='logits',type='tensor(float)',shape=['N',classes]),
             SimpleNamespace(name='log_em_ratio',type='tensor(float)',shape=['N'])]
    def session(data,sess_options,providers):
        assert data==b'unified-weights' and providers==['CPUExecutionProvider']
        assert sess_options.intra_op_num_threads==sess_options.inter_op_num_threads==1
        return SimpleNamespace(get_inputs=lambda:[input_node],get_outputs=lambda:outputs)
    monkeypatch.setattr(module.ort,'InferenceSession',session)
    assert len(module.UnifiedFontClassifier(tmp_path).families)==classes-1
    outputs[0].shape=['N',classes+1]
    with pytest.raises(ValueError,match='contract'):module.UnifiedFontClassifier(tmp_path)
    (tmp_path/'model.onnx').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'):module.UnifiedFontClassifier(tmp_path)
