from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

import flux_glyph.region_font as module
from flux_glyph.region_font import RegionFontClassifier,preprocess_region,aggregate_predictions,rejection_metadata


def image(text='苹方未',size=35):
    canvas=Image.new('RGB',(size*len(text)+16,size*2),'#F3F6FA')
    font=ImageFont.truetype(str(Path(__file__).resolve().parents[1]/'assets/annotation.otf'),size)
    ImageDraw.Draw(canvas).text((8,8),text,font=font,fill='#153667')
    return canvas


def fake_model():
    model=RegionFontClassifier.__new__(RegionFontClassifier)
    model.font_label_groups={}
    model.families=['One','Two']
    model.meta={'temperature':1.,'gates':{'min_score':.7,'min_margin':.15,'min_patch_agreement':.7},
                'max_size_relative_spread':.2}
    def run(names,data):
        assert set(data)=={'tiles'}
        tiles=data['tiles'];assert tiles.dtype==np.float32 and tiles.shape[1:]==(1,64,256)
        return np.tile([[5.,0.]],(len(tiles),1)).astype(np.float32),np.full(len(tiles),np.log(1.2),dtype=np.float32)
    model.session=SimpleNamespace(run=run)
    return model


def test_grouped_font_prediction_names_family_without_inventing_region_variant():
    model=fake_model()
    model.families=['PingFang','SF Pro']
    model.font_label_groups={'PingFang':['PingFang SC','PingFang TC','PingFang HK']}
    result=model.predict(image())
    assert result['family']=='PingFang'
    assert result['font_family_variants']==['PingFang SC','PingFang TC','PingFang HK']
    assert result['candidates'][0]['family']=='PingFang'


def test_region_pixels_preserve_proportions_and_need_no_text_or_script():
    prepared=preprocess_region(image())
    assert prepared['status']=='ok' and prepared['tiles'].shape[1:]==(1,64,256)
    assert 1<=prepared['tile_count']<=8 and prepared['whole_width_covered']
    assert prepared['tiles'].min()>=0 and prepared['tiles'].max()<=1
    result=fake_model().predict(image())
    assert result['family']=='One' and result['ocr_performed'] is False
    assert result['font_size_px_estimate']==pytest.approx(prepared['ink_height_px']*1.2,abs=.01)


def test_absolute_size_tracks_source_pixels_after_resize():
    original=image();double=original.resize((original.width*2,original.height*2),Image.Resampling.NEAREST)
    assert fake_model().predict(double)['font_size_px_estimate']==pytest.approx(
        fake_model().predict(original)['font_size_px_estimate']*2,abs=.02)


def test_blank_low_contrast_and_oversized_regions_abstain():
    model=fake_model()
    for source in [Image.new('RGB',(150,40),'white'),Image.new('RGB',(3,40)),Image.new('RGB',(4001,1001))]:
        result=model.predict(source)
        assert result['status']=='uncertain' and result['family'] is None
        assert result['font_size_px_estimate'] is None


def test_patch_disagreement_cannot_emit_an_average_font():
    model=fake_model()
    def run(names,data):
        n=len(data['tiles']);assert n>=3
        logits=np.tile([5.,0.],(n,1)).astype(np.float32);logits[1::2]=[0.,5.]
        return logits,np.zeros(n,dtype=np.float32)
    model.session.run=run
    result=model.predict(image('苹方未'*4))
    assert result['status']=='uncertain' and result['reason_code']=='mixed_or_ambiguous_region'


def test_unsampled_long_region_is_not_reported_as_fully_classified():
    model=fake_model()
    result=model.predict(image('苹方未'*15,20))
    assert result['reason_code']=='region_too_long' and result['family'] is None


def test_aggregation_size_and_scores_are_shared_with_training():
    value=aggregate_predictions(np.array([[3,0],[2,0]]),np.log([1.2,1.4]),temperature=1.)
    assert value['patch_agreement']==1 and value['order'][0]==0
    assert value['em_ratio']==pytest.approx(np.sqrt(1.2*1.4))


def attach_rejection(model, logits=(0., 5.), threshold=.8, temperature=1.):
    model.rejection_meta={'temperature':temperature,'min_known_score':threshold}
    def run(names,data):
        assert names==['known_logits'] and set(data)=={'tiles'}
        tiles=data['tiles']
        assert tiles.dtype==np.float32 and tiles.shape[1:]==(1,64,256)
        values=logits(len(tiles)) if callable(logits) else np.tile(logits,(len(tiles),1)).astype(np.float32)
        return [values]
    model.rejection_session=SimpleNamespace(run=run)
    return model


def test_unknown_network_blocks_even_a_high_scoring_family_before_naming():
    model=attach_rejection(fake_model(),(12.,-12.))
    model.session.run=lambda *args: pytest.fail('Do not run or expose closed-set family ranking after rejection')
    result=model.predict(image())
    assert result['status']=='out_of_scope' and result['reason_code']=='unknown_font_rejected'
    assert result['family'] is result['score'] is result['font_size_px_estimate'] is None
    assert result['candidates']==[] and 'font_family_variants' not in result
    assert result['rejection']['status']=='rejected' and result['rejection']['known_score']<.001
    assert result['ocr_performed'] is False


def test_passed_rejection_preserves_every_original_font_score_gate_and_size_value():
    original=fake_model().predict(image())
    result=attach_rejection(fake_model()).predict(image())
    rejection=result.pop('rejection')
    assert rejection['status']=='passed' and rejection['known_score']==pytest.approx(1/(1+np.exp(-5)))
    assert result==original


def test_rejection_temperature_tile_mean_and_threshold_equality():
    model=attach_rejection(fake_model(),(0.,0.),threshold=.5)
    assert model.predict(image())['rejection']['status']=='passed'
    model.rejection_meta['min_known_score']=.50001
    assert model.predict(image())['status']=='out_of_scope'
    model=attach_rejection(fake_model(),(0.,2.),threshold=.7,temperature=2.)
    assert model.predict(image())['rejection']['known_score']==pytest.approx(1/(1+np.exp(-1)))
    def patches(count):
        assert count>=3
        values=np.tile([0.,8.],(count,1)).astype(np.float32)
        values[0]=[8.,0.]
        return values
    result=attach_rejection(fake_model(),patches,threshold=.8).predict(image('苹方未'*4))
    values=patches(result['tile_count']).astype(np.float64)
    expected=np.mean(np.exp(values[:,1]) / np.exp(values).sum(axis=1))
    assert result['rejection']['known_score']==pytest.approx(expected)
    assert (result['rejection']['status']=='passed')==(expected>=.8)


@pytest.mark.parametrize('invalid', [
    lambda n:np.full((n,2),np.nan,dtype=np.float32),
    lambda n:np.full((n,2),np.inf,dtype=np.float32),
    lambda n:np.zeros((n,3),dtype=np.float32),
    lambda n:np.zeros((n+1,2),dtype=np.float32),
    lambda n:np.zeros((n,2),dtype=np.float64),
    lambda n:[[0.,1.]]*n,
])
def test_invalid_rejection_output_never_falls_back_to_named_candidates(invalid):
    model=attach_rejection(fake_model(),invalid)
    model.session.run=lambda *args:pytest.fail('Invalid rejection must fail closed')
    result=model.predict(image())
    assert result['reason_code']=='invalid_rejection_output' and result['status']=='uncertain'
    assert result['family'] is result['score'] is result['font_size_px_estimate'] is None
    assert result['candidates']==[] and result['rejection']['status']=='unavailable'


def test_rejection_session_failure_and_zero_score_are_safe():
    model=attach_rejection(fake_model())
    def fail(*args):raise RuntimeError('ONNX failure')
    model.rejection_session.run=fail
    assert model.predict(image())['reason_code']=='invalid_rejection_output'
    zero=attach_rejection(fake_model(),(1000.,-1000.)).predict(image())
    assert zero['rejection']['known_score']==0 and zero['status']=='out_of_scope'


def rejection_bundle(tmp_path,monkeypatch):
    primary,reject=b'primary font ONNX',b'independent rejection ONNX'
    families=['PingFang','SF Pro']
    metadata={'schema':module.SCHEMA,'algorithm':module.REJECTION_ALGORITHM,'families':families,
              'model':{'path':'model.onnx','sha256':hashlib.sha256(primary).hexdigest()},
              'temperature':1.,'gates':{'min_score':.7,'min_margin':.15,'min_patch_agreement':.7},
              'max_size_relative_spread':.2}
    metadata['rejection']={'schema':'flux-glyph-region-rejection-v1','algorithm':'region-known-unknown-cnn64x256-v1',
        'model':{'path':'rejection.onnx','sha256':hashlib.sha256(reject).hexdigest()},
        'labels':['unknown','known'],'known_families':families.copy(),'base_model_sha256':metadata['model']['sha256'],
        'aggregation':'mean_softmax_known_probability','temperature':1.2,'min_known_score':.8}
    (tmp_path/'model.onnx').write_bytes(primary);(tmp_path/'rejection.onnx').write_bytes(reject)
    def node(name,shape):return SimpleNamespace(name=name,shape=shape,type='tensor(float)')
    primary_session=SimpleNamespace(get_inputs=lambda:[node('tiles',['batch',1,64,256])],
                    get_outputs=lambda:[node('logits',['batch',2]),node('log_em_ratio',['batch'])])
    rejection_session=SimpleNamespace(get_inputs=lambda:[node('tiles',['batch',1,64,256])],
                    get_outputs=lambda:[node('known_logits',['batch',2])])
    def session(data,*,sess_options,providers):
        assert providers==['CPUExecutionProvider']
        assert sess_options.intra_op_num_threads==sess_options.inter_op_num_threads==1
        assert isinstance(data,bytes)
        return primary_session if data==primary else rejection_session
    monkeypatch.setattr(module.ort,'InferenceSession',session)
    def save():(tmp_path/'metadata.json').write_text(json.dumps(metadata))
    save()
    return metadata,save,rejection_session


def test_v2_loads_both_hash_bound_cpu_models_and_v1_does_not_need_rejection(tmp_path,monkeypatch):
    metadata,save,session=rejection_bundle(tmp_path,monkeypatch)
    model=RegionFontClassifier(tmp_path)
    assert model.rejection_session is session and model.rejection_meta==metadata['rejection']
    metadata['algorithm']=module.ALGORITHM;metadata.pop('rejection');save()
    (tmp_path/'rejection.onnx').unlink()
    assert RegionFontClassifier(tmp_path).rejection_session is None


@pytest.mark.parametrize('fault',['missing','v1_rejection','labels','families','base','temperature','nan_temperature',
    'threshold','nan_threshold','bool_threshold','aggregation','schema','algorithm','path','same_path','sha'])
def test_malformed_rejection_metadata_is_rejected(tmp_path,monkeypatch,fault):
    metadata,save,_=rejection_bundle(tmp_path,monkeypatch)
    rejection=metadata['rejection']
    if fault=='missing':metadata.pop('rejection')
    elif fault=='v1_rejection':metadata['algorithm']=module.ALGORITHM
    elif fault=='labels':rejection['labels']=['known','unknown']
    elif fault=='families':rejection['known_families']=['SF Pro','PingFang']
    elif fault=='base':rejection['base_model_sha256']='0'*64
    elif fault=='temperature':rejection['temperature']=0
    elif fault=='nan_temperature':rejection['temperature']=float('nan')
    elif fault=='threshold':rejection['min_known_score']=1.01
    elif fault=='nan_threshold':rejection['min_known_score']=float('nan')
    elif fault=='bool_threshold':rejection['min_known_score']=True
    elif fault=='aggregation':rejection['aggregation']='max'
    elif fault=='schema':rejection['schema']='other'
    elif fault=='algorithm':rejection['algorithm']='other'
    elif fault=='path':rejection['model']['path']='../outside.onnx'
    elif fault=='same_path':rejection['model']['path']='model.onnx'
    elif fault=='sha':rejection['model']['sha256']='no-hash'
    save()
    with pytest.raises(ValueError):RegionFontClassifier(tmp_path)


@pytest.mark.parametrize('fault',['missing','changed','input_name','input_shape','output_name','output_shape','output_type'])
def test_rejection_assets_and_onnx_contract_cannot_be_omitted_or_substituted(tmp_path,monkeypatch,fault):
    _,_,session=rejection_bundle(tmp_path,monkeypatch)
    if fault=='missing':(tmp_path/'rejection.onnx').unlink()
    elif fault=='changed':(tmp_path/'rejection.onnx').write_bytes(b'changed model')
    elif fault.startswith('input'):
        session.get_inputs=lambda:[SimpleNamespace(name='other' if fault=='input_name' else 'tiles',
                     type='tensor(float)',shape=['batch',1,32,256] if fault=='input_shape' else ['batch',1,64,256])]
    else:
        session.get_outputs=lambda:[SimpleNamespace(name='other' if fault=='output_name' else 'known_logits',
             type='tensor(double)' if fault=='output_type' else 'tensor(float)',
             shape=['batch',1] if fault=='output_shape' else ['batch',2])]
    with pytest.raises(ValueError):RegionFontClassifier(tmp_path)
