from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from flux_glyph.region_font import RegionFontClassifier,preprocess_region,aggregate_predictions


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
