"""Routing and source-pixel contracts for mixed mobile screenshot regions."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import flux_glyph.pipeline as pipeline_module
from flux_glyph.font_matcher import han
from flux_glyph.latin_matcher import latin_character
from flux_glyph.pipeline import FontPipeline


def run_region(monkeypatch,tmp_path,text,*,confidence=.99,rotation=0,latin_enabled=True,missing_han=False):
    # Asymmetric source pixels detect accidental use of upright coordinates on
    # the unrotated source. The classifier is mocked; disk crops are real.
    yy,xx=np.indices((64,128))
    pixels=np.stack((xx,yy,(xx+3*yy)%256),axis=2).astype(np.uint8)
    source=tmp_path/'source.png';Image.fromarray(pixels).save(source)
    detector_box=[15,10,105,50]
    reading={'text':text,'confidence':confidence,'tokens':[],
             'metadata':{'orientation_degrees':rotation}}
    classified_han=[]
    def match(samples):
        classified_han.extend(sample['character'] for sample in samples)
        return {'family_candidate':'PingFang SC','family_scores':{'PingFang SC':0.001}}
    def score_han(bank,samples,complete):
        assert all(han(sample['character']) for sample in samples)
        return {'accepted_pingfang':complete,'family':'PingFang SC' if complete else None,
                'candidates':[{'family':'PingFang SC','distance':0.001}],
                'complete':complete}
    def segmentation(image,text,tokens,**kwargs):
        return {'characters':[{'index':i,'character':c,'status':'ok','reason':'source_ink',
                               'bbox':[18+i*2,8,34+i*2,31]}
                              for i,c in enumerate(text)]}
    monkeypatch.setattr(pipeline_module,'score_font',score_han)
    monkeypatch.setattr(pipeline_module,'segment_characters',segmentation)
    latin_calls=[]
    def score_latin(image,text,tokens,metadata):
        latin_calls.append(image.copy())
        return {'status':'candidate','family':'SF Pro','reason':'stable_latin_family_candidate',
                'candidates':[{'family':'SF Pro','distance':0.001}],
                'glyphs':[{'index':i,'character':c,'status':'ok','reason':'source_ink',
                           'bbox':[4+i*2,6,14+i*2,28]}
                          for i,c in enumerate(text) if latin_character(c)],
                'segmentation':segmentation(image,text,tokens),
                'evidence':{'candidate_only':True,'complete':True}}
    engine=FontPipeline.__new__(FontPipeline)
    engine.detector=SimpleNamespace(detect=lambda image:[{'source_bbox':detector_box.copy(),
        'quad':[[15,10],[105,10],[105,50],[15,50]],'score':.99}])
    engine.reader=SimpleNamespace(read=lambda images:[deepcopy(reading) for _ in images])
    engine.bank=SimpleNamespace(entries={c:c for c in text if han(c) and not (missing_han and c=='未')},
                                match=match,cache_bytes=0)
    engine.latin_bank=SimpleNamespace(score=score_latin,cache_bytes=0) if latin_enabled else None
    engine.max_regions=200;engine.version='test-mobile-contract'
    output=tmp_path/'result'
    result=engine.run(source,output,'mobile-contract')
    return result,output,Image.fromarray(pixels),classified_han,latin_calls


def test_numeric_region_reaches_latin_matcher(monkeypatch,tmp_path):
    result,_,_,classified_han,calls=run_region(monkeypatch,tmp_path,'22:43')
    region=result['regions'][0]
    assert region['font']['status']=='candidate'
    assert region['font']['family']=='SF Pro'
    assert region['font']['scope']=='Latin letters and digits only'
    assert region['font']['character_indices']==[0,1,3,4]
    assert result['summary']['latin_candidates']==1
    assert result['summary']['pingfang_supported']==0
    assert classified_han==[] and len(calls)==1
    assert [g['character'] for g in region['glyphs']]==list('2243')


@pytest.mark.parametrize('missing_han',[False,True])
def test_mixed_family_evidence_does_not_cross_script_scope(monkeypatch,tmp_path,missing_han):
    result,_,_,classified_han,_=run_region(monkeypatch,tmp_path,'未12中',missing_han=missing_han)
    region=result['regions'][0];chinese,latin=region['font']['components']
    assert chinese['scope']=='Chinese glyphs only'
    assert latin['scope']=='Latin letters and digits only'
    assert chinese['character_indices']==[0,3]
    assert latin['character_indices']==[1,2]
    assert latin['family']=='SF Pro' and latin['status']=='candidate'
    assert region['font']['family']==chinese['family']
    assert chinese['family']==(None if missing_han else 'PingFang SC')
    assert chinese['status']==('uncertain' if missing_han else 'supported')
    assert all(han(c) for c in classified_han)
    assert sorted(g['index'] for g in region['glyphs'])==list(range(4))


@pytest.mark.parametrize('text',['22:43','中12文'])
def test_low_ocr_confidence_cannot_emit_family_candidate(monkeypatch,tmp_path,text):
    result,_,_,_,_=run_region(monkeypatch,tmp_path,text,confidence=.79)
    font=result['regions'][0]['font']
    assert font['status']=='uncertain' and font['family'] is None
    for component in font.get('components',[]):
        assert component['status']=='uncertain' and component['family'] is None
    assert result['summary']['latin_candidates']==0
    assert result['summary']['pingfang_supported']==0


@pytest.mark.parametrize('rotation',[0,180])
def test_mixed_saved_glyph_pixels_match_reported_source_boxes(monkeypatch,tmp_path,rotation):
    result,output,source,_,calls=run_region(monkeypatch,tmp_path,'中12文',rotation=rotation)
    region=result['regions'][0]
    expected_roi=source.crop(region['source_bbox'])
    if rotation:
        expected_roi=expected_roi.transpose(Image.Transpose.ROTATE_180)
    np.testing.assert_array_equal(np.asarray(calls[0]),np.asarray(expected_roi))
    for glyph in region['glyphs']:
        expected=source.crop(glyph['source_bbox'])
        if rotation:
            expected=expected.transpose(Image.Transpose.ROTATE_180)
        with Image.open(output/glyph['crop_file']) as actual:
            np.testing.assert_array_equal(np.asarray(actual),np.asarray(expected))
        assert glyph['source_rotation_degrees']==rotation


def test_legacy_bundle_without_latin_keeps_chinese_pipeline_usable(monkeypatch,tmp_path):
    # A stray local latin/ directory must not be loaded unless declared by the
    # hash-verified bundle manifest.
    (tmp_path/'latin').mkdir()
    monkeypatch.setattr(pipeline_module,'load_active',lambda root:(tmp_path,'legacy',
        {'files':[{'path':'font/metadata.json'}]}))
    monkeypatch.setattr(pipeline_module,'PPRegionDetector',lambda path:object())
    monkeypatch.setattr(pipeline_module,'PPReader',lambda path:object())
    monkeypatch.setattr(pipeline_module,'CompactFontBank',lambda path,size:object())
    def unexpected_latin(*args):
        pytest.fail('Undeclared Latin bank must not be loaded for legacy bundle')
    monkeypatch.setattr(pipeline_module,'CompactLatinBank',unexpected_latin)
    assert FontPipeline(tmp_path).latin_bank is None
    result,_,_,classified_han,calls=run_region(monkeypatch,tmp_path,'中文',latin_enabled=False)
    assert result['regions'][0]['font']['status']=='supported'
    assert result['regions'][0]['font']['family']=='PingFang SC'
    assert classified_han==list('中文') and calls==[]
