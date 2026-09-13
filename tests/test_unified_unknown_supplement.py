"""New native TRAIN supplement rejects leakage, fallback and changed pixels."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image,ImageDraw
from training import prepare_unified_unknown_supplement as m


def owners():
    return {key:set() for key in ('source_id','page_id','source_sha256','native_crop_sha256',
        'normalized_text_sha256','decoded_pixel_sha256','source_font_family','font_file_sha256')}


def natives(tmp_path):
    result=[];fonts=[]
    for i,family in enumerate(sorted(m.EXPECTED_SOURCES)):
        image=Image.new('RGB',(135,49),'white');draw=ImageDraw.Draw(image)
        for n in range(6):
            x=8+n*18;draw.rectangle((x,9+i,x+4,37-i),fill='#333333')
            draw.rectangle((x,10+n,x+10,14+n+i),fill='#333333')
        path=tmp_path/f'image{i}.png';image.save(path);(tmp_path/f'proof{i}.json').write_text('{}')
        font={'id':f'font{i}','family':family,'postscript':f'font{i}PS','sha256':str(i)*64,'ttc_index':0,'allowed_splits':['train']}
        fonts.append(font)
        result.append({'split':'train','training_family':m.UNKNOWN,'native_font_verified':True,
            'source_id':f'source{i}','page_id':f'page{i}','region_id':f'region{i}',
            'font_id':font['id'],'font_family':family,'font_face':font['postscript'],'font_file_sha256':font['sha256'],
            'ttc_index':0,'image':path.name,'image_sha256':m.sha(path),'image_width':135,'image_height':49,
            'bbox':[0,0,135,49],'ink_bbox':[8,8,109,38],'font_size_screen_px':36.,'script':'han',
            'text':f'new native fixture {i}','color':'#333333','proof':f'proof{i}.json'})
    return result,fonts


def plan():return {'regions':2,'pages':2,'per_source_family_regions':1}


@pytest.mark.parametrize('key',['source_id','page_id','source_sha256','normalized_text_sha256','source_font_family','font_file_sha256'])
def test_any_old_partition_identity_or_font_collision_rejects(tmp_path,key):
    rows,fonts=natives(tmp_path);old=owners();row=rows[0]
    value={'source_sha256':row['image_sha256'],'normalized_text_sha256':m.text_sha(row['text']),
        'source_font_family':m.normalized_text(row['font_family'])}.get(key,row.get(key))
    old[key].add(value)
    with pytest.raises(ValueError):m.native_audit(rows,fonts,old,plan())


@pytest.mark.parametrize('change',[{'split':'calibration'},{'split':'test'},{'split':'development_holdout'},
    {'training_family':'PingFang'},{'native_font_verified':False},{'ttc_index':1},{'font_face':'fallbackPS'}])
def test_nontrain_named_or_fallback_identity_rejects_before_pixels(tmp_path,change,monkeypatch):
    rows,fonts=natives(tmp_path);rows[0].update(change)
    monkeypatch.setattr(m.Image,'open',lambda *a,**k:pytest.fail('Pixels opened during metadata check'))
    with pytest.raises(ValueError):m.native_audit(rows,fonts,owners(),plan())


def test_new_unknown_target_is_unified24_and_han_balance_required(tmp_path):
    rows,fonts=natives(tmp_path);assert m.FAMILIES.index(m.UNKNOWN)==24
    assert m.native_audit(rows,fonts,owners(),plan())['script_counts']=={'han':2}
    rows[0]['script']='latin'
    with pytest.raises(ValueError,match='Han'):m.native_audit(rows,fonts,owners(),plan())


def setup_prepare(tmp_path,monkeypatch):
    capture=tmp_path/'capture';capture.mkdir();rows,fonts=natives(capture)
    old=owners();audit=m.native_audit(rows,fonts,old,plan());bindings={str(p):m.sha(p) for p in capture.iterdir()}
    monkeypatch.setattr(m,'context',lambda *args:(deepcopy(rows),old,bindings,audit))
    calls=[]
    monkeypatch.setattr(m,'verify_region',lambda root,row:calls.append(row['region_id']))
    output=tmp_path/'data';m.prepare(capture,tmp_path/'old',tmp_path/'plan',output)
    return capture,output,rows,calls


def test_roundtrip_real_preprocessing_and_size_four_views_with_new_only_inputs(tmp_path,monkeypatch):
    capture,output,_,calls=setup_prepare(tmp_path,monkeypatch);loaded=m.load_supplement(output)
    assert set(calls)=={'region0','region1'} and len(calls)==4
    assert isinstance(loaded['tiles'],np.memmap) and loaded['tiles'].shape[1:]==(1,64,256)
    assert len(loaded['rows'])==8 and all(r['target']==24 and r['split']=='train' for r in loaded['rows'])
    assert set(r['source_font_family'] for r in loaded['rows'])==m.EXPECTED_SOURCES
    assert all(np.isclose(np.exp(r['log_em_ratio'])*r['ink_height_px'],r['font_size_px']) for r in loaded['rows'])
    assert loaded['manifest']['calibration_or_development_or_test_pixels_read'] is False
    assert not (output/'calibration').exists()


@pytest.mark.parametrize('fault',['array','target','crop','size','source'])
def test_rehashed_data_still_must_equal_native_pixels_and_identity(tmp_path,monkeypatch,fault):
    capture,output,_,_=setup_prepare(tmp_path,monkeypatch);folder=output/'train';part=m.read(folder/'MANIFEST.json')
    if fault=='source':
        path=capture/'image0.png';path.write_bytes(path.read_bytes()+b'changed')
    elif fault=='array':
        block=np.memmap(folder/'tiles.raw',dtype='<f4',mode='r+');block[0]=.123;block.flush();del block
        part['array']['sha256']=m.sha(folder/'tiles.raw')
    else:
        rows=m.read(folder/'rows.json')
        if fault=='target':rows[0]['target']=0;rows[0]['family']=m.FAMILIES[0]
        elif fault=='crop':rows[0]['source_crop_bbox'][0]+=1
        else:rows[0]['font_size_px']+=1
        m.dump(folder/'rows.json',rows);part['metadata']['sha256']=m.sha(folder/'rows.json')
    m.dump(folder/'MANIFEST.json',part)
    with pytest.raises(ValueError):m.load_supplement(output)


def test_native_glyph_verifier_failure_prevents_loading(tmp_path,monkeypatch):
    _,output,_,_=setup_prepare(tmp_path,monkeypatch)
    def fallback(*a):raise ValueError('native fallback or missing glyph detected')
    monkeypatch.setattr(m,'verify_region',fallback)
    with pytest.raises(ValueError,match='fallback'):m.load_supplement(output)


@pytest.mark.parametrize('duplicate',['old_crop','old_decoded','new_crop'])
def test_duplicate_pixels_rejected(tmp_path,monkeypatch,duplicate):
    rows,_=natives(tmp_path);old=owners();monkeypatch.setattr(m,'verify_region',lambda *a:None)
    with Image.open(tmp_path/rows[0]['image']) as image:
        if duplicate=='old_crop':old['native_crop_sha256'].add(m.pixels_sha(image.crop(m.source_crop_bbox(rows[0]))))
        elif duplicate=='old_decoded':old['decoded_pixel_sha256'].add(m.pixels_sha(image))
        else:
            rows[1]={**rows[0],'region_id':'different'}
    with pytest.raises(ValueError):list(m.verified_views(tmp_path,rows,old))


def test_historical_audit_uses_metadata_without_any_old_image_or_array(tmp_path,monkeypatch):
    data=tmp_path/'old';data.mkdir();m.dump(data/'MANIFEST.json',{'families':m.FAMILIES})
    for split in ('train','calibration','development_holdout'):
        folder=data/split;folder.mkdir();m.dump(folder/'rows.json',[{'source_id':split,'normalized_text_sha256':m.text_sha(split)}])
        m.dump(folder/'MANIFEST.json',{'root_manifest_sha256':m.sha(data/'MANIFEST.json'),'split':split,
            'families':m.FAMILIES,'views':1,'metadata':{'path':'rows.json','sha256':m.sha(folder/'rows.json')},
            'array':{'path':'does-not-exist.raw','sha256':'0'*64}})
    labels=tmp_path/'old-labels.jsonl';labels.write_text(json.dumps({'split':'test','text':' ＯＬＤ Text ','image':'never-open.png','font_family':'OldFont'})+'\n')
    rows_path=data/'train/rows.json'
    exclusion=tmp_path/'EXCLUSION.json';m.dump(exclusion,{'schema':'flux-glyph-new-train-text-exclusion-v1',
        'source_metadata':[{'path':str(labels),'sha256':m.sha(labels)},
            {'path':str(rows_path),'sha256':m.sha(rows_path)}],
        'hashes':[m.text_sha('oldtext'),m.text_sha('train')]})
    monkeypatch.setattr(m.Image,'open',lambda *a,**k:pytest.fail('Historical image opened'))
    found,audit=m.historical_audit(data,exclusion,{})
    assert m.text_sha('oldtext') in found['normalized_text_sha256'] and 'oldfont' in found['source_font_family']
    assert audit['old_images_or_arrays_read'] is False
    changed=m.read(exclusion);changed['hashes']=[];m.dump(exclusion,changed)
    with pytest.raises(ValueError,match='incomplete'):m.historical_audit(data,exclusion,{})
