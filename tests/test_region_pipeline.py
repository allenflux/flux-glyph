"""The region model must run using source pixels without any text recognizer."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import flux_glyph.pipeline as module
from flux_glyph.models import file_sha, verify_bundle


def write_bundle(root, omitted=None):
    files = {
        'pp/onnx/paddle_ocr_det.onnx': b'detector',
        'pp/paddle_ocr_delivery.contract.json': b'{}',
        'region_neural/metadata.json': json.dumps({'model': {'path': 'model.onnx'}}).encode(),
        'region_neural/model.onnx': b'region weights',
    }
    rows = []
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        if name != omitted:
            rows.append({'path': name, 'bytes': len(value), 'sha256': file_sha(path)})
    (root / 'MANIFEST.json').write_text(json.dumps({'version': 'region-test', 'files': rows}))


def test_detector_and_region_weights_are_a_complete_bundle(tmp_path):
    write_bundle(tmp_path)
    assert len(verify_bundle(tmp_path)['files']) == 4


def test_region_package_manifest_does_not_require_legacy_font_directory(tmp_path):
    from scripts.package_models import write_models_manifest
    write_bundle(tmp_path)
    manifest = write_models_manifest(tmp_path)
    assert len(manifest['files']) == 4
    assert all(row['path'].startswith(('pp/', 'region_neural/')) for row in manifest['files'])
    assert verify_bundle(tmp_path) == manifest


@pytest.mark.parametrize('omitted', [
    'region_neural/model.onnx', 'region_neural/metadata.json',
    'pp/onnx/paddle_ocr_det.onnx', 'pp/paddle_ocr_delivery.contract.json',
])
def test_even_on_disk_region_dependencies_must_be_hash_declared(tmp_path, omitted):
    write_bundle(tmp_path, omitted)
    with pytest.raises(ValueError, match='lacks'):
        verify_bundle(tmp_path)


def test_region_constructor_and_run_never_use_ocr_or_reference_banks(monkeypatch, tmp_path):
    write_bundle(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('OCR, segmentation and reference matching must never be used')
    for name in ('PPReader', 'CompactFontBank', 'CompactLatinBank',
                 'NeuralFontClassifier', 'segment_characters', 'score_font'):
        monkeypatch.setattr(module, name, forbidden)
    boxes = [dict(source_bbox=[12, 10, 80, 40],
                  quad=[[12, 10], [80, 10], [80, 40], [12, 40]], score=.99)] * 2
    monkeypatch.setattr(module, 'PPRegionDetector', lambda _: SimpleNamespace(detect=lambda _: boxes))
    crops = []
    def predict(crop):
        crops.append(crop.copy())
        return dict(status='candidate', family='SF Pro', candidates=[dict(family='SF Pro', score=.99)],
                    score=.99, margin=.9, patch_agreement=1., reason_code='region_neural_family_candidate',
                    font_size_px_estimate=23.5, size_relative_spread=.01, ocr_performed=False)
    monkeypatch.setattr(module, 'RegionFontClassifier', lambda _: SimpleNamespace(predict=predict))
    engine = module.FontPipeline(tmp_path, max_regions=1)
    assert engine.reader is engine.bank is engine.latin_bank is engine.neural is engine.size_metrics is None
    yy, xx = np.indices((60, 100))
    source = Image.fromarray(np.stack((xx, yy, xx + yy), axis=2).astype(np.uint8))
    source_path = tmp_path / 'source.png'
    source.save(source_path)
    result = engine.run(source_path, tmp_path / 'output', 'test')
    assert result['ocr_performed'] is False
    assert result['font_method'] == 'region_neural_network'
    assert result['summary']['processing_limited_regions'] == 1
    assert 'ocr' not in result['timing_seconds']
    assert len(crops) == 1
    first, capped = result['regions']
    np.testing.assert_array_equal(np.asarray(crops[0]), np.asarray(source.crop(first['source_bbox'])))
    assert first['font']['family'] == 'SF Pro'
    assert first['text_style']['font_size_px_estimate'] == 23.5
    assert first['text_style']['font_size_px_interval'] is None
    assert capped['font']['reason_code'] == 'too_many_regions'
    for region in result['regions']:
        assert region['text'] is None and region['ocr_performed'] is False and region['glyphs'] == []
        with Image.open(tmp_path / 'output' / region['crop_file']) as crop:
            np.testing.assert_array_equal(np.asarray(crop), np.asarray(source.crop(region['source_bbox'])))


def rejection_bundle(root):
    import hashlib
    write_bundle(root)
    directory=root/'region_neural'
    weights=b'independent rejection model'
    (directory/'rejection.onnx').write_bytes(weights)
    model={'path':'model.onnx','sha256':file_sha(directory/'model.onnx')}
    metadata={'schema':'flux-glyph-region-font-v1','algorithm':'region-cnn64x256-rejection-v2',
              'families':['PingFang','SF Pro'],'model':model}
    metadata['rejection']={'schema':'flux-glyph-region-rejection-v1','algorithm':'region-known-unknown-cnn64x256-v1',
        'model':{'path':'rejection.onnx','sha256':hashlib.sha256(weights).hexdigest()},'labels':['unknown','known'],
        'known_families':metadata['families'].copy(),'base_model_sha256':model['sha256'],
        'aggregation':'mean_softmax_known_probability','temperature':1.,'min_known_score':.8}
    (directory/'metadata.json').write_text(json.dumps(metadata))
    from scripts.package_models import write_models_manifest
    write_models_manifest(root)
    return metadata


def test_v2_manifest_must_close_over_both_declared_onnx_models(tmp_path):
    rejection_bundle(tmp_path)
    assert len(verify_bundle(tmp_path)['files'])==5
    manifest=json.loads((tmp_path/'MANIFEST.json').read_text())
    manifest['files']=[row for row in manifest['files'] if row['path']!='region_neural/rejection.onnx']
    (tmp_path/'MANIFEST.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='lacks'):verify_bundle(tmp_path)


def test_rejection_metadata_cannot_claim_another_weight_digest_than_manifest(tmp_path):
    metadata=rejection_bundle(tmp_path)
    metadata['rejection']['model']['sha256']='0'*64
    (tmp_path/'region_neural/metadata.json').write_text(json.dumps(metadata))
    from scripts.package_models import write_models_manifest
    write_models_manifest(tmp_path)
    with pytest.raises(ValueError,match='SHA differs'):verify_bundle(tmp_path)


def test_unknown_region_keeps_color_but_no_font_size_and_counts_out_of_scope(monkeypatch,tmp_path):
    from PIL import ImageDraw
    write_bundle(tmp_path)
    box=dict(source_bbox=[10,10,100,40],quad=[[10,10],[100,10],[100,40],[10,40]],score=.99)
    monkeypatch.setattr(module,'PPRegionDetector',lambda _:SimpleNamespace(detect=lambda _:[box]))
    rejection=dict(method='neural_network',status='rejected',known_score=.01,min_known_score=.8)
    unknown=dict(status='out_of_scope',family=None,candidates=[],score=None,margin=None,patch_agreement=None,
                 reason_code='unknown_font_rejected',font_size_px_estimate=None,rejection=rejection)
    monkeypatch.setattr(module,'RegionFontClassifier',lambda _:SimpleNamespace(predict=lambda _:unknown))
    engine=module.FontPipeline(tmp_path)
    source=Image.new('RGB',(120,60),'white')
    ImageDraw.Draw(source).rectangle((20,18,90,32),fill='#154A6F')
    source_path=tmp_path/'source.png';source.save(source_path)
    result=engine.run(source_path,tmp_path/'output','test-rejected')
    region=result['regions'][0]
    assert region['font']['rejection']==rejection and region['font']['label']=='未知字体'
    assert region['font']['family'] is None and region['font']['candidates']==[]
    assert result['summary']['out_of_scope']==1 and result['summary']['other_candidates']==0
    assert region['text_style']['text_color_hex']=='#154A6F'
    assert region['text_style']['font_size_px_estimate'] is region['text_style']['font_size_px_interval'] is None
    assert region['text_style']['size']['status']=='unavailable'
    assert result['ocr_performed'] is False


def test_packager_preserves_the_independent_rejection_file(monkeypatch,tmp_path):
    import importlib
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    package_module=importlib.import_module('scripts.package_region')
    base=tmp_path/'base'
    metadata=rejection_bundle(base)
    monkeypatch.setattr(package_module,'load_active',lambda _:(base,'r17-test',{}))
    monkeypatch.setattr(package_module,'RegionFontClassifier',lambda _:
        SimpleNamespace(meta=metadata,rejection_meta=metadata['rejection']))
    monkeypatch.setattr(package_module,'validate_runtime',lambda path:verify_bundle(path))
    output=tmp_path/'output'
    result=package_module.package(base,base/'region_neural',output,'r19-rejection-test')
    assert result['files']==5
    assert (output/'region_neural/rejection.onnx').read_bytes()==(base/'region_neural/rejection.onnx').read_bytes()
    assert verify_bundle(output)['version']=='r19-rejection-test'


def consensus_bundle(root):
    metadata=rejection_bundle(root)
    directory=root/'region_neural'
    (directory/'verifier.onnx').write_bytes(b'independent verifier model')
    metadata['algorithm']='region-cnn64x256-consensus-v3'
    metadata['verifier']={'schema':'flux-glyph-region-verifier-v1','algorithm':'region-font-verifier-cnn64x256-v1',
        'model':{'path':'verifier.onnx','sha256':file_sha(directory/'verifier.onnx')},
        'families':['SF Pro','Roboto','PingFang'],'temperature':.75,
        'gates':{'min_score':.7,'min_margin':.1,'min_patch_agreement':.7},
        'base_model_sha256':metadata['model']['sha256']}
    (directory/'metadata.json').write_text(json.dumps(metadata))
    from scripts.package_models import write_models_manifest
    write_models_manifest(root)
    return metadata


@pytest.mark.parametrize('fault',['omitted','wrong_sha','v2','no_rejection'])
def test_consensus_bundle_is_closed_over_all_three_models(tmp_path,fault):
    metadata=consensus_bundle(tmp_path)
    assert len(verify_bundle(tmp_path)['files'])==6
    from scripts.package_models import write_models_manifest
    if fault=='omitted':
        manifest=json.loads((tmp_path/'MANIFEST.json').read_text())
        manifest['files']=[row for row in manifest['files'] if row['path']!='region_neural/verifier.onnx']
        (tmp_path/'MANIFEST.json').write_text(json.dumps(manifest))
    else:
        if fault=='wrong_sha':metadata['verifier']['model']['sha256']='0'*64
        elif fault=='v2':metadata['algorithm']='region-cnn64x256-rejection-v2'
        else:metadata.pop('rejection')
        (tmp_path/'region_neural/metadata.json').write_text(json.dumps(metadata))
        write_models_manifest(tmp_path)
    with pytest.raises(ValueError):verify_bundle(tmp_path)


def test_packager_copies_verifier_and_binds_all_three_models(monkeypatch,tmp_path):
    import importlib
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    packager=importlib.import_module('scripts.package_region')
    base=tmp_path/'base'
    metadata=consensus_bundle(base)
    monkeypatch.setattr(packager,'load_active',lambda _:(base,'r19-test',{}))
    monkeypatch.setattr(packager,'RegionFontClassifier',lambda _:
        SimpleNamespace(meta=metadata,rejection_meta=metadata['rejection'],verifier_meta=metadata['verifier']))
    monkeypatch.setattr(packager,'validate_runtime',lambda path:verify_bundle(path))
    output=tmp_path/'output'
    result=packager.package(base,base/'region_neural',output,'r20-consensus-test')
    assert result['files']==6 and verify_bundle(output)['version']=='r20-consensus-test'
    assert (output/'region_neural/verifier.onnx').read_bytes()==(base/'region_neural/verifier.onnx').read_bytes()


@pytest.mark.parametrize('reason,status,label',[('neural_model_disagreement','uncertain','字体存在分歧'),
    ('verifier_font_out_of_scope','out_of_scope','未知字体')])
def test_verifier_prediction_is_preserved_with_color_and_no_size(monkeypatch,tmp_path,reason,status,label):
    from PIL import ImageDraw
    write_bundle(tmp_path)
    box=dict(source_bbox=[10,10,100,40],quad=[[10,10],[100,10],[100,40],[10,40]],score=.99)
    monkeypatch.setattr(module,'PPRegionDetector',lambda _:SimpleNamespace(detect=lambda _:[box]))
    verifier={'status':'disagreed' if status=='uncertain' else 'out_of_scope','family':None,
              'candidates':[],'score':None,'method':'region_neural_network'}
    result=dict(status=status,family=None,candidates=[],score=None,reason_code=reason,
                font_size_px_estimate=None,verifier=verifier)
    monkeypatch.setattr(module,'RegionFontClassifier',lambda _:SimpleNamespace(predict=lambda _:result))
    engine=module.FontPipeline(tmp_path)
    source=Image.new('RGB',(120,60),'white')
    ImageDraw.Draw(source).rectangle((20,18,90,32),fill='#154A6F')
    path=tmp_path/'source.png';source.save(path)
    output=engine.run(path,tmp_path/'output','test-consensus')
    region=output['regions'][0]
    assert region['font']['verifier']==verifier and region['font']['label']==label
    assert region['font']['family'] is region['text_style']['font_size_px_estimate'] is None
    assert region['text_style']['text_color_hex']=='#154A6F'
    assert output['summary']['out_of_scope']==int(status=='out_of_scope')
