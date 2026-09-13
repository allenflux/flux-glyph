"""One ACTIVE classifier serves every request; old bundles stay loadable offline."""
import hashlib
import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import zipfile

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from flux_glyph import api
from flux_glyph.model_download import font_kit
from test_android_font import metadata
from test_unified_font import metadata as unified_metadata


def png():
    buffer = io.BytesIO()
    Image.new('RGB', (32, 32), 'white').save(buffer, 'PNG')
    return buffer.getvalue()


def engines(monkeypatch, tmp_path, mode='unified'):
    android_root = tmp_path/'android-model'
    android_root.mkdir()
    monkeypatch.setenv('FLUX_ANDROID_MODEL_DIR', str(android_root))
    monkeypatch.setattr(api, 'DATA', tmp_path/'jobs')
    monkeypatch.setattr(api, 'TOKEN', '')
    created = []
    class Engine:
        def __init__(self, root, **kwargs):
            assert Path(root) != android_root, 'Separate Android model must not be loaded'
            self.font_mode = mode
            self.version = 'single-'+mode
            self.calls = []
            self.region_neural = None
            self.release = None
            created.append(self)
        def run(self, source, output, identifier, progress):
            self.calls.append(identifier)
            if self.release is not None:
                assert self.release.wait(10)
            return {'id': identifier, 'width': 32, 'height': 32, 'summary': {}, 'timing_seconds': {},
                    'model_version': self.version, 'source_sha256': 'fixture', 'font_scope': 'Detected text region',
                    'font_identity_verified': False, 'device_inference_performed': False,
                    'font_mode': self.font_mode, 'font_method': 'region_neural_network', 'ocr_performed': False,
                    'regions': []}
    monkeypatch.setattr(api, 'FontPipeline', Engine)
    return created


def finish(client, identifier):
    for _ in range(100):
        data = client.get('/api/jobs/'+identifier).json()
        if data['status'] in ('complete', 'error'):
            return data
        time.sleep(.01)
    raise AssertionError('Fixture worker did not complete')


def test_default_and_explicit_unified_pin_the_same_single_engine(monkeypatch, tmp_path):
    created = engines(monkeypatch, tmp_path)
    release = threading.Event()
    try:
        with TestClient(api.app) as client:
            health = client.get('/api/health').json()
            assert health['default_font_mode'] == 'unified' and health['model_version'] == 'single-unified'
            assert health['font_modes'] == {'unified': {'available': True, 'model_version': 'single-unified'}}
            assert len(created) == 1
            original = created[0]
            original.release = release
            first = client.post('/api/jobs', content=png()).json()
            second = client.post('/api/jobs?mode=unified', content=png()).json()
            assert first['font_mode'] == second['font_mode'] == 'unified'
            # Already queued work retains the one selected instance.
            api.app.state.jobs.engine = SimpleNamespace(version='replacement',font_mode='unified')
            release.set()
            done = finish(client, second['id'])
            assert done['status'] == 'complete' and done['result']['font_mode'] == 'unified'
            assert done['result']['model_version'] == 'single-unified'
            assert done['result']['device_inference_performed'] is False
            assert original.calls == [first['id'], second['id']]
    finally:
        release.set()


@pytest.mark.parametrize('legacy_mode', ['ios', 'android'])
def test_old_root_bundle_is_loadable_without_falsely_claiming_joint_training(monkeypatch, tmp_path, legacy_mode):
    created = engines(monkeypatch, tmp_path, mode=legacy_mode)
    with TestClient(api.app) as client:
        health = client.get('/api/health').json()
        assert health['default_font_mode'] == legacy_mode and len(created) == 1
        assert client.get('/api/models/font?mode=unified').status_code == 503
        assert client.get('/api/models/font/download?mode=unified').status_code == 503
        assert client.post('/api/jobs?mode=unified', content=b'bad').status_code == 503
        assert not list(api.DATA.iterdir())
        job = client.post('/api/jobs', content=png()).json()
        assert finish(client, job['id'])['result']['font_mode'] == legacy_mode


@pytest.mark.parametrize('mode', ['ios', 'android', 'windows', '', 'ANDROID', '../ios'])
def test_legacy_or_invalid_modes_reject_before_upload_and_download(monkeypatch, tmp_path, mode):
    engines(monkeypatch, tmp_path)
    with TestClient(api.app) as client:
        for route in ['/api/models/font', '/api/models/font/download']:
            assert client.get(route, params={'mode': mode}).status_code == 400
        for method, route in [('post', '/api/jobs'), ('post', '/api/predict'), ('post', '/upload'),
                              ('put', '/upload/image.png'), ('post', '/')]:
            assert getattr(client, method)(route, params={'mode': mode}, content=b'bad').status_code == 400
        assert not list(api.DATA.iterdir())


def android_engine(tmp_path):
    tmp_path.mkdir()
    (tmp_path/'model.onnx').write_bytes(b'android-weights')
    meta = metadata()
    meta['private_capture_path'] = '/private/captures'
    model = SimpleNamespace(directory=tmp_path, meta=meta, families=meta['families'][:-1])
    return SimpleNamespace(region_neural=model, font_mode='android', version='android-fixture')


def test_android_download_is_self_contained_and_filters_only_public_supported_list(tmp_path):
    engine = android_engine(tmp_path/'model')
    info, payload = font_kit(engine)
    assert info['download_url'] == '/api/models/font/download'
    assert info['font_mode'] == 'android' and info['unknown_font_rejection'] is True
    assert len(info['families']) == 9 and '__unknown__' not in info['families']
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    assert set(files) == {'model.onnx', 'metadata.json', 'inference.py', 'region_font.py', 'text_style.py',
                          'predict.py', 'requirements.txt', 'README.md', 'SHA256.json'}
    saved = json.loads(files['metadata.json'])
    assert saved['families'] == engine.region_neural.meta['families'] and len(saved['families']) == 10
    assert saved['font_label_groups'] == engine.region_neural.meta['font_label_groups']
    assert 'private_capture_path' not in saved
    assert files['inference.py'] == (Path(api.__file__).parent/'android_font.py').read_bytes()
    assert files['region_font.py'] == (Path(api.__file__).parent/'region_font.py').read_bytes()
    assert files['model.onnx'] == b'android-weights' and b'logits [N,10]' in files['README.md']
    assert 'Android Emulator'.encode() in files['README.md'] and b'iOS Simulator' not in files['README.md']
    assert json.loads(files['SHA256.json']) == {name: hashlib.sha256(data).hexdigest() for name, data in files.items() if name != 'SHA256.json'}
    for name in ['inference.py', 'region_font.py', 'text_style.py', 'predict.py']:
        compile(files[name], name, 'exec')


def test_experimental_android_kit_and_api_keep_failed_validation_visible(monkeypatch,tmp_path):
    engines(monkeypatch,tmp_path,mode='android')
    engine=android_engine(tmp_path/'preview-model')
    publication={'release_tier':'experimental','stable_validation_passed':False,'test_passed':False,
        'validation':{'named_precision':.9259954921111946,'known_correct_coverage':.6847222222222222,
            'unknown_wrongly_named':174,'unknown_views':1200,'test_pages':100,'test_native_regions':1200,
            'test_views':4800,'unknown_test_families':['Smiley Sans'],'derived_views_are_correlated':True,
            'onnx_parity_passed':True,'training_font_faces':33,'all_capture_font_faces':35}}
    engine.region_neural.meta.update(publication)
    with TestClient(api.app) as client:
        serving=api.app.state.jobs.engine;serving.region_neural=engine.region_neural
        serving.version='android-open-fonts-v1-preview'
        info=client.get('/api/models/font').json()
        assert all(info[key]==value for key,value in publication.items())
        response=client.get(info['download_url']);assert response.status_code==200
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            metadata=json.loads(archive.read('metadata.json'));readme=archive.read('README.md').decode()
        assert all(metadata[key]==value for key,value in publication.items())
        assert '__unknown__' not in info['families'] and '__unknown__' in metadata['families']
        assert '覆盖9类字体，尚未通过稳定版验收' in readme and '33 个字体 face' in readme and '35 个字体 face' in readme
        assert 'CPU ONNX 完整数值' in readme and '稳定版 TEST 验收未通过' in readme
        assert '99.08' not in readme and client.get('/api/models/font?mode=android').status_code==400


def unified_engine(tmp_path, count=25):
    tmp_path.mkdir()
    (tmp_path/'model.onnx').write_bytes(b'unified-weights')
    meta = unified_metadata(count)
    meta['private_capture_path'] = '/private/captures'
    model = SimpleNamespace(directory=tmp_path, meta=meta, families=meta['families'][:-1])
    return SimpleNamespace(region_neural=model, font_mode='unified', version='unified-fixture')


def test_unified_kit_has_one_cnn_full_dynamic_classes_and_no_platform_routing(tmp_path):
    engine=unified_engine(tmp_path/'unified')
    engine.region_neural.meta.update(release_tier='experimental',stable_validation_passed=False,
                                    test_passed=False,validation={'test_has_not_passed':True})
    info,payload=font_kit(engine)
    assert info['font_mode']=='unified' and len(info['families'])==24 and '__unknown__' not in info['families']
    assert info['unknown_font_rejection'] is True and info['font_consensus_verification'] is False
    assert info['download_url']=='/api/models/font/download'
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        files={name:archive.read(name) for name in archive.namelist()}
    assert [name for name in files if name.endswith('.onnx')]==['model.onnx']
    saved=json.loads(files['metadata.json']);readme=files['README.md'].decode()
    assert saved['families']==engine.region_neural.meta['families'] and len(saved['families'])==25
    assert saved['font_mode']=='unified' and saved['data_kind']=='native_mobile_screenshots'
    assert saved['stable_validation_passed'] is False and saved['test_passed'] is False
    assert 'private_capture_path' not in saved and 'rejection' not in saved and 'verifier' not in saved
    assert files['inference.py']==(Path(api.__file__).parent/'unified_font.py').read_bytes()
    assert files['region_font.py']==(Path(api.__file__).parent/'region_font.py').read_bytes()
    assert 'iOS Simulator 与 Android Emulator' in readme and 'logits [N,25]' in readme
    assert '联合识别 24 类' in readme and '无需选择 iOS／Android' in readme and '尚未通过稳定版验收' in readme
    assert json.loads(files['SHA256.json'])=={name:hashlib.sha256(value).hexdigest() for name,value in files.items() if name!='SHA256.json'}
    for name in ['inference.py','region_font.py','text_style.py','predict.py']:compile(files[name],name,'exec')


def test_single_download_uses_one_cache_and_auth(monkeypatch, tmp_path):
    engines(monkeypatch, tmp_path)
    monkeypatch.setattr(api, 'TOKEN', 'scope-test-token')
    with TestClient(api.app) as client:
        model_engine = unified_engine(tmp_path/'download-model')
        api.app.state.jobs.engine.region_neural = model_engine.region_neural
        for route in ['/api/models/font', '/api/models/font/download', '/api/models/font?mode=unified']:
            assert client.get(route).status_code == 401
        assert client.post('/auth', json={'token': 'scope-test-token'}).status_code == 200
        info = client.get('/api/models/font').json()
        response = client.get(info['download_url'])
        assert info['version'] == 'single-unified' and response.status_code == 200
        assert response.headers['Content-Disposition'] == 'attachment; filename="flux-glyph-font-onnx.zip"'
        assert hashlib.sha256(response.content).hexdigest() == info['sha256']
        assert client.get('/api/models/font?mode=unified').json() == info
        assert client.get('/api/models/font/download?mode=unified').content == response.content
        assert client.get('/api/models/font?mode=ios').status_code == 400
        assert hasattr(api.app.state.jobs.engine, '_font_download')


def test_all_compat_upload_routes_use_the_one_unified_model(monkeypatch, tmp_path):
    created = engines(monkeypatch, tmp_path)
    with TestClient(api.app) as client:
        for method, route in [('post', '/api/predict'), ('post', '/upload'), ('post', '/'),
                              ('put', '/upload/test.png')]:
            for suffix in ['?wait=false','?mode=unified&wait=false']:
                response = getattr(client, method)(route+suffix, content=png())
                assert response.status_code in (200, 202)
                job = response.json()
                assert job['font_mode'] == 'unified'
                assert finish(client, job['id'])['result']['font_mode'] == 'unified'
        assert len(created)==1 and len(created[0].calls)==8


def test_android_pipeline_never_loads_ios_classifier_or_ocr(monkeypatch, tmp_path):
    import flux_glyph.pipeline as pipeline
    from flux_glyph.models import verify_bundle
    from scripts.package_models import write_models_manifest
    from test_region_pipeline import write_bundle
    write_bundle(tmp_path)
    meta = metadata()
    meta['model']['sha256'] = hashlib.sha256((tmp_path/'region_neural/model.onnx').read_bytes()).hexdigest()
    (tmp_path/'region_neural/metadata.json').write_text(json.dumps(meta))
    manifest = write_models_manifest(tmp_path)
    manifest['version'] = 'android-fixture'
    (tmp_path/'MANIFEST.json').write_text(json.dumps(manifest))
    assert len(verify_bundle(tmp_path)['files']) == 4
    for name in ['RegionFontClassifier', 'PPReader', 'CompactFontBank', 'CompactLatinBank', 'NeuralFontClassifier']:
        monkeypatch.setattr(pipeline, name, lambda *args, **kwargs: pytest.fail('Android must not use the iOS font engine or OCR'))
    calls = []
    def predict(image):
        calls.append(image.size)
        return {'status': 'out_of_scope', 'family': None, 'candidates': [], 'score': None,
                'reason_code': 'unknown_font_rejected', 'font_size_px_estimate': None}
    monkeypatch.setattr(pipeline, 'AndroidFontClassifier', lambda _: SimpleNamespace(predict=predict))
    monkeypatch.setattr(pipeline, 'PPRegionDetector', lambda _: SimpleNamespace(detect=lambda image: [
        {'source_bbox': [4, 4, 28, 28], 'quad': [[4,4],[28,4],[28,28],[4,28]], 'score': .99}]))
    engine = pipeline.FontPipeline(tmp_path)
    source = tmp_path/'source.png'
    source.write_bytes(png())
    value = engine.run(source, tmp_path/'output', 'android-source')
    assert engine.font_mode == value['font_mode'] == 'android' and calls
    assert value['ocr_performed'] is False and value['device_inference_performed'] is False
    assert value['regions'][0]['font']['font_mode'] == 'android'
    assert value['regions'][0]['font']['status'] == 'out_of_scope'
    assert value['regions'][0]['font']['family'] is value['regions'][0]['text_style']['font_size_px_estimate'] is None
    # A self-consistent manifest cannot silently change the model's own SHA binding.
    meta['model']['sha256'] = '0'*64
    (tmp_path/'region_neural/metadata.json').write_text(json.dumps(meta))
    write_models_manifest(tmp_path)
    with pytest.raises(ValueError, match='SHA differs'):
        verify_bundle(tmp_path)


def test_unified_pipeline_loads_only_joint_classifier_and_preserves_pixel_results(monkeypatch,tmp_path):
    import flux_glyph.pipeline as pipeline
    from flux_glyph.models import verify_bundle
    from scripts.package_models import write_models_manifest
    from test_region_pipeline import write_bundle
    write_bundle(tmp_path)
    meta=unified_metadata(25)
    meta['model']['sha256']=hashlib.sha256((tmp_path/'region_neural/model.onnx').read_bytes()).hexdigest()
    meta.update(release_tier='experimental',stable_validation_passed=False)
    (tmp_path/'region_neural/metadata.json').write_text(json.dumps(meta))
    manifest=write_models_manifest(tmp_path)
    manifest['version']='unified-fixture'
    (tmp_path/'MANIFEST.json').write_text(json.dumps(manifest))
    assert len(verify_bundle(tmp_path)['files'])==4
    def forbidden(*args,**kwargs):pytest.fail('Unified pipeline must not load platform classifiers or OCR')
    for name in ['RegionFontClassifier','AndroidFontClassifier','PPReader','CompactFontBank','CompactLatinBank','NeuralFontClassifier']:
        monkeypatch.setattr(pipeline,name,forbidden)
    calls=[]
    def predict(image):
        calls.append(image.copy())
        return {'status':'candidate','family':'Roboto','candidates':[{'family':'Roboto','score':.91}],
                'score':.91,'reason_code':'region_neural_family_candidate','font_size_px_estimate':21.5}
    monkeypatch.setattr(pipeline,'UnifiedFontClassifier',lambda _:SimpleNamespace(predict=predict,meta=meta))
    monkeypatch.setattr(pipeline,'PPRegionDetector',lambda _:SimpleNamespace(detect=lambda image:[
        {'source_bbox':[4,4,28,28],'quad':[[4,4],[28,4],[28,28],[4,28]],'score':.99}]))
    engine=pipeline.FontPipeline(tmp_path)
    source=tmp_path/'source.png';source.write_bytes(png())
    value=engine.run(source,tmp_path/'output','joint-source')
    assert engine.font_mode==value['font_mode']=='unified' and len(calls)==1
    assert value['ocr_performed'] is False and value['device_inference_performed'] is False
    assert value['model_release_tier']=='experimental' and value['stable_validation_passed'] is False
    assert value['regions'][0]['font']['family']=='Roboto'
    assert value['regions'][0]['font']['font_mode']=='unified'
    assert value['regions'][0]['text_style']['font_size_px_estimate']==21.5
    # Artifact claims are checked independently of the enclosing manifest.
    meta['model']['sha256']='0'*64
    (tmp_path/'region_neural/metadata.json').write_text(json.dumps(meta))
    write_models_manifest(tmp_path)
    with pytest.raises(ValueError,match='SHA differs'):verify_bundle(tmp_path)


def test_unified_packager_includes_one_joint_onnx_and_detector(monkeypatch,tmp_path):
    import importlib
    from flux_glyph.models import verify_bundle
    from test_region_pipeline import write_bundle
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    packager=importlib.import_module('scripts.package_region')
    base=tmp_path/'base';write_bundle(base)
    meta=unified_metadata()
    meta['model']['sha256']=hashlib.sha256((base/'region_neural/model.onnx').read_bytes()).hexdigest()
    (base/'region_neural/metadata.json').write_text(json.dumps(meta))
    monkeypatch.setattr(packager,'load_active',lambda _:(base,'historical-base',{}))
    monkeypatch.setattr(packager,'UnifiedFontClassifier',lambda _:SimpleNamespace(meta=meta,font_mode='unified'))
    for name in ['RegionFontClassifier','AndroidFontClassifier']:
        monkeypatch.setattr(packager,name,lambda _:pytest.fail('No platform model must be loaded'))
    monkeypatch.setattr(packager,'validate_runtime',verify_bundle)
    output=tmp_path/'output'
    result=packager.package(base,base/'region_neural',output,'unified-font-fixture')
    assert result['files']==4
    manifest=verify_bundle(output)
    assert manifest['font_mode']=='unified' and manifest['ocr_performed'] is False
    assert sorted(path.name for path in (output/'region_neural').iterdir())==['metadata.json','model.onnx']
