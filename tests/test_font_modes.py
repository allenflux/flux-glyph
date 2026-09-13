"""Separate trained engines share queue capacity, never font logits or fallback."""
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


def png():
    buffer = io.BytesIO()
    Image.new('RGB', (32, 32), 'white').save(buffer, 'PNG')
    return buffer.getvalue()


def engines(monkeypatch, tmp_path, android=True, broken=False):
    android_root = tmp_path/'android-model'
    if android:
        android_root.mkdir()
    monkeypatch.setenv('FLUX_ANDROID_MODEL_DIR', str(android_root))
    monkeypatch.setattr(api, 'DATA', tmp_path/'jobs')
    monkeypatch.setattr(api, 'TOKEN', '')
    created = {}
    class Engine:
        def __init__(self, root, **kwargs):
            self.font_mode = 'android' if Path(root) == android_root else 'ios'
            if self.font_mode == 'android' and broken:
                raise ValueError('Invalid model SHA')
            self.version = 'separate-'+self.font_mode
            self.calls = []
            self.region_neural = None
            self.release = None
            created[self.font_mode] = self
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


def test_default_stays_ios_android_uses_separate_engine_and_pins_queued_instance(monkeypatch, tmp_path):
    created = engines(monkeypatch, tmp_path)
    release = threading.Event()
    try:
        with TestClient(api.app) as client:
            health = client.get('/api/health').json()
            assert health['default_font_mode'] == 'ios' and health['model_version'] == 'separate-ios'
            assert health['font_modes']['android'] == {'available': True, 'model_version': 'separate-android'}
            created['ios'].release = release
            first = client.post('/api/jobs', content=png()).json()
            second = client.post('/api/jobs?mode=android', content=png()).json()
            assert first['font_mode'] == 'ios' and second['font_mode'] == 'android'
            # The queued job retains its original engine even if the registry changes.
            api.app.state.jobs.engines['android'] = created['ios']
            release.set()
            done = finish(client, second['id'])
            assert done['status'] == 'complete' and done['result']['font_mode'] == 'android'
            assert done['result']['model_version'] == 'separate-android'
            assert done['result']['device_inference_performed'] is False
            assert created['android'].calls == [second['id']] and created['ios'].calls == [first['id']]
    finally:
        release.set()


@pytest.mark.parametrize('present,broken', [(False, False), (True, True)])
def test_unavailable_android_has_no_upload_writes_or_ios_fallback(monkeypatch, tmp_path, present, broken):
    created = engines(monkeypatch, tmp_path, android=present, broken=broken)
    with TestClient(api.app) as client:
        assert client.get('/api/health').json()['font_modes']['android']['available'] is False
        assert client.get('/api/models/font?mode=android').json() == {'available': False, 'format': 'ONNX', 'font_mode': 'android'}
        assert client.get('/api/models/font/download?mode=android').status_code == 503
        for method, route in [('post', '/api/jobs'), ('post', '/api/predict'), ('post', '/upload'), ('put', '/upload/image.png'), ('post', '/')]:
            response = getattr(client, method)(route+'?mode=android', content=b'not even an image')
            assert response.status_code == 503
        assert not list(api.DATA.iterdir()) and created['ios'].calls == []


@pytest.mark.parametrize('mode', ['windows', '', 'ANDROID', '../ios'])
def test_invalid_modes_reject_before_upload_and_download(monkeypatch, tmp_path, mode):
    engines(monkeypatch, tmp_path)
    with TestClient(api.app) as client:
        for route in ['/api/models/font', '/api/models/font/download']:
            assert client.get(route, params={'mode': mode}).status_code == 400
        assert client.post('/api/jobs', params={'mode': mode}, content=b'bad').status_code == 400
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
    assert info['download_url'] == '/api/models/font/download?mode=android'
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
    engines(monkeypatch,tmp_path)
    engine=android_engine(tmp_path/'preview-model')
    publication={'release_tier':'experimental','stable_validation_passed':False,'test_passed':False,
        'validation':{'named_precision':.9259954921111946,'known_correct_coverage':.6847222222222222,
            'unknown_wrongly_named':174,'unknown_views':1200,'test_pages':100,'test_native_regions':1200,
            'test_views':4800,'unknown_test_families':['Smiley Sans'],'derived_views_are_correlated':True,
            'onnx_parity_passed':True,'training_font_faces':33,'all_capture_font_faces':35}}
    engine.region_neural.meta.update(publication)
    with TestClient(api.app) as client:
        serving=api.app.state.jobs.engines['android'];serving.region_neural=engine.region_neural
        serving.version='android-open-fonts-v1-preview'
        info=client.get('/api/models/font?mode=android').json()
        assert all(info[key]==value for key,value in publication.items())
        response=client.get(info['download_url']);assert response.status_code==200
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            metadata=json.loads(archive.read('metadata.json'));readme=archive.read('README.md').decode()
        assert all(metadata[key]==value for key,value in publication.items())
        assert '__unknown__' not in info['families'] and '__unknown__' in metadata['families']
        assert '覆盖9类字体，尚未通过稳定版验收' in readme and '33 个字体 face' in readme and '35 个字体 face' in readme
        assert 'CPU ONNX 完整数值' in readme and '稳定版 TEST 验收未通过' in readme
        assert '99.08' not in readme and 'test_passed' not in client.get('/api/models/font').json()


def test_mode_downloads_use_matching_engine_cache_and_auth(monkeypatch, tmp_path):
    engines(monkeypatch, tmp_path)
    monkeypatch.setattr(api, 'TOKEN', 'scope-test-token')
    with TestClient(api.app) as client:
        model_engine = android_engine(tmp_path/'download-model')
        api.app.state.jobs.engines['android'].region_neural = model_engine.region_neural
        for route in ['/api/models/font?mode=android', '/api/models/font/download?mode=android']:
            assert client.get(route).status_code == 401
        assert client.post('/auth', json={'token': 'scope-test-token'}).status_code == 200
        info = client.get('/api/models/font?mode=android').json()
        response = client.get(info['download_url'])
        assert info['version'] == 'separate-android' and response.status_code == 200
        assert response.headers['Content-Disposition'] == 'attachment; filename="flux-glyph-android-font-onnx.zip"'
        assert hashlib.sha256(response.content).hexdigest() == info['sha256']
        assert client.get('/api/models/font').json() == {'available': False, 'format': 'ONNX'}
        assert not hasattr(api.app.state.jobs.engine, '_font_download')


def test_all_compat_upload_routes_honor_android_mode(monkeypatch, tmp_path):
    created = engines(monkeypatch, tmp_path)
    with TestClient(api.app) as client:
        for method, route in [('post', '/api/predict'), ('post', '/upload'), ('post', '/'),
                              ('put', '/upload/test.png')]:
            response = getattr(client, method)(route+'?mode=android&wait=false', content=png())
            assert response.status_code in (200, 202)
            job = response.json()
            assert job['font_mode'] == 'android'
            assert finish(client, job['id'])['result']['font_mode'] == 'android'
        assert created['ios'].calls == [] and len(created['android'].calls) == 4


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
