"""Download kits are bounded public model artifacts, never upload directories."""
import ast
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

from fastapi.testclient import TestClient
import pytest

from flux_glyph import api
from flux_glyph.model_download import PREDICT, font_kit


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fake_engine(directory):
    directory.mkdir(parents=True, exist_ok=True)
    weights = b'unit-test-only ONNX stand-in; no inference is performed'
    (directory / 'model.onnx').write_bytes(weights)
    meta = {'schema': 'flux-glyph-region-font-v1', 'algorithm': 'region-cnn64x256-v1',
            'families': ['PingFang SC', 'SF Pro'], 'model': {'path': 'model.onnx', 'sha256': sha(weights)},
            'temperature': 1.2, 'gates': {'min_score': .9, 'min_margin': .2, 'min_patch_agreement': .75},
            'max_size_relative_spread': .2,
            'private_training_path': '/private/training/not-for-download',
            'source_screenshot_paths': ['/private/upload.png']}
    model = SimpleNamespace(directory=directory, meta=meta, families=meta['families'])
    return SimpleNamespace(version='r16-test-region', region_neural=model)


def zip_files(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.testzip() is None
        return {name: archive.read(name) for name in archive.namelist()}


def test_zip_contains_only_explicit_model_runtime_and_usage_files(tmp_path):
    engine = fake_engine(tmp_path / 'model')
    secrets = {'upload.png': b'private screenshot payload', '.env': b'FLUX_API_TOKEN=do-not-copy',
               'training-data.json': b'private training annotations', 'extra.onnx': b'unrelated model'}
    for name, content in secrets.items():
        (engine.region_neural.directory / name).write_bytes(content)
    info, payload = font_kit(engine)
    files = zip_files(payload)
    assert set(files) == {'model.onnx', 'metadata.json', 'inference.py', 'text_style.py',
                          'predict.py', 'README.md', 'requirements.txt', 'SHA256.json'}
    assert all(not name.startswith('/') and '..' not in Path(name).parts for name in files)
    assert not any(secret in content for content in files.values() for secret in secrets.values())
    metadata = json.loads(files['metadata.json'])
    assert 'private_training_path' not in metadata and 'source_screenshot_paths' not in metadata
    assert metadata['model_version'] == engine.version
    assert metadata['model']['sha256'] == sha(files['model.onnx'])
    assert metadata['families'] == engine.region_neural.families
    assert metadata['gates'] == engine.region_neural.meta['gates']
    assert info['bytes'] == len(payload) and info['sha256'] == sha(payload)
    assert info['input_shape'] == [None, 1, 64, 256] and info['ocr_required'] is False
    assert info['download_url'] == '/api/models/font/download'
    assert info['usage'] == {'install': 'pip install -r requirements.txt', 'predict': 'python predict.py text-region.png'}


def test_every_download_member_is_bound_by_sha_and_contains_no_ocr_runtime(tmp_path):
    _, payload = font_kit(fake_engine(tmp_path / 'model'))
    files = zip_files(payload)
    checksums = json.loads(files.pop('SHA256.json'))
    assert checksums == {name: sha(content) for name, content in files.items()}
    assert files['inference.py'] == (Path(api.__file__).parent / 'region_font.py').read_bytes()
    assert files['text_style.py'] == (Path(api.__file__).parent / 'text_style.py').read_bytes()
    assert b'ppocr' not in files['inference.py'] and b'PPReader' not in files['predict.py']
    for filename in ('inference.py', 'text_style.py', 'predict.py'):
        compile(files[filename], filename, 'exec')
    tree = ast.parse(files['predict.py'])
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imports == {'pathlib', 'PIL', 'inference', 'text_style'}
    assert 'ocr_performed' in PREDICT and 'False' in PREDICT
    requirements = files['requirements.txt'].decode().splitlines()
    assert len(requirements) == 3 and all(name in '\n'.join(requirements).lower() for name in ('numpy==', 'pillow==', 'onnxruntime=='))
    assert b'python predict.py text-region.png' in files['README.md']


def test_model_bytes_changed_before_first_download_are_rejected(tmp_path):
    engine = fake_engine(tmp_path / 'model')
    (engine.region_neural.directory / 'model.onnx').write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='Model changed'):
        font_kit(engine)
    assert not hasattr(engine, '_font_download')


def test_wrong_declared_model_sha_is_rejected(tmp_path):
    engine = fake_engine(tmp_path / 'model')
    engine.region_neural.meta['model']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Model changed'):
        font_kit(engine)


def test_repeat_download_is_deterministic_and_cached_content_is_immutable(tmp_path):
    engine = fake_engine(tmp_path / 'model')
    first = font_kit(engine)
    assert font_kit(engine) is first
    # The live serving instance pins its first verified kit. It never reads a
    # later unverified model into that existing ZIP cache.
    (engine.region_neural.directory / 'model.onnx').write_bytes(b'changed on disk')
    assert font_kit(engine)[1] == first[1]
    independent = fake_engine(tmp_path / 'other-model')
    assert font_kit(independent)[1] == first[1]


@pytest.mark.parametrize('engine', [SimpleNamespace(version='legacy'), SimpleNamespace(version='legacy', region_neural=None)])
def test_legacy_non_region_models_have_no_download(engine):
    assert font_kit(engine) is None


def client_engine(monkeypatch, tmp_path, *, region=True, token=''):
    engine = fake_engine(tmp_path / 'model') if region else SimpleNamespace(version='legacy', region_neural=None)
    monkeypatch.setattr(api, 'FontPipeline', lambda *args, **kwargs: engine)
    monkeypatch.setattr(api, 'DATA', tmp_path / 'uploads')
    monkeypatch.setattr(api, 'TOKEN', token)
    return engine


def test_api_model_information_and_zip_match_exact_bytes(monkeypatch, tmp_path):
    engine = client_engine(monkeypatch, tmp_path)
    with TestClient(api.app) as client:
        info = client.get('/api/models/font')
        assert info.status_code == 200 and info.headers['Cache-Control'] == 'no-store'
        model = info.json()
        response = client.get(model['download_url'])
        assert response.status_code == 200 and response.headers['Content-Type'] == 'application/zip'
        assert response.headers['Content-Disposition'] == 'attachment; filename="flux-glyph-font-onnx.zip"'
        assert response.headers['Cache-Control'] == 'no-store'
        assert response.headers['X-Content-Type-Options'] == 'nosniff'
        assert response.headers['X-Model-SHA256'] == model['sha256'] == sha(response.content)
        assert model['bytes'] == len(response.content)
        assert zip_files(response.content)['model.onnx'] == (engine.region_neural.directory / 'model.onnx').read_bytes()


def test_legacy_api_reports_unavailable_and_download_404(monkeypatch, tmp_path):
    client_engine(monkeypatch, tmp_path, region=False)
    with TestClient(api.app) as client:
        assert client.get('/api/models/font').json() == {'available': False, 'format': 'ONNX'}
        assert client.get('/api/models/font/download').status_code == 404


def test_both_download_routes_require_same_bearer_or_cookie_as_inference(monkeypatch, tmp_path):
    engine = client_engine(monkeypatch, tmp_path, token='font-download-test-secret')
    routes = ('/api/models/font', '/api/models/font/download')
    with TestClient(api.app) as client:
        assert client.get('/api/health').status_code == 200
        for route in routes:
            assert client.get(route).status_code == 401
            assert client.get(route, headers={'Authorization': 'Bearer wrong'}).status_code == 401
            assert client.get(route + '?token=font-download-test-secret').status_code == 401
        assert not hasattr(engine, '_font_download')  # Auth runs before creating the kit.
        for route in routes:
            assert client.get(route, headers={'Authorization': 'Bearer font-download-test-secret'}).status_code == 200
        login = client.post('/auth', json={'token': 'font-download-test-secret'})
        assert login.status_code == 200 and 'HttpOnly' in login.headers['set-cookie']
        for route in routes:
            assert client.get(route).status_code == 200
        assert 'token' not in client.get('/api/models/font').json()['download_url']
        client.cookies.clear()
        assert all(client.get(route).status_code == 401 for route in routes)


def test_api_does_not_send_changed_model_as_a_successful_zip(monkeypatch, tmp_path):
    engine = client_engine(monkeypatch, tmp_path)
    (engine.region_neural.directory / 'model.onnx').write_bytes(b'invalid modified weights')
    with TestClient(api.app, raise_server_exceptions=False) as client:
        response = client.get('/api/models/font/download')
        assert response.status_code == 500 and response.headers.get('Content-Type') != 'application/zip'
        assert b'invalid modified weights' not in response.content


def fake_rejection_engine(directory):
    engine=fake_engine(directory)
    meta=engine.region_neural.meta
    weights=b'independent known/unknown ONNX stand-in'
    (directory/'rejection.onnx').write_bytes(weights)
    meta['algorithm']='region-cnn64x256-rejection-v2'
    meta['rejection']={'schema':'flux-glyph-region-rejection-v1','algorithm':'region-known-unknown-cnn64x256-v1',
        'model':{'path':'rejection.onnx','sha256':sha(weights)},'labels':['unknown','known'],
        'known_families':meta['families'].copy(),'base_model_sha256':meta['model']['sha256'],
        'aggregation':'mean_softmax_known_probability','temperature':1.25,'min_known_score':.8}
    return engine


def test_rejection_kit_contains_complete_independent_model_metadata_and_runtime(tmp_path):
    engine=fake_rejection_engine(tmp_path/'model')
    original=json.loads(json.dumps(engine.region_neural.meta['rejection']))
    engine.region_neural.meta['rejection']['training_source']='/private/unknown-captures'
    engine.region_neural.meta['rejection']['model']['private_path']='/private/checkpoint'
    info,payload=font_kit(engine)
    files=zip_files(payload)
    assert len(files)==9 and files['rejection.onnx']==(engine.region_neural.directory/'rejection.onnx').read_bytes()
    metadata=json.loads(files['metadata.json'])
    assert metadata['rejection']==original
    assert metadata['algorithm']=='region-cnn64x256-rejection-v2'
    assert info['unknown_font_rejection'] is True
    checksums=json.loads(files['SHA256.json'])
    assert checksums=={name:sha(content) for name,content in files.items() if name!='SHA256.json'}
    assert all(b'/private/unknown-captures' not in content and b'/private/checkpoint' not in content for content in files.values())
    assert b'known_logits' in files['inference.py'] and '两个 ONNX'.encode() in files['README.md']
    assert len(files['requirements.txt'].decode().splitlines())==3


@pytest.mark.parametrize('problem',['missing','changed','wrong_sha','v1_silent_ignore'])
def test_download_cannot_drop_or_replace_the_rejection_gate(tmp_path,problem):
    engine=fake_rejection_engine(tmp_path/'model')
    if problem=='missing':(engine.region_neural.directory/'rejection.onnx').unlink()
    elif problem=='changed':(engine.region_neural.directory/'rejection.onnx').write_bytes(b'wrong gate')
    elif problem=='wrong_sha':engine.region_neural.meta['rejection']['model']['sha256']='0'*64
    else:engine.region_neural.meta['algorithm']='region-cnn64x256-v1'
    with pytest.raises(ValueError):font_kit(engine)
    assert not hasattr(engine,'_font_download')


def test_verified_rejection_download_cache_pins_both_models(tmp_path):
    engine=fake_rejection_engine(tmp_path/'model')
    first=font_kit(engine)
    (engine.region_neural.directory/'rejection.onnx').write_bytes(b'later unverified model')
    assert font_kit(engine) is first
    assert zip_files(first[1])['rejection.onnx']!=b'later unverified model'


def test_authenticated_rejection_download_serves_the_full_two_model_kit(monkeypatch,tmp_path):
    engine=fake_rejection_engine(tmp_path/'model')
    monkeypatch.setattr(api,'FontPipeline',lambda *a,**k:engine)
    monkeypatch.setattr(api,'DATA',tmp_path/'uploads')
    monkeypatch.setattr(api,'TOKEN','rejection-test-token')
    with TestClient(api.app) as client:
        assert client.get('/api/models/font/download').status_code==401
        headers={'Authorization':'Bearer rejection-test-token'}
        info=client.get('/api/models/font',headers=headers).json()
        response=client.get(info['download_url'],headers=headers)
        assert info['available'] is True and info['unknown_font_rejection'] is True
        assert response.status_code==200 and info['sha256']==sha(response.content)
        assert {'model.onnx','rejection.onnx'}.issubset(zip_files(response.content))
