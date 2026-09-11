"""Queue and access-control boundaries, using a blocked worker instead of ML."""
import io
import os
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from flux_glyph import api


def png():
    output=io.BytesIO();Image.new('RGB',(32,32),'white').save(output,'PNG')
    return output.getvalue()


def test_authenticated_upload_queue_and_private_assets(monkeypatch,tmp_path):
    release=threading.Event();started=threading.Event()
    class SlowEngine:
        version='test-only'
        def __init__(self,*args,**kwargs):pass
        def run(self,source,output,identifier,progress):
            started.set();assert release.wait(10)
            Image.new('RGB',(32,32),'white').save(Path(output)/'original.png')
            return {'id':identifier,'width':32,'height':32,'regions':[],
                    'summary':{},'timing_seconds':{},'model_version':self.version,
                    'source_sha256':'test','font_scope':'test',
                    'font_identity_verified':False,'device_inference_performed':False}
    monkeypatch.setattr(api,'FontPipeline',SlowEngine)
    monkeypatch.setattr(api,'DATA',tmp_path)
    monkeypatch.setattr(api,'MAX_QUEUE',1)
    monkeypatch.setattr(api,'TOKEN','test-secret')
    try:
        with TestClient(api.app) as client:
            assert client.get('/api/health').json()['authentication_required'] is True
            assert client.post('/api/jobs',content=png()).status_code==401
            assert client.post('/auth',json={'token':'错误令牌'}).status_code==401
            assert client.post('/auth',json={'token':['wrong']}).status_code==401
            login=client.post('/auth',json={'token':'test-secret'})
            assert login.status_code==200
            assert 'HttpOnly' in login.headers['set-cookie'] and 'SameSite=strict' in login.headers['set-cookie']
            assert client.post('/api/jobs',content=b'bad',headers={'Content-Type':'multipart/form-data; boundary=x'}).status_code==400
            first=client.post('/api/jobs',content=png());assert first.status_code==202
            assert started.wait(2)
            second=client.post('/api/jobs',content=png());assert second.status_code==202
            assert client.post('/api/jobs',content=png()).status_code==429
            assert client.get('/api/jobs/'+first.json()['id']+'/json').status_code==409
            # A result URL alone never grants access to a private image.
            client.cookies.clear()
            assert client.get('/api/jobs/'+first.json()['id']).status_code==401
            assert client.get('/assets/'+first.json()['id']+'/original.png').status_code==401
            bearer={'Authorization':'Bearer test-secret'}
            assert client.get('/api/jobs/'+first.json()['id'],headers=bearer).status_code==200
            release.set()
            for _ in range(100):
                done=client.get('/api/jobs/'+second.json()['id'],headers=bearer).json()
                if done['status']=='complete':break
                time.sleep(.01)
            assert done['status']=='complete'
            assert client.get('/assets/'+first.json()['id']+'/original.png',headers=bearer).status_code==200
            assert client.get('/assets/'+first.json()['id']+'/%2E%2E%2F%2E%2E%2Fmodels%2FMANIFEST.json',headers=bearer).status_code==404
            # OpenAPI and docs work without fetching third-party assets.
            assert client.get('/docs').status_code==200
            assert client.get('/openapi.json').status_code==200
    finally:
        release.set()


def test_retention_includes_orphans_and_keeps_failed_deletions(monkeypatch,tmp_path):
    monkeypatch.setattr(api,'DATA',tmp_path)
    monkeypatch.setattr(api,'TTL_HOURS',1)
    old=time.time()-7200
    orphan='1'*32;active='2'*32;retry='3'*32
    for identifier in (orphan,active,retry):
        directory=tmp_path/identifier;directory.mkdir()
        (directory/'uploaded-image').write_bytes(b'private test data')
        os.utime(directory,(old,old))
    jobs=api.Jobs.__new__(api.Jobs);jobs.lock=threading.RLock()
    jobs.active={active};jobs.jobs={active:{'created_at':old},retry:{'created_at':old}}
    original_delete=api.shutil.rmtree
    def transient_failure(path):
        if path.name==retry:raise PermissionError('simulated transient filesystem failure')
        original_delete(path)
    with monkeypatch.context() as patch:
        patch.setattr(api.shutil,'rmtree',transient_failure)
        jobs.cleanup()
    assert not (tmp_path/orphan).exists()
    assert (tmp_path/active).is_dir()
    assert (tmp_path/retry).is_dir() and retry in jobs.jobs
    jobs.cleanup()
    assert not (tmp_path/retry).exists() and retry not in jobs.jobs
