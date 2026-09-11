"""Queue positions and progress represent actual completed server operations."""
import io
import threading
import time
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
from flux_glyph import api
from flux_glyph.pipeline import FontPipeline


def test_fifo_positions_and_completion_progress(monkeypatch,tmp_path):
    release=threading.Event();started=threading.Event();order=[]
    class Engine:
        version='test'
        def __init__(self,*args,**kwargs):pass
        def run(self,source,output,identifier,progress):
            order.append(identifier)
            progress({'stage':'正在识读文字 2/4','progress':{'stage_code':'recognizing','percent':25,'current':2,'total':4}})
            started.set()
            assert release.wait(5)
            return {'id':identifier,'width':16,'height':16,'regions':[],
                    'summary':{},'timing_seconds':{},'model_version':self.version,
                    'source_sha256':'test','font_scope':'test','font_identity_verified':False,'device_inference_performed':False}
    monkeypatch.setattr(api,'FontPipeline',Engine);monkeypatch.setattr(api,'DATA',tmp_path)
    monkeypatch.setattr(api,'MAX_QUEUE',2);monkeypatch.setattr(api,'TOKEN','')
    raw=io.BytesIO();Image.new('RGB',(16,16),'white').save(raw,'PNG')
    try:
        with TestClient(api.app) as client:
            first=client.post('/api/jobs',content=raw.getvalue()).json()
            assert started.wait(2)
            live=client.get('/api/jobs/'+first['id']).json()
            assert live['status']=='running' and live['queue_position']==0
            assert live['progress']=={'stage_code':'recognizing','percent':25,'current':2,'total':4}
            second=client.post('/api/jobs',content=raw.getvalue()).json()
            third=client.post('/api/jobs',content=raw.getvalue()).json()
            assert (second['queue_position'],second['queue_ahead'])==(1,1)
            assert (third['queue_position'],third['queue_ahead'])==(2,2)
            assert third['progress']['percent']==0
            health=client.get('/api/health').json()
            assert health['running_jobs']==1 and health['waiting_jobs']==2
            full=client.post('/api/jobs',content=raw.getvalue())
            assert full.status_code==429 and full.headers['Retry-After']=='2'
            release.set()
            for _ in range(100):
                done=client.get('/api/jobs/'+third['id']).json()
                if done['status']=='complete':break
                time.sleep(.01)
            assert order==[first['id'],second['id'],third['id']]
            assert done['progress']['percent']==100 and done['progress']['stage_code']=='complete'
            assert done['queue_position'] is None and done['queue_ahead']==0
            assert 'result' in done
    finally:release.set()


def test_real_pipeline_progress_has_counts_and_never_finishes_before_api(tmp_path):
    events=[];pipeline=FontPipeline()
    result=pipeline.run(Path(__file__).parent/'fixtures/ui_title_billing_details.png',tmp_path/'out','progress',events.append)
    assert result['regions'][0]['font']['family']=='PingFang SC'
    phases=[e['progress']['stage_code'] for e in events]
    assert phases[0:2]==['preparing','detecting']
    assert phases[-2:]==['annotating','finalizing']
    fractions=[e['progress']['percent'] for e in events if e['progress']['percent'] is not None]
    assert fractions==sorted(fractions) and max(fractions)<100
    reading=[e['progress'] for e in events if e['progress']['stage_code']=='recognizing']
    assert reading[0]['current']==0
    assert reading[-1]['current']==reading[-1]['total']==len(result['regions'])


def test_failed_job_releases_queue_and_restart_does_not_claim_it_is_running(monkeypatch,tmp_path):
    import json
    crashed='e'*32
    old=tmp_path/crashed;old.mkdir()
    (old/'job.json').write_text(json.dumps({'id':crashed,'status':'running','created_at':time.time(),
        'progress':{'stage_code':'matching','percent':40,'current':1,'total':3}}))
    release=threading.Event();started=threading.Event();calls=[]
    class Engine:
        version='test'
        def __init__(self,*args,**kwargs):pass
        def run(self,source,output,identifier,progress):
            calls.append(identifier)
            if len(calls)==1:
                progress({'stage':'检测中','progress':{'stage_code':'detecting','percent':None,'current':None,'total':None}})
                started.set();assert release.wait(5)
                raise ValueError('simulated model failure')
            return {'id':identifier,'width':16,'height':16,'regions':[],
                    'summary':{},'timing_seconds':{},'model_version':self.version,
                    'source_sha256':'test','font_scope':'test','font_identity_verified':False,'device_inference_performed':False}
    monkeypatch.setattr(api,'FontPipeline',Engine);monkeypatch.setattr(api,'DATA',tmp_path)
    monkeypatch.setattr(api,'MAX_QUEUE',1);monkeypatch.setattr(api,'TOKEN','')
    raw=io.BytesIO();Image.new('RGB',(16,16),'white').save(raw,'PNG')
    try:
        with TestClient(api.app) as client:
            interrupted=client.get('/api/jobs/'+crashed).json()
            assert interrupted['status']=='error' and interrupted['error_code']=='service_restarted'
            assert interrupted['queue_position'] is None and interrupted['queue_total']==0
            assert interrupted['progress']['percent'] is None
            first=client.post('/api/jobs',content=raw.getvalue()).json()
            assert started.wait(2)
            second=client.post('/api/jobs',content=raw.getvalue()).json()
            assert second['queue_ahead']==1
            release.set()
            for _ in range(100):
                done=client.get('/api/jobs/'+second['id']).json()
                if done['status']=='complete':break
                time.sleep(.01)
            failed=client.get('/api/jobs/'+first['id']).json()
            assert failed['status']=='error' and failed['error_code']=='inference_failed'
            assert failed['progress']['stage_code']=='error' and failed['progress']['percent']!=100
            assert 'result' not in failed and failed['queue_position'] is None
            assert done['status']=='complete' and done['progress']['percent']==100
            assert calls==[first['id'],second['id']]
            assert client.post('/api/jobs',content=raw.getvalue()).status_code==202
    finally:release.set()
