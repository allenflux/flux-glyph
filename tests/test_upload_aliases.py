"""Compatibility upload routes reuse the normal validated inference queue."""
import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from flux_glyph import api


def png():
    output=io.BytesIO();Image.new('RGB',(16,16),'white').save(output,'PNG')
    return output.getvalue()


class StubEngine:
    version='upload-alias-test'
    def __init__(self,*args,**kwargs):pass
    def run(self,source,output,identifier,progress):
        return {'id':identifier,'width':16,'height':16,'regions':[],
                'summary':{},'timing_seconds':{},'model_version':self.version,
                'source_sha256':'test','font_scope':'test',
                'font_identity_verified':False,'device_inference_performed':False}


def test_upload_aliases_keep_existing_get_root_and_job_semantics(monkeypatch,tmp_path):
    monkeypatch.setattr(api,'FontPipeline',StubEngine)
    monkeypatch.setattr(api,'DATA',tmp_path)
    monkeypatch.setattr(api,'TOKEN','')
    with TestClient(api.app) as client:
        assert client.get('/').status_code==200

        root=client.post('/?wait=false',files={'file':('root.png',png(),'image/png')})
        assert root.status_code==202
        assert root.json()['filename']=='root.png'

        form=client.post('/upload',files={'file':('form.png',png(),'image/png')})
        assert form.status_code==200
        assert form.json()['model_version']=='upload-alias-test'

        direct=client.put('/upload/folder/direct.png?wait=false',content=png(),headers={
            'X-Filename':'ignored.png'})
        assert direct.status_code==202
        job=direct.json()
        assert job['filename']=='folder/direct.png'
        directory=tmp_path/job['id']
        assert directory.parent==tmp_path and directory.name==job['id']
        assert (directory/'uploaded-image').read_bytes()==png()
        assert json.loads((directory/'job.json').read_text())['filename']=='folder/direct.png'
        assert not (tmp_path/'folder').exists()

        invalid=client.put('/upload/not-an-image.log',content=b'plain text')
        assert invalid.status_code==400


def test_upload_aliases_require_the_same_bearer_token(monkeypatch,tmp_path):
    monkeypatch.setattr(api,'FontPipeline',StubEngine)
    monkeypatch.setattr(api,'DATA',tmp_path)
    monkeypatch.setattr(api,'TOKEN','secret')
    with TestClient(api.app) as client:
        requests=(
            client.post('/?wait=false',files={'file':('root.png',png(),'image/png')}),
            client.post('/upload?wait=false',files={'file':('form.png',png(),'image/png')}),
            client.put('/upload/direct.png?wait=false',content=png()),
        )
        assert [response.status_code for response in requests]==[401,401,401]
        bearer={'Authorization':'Bearer secret'}
        accepted=client.put('/upload/direct.png?wait=false',content=png(),headers=bearer)
        assert accepted.status_code==202
        assert accepted.headers['Cache-Control']=='no-store'
