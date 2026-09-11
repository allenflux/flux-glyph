import hashlib
import json
import zipfile

import numpy as np
import pytest

from flux_glyph.font_matcher import CompactFontBank
from flux_glyph.models import verify_bundle
from scripts import model_release


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font_directory(tmp_path,reference):
    directory=tmp_path/'font';directory.mkdir()
    font_ids=[f'face-{index}' for index in range(31)]
    families=[f'Family {index}' for index in range(10)]+['PingFang SC']
    face_family={font_id:families[index%len(families)] for index,font_id in enumerate(font_ids)}
    archive=directory/'references.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as output:
        with output.open('cjk_4E2D.npy','w') as stream:
            np.lib.format.write_array(stream,reference,allow_pickle=False)
    metadata={'schema':'flux-glyph-r13-compact-v1','algorithm_version':'r12-normalized-glyph_blur1_resize32_uint8',
              'shape':[31,3,32,32],'dtype':'uint8','font_ids':font_ids,'face_family':face_family,
              'characters':{'中':'cjk_4E2D.npy'},'archive':archive.name,'archive_sha256':sha(archive)}
    (directory/'metadata.json').write_text(json.dumps(metadata))
    (directory/'GATES.json').write_text(json.dumps({'method':'blur32','gates':{'1':None,
        '2':{'max_distance':.02,'min_margin':.005},'4':{'max_distance':.04,'min_margin':.002}}}))
    return directory


def test_font_bank_rejects_empty_reference_vector(tmp_path):
    reference=np.ones((31,3,32,32),dtype=np.uint8)
    reference[7,1]=0
    directory=font_directory(tmp_path,reference)
    with pytest.raises(ValueError,match='empty vector'):
        CompactFontBank(directory,max_characters=1)


def test_activation_validation_failure_preserves_active_file(monkeypatch,tmp_path):
    root=tmp_path/'models';release=root/'releases'/'bad';release.mkdir(parents=True)
    required=['font/metadata.json','font/GATES.json','font/references.zip','pp/onnx/paddle_ocr_det.onnx',
              'pp/onnx/paddle_ocr_rec.onnx','pp/onnx/paddle_ocr_cls.onnx','pp/charset/ppocr_keys_v1.txt',
              'pp/paddle_ocr_delivery.contract.json']
    for relative in required:
        path=release/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'placeholder')
    (release/'font/metadata.json').write_text(json.dumps({'archive':'references.zip'}))
    rows=[{'path':relative,'bytes':(release/relative).stat().st_size,'sha256':sha(release/relative)} for relative in required]
    (release/'MANIFEST.json').write_text(json.dumps({'files':rows}))
    active=root/'ACTIVE.json';active.write_text(json.dumps({'path':'releases/good'}))
    before=active.read_bytes()
    monkeypatch.setattr(model_release,'validate_runtime',lambda _: (_ for _ in ()).throw(ValueError('bad runtime')))
    with pytest.raises(ValueError,match='bad runtime'):
        model_release.activate(root,'releases/bad')
    assert active.read_bytes()==before


def test_manifest_rejects_noncanonical_archive_names(tmp_path):
    root=tmp_path/'model';path=root/'font'/'metadata.json';path.parent.mkdir(parents=True);path.write_text('{}')
    (root/'MANIFEST.json').write_text(json.dumps({'files':[{
        'path':'font/../font/metadata.json','bytes':path.stat().st_size,'sha256':sha(path)}]}))
    with pytest.raises(ValueError,match='manifest entry'):
        verify_bundle(root)
