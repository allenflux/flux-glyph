"""Every runtime Latin dependency must belong to the verified bundle manifest."""
import json

import pytest

from flux_glyph.models import file_sha,verify_bundle


def bundle(tmp_path,*,latin=True,omitted=None):
    files={'font/metadata.json':json.dumps({'archive':'references.zip'}).encode(),
           'font/GATES.json':b'{}','font/references.zip':b'references',
           'pp/onnx/paddle_ocr_det.onnx':b'detector',
           'pp/onnx/paddle_ocr_rec.onnx':b'recognizer',
           'pp/onnx/paddle_ocr_cls.onnx':b'orientation',
           'pp/charset/ppocr_keys_v1.txt':b'characters',
           'pp/paddle_ocr_delivery.contract.json':b'{}'}
    if latin:
        files.update({'latin/metadata.json':json.dumps({'archive':'references.zip',
                      'gates':{'path':'GATES.json'}}).encode(),
                      'latin/GATES.json':b'{}','latin/references.zip':b'latin references'})
    rows=[]
    for relative,content in files.items():
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        if relative!=omitted:
            rows.append({'path':relative,'bytes':len(content),'sha256':file_sha(path)})
    (tmp_path/'MANIFEST.json').write_text(json.dumps({'files':rows}))
    return tmp_path


def test_legacy_manifest_does_not_require_latin_assets(tmp_path):
    root=bundle(tmp_path,latin=False)
    assert verify_bundle(root)['files']


@pytest.mark.parametrize('omitted',['latin/metadata.json','latin/GATES.json','latin/references.zip'])
def test_latin_assets_on_disk_cannot_escape_manifest_hash_closure(tmp_path,omitted):
    root=bundle(tmp_path,omitted=omitted)
    assert (root/omitted).is_file()
    with pytest.raises(ValueError,match='lacks'):
        verify_bundle(root)


def test_latin_gate_tampering_is_rejected_before_loading(tmp_path):
    root=bundle(tmp_path)
    verify_bundle(root)
    (root/'latin/GATES.json').write_text('{"gates": "changed"}')
    with pytest.raises(ValueError,match='checksum mismatch: latin/GATES.json'):
        verify_bundle(root)
