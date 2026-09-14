"""Tests for the one-field post-DEV spatial-refinement delivery wrapper."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import prepare_spatial_refinement_delivery as delivery


def metadata(model_sha, version=delivery.upstream.VERSION):
    return {'schema':'flux-glyph-unified-region-font-v1',
        'algorithm':'unified-region-cnn64x256-v1','font_mode':'unified',
        'data_kind':'native_mobile_screenshots','network_architecture':
            'region-cnn64x256-unified-spatial4x16-v1',
        'model':{'path':'model.onnx','sha256':model_sha},
        'families':[*(f'Known {i}' for i in range(24)),'__unknown__'],'temperature':1.,
        'gates':{'min_score':.7,'min_margin':.01,'min_patch_agreement':2/3},
        'max_size_relative_spread':.2,'font_label_groups':{},'font_sources':{},
        'version':version,'release_tier':'experimental','stable_validation_passed':False,'test_passed':False,
        'pre_development':True,
        'validation':{'promotion_allowed':True,'development_holdout_evaluated':True,
            'development_checks':36,'stable_validation_passed':False,'test_passed':False,
            'test_read':False,'blind_test_performed':False},
        'training':{'data_manifest_sha256':'a'*64,'ocr_text_used':False,'model_count':1,
                    'platform_routing':False,'score_merging':False}}


def upstream_stub(tmp_path, *, metadata_update=None, bad_model_binding=False, failure=None):
    source = tmp_path/'original-input.json'; source.write_text('{}\n')
    def prepare(run, region, development_report, staging, development_freeze, version):
        if failure:
            raise ValueError(failure)
        staging=Path(staging);staging.mkdir()
        model=staging/'model.onnx';model.write_bytes(b'folded-spatial-cnn')
        value=metadata(hashlib.sha256(model.read_bytes()).hexdigest(),version)
        if metadata_update:
            metadata_update(value)
        meta=staging/'metadata.json';meta.write_text(json.dumps(value,indent=2)+'\n')
        evidence={'schema':delivery.upstream.EVIDENCE_SCHEMA,'version':version,
            'release_tier':'preview','promotion_allowed':True,
            'stable_validation_passed':False,'test_passed':False,'test_read':False,
            'blind_test_performed':False,
            'input_bindings':{str(source):delivery.sha(source)},
            'output_bindings':{str(model):('0'*64 if bad_model_binding else delivery.sha(model)),
                               str(meta):delivery.sha(meta)}}
        (staging/'RELEASE_EVIDENCE.json').write_bytes(delivery.encoded(evidence))
        return evidence
    return prepare


def invoke(monkeypatch,tmp_path,**stub_options):
    monkeypatch.setattr(delivery.upstream,'prepare_release',upstream_stub(tmp_path,**stub_options))
    paths={name:tmp_path/name for name in ('run','region','development.json','staging','final')}
    return paths, lambda:delivery.prepare_delivery(paths['run'],paths['region'],
        paths['development.json'],paths['staging'],paths['final'])


def test_wrapper_changes_only_pre_development_and_preserves_model(monkeypatch,tmp_path):
    paths,call=invoke(monkeypatch,tmp_path);report=call()
    staging,final=paths['staging'],paths['final']
    old=json.loads((staging/'metadata.json').read_text());new=json.loads((final/'metadata.json').read_text())
    restored=deepcopy(new);restored['pre_development']=True
    assert restored==old and old['pre_development'] is True and new['pre_development'] is False
    assert new['validation']['development_holdout_evaluated'] is True
    assert (final/'model.onnx').read_bytes()==(staging/'model.onnx').read_bytes()
    assert set(p.name for p in final.iterdir())=={'model.onnx','metadata.json','RELEASE_EVIDENCE.json'}
    assert report['schema']==delivery.SCHEMA and report['metadata_change']=={
        'field':'pre_development','before':True,'after':False,'only_metadata_change':True,
        'reason':'The frozen dual-DEV release validation completed successfully.'}
    assert report['model_bytes_unchanged'] is True


def test_evidence_binds_staging_upstream_inputs_sources_and_final_outputs(monkeypatch,tmp_path):
    paths,call=invoke(monkeypatch,tmp_path);report=call();final=paths['final']
    saved=json.loads((final/'RELEASE_EVIDENCE.json').read_text())
    assert saved==report
    assert set(Path(p).name for p in report['staging_bindings'])=={
        'RELEASE_EVIDENCE.json','model.onnx','metadata.json'}
    assert set(Path(p).name for p in report['source_bindings'])=={
        'prepare_spatial_refinement_release.py','prepare_spatial_refinement_delivery.py'}
    for bindings in (report['staging_bindings'],report['upstream_input_bindings'],
                     report['source_bindings'],report['output_bindings']):
        for path,digest in bindings.items():
            assert delivery.sha(path)==digest


@pytest.mark.parametrize('existing',('staging','final'))
def test_existing_staging_or_final_is_never_overwritten(monkeypatch,tmp_path,existing):
    paths,call=invoke(monkeypatch,tmp_path);paths[existing].mkdir();marker=paths[existing]/'keep';marker.write_text('x')
    with pytest.raises(ValueError):call()
    assert marker.read_text()=='x'
    assert not paths['final' if existing=='staging' else 'staging'].exists()


def test_upstream_validation_failure_produces_no_staging_or_final(monkeypatch,tmp_path):
    paths,call=invoke(monkeypatch,tmp_path,failure='CAL or DEV evidence failed')
    with pytest.raises(ValueError,match='CAL or DEV'):call()
    assert not paths['staging'].exists() and not paths['final'].exists()


@pytest.mark.parametrize('stub_options',[
    {'metadata_update':lambda value:value.update(pre_development=False)},
    {'metadata_update':lambda value:value['validation'].update(development_holdout_evaluated=False)},
    {'bad_model_binding':True},
])
def test_invalid_upstream_staging_never_produces_final(monkeypatch,tmp_path,stub_options):
    paths,call=invoke(monkeypatch,tmp_path,**stub_options)
    with pytest.raises(ValueError):call()
    assert paths['staging'].is_dir() and not paths['final'].exists()


def test_staging_and_final_must_be_independent(monkeypatch,tmp_path):
    monkeypatch.setattr(delivery.upstream,'prepare_release',lambda *args:pytest.fail('must not call'))
    staging=tmp_path/'artifact'; final=staging/'final'
    with pytest.raises(ValueError,match='independent'):
        delivery.prepare_delivery(tmp_path/'run',tmp_path/'region',tmp_path/'dev',staging,final)
