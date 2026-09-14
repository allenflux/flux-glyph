import json
from pathlib import Path
import pytest
import numpy as np
from PIL import Image
from test_prepare_captured import dataset, write_labels, capture
from training import prepare_ios_short_train as m

def test_canonical_families_and_view_contract():
    assert [m.canonical(x) for x in ('PingFang SC','PingFang TC','PingFang HK','SF Pro','Helvetica')] == ['PingFang','PingFang','PingFang','SF Pro','Helvetica']
    assert m.VIEWS == ('native','half','three_quarters_jpeg75','jpeg75')
    assert all(v in m.RECIPES for v in m.VIEWS)

def test_test_rows_are_rejected_before_preparation(tmp_path):
    capture = tmp_path/'capture'; capture.mkdir()
    (capture/'labels.jsonl').write_text(json.dumps({'split':'test'})+'\n')
    (capture/'Scenes.json').write_text(json.dumps({'schema':'flux-glyph-capture-scenes-v1','platform':'ios','pages':[]}))
    (capture/'CAPTURE_PROTOCOL.json').write_text('{}')
    with pytest.raises(ValueError, match='TEST/CAL'):
        m.prepare(capture/'labels.jsonl', tmp_path/'out')

def test_output_overwrite_is_rejected(tmp_path):
    capture = tmp_path/'capture'; capture.mkdir(); out = tmp_path/'out'; out.mkdir()
    (capture/'labels.jsonl').write_text('')
    with pytest.raises(ValueError, match='overwrite'):
        m.prepare(capture/'labels.jsonl', out)

def valid_capture(tmp_path):
    labels, records, _ = dataset(tmp_path, texts=('青甲',), splits=('train',))
    driver = m.ROOT/'training/capture/capture_ios.py'; swift = m.ROOT/'training/capture/ios/FluxFontCapture/AppDelegate.swift'
    protocol = json.loads((tmp_path/'CAPTURE_PROTOCOL.json').read_text())
    protocol.update(capture_driver_sha256=m.sha(driver), capture_app_source_sha256=m.sha(swift),
                    capture_app_executable_sha256='a'*64)
    (tmp_path/'CAPTURE_PROTOCOL.json').write_text(json.dumps(protocol))
    (tmp_path/'frames').mkdir()
    (tmp_path/'frames'/'page-0.json').write_text(json.dumps(records[0], ensure_ascii=False))
    write_labels(labels, records)
    return labels, records

def test_real_native_fixture_contract_produces_complete_four_view_train(tmp_path):
    labels, records = valid_capture(tmp_path)
    manifest = m.prepare(labels, tmp_path/'prepared')
    assert manifest['families'] == list(m.FAMILIES) and manifest['native_regions'] == 1
    assert manifest['tiles'] == 4 and manifest['test_read'] is False
    rows = json.loads((tmp_path/'prepared'/'rows.json').read_text())['rows']
    assert {r['view'] for r in rows} == set(m.VIEWS)
    assert all(r['target'] == m.FAMILIES.index('PingFang') for r in rows)
    assert all(r['native_glyph_provenance']['glyphs'] for r in rows)

@pytest.mark.parametrize('mutation', ['invisible', 'fontwrong', 'bboxoutside'])
def test_native_glyph_proof_mutations_are_rejected(tmp_path, mutation):
    labels, records = valid_capture(tmp_path); glyph = records[0]['regions'][0]['glyphs'][0]
    if mutation == 'invisible': glyph['visible'] = False
    elif mutation == 'fontwrong': glyph['font_postscript'] = 'Other'
    else: glyph['bbox'] = [-1, 21, 20, 48]
    write_labels(labels, records)
    with pytest.raises(ValueError): m.prepare(labels, tmp_path/'prepared')

def test_frame_mismatch_and_input_mutation_are_fatal(tmp_path):
    labels, records = valid_capture(tmp_path); records[0]['page_id'] = 'changed'; write_labels(labels, records)
    with pytest.raises(ValueError): m.prepare(labels, tmp_path/'prepared')
    second = tmp_path/'second'; second.mkdir()
    labels, records = valid_capture(second); image = Path(records[0]['image'])
    with image.open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError): m.prepare(labels, tmp_path/'second-out')

def test_missing_planned_page_is_rejected(tmp_path):
    labels, records = valid_capture(tmp_path)
    scenes = json.loads((tmp_path/'Scenes.json').read_text()); scenes['pages'].append({'id':'page-1','split':'train','regions':scenes['pages'][0]['regions']})
    (tmp_path/'Scenes.json').write_text(json.dumps(scenes))
    write_labels(labels, records)
    with pytest.raises(ValueError): m.prepare(labels, tmp_path/'prepared')

def test_conflicting_tile_or_region_identity_removes_both_families():
    rows = [dict(source_id='s', region_id='r1', family='PingFang', tiles_sha256='x', region_rgb_sha256='a'),
            dict(source_id='s', region_id='r1', family='SF Pro', tiles_sha256='x', region_rgb_sha256='b'),
            dict(source_id='s', region_id='r2', family='Helvetica', tiles_sha256='z', region_rgb_sha256='c')]
    kept, tensors, conflicts = m.remove_conflicting_identities(rows, [1,2,3])
    assert conflicts == 1 and [r['family'] for r in kept] == ['Helvetica'] and tensors == [3]
