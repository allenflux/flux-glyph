"""Candidate alignment must improve raster phase errors without changing gates."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from flux_glyph.font_matcher import CompactFontBank,aligned_query_vectors,score_font


ROOT=Path(__file__).resolve().parents[1]
FIXTURES=ROOT/'tests/fixtures/font_accuracy'


@pytest.fixture(scope='module')
def bank():
    value=CompactFontBank(ROOT/'models/font')
    yield value
    value.archive.close()


def fixture_samples(case_id):
    manifest=json.loads((FIXTURES/'generated_manifest.json').read_text())
    case=next(row for row in manifest['cases'] if row['case_id']==case_id)
    with Image.open(FIXTURES/case['input_file']) as opened:
        image=opened.convert('RGB')
    return [{'character':char,'image':image.crop(box)}
            for char,box in zip(case['text'],case['expected_glyph_bboxes'])]


@pytest.mark.parametrize(('case_id','expected'),[
    ('misans_regular--white_png_26','MiSans'),
    ('oppo_regular--blue_jpeg82_36','OPPO Sans'),
])
def test_frozen_android_phase_regressions(bank,case_id,expected):
    samples=fixture_samples(case_id)
    # These oracle boxes in the frozen fixtures expose the matcher error itself.
    assert bank.match(samples)['family_candidate']!=expected
    scored=score_font(bank,samples,complete=True)
    assert scored['family']==expected
    assert all(row['family_candidate']==expected for row in scored['candidate_views'].values())
    assert scored['candidate_method']=='blur32_half_pixel_alignment'
    assert not scored['accepted_pingfang']
    assert score_font(bank,samples,complete=False)['family'] is None


def test_alignment_does_not_discard_strokes_outside_original_canvas():
    z=np.zeros((32,32));z[0,0]=1
    queries=aligned_query_vectors(z.reshape(-1))
    assert queries.shape==(9,1024)
    assert np.isfinite(queries).all()
    # At (-.5,-.5), three quarters of the stroke leave the original canvas.
    # Keeping their energy yields an inner vector of norm .5, not 1.0.
    assert np.linalg.norm(queries[0])==pytest.approx(.5)
    np.testing.assert_array_equal(queries[4],z.reshape(-1))
    assert np.count_nonzero(queries[:,31*32:])==0


def test_aligned_distance_cannot_pass_frozen_pingfang_gate():
    def match(samples,*,align=False):
        distance=0.001 if align else 0.05
        return {'status':'ok','family_candidate':'PingFang SC',
                'leading_family_ties':['PingFang SC'],
                'family_scores':{'PingFang SC':distance,'MiSans':0.2},
                'margin_to_next_family':0.2-distance}
    fake=SimpleNamespace(match=match,gates={'4':{'max_distance':0.04,'min_margin':0.002}})
    image=Image.new('RGB',(20,20),'white')
    scored=score_font(fake,[{'character':c,'image':image} for c in '账单详情'],True)
    assert not scored['accepted_pingfang']
    assert scored['views']['identity']['family_scores']['PingFang SC']==0.05
    assert scored['candidate_views']['identity']['family_scores']['PingFang SC']==0.001


def test_supported_pingfang_keeps_original_distance_and_decision(bank):
    scored=score_font(bank,fixture_samples('pingfang_21.0d1e1_regular--white_png_26'),True)
    assert scored['accepted_pingfang']
    assert scored['candidate_method']=='blur32'
    assert scored['candidate_views']==scored['views']
