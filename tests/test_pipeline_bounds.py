"""Large overlapping detector regions must not all be materialized together."""
from types import SimpleNamespace
from PIL import Image
from flux_glyph.pipeline import FontPipeline


def test_large_overlapping_regions_are_bounded_and_order_preserved(tmp_path):
    boxes=[{'source_bbox':[0,0,1000,1500],'quad':[[0,0],[1000,0],[1000,1500],[0,1500]],'score':.99} for _ in range(8)]
    areas=[]
    class Reader:
        def read(self,crops):
            areas.append(sum(x.width*x.height for x in crops))
            return [{'text':'123','confidence':.99} for _ in crops]
    pipeline=FontPipeline.__new__(FontPipeline)
    pipeline.detector=SimpleNamespace(detect=lambda image:boxes)
    pipeline.reader=Reader();pipeline.bank=SimpleNamespace(cache_bytes=0)
    pipeline.max_regions=200;pipeline.version='test-only'
    source=tmp_path/'input.png';Image.new('RGB',(1000,1500),'white').save(source)
    result=pipeline.run(source,tmp_path/'result','bounded')
    assert max(areas)<=4_000_000
    assert sum(areas)==8*1000*1500
    assert [r['id'] for r in result['regions']]==[f'R{i:03d}' for i in range(1,9)]
    assert len(result['regions'])==8
    assert all(r['text']=='123' for r in result['regions'])
