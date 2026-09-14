"""Strict loader and face-balanced sampler for new native iOS short TRAIN."""
from __future__ import annotations
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from prepare_unified_regions import FAMILIES
from train_regions import require, sha

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT/'artifacts/ios-native-short-train-v1/prepared'
MANIFEST_SHA = '7b2058625495dea7e56242a27d7feaa7a245eaad2915c8674edc6fcf9ef7d3af'
IOS_FAMILIES = ('PingFang', 'SF Pro', 'Helvetica')
VIEWS = ('native', 'half', 'three_quarters_jpeg75', 'jpeg75')
SAMPLING = {'schema':'flux-glyph-ios-native-short-sampling-v1', 'batch_rows':24,
    'families':list(IOS_FAMILIES), 'glyph_counts':[1,2,3,4], 'rows_per_family_length':2,
    'view_weights':dict(zip(VIEWS,(.4,.2,.2,.2))),
    'face_selection':'exact round robin within family and glyph count',
    'identity_weighting':'uniform native identity within selected face, before choosing view',
    'rng_independent':True, 'label_source':'verified complete native TRAIN glyphs only',
    'related_views_are_independent_samples':False, 'inference':False}

def load_prepared(root=DEFAULT_ROOT):
    root = Path(root).resolve()
    mp,rp,ap = (root/name for name in ('MANIFEST.json','rows.json','tiles.npy'))
    require(sha(mp)==MANIFEST_SHA, 'iOS short manifest is not the frozen capture')
    m=json.loads(mp.read_text())
    require(m.get('schema')=='flux-glyph-ios-native-short-train-v1' and m.get('families')==FAMILIES
            and m.get('split')=='train' and m.get('test_read') is False
            and m.get('calibration_read') is False and m.get('accepted_native_verified_only') is True,
            'Invalid iOS TRAIN manifest')
    bindings=dict(m['bindings'])
    bindings.update({str(mp):MANIFEST_SHA,str(rp):m['rows_sha256'],str(ap):m['array_sha256']})
    require(all(sha(p)==h for p,h in bindings.items()), 'iOS native source closure differs')
    payload=json.loads(rp.read_text())
    require(payload.get('schema')=='flux-glyph-ios-native-short-rows-v1' and payload.get('split')=='train',
            'iOS metadata is not TRAIN')
    rows=payload['rows']; tiles=np.load(ap,mmap_mode='r',allow_pickle=False)
    require(len(rows)==m['tiles']==2380 and tiles.shape==(len(rows),1,64,256) and tiles.dtype==np.dtype('<f4'),
            'iOS array shape/order differs')
    identities,owners,pools={},{},{}
    for i,r in enumerate(rows):
        require(r.get('split')=='train' and r.get('domain')=='ios'
                and r.get('source_kind')=='ios_simulator_screenshot' and r.get('native_font_verified') is True
                and r.get('whole_width_covered') is True and r.get('family') in IOS_FAMILIES
                and r.get('glyph_count') in (1,2,3,4) and r.get('view') in VIEWS, 'Unverified iOS row')
        require(r.get('index')==r.get('tile_start')==i and r.get('tile_count')==1
                and r.get('target')==FAMILIES.index(r['family']), 'iOS target/tile identity differs')
        tile=tiles[i]
        require(np.isfinite(tile).all() and tile.min()>=0 and tile.max()<=1
                and hashlib.sha256(tile.tobytes()).hexdigest()==r.get('tiles_sha256'), 'iOS tile bytes differ')
        require(math.isfinite(r['log_em_ratio']) and abs(r['log_em_ratio'])<=3
                and r['font_size_px']>0 and r['ink_height_px']>0
                and abs(math.log(r['font_size_px']/r['ink_height_px'])-r['log_em_ratio'])<1e-10,
                'iOS native size target differs')
        require(r.get('source_id') and r.get('region_id') and r.get('font_face')
                and r.get('normalized_text_sha256') and r.get('native_glyph_provenance',{}).get('glyphs')
                and len(r['native_glyph_provenance']['glyphs'])==r['glyph_count']
                and bindings.get(r['frame_path'])==r['frame_sha256'], 'iOS glyph/frame proof missing')
        ident=(r['source_id'],r['region_id']); group=identities.setdefault(ident,{})
        require(r['view'] not in group,'Duplicate iOS view');group[r['view']]=r
        for key in (('tile',r['tiles_sha256']),('region',r['region_rgb_sha256'])):
            require(owners.setdefault(key,r['family'])==r['family'],'Conflicting iOS pixel labels')
    fields=('family','target','font_face','glyph_count','source_id','page_id','region_id',
            'normalized_text_sha256','native_font_size_px','text_color_hex','source_sha256','frame_sha256')
    for ident,views in identities.items():
        require(set(views)==set(VIEWS),'iOS identity lacks four views')
        native=views['native']
        require(all([r[k] for k in fields]==[native[k] for k in fields] for r in views.values()),
                'iOS related views disagree on native truth')
        pools.setdefault((native['family'],native['glyph_count']),{}).setdefault(native['font_face'],{})[ident]=views
    require(len(identities)==m['native_regions']==595
            and set(pools)=={(f,n) for f in IOS_FAMILIES for n in (1,2,3,4)},'iOS family/length coverage differs')
    proof={'schema':'flux-glyph-ios-native-short-proof-v1','sampling':SAMPLING,
           'families':list(IOS_FAMILIES),'identity_count':len(identities),'views':len(rows),
           'by_family':dict(Counter(v['native']['family'] for v in identities.values())),
           'bindings':bindings,'test_read':False,'development_read':False,'calibration_read':False,
           'model_inference':False}
    return {'rows':rows,'tiles':tiles,'families':FAMILIES,'pools':pools},proof

class IOSShortSampler:
    def __init__(self,data,seed):
        self.data,self.rng,self.steps=data,np.random.default_rng(seed),0
        self.cursors,self.counts,self.faces,self.identities=Counter(),Counter(),Counter(),Counter()
    def batch(self):
        result=[]
        for family in IOS_FAMILIES:
            for length in (1,2,3,4):
                key=(family,length);pool=self.data['pools'][key];faces=sorted(pool)
                require(faces,'Empty iOS face pool')
                for _ in range(2):
                    face=faces[self.cursors[key]%len(faces)];self.cursors[key]+=1
                    ids=sorted(pool[face]);ident=ids[int(self.rng.integers(len(ids)))]
                    view=str(self.rng.choice(VIEWS,p=(.4,.2,.2,.2)))
                    result.append(copy.deepcopy(pool[face][ident][view]))
                    self.counts[family,length,view]+=1;self.faces[family,length,face]+=1;self.identities[ident]+=1
        self.steps+=1
        return result
    def report(self):
        return {'schema':SAMPLING['schema'],'steps':self.steps,'rows':self.steps*24,'sampling':SAMPLING,
                'by_family':{f:sum(n for (x,_,_),n in self.counts.items() if x==f) for f in IOS_FAMILIES},
                'by_glyph_count':{str(l):sum(n for (_,x,_),n in self.counts.items() if x==l) for l in (1,2,3,4)},
                'by_view':{v:sum(n for (_,_,x),n in self.counts.items() if x==v) for v in VIEWS},
                'by_face':{json.dumps(k):n for k,n in sorted(self.faces.items())},
                'by_identity':{json.dumps(k):n for k,n in sorted(self.identities.items())}}
