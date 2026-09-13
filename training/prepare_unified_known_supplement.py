#!/usr/bin/env python3
"""Verify paired TRAIN-only native known-font screenshots without altering old data."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'training')]
from train_regions import require,sha,dump
from prepare_android_regions import inside,source_crop_bbox,CROP_RULE,tile_sha,normalized_text
from prepare_sans_views import apply_view,RECIPES,pixels_sha
from prepare_unified_regions import FAMILIES,UNKNOWN
from flux_glyph.region_font import preprocess_region
from training.capture.capture_android import load_capture,verify_region

SCHEMA='flux-glyph-unified-known-supplement-v1'
DEFAULT=ROOT/'artifacts/unified-font-v2/paired-known-capture-v1'
EXPECTED_SOURCES={'LXGW WenKai','WenQuanYi Micro Hei'}
PAIR_MAPPING={'Zhuque Fangsong':'LXGW WenKai','WenQuanYi Zen Hei':'WenQuanYi Micro Hei'}
PAIRED_UNKNOWN=ROOT/'artifacts/unified-font-v2/new-unknown-capture-v1'

def read(path):return json.loads(Path(path).read_text())

def bind(bindings,path,expected=None):
    path=Path(path).resolve();digest=sha(path)
    require(expected is None or digest==expected,'Bound supplement source changed: '+str(path))
    require(str(path) not in bindings or bindings[str(path)]==digest,'Conflicting supplement source identity')
    bindings[str(path)]=digest
    return path

def text_sha(text):return hashlib.sha256(normalized_text(text).encode()).hexdigest()

def historical_audit(data,exclusion,bindings):
    """Metadata only: never open old TRAIN/CAL/holdout/TEST image or tile files."""
    data=Path(data).resolve();manifest=read(bind(bindings,data/'MANIFEST.json'))
    require(manifest['families']==FAMILIES and FAMILIES[-1]==UNKNOWN,'Unified 25-class order changed')
    owners={key:set() for key in ('source_id','page_id','source_sha256','native_crop_sha256','normalized_text_sha256',
                                  'decoded_pixel_sha256','source_font_family','font_file_sha256')}
    counts={}
    for split in ('train','calibration','development_holdout'):
        part=read(bind(bindings,data/split/'MANIFEST.json'))
        require(part['root_manifest_sha256']==sha(data/'MANIFEST.json') and part['split']==split
                and part['families']==FAMILIES,'Historical partition identity changed')
        path=inside(data/split,part['metadata']['path']);bind(bindings,path,part['metadata']['sha256'])
        rows=read(path);require(len(rows)==part['views'],'Historical row count changed')
        for row in [*rows,*part.get('rejected',[])]:
            for key in owners:
                if row.get(key):owners[key].add(normalized_text(row[key]) if key=='source_font_family' else row[key])
        counts[split]=len(rows)
    exclusion_path=bind(bindings,exclusion);excluded=read(exclusion_path)
    require(excluded['schema']=='flux-glyph-new-train-text-exclusion-v1','Missing historical text exclusion')
    full_texts=set()
    for descriptor in excluded['source_metadata']:
        path=bind(bindings,descriptor['path'],descriptor['sha256'])
        records=read(path) if path.suffix=='.json' else [json.loads(line) for line in path.read_text().splitlines()]
        require(isinstance(records,list),'Historical label metadata must be a list or JSONL')
        for page in records:
            for row in page.get('regions',[page]):
                if row.get('text') and row['text'].strip():full_texts.add(text_sha(row['text']))
                if row.get('normalized_text_sha256'):full_texts.add(row['normalized_text_sha256'])
                for key in ('font_file_sha256','source_font_family'):
                    value=row.get(key, row.get('font_family') if key=='source_font_family' else row.get('font_source',{}).get('sha256'))
                    if value:owners[key].add(normalized_text(value) if key=='source_font_family' else value)
            for key,aliases in {'source_id':('source_id',),'page_id':('page_id',),
                               'source_sha256':('source_sha256','image_sha256'),
                               'decoded_pixel_sha256':('decoded_pixel_sha256',)}.items():
                for alias in aliases:
                    if page.get(alias):owners[key].add(page[alias])
    require(full_texts==set(excluded['hashes']),'Historical text exclusion is incomplete or changed')
    owners['normalized_text_sha256'].update(full_texts)
    return owners,{'existing_partition_rows':counts,'historical_unique_texts':len(full_texts),
        'old_images_or_arrays_read':False,'overlap_policy':'Reject new source/page/image/crop/text identities found in any old partition; fonts must be new source families and files.'}

def native_audit(rows,fonts,owners,plan):
    require(len(rows)==plan['regions'] and len({r['page_id'] for r in rows})==plan['pages'], 'New capture is incomplete')
    registry={r['id']:r for r in fonts};require(set(f['family'] for f in fonts)==EXPECTED_SOURCES,'Unexpected supplemental font source')
    seen=set();sources={};counts=Counter();scripts=Counter()
    for row in rows:
        require(row.get('split')=='train' and row.get('training_family')==row.get('font_family') and row.get('font_family') in EXPECTED_SOURCES
                and row.get('native_font_verified') is True,'Supplement accepts verified named TRAIN fonts only')
        identity=(row['source_id'],row['region_id']);require(identity not in seen,'Duplicate new native region');seen.add(identity)
        font=registry.get(row['font_id']);require(font is not None and font['family']==row['font_family']
            and font['postscript']==row['font_face'] and font['sha256']==row['font_file_sha256']
            and font.get('ttc_index',0)==row['ttc_index']==0 and font.get('allowed_splits')==['train'],
            'Supplement source face or allowed split differs')
        require((row['font_family'],row['font_face'],row['font_file_sha256'],row['ttc_index']) in owners['known_train_faces'],
                'Paired font face must already be a verified original TRAIN class')
        for key,value in [('source_id',row['source_id']),('page_id',row['page_id']),
                          ('source_sha256',row['image_sha256']),('normalized_text_sha256',text_sha(row['text']))]:
            require(value not in owners[key],'Supplement overlaps historical '+key)
        require(isinstance(row['image_width'],int) and isinstance(row['image_height'],int)
                and 0<=row['bbox'][0]<row['bbox'][2]<=row['image_width']
                and 0<=row['bbox'][1]<row['bbox'][3]<=row['image_height'],'Invalid screenshot bounds')
        source_crop_bbox(row)
        signature=(row['page_id'],row['image'],row['image_sha256'],row['image_width'],row['image_height'])
        require(sources.setdefault(row['source_id'],signature)==signature,'Inconsistent native screenshot identity')
        counts[row['font_family']]+=1;scripts[row['script']]+=1
    require(all(counts[f]==plan['per_family_regions'] for f in EXPECTED_SOURCES)
            and scripts['han']/len(rows)>=.8,'Supplement family count or Han coverage differs')
    return {'native_regions':len(rows),'pages':len(sources),'source_family_counts':dict(counts),'script_counts':dict(scripts)}

def paired_audit(natives,pair_document,known_scenes,unknown_scenes,unknown_rows,bindings):
    """Pair native TRAIN content/style exactly; class labels remain the actual fonts."""
    require(pair_document.get('schema')=='flux-glyph-known-unknown-train-pairs-v1', 'Missing explicit TRAIN pairing declaration')
    pairs=pair_document['pairs'];new={r['region_id']:r for r in natives};old={r['region_id']:r for r in unknown_rows}
    require(len(new)==len(natives)==len(old)==len(unknown_rows)==len(pairs)
        and len({p['new_region_id'] for p in pairs})==len(pairs)
        and len({p['source_region_id'] for p in pairs})==len(pairs),'Pairs must cover both native captures exactly once')
    def index(scenes):
        return {r['id']:(p,r) for p in scenes['pages'] for r in p['regions']}
    ni,oi=index(known_scenes),index(unknown_scenes)
    require(len(ni)==len(new) and len(oi)==len(old) and known_scenes['canvas_px']==unknown_scenes['canvas_px'],
        'Pair canvas or scene count differs')
    expected_fields=['text','script','font_size_px','color','bbox','page_background','canvas_px']
    evidence={}
    for pair in pairs:
        require(pair.get('matched_fields')==expected_fields and pair.get('source_native_font_verified') is True,
            'Incomplete explicit native pairing evidence')
        a=new.get(pair['new_region_id']);b=old.get(pair['source_region_id'])
        require(a is not None and b is not None and a['split']==b['split']=='train'
            and a['native_font_verified'] is b['native_font_verified'] is True
            and a['training_family']==a['font_family'] in EXPECTED_SOURCES and b['training_family']==UNKNOWN
            and PAIR_MAPPING.get(b['font_family'])==a['font_family'],'Pair must link declared known/unknown TRAIN fonts')
        ap,ar=ni[a['region_id']];bp,br=oi[b['region_id']]
        require(ap['id']==a['page_id']==pair['new_page_id'] and bp['id']==b['page_id']==pair['source_page_id']
            and a['font_id']==ar['font_id']==pair['known_font_id']
            and b['font_id']==br['font_id']==pair['source_unknown_font_id']
            and ap['split']==bp['split']=='train' and ap['background']==bp['background']
            and all(ar[k]==br[k] for k in ('text','script','font_size_px','color','bbox')),
            'Paired scene content, native font, size, color, position or background changed')
        require(all(a[k]==b[k] for k in ('text','script','font_size_screen_px','color','bbox','background','image_width','image_height')),
            'Actual paired native content/style differs')
        proof=bind(bindings,pair['source_proof']['path'],pair['source_proof']['sha256'])
        require(proof==(PAIRED_UNKNOWN/'capture'/b['proof']).resolve(),'Pair points to a different native proof')
        # The old new-unknown capture is TRAIN-only; read glyph proof and hash PNG, never decode it.
        verify_region(PAIRED_UNKNOWN/'capture',b)
        actual=next(r for r in read(proof)['regions'] if r['id']==b['region_id'])
        require(len(actual['glyphs'])==pair['source_glyphs'],'Paired source glyph coverage changed')
        evidence[a['region_id']]={'source_id':b['source_id'],'region_id':b['region_id'],'font_family':b['font_family'],
            'training_family':UNKNOWN,'source_proof':str(proof),'source_proof_sha256':sha(proof),
            'source_image_sha256':b['image_sha256'],'normalized_text_sha256':text_sha(b['text']),
            'matched_fields':expected_fields,'intentional_train_text_pair':True}
    return evidence


def context(capture,data,plan_path):
    capture=Path(capture).resolve();data=Path(data).resolve();bindings={}
    plan=read(bind(bindings,plan_path));require(plan['schema']=='flux-glyph-paired-known-native-capture-plan-v1'
        and plan['only_split']=='train' and Path(plan['capture_directory']).resolve()==capture,'Capture plan scope differs')
    for path,digest in plan['bindings'].items():bind(bindings,path,digest)
    for key in ('source_registry','scenes','pair_bindings'):bind(bindings,plan[key]['path'],plan[key]['sha256'])
    sources=read(plan['source_registry']['path']);fonts=sources['fonts']
    require(sources.get('fonts_existing_known_classes') is True and sources.get('new_classes')==0
        and sources.get('only_split')=='train','Paired sources must retain their existing known classes')
    for font in fonts:
        bind(bindings,font['path'],font['sha256'])
        for item in font.get('license_files',[]):bind(bindings,item['path'],item['sha256'])
    source=load_capture(capture)
    for path,digest in source['bindings'].items():bind(bindings,path,digest)
    manifest=read(capture/'CAPTURE_MANIFEST.json')
    require(manifest['scenes']['sha256']==plan['scenes']['sha256'] and manifest.get('smoke_only') is False
        and manifest.get('user_images_used') is False and manifest['split_counts']=={'train':plan['regions']},
        'Capture differs from the paired TRAIN-only plan')
    for entry in manifest['files']:bind(bindings,inside(capture,entry['path']),entry['sha256'])
    # Keep all historical text exclusions. The only intentional text reuse is the separately bound TRAIN supplement.
    owners,audit=historical_audit(data,PAIRED_UNKNOWN/'TEXT_EXCLUSION.json',bindings)
    train=read(data/'train/rows.json')
    owners['known_train_faces']={(r['family'],r['font_face'],r['font_file_sha256'],r.get('ttc_index',0))
        for r in train if r['family'] in EXPECTED_SOURCES and r.get('native_font_verified') is True}
    audit['overlap_policy']='Allow only the declared new-unknown TRAIN text/style pairs. Reject all original partition text/image identities; reuse the two existing known TRAIN font faces.'
    audit.update(native_audit(source['rows'],fonts,owners,plan))
    pair_document=read(plan['pair_bindings']['path'])
    old_scene_path=bind(bindings,pair_document['source_scenes']['path'],pair_document['source_scenes']['sha256'])
    require(old_scene_path==(PAIRED_UNKNOWN/'Scenes.json').resolve(),'Wrong paired TRAIN source scenes')
    unknown=load_capture(PAIRED_UNKNOWN/'capture')
    for path,digest in unknown['bindings'].items():bind(bindings,path,digest)
    unknown_manifest=read(PAIRED_UNKNOWN/'capture/CAPTURE_MANIFEST.json')
    require(unknown_manifest['split_counts']=={'train':plan['regions']}
        and unknown_manifest['scenes']['sha256']==pair_document['source_scenes']['sha256'],'Paired source is not the unchanged TRAIN capture')
    for entry in unknown_manifest['files']:bind(bindings,inside(PAIRED_UNKNOWN/'capture',entry['path']),entry['sha256'])
    owners['pair_evidence']=paired_audit(source['rows'],pair_document,read(plan['scenes']['path']),read(old_scene_path),unknown['rows'],bindings)
    # Existing supplement metadata prevents same PNG/crop reuse while its paired text remains deliberately allowed.
    old_part=read(bind(bindings,PAIRED_UNKNOWN/'data/train/MANIFEST.json'))
    old_rows_path=inside(PAIRED_UNKNOWN/'data/train',old_part['metadata']['path'])
    bind(bindings,old_rows_path,old_part['metadata']['sha256'])
    for row in [*read(old_rows_path),*old_part['rejected']]:
        for key in ('source_id','page_id','source_sha256','native_crop_sha256','decoded_pixel_sha256'):
            if row.get(key):owners[key].add(row[key])
    # Newly added old-supplement image owners must also reject collisions before any known pixel decode.
    for row in source['rows']:
        require(row['source_id'] not in owners['source_id'] and row['page_id'] not in owners['page_id']
            and row['image_sha256'] not in owners['source_sha256'],'Paired known source reuses an unknown screenshot')
    audit.update(paired_native_regions=len(owners['pair_evidence']),intentional_train_text_pairing=True,
        pairing_manifest_sha256=plan['pair_bindings']['sha256'],known_targets={f:FAMILIES.index(f) for f in sorted(EXPECTED_SOURCES)},
        unknown_source_images_decoded=False,heldout_text_overlap=0)
    for path in [Path(__file__),ROOT/'training/prepare_android_regions.py',ROOT/'training/prepare_sans_views.py',
        ROOT/'training/prepare_unified_regions.py',ROOT/'training/train_regions.py',
        ROOT/'training/capture/capture_android.py',ROOT/'src/flux_glyph/region_font.py']:
        bind(bindings,path)
    return source['rows'],owners,bindings,audit


def verified_views(capture,natives,owners):
    """Only new screenshot pixels are opened; every region gets native glyph proof."""
    capture=Path(capture).resolve();images={};crops={};new_images={}
    for native in natives:
        verify_region(capture,native)
        path=inside(capture,native['image'])
        if path not in images:
            require(sha(path)==native['image_sha256'],'New screenshot changed')
            with Image.open(path) as opened:image=opened.convert('RGB')
            require(list(image.size)==[native['image_width'],native['image_height']],'Screenshot dimensions differ')
            image_sha=pixels_sha(image)
            require(image_sha not in owners['decoded_pixel_sha256'],'Decoded screenshot overlaps old pixels')
            require(new_images.setdefault(image_sha,native['source_id'])==native['source_id'],'Duplicate decoded screenshot source')
            images={path:(image,image_sha)}
        image,image_sha=images[path];bbox=source_crop_bbox(native);crop=image.crop(bbox);crop_sha=pixels_sha(crop)
        identity=(native['source_id'],native['region_id'])
        require(crop_sha not in owners['native_crop_sha256'] and crops.setdefault(crop_sha,identity)==identity,
                'Native crop duplicates an old or new region')
        base={'split':'train','original_split':'train','evaluation_role':'paired_known_train_only_supplement',
            'family':native['training_family'],'target':FAMILIES.index(native['training_family']),'source_id':native['source_id'],'page_id':native['page_id'],
            'region_id':native['region_id'],'source_font_family':native['font_family'],'font_face':native['font_face'],
            'font_id':native['font_id'],'font_file_sha256':native['font_file_sha256'],'ttc_index':native['ttc_index'],
            'font_source_kind':'asset','source_kind':'android_emulator_screenshot','source_sha256':native['image_sha256'],
            'decoded_pixel_sha256':image_sha,'source_crop_bbox':bbox,'native_crop_sha256':crop_sha,
            'domain':'android','source_dataset':'android_paired_known_supplement','script':native['script'],
            'normalized_text_sha256':text_sha(native['text']),'native_font_verified':True,
            'image':str(path),'proof':str(inside(capture,native['proof'])),'proof_sha256':sha(capture/native['proof']),
            'text_color_hex':native['color'],'pair_evidence':owners['pair_evidence'][native['region_id']]}
        for view in RECIPES:
            pixels,scale_y,_=apply_view(crop,view);processed=preprocess_region(pixels)
            if processed['status']!='ok' or not processed.get('whole_width_covered'):
                yield {**base,'view':view,'reason':processed.get('reason','whole_width_not_covered')},None
                continue
            tiles=processed['tiles'];height=processed['ink_height_px'];size=native['font_size_screen_px']*scale_y;ratio=math.log(size/height)
            require(tiles.dtype==np.float32 and tiles.ndim==4 and tiles.shape[1:]==(1,64,256) and 1<=len(tiles)<=8
                and np.isfinite(tiles).all() and np.all((tiles>=0)&(tiles<=1)) and math.isfinite(ratio) and -3<=ratio<=3,
                'Invalid native supplement geometry or tiles')
            yield {**base,'view':view,'tiles_sha256':tile_sha(tiles),'tile_count':len(tiles),'ink_height_px':height,
                'font_size_px':size,'log_em_ratio':ratio},tiles

def prepare(capture,data,plan,output):
    output=Path(output).resolve();require(not output.exists(),'Supplement output must be new')
    natives,owners,bindings,audit=context(capture,data,plan)
    output.mkdir(parents=True)
    frozen={'schema':'flux-glyph-unified-known-supplement-freeze-v1','families':FAMILIES,'only_split':'train',
        'capture':str(Path(capture).resolve()),'parent_data':str(Path(data).resolve()),'capture_plan':str(Path(plan).resolve()),
        'bindings':bindings,'views':RECIPES,'source_crop':CROP_RULE,'preprocessing_shape':[1,64,256],
        'old_data_modified':False,'old_cache_modified':False,'old_images_or_arrays_read':False,
        'calibration_or_development_or_test_pixels_read':False,'optimizer_steps':0,'audit':audit}
    dump(output/'PREPARATION_FREEZE.json',frozen);freeze_sha=sha(output/'PREPARATION_FREEZE.json')
    folder=output/'train';folder.mkdir();rows=[];rejected=[];offset=0
    with (folder/'tiles.raw').open('wb') as stream:
        for row,tiles in verified_views(capture,natives,owners):
            if tiles is None:rejected.append(row);continue
            row['tile_start']=offset;offset+=len(tiles);rows.append(row);stream.write(tiles.astype('<f4',copy=False).tobytes())
    require(rows and len(rejected)<=.05*(len(rows)+len(rejected)), 'More than 5% supplement views rejected')
    require(all(any(r['source_font_family']==f and r['view']==v for r in rows) for f in EXPECTED_SOURCES for v in RECIPES),
            'Supplement source/view lacks accepted examples')
    require(all(sha(p)==d for p,d in bindings.items()) and sha(output/'PREPARATION_FREEZE.json')==freeze_sha,
            'Supplement source changed during preparation')
    dump(folder/'rows.json',rows)
    manifest={**frozen,'schema':SCHEMA,'preparation_freeze_sha256':freeze_sha,
        'source_kind':'android_emulator_screenshot','model_inputs':['image_tiles'],'ocr_performed':False,
        'text_or_platform_features_used':False,'derived_views_are_correlated':True,'native_font_verified':True}
    dump(output/'MANIFEST.json',manifest)
    part={'schema':SCHEMA,'split':'train','families':FAMILIES,'root_manifest_sha256':sha(output/'MANIFEST.json'),
        'native_regions':len(natives),'views':len(rows),'tiles':offset,'shape':[offset,1,64,256],
        'counts':dict(Counter(r['family'] for r in rows)),'source_family_counts':dict(Counter(r['source_font_family'] for r in rows)),
        'rejected':rejected,'array':{'path':'tiles.raw','sha256':sha(folder/'tiles.raw')},
        'metadata':{'path':'rows.json','sha256':sha(folder/'rows.json')}}
    dump(folder/'MANIFEST.json',part);print(json.dumps({k:part[k] for k in ('native_regions','views','tiles','source_family_counts')}),flush=True)
    return manifest

def load_supplement(root):
    root=Path(root).resolve();manifest=read(root/'MANIFEST.json');part=read(root/'train/MANIFEST.json')
    require(manifest['schema']==part['schema']==SCHEMA and part['split']==manifest['only_split']=='train'
            and manifest['families']==part['families']==FAMILIES and part['root_manifest_sha256']==sha(root/'MANIFEST.json')
            and manifest['preparation_freeze_sha256']==sha(root/'PREPARATION_FREEZE.json'), 'Supplement contract changed')
    frozen=read(root/'PREPARATION_FREEZE.json')
    require(all(manifest.get(k)==v for k,v in frozen.items() if k!='schema'),'Supplement frozen scope changed')
    require(all(sha(p)==d for p,d in manifest['bindings'].items()),'Bound supplement preparation changed')
    natives,owners,bindings,audit=context(manifest['capture'],manifest['parent_data'],manifest['capture_plan'])
    require(bindings==manifest['bindings'] and audit==manifest['audit'],'Supplement provenance differs')
    for key in ('array','metadata'):require(sha(inside(root/'train',part[key]['path']))==part[key]['sha256'],'Supplement tensor/row file changed')
    rows=read(root/'train'/part['metadata']['path']);path=root/'train'/part['array']['path'];count=part['tiles']
    require(type(count) is int and count>0 and part['shape']==[count,1,64,256] and path.stat().st_size==count*64*256*4
            and len(rows)==part['views'],'Supplement tensor shape differs')
    tiles=np.memmap(path,mode='r',dtype='<f4',shape=tuple(part['shape']));offset=0;index=0;rejected=[]
    for actual,block in verified_views(manifest['capture'],natives,owners):
        if block is None:rejected.append(actual);continue
        actual['tile_start']=offset
        require(index<len(rows) and rows[index]==actual,'Supplement row differs from actual native source/preprocessing')
        end=offset+len(block)
        require(np.array_equal(tiles[offset:end],block),'Supplement tiles differ from native screenshot preprocessing')
        offset=end;index+=1
    require(index==len(rows) and offset==count and rejected==part['rejected'] and part['native_regions']==len(natives)
            and part['counts']==dict(Counter(r['family'] for r in rows))
            and part['source_family_counts']==dict(Counter(r['source_font_family'] for r in rows)), 'Supplement coverage changed')
    return {'tiles':tiles,'rows':rows,'families':FAMILIES,'manifest':manifest,'partition':part,
        'manifest_sha256':sha(root/'MANIFEST.json'),'partition_sha256':sha(root/'train/MANIFEST.json')}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,default=DEFAULT/'capture')
    parser.add_argument('--data',type=Path,default=ROOT/'artifacts/unified-font-v1/data-v1')
    parser.add_argument('--plan',type=Path,default=DEFAULT/'CAPTURE_PLAN.json')
    parser.add_argument('--output',type=Path,default=DEFAULT/'data')
    args=parser.parse_args();prepare(args.capture,args.data,args.plan,args.output)
