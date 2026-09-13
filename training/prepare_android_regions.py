#!/usr/bin/env python3
"""Prepare image-only training partitions from verified Android framebuffer captures."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unicodedata
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'training')]
from train_regions import sha,dump,require
from prepare_sans_views import apply_view,RECIPES,pixels_sha
from flux_glyph.region_font import preprocess_region

SCHEMA='flux-glyph-android-region-data-v1'
SPLITS=('train','calibration','test')
UNKNOWN='__unknown__'
CROP_RULE={'method':'native_ink_bbox_with_padding_v1','padding_px':'max(2, ceil(native_font_size_px * .15))',
           'rounding':'floor left/top, ceil right/bottom','clamp':'original verified layout bbox'}


def normalized_text(value):
    require(isinstance(value,str) and value.strip(),'missing native text identity')
    return ''.join(unicodedata.normalize('NFKC',value).casefold().split())


def inside(root,relative):
    require(isinstance(relative,str) and relative and not Path(relative).is_absolute(),'invalid capture relative path')
    path=(root/relative).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(),'capture path is missing or escapes root')
    return path


def source_crop_bbox(row):
    """Crop verified native ink with a fixed margin, retaining complete strokes."""
    bbox,ink,size=row.get('bbox'),row.get('ink_bbox'),row.get('font_size_screen_px')
    require(isinstance(bbox,list) and len(bbox)==4 and all(type(v) is int for v in bbox)
            and bbox[0]<bbox[2] and bbox[1]<bbox[3],'invalid native crop bounds')
    require(isinstance(ink,list) and len(ink)==4 and all(type(v) in (int,float) and math.isfinite(v) for v in ink)
            and bbox[0]<=ink[0]<ink[2]<=bbox[2] and bbox[1]<=ink[1]<ink[3]<=bbox[3],
            'native ink bounds are invalid or outside the verified layout')
    require(type(size) in (int,float) and math.isfinite(size) and 4<=size<=512,'invalid native pixel font size')
    padding=max(2,math.ceil(size*.15))
    return [max(bbox[0],math.floor(ink[0])-padding),max(bbox[1],math.floor(ink[1])-padding),
            min(bbox[2],math.ceil(ink[2])+padding),min(bbox[3],math.ceil(ink[3])+padding)]


def audit_identities(rows,families):
    require(len(families)==10 and len(set(families))==10 and families.count(UNKNOWN)==1,'expected nine named fonts plus unknown')
    seen,owners,unknown_owners,face_targets=set(),{},{},{}
    family_targets,file_splits={},{}
    image_identities,source_identities={},{}
    counts=Counter()
    for row in rows:
        split=row.get('split');family=row.get('training_family',row.get('font_family'))
        require(split in SPLITS and family in families,'invalid split or target family')
        for key in ('source_id','region_id','page_id','content_group_id','image','font_id','font_face','font_family'):
            require(isinstance(row.get(key),str) and row[key].strip(),'missing native identity: '+key)
        require(not Path(row['image']).is_absolute() and '..' not in Path(row['image']).parts,'invalid capture relative path')
        for key in ('image_sha256','font_file_sha256'):
            require(isinstance(row.get(key),str) and re.fullmatch(r'[0-9a-f]{64}',row[key]),'invalid native SHA: '+key)
        require(type(row.get('ttc_index')) is int and row['ttc_index']>=0,'invalid native TTC index')
        identity=(row['source_id'],row['region_id'])
        require(identity not in seen,'duplicate native region identity');seen.add(identity)
        require(row.get('native_font_verified') is True,'unverified native font')
        require(type(row.get('font_size_screen_px')) in (int,float) and math.isfinite(row['font_size_screen_px'])
                and 4<=row['font_size_screen_px']<=512,'invalid native pixel font size')
        require(isinstance(row.get('color'),str) and re.fullmatch(r'#[0-9A-Fa-f]{6}',row['color']),'invalid native text color')
        width,height=row.get('image_width'),row.get('image_height');bbox=row.get('bbox')
        require(type(width) is int and type(height) is int and width>0 and height>0,'invalid native image dimensions')
        require(isinstance(bbox,list) and len(bbox)==4 and all(type(v) is int for v in bbox)
                and 0<=bbox[0]<bbox[2]<=width and 0<=bbox[1]<bbox[3]<=height,'invalid native crop bounds')
        source_crop_bbox(row)
        image_identity=(row['image_sha256'],width,height)
        require(image_identities.setdefault(row['image'],image_identity)==image_identity,'inconsistent native image identity')
        source_identity=(row['page_id'],row['content_group_id'],row['image'],image_identity)
        require(source_identities.setdefault(row['source_id'],source_identity)==source_identity,'inconsistent native source identity')
        face=(row['font_file_sha256'],row['ttc_index'])
        require(face_targets.setdefault(face,family)==family,'same native font face has conflicting training classes')
        source_family=normalized_text(row['font_family'])
        require(family_targets.setdefault(source_family,family)==family,'same native font family has conflicting training classes')
        file_splits.setdefault(row['font_file_sha256'],set()).add(split)
        for kind,value in [('source',row['source_id']),('page',row['page_id']),('content',row['content_group_id']),
                           ('image',row['image']),('image_sha',row['image_sha256']),('text',normalized_text(row['text']))]:
            require(owners.setdefault((kind,value),split)==split,'native content crosses partitions: '+kind)
        if family==UNKNOWN:
            # Unknown holdouts are entire real families and source files, not
            # just another weight or TTC face from a training font.
            for key in (('file',row['font_file_sha256']),('family',normalized_text(row['font_family']))):
                require(unknown_owners.setdefault(key,split)==split,'unknown font source/family crosses train/CAL/TEST')
        counts[(split,family)]+=1
    for (kind,value),split in unknown_owners.items():
        if kind=='file':require(file_splits[value]=={split},'unknown font source file appears in another partition')
    require(all(counts[(s,f)]>0 for s in SPLITS for f in families),'native capture lacks a split/class')
    return counts


def selection_evidence(path, expected_sha=None, data_sha=None):
    require(path is not None,'TEST preparation requires a frozen passed CAL selection')
    path=Path(path).resolve()
    require(path.is_file(),'frozen calibration selection is missing')
    digest=sha(path)
    require(expected_sha is None or digest==expected_sha,'frozen calibration selection changed')
    selected=json.loads(path.read_text())
    require(selected.get('schema')=='flux-glyph-android-training-selection-v1' and selected.get('passed') is True
            and selected.get('test_read') is False,'TEST requires frozen successful calibration')
    require(data_sha is None or selected.get('data_manifest_sha256')==data_sha,
            'TEST source differs from the selected training data')
    return {'path':str(path),'sha256':digest}


def tile_sha(tiles):
    return hashlib.sha256(np.asarray(tiles,dtype='<f4').tobytes()).hexdigest()


def prepare(capture,output,split,selection=None):
    from training.capture.capture_android import load_capture,verify_region
    require(split in SPLITS,'invalid data partition')
    capture,output=Path(capture).resolve(),Path(output).resolve()
    frozen=None
    if split=='test':
        frozen=selection_evidence(selection)
    source=load_capture(capture)
    families=source['families'];audit_identities(source['rows'],families)
    bindings={str(Path(path).resolve()):digest for path,digest in source['bindings'].items()}
    for path in [Path(__file__),ROOT/'training/train_regions.py',ROOT/'training/prepare_sans_views.py',
                 ROOT/'training/capture/capture_android.py',ROOT/'src/flux_glyph/region_font.py',capture/'CAPTURE_MANIFEST.json']:
        bindings[str(path.resolve())]=sha(path)
    root_manifest={'schema':SCHEMA,'families':families,'capture':str(capture),
        'capture_manifest_sha256':source['manifest_sha256'],'bindings':bindings,'views':RECIPES,
        'model_inputs':['image_tiles'],'ocr_performed':False,'text_or_platform_features_used':False,
        'source_crop':CROP_RULE,
        'source_kind':'android_emulator_screenshot','split_policy':'Native source/normalized text partitions; whole unknown font families and files disjoint across all three partitions.',
        'preprocessing_shape':[1,64,256],'size_target':'log(native pixel font size * actual vertical view scale / measured ink height)'}
    output.mkdir(parents=True,exist_ok=True)
    if (output/'MANIFEST.json').exists():
        require(json.loads((output/'MANIFEST.json').read_text())==root_manifest,'source or preprocessing changed between partitions')
    else:dump(output/'MANIFEST.json',root_manifest)
    if frozen:
        selection_evidence(frozen['path'],frozen['sha256'],sha(output/'MANIFEST.json'))
    target=output/split;require(not target.exists(),'prepared partition must be new')
    selected_rows=[r for r in source['rows'] if r['split']==split]
    crop_owners={}
    for other in SPLITS:
        path=output/other/'rows.json'
        if path.exists():
            part=json.loads((output/other/'MANIFEST.json').read_text())
            require(part['root_manifest_sha256']==sha(output/'MANIFEST.json')
                    and part['metadata']['path']=='rows.json' and part['metadata']['sha256']==sha(path),
                    'previous partition metadata changed')
            for row in json.loads(path.read_text()):
                crop_owners[row['native_crop_sha256']]=other
            for row in part.get('rejected',[]):
                crop_owners[row['native_crop_sha256']]=other
    with tempfile.TemporaryDirectory(prefix='.android-data-',dir=output) as temporary:
        stage=Path(temporary);rows=[];rejected=[];tile_count=0;image_cache={}
        with (stage/'tiles.raw').open('wb') as stream:
            for native in selected_rows:
                verify_region(capture,native)
                path=inside(capture,native['image'])
                if path not in image_cache:
                    require(sha(path)==native['image_sha256'],'native screenshot changed')
                    with Image.open(path) as opened:image=opened.convert('RGB')
                    require(list(image.size)==[native['image_width'],native['image_height']],'native image dimensions changed')
                    image_cache={path:image}
                image=image_cache[path]
                bbox=source_crop_bbox(native)
                require(isinstance(bbox,list) and len(bbox)==4 and all(type(v) is int for v in bbox)
                        and 0<=bbox[0]<bbox[2]<=image.width and 0<=bbox[1]<bbox[3]<=image.height,'invalid native crop bounds')
                crop=image.crop(bbox);crop_sha=pixels_sha(crop)
                require(crop_owners.setdefault(crop_sha,split)==split,'identical crop pixels cross partitions')
                family=native.get('training_family',native['font_family'])
                for view in RECIPES:
                    pixels,scale_y,_=apply_view(crop,view);processed=preprocess_region(pixels)
                    if processed['status']!='ok' or not processed.get('whole_width_covered'):
                        rejected.append({'source_id':native['source_id'],'region_id':native['region_id'],'family':family,
                                         'source_crop_bbox':bbox,'native_crop_sha256':crop_sha,
                                         'view':view,'reason':processed.get('reason','whole_width_not_covered')});continue
                    tiles=processed['tiles'];height=processed['ink_height_px'];size=native['font_size_screen_px']*scale_y
                    require(math.isfinite(height) and height>0 and math.isfinite(size) and size>0,'invalid prepared size geometry')
                    ratio=math.log(size/height)
                    require(tiles.dtype==np.float32 and tiles.ndim==4 and tiles.shape[1:]==(1,64,256)
                            and 1<=len(tiles)<=8 and np.isfinite(tiles).all() and np.all((tiles>=0)&(tiles<=1))
                            and -3<=ratio<=3,'invalid prepared tiles/size')
                    stream.write(tiles.astype('<f4',copy=False).tobytes())
                    rows.append({'split':split,'family':family,'target':families.index(family),'source_id':native['source_id'],
                        'page_id':native['page_id'],'region_id':native['region_id'],'source_font_family':native['font_family'],
                        'font_id':native['font_id'],'font_face':native['font_face'],'font_file_sha256':native['font_file_sha256'],
                        'ttc_index':native['ttc_index'],'tiles_sha256':tile_sha(tiles),
                        'source_crop_bbox':bbox,
                        'native_crop_sha256':crop_sha,'view':view,'tile_start':tile_count,'tile_count':len(tiles),
                        'ink_height_px':height,'font_size_px':size,'log_em_ratio':ratio,'text_color_hex':native['color']})
                    tile_count+=len(tiles)
        require(rows and all(any(r['family']==f and r['view']==v for r in rows) for f in families for v in RECIPES),
                'a family/view has no usable native regions')
        require(len(rejected)<=.05*(len(rows)+len(rejected)),'more than 5% views rejected; investigate data before training')
        dump(stage/'rows.json',rows)
        manifest={'schema':SCHEMA,'split':split,'root_manifest_sha256':sha(output/'MANIFEST.json'),'families':families,
            'native_regions':len(selected_rows),'views':len(rows),'tiles':tile_count,'shape':[tile_count,1,64,256],
            'array':{'path':'tiles.raw','sha256':sha(stage/'tiles.raw')},'metadata':{'path':'rows.json','sha256':sha(stage/'rows.json')},
            'counts':dict(Counter(r['family'] for r in rows)),'rejected':rejected,'frozen_selection':frozen}
        dump(stage/'MANIFEST.json',manifest)
        require(all(Path(p).is_file() and sha(p)==s for p,s in bindings.items()),'capture/preparation source changed')
        if frozen:selection_evidence(frozen['path'],frozen['sha256'],sha(output/'MANIFEST.json'))
        shutil.move(str(stage),str(target))
    print(json.dumps({k:manifest[k] for k in ('split','native_regions','views','tiles','counts')},ensure_ascii=False),flush=True)
    return manifest


def load_split(root,split):
    from training.capture.capture_android import load_capture,verify_region
    root=Path(root).resolve();require(split in SPLITS,'invalid data split')
    metadata=json.loads((root/'MANIFEST.json').read_text());partition=json.loads((root/split/'MANIFEST.json').read_text())
    require(metadata['schema']==partition['schema']==SCHEMA and partition['split']==split
            and partition['families']==metadata['families'] and partition['root_manifest_sha256']==sha(root/'MANIFEST.json'),
            'partition source or family order changed')
    require(metadata.get('source_crop')==CROP_RULE,'prepared native crop rule differs')
    if split=='test':
        frozen=partition.get('frozen_selection')
        require(isinstance(frozen,dict) and isinstance(frozen.get('path'),str)
                and isinstance(frozen.get('sha256'),str) and re.fullmatch(r'[0-9a-f]{64}',frozen['sha256']),
                'TEST partition lacks a frozen calibration selection')
        selection_evidence(frozen.get('path'),frozen.get('sha256'),sha(root/'MANIFEST.json'))
    require(all(Path(p).is_file() and sha(p)==digest for p,digest in metadata['bindings'].items()),'bound preparation source changed')
    source=load_capture(Path(metadata['capture']))
    require(source['manifest_sha256']==metadata['capture_manifest_sha256'] and source['families']==metadata['families'],
            'native capture identity changed')
    audit_identities(source['rows'],source['families'])
    native_rows={(r['source_id'],r['region_id']):r for r in source['rows'] if r['split']==split}
    # Verify only this partition's screenshots/proofs; TEST pixels remain closed
    # while loading train or CAL. Metadata audits above read no image pixels.
    images={};native_crops={}
    for native in native_rows.values():
        verify_region(Path(metadata['capture']),native)
        path=inside(Path(metadata['capture']),native['image'])
        if path not in images:
            require(sha(path)==native['image_sha256'],'native screenshot changed')
            with Image.open(path) as opened:image=opened.convert('RGB')
            require(list(image.size)==[native['image_width'],native['image_height']],'native image dimensions changed')
            images={path:image}
        bbox=source_crop_bbox(native)
        native_crops[(native['source_id'],native['region_id'])]=(bbox,pixels_sha(images[path].crop(bbox)))
    for key in ('array','metadata'):
        path=inside(root/split,partition[key]['path']);require(sha(path)==partition[key]['sha256'],'prepared partition bytes changed')
    rows=json.loads((root/split/partition['metadata']['path']).read_text())
    require(len(rows)==partition['views'] and all(r['split']==split and type(r['target']) is int
            and 0<=r['target']<len(metadata['families']) and r['family']==metadata['families'][r['target']] for r in rows),
            'prepared region identity changed')
    offset=0;view_identities=set()
    for row in rows:
        require(type(row['tile_start']) is int and row['tile_start']==offset and type(row['tile_count']) is int
                and 1<=row['tile_count']<=8 and row['view'] in RECIPES,'invalid tile mapping')
        key=(row['source_id'],row['region_id']);native=native_rows.get(key)
        require(native is not None and row['family']==native.get('training_family',native['font_family'])
                and row['page_id']==native['page_id'] and row['source_font_family']==native['font_family']
                and all(row.get(k)==native[k] for k in ('font_id','font_face','font_file_sha256','ttc_index'))
                and row['text_color_hex']==native['color'],'prepared native truth/source differs')
        identity=(*key,row['view']);require(identity not in view_identities,'duplicate prepared region/view')
        view_identities.add(identity)
        bbox,crop_sha=native_crops[key]
        require(row.get('source_crop_bbox')==bbox and row.get('native_crop_sha256')==crop_sha,
                'prepared crop differs from native ink bounds/pixels')
        crop_height=bbox[3]-bbox[1]
        view_height=max(1,round(crop_height*RECIPES[row['view']]['scale']))
        size=native['font_size_screen_px']*view_height/crop_height
        height=row['ink_height_px'];ratio=row['log_em_ratio']
        require(type(height) in (int,float) and math.isfinite(height) and height>0
                and type(ratio) in (int,float) and math.isfinite(ratio) and -3<=ratio<=3
                and math.isclose(row['font_size_px'],size,rel_tol=1e-12)
                and math.isclose(ratio,math.log(size/height),rel_tol=1e-12,abs_tol=1e-12),'prepared native size target differs')
        offset+=row['tile_count']
    path=root/split/partition['array']['path']
    require(offset==partition['tiles'] and partition['shape']==[offset,1,64,256]
            and path.stat().st_size==offset*64*256*4,'invalid native array shape')
    tiles=np.memmap(path,mode='r',dtype='<f4',shape=tuple(partition['shape']))
    for row in rows:
        tensor=tiles[row['tile_start']:row['tile_start']+row['tile_count']]
        require(tile_sha(tensor)==row['tiles_sha256'] and np.isfinite(tensor).all()
                and np.all((tensor>=0)&(tensor<=1)),'prepared tensor pixels/hash differ')
    rejected=partition.get('rejected',[])
    for row in rejected:
        key=(row['source_id'],row['region_id']);native=native_rows.get(key);identity=(*key,row['view'])
        require(native is not None and row['family']==native.get('training_family',native['font_family'])
                and row['view'] in RECIPES and identity not in view_identities,'invalid rejected native region/view')
        bbox,crop_sha=native_crops[key]
        require(row.get('source_crop_bbox')==bbox and row.get('native_crop_sha256')==crop_sha,
                'rejected crop differs from native ink bounds/pixels')
        view_identities.add(identity)
    require(len(view_identities)==len(native_rows)*len(RECIPES) and partition['native_regions']==len(native_rows)
            and partition['counts']==dict(Counter(r['family'] for r in rows)),'native region/view coverage changed')
    require(len(rejected)<=.05*len(view_identities),'prepared rejection coverage exceeds 5%')
    if split=='test':selection_evidence(frozen['path'],frozen['sha256'],sha(root/'MANIFEST.json'))
    return {'tiles':tiles,'rows':rows,
            'families':metadata['families'],'manifest':metadata,'partition':partition,
            'manifest_sha256':sha(root/'MANIFEST.json'),'partition_sha256':sha(root/split/'MANIFEST.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split',choices=SPLITS,required=True);p.add_argument('--frozen-selection',type=Path)
    a=p.parse_args();prepare(a.capture,a.output,a.split,a.frozen_selection)
