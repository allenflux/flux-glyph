"""Supplement native training glyphs with independently rendered FreeType data."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTCollection, TTFont

from data import dump, sha
from network import FAMILIES
from flux_glyph.glyph_preprocess import extract_glyphs


def build(old_root, output):
    source_path=old_root/'runs/alipay-font-generalization-v8-20260911/data/fonts.json'
    splits_path=old_root/'runs/alipay-font-smalltext-v10-20260911/data/SPLIT.json'
    declared=json.loads(splits_path.read_text())['characters']
    chars={'train':declared['train'][:128],'calibration':declared['calibration'][:32],'test':declared['sealed_test'][:32]}
    sizes={'train':[18,28,40],'calibration':[22,34],'test':[24,38]}
    sources=[];excluded=[]
    for row in json.loads(source_path.read_text()):
        path=Path(row['path'])
        if sha(path)!=row['source_sha256']:raise ValueError('Source checksum mismatch')
        count=1
        try:
            if path.suffix.lower()=='.ttc':
                collection=TTCollection(path,lazy=True);count=len(collection.fonts);collection.close()
            found=None
            for index in range(count):
                with TTFont(path,fontNumber=index,lazy=True) as font:
                    if font['name'].getDebugName(6)==row['postscript_name']:
                        found=index;cmap=set(font.getBestCmap());break
            if found is None:raise ValueError('PostScript face unavailable to FreeType')
            ImageFont.truetype(str(path),size=20,index=found)
            sources.append(dict(row,index=found,cmap=cmap))
        except (ValueError,OSError) as e:excluded.append({'font_id':row['font_id'],'reason':str(e)})
    manifest={'schema':'neural-han-freetype-data-v1','source_manifest_sha256':sha(source_path),
              'split_manifest_sha256':sha(splits_path),'characters':chars,'sizes':sizes,
              'excluded_sources':excluded,'sources':[{k:v for k,v in s.items() if k!='cmap'} for s in sources],'splits':{}}
    for split,characters in chars.items():
        path=output/(split+'.npz')
        if path.exists():raise FileExistsError(path)
        xs=[];ys=[];cs=[];faces=[];ss=[];rejected=[]
        for row in sources:
            for size in sizes[split]:
                font=ImageFont.truetype(row['path'],size=size,index=row['index'])
                for char in characters:
                    if ord(char) not in row['cmap']:
                        rejected.append({'face':row['font_id'],'character':char,'reason':'missing cmap'});continue
                    box=font.getbbox(char);image=Image.new('RGB',(box[2]-box[0]+12,box[3]-box[1]+12),'white')
                    ImageDraw.Draw(image).text((6-box[0],6-box[1]),char,font=font,fill='black')
                    for view in (('identity',) if split=='train' else ('identity','jpeg75')):
                        current=image
                        if view=='jpeg75':
                            buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=75);buffer.seek(0);current=Image.open(buffer).convert('RGB')
                        result=extract_glyphs(current,1)
                        if result.glyphs is None:
                            rejected.append({'face':row['font_id'],'character':char,'reason':result.diagnostics['reason']});continue
                        xs.append(result.glyphs[0]);ys.append(FAMILIES.index(row['family_label']));cs.append(char);faces.append(row['font_id']);ss.append(size)
        output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,x=np.stack(xs),y=np.array(ys,dtype=np.int64),chars=np.array(cs),faces=np.array(faces),sizes=np.array(ss))
        manifest['splits'][split]={'rows':len(xs),'sha256':sha(path),'rejected':rejected}
        print(json.dumps({'han_freetype':split,'rows':len(xs),'sources':len(sources),'excluded':len(excluded)}),flush=True)
    dump(output/'manifest.json',manifest)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--old-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();build(a.old_root,a.output)
