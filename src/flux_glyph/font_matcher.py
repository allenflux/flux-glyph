"""Compact, bounded-memory implementation of the frozen R12/R13 glyph matcher."""
from __future__ import annotations
import hashlib
import io
import json
import zipfile
from collections import OrderedDict,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
from .foreground import normalize_foreground
from .glyph_preprocess import extract_glyphs
from .models import file_sha


def han(c):
    return isinstance(c,str) and len(c)==1 and '\u4e00'<=c<='\u9fff'


def raster(image):
    foreground,diagnostic=normalize_foreground(image)
    if foreground is None:
        return None
    extracted=extract_glyphs(foreground,1)
    if extracted.glyphs is None or not extracted.diagnostics.get('ok') or extracted.glyphs.shape!=(1,64,64):
        return None
    a=extracted.glyphs[0].astype(np.float64)
    return np.asarray(Image.fromarray(np.clip(np.rint(a*255),0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1)).resize((32,32),Image.Resampling.BICUBIC),dtype=np.uint8)


def normalize_raster(a):
    z=a.astype(np.float64).reshape(-1)/255.
    n=np.sqrt(np.einsum('i,i->',z,z,optimize=False))
    return z/n if np.isfinite(n) and n>1e-12 else None


def vector(image):
    a=raster(image)
    return normalize_raster(a) if a is not None else None


def _load_reference_member(archive,name):
    try:
        info=archive.getinfo(name)
    except KeyError as error:
        raise ValueError('Font archive lacks declared reference member: '+name) from error
    if info.is_dir() or info.file_size>100000:
        raise ValueError('Reference member exceeds bounded shape: '+name)
    try:
        value=np.load(io.BytesIO(archive.read(info)),allow_pickle=False)
    except Exception as error:
        raise ValueError('Invalid reference raster: '+name) from error
    if value.shape!=(31,3,32,32) or value.dtype!=np.uint8:
        raise ValueError('Invalid reference raster shape: '+name)
    # A zero raster cannot produce the normalized vector required by matching.
    # Check this cheaply here; float64 normalization remains lazy at query time.
    if not np.all(np.any(value,axis=(2,3))):
        raise ValueError('Reference raster contains an empty vector: '+name)
    return value


def _validate_gates(value):
    if not isinstance(value,dict) or value.get('method')!='blur32' or not isinstance(value.get('gates'),dict):
        raise ValueError('Unsupported font gate contract')
    gates=value['gates']
    if set(gates)!= {'1','2','4'} or gates['1'] is not None:
        raise ValueError('Unsupported font gate registry')
    for count in ('2','4'):
        gate=gates[count]
        if not isinstance(gate,dict) or set(gate)!= {'max_distance','min_margin'}:
            raise ValueError('Invalid font gate: '+count)
        for key in ('max_distance','min_margin'):
            number=gate[key]
            if isinstance(number,bool) or not isinstance(number,(int,float)) or not np.isfinite(number) or not 0<number<=2:
                raise ValueError(f'Invalid font gate value: {count}.{key}')
    return gates


def _validate_registry(meta):
    if (meta.get('schema')!='flux-glyph-r13-compact-v1' or
            meta.get('algorithm_version')!='r12-normalized-glyph_blur1_resize32_uint8' or
            meta.get('shape')!=[31,3,32,32] or meta.get('dtype')!='uint8'):
        raise ValueError('Unsupported font archive contract')
    font_ids=meta.get('font_ids');face_family=meta.get('face_family');entries=meta.get('characters')
    if (not isinstance(font_ids,list) or len(font_ids)!=31 or len(set(font_ids))!=31 or
            not all(isinstance(value,str) and value for value in font_ids)):
        raise ValueError('Font ID registry mismatch')
    if (not isinstance(face_family,dict) or set(face_family)!=set(font_ids) or
            not all(isinstance(value,str) and value for value in face_family.values()) or
            len(set(face_family.values()))!=11 or 'PingFang SC' not in face_family.values()):
        raise ValueError('Font family registry mismatch')
    if (not isinstance(entries,dict) or not entries or len(entries)>10000 or
            not all(han(char) for char in entries) or
            not all(isinstance(name,str) and Path(name).name==name and name.endswith('.npy') for name in entries.values()) or
            len(set(entries.values()))!=len(entries)):
        raise ValueError('Font character registry mismatch')
    return font_ids,face_family,entries


class CompactFontBank:
    def __init__(self,directory,max_characters=32):
        self.directory=Path(directory)
        self.meta=json.loads((self.directory/'metadata.json').read_text())
        self.font_ids,self.face_family,self.entries=_validate_registry(self.meta)
        self.families=sorted(set(self.face_family.values()));self.cache=OrderedDict();self.max_characters=max(1,min(128,max_characters));self.loads=0
        archive_name=self.meta.get('archive');archive=self.directory/archive_name if isinstance(archive_name,str) else self.directory/'__invalid__'
        if (not isinstance(archive_name,str) or Path(archive_name).name!=archive_name or
                not archive.resolve().is_relative_to(self.directory.resolve()) or not archive.is_file() or
                file_sha(archive)!=self.meta.get('archive_sha256')):
            raise ValueError('Font archive checksum mismatch')
        if archive.stat().st_size>256*1024*1024:
            raise ValueError('Font archive exceeds size limit')
        opened=zipfile.ZipFile(archive)
        try:
            members=opened.infolist();names=[item.filename for item in members]
            if (len(names)!=len(set(names)) or set(names)!=set(self.entries.values()) or
                    sum(item.file_size for item in members)>256*1024*1024):
                raise ValueError('Font archive member registry mismatch')
            for name in self.entries.values():
                _load_reference_member(opened,name)
            self.gates=_validate_gates(json.loads((self.directory/'GATES.json').read_text()))
        except Exception:
            opened.close();raise
        self.archive=opened

    @property
    def cache_bytes(self):
        return sum(a.nbytes for a in self.cache.values())

    def references(self,char):
        if char not in self.entries:
            return None
        if char in self.cache:
            self.cache.move_to_end(char);return self.cache[char]
        while len(self.cache)>=self.max_characters:
            self.cache.popitem(last=False)
        name=self.entries[char];a=_load_reference_member(self.archive,name)
        # Per-vector reduction order matches the previous reference implementation.
        vectors=np.stack([normalize_raster(x) for x in a.reshape(-1,32,32)]).reshape(31,3,1024)
        if not np.isfinite(vectors).all():
            raise ValueError('Non-finite reference vector')
        self.cache[char]=vectors;self.loads+=1
        return vectors

    def distances(self,char,z):
        if z is None:
            return None
        refs=self.references(char)
        if refs is None:
            return None
        return np.asarray([float(np.min(1-np.einsum('ij,j->i',face,z,optimize=False))) for face in refs])

    def match(self,samples):
        if not samples:
            return {'status':'failed','reason':'no_samples'}
        by=defaultdict(list)
        for sample in samples:
            c=sample['character'];distance=self.distances(c,vector(sample['image']))
            if distance is None:
                return {'status':'failed','reason':'bad_query_or_missing_reference'}
            by[c].append(distance)
        per={c:np.mean(np.stack(v),0) for c,v in by.items()}
        score=np.mean(np.stack(list(per.values())),0)
        family_scores={f:float(np.min(score[[i for i,id in enumerate(self.font_ids) if self.face_family[id]==f]])) for f in self.families}
        ordered=sorted(family_scores,key=lambda f:(family_scores[f],f));best=family_scores[ordered[0]]
        ties=[f for f in ordered if family_scores[f]-best<=1e-9]
        return {'status':'ok','family_scores':family_scores,'face_distances':score.tolist(),
                'family_candidate':ordered[0] if len(ties)==1 else None,'leading_family_ties':ties,
                'margin_to_next_family':None if len(ties)==len(ordered) else family_scores[ordered[len(ties)]]-best}


def views(image):
    image=image.convert('RGB');b=io.BytesIO();image.save(b,'JPEG',quality=75,subsampling=0,optimize=False,progressive=False)
    small=image.resize((max(4,round(image.width*.8)),max(4,round(image.height*.8))),Image.Resampling.BICUBIC)
    return {'identity':image,'jpeg75':Image.open(io.BytesIO(b.getvalue())).convert('RGB'),
            'resize80_restore':small.resize(image.size,Image.Resampling.BICUBIC)}


def rank(match):
    return [{'family':f,'distance':d} for f,d in sorted(match.get('family_scores',{}).items(),key=lambda x:(x[1],x[0]))[:3]]


def score_font(bank,samples,complete):
    distinct=sorted({x['character'] for x in samples});n=1 if len(distinct)==1 else 2 if len(distinct) in (2,3) else 4 if distinct else 0
    subset=distinct[:n];variants={key:[] for key in ('identity','jpeg75','resize80_restore')}
    for sample in samples:
        for key,image in views(sample['image']).items():variants[key].append({'character':sample['character'],'image':image})
    full={key:bank.match(xs) for key,xs in variants.items()};sub={key:bank.match([s for s in xs if s['character'] in subset]) for key,xs in variants.items()}
    gate=bank.gates.get(str(n));pf='PingFang SC'
    def unique(m):return m.get('status')=='ok' and m.get('leading_family_ties')==[pf]
    accepted=bool(complete and gate and all(unique(full[k]) and unique(sub[k]) and sub[k]['family_scores'][pf]<=gate['max_distance'] and sub[k]['margin_to_next_family']>=gate['min_margin'] for k in variants))
    stable=bool(complete and all(v.get('status')=='ok' and len(v.get('leading_family_ties',[]))==1 for v in full.values()) and len({v.get('family_candidate') for v in full.values()})==1)
    return {'accepted_pingfang':accepted,'family':full['identity'].get('family_candidate') if stable else None,
            'candidates':rank(full['identity']),'views':full,'subset_views':sub,'complete':complete,'gate':gate}
