"""Linux CPU pipeline: PP regions and text -> source-ink glyphs -> font references."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib,json,time
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
from .ppocr import PPRegionDetector,PPReader
from .segmentation import segment_characters
from .font_matcher import CompactFontBank,han,rank,score_font
from .models import load_active

ROOT=Path(__file__).resolve().parents[2]
STATUSES={'supported':'支持','candidate':'候选','uncertain':'待确认','out_of_scope':'不支持'}


def save_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n');temp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def annotation(image,regions,path):
    """Draw on an exact-size copy. Original and original crops stay untouched."""
    result=image.copy();draw=ImageDraw.Draw(result)
    size=max(12,min(28,round(image.width*.016)))
    font=ImageFont.truetype(str(ROOT/'assets/annotation.otf'),size)
    colors={'supported':'#14804a','candidate':'#235ab4','uncertain':'#ad6200','out_of_scope':'#6b7280'}
    for region in regions:
        status=region['font']['status'];color=colors[status]
        points=[tuple(p) for p in region['quad']];draw.line(points+[points[0]],fill=color,width=max(2,round(image.width/500)))
        family=region['font'].get('family')
        display='苹方' if family=='PingFang SC' else family or STATUSES[status]
        text=f"{region['id']} {display}"+(' · 候选' if status=='candidate' else '')
        left=min(p[0] for p in points);top=min(p[1] for p in points)
        box=draw.textbbox((0,0),text,font=font);tw,th=box[2]-box[0]+8,box[3]-box[1]+6
        # Keep labels within the original canvas, without resizing the receipt.
        x=max(0,min(left,image.width-tw));y=max(0,top-th)
        draw.rectangle((x,y,min(image.width,x+tw),min(image.height,y+th)),fill=color)
        draw.text((x+4,y+3-box[1]),text,font=font,fill='white')
    result.save(path,'PNG',optimize=False)


class FontPipeline:
    def __init__(self,model_dir=ROOT/'models',cache_characters=32,max_regions=200):
        self.directory,self.version,self.manifest=load_active(model_dir)
        self.detector=PPRegionDetector(self.directory/'pp');self.reader=PPReader(self.directory/'pp')
        self.bank=CompactFontBank(self.directory/'font',cache_characters);self.max_regions=max_regions

    def run(self,source,output,identifier,progress=lambda stage:None):
        def report(code,message,percent=None,current=None,total=None):
            progress({'stage':message,'progress':{'stage_code':code,'percent':percent,
                                                  'current':current,'total':total}})
        report('preparing','正在准备图片')
        started=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=True)
        with Image.open(source) as raw:image=ImageOps.exif_transpose(raw).convert('RGB')
        image.save(output/'original.png')
        report('detecting','正在框选文字区域');mark=time.perf_counter();boxes=self.detector.detect(image);det_seconds=time.perf_counter()-mark
        # Every detection stays visible; cap expensive OCR/glyph work per request.
        # DB polygons sometimes stop at the outer stroke. Add source-pixel context
        # for OCR/segmentation; retain the detector polygon and box separately.
        for box in boxes:
            left,top,right,bottom=box['source_bbox']
            margin=max(2,min(12,round((bottom-top)*.08)))
            box['detector_bbox']=list(box['source_bbox'])
            box['source_bbox']=[max(0,left-margin),max(0,top-margin),
                                min(image.width,right+margin),min(image.height,bottom+margin)]
        selected=boxes[:self.max_regions]
        # Progress counts completed region operations plus annotation/result
        # output steps. It is not an estimate of elapsed or remaining time.
        work_total=len(selected)+len(boxes)+2
        recognized=0
        def percent(done):return min(99,int(100*done/work_total))
        report('recognizing',f'正在识读文字 0/{len(selected)}',0,0,len(selected));mark=time.perf_counter()
        # Preserve PP's aspect-sorted batch order without holding every native
        # ROI in memory. Large/overlapping boxes cannot multiply image memory by
        # the 200-region processing limit. One oversized crop runs on its own.
        def dimensions(index):
            l,t,r,b=selected[index]['source_bbox'];return r-l,b-t
        order=sorted(range(len(selected)),key=lambda i:dimensions(i)[0]/dimensions(i)[1])
        readings=[None]*len(selected);batch=[];pixels=0
        def recognize(indices):
            nonlocal recognized
            crops=[image.crop(selected[i]['source_bbox']) for i in indices]
            for i,reading in zip(indices,self.reader.read(crops)):readings[i]=reading
            recognized+=len(indices)
            report('recognizing',f'正在识读文字 {recognized}/{len(selected)}',percent(recognized),recognized,len(selected))
        for i in order:
            width,height=dimensions(i);area=width*height
            if batch and (len(batch)==6 or pixels+area>4_000_000):
                recognize(batch);batch=[];pixels=0
            batch.append(i);pixels+=area
        if batch:recognize(batch)
        rec_seconds=time.perf_counter()-mark
        regions=[];mark=time.perf_counter();(output/'crops').mkdir();(output/'glyphs').mkdir()
        for index,box in enumerate(boxes):
            report('matching',f'正在匹配字体 {index}/{len(boxes)}',percent(recognized+index),index,len(boxes))
            rid=f'R{index+1:03d}';crop=image.crop(box['source_bbox']);crop_path=output/'crops'/f'{rid}.png';crop.save(crop_path)
            base={'label':'待确认','family':None,'status':'uncertain','reason':'字形证据不足。','candidates':[],
                  'scope':'Chinese glyphs only','font_identity_verified':False}
            region={'id':rid,'quad':box['quad'],'detector_bbox':box['detector_bbox'],'source_bbox':box['source_bbox'],'detector_score':box['score'],
                    'crop_file':crop_path.relative_to(output).as_posix(),'text':'','font':base,'glyphs':[]}
            if index>=len(selected):
                region['font']={**base,'reason':'该图文字区域过多；已保留框，本次超出处理上限。'}
                regions.append(region);continue
            reading=readings[index];text=reading['text'];region['text']=text;region['ocr_confidence']=reading['confidence']
            if not text:
                region['font']={**base,'reason':'文字未可靠读出，或文字行超出长度限制。'};regions.append(region);continue
            if not any(han(c) for c in text):
                region['font']={**base,'status':'out_of_scope','label':'暂不支持','reason':'当前模型识别中文字体，数字和英文保留框。'};regions.append(region);continue
            rotation=reading['metadata']['orientation_degrees']
            oriented=crop.transpose(Image.Transpose.ROTATE_180) if rotation==180 else crop
            segments=segment_characters(oriented,text,reading['tokens'],metadata=reading['metadata'])
            samples=[];glyphs=[];missing=[];all_usable=True
            for char in segments['characters']:
                c=char['character']
                if not han(c):continue
                glyph={'character':c,'index':char['index'],'status':char['status'],'reason':char['reason'],'family_candidate':None,'source_rotation_degrees':rotation}
                if c not in self.bank.entries:
                    glyph['reason']='reference_character_absent';missing.append(c);all_usable=False
                if char['status']=='ok':
                    l,t,r,b=char['bbox'];raw_glyph=oriented.crop((l,t,r,b));name=f'{rid}-{char["index"]:03d}.png'
                    raw_glyph.save(output/'glyphs'/name);glyph['crop_file']='glyphs/'+name
                    source_box=[crop.width-r,crop.height-b,crop.width-l,crop.height-t] if rotation else [l,t,r,b]
                    x0,y0,_,_=box['source_bbox'];glyph['source_bbox']=[source_box[0]+x0,source_box[1]+y0,source_box[2]+x0,source_box[3]+y0]
                    # Add only a constant background border; retain native glyph pixels.
                    a=np.asarray(oriented);bg=tuple(np.median(np.concatenate((a[0],a[-1],a[:,0],a[:,-1])),axis=0).astype(np.uint8).tolist())
                    padded=Image.new('RGB',(raw_glyph.width+8,raw_glyph.height+8),bg);padded.paste(raw_glyph,(4,4))
                    if c in self.bank.entries:
                        match=self.bank.match([{'character':c,'image':padded}]);glyph['family_candidate']=match.get('family_candidate');glyph['candidates']=rank(match)
                        samples.append({'character':c,'image':padded})
                else:all_usable=False
                glyphs.append(glyph)
            complete=all_usable and len(samples)==sum(han(c) for c in text)
            scored=score_font(self.bank,samples,complete)
            all_pf=bool(glyphs) and all(g['family_candidate']=='PingFang SC' for g in glyphs)
            family=scored['family'];status='uncertain';label='待确认';reason='字形证据不足或未通过字体确认门槛。'
            if missing:reason='字库尚未收录：'+'、'.join(sorted(set(missing)))
            elif reading['confidence']<.8:reason='识读文字置信度不足，字体待确认。'
            elif scored['accepted_pingfang'] and all_pf:
                status='supported';family='PingFang SC';label='PingFang SC（苹方）';reason='中文字形通过苹方门槛，逐字候选一致。'
            elif complete and family and family!='PingFang SC':
                status='candidate';label=family+'（候选）';reason='该中文字体在三种图像检查中排名一致，字体名仍为候选。'
            elif not complete:reason='部分汉字无法可靠分字，保留原图区域供复核。'
            region.update(font={**base,'status':status,'label':label,'family':family if status in ('supported','candidate') else None,'reason':reason,'candidates':scored['candidates'],
                                 'unsupported_characters':sorted(set(missing))},glyphs=glyphs,
                          segmentation=segments,ocr=reading,font_evidence=scored)
            regions.append(region)
        match_seconds=time.perf_counter()-mark
        report('annotating','正在生成字体标注图片',percent(recognized+len(boxes)))
        annotation(image,regions,output/'annotated.png')
        counts=Counter(r['font']['status'] for r in regions)
        result={'id':identifier,'width':image.width,'height':image.height,'regions':regions,'model_version':self.version,
                'summary':{'detected_regions':len(regions),'chinese_regions':sum(any(han(c) for c in r['text']) for r in regions),
                           'pingfang_supported':counts['supported'],'other_candidates':counts['candidate'],'uncertain':counts['uncertain'],'out_of_scope':counts['out_of_scope'],
                           'processing_limited_regions':max(0,len(regions)-self.max_regions)},
                'timing_seconds':{'detector':det_seconds,'ocr':rec_seconds,'font_matching':match_seconds,'total':time.perf_counter()-started},
                'source_sha256':sha(source),'font_scope':'Chinese font family candidates; 986 known Han characters; numeric/Latin fonts unsupported',
                'pipeline':'PP DB + PP REC + original-pixel ink character localization + frozen font references',
                'font_identity_verified':False,'device_inference_performed':False,
                'linux_segmentation_accuracy_validated':False,'font_cache_bytes':self.bank.cache_bytes}
        save_json(output/'result.json',result)
        report('finalizing','正在整理识别结果',percent(work_total-1))
        return result
