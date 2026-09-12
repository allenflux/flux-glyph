#!/usr/bin/env python3
"""Reproduce candidate-only validation on new, hash-verified source glyphs.

The text, size and palette cross-product was declared before the holdout run.
None of these strings or native sizes is in the frozen 36-image benchmark.
Source fonts are the same identities, so this measures rasterization robustness
on controlled oracle crops, not real Android screenshot accuracy or OCR quality.
Source font files must be available as declared by the original fixture manifest.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont

from accuracy_eval import ROOT,CASES,font_sources
from flux_glyph.font_matcher import CompactFontBank,score_font,views


TEXTS=('支付宝账户','交易信息查询','银行卡管理','支付服务安全','手机号码验证','生活缴费中心')
SIZES=(19,23,29,35,41,47)
PALETTES=(((255,255,255),(30,30,30)),((20,100,210),(255,255,255)))
OUTPUT=ROOT/'docs/android-alignment-validation.json'


def legacy_family(bank,samples):
    variants={key:[] for key in ('identity','jpeg75','resize80_restore')}
    for sample in samples:
        for key,image in views(sample['image']).items():
            variants[key].append({'character':sample['character'],'image':image})
    matches=[bank.match(xs) for xs in variants.values()]
    if (all(m.get('status')=='ok' and len(m.get('leading_family_ties',[]))==1 for m in matches)
            and len({m['family_candidate'] for m in matches})==1):
        return matches[0]['family_candidate']
    return None


def evaluate():
    sources=font_sources(json.loads(CASES.read_text()))
    bank=CompactFontBank(ROOT/'models/font',128)
    counts=defaultdict(lambda:defaultdict(lambda:dict(total=0,correct=0,wrong=0,abstention=0)))
    rows=[]
    for font_id,source in sources.items():
        for text in TEXTS:
            for size in SIZES:
                face=ImageFont.truetype(source['path'],size,index=source['face_index'])
                for palette,(background,foreground) in enumerate(PALETTES):
                    samples=[];digest=hashlib.sha256()
                    for char in text:
                        if char not in bank.entries:
                            raise ValueError('Holdout character absent from references: '+char)
                        image=Image.new('RGB',(size+16,size+20),background)
                        ImageDraw.Draw(image).text((8,4),char,font=face,fill=foreground)
                        digest.update(image.tobytes())
                        samples.append({'character':char,'image':image})
                    predictions={'legacy':legacy_family(bank,samples),
                                 'aligned':score_font(bank,samples,True)['family']}
                    for method,prediction in predictions.items():
                        count=counts[method][source['expected_family']]
                        count['total']+=1
                        count['correct']+=prediction==source['expected_family']
                        count['wrong']+=prediction is not None and prediction!=source['expected_family']
                        count['abstention']+=prediction is None
                    rows.append({'font_id':font_id,'expected_family':source['expected_family'],
                                 'text':text,'size':size,'palette':palette,'glyph_pixels_sha256':digest.hexdigest(),
                                 'predictions':predictions})
        print(font_id,dict(counts['legacy'][source['expected_family']]),
              dict(counts['aligned'][source['expected_family']]),flush=True)
    bank.archive.close()
    report={'schema':'flux-glyph-half-pixel-candidate-validation-v1',
            'evaluated_at_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'controlled oracle glyph crops only; not real screenshots or end-to-end accuracy',
            'frozen_screenshot_inputs_modified':False,
            'candidate_decision':'Complete crops and unique same-family winner across all three image views; abstention is incorrect.',
            'pingfang_gates_modified':False,'texts':TEXTS,'sizes':SIZES,'palettes':PALETTES,
            'font_sources':[{key:source[key] for key in ('font_id','expected_family','source_sha256','postscript_name')}
                            for source in sources.values()],
            'font_matcher_sha256':hashlib.sha256((ROOT/'src/flux_glyph/font_matcher.py').read_bytes()).hexdigest(),
            'counts':dict(counts),'cases':rows}
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


if __name__=='__main__':
    evaluate()
