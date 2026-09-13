#!/usr/bin/env python3
"""Paired, split-disjoint Android native font scenes; no font labels in pixels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import string
import sys
import unicodedata

sys.path.insert(0,str(Path(__file__).resolve().parent))
from capture_android import dump,read,require,sha,validate_scenes

SEED=2026091302
SPLITS={'train':600,'calibration':100,'test':100}
SIZES=[16,18,20,24,28,32,40,48,56,64,72,84]
PALETTES=[('#FFFFFF','#000000'),('#FFFFFF','#333333'),('#F5F5F5','#555555'),
          ('#111111','#FFFFFF'),('#F5F1E8','#63543B'),('#FFFFFF','#1677FF')]
HAN='账单详情全部交易成功支付时间金额订单商家服务管理设置消息首页收藏搜索登录账户余额收入支出确认取消返回活动记录中心安全优惠提醒状态完成退款收款转账红包通知选择更新帮助我的朋友客户快速查看方式打开关闭历史会员今日消费充值明细余额商店城市银行付款查询客服内容显示名称号码介绍应用字体数字正文标题北京上海广州深圳杭州南京武汉成都西安长沙苏州郑州重庆厦门天津青岛昆明南宁福州合肥济南南昌沈阳大连太原兰州贵阳海口长春哈尔滨温州宁波无锡常州东莞佛山珠海泉州湛江惠州中山四五六七八九十百千万上下左右天地人和轻重大小春夏秋冬新旧实用清晰简洁系统自动选择'
PHRASES=['账单详情','全部账单','交易成功','支付时间','支付方式','订单编号','查看明细','返回首页','账户余额','消息通知','确认付款','收款金额',
         '服务管理','设置字体','申请退款','我的收藏','安全中心','充值记录','收入支出','优惠活动','打开应用','取消订单','联系商家','搜索内容',
         '历史记录','更新时间','选择城市','今日消费','客户服务','全部消息']


def text_split(text):
    number=int(hashlib.sha256(text.encode()).hexdigest()[:8],16)%8
    return 'train' if number<6 else 'calibration' if number==6 else 'test'


def main(args):
    require(not args.output.exists(),'scene file must be new')
    registry=read(args.fonts);require(registry['schema']=='flux-glyph-android-font-sources-v1','source schema differs')
    fonts=registry['fonts'];known=registry['families'];require(len(known)==9,'nine known families required')
    bindings={str(args.fonts.resolve()):sha(args.fonts),str(Path(__file__).resolve()):sha(__file__)}
    maps={}
    for font in fonts:
        descriptor=font['cmap'];require(sha(descriptor['path'])==descriptor['sha256'],'source cmap changed')
        maps[font['id']]=set(read(descriptor['path']));bindings[str(Path(descriptor['path']).resolve())]=descriptor['sha256']
    rng=random.Random(SEED);pages=[];seen={}
    def unique_text(split,alphabet,latin=False):
        while True:
            if latin:
                choice=rng.randrange(3)
                if choice==0: value=''.join(rng.choices(string.digits,k=rng.randrange(4,11)))
                elif choice==1: value=''.join(rng.choices(string.ascii_letters,k=rng.randrange(3,10)))
                else: value=''.join(rng.choices(string.ascii_letters+string.digits,k=rng.randrange(4,12)))
            else:
                choices=[text for text in PHRASES if text_split(text)==split and all(ch in alphabet for ch in text)]
                value=rng.choice(choices) if choices and rng.random()<.35 else ''.join(rng.choices(alphabet,k=rng.randrange(3,9)))
            normalized=unicodedata.normalize('NFKC',value).casefold()
            if all(ch in alphabet for ch in value) and seen.setdefault(normalized,split)==split:
                return value
    for split,page_count in SPLITS.items():
        allowed=[font for font in fonts if split in font['allowed_splits']]
        han_maps=[maps[font['id']] for font in allowed if font['family']!='Roboto']
        common=set.intersection(*han_maps)
        han=''.join(dict.fromkeys(ch for ch in HAN if ord(ch) in common))
        require(len(han)>100,'insufficient common CJK coverage')
        latin=string.ascii_letters+string.digits
        require(all(all(ord(ch) in maps[font['id']] for ch in latin) for font in allowed),'Latin scene coverage is incomplete')
        for page_index in range(page_count):
            page_id=f'android-native-v1-{split}-{page_index:05d}'
            size=SIZES[page_index%len(SIZES)];background,color=PALETTES[(page_index//len(SIZES))%len(PALETTES)]
            latin_page=(page_index//len(SIZES)+page_index)%4==0
            chinese_text=unique_text(split,han);latin_text=unique_text(split,latin,latin=True)
            entries=[]
            for class_index,family in enumerate(known):
                faces=[font for font in allowed if font['training_family']==family]
                face=faces[(page_index//len(SIZES)+page_index+class_index)%len(faces)]
                use_latin=latin_page or family=='Roboto'
                text=latin_text if use_latin else chinese_text
                entries.append((face,text,'numeric' if text.isdecimal() else 'latin' if use_latin else 'han'))
            negatives=[font for font in allowed if font['training_family']=='__unknown__']
            for negative_index in range(3):
                face=negatives[(page_index//len(SIZES)+page_index+negative_index)%len(negatives)]
                text=latin_text if latin_page else chinese_text
                entries.append((face,text,'numeric' if text.isdecimal() else 'latin' if latin_page else 'han'))
            rng.shuffle(entries)
            regions=[]
            for slot,(font,text,script) in enumerate(entries):
                require(all(ord(ch) in maps[font['id']] for ch in text),'scene text requests fallback')
                top=40+slot*190
                regions.append({'id':page_id+f'-r{slot:02d}','font_id':font['id'],'text':text,'script':script,
                    'font_size_px':size,'color':color,'bbox':[52,top,1028,top+170]})
            pages.append({'id':page_id,'content_group_id':page_id,'split':split,'background':background,'regions':regions})
    scenes={'schema':'flux-glyph-android-scenes-v1','seed':SEED,'canvas_px':[1080,2400],
        'families':known+['__unknown__'],'fonts':fonts,'pages':pages,'bindings':bindings,
        'design':{'split_pages':SPLITS,'rows_per_page':12,'paired_size_palette_and_text_across_families':True,
            'font_labels_rendered':False,'text_split_overlap':False,'unknown_font_sources_disjoint':True,
            'sizes_px':SIZES,'palettes':PALETTES,'source_rendering':'Android native only'}}
    validate_scenes(scenes)
    args.output.parent.mkdir(parents=True,exist_ok=True);dump(args.output,scenes)
    print(json.dumps({'scenes':str(args.output),'sha256':sha(args.output),'pages':len(pages),'regions':sum(len(p['regions']) for p in pages),'faces':len(fonts)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fonts',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
