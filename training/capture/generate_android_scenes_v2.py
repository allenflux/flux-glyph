#!/usr/bin/env python3
"""Second independent Android capture: fresh text and script-balanced negatives.

Only prior label metadata is read. No prior or new TEST pixels are opened here.
The original collector, native renderer, and first-round scenes remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import string
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_android import dump, read, require, sha, validate_scenes

SEED = 2026091303
SPLITS = {'train': 1000, 'calibration': 150, 'test': 150}
SIZES = [16, 17, 18, 19, 20, 21, 22, 24, 26, 28, 30, 32, 36, 40, 48, 56, 64, 72, 84]
PALETTES = [('#FFFFFF', '#000000'), ('#FFFFFF', '#333333'), ('#F5F5F5', '#555555'),
            ('#111111', '#FFFFFF'), ('#F5F1E8', '#63543B'), ('#FFFFFF', '#1677FF')]
HAN = '账单详情全部交易成功支付时间金额订单商家服务管理设置消息首页收藏搜索登录账户余额收入支出确认取消返回活动记录中心安全优惠提醒状态完成退款收款转账红包通知选择更新帮助我的朋友客户快速查看方式打开关闭历史会员今日消费充值明细商店城市银行付款查询客服内容显示名称号码介绍应用字体数字正文标题北京上海广州深圳杭州南京武汉成都西安长沙苏州郑州重庆厦门天津青岛昆明南宁福州合肥济南南昌沈阳大连太原兰州贵阳海口长春哈尔滨温州宁波无锡常州东莞佛山珠海泉州湛江惠州中山四五六七八九十百千万上下左右天地人和轻重大小春夏秋冬新旧实用清晰简洁系统自动选择待处理已接收用户资料隐私政策帮助反馈网络连接设备语言版本密码验证重新加载正在读取银行卡电子发票物流配送地址购买商品最近浏览时间日期订单状态余额明细当前页面其他更多关于退出账户'
PHRASES = ['账单详情', '全部账单', '交易成功', '支付时间', '订单编号', '消息通知', '确认付款',
           '收款金额', '充值记录', '收入支出', '取消订单', '联系商家', '订单状态', '付款成功',
           '我的订单', '当前余额', '最近交易', '支付设置', '交易详情', '查看订单', '到账时间',
           '返回列表', '收款账户', '付款账户', '客服帮助', '用户资料', '网络连接', '设备信息',
           '语言设置', '版本更新', '正在加载', '重新打开', '隐私政策', '安全验证', '修改密码',
           '电子发票', '配送地址', '购买商品', '最近浏览', '其他服务', '更多消息', '系统设置']
ENGLISH = ['PayNow', 'Payment', 'Balance', 'Settings', 'Details', 'Transfer', 'Success', 'Cancel',
           'Continue', 'Confirm', 'Refund', 'History', 'Amount', 'Receive', 'Account', 'Message',
           'Search', 'Loading', 'Updated', 'Complete', 'Next', 'Done', 'More', 'Back']


def normalize_text(value):
    """Match preparation's stronger identity: compatibility/case/space variants."""
    return ''.join(unicodedata.normalize('NFKC', value).casefold().split())


def prior_texts(path):
    texts = set()
    partitions = Counter()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            require(row.get('split') in SPLITS and isinstance(row.get('text'), str) and row['text'],
                    'prior label must contain text and a valid partition')
            texts.add(normalize_text(row['text']))
            partitions[row['split']] += 1
    require(set(partitions) == set(SPLITS), 'prior exclusions must include all three partitions')
    return texts, dict(partitions)


def text_partition(value):
    number = int(hashlib.sha256(normalize_text(value).encode()).hexdigest()[:8], 16) % 26
    return 'train' if number < 20 else 'calibration' if number < 23 else 'test'


class TextSampler:
    def __init__(self, rng, excluded):
        self.rng, self.excluded, self.owners = rng, excluded, {}

    def sample(self, split, alphabet, *, latin=False):
        rng = self.rng
        phrases = ENGLISH if latin else PHRASES
        choices = [text for text in phrases if text_partition(text) == split and
                   normalize_text(text) not in self.excluded and set(text) <= set(alphabet)]
        for _ in range(10000):
            if latin:
                style = rng.randrange(7)
                if style == 0:
                    value = f'{"-" if rng.randrange(2) else ""}{rng.randrange(1, 100000)}.{rng.randrange(100):02d}'
                elif style == 1:
                    value = f'{rng.randrange(2018, 2036):04d}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}'
                elif style == 2:
                    value = f'{rng.randrange(24):02d}:{rng.randrange(60):02d}:{rng.randrange(60):02d}'
                elif style == 3 and choices:
                    value = rng.choice(choices)
                elif style == 4:
                    value = ''.join(rng.choices(string.digits, k=rng.randrange(3, 11)))
                else:
                    value = ''.join(rng.choices(string.ascii_letters + string.digits, k=rng.randrange(3, 11)))
            else:
                value = rng.choice(choices) if choices and rng.random() < .35 else ''.join(
                    rng.choices(alphabet, k=rng.randrange(3, 9)))
            identity = normalize_text(value)
            if (set(value) <= set(alphabet) and identity not in self.excluded and
                    self.owners.get(identity, split) == split):
                self.owners[identity] = split
                return value
        raise ValueError('could not sample fresh supported text')


class FamilyCycle:
    """Balance source families before cycling their individual face files."""
    def __init__(self, fonts):
        self.groups = defaultdict(list)
        for font in fonts:
            self.groups[font['family']].append(font)
        require(bool(self.groups), 'negative script pool is empty')
        self.families = sorted(self.groups)
        self.index = 0
        self.counts = Counter()

    def next(self):
        family = self.families[self.index % len(self.families)]
        faces = self.groups[family]
        font = faces[self.counts[family] % len(faces)]
        self.index += 1
        self.counts[family] += 1
        return font


def split_alphabets(allowed, maps):
    han_fonts = [font for font in allowed if 'han' in font['scripts']]
    require(bool(han_fonts), 'no Han-capable font source')
    han_common = set.intersection(*(maps[font['id']] for font in han_fonts))
    han = ''.join(dict.fromkeys(ch for ch in HAN if ord(ch) in han_common))
    require(len(han) >= 32, 'insufficient shared native Han coverage')
    latin_common = set.intersection(*(maps[font['id']] for font in allowed))
    latin = ''.join(ch for ch in string.ascii_letters + string.digits + '-.:/' if ord(ch) in latin_common)
    require(set(string.ascii_letters + string.digits) <= set(latin), 'incomplete native Latin coverage')
    return han, latin


def build_scenes(registry, maps, excluded, bindings, split_pages=None):
    split_pages = SPLITS if split_pages is None else split_pages
    require(set(split_pages) == set(SPLITS) and all(type(n) is int and n > 0 for n in split_pages.values()),
            'three positive page counts required')
    require(registry['schema'] == 'flux-glyph-android-font-sources-v1', 'source schema differs')
    known, fonts = registry['families'], registry['fonts']
    require(len(known) == 9 and known[-1] == 'Roboto', 'known family order differs')
    rng = random.Random(SEED)
    sampler = TextSampler(rng, excluded)
    pages, coverage = [], {}
    for split, count in split_pages.items():
        allowed = [font for font in fonts if split in font['allowed_splits']]
        han, latin = split_alphabets(allowed, maps)
        coverage[split] = {'shared_han_alphabet': han, 'shared_latin_alphabet': latin}
        negative_han = FamilyCycle([font for font in allowed if font['training_family'] == '__unknown__'
                                    and 'han' in font['scripts']])
        negative_latin = FamilyCycle([font for font in allowed if font['training_family'] == '__unknown__'
                                      and 'han' not in font['scripts'] and 'latin' in font['scripts']])
        face_counts = Counter()
        for index in range(count):
            page_id = f'android-native-v2-{split}-{index:05d}'
            size = SIZES[index % len(SIZES)]
            background, color = PALETTES[(index // len(SIZES)) % len(PALETTES)]
            # Use the size index, not the absolute index: 19 + 1 is divisible
            # by four and would otherwise permanently tie each size to a script.
            latin_page = (index // len(SIZES) + index % len(SIZES)) % 4 == 0
            chinese_text = sampler.sample(split, han)
            latin_text = sampler.sample(split, latin, latin=True)
            entries = []
            for class_index, family in enumerate(known):
                faces = [font for font in allowed if font['training_family'] == family]
                require(bool(faces), 'missing known family')
                use_latin = latin_page or family == 'Roboto'
                face_key = (family, size, use_latin)
                face = faces[(face_counts[face_key] + index % len(SIZES) + class_index) % len(faces)]
                face_counts[face_key] += 1
                require(use_latin or 'han' in face['scripts'], 'known Han row has a Latin-only font')
                entries.append((face, latin_text if use_latin else chinese_text, use_latin))
            for slot in range(3):
                latin_only = index % 3 == 0 and slot == 0
                font = (negative_latin if latin_only else negative_han).next()
                use_latin = latin_only or latin_page
                entries.append((font, latin_text if use_latin else chinese_text, use_latin))
            rng.shuffle(entries)
            regions = []
            for slot, (font, text, use_latin) in enumerate(entries):
                require(all(ord(ch) in maps[font['id']] for ch in text), 'scene text requests fallback')
                top = 40 + slot * 190
                regions.append({'id': page_id + f'-r{slot:02d}', 'font_id': font['id'], 'text': text,
                    'script': ('numeric' if not any(ch.isalpha() for ch in text) else 'latin') if use_latin else 'han',
                    'font_size_px': size, 'color': color, 'bbox': [52, top, 1028, top + 170]})
            pages.append({'id': page_id, 'content_group_id': page_id, 'split': split,
                          'background': background, 'regions': regions})
    return {'schema': 'flux-glyph-android-scenes-v1', 'seed': SEED, 'canvas_px': [1080, 2400],
        'families': known + ['__unknown__'], 'fonts': fonts, 'pages': pages, 'bindings': bindings,
        'design': {'protocol': 'flux-glyph-android-native-round-two-v1', 'split_pages': split_pages,
            'rows_per_page': 12, 'paired_known_size_palette_script_and_text': True,
            'font_labels_rendered': False, 'prior_all_partition_text_excluded': True,
            'text_identity': 'NFKC, casefold, remove whitespace', 'text_split_overlap': False,
            'unknown_font_sources_disjoint': True, 'unknown_sampling': 'round-robin family then face',
            'unknown_latin_only_slots': 'one of three slots every third page; otherwise Han-capable source',
            'han_capable_latin_schedule': '(page_index // 19 + page_index % 19) % 4 == 0',
            'known_face_sampling': 'cycle faces independently within each family, size, and script',
            'sizes_px': SIZES, 'palettes': PALETTES, 'coverage': coverage,
            'source_rendering': 'Android native only', 'old_test_reused_for_development': True,
            'old_test_remains_failed': True, 'test_pixels_read_for_scene_generation': False}}


def validate_round(scenes, maps, excluded):
    fonts = validate_scenes(scenes)
    owners = {}
    for page in scenes['pages']:
        require(len(page['regions']) == 12, 'round requires twelve rows per page')
        classes = Counter(fonts[row['font_id']]['training_family'] for row in page['regions'])
        require(classes == Counter({**{family: 1 for family in scenes['families'][:-1]}, '__unknown__': 3}),
                'round class counts differ')
        require(len({(row['font_size_px'], row['color']) for row in page['regions']}) == 1,
                'font size and palette must be paired')
        for row in page['regions']:
            identity = normalize_text(row['text'])
            require(identity not in excluded, 'prior partition text reused')
            require(owners.setdefault(identity, page['split']) == page['split'], 'new text crosses partitions')
            require(all(ord(ch) in maps[row['font_id']] for ch in row['text']), 'unsupported source glyph')
            require(row['script'] != 'han' or 'han' in fonts[row['font_id']]['scripts'],
                    'Latin-only font received Han text')


def main(args):
    require(not args.output.exists(), 'scene output must be new')
    registry = read(args.fonts)
    excluded, prior_counts = prior_texts(args.exclude_labels)
    bindings = {str(path.resolve()): sha(path) for path in (args.fonts, args.exclude_labels, Path(__file__))}
    maps = {}
    for font in registry['fonts']:
        descriptor = font['cmap']
        require(sha(descriptor['path']) == descriptor['sha256'], 'source cmap changed')
        maps[font['id']] = set(read(descriptor['path']))
        bindings[str(Path(descriptor['path']).resolve())] = descriptor['sha256']
    scenes = build_scenes(registry, maps, excluded, bindings)
    scenes['design']['prior_exclusion'] = {'labels': str(args.exclude_labels.resolve()),
        'sha256': bindings[str(args.exclude_labels.resolve())], 'rows_by_split': prior_counts,
        'unique_normalized_texts': len(excluded)}
    validate_round(scenes, maps, excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(args.output, scenes)
    print(json.dumps({'scenes': str(args.output), 'sha256': sha(args.output), 'pages': len(scenes['pages']),
                      'regions': sum(len(page['regions']) for page in scenes['pages']), 'faces': len(registry['fonts'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fonts', type=Path, required=True)
    parser.add_argument('--exclude-labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
