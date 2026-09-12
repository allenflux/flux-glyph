"""Generate reproducible, labelled *requests* for native screen capture.

The JSON is not a screenshot dataset or font truth. Collectors must confirm the
font used by every rendered run/glyph and capture the real OS screen. Run with
the development environment (fontTools is only needed to audit asset sources).
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (ROOT.parent / 'alipay-ai-inference/runs/'
                    'alipay-font-generalization-v8-20260911/data/fonts.json')
SCHEMA = 'flux-glyph-capture-scenes-v1'
SPLITS = ('train', 'calibration', 'test')
ANDROID_FAMILIES = {'HarmonyOS Sans SC', 'MiSans', 'Noto Sans CJK SC', 'OPPO Sans'}
LATIN_FAMILIES = {'HarmonyOS Sans SC', 'MiSans', 'OPPO Sans', 'SF Pro',
                  'Helvetica', 'Alipay Number', 'Roboto'}
ROBOTO_SHA = 'd7598e12c5dbef095ff8272cfc55da0250bd07fbdecbac8a530b9b277872a134'
ALIPAY_SHA = '6074082d8cb92e175184b177e28335e0171f3ddfb2b5818d7853295f7fb0fada'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def families():
    tree = ast.parse((ROOT / 'training/network.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FAMILIES' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing literal training/network.py FAMILIES')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def prepare_assets(output, manifest=DEFAULT_MANIFEST, platform='android', include_ios_cjk=True):
    from fontTools.ttLib import TTFont
    records = json.loads(Path(manifest).read_text())
    records = ([dict(row) for row in records if row['family_label'] in ANDROID_FAMILIES]
               if platform == 'android' or include_ios_cjk else [])
    roboto = ROOT / 'artifacts/neural-font-v1/downloads/roboto-google-fonts/Roboto[wdth,wght].ttf'
    if platform == 'android':
        records.append({'font_id': 'roboto_variable', 'family_label': 'Roboto', 'path': str(roboto),
                        'postscript_name': 'Roboto-Regular', 'source_sha256': ROBOTO_SHA})
    else:
        alipay = Path(manifest).parents[2] / 'alipay-font-real-fields-v9-20260911/discovery/AlipayNumber-Regular.ttf'
        records.append({'font_id': 'alipay_number_regular', 'family_label': 'Alipay Number', 'path': str(alipay),
                        'postscript_name': 'AlipayNumber-Regular', 'source_sha256': ALIPAY_SHA})
    result = []
    for record in records:
        source = Path(record['path'])
        if sha(source) != record['source_sha256']:
            raise ValueError(f'Font source SHA mismatch: {source}')
        font = TTFont(source, lazy=False)
        postscript = font['name'].getDebugName(6)
        if postscript != record['postscript_name']:
            raise ValueError(f'Font PostScript mismatch: {source}: {postscript}')
        cmap = font.getBestCmap() or {}
        weight = int(font['OS/2'].usWeightClass)
        axes = ({axis.axisTag: [axis.minValue, axis.defaultValue, axis.maxValue]
                 for axis in font['fvar'].axes} if 'fvar' in font else {})
        font.close()
        target = output / 'assets' / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) != record['source_sha256']:
            raise ValueError(f'Refusing to replace different asset: {target}')
        if not target.exists():
            shutil.copyfile(source, target)
        if sha(target) != record['source_sha256']:
            raise ValueError(f'Copied asset SHA mismatch: {target}')
        family = record['family_label']
        result.append({'id': record['font_id'], 'family': family, 'postscript': postscript,
                       'kind': 'asset', 'path': str(target.resolve()), 'sha256': sha(target),
                       'source_path': str(source.resolve()),
                       'source_manifest': (str(roboto.parent / 'SOURCE_MANIFEST.json') if family == 'Roboto' else
                                           str(ROOT / 'scripts/build_latin_references.py') if family == 'Alipay Number'
                                           else str(Path(manifest).resolve())),
                       'ttc_index': 0, 'weight': weight, 'variable': bool(axes),
                       'variation_axes': axes,
                       'scripts': (['han'] if family == 'Noto Sans CJK SC' else
                                   ['latin'] if family in {'Roboto', 'Alipay Number'} else ['han', 'latin']),
                       'numeric_only': family == 'Alipay Number',
                       'cmap_codepoints': sorted(cmap)})
    manifest_data = {'schema': 'flux-glyph-capture-font-assets-v1',
                     'created_at': datetime.now(timezone.utc).isoformat(),
                     'label_status': 'source font identities; actual native glyph use must be verified',
                     'source_font_manifest': str(Path(manifest).resolve()),
                     'source_font_manifest_sha256': sha(manifest),
                     'fonts': [{key: value for key, value in font.items() if key != 'cmap_codepoints'} for font in result]}
    write_json(output / f'assets/{platform.upper()}_SOURCE_MANIFEST.json', manifest_data)
    return result


def ios_fonts(inventory=None):
    """Inventories may provide `fonts` objects, or a list of PostScript names.

    System requests deliberately have null file identity until the iOS collector
    resolves them inside the running OS. The public system-font API is explicit.
    """
    requested = []
    for suffix, weight in [('Light', 300), ('Regular', 400), ('Medium', 500), ('Semibold', 600)]:
        requested.append({'id': 'pingfang_' + suffix.lower(), 'family': 'PingFang SC',
                          'postscript': 'PingFangSC-' + suffix, 'weight': weight, 'scripts': ['han']})
    for weight in [300, 400, 500, 600, 700]:
        requested.append({'id': f'system_sf_{weight}', 'family': 'SF Pro', 'postscript': '-system',
                          'weight': weight, 'scripts': ['latin'], 'system_font': True})
    for ps, weight in [('Helvetica', 400), ('Helvetica-Bold', 700)]:
        requested.append({'id': ps.lower(), 'family': 'Helvetica', 'postscript': ps,
                          'weight': weight, 'scripts': ['latin']})
    for family, ps in [('Songti SC', 'STSongti-SC-Regular'), ('Kaiti SC', 'STKaitiSC-Regular'),
                       ('Heiti SC', 'STHeitiSC-Medium'), ('Hiragino Sans GB', 'HiraginoSansGB-W3'),
                       ('Baoli SC', 'STBaoliSC-Regular'), ('Yuanti SC', 'STYuanti-SC-Regular')]:
        requested.append({'id': ps.lower(), 'family': family, 'postscript': ps,
                          'weight': 400, 'scripts': ['han']})
    if inventory:
        data = json.loads(Path(inventory).read_text())
        if isinstance(data, dict) and isinstance(data.get('families'), list):
            entries = [name for family in data['families'] for name in family['postscript_names']]
        else:
            entries = data.get('fonts', data.get('postscript_names', [])) if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise ValueError('iOS inventory must contain a fonts or postscript_names list')
        names = {entry if isinstance(entry, str) else entry.get('postscript', entry.get('postscript_name'))
                 for entry in entries}
        requested = [font for font in requested if font.get('system_font') or font['postscript'] in names]
    for font in requested:
        font.update(kind='system', path=None, sha256=None,
                    inventory_status='listed_in_ios_inventory' if inventory and not font.get('system_font')
                    else 'resolve_during_native_capture')
    return requested


HAN = {
    'time': '今日 昨日 本月 上月 最近 当天 每日 每周 明天 今晚 后天 周末 上午 下午 晚上 早上 稍后 现在 此刻 下周'.split(),
    'item': '交通 餐饮 购物 住房 通信 电影 外卖 出行 公交 地铁 快递 医疗 教育 图书 旅行 水电 会员 服务 商城 酒店 车票 机票 充值 转账 余额 账单 订单 礼物 早餐 午餐 晚餐 咖啡 生鲜 超市 运动 门票 话费 流量 团购 保险'.split(),
    'bill': '支付 付款 退款 收款 交易 结算 转入 转出 扣款 入账 查询 统计'.split(),
    'state': '完成 成功 记录 详情 通知 提醒 确认 待处理 已更新 已确认 已完成 正在处理'.split(),
    'setting': '账号 消息 隐私 安全 通知 声音 显示 语言 网络 存储 位置 相册 通讯录 备份 更新 设备 登录 密码 主题 支付'.split(),
    'option': '管理 设置 权限 选项 提醒 记录 同步 检查 保护 中心'.split(),
    'action': '开启 关闭 更新 查看 修改 保存 检查 确认 重置 完成'.split(),
    'chat': '记得 帮忙 一起 准备 需要 可以 请先 继续 及时 稍后'.split(),
    'verb': '查看 确认 处理 分享 发送 核对 检查 更新 安排 整理'.split(),
}
ENGLISH = {
    'verb': 'Review Confirm Update Manage Check Open Save Share Find Track View Schedule'.split(),
    'time': 'recent monthly pending saved new daily weekly current previous latest incoming outgoing scheduled completed selected archived'.split(),
    'item': 'payments orders messages settings balance transfers receipts activity privacy accounts devices updates bookings alerts contacts invoices purchases statements reminders history'.split(),
}


def text_candidate(rng, script, category, numeric=False):
    if numeric:
        # Digits only: no model can use an amount prefix as the family label.
        return ''.join(rng.choice('0123456789') for _ in range(rng.randint(8, 14)))
    if script == 'latin':
        return ' '.join(rng.choice(ENGLISH[k]) for k in ['verb', 'time', 'item'])
    keys = (['time', 'item', 'bill', 'state'] if category == 'bill' else
            ['time', 'setting', 'option', 'action'] if category == 'settings' else
            ['time', 'chat', 'verb', 'item'])
    return ''.join(rng.choice(HAN[key]) for key in keys)


def content_pages(count=1000, seed=20260912):
    if count < 10:
        raise ValueError('At least 10 pages are required to populate all splits')
    rng = random.Random(seed)
    order = list(range(count))
    rng.shuffle(order)
    train = count * 8 // 10
    calibration = (count - train) // 2
    mapping = {identifier: ('train' if i < train else 'calibration' if i < train + calibration else 'test')
               for i, identifier in enumerate(order)}
    seen = set()
    pages = []
    for identifier in range(count):
        group = f'scene-{identifier:05d}'
        category = rng.choice(['bill', 'settings', 'chat'])
        rows = rng.randint(10, 12)
        scripts = ['han'] * (rows - 4) + ['latin'] * 4
        rng.shuffle(scripts)
        latin_count = 0
        regions = []
        for i, script in enumerate(scripts):
            numeric = script == 'latin' and latin_count % 2 == 0
            latin_count += int(script == 'latin')
            for attempt in range(10000):
                text = text_candidate(rng, script, category, numeric)
                normalized = ''.join(text.split())
                if normalized not in seen:
                    seen.add(normalized)
                    break
            else:
                raise ValueError('Text space exhausted; cannot create split-isolated combinations')
            font_size = rng.choice([13, 15, 17, 19, 21, 23, 25, 27])
            # A conservative length bound, independent of font family.
            max_size = int(328 / (len(text) * (1.0 if script == 'han' else 0.75) + 1))
            font_size = min(font_size, max_size)
            left = rng.choice([20, 24, 28, 32])
            top = 66 + i * 64 + rng.choice([-2, 0, 2])
            regions.append({'id': f'{group}-r{i:02d}', 'text': text, 'script': script,
                            'text_kind': 'numeric' if numeric else 'han' if script == 'han' else 'english',
                            'font_size': font_size, 'bbox_points': [left, top, 382, top + 52]})
        pages.append({'id': group, 'content_group_id': group, 'split': mapping[identifier],
                      'category': category, 'regions': regions})
    return pages


def scene_document(platform, fonts, content, seed=20260912):
    allowed = set(families())
    if not fonts or any(font['family'] not in allowed for font in fonts):
        raise ValueError('Capture fonts must belong to training/network.FAMILIES')
    by_script = {script: [font for font in fonts if script in font['scripts']] for script in ['han', 'latin']}
    if not all(by_script.values()):
        raise ValueError('Both Han and Latin fonts are needed')
    for font in fonts:
        if 'latin' in font['scripts'] and font['family'] not in LATIN_FAMILIES:
            raise ValueError('Font family is not enabled in the trainer Latin mask')
    rng = random.Random(seed + (17 if platform == 'ios' else 41))
    pools = {}
    cmaps = {font['id']: frozenset(font['cmap_codepoints']) for font in fonts if font.get('cmap_codepoints')}
    def pick(script, split, text, text_kind):
        key = (script, split, text_kind)
        if not pools.get(key):
            pools[key] = [font for font in by_script[script]
                          if not font.get('numeric_only') or text_kind == 'numeric']
            rng.shuffle(pools[key])
        # Require cmap coverage for explicit assets; system truth is captured later.
        for index, font in enumerate(pools[key]):
            if font['id'] not in cmaps or all(ord(c) in cmaps[font['id']] for c in text if not c.isspace()):
                return pools[key].pop(index)
        raise ValueError(f'Asset fonts cannot cover scene text: {text}')
    pages = []
    counts = {split: Counter() for split in SPLITS}
    for original in content:
        page = {key: value for key, value in original.items() if key != 'regions'}
        page['id'] = platform + '-' + original['id']
        page['source_id'] = page['id']
        dark = rng.randrange(5) == 0
        page['background'] = rng.choice(['#1C1C1E', '#242426'] if dark else ['#FFFFFF', '#F5F5F7', '#FAFAF8'])
        page['regions'] = []
        for original_region in original['regions']:
            region = dict(original_region)
            font = pick(region['script'], page['split'], region['text'], region['text_kind'])
            weight = rng.choice([300, 400, 500, 600, 700]) if font.get('variable') else font['weight']
            region.update(font_id=font['id'], font_postscript=font['postscript'], font_family=font['family'],
                          weight=weight, color=rng.choice(['#FFFFFF', '#ECECEF', '#C7C7CC', '#8CC8FF', '#FFB4AB', '#81D8B0'] if dark else
                                                         ['#111111', '#333333', '#555555', '#303A45', '#005BBB', '#B42318', '#1D6B44']))
            page['regions'].append(region)
            counts[page['split']][font['id']] += 1
        pages.append(page)
    for split, frequencies in counts.items():
        if len(content) >= 100 and set(frequencies) != {font['id'] for font in fonts}:
            raise ValueError(f'{split} has an uncovered font face')
    return {'schema': SCHEMA, 'platform': platform, 'canvas_points': [402, 874],
            'seed': seed, 'label_status': 'render requests; actual glyph font identity must pass native capture validation',
            'intended_capture_kind': 'ios_simulator_controlled_scene' if platform == 'ios' else 'android_emulator_controlled_scene',
            'ui_content_is_generated': True,
            'split_unit': 'content_group_id shared across platforms; all normalized region text combinations are disjoint',
            'characters_may_overlap_across_splits': True,
            'font_assignment': 'shuffled face cycles per script and split; independent of text, geometry and color',
            'fonts': [{key: value for key, value in font.items() if key != 'cmap_codepoints'} for font in fonts],
            'pages': pages, 'counts': {'pages': dict(Counter(page['split'] for page in pages)),
                                      'regions': {split: dict(values) for split, values in counts.items()}}}


def smoke_document(document):
    """Two diagnostic screens cover each requested face before mass capture."""
    fonts = document['fonts']
    if len(fonts) > 24:
        raise ValueError('Two-page smoke supports at most 24 font faces')
    pools = {kind: [r for page in document['pages'] for r in page['regions'] if r['text_kind'] == kind]
             for kind in ('han', 'english', 'numeric')}
    chosen = (list(fonts) * math.ceil(24 / len(fonts)))[:24]
    # Separate the two scripts across each screen without adding font-name text.
    han = [font for font in chosen if 'han' in font['scripts']]
    latin = [font for font in chosen if 'han' not in font['scripts']]
    ordered = []
    for page in range(2):
        take_han = min(8, len(han)) if page == 0 else len(han)
        part, han = han[:take_han], han[take_han:]
        needed = 12 - len(part)
        part += latin[:needed]
        latin = latin[needed:]
        ordered.append(part)
    pages = []
    for page_index, group in enumerate(ordered):
        identifier = f'{document["platform"]}-smoke-{page_index:02d}'
        page = {'id': identifier, 'source_id': identifier, 'content_group_id': identifier,
                'split': 'train', 'background': '#FFFFFF', 'diagnostic_only': True, 'regions': []}
        for row, font in enumerate(group):
            kind = 'han' if 'han' in font['scripts'] else 'numeric' if font.get('numeric_only') else 'english'
            region = dict(pools[kind].pop(0))
            region.update(id=f'{identifier}-r{row:02d}', font_id=font['id'], font_postscript=font['postscript'],
                          font_family=font['family'], weight=font['weight'], color='#111111',
                          bbox_points=[24, 66 + row * 64, 382, 118 + row * 64])
            page['regions'].append(region)
        pages.append(page)
    return dict(document, pages=pages, counts={'pages': 2, 'faces': len(fonts)}, diagnostic_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/mobile-font-capture')
    parser.add_argument('--platform', choices=['ios', 'android', 'both'], default='ios')
    parser.add_argument('--pages', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20260912)
    parser.add_argument('--font-manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--ios-font-inventory', type=Path)
    parser.add_argument('--no-alipay-asset', action='store_true', help='Use only iOS built-in fonts')
    parser.add_argument('--no-ios-cjk-assets', action='store_true', help='Omit imported CJK comparison fonts from iOS')
    args = parser.parse_args()
    content = content_pages(args.pages, args.seed)
    platforms = ['ios', 'android'] if args.platform == 'both' else [args.platform]
    result = {}
    for platform in platforms:
        fonts = (prepare_assets(args.output, args.font_manifest) if platform == 'android'
                 else ios_fonts(args.ios_font_inventory))
        if platform == 'ios' and not args.no_alipay_asset:
            fonts += prepare_assets(args.output, args.font_manifest, platform='ios', include_ios_cjk=not args.no_ios_cjk_assets)
            for font in fonts:
                if font['kind'] == 'asset':
                    font['host_path'], font['path'] = font['path'], Path(font['path']).name
        document = scene_document(platform, fonts, content, args.seed)
        destination = args.output / f'{platform}-scenes.json'
        write_json(destination, document)
        smoke = smoke_document(document)
        write_json(args.output / f'{platform}-scenes-smoke.json', smoke)
        result[platform] = {'path': str(destination.resolve()), 'sha256': sha(destination),
                            'fonts': len(fonts), 'pages': document['counts']['pages'],
                            'regions': sum(len(page['regions']) for page in document['pages'])}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
