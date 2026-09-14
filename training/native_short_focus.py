"""Build and sample verified complete native 3/4-glyph TRAIN identities."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import re
import statistics
import unicodedata

import numpy as np

from prepare_unified_regions import FAMILIES
from retention_paired_known_sampler import KNOWN_SUPPLEMENT_MARKER
from retention_supplement_sampler import NEW_SOURCES, SUPPLEMENT_MARKER
from short_region_objective import SHORT_UNKNOWN_SOURCES
from train_regions import require, sha

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ('native', 'half', 'three_quarters_jpeg75', 'jpeg75')
VIEW_WEIGHTS = {'native': .4, 'half': .2, 'three_quarters_jpeg75': .2, 'jpeg75': .2}
FOCUS_KNOWN_FAMILIES = tuple(family for family in FAMILIES[:-1] if family in {
    'Noto Sans CJK SC', 'Roboto', 'Noto Serif CJK SC', 'LXGW WenKai',
    'WenQuanYi Micro Hei', 'ZCOOL KuaiLe', 'ZCOOL XiaoWei',
    'ZCOOL QingKe HuangYou', 'Ma Shan Zheng', 'FandolHei', 'FandolKai',
    'FandolSong', 'Long Cang', 'Zhi Mang Xing',
})
ALL_TRAIN_UNKNOWN_SOURCES = tuple(sorted((*SHORT_UNKNOWN_SOURCES, *NEW_SOURCES)))
FOCUS_UNKNOWN_SOURCES = ('Lato', 'Liu Jian Mao Cao', 'Open Sans', 'Smiley Sans',
                         'WenQuanYi Zen Hei', 'Zhuque Fangsong')
SAMPLING = {
    'batch_size': 32, 'known_rows': 28, 'unknown_rows': 4,
    'known_families': list(FOCUS_KNOWN_FAMILIES),
    'known_per_family_and_glyph_count': 1,
    'glyph_counts': [3, 4], 'unknown_per_glyph_count': 2,
    'unknown_sources': list(FOCUS_UNKNOWN_SOURCES),
    'allowed_train_unknown_sources': list(ALL_TRAIN_UNKNOWN_SOURCES),
    'unknown_selection': 'independent exact source round robin for each glyph count, then uniform native identity',
    'known_selection': 'one row per family and glyph count; exact font faces cycle before uniform native identity',
    'identity_weighting': 'uniform native identity before view; derived view multiplicity never weights identity',
    'view_weights': VIEW_WEIGHTS,
    'required_views': list(VIEWS), 'required_tile_count_per_view': 1,
    'proof': 'exact TRAIN Android glyph list and font identity; text length is never a glyph-count fallback',
    'inference': False, 'independent_rng': True,
}
DATASETS = {
    'original': {
        'root': ROOT/'artifacts/unified-font-v1/data-v1',
        'manifest_sha256': 'dfdd9fa746b28b3665eed10491ce83c5c872f16b2499fe3ae6c6e6bb9ed40a23',
        'partition_sha256': '4a14ef83b39181f5da55d0c5a7c3a0f868547e6dd5ee497f5887b70cec479ca5',
        'tiles': 130854,
    },
    'supplement': {
        'root': ROOT/'artifacts/unified-font-v2/new-unknown-capture-v1/data',
        'manifest_sha256': '43aac66dfc3ce3a09fd0acd088dcda82450a7203997befd1e35aebe016713b05',
        'partition_sha256': 'df2c79fc8f99a4bd364e631a97e39a3c8d0281cf68a01199316143898664425c',
        'tiles': 3156,
    },
    'known': {
        'root': ROOT/'artifacts/unified-font-v3/paired-wenkai-weight-capture-v1/merged-data',
        'manifest_sha256': 'ec9b616538444741742f91414c7e66ae51bdf036c3956fdca9d6bdbbb7873b83',
        'partition_sha256': '546dda53c827f95db3a7ecd121d9321524c5f9a10a36c17507227053fdc1dd46',
        'tiles': 4820,
    },
}
ANDROID_LABELS = ROOT/'artifacts/android-font-v1/capture-v3/labels.jsonl'
TRAIN_MARKER = re.compile(rb'"split"\s*:\s*"train"')


def _json_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _text_sha(value):
    value = ''.join(unicodedata.normalize('NFKC', value).casefold().split())
    return hashlib.sha256(value.encode()).hexdigest()


def _inside(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), 'Native proof is missing or outside the repository')
    return path


def _original_labels(wanted):
    labels = {}
    with ANDROID_LABELS.open('rb') as stream:
        for raw in stream:
            if not TRAIN_MARKER.search(raw):
                continue
            row = json.loads(raw)
            require(row.get('split') == 'train', 'Android TRAIN marker differs from decoded split')
            key = (row.get('source_id'), row.get('region_id'))
            if key in wanted:
                require(key not in labels, 'Duplicate original Android native identity')
                labels[key] = row
    return labels


def _proof_region(raw, label, bindings):
    proof_path = _inside(label['proof'] if Path(label['proof']).is_absolute()
                         else ANDROID_LABELS.parent/label['proof'])
    digest = sha(proof_path)
    if label.get('proof_sha256') is not None:
        require(label['proof_sha256'] == digest, 'Native proof SHA differs from its row')
    bindings[str(proof_path)] = digest
    proof = json.loads(proof_path.read_text())
    require(proof.get('split') == 'train'
            and proof.get('page_id') == raw.get('page_id'),
            'Native proof is not the exact TRAIN page')
    region = next((item for item in proof.get('regions', [])
                   if item.get('id') == raw.get('region_id')), None)
    require(region is not None and region.get('font_verified') is True,
            'Native proof lacks the exact region/font verification')
    glyphs = region.get('glyphs')
    require(isinstance(glyphs, list) and len(glyphs) in (3, 4),
            'Focus identity is not an exact three/four-glyph proof')
    face = raw.get('font_face'); file_sha = raw.get('font_file_sha256'); ttc = raw.get('ttc_index', 0)
    for glyph in glyphs:
        font = glyph.get('font', {})
        box = glyph.get('bbox')
        require(type(glyph.get('glyph_id')) is int and glyph['glyph_id'] != 0
                and glyph.get('font_verified') is True
                and font.get('postscript') == face and font.get('sha256') == file_sha
                and font.get('ttc_index', 0) == ttc
                and isinstance(box, list) and len(box) == 4
                and all(type(value) in (int, float) for value in box)
                and box[0] < box[2] and box[1] < box[3],
                'Glyph is invisible or its exact font identity differs')
    text = region.get('text')
    require(isinstance(text, str) and text and _text_sha(text) == raw.get('normalized_text_sha256'),
            'Native proof text identity differs from the frozen row hash')
    return len(glyphs), proof_path, digest


def _validate_views(source, rows, label, bindings):
    require(len(rows) == len(VIEWS) and {row.get('view') for row in rows} == set(VIEWS),
            'Focus identity requires exactly the four frozen TRAIN views')
    raw_by_view = {}
    truth_fields = ('split', 'family', 'target', 'source_id', 'region_id', 'source_font_family',
                    'font_face', 'font_file_sha256', 'domain', 'native_font_verified',
                    'normalized_text_sha256', 'page_id')
    expected = {key: rows[0].get(key) for key in truth_fields}
    require(expected['split'] == 'train' and expected['native_font_verified'] is True
            and expected['domain'] == 'android' and expected['family'] == FAMILIES[expected['target']]
            and SUPPLEMENT_MARKER not in rows[0] and KNOWN_SUPPLEMENT_MARKER not in rows[0],
            'Focus row is not verified Android TRAIN truth')
    for row in rows:
        require({key: row.get(key) for key in truth_fields} == expected
                and row.get('tile_count') == 1 and type(row.get('tile_start')) is int
                and row['tile_start'] >= 0 and row.get('tiles_sha256'),
                'Focus views disagree on truth or are not single-tile')
        raw_by_view[row['view']] = copy.deepcopy(row)
    native = raw_by_view['native']
    if source == 'original':
        require(label.get('split') == 'train' and label.get('source_id') == native['source_id']
                and label.get('region_id') == native['region_id']
                and label.get('image_sha256') == native.get('source_sha256')
                and label.get('font_face') == native['font_face']
                and label.get('font_file_sha256') == native['font_file_sha256']
                and _text_sha(label.get('text', '')) == native['normalized_text_sha256'],
                'Original native label differs from the frozen unified row')
    count, proof_path, proof_sha = _proof_region(native, label, bindings)
    marker = SUPPLEMENT_MARKER if source == 'supplement' else KNOWN_SUPPLEMENT_MARKER if source == 'known' else None
    return {'source': source, 'family': native['family'], 'source_font_family': native['source_font_family'],
            'font_face': native['font_face'], 'glyph_count': count,
            'identity': (native['source_id'], native['region_id']), 'rows': raw_by_view,
            'row_sha256': {view: _json_sha(raw_by_view[view]) for view in VIEWS},
            'proof': str(proof_path), 'proof_sha256': proof_sha, 'marker': marker}


def build_focus(datasets):
    """Return strict complete-native pools and a serializable source/row proof."""
    require(set(datasets) == set(DATASETS), 'Focus datasets differ from the frozen R22 TRAIN union')
    bindings = {str(Path(__file__).resolve()): sha(__file__), str(ANDROID_LABELS.resolve()): sha(ANDROID_LABELS)}
    groups = {}
    original_wanted = set()
    for source, expected in DATASETS.items():
        data = datasets[source]; root = expected['root']; part_path = root/'train/MANIFEST.json'
        rows_path = root/'train'/data['partition']['metadata']['path']
        require(data.get('manifest_sha256') == expected['manifest_sha256']
                and data.get('partition_sha256') == expected['partition_sha256']
                and len(data.get('tiles', ())) == expected['tiles']
                and data.get('partition', {}).get('split') == 'train'
                and data.get('families') == FAMILIES
                and sha(root/'MANIFEST.json') == expected['manifest_sha256']
                and sha(part_path) == expected['partition_sha256']
                and sha(rows_path) == data['partition']['metadata']['sha256']
                and json.loads(rows_path.read_text()) == data.get('rows'),
                'Focus dataset or on-disk TRAIN row order differs')
        for path in (root/'MANIFEST.json', part_path, rows_path):
            bindings[str(path.resolve())] = sha(path)
        by_identity = defaultdict(list)
        for row in data['rows']:
            require(row.get('split') == 'train', 'Non-TRAIN row entered the focus builder')
            by_identity[(row.get('source_id'), row.get('region_id'))].append(row)
        groups[source] = by_identity
        if source == 'original':
            original_wanted = set(by_identity)
    labels = _original_labels(original_wanted)
    records = []
    for source, identities in groups.items():
        for key, rows in identities.items():
            native = next((row for row in rows if row.get('view') == 'native'), None)
            if (native is None or native.get('domain') != 'android'
                    or len(rows) != len(VIEWS) or {row.get('view') for row in rows} != set(VIEWS)
                    or any(row.get('tile_count') != 1 for row in rows)):
                continue
            if source == 'original':
                label = labels.get(key)
                if label is None:
                    continue
            else:
                label = {'proof': native.get('proof'), 'proof_sha256': native.get('proof_sha256')}
            try:
                record = _validate_views(source, rows, label, bindings)
            except ValueError as error:
                if 'not an exact three/four-glyph proof' in str(error):
                    continue
                raise
            records.append(record)
    known = defaultdict(lambda: defaultdict(list)); unknown = defaultdict(list)
    for record in records:
        if record['family'] == '__unknown__':
            require(record['source_font_family'] in ALL_TRAIN_UNKNOWN_SOURCES,
                    'Held-out or undeclared unknown source entered focus')
            unknown[(record['source_font_family'], record['glyph_count'])].append(record)
        else:
            require(record['family'] in FOCUS_KNOWN_FAMILIES,
                    'Complete focus pool contains an unexpected named family')
            known[(record['family'], record['glyph_count'])][record['font_face']].append(record)
    require(set(known) == {(family, count) for family in FOCUS_KNOWN_FAMILIES for count in (3, 4)}
            and all(pool for faces in known.values() for pool in faces.values()),
            'Exactly fourteen known families need nonempty three/four-glyph pools')
    require(set(unknown) == {(source, count) for source in FOCUS_UNKNOWN_SOURCES for count in (3, 4)}
            and all(unknown.values()),
            'The six proof-qualified focus unknown sources need both three/four-glyph pools')
    identity_proof = [{'source': r['source'], 'family': r['family'],
        'source_font_family': r['source_font_family'], 'font_face': r['font_face'],
        'glyph_count': r['glyph_count'], 'source_id': r['identity'][0], 'region_id': r['identity'][1],
        'proof': r['proof'], 'proof_sha256': r['proof_sha256'], 'row_sha256': r['row_sha256']}
        for r in sorted(records, key=lambda item: (item['family'], item['glyph_count'], item['identity']))]
    proof = {'schema': 'flux-glyph-native-short-focus-proof-v1', 'sampling': SAMPLING,
        'families': FAMILIES, 'focus_known_families': list(FOCUS_KNOWN_FAMILIES),
        'unknown_sources': list(FOCUS_UNKNOWN_SOURCES), 'identity_count': len(records),
        'source_partition_counts': dict(sorted(Counter(r['source'] for r in records).items())),
        'known_pool_counts': [{'family': family, 'glyph_count': count, 'font_face': face, 'identities': len(pool)}
            for (family, count), faces in sorted(known.items()) for face, pool in sorted(faces.items())],
        'unknown_pool_counts': [{'source_font_family': source, 'glyph_count': count, 'identities': len(pool)}
            for (source, count), pool in sorted(unknown.items())],
        'identities': identity_proof, 'row_binding_sha256': _json_sha(identity_proof),
        'bindings': dict(sorted(bindings.items())), 'model_inference': False,
        'calibration_read': False, 'development_read': False, 'test_read': False}
    pools = {'known': {key: dict(value) for key, value in known.items()},
             'unknown': dict(unknown), 'row_binding_sha256': proof['row_binding_sha256']}
    return pools, proof


class NativeShortFocusSampler:
    def __init__(self, pools, seed):
        require(isinstance(seed, int) and set(pools) == {'known', 'unknown', 'row_binding_sha256'},
                'Invalid native focus pools or seed')
        self.pools = pools
        self.rng = np.random.default_rng(seed)
        self.face_positions = Counter(); self.unknown_positions = Counter()
        self.steps = 0; self.counts = Counter(); self.identity_counts = Counter()

    def _choose(self, records, glyph_count):
        record = records[int(self.rng.integers(len(records)))]
        view = str(self.rng.choice(list(VIEWS), p=[VIEW_WEIGHTS[v] for v in VIEWS]))
        raw = copy.deepcopy(record['rows'][view])
        require(_json_sha(raw) == record['row_sha256'][view]
                and raw['tile_count'] == 1 and raw['view'] == view,
                'Focus row changed after proof construction')
        if record['marker'] is not None:
            raw[record['marker']] = True
        self.counts[(raw['family'], glyph_count, raw['source_font_family'], raw['font_face'], view)] += 1
        self.identity_counts[(record['source'], *record['identity'])] += 1
        return raw

    def batch(self):
        rows = []
        for family in FOCUS_KNOWN_FAMILIES:
            for count in (3, 4):
                faces = self.pools['known'][(family, count)]
                names = sorted(faces)
                position = self.face_positions[(family, count)]
                face = names[position % len(names)]
                self.face_positions[(family, count)] += 1
                rows.append(self._choose(faces[face], count))
        for count in (3, 4):
            for _ in range(2):
                position = self.unknown_positions[count]
                source = FOCUS_UNKNOWN_SOURCES[position % len(FOCUS_UNKNOWN_SOURCES)]
                self.unknown_positions[count] += 1
                rows.append(self._choose(self.pools['unknown'][(source, count)], count))
        require(len(rows) == 32, 'Native short focus batch differs from 28 known plus four unknown')
        self.rng.shuffle(rows); self.steps += 1
        return rows

    def report(self):
        by_family = Counter(); by_length = Counter(); by_source = Counter(); by_source_length = Counter()
        by_face = Counter(); by_view = Counter()
        for (family, count, source, face, view), rows in self.counts.items():
            by_family[family] += rows; by_length[str(count)] += rows; by_view[view] += rows
            by_face[(family, face)] += rows
            if family == '__unknown__':
                by_source[source] += rows
                by_source_length[(source, str(count))] += rows
        identity_draws = list(self.identity_counts.values())
        return {'schema': 'flux-glyph-native-short-focus-sampling-v1', 'steps': self.steps,
            'rows': sum(self.counts.values()), 'by_family': dict(sorted(by_family.items())),
            'by_glyph_count': dict(sorted(by_length.items())),
            'by_unknown_source': dict(sorted(by_source.items())),
            'by_unknown_source_glyph_count': [
                {'source_font_family': source, 'glyph_count': count, 'rows': rows}
                for (source, count), rows in sorted(by_source_length.items())],
            'by_face': [{'family': family, 'font_face': face, 'rows': rows}
                        for (family, face), rows in sorted(by_face.items())],
            'by_view': dict(sorted(by_view.items())),
            'unique_native_identities_drawn': len(identity_draws),
            'per_identity_draws': {'minimum': min(identity_draws) if identity_draws else 0,
                'median': statistics.median(identity_draws) if identity_draws else 0,
                'maximum': max(identity_draws) if identity_draws else 0,
                'histogram': dict(sorted(Counter(map(str, identity_draws)).items(), key=lambda item: int(item[0])))},
            'sampling': SAMPLING,
            'row_binding_sha256': self.pools['row_binding_sha256']}
