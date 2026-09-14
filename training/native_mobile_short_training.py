"""Family-balanced native short TRAIN pools; platform is used only to locate pixels."""
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from ios_short_training import load_prepared as load_ios, VIEWS
from prepare_android_regions import normalized_text
from prepare_unified_regions import FAMILIES, family_label
from train_regions import require, sha

ANDROID_KNOWN = ('Noto Sans CJK SC', 'Noto Serif CJK SC', 'LXGW WenKai',
    'WenQuanYi Micro Hei', 'ZCOOL KuaiLe', 'ZCOOL XiaoWei', 'ZCOOL QingKe HuangYou',
    'Ma Shan Zheng', 'Roboto', 'FandolHei', 'FandolKai', 'FandolSong', 'Long Cang', 'Zhi Mang Xing')
KNOWN = tuple(f for f in FAMILIES if f in (*ANDROID_KNOWN, 'PingFang', 'SF Pro', 'Helvetica'))
UNKNOWN_SOURCES = ('Lato', 'Liu Jian Mao Cao', 'Open Sans', 'Smiley Sans')
SAMPLING = {'schema': 'flux-glyph-native-mobile-short-sampling-v1', 'batch_rows': 20,
    'known_families': list(KNOWN), 'rows_per_known_family': 1, 'unknown_rows_per_batch': 3,
    'unknown_sources': list(UNKNOWN_SOURCES), 'glyph_counts': [1, 2, 3, 4],
    'source_selection': 'known families each once; unknown sources in one global round robin',
    'script_and_length_selection': 'complete script/length Cartesian cycle per source family',
    'face_selection': 'round robin within source family, script and length',
    'identity_selection': 'uniform native identity within selected face before choosing view',
    'view_weights': dict(zip(VIEWS, (.4, .2, .2, .2))), 'rng_independent': True,
    'label_source': 'verified native font proof only', 'platform_features': False,
    'related_views_are_independent_samples': False, 'model_inference': False}


def load_android(root, expected_sha):
    root = Path(root).resolve()
    mp, rp, ap = (root / name for name in ('MANIFEST.json', 'rows.json', 'tiles.npy'))
    require(sha(mp) == expected_sha, 'Android short manifest differs from fixed plan')
    manifest = json.loads(mp.read_text())
    require(manifest.get('schema') == 'flux-glyph-android-native-short-train-v2'
            and manifest.get('split') == 'train' and manifest.get('families') == FAMILIES
            and manifest.get('test_read') is False and manifest.get('calibration_read') is False
            and manifest.get('accepted_native_verified_only') is True, 'Android native TRAIN proof missing')
    bindings = dict(manifest['bindings'])
    bindings.update({str(mp): expected_sha, str(rp): manifest['rows_sha256'], str(ap): manifest['array_sha256']})
    require(all(sha(path) == digest for path, digest in bindings.items()), 'Android short sources changed')
    payload = json.loads(rp.read_text())
    require(payload.get('schema') == 'flux-glyph-android-native-short-rows-v2'
            and payload.get('split') == 'train', 'Android short row partition differs')
    rows = payload['rows']; tiles = np.load(ap, mmap_mode='r', allow_pickle=False)
    require(len(rows) == manifest['tiles'] and tiles.shape == (len(rows), 1, 64, 256)
            and tiles.dtype == np.dtype('<f4'), 'Android short tile array differs')
    for index, row in enumerate(rows):
        require(row.get('split') == 'train' and row.get('domain') == 'android'
                and row.get('native_font_verified') is True and row.get('whole_width_covered') is True
                and row.get('family') == family_label(row.get('source_font_family'))
                and row.get('target') == FAMILIES.index(row['family'])
                and row.get('index') == row.get('tile_start') == index and row.get('tile_count') == 1,
                'Android short row truth/identity differs')
        require(row['family'] in ANDROID_KNOWN or
                (row['family'] == '__unknown__' and row['source_font_family'] in UNKNOWN_SOURCES),
                'Android short source is outside the fixed TRAIN families')
        value = tiles[index]
        require(np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
                and hashlib.sha256(value.tobytes()).hexdigest() == row['tiles_sha256'], 'Android tile bytes differ')
        require(math.isfinite(row['log_em_ratio']) and abs(row['log_em_ratio']) <= 3
                and row['font_size_px'] > 0 and row['ink_height_px'] > 0
                and abs(math.log(row['font_size_px'] / row['ink_height_px']) - row['log_em_ratio']) < 1e-10,
                'Android short native size target differs')
        require(row.get('glyph_count') in (1, 2, 3, 4)
                and len(row.get('native_glyph_provenance', {}).get('glyphs', [])) == row['glyph_count']
                and bindings.get(row.get('frame_path')) == row.get('frame_sha256')
                and row.get('font_face') and row.get('font_file_sha256'), 'Android native glyph proof missing')
        # Android's frozen rows carry verified plain text. Derive the same
        # identity field in memory; never rewrite the captured/prepared files.
        text_digest = hashlib.sha256(normalized_text(row['text']).encode()).hexdigest()
        require(row.get('normalized_text_sha256', text_digest) == text_digest,
                'Android normalized native text identity differs')
        row['normalized_text_sha256'] = text_digest
    return {'rows': rows, 'tiles': tiles}, {'bindings': bindings, 'manifest_sha256': expected_sha,
        'native_regions': manifest['native_regions'], 'views': len(rows)}


def build_pools(datasets):
    identities, owners = {}, defaultdict(set)
    for dataset, data in datasets.items():
        for original in data['rows']:
            row = copy.deepcopy(original); row['short_dataset'] = dataset
            identity = (dataset, row['source_id'], row['region_id'])
            views = identities.setdefault(identity, {})
            require(row['view'] in VIEWS and row['view'] not in views, 'Duplicate mobile short view')
            views[row['view']] = row
            for key in (('tile', row['tiles_sha256']), ('region', row['region_rgb_sha256'])):
                owners[key].add((row['family'], identity))
    excluded = {identity for values in owners.values() if len({family for family, _ in values}) > 1
                for _, identity in values}
    pools = {}
    for identity, views in identities.items():
        require(set(views) == set(VIEWS), 'Mobile identity must have all four related views')
        native = views['native']
        keys = ('family', 'target', 'font_face', 'glyph_count', 'source_id', 'region_id', 'source_font_family',
                'normalized_text_sha256', 'native_font_size_px', 'text_color_hex', 'source_sha256', 'frame_sha256')
        require(all([row[key] for key in keys] == [native[key] for key in keys] for row in views.values()),
                'Mobile views disagree on native truth')
        if identity in excluded:
            continue
        source = native['source_font_family'] if native['family'] == '__unknown__' else native['family']
        script = 'han' if native['script'] == 'han' else 'latin'
        require(native['script'] in ('han', 'latin', 'numeric') and native['glyph_count'] in (1, 2, 3, 4),
                'Unsupported native short script or length')
        key = (source, script, native['glyph_count'])
        pools.setdefault(key, {}).setdefault(native['font_face'], {})[identity] = views
    require({key[0] for key in pools} == set(KNOWN) | set(UNKNOWN_SOURCES), 'Mobile source coverage incomplete')
    for source in (*KNOWN, *UNKNOWN_SOURCES):
        scripts = {key[1] for key in pools if key[0] == source}
        require(scripts and {(source, script, length) for script in scripts for length in (1, 2, 3, 4)}
                <= set(pools), 'Mobile source lacks a complete script/length cycle: ' + source)
    proof = {'schema': 'flux-glyph-native-mobile-short-pools-v1', 'sampling': SAMPLING,
        'native_regions_before_conflict_exclusion': len(identities), 'native_regions': len(identities) - len(excluded),
        'cross_dataset_conflicting_native_identities': [list(identity) for identity in sorted(excluded)],
        'source_native_regions': dict(Counter(key[0] for key, faces in pools.items()
            for groups in faces.values() for _ in groups)),
        'pool_native_regions': {json.dumps(key): sum(map(len, faces.values())) for key, faces in sorted(pools.items())},
        'test_read': False, 'development_read': False, 'calibration_read': False, 'model_inference': False}
    return pools, proof


def load_mobile(ios_root, android_root, android_manifest_sha):
    ios, ios_proof = load_ios(ios_root)
    android, android_proof = load_android(android_root, android_manifest_sha)
    datasets = {'ios': ios, 'android': android}; pools, proof = build_pools(datasets)
    bindings = dict(ios_proof['bindings'])
    for path, digest in android_proof['bindings'].items():
        require(path not in bindings or bindings[path] == digest, 'Mobile source bindings conflict')
        bindings[path] = digest
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    proof.update(bindings=bindings, ios_proof=ios_proof, android_proof=android_proof)
    return {'datasets': datasets, 'pools': pools}, proof


class NativeMobileShortSampler:
    def __init__(self, data, seed):
        self.data, self.rng, self.steps = data, np.random.default_rng(seed), 0
        self.source_cursor, self.face_cursor = Counter(), Counter()
        self.unknown_cursor = 0; self.counts, self.faces, self.identities = Counter(), Counter(), Counter()

    def _draw(self, source):
        keys = sorted(key for key in self.data['pools'] if key[0] == source)
        key = keys[self.source_cursor[source] % len(keys)]; self.source_cursor[source] += 1
        faces = self.data['pools'][key]; names = sorted(faces)
        face = names[self.face_cursor[key] % len(names)]; self.face_cursor[key] += 1
        groups = faces[face]; identities = sorted(groups)
        identity = identities[int(self.rng.integers(len(identities)))]
        view = str(self.rng.choice(VIEWS, p=(.4, .2, .2, .2)))
        row = copy.deepcopy(groups[identity][view])
        self.counts[source, key[1], key[2], view] += 1; self.faces[key + (face,)] += 1
        self.identities[identity] += 1
        return row

    def batch(self):
        rows = [self._draw(family) for family in KNOWN]
        for _ in range(3):
            rows.append(self._draw(UNKNOWN_SOURCES[self.unknown_cursor % len(UNKNOWN_SOURCES)]))
            self.unknown_cursor += 1
        self.steps += 1
        require(len(rows) == 20, 'Mobile batch size differs')
        return rows

    def report(self):
        return {'schema': SAMPLING['schema'], 'sampling': SAMPLING, 'steps': self.steps, 'rows': self.steps * 20,
            'by_source': {s: sum(n for (source, _, _, _), n in self.counts.items() if source == s)
                          for s in (*KNOWN, *UNKNOWN_SOURCES)},
            'by_source_script_length_view': {json.dumps(key): value for key, value in sorted(self.counts.items())},
            'by_face': {json.dumps(key): value for key, value in sorted(self.faces.items())},
            'by_identity': {json.dumps(key): value for key, value in sorted(self.identities.items())}}
