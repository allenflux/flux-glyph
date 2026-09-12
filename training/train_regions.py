#!/usr/bin/env python3
"""Train font family and pixel font size directly from native line images.

No recognized characters, scripts, CTC tokens or glyph segmentation are model
inputs. Offline annotations establish native font/size truth and source splits.
Prepare with the inference environment; train with the standalone torch env:

  python training/train_regions.py --captures .../labels.jsonl --data .../data --prepare-only
  python training/train_regions.py --data .../data --output .../run --device mps
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'training')]
SPLITS = ('train', 'calibration', 'test')
DATA_SCHEMA = 'flux-glyph-native-region-training-data-v1'
AGREEMENT = .7
MAX_SIZE_SPREAD = .2


def require(condition, message):
    if not condition:
        raise ValueError('Region font training: ' + message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


def state_sha(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare(captures, output, *, group_pingfang=False, test_history='reused'):
    """Extract actual whole-region PNG crops; native text never enters pixels."""
    from training.capture import prepare_captured as native
    from flux_glyph.region_font import preprocess_region
    from training.capture.generate_scenes import capture_families
    from training.region_labels import label_groups, font_label, region_families

    captures, output = Path(captures).resolve(), Path(output).resolve()
    require(not output.exists(), 'prepared output must be new')
    scenes_path = captures.parent / 'Scenes.json'
    labels_sha, scenes_sha = sha(captures), sha(scenes_path)
    scenes, pages, isolation = native.load_scenes(scenes_path)
    protocol_path, protocol = native.capture_protocol(captures.parent, scenes_sha)
    protocol_sha = sha(protocol_path)
    font_registry = {font['id']: font for font in scenes['fonts']}
    native_families = capture_families()
    groups = label_groups(group_pingfang)
    require(test_history in ('fresh', 'reused'), 'unknown test history')
    families = region_families(groups)
    preprocess_sha = sha(ROOT / 'src/flux_glyph/region_font.py')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.native-regions-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'prepared'
        stage.mkdir()
        metadata = {split: [] for split in SPLITS}
        counts = {split: 0 for split in SPLITS}
        streams = {}
        for split in SPLITS:
            (stage / split).mkdir()
            streams[split] = (stage / split / 'tiles.raw').open('wb')
        sources, seen, rejected = [], set(), Counter()
        try:
            with captures.open() as incoming:
                for line in incoming:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    page, image_path, image, requests = native.native_source(record, scenes_sha, pages, captures.parent)
                    try:
                        require(record['page_id'] not in seen, 'duplicate source page')
                        require(record.get('simulator_id') == protocol.get('simulator_id') and record.get('bundle_id') == protocol.get('bundle_id'),
                                'capture simulator/app differs from protocol')
                        seen.add(record['page_id'])
                        split = record['split']
                        pixels_sha = native.pixels_sha(image)
                        identities = {'source_id': record['source_id'], 'source_file_sha256': record['source_sha256'],
                                      'decoded_pixel_sha256': pixels_sha, 'page_id': page['id'],
                                      'content_group_id': page.get('content_group_id', page['id'])}
                        for kind, value in identities.items():
                            isolation.bind(kind, value, split)
                        sources.append({'source_id': record['source_id'], 'page_id': page['id'], 'split': split,
                                        'source_kind': 'ios_simulator_screenshot', 'image': str(image_path),
                                        'source_sha256': record['source_sha256'], 'decoded_pixel_sha256': pixels_sha,
                                        'content_group_id': identities['content_group_id'], 'scenes_sha256': scenes_sha})
                        for region in record['regions']:
                            native_family = region['font_family']
                            require(native_family in native_families, 'native font family outside training registry')
                            reason = native.font_evidence(region, native_family, region['script'])
                            reason = reason or native.asset_evidence(region, requests[region['id']], font_registry)
                            if reason:
                                rejected['native:' + reason] += 1
                                continue
                            family = font_label(native_family, groups)
                            style = native.native_style(region, record['native'])
                            require(native.bounds(region.get('bbox'), image.size), 'invalid native region bbox')
                            crop = image.crop(region['bbox'])
                            crop_sha = native.pixels_sha(crop)
                            isolation.bind('region_rgb_sha256', crop_sha, split)
                            # Text is used exclusively to enforce the fixed split
                            # groups, not as a feature or a preprocessing hint.
                            isolation.bind('normalized_region_text', native.normalized_text(region['text']), split)
                            processed = preprocess_region(crop)
                            if not isinstance(processed, dict) or processed.get('status') != 'ok' or processed.get('tiles') is None:
                                rejected['preprocess_region_rejected'] += 1
                                continue
                            if processed.get('whole_width_covered') is not True:
                                rejected['region_too_long'] += 1
                                continue
                            tiles = np.ascontiguousarray(processed['tiles'], dtype='<f4')
                            height = processed.get('ink_height_px')
                            require(tiles.ndim == 4 and tiles.shape[1:] == (1, 64, 256) and 1 <= len(tiles) <= 8
                                    and np.isfinite(tiles).all() and tiles.min() >= 0 and tiles.max() <= 1,
                                    'preprocess_region returned invalid tiles')
                            require(type(height) in (float, int) and math.isfinite(height) and height > 0, 'invalid native ink height')
                            ratio = math.log(style['font_size_screen_px'] / height)
                            require(math.isfinite(ratio) and -3 <= ratio <= 3, 'invalid pixel em/ink ratio')
                            raw = tiles.tobytes()
                            entry = {'index': len(metadata[split]), 'tile_start': counts[split], 'tile_count': len(tiles),
                                     'family': family, 'font_face': region['actual_font_postscript'],
                                     'native_font_family': native_family,
                                     'source_id': record['source_id'], 'page_id': page['id'], 'region_id': region['id'],
                                     'content_group_id': identities['content_group_id'], 'split': split,
                                     'source_sha256': record['source_sha256'], 'decoded_pixel_sha256': pixels_sha,
                                     'region_rgb_sha256': crop_sha, 'bbox': region['bbox'],
                                     'ink_height_px': height, 'font_size_screen_px': style['font_size_screen_px'],
                                     'log_em_ratio': ratio, 'tiles_sha256': hashlib.sha256(raw).hexdigest(),
                                     'native_font_verified': True, 'whole_width_covered': True,
                                     'actual_font_size_points': style['actual_font_size_points'],
                                     'source_screen_scale': style['source_screen_scale'], 'text_color_hex': style['text_color_hex'],
                                     'native_style': 'actual CTFont em size; pixels = points * screen scale'}
                            metadata[split].append(entry)
                            streams[split].write(raw)
                            counts[split] += len(tiles)
                    finally:
                        image.close()
                    if len(seen) % 50 == 0:
                        print(json.dumps({'prepared_pages': len(seen), 'regions': {s: len(v) for s, v in metadata.items()}, 'tiles': counts}), flush=True)
        finally:
            for stream in streams.values():
                stream.close()
        require(seen == set(pages), 'all planned capture pages are required for source split audit')
        active = [family for family in families if any(row['family'] == family for row in metadata['train'])]
        require(len(active) >= 2, 'training needs at least two actually captured font families')
        splits = {}
        for split in SPLITS:
            rows, folder = metadata[split], stage / split
            require(rows, split + ' contains no usable line regions')
            require(all(row['family'] in active for row in rows), 'evaluation family absent from captured train')
            for row in rows:
                row['target'] = active.index(row['family'])
            with (folder / 'tiles.npy').open('wb') as destination, (folder / 'tiles.raw').open('rb') as incoming:
                np.lib.format.write_array_header_2_0(destination, {'descr': '<f4', 'fortran_order': False,
                                                                   'shape': (counts[split], 1, 64, 256)})
                shutil.copyfileobj(incoming, destination, length=8 << 20)
            (folder / 'tiles.raw').unlink()
            dump(folder / 'metadata.json', {'schema': DATA_SCHEMA, 'split': split, 'rows': rows})
            splits[split] = {'regions': len(rows), 'tiles': counts[split],
                             'family_counts': dict(Counter(row['family'] for row in rows)),
                             'array': {'path': f'{split}/tiles.npy', 'sha256': sha(folder / 'tiles.npy')},
                             'metadata': {'path': f'{split}/metadata.json', 'sha256': sha(folder / 'metadata.json')}}
        require(sha(captures) == labels_sha and sha(scenes_path) == scenes_sha and sha(protocol_path) == protocol_sha,
                'capture inputs changed during preparation')
        require(sha(ROOT / 'src/flux_glyph/region_font.py') == preprocess_sha, 'preprocessing code changed during preparation')
        manifest = {'schema': DATA_SCHEMA, 'families': active, 'splits': splits, 'sources': sources,
                    'font_label_groups': groups,
                    'font_sources': {family: sorted({font['kind'] for font in font_registry.values()
                                     if font_label(font['family'], groups) == family}) for family in active},
                    'source_kind': 'ios_simulator_screenshot', 'capture_domain': 'ios_simulator_controlled_scene',
                    'ui_content_is_generated': True, 'image_source': 'simctl_png', 'accepted_native_verified_only': True,
                    'input_labels': {'path': str(captures), 'sha256': labels_sha},
                    'scenes': {'path': str(scenes_path), 'sha256': scenes_sha},
                    'capture_protocol': {'path': str(protocol_path), 'sha256': protocol_sha},
                    'split_isolation': {key: 0 for key in ('source_id', 'source_file_sha256', 'decoded_pixel_sha256',
                                                          'page_id', 'content_group_id', 'region_rgb_sha256', 'normalized_region_text')},
                    'preprocessing': {'function': 'flux_glyph.region_font.preprocess_region',
                                      'source_sha256': preprocess_sha, 'input': 'original screenshot RGB region pixels only',
                                      'shape': [1, 64, 256], 'dtype': 'float32', 'max_tiles': 8},
                    'network_inputs': ['image_tiles'], 'ocr_performed': False, 'script_features_used': False,
                    'text_features_used': False, 'character_segmentation_performed': False,
                    'label_text_use': 'native evidence validation and pre-existing source/content split audit only',
                    'regression_target': 'log(native_font_size_screen_px / image_measured_ink_height_px)',
                    'test_history': test_history,
                    'test_scope': ('New fixed capture pages; same controlled simulator domain as the parent model.'
                                   if test_history == 'fresh' else 'Previously used fixed test pages; regression suite, not a new blind test.'),
                    'rejected': dict(rejected), 'preparation_code_sha256': sha(Path(__file__))}
        dump(stage / 'MANIFEST.json', manifest)
        stage.rename(output)
        return manifest


class RegionData:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest_path = self.directory / 'MANIFEST.json'
        self.manifest_sha = sha(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        m = self.manifest
        require(m.get('schema') == DATA_SCHEMA and m.get('source_kind') == 'ios_simulator_screenshot'
                and m.get('image_source') == 'simctl_png' and m.get('accepted_native_verified_only') is True,
                'only verified native region datasets are accepted')
        require(m.get('network_inputs') == ['image_tiles'] and m.get('ocr_performed') is False
                and m.get('text_features_used') is False and m.get('script_features_used') is False
                and m.get('character_segmentation_performed') is False, 'region dataset includes OCR/text/script inputs')
        require(set(m['splits']) == set(SPLITS), 'three fixed region splits are required')
        require(m['preprocessing']['source_sha256'] == sha(ROOT / 'src/flux_glyph/region_font.py'), 'region preprocessing code differs')
        require(m['preprocessing']['shape'] == [1, 64, 256], 'region tensor geometry differs')
        self.families = m['families']
        from training.region_labels import validate_groups, region_families
        self.groups = validate_groups(m.get('font_label_groups', {}))
        require(isinstance(self.families, list) and 2 <= len(self.families) <= 64
                and len(set(self.families)) == len(self.families), 'invalid active region classes')
        require(self.families == [name for name in region_families(self.groups) if name in self.families],
                'active region classes differ from the ordered family registry')
        for name in ('input_labels', 'scenes', 'capture_protocol'):
            entry = m[name]
            require(Path(entry['path']).is_file() and sha(entry['path']) == entry['sha256'], 'native source manifest changed: ' + name)
        identity = {}
        self.sources = {}
        for source in m['sources']:
            split = source['split']
            require(split in SPLITS and source['source_id'] not in self.sources, 'duplicate/invalid source')
            require(source['source_kind'] == 'ios_simulator_screenshot' and sha(source['image']) == source['source_sha256'],
                    'actual screenshot source changed')
            self.sources[source['source_id']] = source
            for key in ('source_id', 'source_sha256', 'decoded_pixel_sha256', 'page_id', 'content_group_id'):
                value = source[key]
                require(isinstance(value, str) and value and identity.setdefault((key, value), split) == split,
                        'source/content identity crosses splits: ' + key)
        require(set(m['split_isolation']) == {'source_id', 'source_file_sha256', 'decoded_pixel_sha256', 'page_id',
                                            'content_group_id', 'region_rgb_sha256', 'normalized_region_text'}
                and all(type(v) is int and v == 0 for v in m['split_isolation'].values()), 'split isolation has failures')
        self.loaded = {}
        self.region_hash_splits = {}

    def load(self, split):
        require(split in SPLITS and split not in self.loaded, 'partition already loaded or invalid')
        require(sha(self.manifest_path) == self.manifest_sha, 'prepared manifest changed')
        descriptor = self.manifest['splits'][split]
        paths = {}
        for name in ('array', 'metadata'):
            item = descriptor[name]
            path = (self.directory / item['path']).resolve()
            require(path.is_relative_to(self.directory) and path.is_file() and sha(path) == item['sha256'],
                    'prepared partition asset changed: ' + name)
            paths[name] = path
        meta = json.loads(paths['metadata'].read_text())
        require(meta['schema'] == DATA_SCHEMA and meta['split'] == split, 'region metadata split/schema differs')
        rows = meta['rows']
        require(rows and len(rows) == descriptor['regions'], 'empty/incomplete region partition')
        tiles = np.load(paths['array'], mmap_mode='r', allow_pickle=False)
        require(tiles.dtype == np.float32 and tiles.shape == (descriptor['tiles'], 1, 64, 256), 'invalid region tile array')
        next_tile, regions = 0, set()
        for i, row in enumerate(rows):
            require(row['index'] == i and row['split'] == split and row['tile_start'] == next_tile
                    and type(row['tile_count']) is int and 1 <= row['tile_count'] <= 8,
                    'invalid contiguous region tile mapping')
            next_tile += row['tile_count']
            require(row['family'] in self.families and row['target'] == self.families.index(row['family'])
                    and row['native_font_verified'] is True and row['whole_width_covered'] is True, 'invalid native family target or incomplete region')
            from training.region_labels import font_label
            native_family = row.get('native_font_family', row['family'])
            require(font_label(native_family, self.groups) == row['family']
                    and (not self.groups or native_family not in self.groups), 'native font identity differs from label group')
            require(not any(key in row for key in ('text', 'script', 'tokens', 'characters')), 'region metadata leaks input text/script')
            source = self.sources.get(row['source_id'])
            require(source is not None and source['split'] == split, 'region has no source in this split')
            for key in ('source_sha256', 'decoded_pixel_sha256', 'page_id', 'content_group_id'):
                require(row[key] == source[key], 'region source identity differs: ' + key)
            key = (row['source_id'], row['region_id'])
            require(key not in regions, 'duplicate source region')
            regions.add(key)
            require(self.region_hash_splits.setdefault(row['region_rgb_sha256'], split) == split,
                    'identical region pixels cross splits')
            require(math.isclose(row['log_em_ratio'], math.log(row['font_size_screen_px'] / row['ink_height_px']),
                                 abs_tol=1e-9), 'native size regression target differs')
            block = np.asarray(tiles[row['tile_start']:next_tile])
            require(np.isfinite(block).all() and block.min() >= 0 and block.max() <= 1
                    and hashlib.sha256(block.tobytes()).hexdigest() == row['tiles_sha256'], 'region tiles differ from source crop hash')
        require(next_tile == len(tiles), 'orphan region tiles')
        require(dict(Counter(row['family'] for row in rows)) == descriptor['family_counts'], 'region family counts differ')
        result = {'split': split, 'tiles': tiles, 'rows': rows, 'targets': np.asarray([row['target'] for row in rows]),
                  'log_em_ratio': np.asarray([row['log_em_ratio'] for row in rows], dtype=np.float32)}
        self.loaded[split] = result
        return result


class RegionSampler:
    def __init__(self, data, seed, *, families=None, system_family_weight=1):
        require(type(system_family_weight) is int and system_family_weight in (1, 2), 'system family weight must be 1 or 2')
        self.data, self.rng = data, np.random.default_rng(seed)
        self.pools = {int(target): np.where(data['targets'] == target)[0] for target in np.unique(data['targets'])}
        require(system_family_weight == 1 or (isinstance(families, list) and all(0 <= i < len(families) for i in self.pools)),
                'weighted sampling requires the actual output family registry')
        self.weights = {target: system_family_weight if families and families[target] in ('PingFang', 'SF Pro', 'Helvetica') else 1
                        for target in self.pools}
        self.weighted_cycle = []
        self.queues = {target: list(self.rng.permutation(rows)) for target, rows in self.pools.items()}
        self.visits = np.zeros(len(data['rows']), dtype=np.int32)
        self.family_cursor = 0

    def batch(self, size):
        families = sorted(self.pools)
        weighted = any(weight != 1 for weight in self.weights.values())
        if not weighted:
            self.rng.shuffle(families)
        rows = []
        for i in range(size):
            if weighted:
                if not self.weighted_cycle:
                    cycle = [target for target in families for _ in range(self.weights[target])]
                    self.weighted_cycle = list(self.rng.permutation(cycle))
                target = int(self.weighted_cycle.pop())
            else:
                target = families[(self.family_cursor + i) % len(families)]
            if not self.queues[target]:
                self.queues[target] = list(self.rng.permutation(self.pools[target]))
            rows.append(int(self.queues[target].pop()))
        self.family_cursor += size
        indices = np.asarray(rows, dtype=np.int64)
        np.add.at(self.visits, indices, 1)
        tiles = np.asarray([self.data['rows'][i]['tile_start'] + int(self.rng.integers(self.data['rows'][i]['tile_count'])) for i in indices])
        return tiles, self.data['targets'][indices], self.data['log_em_ratio'][indices]


def training_test_history(manifest, override=None):
    recorded = manifest.get('test_history', 'reused')
    history = override or recorded
    require(history in ('fresh', 'reused') and not (history == 'fresh' and recorded != 'fresh'),
            'previously used test pages cannot become fresh')
    scope = ('Previously used fixed test pages; regression suite, not a new blind test.' if history == 'reused'
             else manifest.get('test_scope', 'New fixed capture pages; same controlled simulator domain as the parent model.'))
    return {'test_history': history, 'test_scope': scope}


def system_focus_policy(args, families):
    """Validate the predeclared continuation budget and inherit parent gates."""
    from training.system_finetune import SYSTEM_FAMILIES, PARENT_TEMPERATURE, PARENT_GATES, MAX_SIZE_SPREAD
    frozen = {'seed': 2026091294, 'steps': 4000, 'batch_size': 64, 'learning_rate': 3e-5,
              'eval_every': 250, 'system_family_weight': 2, 'preserve_size_head': True}
    require(all(getattr(args, name, None) == value for name, value in frozen.items()),
            'system-focused mode requires the frozen seed/steps/batch/lr/eval/weight/preserve-size-head configuration')
    require(not getattr(args, 'allow_new_traditional_families', False) and not getattr(args, 'benchmark_steps', 0),
            'system-focused mode continues existing classes without a benchmark')
    require(len(families) == 8 and set(SYSTEM_FAMILIES).issubset(families), 'system-focused mode requires all eight competing classes')
    require(getattr(args, 'parent_metadata', None) is not None, '--parent-metadata is required for system-focused continuation')
    path = Path(args.parent_metadata).resolve()
    meta = json.loads(path.read_text())
    require(meta.get('schema') == 'flux-glyph-region-font-v1' and meta.get('algorithm') == 'region-cnn64x256-v1'
            and meta.get('families') == families and not meta.get('family_gates'), 'parent model registry/algorithm differs')
    require(meta.get('temperature') == PARENT_TEMPERATURE and meta.get('gates') == PARENT_GATES
            and meta.get('max_size_relative_spread') == MAX_SIZE_SPREAD, 'parent fixed calibration policy differs')
    expected_state = meta.get('training', {}).get('state_after_sha256')
    require(isinstance(expected_state, str) and len(expected_state) == 64, 'parent metadata must pin its trained state')
    return {'enabled': True, 'system_families': list(SYSTEM_FAMILIES), 'system_family_weight': 2,
            'parent_metadata': {'path': str(path), 'sha256': sha(path)}, 'parent_state_sha256': expected_state,
            'fixed_policy': {'temperature': meta['temperature'], 'gates': dict(meta['gates']),
                             'max_size_relative_spread': meta['max_size_relative_spread']},
            'selection_policy': {'source': 'calibration_only', 'global_wrong_must_not_increase': True,
                                 'system_wrong_by_true_and_predicted_family_must_not_increase': True,
                                 'nonsystem_correct_must_not_decrease': True, 'nonsystem_wrong_must_not_increase': True,
                                 'preserve_all_parent_accepted_correct_rows': True, 'system_correct_must_increase': True,
                                 'rank': ['system_correct', 'minimum_system_correct_coverage', 'overall_correct',
                                          'negative_mean_size_relative_error'], 'tie': 'earliest_checkpoint',
                                 'recalibrate': False}}


def sampling_report(sampler, training, families):
    native_visits = Counter()
    for row, visits in zip(training['rows'], sampler.visits):
        native_visits[row.get('native_font_family', row.get('family', families[row.get('target', 0)]))] += int(visits)
    return {'rows': len(training['rows']), 'visited': int(np.count_nonzero(sampler.visits)),
            'minimum_visits': int(sampler.visits.min()), 'maximum_visits': int(sampler.visits.max()),
            'family_weights': {families[i]: weight for i, weight in sampler.weights.items()},
            'family_visits': {families[i]: int(sampler.visits[training['targets'] == i].sum()) for i in sampler.pools},
            'native_family_visits': dict(native_visits)}


def softmax(logits, temperature):
    values = logits.astype(np.float64)
    values = np.exp((values - values.max(axis=1, keepdims=True)) / temperature)
    return values / values.sum(axis=1, keepdims=True)


def region_observations(logits, log_ratio, data, temperature=1.):
    require(logits.shape[0] == len(log_ratio) == len(data['tiles']) and np.isfinite(logits).all()
            and np.isfinite(log_ratio).all(), 'invalid region predictions')
    patch_prob = softmax(logits, temperature)
    probabilities, agreement, ratios, spread, valid = [], [], [], [], []
    for row in data['rows']:
        start, end = row['tile_start'], row['tile_start'] + row['tile_count']
        prob = patch_prob[start:end].mean(axis=0)
        top = int(np.argmax(prob))
        probabilities.append(prob)
        agreement.append(float((np.argmax(patch_prob[start:end], axis=1) == top).mean()))
        logs = log_ratio[start:end]
        valid.append(bool(np.all(np.abs(logs) <= 3) and row['whole_width_covered']))
        # Runtime rejects out-of-range output. Clipping here bounds diagnostics
        # only: invalid regions are excluded from every accepted result below.
        safe_logs = np.clip(logs, -3., 3.)
        predicted_ratio = np.exp(safe_logs.astype(np.float64))
        median_ratio = float(np.exp(np.median(safe_logs)))
        ratios.append(median_ratio)
        spread.append(float((np.quantile(predicted_ratio, .9) - np.quantile(predicted_ratio, .1)) / median_ratio))
    probs = np.asarray(probabilities)
    rank = np.argsort(-probs, axis=1, kind='stable')
    heights = np.asarray([row['ink_height_px'] for row in data['rows']])
    expected = np.asarray([row['font_size_screen_px'] for row in data['rows']])
    sizes = heights * np.asarray(ratios)
    return {'probabilities': probs, 'predicted': rank[:, 0], 'correct': rank[:, 0] == data['targets'],
            'score': probs[np.arange(len(probs)), rank[:, 0]],
            'margin': probs[np.arange(len(probs)), rank[:, 0]] - probs[np.arange(len(probs)), rank[:, 1]],
            'agreement': np.asarray(agreement), 'size_spread': np.asarray(spread), 'valid_output': np.asarray(valid),
            'size_px': sizes, 'size_abs_error': np.abs(sizes - expected),
            'size_relative_error': np.abs(sizes - expected) / expected,
            'nll': float(-np.log(np.clip(probs[np.arange(len(probs)), data['targets']], 1e-12, 1)).mean())}


def metrics(obs, data, families, gates=None):
    by_family = {family: {'rows': int((data['targets'] == i).sum()),
                           'accuracy': float(obs['correct'][data['targets'] == i].mean())}
                 for i, family in enumerate(families) if (data['targets'] == i).any()}
    valid_size = (obs['size_spread'] <= MAX_SIZE_SPREAD) & obs['valid_output']
    result = {'regions': len(data['rows']), 'accuracy': float(obs['correct'].mean()),
              'macro_accuracy': float(np.mean([row['accuracy'] for row in by_family.values()])),
              'families': by_family, 'negative_log_likelihood': obs['nll'],
              'invalid_output_regions': int((~obs['valid_output']).sum()),
              'size': {'rows': len(valid_size), 'mae_px': float(obs['size_abs_error'].mean()),
                       'mean_relative_error': float(obs['size_relative_error'].mean()),
                       'within_10_percent': float((obs['size_relative_error'] <= .1).mean()),
                       'consistent_patch_rows': int(valid_size.sum()),
                       'consistent_patch_mae_px': float(obs['size_abs_error'][valid_size].mean()) if valid_size.any() else None,
                       'max_size_relative_spread': MAX_SIZE_SPREAD}}
    if gates:
        accepted = (obs['valid_output'] & (obs['score'] >= gates['min_score']) & (obs['margin'] >= gates['min_margin'])
                    & (obs['margin'] > 1e-8) & (obs['agreement'] >= gates['min_patch_agreement']))
        result['accepted'] = {'regions': int(accepted.sum()), 'correct': int((accepted & obs['correct']).sum()),
                              'wrong': int((accepted & ~obs['correct']).sum()), 'coverage': float(accepted.mean()),
                              'precision': float(obs['correct'][accepted].mean()) if accepted.any() else None}
        accepted_size = accepted & valid_size
        result['accepted_size'] = {'regions': int(accepted_size.sum()), 'coverage': float(accepted_size.mean()),
                                   'mae_px': float(obs['size_abs_error'][accepted_size].mean()) if accepted_size.any() else None,
                                   'mean_relative_error': float(obs['size_relative_error'][accepted_size].mean()) if accepted_size.any() else None}
    return result


def calibrate(logits, ratios, data, families):
    choices = [(region_observations(logits, ratios, data, temperature), temperature)
               for temperature in (.5, .75, 1., 1.25, 1.5, 2., 3., 4.)]
    obs, temperature = min(choices, key=lambda item: (item[0]['nll'], item[1]))
    options = []
    for score in (.5, .6, .7, .75, .8, .85, .9, .925, .95, .975, .99):
        for margin in (.01, .025, .05, .1, .15, .2, .3, .4, .5):
            gate = {'min_score': score, 'min_margin': margin, 'min_patch_agreement': AGREEMENT}
            result = metrics(obs, data, families, gate)
            accepted = result['accepted']
            if accepted['regions'] >= 20 and accepted['precision'] >= .97:
                options.append((accepted['regions'], -score, -margin, gate))
    gates = max(options, key=lambda item: item[:3])[-1] if options else {'min_score': 1., 'min_margin': 1., 'min_patch_agreement': AGREEMENT}
    return temperature, gates, {'gate_found': bool(options), 'temperature': temperature,
                                'gate_selection_level': 'actual_whole_native_region', 'target_precision': .97,
                                'minimum_accepted_regions': 20, 'metrics': metrics(obs, data, families, gates),
                                'unknown_font_rejection_validated': False}


def predict(net, data, device, batch=128):
    import torch
    net.eval()
    logits, ratios = [], []
    with torch.inference_mode():
        for start in range(0, len(data['tiles']), batch):
            x = torch.from_numpy(np.array(data['tiles'][start:start + batch], copy=True)).to(device)
            family, ratio = net(x)
            logits.append(family.cpu().numpy())
            ratios.append(ratio.cpu().numpy())
    return np.concatenate(logits), np.concatenate(ratios)


def benchmark(args, data, training):
    """Discard a bounded train-only speed probe; never choose or save weights."""
    import torch
    from torch.nn import functional as F
    from region_network import RegionFontClassifier
    require(args.device in ('cpu', 'mps'), 'benchmark requires an explicit device')
    require(args.device != 'mps' or torch.backends.mps.is_available(), 'MPS unavailable')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    net = RegionFontClassifier(len(data.families))
    net.warm_start(args.warm_start, data.families, new_family_parents=family_initializers(args, data.families),
                   preserve_size_head=getattr(args, 'preserve_size_head', False))
    if not getattr(args, 'preserve_size_head', False):
        with torch.no_grad():
            net.size_head.bias.fill_(float(np.median(training['log_em_ratio'])))
    net.to(args.device).train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    sampler, elapsed = RegionSampler(training, args.seed), []
    for step in range(args.benchmark_steps + 2):
        if args.device == 'mps':
            torch.mps.synchronize()
        started = time.perf_counter()
        indices, labels, ratio = sampler.batch(args.batch_size)
        x = torch.from_numpy(np.array(training['tiles'][indices], copy=True)).to(args.device)
        y, ratio = torch.tensor(labels, device=args.device), torch.tensor(ratio, device=args.device)
        optimizer.zero_grad(set_to_none=True)
        logits, prediction = net(x)
        loss = F.cross_entropy(logits, y) + .5 * F.smooth_l1_loss(prediction, ratio, beta=.05)
        require(bool(torch.isfinite(loss)), 'benchmark nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
        optimizer.step()
        if args.device == 'mps':
            torch.mps.synchronize()
        if step >= 2:
            elapsed.append(time.perf_counter() - started)
    result = {'schema': 'flux-glyph-region-speed-benchmark-v1', 'device': args.device,
              'steps': args.benchmark_steps, 'discarded_warmup_steps': 2, 'batch_size': args.batch_size,
              'seconds_per_update_median': float(np.median(elapsed)),
              'estimated_3000_update_seconds_excluding_calibration': float(np.median(elapsed) * 3000),
              'data_manifest_sha256': data.manifest_sha, 'model_weights_saved': False,
              'only_train_data_opened': True, 'calibration_or_test_used': False,
              'training_code_sha256': sha(__file__)}
    require(args.output is not None, 'benchmark requires --output')
    destination = args.output / 'benchmark' / (args.device + '.json')
    require(not destination.exists(), 'benchmark result already exists')
    dump(destination, result)
    print(json.dumps(result), flush=True)


def train(args):
    import torch
    from torch.nn import functional as F
    from region_network import RegionFontClassifier

    data = RegionData(args.data)
    training = data.load('train')
    if args.benchmark_steps:
        require(not getattr(args, 'system_focused', False) and getattr(args, 'system_family_weight', 1) == 1,
                'benchmark is separate from system-focused/weighted training')
        benchmark(args, data, training)
        return
    calibration = data.load('calibration')
    families = data.families
    focused = system_focus_policy(args, families) if getattr(args, 'system_focused', False) else None
    history_override = getattr(args, 'test_history', None)
    if focused:
        require(history_override != 'fresh', 'system-focused continuation uses the existing regression test partition')
        history_override = 'reused'
    test_history = training_test_history(data.manifest, history_override)
    if args.validate_only:
        print(json.dumps({'validated': True, 'families': families,
                          'train_regions': len(training['rows']), 'calibration_regions': len(calibration['rows']),
                          'test_arrays_opened': False}, ensure_ascii=False))
        return
    output = args.output.resolve()
    require(not output.exists() or set(p.name for p in output.iterdir()) <= {'data', 'benchmark'},
            'training output must be new or contain only prepared data/benchmark')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = 'mps' if args.device == 'auto' and torch.backends.mps.is_available() else 'cpu' if args.device == 'auto' else args.device
    require(device != 'mps' or torch.backends.mps.is_available(), 'MPS is unavailable')
    net = RegionFontClassifier(len(families))
    initializers = family_initializers(args, families)
    parent_families = net.warm_start(args.warm_start, families, new_family_parents=initializers,
                                    preserve_size_head=getattr(args, 'preserve_size_head', False))
    if not getattr(args, 'preserve_size_head', False):
        with torch.no_grad():
            net.size_head.bias.fill_(float(np.median(training['log_em_ratio'])))
    initial = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
    before = state_sha(initial)
    if focused:
        require(before == focused['parent_state_sha256'], 'warm-start parameters differ from the pinned parent metadata')
    net.to(device)
    sampler = RegionSampler(training, args.seed, families=families,
                            system_family_weight=getattr(args, 'system_family_weight', 1))
    code = {str(path.relative_to(ROOT)): sha(path) for path in
            [Path(__file__), ROOT / 'training/region_network.py', ROOT / 'training/network.py',
             ROOT / 'training/region_labels.py', ROOT / 'training/capture/generate_scenes.py',
             ROOT / 'src/flux_glyph/region_font.py']}
    if focused:
        code['training/system_finetune.py'] = sha(ROOT / 'training/system_finetune.py')
    freeze = {'schema': 'flux-glyph-region-training-freeze-v1', 'families': families, 'seed': args.seed,
              'device': device, 'torch': torch.__version__, 'optimizer_steps_planned': args.steps,
              'batch_size': args.batch_size, 'evaluate_every_steps': args.eval_every,
              'optimizer': {'name': 'AdamW', 'learning_rate': args.learning_rate, 'weight_decay': .0001},
              'loss': f'{len(families)}-family unmasked cross entropy + 0.5 smooth-L1(log em/ink ratio), beta=0.05',
              'selection': 'calibration family macro accuracy minus 0.1 times min(mean size relative error,1); earliest tie',
              'model_inputs': ['RGB-derived region tiles'], 'ocr_inputs': False, 'script_mask': False,
              'font_label_groups': data.groups,
              'region_aggregation': 'mean patch softmax; size=ink_height*exp(median(log_em_ratio))',
              'invalid_output': 'reject entire region when any abs(log_em_ratio)>3 or whole width not covered',
              'size_spread': '(p90_ratio-p10_ratio)/median_ratio',
              'calibration': {'gate_target_region_precision': .97, 'min_regions': 20,
                              'min_patch_agreement': AGREEMENT, 'max_size_relative_spread': MAX_SIZE_SPREAD},
              'data_manifest_sha256': data.manifest_sha, 'source_kind': 'ios_simulator_screenshot',
              'source_counts': dict(Counter(source['split'] for source in data.sources.values())),
              'regions': {split: data.manifest['splits'][split]['regions'] for split in SPLITS},
              'warm_start': {'path': str(args.warm_start.resolve()), 'sha256': sha(args.warm_start), 'families': parent_families,
                             'new_family_initializers': initializers,
                             'preserved_size_head': getattr(args, 'preserve_size_head', False)},
              'initial_state_sha256': before, 'code_sha256': code, 'test_arrays_opened': False,
              **test_history}
    freeze['training_region_sampling'] = {'family_weights': {families[i]: weight for i, weight in sampler.weights.items()},
                                         'system_family_weight': getattr(args, 'system_family_weight', 1),
                                         'row_policy': 'exhaust each shuffled family row queue before reuse'}
    protocol = getattr(args, 'regression_protocol', None)
    if protocol:
        protocol = Path(protocol).resolve()
        freeze['regression_protocol'] = {'path': str(protocol), 'sha256': sha(protocol)}
    parent_stats = None
    if focused:
        from training.system_finetune import selection_stats, assess_checkpoint
        policy = focused['fixed_policy']
        parent_obs = region_observations(*predict(net, calibration, device), calibration, policy['temperature'])
        parent_stats = selection_stats(parent_obs, calibration, families, policy['gates'])
        baseline = {'schema': 'flux-glyph-parent-calibration-baseline-v1', 'optimizer_steps_executed': 0,
                    'parent_state_sha256': before, 'parent_metadata': focused['parent_metadata'], 'fixed_policy': policy,
                    'calibration_partition': data.manifest['splits']['calibration'], 'data_manifest_sha256': data.manifest_sha,
                    'metrics': metrics(parent_obs, calibration, families, policy['gates']), 'selection_stats': parent_stats,
                    'test_arrays_opened': False}
        dump(output / 'PARENT_CALIBRATION.json', baseline)
        focused['parent_calibration'] = {'path': 'PARENT_CALIBRATION.json', 'sha256': sha(output / 'PARENT_CALIBRATION.json')}
        freeze['system_focused'] = focused
        freeze['selection'] = focused['selection_policy']
        freeze['calibration'] = {**policy, 'policy_source': 'frozen_parent_metadata', 'recalibrate': False}
    dump(output / 'TRAINING_FREEZE.json', freeze)
    for path in code:
        destination = output / 'frozen-code' / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, destination)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history, best, losses = [], None, []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        net.train()
        indices, targets, size_targets = sampler.batch(args.batch_size)
        x = torch.from_numpy(np.array(training['tiles'][indices], copy=True)).to(device)
        y = torch.tensor(targets, dtype=torch.long, device=device)
        target_ratio = torch.tensor(size_targets, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits, ratio = net(x)
        classification_loss = F.cross_entropy(logits, y)
        size_loss = F.smooth_l1_loss(ratio, target_ratio, beta=.05)
        loss = classification_loss + .5 * size_loss
        require(bool(torch.isfinite(loss)), 'nonfinite region loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
        optimizer.step()
        optimizer.param_groups[0]['lr'] = args.learning_rate * (.1 + .45 * (1 + math.cos(math.pi * step / args.steps)))
        losses.append((float(classification_loss.detach().cpu()), float(size_loss.detach().cpu())))
        if step % 25 == 0:
            print(json.dumps({'step': step, 'classification_loss': float(np.mean([v[0] for v in losses[-25:]])),
                              'size_loss': float(np.mean([v[1] for v in losses[-25:]])), 'seconds': time.perf_counter() - started}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            predictions = predict(net, calibration, device)
            if focused:
                obs = region_observations(*predictions, calibration, policy['temperature'])
                metrics_cal = metrics(obs, calibration, families, policy['gates'])
                stats = selection_stats(obs, calibration, families, policy['gates'])
                assessment = assess_checkpoint(stats, parent_stats)
                score = tuple(assessment['rank'])
                eligible = assessment['eligible']
            else:
                metrics_cal = metrics(region_observations(*predictions, calibration), calibration, families)
                score = metrics_cal['macro_accuracy'] - .1 * min(metrics_cal['size']['mean_relative_error'], 1.)
                eligible = True
            record = {'step': step, 'selection_score': score, 'calibration': metrics_cal,
                      'unique_train_regions_sampled': int(np.count_nonzero(sampler.visits)),
                      'seconds': time.perf_counter() - started}
            if focused:
                record.update(eligibility=assessment, selection_stats=stats)
            history.append(record)
            if eligible and (best is None or score > best['score']):
                state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
                torch.save({'state_dict': state, 'families': families, 'optimizer_steps': step}, output / 'best-calibration.pth')
                best = {'step': step, 'score': score, 'checkpoint_sha256': sha(output / 'best-calibration.pth')}
            dump(output / 'history.json', history)
            print(json.dumps({'calibration': record, 'best_step': best['step'] if best else None}), flush=True)
    if focused:
        final_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        final_sha = state_sha(final_state)
        torch.save({'state_dict': final_state, 'families': families, 'optimizer_steps': args.steps}, output / 'last-trained.pth')
        require(all(sha(ROOT / path) == expected for path, expected in code.items()), 'training/runtime code changed during the frozen run')
        require(sha(data.manifest_path) == data.manifest_sha, 'prepared manifest changed during training')
        require(sha(focused['parent_metadata']['path']) == focused['parent_metadata']['sha256'], 'parent policy changed during training')
        if protocol:
            require(sha(protocol) == freeze['regression_protocol']['sha256'], 'regression protocol changed during training')
        if best is None:
            no_candidate = {'status': 'NO_PROMOTABLE_CHECKPOINT', 'optimizer_steps_executed': args.steps,
                            'state_before_sha256': before, 'state_after_sha256': final_sha,
                            'parameters_changed': final_sha != before,
                            'last_checkpoint': {'path': 'last-trained.pth', 'sha256': sha(output / 'last-trained.pth')},
                            'history_sha256': sha(output / 'history.json'), 'training_freeze_sha256': sha(output / 'TRAINING_FREEZE.json'),
                            'parent_calibration': focused['parent_calibration'], 'test_arrays_opened': False,
                            'training_region_sampling': sampling_report(sampler, training, families), **test_history}
            dump(output / 'NO_PROMOTABLE_CHECKPOINT.json', no_candidate)
            dump(output / 'report.json', no_candidate)
            print(json.dumps({'finished': True, 'output': str(output), **no_candidate}), flush=True)
            return
    require(best is not None and sha(output / 'best-calibration.pth') == best['checkpoint_sha256'], 'selected region model changed')
    selected = torch.load(output / 'best-calibration.pth', map_location='cpu', weights_only=True)
    net.load_state_dict(selected['state_dict'])
    after = state_sha(selected['state_dict'])
    require(after != before, 'region model parameters did not change')
    if focused:
        temperature, gates = policy['temperature'], policy['gates']
        selected_obs = region_observations(*predict(net, calibration, device), calibration, temperature)
        selected_stats = selection_stats(selected_obs, calibration, families, gates)
        require(assess_checkpoint(selected_stats, parent_stats)['eligible'], 'selected checkpoint no longer satisfies the fixed parent-relative policy')
        cal_result = {'gate_found': True, 'temperature': temperature, 'recalibrated': False,
                      'gate_selection_level': 'inherited_parent_metadata', 'parent_metadata': focused['parent_metadata'],
                      'metrics': metrics(selected_obs, calibration, families, gates), 'selection_stats': selected_stats}
    else:
        temperature, gates, cal_result = calibrate(*predict(net, calibration, device), calibration, families)
    selection = {'schema': 'flux-glyph-region-selection-before-test-v1', 'selected': best,
                 'optimizer_steps_executed': args.steps, 'state_before_sha256': before, 'state_after_sha256': after,
                 'temperature': temperature, 'gates': gates, 'max_size_relative_spread': MAX_SIZE_SPREAD,
                 'calibration': cal_result, 'test_arrays_opened': False,
                 'training_seconds': time.perf_counter() - started,
                 'parameter_l2_change': {key: float((selected['state_dict'][key] - initial[key]).norm())
                                         for key in ('trunk.0.weight', 'style.0.weight', 'family_head.weight', 'size_head.weight')},
                 'training_region_sampling': sampling_report(sampler, training, families), **test_history}
    dump(output / 'SELECTION_FREEZE.json', selection)
    # The test arrays are used only after every model/gate choice above is fixed.
    test = data.load('test')
    test_result = metrics(region_observations(*predict(net, test, device), test, temperature), test, families, gates)
    require(all(sha(ROOT / path) == expected for path, expected in code.items()), 'training/runtime code changed during the frozen run')
    neural = output / 'region'
    neural.mkdir()
    net.cpu().eval()
    torch.onnx.export(net, torch.zeros(2, 1, 64, 256), neural / 'model.onnx', input_names=['tiles'],
                      output_names=['logits', 'log_em_ratio'], dynamic_axes={'tiles': {0: 'batch'}, 'logits': {0: 'batch'}, 'log_em_ratio': {0: 'batch'}},
                      opset_version=17, dynamo=False)
    torch.save(selected, neural / 'model.pth')
    meta = {'schema': 'flux-glyph-region-font-v1', 'algorithm': 'region-cnn64x256-v1',
            'families': families, 'model': {'path': 'model.onnx', 'sha256': sha(neural / 'model.onnx')},
            'font_label_groups': data.groups,
            'font_sources': data.manifest.get('font_sources', {}),
            'temperature': temperature, 'gates': gates, 'max_size_relative_spread': MAX_SIZE_SPREAD,
            'preprocessing': {'function': 'preprocess_region', 'shape': [1, 64, 256], 'dtype': 'float32', 'max_tiles': 8,
                              'source_sha256': data.manifest['preprocessing']['source_sha256']},
            'size': {'target': 'log(font_size_screen_px/ink_height_px)', 'valid_log_ratio_range': [-3, 3],
                     'aggregation': 'exp(median(log_em_ratio))', 'spread': '(p90_ratio-p10_ratio)/exp(median(log_em_ratio))'},
            'aggregation': 'mean_softmax_probability_across_patches', 'ocr_required': False, 'script_input_required': False,
            'training': {'data_manifest_sha256': data.manifest_sha, 'source_kind': 'ios_simulator_screenshot',
                         'optimizer_steps_executed': args.steps, 'selected_step': best['step'],
                         'parent_sha256': freeze['warm_start']['sha256'], 'state_after_sha256': after,
                         'selection_sha256': sha(output / 'SELECTION_FREEZE.json')},
            'calibration': ({'gate_found': True, 'recalibrated': False, 'source': 'parent_metadata',
                             'parent_metadata_sha256': focused['parent_metadata']['sha256']} if focused
                            else {'gate_found': cal_result['gate_found'], 'target_region_precision': .97}),
            'release_status': 'experimental',
            'scope': 'OCR-free whole-region font candidates and pixel em size; ' + freeze['test_scope']}
    dump(neural / 'metadata.json', meta)
    report = {'schema': 'flux-glyph-region-font-training-report-v1', 'families': families, 'selection': selection,
              'test': test_result, 'test_arrays_opened_after_freeze': True,
              'test_reused_from_previous_version': test_history['test_history'] != 'fresh', **test_history,
              'font_label_groups': data.groups,
              'source_counts': freeze['source_counts'], 'regions': freeze['regions'], 'model_sha256': sha(neural / 'model.onnx'),
              'ocr_performed': False, 'model_inputs': ['image_tiles'],
              'limitations': [freeze['test_scope'],
                              'Native region boxes isolate classification/size quality; detector quality must be evaluated separately.',
                              'Screenshots are actual iOS Simulator frames of generated controlled native pages.',
                              'Physical-device/third-party-app accuracy and unknown-font rejection are not established.']}
    dump(output / 'report.json', report)
    print(json.dumps({'finished': True, 'output': str(output), 'selected_step': best['step'], 'model': str(neural / 'model.onnx')}), flush=True)


def family_initializers(args, families):
    if not getattr(args, 'allow_new_traditional_families', False):
        return {}
    return {name: 'PingFang SC' for name in ('PingFang', 'PingFang TC', 'PingFang HK') if name in families}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--test-history', choices=['fresh', 'reused'], default=None,
                        help='Override report test history without changing prepared data; default inherits its manifest')
    parser.add_argument('--group-pingfang', action='store_true',
                        help='Prepare SC/TC/HK as the visually identifiable PingFang family; retain native source identity')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--benchmark-steps', type=int, default=0)
    parser.add_argument('--warm-start', type=Path, default=ROOT / 'artifacts/ios-font-screenshots-v1/neural/model.pth')
    parser.add_argument('--allow-new-traditional-families', action='store_true',
                        help='Initialize new PingFang TC/HK heads from SC; requires actual native training samples')
    parser.add_argument('--preserve-size-head', action='store_true',
                        help='Continue the parent region model size head instead of reinitializing it')
    parser.add_argument('--system-focused', action='store_true',
                        help='Frozen eight-class system-font continuation with parent-relative fixed-gate CAL selection')
    parser.add_argument('--system-family-weight', type=int, choices=[1, 2], default=1,
                        help='Sampling copies for PingFang/SF Pro/Helvetica; 1 preserves the existing sampler')
    parser.add_argument('--parent-metadata', type=Path,
                        help='Parent region metadata that pins the unchanged calibration policy and warm-start state')
    parser.add_argument('--regression-protocol', type=Path,
                        help='Read-only predeclared regression protocol to bind into TRAINING_FREEZE')
    parser.add_argument('--device', choices=['cpu', 'mps', 'auto'], default='auto')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2026091243)
    args = parser.parse_args()
    require(args.steps > 0 and args.batch_size > 0 and args.eval_every > 0 and 0 < args.learning_rate < 1, 'invalid training budget')
    require(0 <= args.benchmark_steps <= 100 and not (args.benchmark_steps and (args.prepare_only or args.validate_only)),
            'benchmark must be a separate bounded invocation')
    if args.prepare_only:
        require(args.captures is not None, '--captures is required for preparation')
        report = prepare(args.captures, args.data, group_pingfang=args.group_pingfang, test_history=args.test_history or 'reused')
        print(json.dumps({'prepared': str(args.data), 'families': report['families'],
                          'regions': {s: report['splits'][s]['regions'] for s in SPLITS}}, ensure_ascii=False))
        return
    require(args.output is not None or args.validate_only, '--output is required for training')
    train(args)


if __name__ == '__main__':
    main()
