#!/usr/bin/env python3
"""Train the font CNN on verified crops of actual iOS Simulator screenshots.

Uses only capture/prepare_captured.py output. The synthetic checkpoint supplies
initial parameters, never examples or teacher predictions. Test inference starts
only after checkpoint, temperature and acceptance gates are frozen.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from network import FAMILIES, SCRIPTS, FontClassifier

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'calibration', 'test')
SOURCE_KIND = 'ios_simulator_screenshot'
IDENTITIES = ('source_id', 'source_file_sha256', 'decoded_pixel_sha256', 'page_id',
              'region_rgb_sha256', 'content_group_id', 'normalized_region_text')
HEX = re.compile(r'[a-f0-9]{64}\Z')


def require(condition, message):
    if not condition:
        raise ValueError('Native screenshot training: ' + message)


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


def script_of(character):
    if not isinstance(character, str) or len(character) != 1:
        return None
    if '\u4e00' <= character <= '\u9fff':
        return 'han'
    if character.isascii() and character.isalnum():
        return 'latin'
    return None


class CapturedData:
    """Validate identities and cryptographic partitions; mmap each requested split.

    The manifest includes test source annotations. They are used for identity
    checks only. Test glyph arrays and per-glyph metadata are opened after freeze.
    """
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.manifest_path = self.folder / 'MANIFEST.json'
        self.manifest_sha = sha(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        m = self.manifest
        require(m.get('schema') == 'flux-glyph-native-captured-glyphs-v1', 'unsupported prepared schema')
        require(m.get('families') == FAMILIES and m.get('scripts') == SCRIPTS, 'prepared family registry differs')
        require(m.get('desktop_rendered_pixels_accepted') is False and m.get('instrumented_boxes_used') is True,
                'only instrumented native screenshots are accepted')
        require(m.get('accepted_native_verified_only') is True and m.get('image_source') == 'simctl_png'
                and m.get('ui_content_is_generated') is True, 'missing native screen provenance')
        require(m.get('ocr_performed') is False, 'expected native glyph instrumentation, not OCR guesses')
        require(isinstance(m.get('capture_protocol_sha256'), str) and HEX.fullmatch(m['capture_protocol_sha256']),
                'missing capture protocol SHA')
        for key in ('input_labels', 'scenes', 'capture_protocol'):
            descriptor = m.get(key, {})
            source_path = Path(descriptor.get('path', ''))
            require(source_path.is_file() and sha(source_path) == descriptor.get('sha256'),
                    'capture provenance file changed or is absent: ' + key)
        require(m['capture_protocol']['sha256'] == m['capture_protocol_sha256'], 'capture protocol SHA aliases differ')
        protocol = json.loads(Path(m['capture_protocol']['path']).read_text())
        require(protocol.get('schema') == 'ios-native-screen-capture-v1'
                and protocol.get('source_kind') == SOURCE_KIND
                and protocol.get('scenes_sha256') == m['scenes']['sha256'], 'capture protocol is not bound to these scenes')
        require(set(m.get('splits', {})) == set(SPLITS), 'three declared partitions are required')
        require(set(m.get('source_kind_counts', {})) == {SOURCE_KIND}, 'input includes another image source kind')
        isolation = m.get('split_isolation', {})
        require(all(type(isolation.get(key)) is int and isolation[key] == 0 for key in IDENTITIES),
                'source/page/content/pixel split isolation is unproven')
        prep = m.get('preprocessing', {})
        require(prep.get('function') == 'flux_glyph.neural_font.preprocess_glyph'
                and prep.get('padding_pixels') == 4 and prep.get('shape') == [64, 64]
                and prep.get('dtype') == 'float32' and prep.get('range') == [0, 1],
                'preprocessing differs from deployed neural inference')
        require(prep.get('runtime_source_sha256') == sha(ROOT / 'src/flux_glyph/neural_font.py')
                and prep.get('glyph_extraction_source_sha256') == sha(ROOT / 'src/flux_glyph/glyph_preprocess.py'),
                'preprocessing implementation changed after preparation')
        self.registry = {}
        self.sources = {}
        self.regions = {}
        for source in m.get('sources', []):
            split = source.get('split')
            require(split in SPLITS and source.get('source_kind') == SOURCE_KIND, 'invalid source kind/split')
            sid = source.get('source_id')
            require(isinstance(sid, str) and sid and sid not in self.sources, 'duplicate/empty source_id')
            for key in ('source_sha256', 'decoded_pixel_sha256', 'native_frame_sha256', 'scenes_sha256'):
                require(isinstance(source.get(key), str) and HEX.fullmatch(source[key]), 'invalid source SHA: ' + key)
            require(source['scenes_sha256'] == m['scenes']['sha256'], 'source scene provenance differs')
            image_path = Path(source.get('image_path', source.get('image', '')))
            require(image_path.is_file() and sha(image_path) == source['source_sha256'],
                    'original simctl screenshot missing or changed')
            self.sources[sid] = source
            for key, value in [('source_id', sid), ('source_file_sha256', source['source_sha256']),
                               ('decoded_pixel_sha256', source['decoded_pixel_sha256']),
                               ('page_id', source.get('page_id')), ('content_group_id', source.get('content_group_id'))]:
                self.bind(key, value, split)
            for region in source.get('regions', []):
                require(region.get('font_match_verified') is True and region.get('fallback_detected') is False,
                        'source region has no verified actual font')
                key = (sid, region['id'])
                require(key not in self.regions, 'duplicate native source region')
                self.regions[key] = region
                self.bind('region_rgb_sha256', region['region_rgb_sha256'], split)
                self.bind('normalized_region_text', ''.join(region['text'].split()), split)
        require(self.sources and len(self.sources) == m.get('screenshot_count')
                == m['source_kind_counts'][SOURCE_KIND], 'screenshot source count differs')
        self.loaded = {}
        self.families = None
        self.scripts = None
        self.audit = {'manifest_sha256': self.manifest_sha, 'source_kind': SOURCE_KIND,
                      'capture_domain': 'ios_simulator_controlled_scene', 'ui_content_is_generated': True,
                      'test_source_metadata_inspected_for_identity_validation': True,
                      'test_glyph_pixels_loaded': False, 'characters_may_overlap_across_splits': True,
                      'source_overlap': {key: 0 for key in IDENTITIES},
                      'screenshots_by_split': dict(Counter(s['split'] for s in self.sources.values())),
                      'verified_partition_sha256': {}}

    def bind(self, kind, value, split):
        require(isinstance(value, str) and bool(value), 'empty source identity: ' + kind)
        prior = self.registry.setdefault((kind, value), split)
        require(prior == split, f'{kind} crosses screenshot partitions')

    def asset(self, descriptor):
        relative = descriptor.get('path')
        require(isinstance(relative, str), 'invalid prepared asset path')
        path = (self.folder / relative).resolve()
        require(path.is_relative_to(self.folder) and path.is_file(), 'prepared asset escapes folder or is absent')
        require(isinstance(descriptor.get('sha256'), str) and HEX.fullmatch(descriptor['sha256']), 'invalid asset SHA')
        require(sha(path) == descriptor['sha256'], 'prepared asset SHA differs: ' + relative)
        return path

    def load(self, split):
        require(split in SPLITS, 'unknown split')
        require(sha(self.manifest_path) == self.manifest_sha, 'prepared manifest changed')
        require(split not in self.loaded, 'partition already loaded')
        part = self.manifest['splits'][split]
        metadata_path = self.asset(part['metadata'])
        metadata = json.loads(metadata_path.read_text())
        require(metadata.get('schema') == 'flux-glyph-native-captured-metadata-v1'
                and metadata.get('split') == split, 'metadata schema/split differs')
        rows = metadata.get('rows')
        require(isinstance(rows, list) and len(rows) == part['rows'] and rows, split + ' partition is empty or incomplete')
        x_path = self.asset(part['glyphs'])
        x = np.load(x_path, mmap_mode='r', allow_pickle=False)
        require(x.dtype == np.float32 and x.shape == (len(rows), 64, 64)
                and part['glyphs'].get('shape') == list(x.shape)
                and part['glyphs'].get('dtype') == 'float32', 'glyph array shape/dtype differs')
        for i, row in enumerate(rows):
            require(row.get('array_row') == i and row.get('source_kind') == SOURCE_KIND, 'metadata array row/source differs')
            source = self.sources.get(row.get('source_id'))
            require(source is not None and source['split'] == split, 'glyph source is not in this split')
            for key in ('page_id', 'content_group_id', 'source_sha256', 'decoded_pixel_sha256', 'scenes_sha256'):
                require(row.get(key) == source[key], 'glyph source identity differs: ' + key)
            region = self.regions.get((row['source_id'], row.get('region_id')))
            require(region is not None, 'glyph has no native source region')
            for key, source_key in [('family', 'font_family'), ('script', 'script'), ('text', 'text'),
                                    ('region_rgb_sha256', 'region_rgb_sha256'), ('region_bbox', 'bbox')]:
                require(row.get(key) == region[source_key], 'glyph label/region provenance differs: ' + key)
            family, script = row['family'], row['script']
            require(script in SCRIPTS and family in SCRIPTS[script]
                    and row.get('target') == FAMILIES.index(family), 'glyph family/script target differs')
            require(script_of(row.get('character')) == script, 'glyph character/script differs')
            index = row.get('text_index')
            require(type(index) is int and 0 <= index < len(row['text']) and row['text'][index] == row['character'],
                    'glyph character does not occur at annotated text index')
            native = row.get('native_glyph_provenance', {})
            require(native.get('font_match_verified') is True and native.get('visible') is True
                    and native.get('font_family') == family and native.get('character') == row['character']
                    and native.get('font_postscript') == row.get('font_face')
                    and native.get('text_index') == index and native.get('bbox') == row.get('bbox')
                    and type(native.get('glyph_id')) is int and native['glyph_id'] > 0,
                    'glyph actual native font evidence differs')
            require(isinstance(row.get('glyph_sha256'), str) and HEX.fullmatch(row['glyph_sha256']), 'invalid glyph SHA')
        require(dict(Counter(row['family'] for row in rows)) == part['family_counts']
                and dict(Counter(row['script'] for row in rows)) == part['script_counts'], 'partition label counts differ')
        if split == 'train':
            require(self.families is None, 'train can only define the registry once')
            self.families = [family for family in FAMILIES if family in part['family_counts']]
            self.scripts = {script: [family for family in self.families if any(
                row['family'] == family and row['script'] == script for row in rows)] for script in SCRIPTS}
            require(all(len(names) >= 2 for names in self.scripts.values()), 'each script needs at least two actually trained families')
        require(self.families is not None, 'train must load before evaluation partitions')
        require(all(row['family'] in self.scripts[row['script']] for row in rows),
                'evaluation contains a font/script absent from captured train')
        y = np.asarray([self.families.index(row['family']) for row in rows], dtype=np.int64)
        scripts = np.asarray([row['script'] for row in rows])
        # Check every captured glyph once, sequentially, not on every optimizer step.
        for start in range(0, len(x), 512):
            block = np.asarray(x[start:start + 512])
            require(np.isfinite(block).all() and block.min() >= 0 and block.max() <= 1, 'invalid glyph pixels')
            for j, raster in enumerate(block):
                require(hashlib.sha256(raster.tobytes()).hexdigest() == rows[start + j]['glyph_sha256'],
                        'per-glyph bytes differ from native crop provenance')
        dataset = {'name': split, 'x': x, 'y': y, 'scripts': scripts, 'rows': rows,
                   'regions': group_regions(rows), 'source_ids': {row['source_id'] for row in rows}}
        self.loaded[split] = dataset
        self.audit['verified_partition_sha256'][split] = {'metadata': part['metadata']['sha256'], 'glyphs': part['glyphs']['sha256']}
        if split == 'test':
            self.audit['test_glyph_pixels_loaded'] = True
        return dataset


def group_regions(rows):
    groups = {}
    for i, row in enumerate(rows):
        key = (row['source_id'], row['region_id'], row['script'])
        group = groups.setdefault(key, {'script': row['script'], 'family': row['family'], 'text': row['text'],
                                        'source_id': row['source_id'], 'region_id': row['region_id'],
                                        'characters': defaultdict(list), 'text_indices': []})
        require(group['family'] == row['family'] and group['text'] == row['text'], 'one region has conflicting labels/text')
        group['characters'][row['character']].append(i)
        group['text_indices'].append(row['text_index'])
    for group in groups.values():
        observed = group['text_indices']
        require(len(observed) == len(set(observed)), 'duplicate glyph text index within native region')
        expected = [i for i, char in enumerate(group['text']) if script_of(char) == group['script']]
        group['complete'] = sorted(observed) == expected
    return list(groups.values())


def active_model(families, checkpoint):
    stored = torch.load(checkpoint, map_location='cpu', weights_only=True)
    source_families = stored.get('families')
    require(isinstance(source_families, list) and len(set(source_families)) == len(source_families)
            and set(families) <= set(source_families), 'warm-start class registry cannot cover captured train')
    net = FontClassifier()
    net.family_head = nn.Linear(net.family_head.in_features, len(families))
    selected = [source_families.index(family) for family in families]
    state = {key: value[selected].clone() if key.startswith('family_head.') else value.clone()
             for key, value in stored['state_dict'].items()}
    net.load_state_dict(state, strict=True)
    return net


class BalancedSampler:
    """Use all train rows; shuffled per-family queues avoid silent row truncation."""
    def __init__(self, data, seed):
        self.rng = np.random.default_rng(seed)
        self.pools = {(script, int(target)): np.where((data['scripts'] == script) & (data['y'] == target))[0]
                      for script in SCRIPTS for target in np.unique(data['y'][data['scripts'] == script])}
        self.queues = {key: self.rng.permutation(values) for key, values in self.pools.items()}
        self.cursors = {key: 0 for key in self.pools}
        self.families = {script: [key for key in self.pools if key[0] == script] for script in SCRIPTS}
        self.visits = np.zeros(len(data['x']), dtype=np.int32)

    def take(self, key, count):
        pieces = []
        while count:
            if self.cursors[key] == len(self.queues[key]):
                self.queues[key] = self.rng.permutation(self.pools[key])
                self.cursors[key] = 0
            n = min(count, len(self.queues[key]) - self.cursors[key])
            pieces.append(self.queues[key][self.cursors[key]:self.cursors[key] + n])
            self.cursors[key] += n
            count -= n
        return np.concatenate(pieces)

    def batch(self, size):
        indices = []
        for script, count in [('han', size // 2), ('latin', size - size // 2)]:
            keys = list(self.families[script])
            self.rng.shuffle(keys)
            selected = [keys[i % len(keys)] for i in range(count)]
            for key, n in Counter(selected).items():
                indices.extend(self.take(key, n))
        indices = np.asarray(indices, dtype=np.int64)
        self.rng.shuffle(indices)
        np.add.at(self.visits, indices, 1)
        return indices


def predict(net, data, device, batch_size):
    net.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(data['x']), batch_size):
            x = torch.from_numpy(np.array(data['x'][start:start + batch_size, None], copy=True)).to(device)
            outputs.append(net(x).cpu().numpy())
    return np.concatenate(outputs)


def probabilities(logits, indices, temperature):
    scores = logits[:, indices].astype(np.float64)
    scores = (scores - scores.max(axis=1, keepdims=True)) / temperature
    scores = np.exp(scores)
    return scores / scores.sum(axis=1, keepdims=True)


def observations(logits, data, script, families, scripts, temperature=1.):
    mask = data['scripts'] == script
    positions = np.where(mask)[0]
    indices = [families.index(family) for family in scripts[script]]
    probs = probabilities(logits[positions], indices, temperature)
    glyph = {'probabilities': probs, 'y': data['y'][positions], 'complete': np.ones(len(probs), dtype=bool)}
    local = {int(index): i for i, index in enumerate(positions)}
    region_probs, targets, complete = [], [], []
    for group in data['regions']:
        if group['script'] != script:
            continue
        votes = [probs[[local[index] for index in rows]].mean(axis=0) for rows in group['characters'].values()]
        region_probs.append(np.mean(votes, axis=0))
        targets.append(families.index(group['family']))
        complete.append(group['complete'])
    require(region_probs, 'evaluation has no native regions for ' + script)
    region = {'probabilities': np.asarray(region_probs), 'y': np.asarray(targets), 'complete': np.asarray(complete, dtype=bool)}
    for group in (glyph, region):
        ranking = np.argsort(-group['probabilities'], axis=1, kind='stable')
        group['confidence'] = group['probabilities'][np.arange(len(ranking)), ranking[:, 0]]
        group['margin'] = group['confidence'] - group['probabilities'][np.arange(len(ranking)), ranking[:, 1]]
        group['predicted'] = np.asarray(indices)[ranking[:, 0]]
        group['correct'] = group['predicted'] == group['y']
        local_y = np.asarray([indices.index(int(target)) for target in group['y']])
        usable = group['complete']
        group['nll'] = float(-np.log(np.clip(group['probabilities'][np.arange(len(local_y)), local_y], 1e-12, 1))[usable].mean()) if usable.any() else None
    return glyph, region


def summary(group, families, gate=None):
    correct, complete = group['correct'], group['complete']
    per_family = {families[int(target)]: {'rows': int((group['y'] == target).sum()),
                                         'accuracy': float(correct[group['y'] == target].mean())}
                  for target in sorted(set(group['y'].tolist()))}
    result = {'rows': len(correct), 'complete_rows': int(complete.sum()), 'accuracy': float(correct.mean()),
              'macro_accuracy': float(np.mean([item['accuracy'] for item in per_family.values()])),
              'families': per_family, 'negative_log_likelihood': group['nll']}
    if gate:
        accepted = (group['confidence'] >= gate['min_score']) & (group['margin'] >= gate['min_margin']) & (group['margin'] > 1e-8) & complete
        result['accepted'] = {'rows': int(accepted.sum()), 'correct': int((accepted & correct).sum()),
                              'wrong': int((accepted & ~correct).sum()), 'coverage': float(accepted.mean()),
                              'precision': float(correct[accepted].mean()) if accepted.any() else None}
    return result


def evaluate(logits, data, families, scripts, temperatures=None, gates=None):
    return {script: {kind: summary(group, families, gates[script] if gates else None)
                     for kind, group in zip(('glyph', 'region'), observations(
                         logits, data, script, families, scripts, temperatures[script] if temperatures else 1.))}
            for script in scripts}


def calibrate(logits, data, families, scripts, target_precision=.97):
    temperatures, gates, reports = {}, {}, {}
    for script in scripts:
        choices = []
        for temperature in (.5, .75, 1., 1.25, 1.5, 2., 3., 4.):
            obs = observations(logits, data, script, families, scripts, temperature)
            require(obs[1]['nll'] is not None, 'no complete native calibration regions for ' + script)
            choices.append((obs[1]['nll'], temperature, obs))
        _, temperature, obs = min(choices, key=lambda item: item[:2])
        temperatures[script] = temperature
        options = []
        for score in (.5, .6, .7, .75, .8, .85, .9, .925, .95, .975, .99, .995):
            for gap in (.01, .025, .05, .1, .15, .2, .3, .4, .5):
                gate = {'min_score': score, 'min_margin': gap}
                metric = summary(obs[1], families, gate)
                if metric['accepted']['rows'] >= 20 and metric['accepted']['precision'] >= target_precision:
                    coverage = metric['accepted']['coverage']
                    options.append((float(coverage), -score, -gap, gate))
        chosen = max(options, key=lambda item: item[:3])[-1] if options else {'min_score': 1., 'min_margin': 1.}
        gates[script] = chosen
        reports[script] = {'temperature': temperature, 'gate_found': bool(options),
                           'glyph': summary(obs[0], families, chosen), 'region': summary(obs[1], families, chosen),
                           'target_precision': target_precision, 'minimum_accepted_complete_regions': 20,
                           'gate_calibration_level': 'complete_native_region',
                           'temperature_selection': 'minimum complete native region negative log likelihood',
                           'glyph_acceptance_precision_target_validated': False,
                           'unknown_font_rejection_validated': False}
    return temperatures, gates, reports


def requested_device(value):
    device = 'mps' if value == 'auto' and torch.backends.mps.is_available() else 'cpu' if value == 'auto' else value
    require(device != 'mps' or torch.backends.mps.is_available(), 'MPS requested but unavailable')
    return device


def benchmark(args, bundle, train):
    """Disposable optimizer timing; no calibration/test reads or model export."""
    output = args.output.resolve()
    require(not output.exists() or not any(output.iterdir()), 'benchmark output must be a new/empty directory')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = requested_device(args.device)
    model = active_model(bundle.families, args.warm_start).to(device)
    before = state_sha(model.state_dict())
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    sampler = BalancedSampler(train, args.seed)
    indices = {script: [bundle.families.index(family) for family in names] for script, names in bundle.scripts.items()}
    times, losses = [], []
    for step in range(args.benchmark_steps):
        if device == 'mps':
            torch.mps.synchronize()
        start = time.perf_counter()
        rows = sampler.batch(args.batch_size)
        x = torch.from_numpy(np.array(train['x'][rows, None], copy=True)).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = 0.
        for script, columns in indices.items():
            mask = train['scripts'][rows] == script
            target = torch.tensor([columns.index(int(y)) for y in train['y'][rows[mask]]], dtype=torch.long, device=device)
            loss = loss + F.cross_entropy(logits[torch.from_numpy(mask).to(device)][:, columns], target) / len(indices)
        require(bool(torch.isfinite(loss)), 'nonfinite benchmark loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if device == 'mps':
            torch.mps.synchronize()
        times.append(time.perf_counter() - start)
        print(json.dumps({'benchmark_step': step + 1, 'seconds': times[-1], 'loss': losses[-1]}), flush=True)
    stable = times[min(5, len(times) // 2):]
    mean = float(np.mean(stable))
    report = {'schema': 'native-screenshot-training-speed-benchmark-v1', 'device': device,
              'torch': torch.__version__, 'optimizer_steps': args.benchmark_steps, 'batch_size': args.batch_size,
              'actual_train_screenshot_count': len(train['source_ids']), 'available_train_glyphs': len(train['x']),
              'state_changed': state_sha(model.state_dict()) != before,
              'stable_step_seconds_mean': mean, 'stable_step_seconds_median': float(np.median(stable)),
              'estimated_5000_update_seconds_excluding_validation': mean * 5000,
              'step_seconds': times, 'losses': losses, 'data_audit': bundle.audit,
              'calibration_or_test_loaded': False, 'model_exported': False,
              'scope': 'Disposable speed measurement; no accuracy claim, checkpoint selection or formal training budget change.'}
    dump(output / 'benchmark.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--warm-start', type=Path, default=ROOT / 'artifacts/neural-font-v1/neural/model.pth')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--steps-per-epoch', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2026091217)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--benchmark-steps', type=int, default=0, help='Run an isolated train-only speed benchmark and export no model')
    args = parser.parse_args()
    require(args.epochs > 0 and args.steps_per_epoch > 0 and args.batch_size >= 4 and args.eval_batch_size > 0,
            'invalid training budget/batch size')
    require(0 < args.learning_rate < 1, 'invalid learning rate')
    require(args.benchmark_steps >= 0 and not (args.benchmark_steps and args.validate_only), 'invalid benchmark option combination')
    bundle = CapturedData(args.data)
    train = bundle.load('train')
    if args.benchmark_steps:
        benchmark(args, bundle, train)
        return
    cal = bundle.load('calibration')
    families, scripts = bundle.families, bundle.scripts
    if args.validate_only:
        print(json.dumps({'families': families, 'scripts': scripts, 'audit': bundle.audit,
                          'train_rows': len(train['x']), 'calibration_rows': len(cal['x'])}, ensure_ascii=False, indent=2))
        return
    output = args.output.resolve()
    require(not output.exists() or not any(output.iterdir()), 'training output must be new/empty; frozen runs are never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = requested_device(args.device)
    net = active_model(families, args.warm_start)
    initial = copy.deepcopy(net.state_dict())
    before = state_sha(initial)
    code = {str(path.relative_to(ROOT)): sha(path) for path in [Path(__file__), ROOT / 'training/network.py']}
    freeze = {'schema': 'native-screenshot-training-freeze-v1', 'seed': args.seed, 'device': device,
              'torch': torch.__version__, 'families': families, 'scripts': scripts,
              'epochs': args.epochs, 'steps_per_epoch': args.steps_per_epoch, 'batch_size': args.batch_size,
              'optimizer_steps_planned': args.epochs * args.steps_per_epoch,
              'optimizer': {'name': 'AdamW', 'learning_rate': args.learning_rate, 'weight_decay': 1e-4},
              'loss': 'equal Han/Latin script-masked cross entropy; balanced actual screenshot family queues',
              'selection': 'maximum mean calibration macro accuracy across scripts and glyph/real-region levels; earliest tie',
              'calibration_policy': {'level': 'complete_native_region', 'temperature': 'minimum region negative log likelihood',
                                     'gate': 'maximum calibration region coverage with >=20 accepted regions and >=0.97 precision',
                                     'glyph_acceptance_precision_target_validated': False},
              'region_aggregation': 'mean probabilities per repeated character, then mean distinct characters within source_id+region_id',
              'warm_start': {'path': str(args.warm_start.resolve()), 'sha256': sha(args.warm_start),
                             'head': 'slice by actually captured train family names; no teacher loss'},
              'state_before_sha256': before, 'data_audit': copy.deepcopy(bundle.audit), 'code_sha256': code,
              'training_pixels': 'Only actual simctl screenshot crops. UI text/layout is generated inside a native iOS app.',
              'synthetic_training_examples': 0, 'physical_device_screenshots': 0}
    dump(output / 'TRAINING_FREEZE.json', freeze)
    for name in code:
        destination = output / 'frozen-code' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    net.to(device)
    start = time.perf_counter()
    baseline = evaluate(predict(net, cal, device, args.eval_batch_size), cal, families, scripts)
    dump(output / 'calibration-baseline.json', baseline)
    sampler = BalancedSampler(train, args.seed)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    script_indices = {script: [families.index(family) for family in names] for script, names in scripts.items()}
    local_target = {script: np.asarray([indices.index(int(y)) if int(y) in indices else -1 for y in train['y']])
                    for script, indices in script_indices.items()}
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        net.train()
        losses = []
        for step in range(args.steps_per_epoch):
            rows = sampler.batch(args.batch_size)
            x = torch.from_numpy(np.array(train['x'][rows, None], copy=True)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = net(x)
            loss = 0.
            for script, indices in script_indices.items():
                mask = train['scripts'][rows] == script
                target = torch.tensor(local_target[script][rows[mask]], dtype=torch.long, device=device)
                loss = loss + F.cross_entropy(logits[torch.from_numpy(mask).to(device)][:, indices], target) / len(scripts)
            require(bool(torch.isfinite(loss)), 'nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step()
            updates = (epoch - 1) * args.steps_per_epoch + step + 1
            progress = updates / (args.epochs * args.steps_per_epoch)
            optimizer.param_groups[0]['lr'] = args.learning_rate * (.1 + .45 * (1 + math.cos(math.pi * progress)))
            losses.append(float(loss.detach().cpu()))
            if updates % 25 == 0:
                print(json.dumps({'epoch': epoch, 'updates': updates, 'loss': float(np.mean(losses[-25:])),
                                  'seconds': time.perf_counter() - start}), flush=True)
        cal_logits = predict(net, cal, device, args.eval_batch_size)
        evaluation = evaluate(cal_logits, cal, families, scripts)
        score = float(np.mean([evaluation[script][kind]['macro_accuracy'] for script in scripts for kind in ('glyph', 'region')]))
        record = {'epoch': epoch, 'updates': updates, 'loss': float(np.mean(losses)), 'selection_score': score,
                  'calibration': evaluation, 'seconds': time.perf_counter() - start,
                  'unique_train_glyphs_sampled': int(np.count_nonzero(sampler.visits))}
        history.append(record)
        if best is None or score > best['score']:
            state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
            checkpoint = {'state_dict': state, 'families': families, 'scripts': scripts,
                          'epoch': epoch, 'optimizer_steps': updates}
            torch.save(checkpoint, output / 'best-calibration.pth')
            best = {'epoch': epoch, 'updates': updates, 'score': score,
                    'checkpoint_sha256': sha(output / 'best-calibration.pth')}
        dump(output / 'training-history.json', history)
        print(json.dumps({'epoch_finished': epoch, 'selection_score': score, 'best_epoch': best['epoch'],
                          'seconds': record['seconds']}), flush=True)
    require(sha(output / 'best-calibration.pth') == best['checkpoint_sha256'], 'selected checkpoint changed')
    selected = torch.load(output / 'best-calibration.pth', map_location='cpu', weights_only=True)
    net.load_state_dict(selected['state_dict'])
    after = state_sha(selected['state_dict'])
    require(before != after, 'selected parameters did not change')
    temperatures, gates, calibration = calibrate(predict(net, cal, device, args.eval_batch_size), cal, families, scripts)
    parameter_change = {key: float((selected['state_dict'][key] - initial[key]).norm())
                        for key in ('trunk.0.weight', 'style.0.weight', 'family_head.weight')}
    selection = {'schema': 'native-screenshot-selection-before-test-v1', 'selected': best,
                 'optimizer_steps_executed': args.epochs * args.steps_per_epoch,
                 'state_before_sha256': before, 'state_after_sha256': after, 'parameter_l2_change': parameter_change,
                 'temperatures': temperatures, 'gates': gates, 'calibration': calibration,
                 'test_glyph_pixels_loaded': False, 'elapsed_training_seconds': time.perf_counter() - start,
                 'train_sampling': {'available_rows': len(train['x']), 'unique_rows_sampled': int(np.count_nonzero(sampler.visits)),
                                    'minimum_visits': int(sampler.visits.min()), 'maximum_visits': int(sampler.visits.max())}}
    dump(output / 'SELECTION_FREEZE.json', selection)
    # No test arrays, metrics or temperature tuning above this point.
    require(sha(args.warm_start) == freeze['warm_start']['sha256'], 'warm-start checkpoint changed during training')
    test = bundle.load('test')
    test_logits = predict(net, test, device, args.eval_batch_size)
    baseline_net = active_model(families, args.warm_start).to(device)
    baseline_test = evaluate(predict(baseline_net, test, device, args.eval_batch_size), test, families, scripts)
    test_report = evaluate(test_logits, test, families, scripts, temperatures, gates)
    neural = output / 'neural'
    neural.mkdir()
    net.cpu().eval()
    torch.onnx.export(net, torch.zeros(2, 1, 64, 64), neural / 'model.onnx', input_names=['glyphs'], output_names=['logits'],
                      dynamic_axes={'glyphs': {0: 'batch'}, 'logits': {0: 'batch'}}, opset_version=17, dynamo=False)
    torch.save(selected, neural / 'model.pth')
    metadata = {'schema': 'flux-glyph-neural-font-v1', 'algorithm': 'glyph-cnn64-v1',
                'families': families, 'scripts': scripts, 'model': {'path': 'model.onnx', 'sha256': sha(neural / 'model.onnx')},
                'temperature': temperatures, 'gates': gates,
                'preprocessing': {'algorithm': 'extract_glyphs-v1', 'shape': [1, 64, 64], 'dtype': 'float32'},
                'training': {'data': 'actual iOS Simulator screen captures of controlled native app scenes',
                             'source_kind': SOURCE_KIND, 'selected_epoch': best['epoch'],
                             'selected_optimizer_steps': best['updates'], 'optimizer_steps_executed': args.epochs * args.steps_per_epoch,
                             'data_manifest_sha256': bundle.manifest_sha, 'selection_sha256': sha(output / 'SELECTION_FREEZE.json'),
                             'parent_sha256': sha(args.warm_start), 'state_before_sha256': before, 'state_after_sha256': after},
                'calibration': {'acceptance_precision_target': .97, 'minimum_accepted_rows': 20,
                                'gate_calibration_level': 'complete_native_region',
                                'glyph_acceptance_precision_target_validated': False,
                                'gate_found': {script: info['gate_found'] for script, info in calibration.items()},
                                'fallback_gate_note': 'No validated gate uses score=margin=1; this is not an unknown-font guarantee.'},
                'release_status': 'experimental',
                'scope': 'Known-font candidates evaluated on held-out controlled iOS Simulator screens; physical-device and unknown-font accuracy unestablished.'}
    dump(neural / 'metadata.json', metadata)
    report = {'schema': 'native-screenshot-font-test-report-v1', 'data_audit': bundle.audit,
              'families': families, 'scripts': scripts, 'calibration_baseline': baseline,
              'selection': selection, 'test_before': baseline_test, 'test_after': test_report,
              'glyph_counts': {split: len(bundle.loaded[split]['x']) for split in SPLITS},
              'region_counts': {split: len(bundle.loaded[split]['regions']) for split in SPLITS},
              'onnx_sha256': sha(neural / 'model.onnx'), 'checkpoint_sha256': sha(neural / 'model.pth'),
              'limitations': ['Screenshots are actual iOS Simulator frames; page text and layouts are generated.',
                              'No physical-device or natural third-party app screenshot accuracy is established.',
                              'Native glyph boxes isolate font classification; OCR detection/segmentation accuracy is measured separately.',
                              'Source-font synthetic checkpoint initializes parameters; all optimizer examples are native screenshot crops.',
                              'Unknown-font rejection is not validated.']}
    dump(output / 'report.json', report)
    print(json.dumps({'finished': True, 'output': str(output), 'selected_epoch': best['epoch'],
                      'updates_executed': args.epochs * args.steps_per_epoch, 'onnx': str(neural / 'model.onnx')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
