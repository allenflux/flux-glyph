#!/usr/bin/env python3
"""Cache only verified TRAIN features from the unchanged R22 convolution trunk."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'training')]
from cache_r22_train_teacher import (load_training, load_base, source_bindings,
    BASE_SHA, BASE_STATE_SHA, read, write_new)
from native_mobile_short_training import load_mobile
from spatial_region_network import expand_spatial_pool
from train_regions import require, sha, state_sha

IOS = ROOT / 'artifacts/ios-native-short-train-v1/prepared'
ANDROID = ROOT / 'artifacts/android-native-short-train-v1/prepared'
ANDROID_SHA = '09584bd629c966148ed27a2c0a82698b3e09bc0f7b5e8c974f0bc0b20ceae872'
SCHEMA = 'flux-glyph-frozen-spatial-train-features-v1'


def cache(output, device):
    import torch
    require(not output.exists(), 'Preserve earlier feature caches')
    torch.set_num_threads(4)
    datasets = load_training()
    mobile, mobile_proof = load_mobile(IOS, ANDROID, ANDROID_SHA)
    source, _ = load_base()
    model, _ = expand_spatial_pool(source)
    del source
    before = state_sha(model.state_dict())
    bindings = {**source_bindings(datasets), **mobile_proof['bindings']}
    for path in [Path(__file__), ROOT / 'training/spatial_region_network.py',
                 ROOT / 'training/native_mobile_short_training.py']:
        bindings[str(path.resolve())] = sha(path)
    sources = {**datasets, **mobile['datasets']}
    expected_counts = {'original': 130854, 'supplement': 3156, 'known': 4820,
                       'ios': 2380, 'android': 4140}
    require(set(sources) == set(expected_counts), 'TRAIN source set differs')
    output.mkdir(parents=True)
    freeze = {'schema': SCHEMA, 'bindings': bindings, 'checkpoint_sha256': BASE_SHA,
              'source_state_sha256': BASE_STATE_SHA, 'spatial_initial_state_sha256': before,
              'device': device, 'feature_dtype': 'float32', 'feature_width': 8192,
              'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
              'sources': expected_counts, 'optimizer_steps': 0,
              'calibration_read': False, 'development_read': False, 'test_read': False,
              'purpose': 'Exact frozen trunk reuse during single-CNN residual training'}
    write_new(output / 'CACHE_FREEZE.json', freeze)
    model.to(device).eval(); entries = {}; started = time.monotonic()
    with torch.inference_mode():
        for name, data in sources.items():
            tiles = data['tiles']; require(len(tiles) == expected_counts[name], 'TRAIN tile count differs')
            require(all(row['split'] == 'train' and row['native_font_verified'] is True
                        for row in data['rows']), 'Unverified or non-TRAIN features requested')
            path = output / (name + '.npy')
            array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                             shape=(len(tiles), 8192))
            for start in range(0, len(tiles), 128):
                pixels = np.array(tiles[start:start + 128], copy=True)
                require(np.isfinite(pixels).all() and pixels.min() >= 0 and pixels.max() <= 1,
                        'Invalid TRAIN input pixels')
                feature = model.pool(model.trunk(torch.from_numpy(pixels).to(device))).cpu().numpy()
                require(feature.shape == (len(pixels), 8192) and np.isfinite(feature).all(),
                        'Invalid frozen TRAIN features')
                array[start:start + len(pixels)] = feature
                if start == 0 or (start + len(pixels)) % 16384 < 128:
                    print({'partition': name, 'tiles': start + len(pixels), 'total': len(tiles)}, flush=True)
            array.flush(); del array
            cached = np.load(path, mmap_mode='r', allow_pickle=False)
            indices = np.unique(np.linspace(0, len(tiles) - 1, 17).astype(int))
            pixels = torch.from_numpy(np.array(tiles[indices], copy=True)).to(device)
            direct = model(pixels)
            pooled = torch.from_numpy(np.array(cached[indices], copy=True)).to(device)
            features = model.style(pooled)
            reused = model.family_head(features), model.size_head(features).squeeze(-1)
            errors = {}
            for label, a, b in zip(('font', 'size'), direct, reused):
                torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-5)
                errors[label] = float((a-b).abs().max().cpu())
            source_path = Path(tiles.filename).resolve()
            require(bindings.get(str(source_path)) == sha(source_path), 'TRAIN pixels are not bound')
            entries[name] = {'path': path.name, 'sha256': sha(path), 'shape': list(cached.shape),
                'dtype': str(cached.dtype), 'source_pixels_path': str(source_path),
                'source_pixels_sha256': sha(source_path), 'reuse_checked_tiles': len(indices),
                'maximum_output_error': errors}
            print({'completed': name, 'seconds': round(time.monotonic() - started, 1)}, flush=True)
    model.cpu()
    require(state_sha(model.state_dict()) == before and all(sha(p) == h for p,h in bindings.items()),
            'Frozen trunk or TRAIN evidence changed while caching')
    write_new(output / 'MANIFEST.json', {**freeze, 'cache_freeze_sha256': sha(output / 'CACHE_FREEZE.json'),
        'files': entries, 'complete': True, 'reuse_parity_passed': True,
        'parameters_unchanged': True, 'seconds': round(time.monotonic() - started, 1)})
    print({'complete': True, 'manifest_sha256': sha(output / 'MANIFEST.json')}, flush=True)


def load_features(root):
    manifest = read(root / 'MANIFEST.json')
    require(manifest.get('schema') == SCHEMA and manifest.get('complete') is True
            and manifest.get('checkpoint_sha256') == BASE_SHA
            and manifest.get('source_state_sha256') == BASE_STATE_SHA
            and manifest.get('parameters_unchanged') is True
            and manifest.get('reuse_parity_passed') is True
            and all(manifest.get(k) is False for k in ('calibration_read','development_read','test_read')),
            'Frozen TRAIN feature cache does not satisfy its contract')
    require(sha(root / 'CACHE_FREEZE.json') == manifest['cache_freeze_sha256'], 'Feature freeze differs')
    require(all(sha(p) == h for p,h in manifest['bindings'].items()), 'Feature source changed')
    result = {}
    require(set(manifest['files']) == {'original','supplement','known','ios','android'},
            'Unexpected feature source')
    for name, entry in manifest['files'].items():
        path = root / entry['path']
        require(path.parent == root and sha(path) == entry['sha256'], 'Feature bytes changed')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        require(list(array.shape) == entry['shape'] == [manifest['sources'][name], 8192]
                and array.dtype == np.float32, 'Feature shape/dtype changed')
        result[name] = array
    return result, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument('--device', choices=['cpu','mps'], default='mps')
    args = parser.parse_args(); cache(args.output, args.device)
